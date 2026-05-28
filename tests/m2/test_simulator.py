import csv
import subprocess
import sys
from pathlib import Path

import yaml

from benchmarks.m2.simulator import (
    SimulationConfig,
    load_calibration,
    run_simulation,
    write_outputs,
)
from benchmarks.m2.workloads import build_workload


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def minimal_params() -> dict:
    return {
        "verification_status": "MEASURED",
        "hardware": {
            "gpu_name": "Test GPU",
            "hbm_total_bytes": 48 * 1024**3,
            "dram_total_bytes": 512 * 1024**3,
            "nvme_read_latency_by_size": {
                "64M": {"bandwidth_gbps": 2.5, "p50_ms": 27.0}
            },
            "h2d_gbps": {"1GB": 25.0},
        },
        "vllm": {
            "version": "0.19.0",
            "model_id": "/root/models/Qwen3-32B",
            "kv_bytes_per_token": 262144,
            "ttft_by_input_len": {128: {"p50_ms": 100.0}},
        },
        "notes": {"storage_path": "local_nvme_lower_bound"},
    }


def test_load_calibration_derives_qwen_kv_bytes_and_capacity(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    capacity = tmp_path / "qwen3_capacity_probe.csv"
    write_yaml(params, minimal_params())
    capacity.write_text(
        "max_model_len,status,gpu_kv_cache_tokens,max_concurrency,"
        "cpu_offloaded_parameters_gb,needed_kv_cache_gib,available_kv_cache_gib,"
        "estimated_max_model_len,startup_seconds,exit_code,log_path,error_summary\n"
        "32768,STARTED,44128,1.35,32.45,,,,,,log,\n",
        encoding="utf-8",
    )

    calibration = load_calibration(params, capacity)

    assert calibration.kv_bytes_per_token == 262144
    assert calibration.measured_gpu_kv_tokens == 44128
    assert calibration.h2d_gbps == 25.0
    assert calibration.nvme_gbps == 2.5


def test_load_calibration_uses_model_specific_kv_bytes(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    data = minimal_params()
    data["vllm"]["model_id"] = "/root/models/Qwen2.5-14B-Instruct"
    data["vllm"]["kv_bytes_per_token"] = 196608
    write_yaml(params, data)

    calibration = load_calibration(params, None)

    assert calibration.kv_bytes_per_token == 196608


def test_workloads_encode_reuse_and_negative_control():
    sessions = build_workload("session_append", num_requests=3)
    shared = build_workload("shared_prefix", num_requests=4)
    random = build_workload("low_locality", num_requests=4)

    assert [request.reuse_tokens for request in sessions] == [0, 262144, 524288]
    assert shared[1].prefix_id == shared[0].prefix_id
    assert all(request.reuse_tokens == 0 for request in random)
    assert len({request.prefix_id for request in random}) == 4


def test_simulation_distinguishes_reuse_from_low_locality(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)
    config = SimulationConfig(
        hbm_capacity_tokens=32768,
        dram_capacity_tokens=900_000,
        request_deadline_ms=250.0,
        admission_prefetch_window_ms=10_000.0,
        num_requests=4,
    )

    session_result = run_simulation(calibration, "session_append", config)
    low_result = run_simulation(calibration, "low_locality", config)

    assert session_result.summary.effective_hit_rate > low_result.summary.effective_hit_rate
    assert session_result.summary.prefill_tokens_saved > 0
    assert low_result.summary.rejected_or_delayed_requests > 0
    assert session_result.summary.sync_ssd_miss_rate == 0.0
    assert low_result.summary.sync_ssd_miss_rate == 0.0


def test_deadline_miss_requests_are_delayed_not_admitted(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)

    result = run_simulation(
        calibration,
        "shared_prefix",
        SimulationConfig(
            num_requests=2,
            hbm_capacity_tokens=32768,
            dram_capacity_tokens=900_000,
            request_deadline_ms=1.0,
            admission_prefetch_window_ms=10_000.0,
        ),
    )

    deadline_miss = [row for row in result.requests if row.deadline_miss]
    assert deadline_miss
    assert all(not row.admitted for row in deadline_miss)
    assert all(row.delayed for row in deadline_miss)


def test_prefetch_lead_and_parallelism_can_admit_reuse_request(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)

    no_lead = run_simulation(
        calibration,
        "shared_prefix",
        SimulationConfig(
            num_requests=2,
            hbm_capacity_tokens=32768,
            dram_capacity_tokens=900_000,
            request_deadline_ms=250.0,
            admission_prefetch_window_ms=20_000.0,
            h2d_parallelism=1,
            prefetch_lead_ms=0.0,
        ),
    )
    with_lead = run_simulation(
        calibration,
        "shared_prefix",
        SimulationConfig(
            num_requests=2,
            hbm_capacity_tokens=32768,
            dram_capacity_tokens=900_000,
            request_deadline_ms=250.0,
            admission_prefetch_window_ms=20_000.0,
            h2d_parallelism=2,
            prefetch_lead_ms=5_000.0,
        ),
    )

    assert no_lead.summary.admitted_requests == 0
    assert with_lead.summary.admitted_requests == 1
    assert with_lead.summary.prefetch_deadline_miss_rate < no_lead.summary.prefetch_deadline_miss_rate


def test_reuse_prediction_error_reduces_effective_hit_rate(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)

    clean = run_simulation(
        calibration,
        "shared_prefix",
        SimulationConfig(
            num_requests=3,
            hbm_capacity_tokens=32768,
            dram_capacity_tokens=900_000,
            request_deadline_ms=20_000.0,
            reuse_prediction_error_rate=0.0,
        ),
    )
    noisy = run_simulation(
        calibration,
        "shared_prefix",
        SimulationConfig(
            num_requests=3,
            hbm_capacity_tokens=32768,
            dram_capacity_tokens=900_000,
            request_deadline_ms=20_000.0,
            reuse_prediction_error_rate=0.25,
        ),
    )

    assert noisy.summary.effective_hit_rate < clean.summary.effective_hit_rate
    assert noisy.summary.prefill_tokens_saved < clean.summary.prefill_tokens_saved


def test_write_outputs_creates_csv_and_report(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)
    result = run_simulation(
        calibration,
        "shared_prefix",
        SimulationConfig(num_requests=2, hbm_capacity_tokens=32768),
    )

    write_outputs(tmp_path, [result])

    summary_rows = read_csv(tmp_path / "m2_summary.csv")
    request_rows = read_csv(tmp_path / "m2_requests.csv")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert summary_rows[0]["workload"] == "shared_prefix"
    assert request_rows[0]["workload"] == "shared_prefix"
    assert "M2 Tiered KV Simulation Report" in report
    assert "shared_prefix" in report


def test_cli_script_runs_from_repo_root(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    output_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m2/simulator.py",
            "--simulator-params",
            str(params),
            "--output-dir",
            str(output_dir),
            "--workloads",
            "shared_prefix",
            "--num-requests",
            "2",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "m2_summary.csv").exists()
