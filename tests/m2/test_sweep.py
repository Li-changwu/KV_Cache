import csv
import subprocess
import sys
from pathlib import Path

import yaml

from benchmarks.m2.simulator import SimulationConfig, load_calibration
from benchmarks.m2.sweep import (
    SweepConfig,
    find_boundaries,
    find_deadline_frontier,
    run_sweep,
    write_sweep_outputs,
)


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
        },
    }


def test_sweep_generates_grid_and_boundary_rows(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)

    sweep = SweepConfig(
        workloads=["shared_prefix", "low_locality"],
        deadlines_ms=[250.0, 12_000.0],
        prefetch_leads_ms=[0.0, 8_000.0],
        h2d_gbps=[25.0],
        nvme_gbps=[2.5],
        dram_capacity_tokens=[900_000],
        hbm_capacity_tokens=[32_768],
        h2d_parallelism=[1],
        nvme_parallelism=[1],
        reuse_prediction_error_rates=[0.0],
        target_admitted_rate=0.5,
        base_config=SimulationConfig(num_requests=2, admission_prefetch_window_ms=20_000.0),
    )

    rows = run_sweep(calibration, sweep)
    boundaries = find_boundaries(rows, target_admitted_rate=0.5)

    assert len(rows) == 8
    assert {row.workload for row in boundaries} == {"shared_prefix", "low_locality"}
    shared = next(row for row in boundaries if row.workload == "shared_prefix")
    low = next(row for row in boundaries if row.workload == "low_locality")
    assert shared.status == "MEETS_TARGET"
    assert shared.deadline_ms == 250.0
    assert shared.prefetch_lead_ms == 8_000.0
    assert low.status == "NO_SLA_REGION"


def test_boundary_prefers_lower_prediction_error_before_resource_cost():
    rows = [
        # This row meets the target with cheap resources only because it reuses less KV.
        # It should not be chosen as the representative benefit boundary.
        _row(
            workload="shared_prefix",
            deadline_ms=250.0,
            h2d_gbps=25.0,
            reuse_prediction_error_rate=0.25,
            admitted_rate=0.875,
            effective_hit_rate=0.49,
        ),
        _row(
            workload="shared_prefix",
            deadline_ms=250.0,
            h2d_gbps=200.0,
            reuse_prediction_error_rate=0.0,
            admitted_rate=0.875,
            effective_hit_rate=0.66,
        ),
    ]

    [boundary] = find_boundaries(rows, target_admitted_rate=0.875)

    assert boundary.status == "MEETS_TARGET"
    assert boundary.reuse_prediction_error_rate == 0.0
    assert boundary.effective_hit_rate == 0.66


def test_deadline_frontier_emits_one_row_per_workload_and_deadline():
    rows = [
        _row(workload="shared_prefix", deadline_ms=250.0, admitted_rate=0.0),
        _row(
            workload="shared_prefix",
            deadline_ms=1000.0,
            admitted_rate=0.875,
            h2d_gbps=100.0,
        ),
        _row(
            workload="shared_prefix",
            deadline_ms=1000.0,
            admitted_rate=0.875,
            h2d_gbps=200.0,
        ),
        _row(workload="low_locality", deadline_ms=250.0, admitted_rate=0.0),
    ]

    frontier = find_deadline_frontier(rows, target_admitted_rate=0.875)

    assert len(frontier) == 3
    shared_250 = next(
        row for row in frontier if row.workload == "shared_prefix" and row.deadline_ms == 250.0
    )
    shared_1000 = next(
        row for row in frontier if row.workload == "shared_prefix" and row.deadline_ms == 1000.0
    )
    assert shared_250.status == "NO_SLA_REGION"
    assert shared_1000.status == "MEETS_TARGET"
    assert shared_1000.h2d_gbps == 100.0


def test_sweep_writes_csv_and_report(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    calibration = load_calibration(params, None)
    sweep = SweepConfig(
        workloads=["shared_prefix"],
        deadlines_ms=[250.0],
        prefetch_leads_ms=[8_000.0],
        h2d_gbps=[25.0],
        nvme_gbps=[2.5],
        dram_capacity_tokens=[900_000],
        hbm_capacity_tokens=[32_768],
        target_admitted_rate=0.5,
        base_config=SimulationConfig(num_requests=2, admission_prefetch_window_ms=20_000.0),
    )
    rows = run_sweep(calibration, sweep)
    boundaries = find_boundaries(rows, target_admitted_rate=0.5)

    write_sweep_outputs(tmp_path, rows, boundaries, sweep)

    summary = read_csv(tmp_path / "m2_sweep_summary.csv")
    boundary = read_csv(tmp_path / "m2_sweep_boundaries.csv")
    frontier = read_csv(tmp_path / "m2_sweep_deadline_frontier.csv")
    report = (tmp_path / "sweep_report.md").read_text(encoding="utf-8")

    assert summary[0]["workload"] == "shared_prefix"
    assert boundary[0]["status"] == "MEETS_TARGET"
    assert frontier[0]["workload"] == "shared_prefix"
    assert "M2 Sensitivity Sweep Report" in report
    assert "shared_prefix" in report


def test_sweep_cli_writes_outputs(tmp_path):
    params = tmp_path / "simulator_params.yaml"
    write_yaml(params, minimal_params())
    output_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m2/sweep.py",
            "--simulator-params",
            str(params),
            "--output-dir",
            str(output_dir),
            "--workloads",
            "shared_prefix",
            "--deadlines-ms",
            "250",
            "--prefetch-leads-ms",
            "8000",
            "--h2d-gbps",
            "25",
            "--nvme-gbps",
            "2.5",
            "--dram-capacity-tokens",
            "900000",
            "--hbm-capacity-tokens",
            "32768",
            "--num-requests",
            "2",
            "--admission-prefetch-window-ms",
            "20000",
            "--target-admitted-rate",
            "0.5",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "m2_sweep_summary.csv").exists()
    assert (output_dir / "m2_sweep_boundaries.csv").exists()


def _row(**overrides):
    defaults = {
        "workload": "shared_prefix",
        "deadline_ms": 250.0,
        "prefetch_lead_ms": 0.0,
        "h2d_gbps": 25.0,
        "nvme_gbps": 2.5,
        "dram_capacity_tokens": 500_000,
        "hbm_capacity_tokens": 44_128,
        "h2d_parallelism": 1,
        "nvme_parallelism": 1,
        "reuse_prediction_error_rate": 0.0,
        "requests": 8,
        "admitted_requests": 7,
        "admitted_rate": 0.875,
        "rejected_or_delayed_requests": 1,
        "effective_hit_rate": 0.5,
        "prefill_tokens_saved": 1,
        "prefetch_deadline_miss_rate": 0.0,
        "sync_ssd_miss_rate": 0.0,
        "nvme_read_gb": 0.0,
        "h2d_gb": 0.0,
        "max_ttft_ms": 0.0,
        "p50_ttft_ms": 0.0,
    }
    defaults.update(overrides)
    from benchmarks.m2.sweep import SweepRow

    return SweepRow(**defaults)
