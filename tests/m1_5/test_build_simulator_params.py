import csv
import json
from pathlib import Path

import pytest
import yaml

from benchmarks.m1_5.build_simulator_params import build_params


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_params_requires_calibration_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="vllm_latency.csv"):
        build_params(tmp_path)


def test_build_params_uses_measured_csvs(tmp_path):
    (tmp_path / "env.json").write_text(
        json.dumps(
            {
                "gpu": [{"name": "NVIDIA RTX A6000", "memory_total_mib": 46068}],
                "dram": {"total_bytes": 540000000000},
                "vllm": {"version": "0.19.0"},
                "model_config": {
                    "num_hidden_layers": 24,
                    "num_key_value_heads": 8,
                    "hidden_size": 4096,
                    "num_attention_heads": 64,
                    "head_dim": 64,
                    "torch_dtype": "bfloat16",
                },
                "notes": {
                    "dense_attention_representativeness": "limited_by_sliding_window_attention"
                },
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        tmp_path / "vllm_latency.csv",
        [
            {
                "run_id": "lat512",
                "model": "/root/models/gpt-oss-20b",
                "dataset": "random",
                "input_len": "512",
                "output_len": "16",
                "prefix_len": "",
                "suffix_len": "",
                "num_prefixes": "",
                "request_rate": "1",
                "max_concurrency": "1",
                "num_prompts": "16",
                "ttft_p50_ms": "90",
                "ttft_p90_ms": "130",
                "ttft_p99_ms": "180",
                "tpot_p50_ms": "12",
                "tpot_p90_ms": "18",
                "tpot_p99_ms": "25",
                "itl_p50_ms": "11",
                "itl_p90_ms": "17",
                "itl_p99_ms": "24",
                "e2el_p50_ms": "300",
                "e2el_p90_ms": "420",
                "e2el_p99_ms": "500",
                "request_throughput": "2.5",
                "output_throughput": "40",
                "total_errors": "0",
                "status": "OK",
            }
        ],
    )
    write_csv(
        tmp_path / "apc_prefix_reuse.csv",
        [
            {
                "run_id": "apc1024",
                "model": "/root/models/gpt-oss-20b",
                "dataset": "prefix_repetition",
                "input_len": "",
                "output_len": "128",
                "prefix_len": "1024",
                "suffix_len": "128",
                "num_prefixes": "1",
                "request_rate": "1",
                "max_concurrency": "4",
                "num_prompts": "64",
                "ttft_p50_ms": "80",
                "ttft_p90_ms": "120",
                "ttft_p99_ms": "160",
                "tpot_p50_ms": "10",
                "tpot_p90_ms": "14",
                "tpot_p99_ms": "20",
                "itl_p50_ms": "9",
                "itl_p90_ms": "13",
                "itl_p99_ms": "19",
                "e2el_p50_ms": "1000",
                "e2el_p90_ms": "1300",
                "e2el_p99_ms": "1700",
                "request_throughput": "3",
                "output_throughput": "384",
                "total_errors": "0",
                "status": "OK",
            }
        ],
    )
    write_csv(
        tmp_path / "hardware_io.csv",
        [
            {
                "method": "python",
                "mode": "sequential_read",
                "block_size": "1048576",
                "block_size_label": "1M",
                "bandwidth_gbps": "5.5",
                "latency_p50_ms": "0.2",
                "latency_p90_ms": "0.3",
                "latency_p99_ms": "0.5",
                "status": "OK",
            }
        ],
    )
    write_csv(
        tmp_path / "h2d_bandwidth.csv",
        [
            {
                "size_bytes": "67108864",
                "size_label": "64MB",
                "bandwidth_gbps": "12.5",
                "latency_p50_ms": "5",
                "latency_p90_ms": "6",
                "latency_p99_ms": "7",
                "status": "OK",
            }
        ],
    )

    params = build_params(tmp_path)

    assert params["hardware"]["gpu_name"] == "NVIDIA RTX A6000"
    assert params["vllm"]["version"] == "0.19.0"
    assert params["vllm"]["kv_bytes_per_token"] == 49152
    assert params["vllm"]["ttft_by_input_len"][512]["p50_ms"] == 90.0
    assert params["hardware"]["h2d_gbps"]["64MB"] == 12.5

    output = tmp_path / "simulator_params.yaml"
    assert output.exists()
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded["notes"]["dense_attention_representativeness"] == "limited_by_sliding_window_attention"

    report = tmp_path / "report.md"
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "# Benchmark Calibration Report" in report_text
    assert "NVIDIA RTX A6000" in report_text
    assert "/root/models/gpt-oss-20b" in report_text
    assert "limited_by_sliding_window_attention" in report_text
    assert "Verification Status: MEASURED" in report_text


def test_build_params_derives_qwen25_kv_bytes_from_env(tmp_path):
    (tmp_path / "env.json").write_text(
        json.dumps(
            {
                "gpu": [{"name": "NVIDIA RTX A6000", "memory_total_mib": 46068}],
                "dram": {"total_bytes": 540000000000},
                "vllm": {"version": "0.19.0"},
                "model_config": {
                    "num_hidden_layers": 48,
                    "num_attention_heads": 40,
                    "num_key_value_heads": 8,
                    "hidden_size": 5120,
                    "torch_dtype": "bfloat16",
                },
                "notes": {
                    "dense_attention_representativeness": "full_attention_path_for_configured_context"
                },
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        tmp_path / "vllm_latency.csv",
        [
            {
                "run_id": "lat128",
                "model": "/root/models/Qwen2.5-14B-Instruct",
                "dataset": "random",
                "input_len": "128",
                "output_len": "1",
                "prefix_len": "",
                "suffix_len": "",
                "num_prefixes": "",
                "request_rate": "1",
                "max_concurrency": "1",
                "num_prompts": "1",
                "ttft_p50_ms": "100",
                "ttft_p90_ms": "130",
                "ttft_p99_ms": "180",
                "tpot_p50_ms": "0",
                "tpot_p90_ms": "0",
                "tpot_p99_ms": "0",
                "itl_p50_ms": "0",
                "itl_p90_ms": "0",
                "itl_p99_ms": "0",
                "e2el_p50_ms": "100",
                "e2el_p90_ms": "130",
                "e2el_p99_ms": "180",
                "request_throughput": "1",
                "output_throughput": "1",
                "total_errors": "0",
                "status": "OK",
            }
        ],
    )
    write_csv(
        tmp_path / "apc_prefix_reuse.csv",
        [
            {
                "run_id": "apc128",
                "model": "/root/models/Qwen2.5-14B-Instruct",
                "dataset": "prefix_repetition",
                "input_len": "",
                "output_len": "1",
                "prefix_len": "128",
                "suffix_len": "16",
                "num_prefixes": "1",
                "request_rate": "1",
                "max_concurrency": "1",
                "num_prompts": "1",
                "ttft_p50_ms": "90",
                "ttft_p90_ms": "120",
                "ttft_p99_ms": "160",
                "tpot_p50_ms": "0",
                "tpot_p90_ms": "0",
                "tpot_p99_ms": "0",
                "itl_p50_ms": "0",
                "itl_p90_ms": "0",
                "itl_p99_ms": "0",
                "e2el_p50_ms": "90",
                "e2el_p90_ms": "120",
                "e2el_p99_ms": "160",
                "request_throughput": "1",
                "output_throughput": "1",
                "total_errors": "0",
                "status": "OK",
            }
        ],
    )
    write_csv(
        tmp_path / "hardware_io.csv",
        [
            {
                "method": "python",
                "mode": "sequential_read",
                "block_size": "1048576",
                "block_size_label": "1M",
                "bandwidth_gbps": "5.5",
                "latency_p50_ms": "0.2",
                "latency_p90_ms": "0.3",
                "latency_p99_ms": "0.5",
                "status": "OK",
            }
        ],
    )
    write_csv(
        tmp_path / "h2d_bandwidth.csv",
        [
            {
                "size_bytes": "67108864",
                "size_label": "64MB",
                "bandwidth_gbps": "12.5",
                "latency_p50_ms": "5",
                "latency_p90_ms": "6",
                "latency_p99_ms": "7",
                "status": "OK",
            }
        ],
    )

    params = build_params(tmp_path)

    assert params["vllm"]["kv_bytes_per_token"] == 196608
