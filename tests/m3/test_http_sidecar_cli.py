import subprocess
import sys
from pathlib import Path
import json

from benchmarks.m3.http_sidecar_cli import build_config_from_args, parse_args


def test_build_config_from_qwen25_calibration(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    args = parse_args(
        [
            "--simulator-params",
            "results/qwen25_14b_calibration/simulator_params.yaml",
            "--capacity-csv",
            "results/qwen25_14b_calibration/qwen25_capacity_probe.csv",
            "--decision-log",
            str(tmp_path / "decisions.jsonl"),
            "--upstream-base-url",
            "http://127.0.0.1:8000",
            "--inject-kv-transfer-params",
            "--tensor-store-path",
            str(tensor_store),
            "--tensor-store-block-size",
            "32",
            "--async-prefetch",
        ]
    )

    config = build_config_from_args(args)

    assert config.model_id == "/root/models/Qwen2.5-14B-Instruct"
    assert config.kv_bytes_per_token == 196_608
    assert config.hbm_capacity_tokens >= 64_000
    assert config.upstream_base_url == "http://127.0.0.1:8000"
    assert config.decision_log_path == tmp_path / "decisions.jsonl"
    assert config.inject_kv_transfer_params is True
    assert config.tensor_store_path == tensor_store
    assert config.tensor_store_block_size == 32
    assert config.async_prefetch is True


def test_build_config_can_select_3fs_posix_cold_backend(tmp_path):
    cold_root = tmp_path / "threefs_mount"
    args = parse_args(
        [
            "--simulator-params",
            "results/qwen25_14b_calibration/simulator_params.yaml",
            "--capacity-csv",
            "results/qwen25_14b_calibration/qwen25_capacity_probe.csv",
            "--decision-log",
            str(tmp_path / "decisions.jsonl"),
            "--tensor-store-path",
            str(tmp_path / "tensor_store"),
            "--cold-root",
            str(cold_root),
            "--cold-backend",
            "3fs_posix",
        ]
    )

    config = build_config_from_args(args)

    assert config.cold_root == cold_root
    assert config.cold_backend == "3fs_posix"


def test_http_sidecar_cli_dry_run_prints_config(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/http_sidecar_cli.py",
            "--simulator-params",
            "results/qwen25_14b_calibration/simulator_params.yaml",
            "--capacity-csv",
            "results/qwen25_14b_calibration/qwen25_capacity_probe.csv",
            "--decision-log",
            str(tmp_path / "decisions.jsonl"),
            "--tensor-store-path",
            str(tmp_path / "tensor_store"),
            "--async-prefetch",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "/root/models/Qwen2.5-14B-Instruct" in completed.stdout
    assert "decision_log_path" in completed.stdout
    assert "inject_kv_transfer_params" in completed.stdout
    assert "tensor_store_path" in completed.stdout
    assert '"async_prefetch": true' in completed.stdout


def test_http_sidecar_cli_dry_run_prints_cold_backend(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/http_sidecar_cli.py",
            "--simulator-params",
            "results/qwen25_14b_calibration/simulator_params.yaml",
            "--capacity-csv",
            "results/qwen25_14b_calibration/qwen25_capacity_probe.csv",
            "--decision-log",
            str(tmp_path / "decisions.jsonl"),
            "--tensor-store-path",
            str(tmp_path / "tensor_store"),
            "--cold-root",
            str(tmp_path / "threefs_mount"),
            "--cold-backend",
            "3fs_posix",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["cold_root"] == str(tmp_path / "threefs_mount")
    assert payload["cold_backend"] == "3fs_posix"
