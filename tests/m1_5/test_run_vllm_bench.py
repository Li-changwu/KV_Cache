from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from benchmarks.m1_5.parse_vllm_results import parse_result
from benchmarks.m1_5.run_vllm_bench import load_config, render_commands, run_commands


def test_load_config_rejects_missing_model(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model: /definitely/missing/model
base_url: http://127.0.0.1:8000
endpoint: /v1/completions
result_dir: results/benchmark_calibration
raw_result_dir: results/benchmark_calibration/raw
random_smoke:
  input_lens: [512]
  output_lens: [16]
  request_rates: [1]
  max_concurrencies: [1]
  num_prompts: 16
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model path does not exist"):
        load_config(config)


def test_render_commands_includes_random_and_prefix_repetition(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
model: {model}
base_url: http://127.0.0.1:8000
endpoint: /v1/completions
result_dir: {tmp_path}/results
raw_result_dir: {tmp_path}/results/raw
random_smoke:
  input_lens: [512]
  output_lens: [16]
  request_rates: [1]
  max_concurrencies: [1]
  num_prompts: 16
apc_sweep:
  num_prefixes: [1]
  prefix_lens: [1024]
  suffix_lens: [128]
  output_lens: [128]
  request_rates: [1]
  max_concurrencies: [4]
  num_prompts: 64
""",
        encoding="utf-8",
    )

    commands = render_commands(load_config(config), suites=["random_smoke", "apc_sweep"])

    joined = "\n".join(" ".join(cmd) for cmd in commands)
    assert "--dataset-name random" in joined
    assert "--random-input-len 512" in joined
    assert "--dataset-name prefix_repetition" in joined
    assert "--prefix-repetition-prefix-len 1024" in joined
    assert "--save-detailed" in joined


def test_run_commands_records_failed_rows(tmp_path, monkeypatch):
    result_dir = tmp_path / "raw"
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        "/root/models/gpt-oss-20b",
        "--result-dir",
        str(result_dir),
        "--label",
        "m1_5_latency_oom",
        "--metadata",
        "suite=latency_sweep",
        "dataset_name=random",
        "input_len=32768",
        "output_len=128",
        "--dataset-name",
        "random",
        "--random-input-len",
        "32768",
        "--random-output-len",
        "128",
        "--num-prompts",
        "32",
        "--request-rate",
        "1",
        "--max-concurrency",
        "1",
        "--result-filename",
        "latency-oom.json",
    ]

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="CUDA out of memory while allocating KV cache",
        )

    monkeypatch.setattr("benchmarks.m1_5.run_vllm_bench.subprocess.run", fake_run)

    assert run_commands([command], dry_run=False) == 0
    failure_path = result_dir / "latency-oom.json"
    assert failure_path.exists()
    raw = json.loads(failure_path.read_text(encoding="utf-8"))
    assert raw["errors"] == ["CUDA out of memory while allocating KV cache"]
    assert parse_result(failure_path)["status"] == "OOM"
