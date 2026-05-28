import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_async_prefetch_smoke import run_async_prefetch_smoke


def _write_manifest(path: Path, prefix_id: str, token_end: int = 64) -> None:
    prefix_dir = path / prefix_id
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": token_end,
                "correctness_key": {
                    "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
                    "tokenizer_fingerprint": "qwen2.5-tokenizer",
                    "rope_config": "native-32768",
                    "dtype": "bf16",
                    "kv_layout": "vllm-paged",
                },
                "token_ids": list(range(min(token_end, 16))),
                "block_size": 16,
                "layout": "NHD",
                "tier": "DRAM",
                "ready": True,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )


def test_async_prefetch_smoke_delays_until_advance_then_admits(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a")

    summary = run_async_prefetch_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        prefix_id="session-a",
    )

    assert summary["status"] == "OK"
    assert summary["first_decision"] == "DELAY"
    assert summary["second_decision_before_advance"] == "DELAY"
    assert summary["third_decision_after_advance"] == "ADMIT"
    assert summary["queued_status"] == "QUEUED"
    assert summary["completed_status"] == "COMPLETED"
    assert summary["pending_after_advance"] == 0
    csv_text = (tmp_path / "result" / "async_prefetch_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "prefix_id,queued_status,completed_status" in csv_text
    assert "session-a,QUEUED,COMPLETED" in csv_text
    assert (tmp_path / "result" / "async_prefetch_report.md").exists()


def test_async_prefetch_smoke_help_runs():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_async_prefetch_smoke.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--prefix-id" in completed.stdout
