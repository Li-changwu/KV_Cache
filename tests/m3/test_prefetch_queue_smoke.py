import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_prefetch_queue_smoke import run_prefetch_queue_smoke


def _write_manifest(path: Path, prefix_id: str, token_end: int) -> None:
    prefix_dir = path / prefix_id
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": token_end,
                "correctness_key": {"model_fingerprint": "model"},
                "token_ids": list(range(min(token_end, 16))),
                "block_size": 16,
                "layout": "NHD",
                "tier": "NVME",
                "ready": False,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )


def test_prefetch_queue_smoke_writes_rows_summary_and_report(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", 64)
    _write_manifest(tensor_store, "session-b", 128)
    _write_manifest(tensor_store, "session-c", 1_000_000)

    summary = run_prefetch_queue_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        prefix_ids=["session-a", "session-b", "session-c"],
        storage_gbps=0.004,
    )

    assert summary["status"] == "OK"
    assert summary["requests"] == 3
    assert summary["completed"] == 2
    assert summary["deadline_miss"] == 1
    csv_text = (tmp_path / "result" / "prefetch_queue_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "prefix_id,status,tokens,bytes" in csv_text
    assert "session-c,DEADLINE_MISS" in csv_text
    assert (tmp_path / "result" / "prefetch_queue_report.md").exists()


def test_prefetch_queue_smoke_help_runs():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_prefetch_queue_smoke.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--prefix-ids" in completed.stdout
