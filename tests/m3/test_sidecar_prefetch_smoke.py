import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_sidecar_prefetch_smoke import run_sidecar_prefetch_smoke


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


def test_sidecar_prefetch_smoke_delays_then_admits_after_prefetch(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a")

    summary = run_sidecar_prefetch_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        prefix_id="session-a",
    )

    assert summary["status"] == "OK"
    assert summary["first_decision"] == "DELAY"
    assert summary["second_decision"] == "ADMIT"

    manifest = json.loads(
        (tensor_store / "session-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tier"] == "DRAM"
    assert manifest["ready"] is True

    csv_text = (tmp_path / "result" / "sidecar_prefetch_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "prefix_id,first_decision,first_reason,second_decision" in csv_text
    assert "session-a,DELAY,required_kv_not_ready_before_decode,ADMIT" in csv_text

    records = [
        json.loads(line)
        for line in (tmp_path / "result" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(record["source"] == "prefetch_to_dram" for record in records)
    assert (tmp_path / "result" / "sidecar_prefetch_report.md").exists()


def test_sidecar_prefetch_smoke_records_deadline_miss(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", token_end=1_000_000)

    summary = run_sidecar_prefetch_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        prefix_id="session-a",
        token_end=1_000_000,
        request_token_count=1_000_016,
        storage_gbps=0.001,
    )

    assert summary["status"] == "DEADLINE_MISS"
    assert summary["first_decision"] == "DELAY"
    assert summary["second_decision"] == "DELAY"
    assert summary["prefetch_deadline_miss"] is True

    csv_text = (tmp_path / "result" / "sidecar_prefetch_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "prefetch_deadline_miss" in csv_text
    assert "DEADLINE_MISS" in csv_text


def test_sidecar_prefetch_smoke_help_runs():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_sidecar_prefetch_smoke.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--tensor-store" in completed.stdout


def test_sidecar_prefetch_smoke_cli_accepts_storage_gbps(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a")
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/run_sidecar_prefetch_smoke.py",
            "--tensor-store",
            str(tensor_store),
            "--result-dir",
            str(tmp_path / "result"),
            "--prefix-id",
            "session-a",
            "--storage-gbps",
            "0.000001",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "DEADLINE_MISS"
