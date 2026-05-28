import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_residency_aware_prefetch_smoke import (
    run_residency_aware_prefetch_smoke,
)


def _write_manifest(
    path: Path,
    prefix_id: str,
    token_end: int,
    *,
    tier: str,
    ready: bool,
) -> None:
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
                "tier": tier,
                "ready": ready,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )


def test_residency_aware_prefetch_smoke_distinguishes_hit_completed_and_miss(
    tmp_path,
):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "dram-hit", 64, tier="DRAM", ready=True)
    _write_manifest(tensor_store, "nvme-ok", 64, tier="NVME", ready=False)
    _write_manifest(tensor_store, "nvme-miss", 1_000_000, tier="NVME", ready=False)

    summary = run_residency_aware_prefetch_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        prefix_ids=["dram-hit", "nvme-ok", "nvme-miss"],
        storage_gbps=0.002,
    )

    assert summary["status"] == "OK"
    assert summary["residency_hits"] == 1
    assert summary["prefetch_completed"] == 1
    assert summary["deadline_miss"] == 1
    assert summary["queue_requests"] == 2

    csv_text = (tmp_path / "result" / "residency_aware_prefetch_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "dram-hit,RESIDENCY_HIT" in csv_text
    assert "nvme-ok,PREFETCH_COMPLETED" in csv_text
    assert "nvme-miss,DEADLINE_MISS" in csv_text
    assert (tmp_path / "result" / "residency_aware_prefetch_report.md").exists()


def test_residency_aware_prefetch_smoke_help_runs():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/run_residency_aware_prefetch_smoke.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--prefix-ids" in completed.stdout


def test_residency_aware_prefetch_smoke_uses_configured_kv_bytes_for_hits(
    tmp_path,
):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "dram-hit", 64, tier="DRAM", ready=True)

    summary = run_residency_aware_prefetch_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        prefix_ids=["dram-hit"],
        kv_bytes_per_token=1024,
    )

    assert summary["status"] == "OK"
    csv_text = (tmp_path / "result" / "residency_aware_prefetch_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "dram-hit,RESIDENCY_HIT,64,65536" in csv_text
