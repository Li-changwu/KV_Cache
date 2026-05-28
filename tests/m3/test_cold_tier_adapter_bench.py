import csv
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_cold_tier_adapter_bench import (
    ColdTierAdapterBenchConfig,
    run_cold_tier_adapter_bench,
)


def test_cold_tier_adapter_bench_writes_latency_and_bandwidth_outputs(tmp_path):
    result = run_cold_tier_adapter_bench(
        ColdTierAdapterBenchConfig(
            result_dir=tmp_path / "result",
            cold_root=tmp_path / "cold",
            cold_backend="local_posix",
            prefix_tokens=[16, 32],
            repeats=2,
            layers=2,
            block_size=16,
            kv_heads=1,
            head_dim=4,
        )
    )

    assert result["status"] == "OK"
    assert result["rows"] == 4
    assert result["cold_backend"] == "local_posix"
    assert result["bytes_total"] > 0
    assert result["demote_ms_p50"] >= 0
    assert result["restore_ms_p95"] >= 0
    assert Path(result["csv_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()

    rows = list(csv.DictReader(Path(result["csv_path"]).open()))
    assert len(rows) == 4
    assert rows[0]["cold_backend"] == "local_posix"
    assert rows[0]["checksum_status"] == "ok"
    assert rows[0]["hot_files_restored"] == "yes"
    assert float(rows[0]["demote_mib_per_s"]) >= 0
    assert float(rows[0]["restore_mib_per_s"]) >= 0

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["restore_ms_p99"] >= summary["restore_ms_p50"]


def test_cold_tier_adapter_bench_supports_3fs_posix_backend(tmp_path):
    result = run_cold_tier_adapter_bench(
        ColdTierAdapterBenchConfig(
            result_dir=tmp_path / "result",
            cold_root=tmp_path / "threefs_mount",
            cold_backend="3fs_posix",
            prefix_tokens=[16],
            repeats=1,
            layers=1,
            block_size=16,
            kv_heads=1,
            head_dim=2,
        )
    )

    assert result["status"] == "OK"
    assert result["cold_backend"] == "3fs_posix"
    rows = list(csv.DictReader(Path(result["csv_path"]).open()))
    assert rows[0]["cold_backend"] == "3fs_posix"
    prefix_id = rows[0]["prefix_id"]
    events = [
        json.loads(line)
        for line in (Path(result["tensor_store"]) / prefix_id / "migration_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["cold_backend"] == "3fs_posix"
    assert events[1]["cold_backend"] == "3fs_posix"


def test_cold_tier_adapter_bench_supports_packed_v1_backend(tmp_path):
    result = run_cold_tier_adapter_bench(
        ColdTierAdapterBenchConfig(
            result_dir=tmp_path / "result",
            cold_root=tmp_path / "packed",
            cold_backend="packed_v1",
            prefix_tokens=[16],
            repeats=1,
            layers=2,
            block_size=16,
            kv_heads=1,
            head_dim=2,
        )
    )

    assert result["status"] == "OK"
    assert result["cold_backend"] == "packed_v1"
    rows = list(csv.DictReader(Path(result["csv_path"]).open()))
    assert rows[0]["cold_backend"] == "packed_v1"
    assert rows[0]["checksum_status"] == "ok"
    prefix_id = rows[0]["prefix_id"]
    object_id = rows[0]["object_id"]
    assert (tmp_path / "packed" / object_id / "packed_object.bin").exists()
    assert (tmp_path / "packed" / object_id / "packed_manifest.json").exists()
    events = [
        json.loads(line)
        for line in (Path(result["tensor_store"]) / prefix_id / "migration_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["cold_backend"] == "packed_v1"
    assert events[1]["cold_backend"] == "packed_v1"


def test_cold_tier_adapter_bench_records_queue_depth_batches(tmp_path):
    result = run_cold_tier_adapter_bench(
        ColdTierAdapterBenchConfig(
            result_dir=tmp_path / "result",
            cold_root=tmp_path / "cold",
            cold_backend="local_posix",
            prefix_tokens=[16, 32],
            repeats=2,
            layers=1,
            block_size=16,
            kv_heads=1,
            head_dim=2,
            queue_depth=2,
        )
    )

    assert result["status"] == "OK"
    assert result["queue_depth"] == 2
    assert result["restore_batch_ms_p95"] >= result["restore_batch_ms_p50"]
    assert result["restore_effective_mib_per_s_p50"] >= 0
    rows = list(csv.DictReader(Path(result["csv_path"]).open()))
    assert set(rows[0]) >= {
        "batch_id",
        "queue_depth",
        "restore_batch_ms",
        "restore_effective_mib_per_s",
    }
    assert {row["queue_depth"] for row in rows} == {"2"}
    assert all(float(row["restore_batch_ms"]) >= float(row["restore_ms"]) for row in rows)


def test_cold_tier_adapter_bench_qwen25_profile_sets_shape_defaults(tmp_path):
    result = run_cold_tier_adapter_bench(
        ColdTierAdapterBenchConfig(
            result_dir=tmp_path / "result",
            cold_root=tmp_path / "cold",
            cold_backend="local_posix",
            prefix_tokens=[16],
            repeats=1,
            block_size=16,
            profile="qwen25_14b_tiny",
        )
    )

    assert result["profile"] == "qwen25_14b_tiny"
    assert result["layers"] == 4
    assert result["kv_heads"] == 8
    assert result["head_dim"] == 128
    assert result["dtype"] == "bfloat16"


def test_cold_tier_adapter_bench_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_cold_tier_adapter_bench.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--cold-backend" in completed.stdout
    assert "--prefix-tokens" in completed.stdout
    assert "--queue-depth" in completed.stdout
    assert "--profile" in completed.stdout
