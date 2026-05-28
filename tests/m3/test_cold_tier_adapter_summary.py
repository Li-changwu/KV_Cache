import csv
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.summarize_cold_tier_adapter_bench import summarize_adapter_benches


def _write_summary(path: Path, backend: str, restore_p95: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "OK",
                "rows": 2,
                "profile": "synthetic",
                "queue_depth": 2,
                "cold_backend": backend,
                "cold_root": f"/tmp/{backend}",
                "bytes_total": 1024,
                "demote_ms_p50": 1.0,
                "demote_ms_p95": 2.0,
                "demote_ms_p99": 3.0,
                "restore_ms_p50": 4.0,
                "restore_ms_p95": restore_p95,
                "restore_ms_p99": 6.0,
                "restore_batch_ms_p50": 7.0,
                "restore_batch_ms_p95": 8.0,
                "restore_batch_ms_p99": 9.0,
                "demote_mib_per_s_p50": 7.0,
                "restore_mib_per_s_p50": 8.0,
                "restore_effective_mib_per_s_p50": 9.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_adapter_benches_writes_comparison_outputs(tmp_path):
    local = tmp_path / "local" / "adapter_bench_summary.json"
    threefs = tmp_path / "threefs" / "adapter_bench_summary.json"
    _write_summary(local, "local_posix", 5.0)
    _write_summary(threefs, "3fs_posix", 9.0)

    result = summarize_adapter_benches(
        [local, threefs],
        result_dir=tmp_path / "summary",
    )

    assert result["status"] == "OK"
    assert result["rows"] == 2
    rows = list(csv.DictReader(Path(result["csv_path"]).open()))
    assert [row["cold_backend"] for row in rows] == ["local_posix", "3fs_posix"]
    assert rows[1]["restore_ms_p95"] == "9.0"
    assert rows[1]["queue_depth"] == "2"
    assert rows[1]["restore_batch_ms_p95"] == "8.0"
    assert rows[1]["restore_effective_mib_per_s_p50"] == "9.0"
    assert Path(result["report_path"]).exists()


def test_summarize_adapter_bench_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/summarize_cold_tier_adapter_bench.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--summary" in completed.stdout
