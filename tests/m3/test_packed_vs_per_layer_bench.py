import csv
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_packed_vs_per_layer_bench import (
    PackedVsPerLayerBenchConfig,
    run_packed_vs_per_layer_bench,
)


def test_packed_vs_per_layer_bench_writes_comparison_outputs(tmp_path):
    result = run_packed_vs_per_layer_bench(
        PackedVsPerLayerBenchConfig(
            result_dir=tmp_path / "result",
            prefix_tokens=[16],
            repeats=1,
            layers=1,
            block_size=16,
            kv_heads=1,
            head_dim=2,
            queue_depth=1,
        )
    )

    assert result["status"] == "OK"
    assert result["rows"] == 2
    assert result["prefix_tokens"] == [16]
    assert set(result["backends"]) == {"local_posix", "packed_v1"}
    assert Path(result["csv_path"]).exists()
    assert Path(result["by_token_csv_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["summary_path"]).exists()

    rows = list(csv.DictReader(Path(result["csv_path"]).open()))
    assert [row["cold_backend"] for row in rows] == ["local_posix", "packed_v1"]
    assert all(row["restore_ms_p95"] for row in rows)
    assert all(row["cold_file_count"] for row in rows)

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["restore_p95_delta_ms"] is not None
    assert "packed_vs_per_layer_report.md" in summary["report_path"]

    by_token_rows = list(csv.DictReader(Path(result["by_token_csv_path"]).open()))
    assert [row["tokens"] for row in by_token_rows] == ["16", "16"]
    assert [row["cold_backend"] for row in by_token_rows] == [
        "local_posix",
        "packed_v1",
    ]


def test_packed_vs_per_layer_bench_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/run_packed_vs_per_layer_bench.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--prefix-tokens" in completed.stdout
    assert "--profile" in completed.stdout
