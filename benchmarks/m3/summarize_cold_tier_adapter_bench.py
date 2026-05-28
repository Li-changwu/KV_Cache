#!/usr/bin/env python3
"""Summarize M3 cold-tier adapter benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def summarize_adapter_benches(
    summaries: list[Path],
    *,
    result_dir: Path,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row_from_summary(path) for path in summaries]
    csv_path = result_dir / "adapter_bench_comparison.csv"
    report_path = result_dir / "adapter_bench_comparison.md"
    summary_path = result_dir / "adapter_bench_comparison_summary.json"
    _write_csv(csv_path, rows)
    _write_report(report_path, rows)
    result = {
        "status": "OK" if rows else "EMPTY",
        "rows": len(rows),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _row_from_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "cold_backend": payload.get("cold_backend", ""),
        "profile": payload.get("profile", ""),
        "queue_depth": payload.get("queue_depth", 1),
        "cold_root": payload.get("cold_root", ""),
        "status": payload.get("status", ""),
        "rows": payload.get("rows", 0),
        "bytes_total": payload.get("bytes_total", 0),
        "demote_ms_p50": payload.get("demote_ms_p50", 0),
        "demote_ms_p95": payload.get("demote_ms_p95", 0),
        "demote_ms_p99": payload.get("demote_ms_p99", 0),
        "restore_ms_p50": payload.get("restore_ms_p50", 0),
        "restore_ms_p95": payload.get("restore_ms_p95", 0),
        "restore_ms_p99": payload.get("restore_ms_p99", 0),
        "restore_batch_ms_p50": payload.get("restore_batch_ms_p50", 0),
        "restore_batch_ms_p95": payload.get("restore_batch_ms_p95", 0),
        "restore_batch_ms_p99": payload.get("restore_batch_ms_p99", 0),
        "demote_mib_per_s_p50": payload.get("demote_mib_per_s_p50", 0),
        "restore_mib_per_s_p50": payload.get("restore_mib_per_s_p50", 0),
        "restore_effective_mib_per_s_p50": payload.get(
            "restore_effective_mib_per_s_p50",
            0,
        ),
        "summary_path": str(path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "cold_backend",
        "profile",
        "queue_depth",
        "cold_root",
        "status",
        "rows",
        "bytes_total",
        "demote_ms_p50",
        "demote_ms_p95",
        "demote_ms_p99",
        "restore_ms_p50",
        "restore_ms_p95",
        "restore_ms_p99",
        "restore_batch_ms_p50",
        "restore_batch_ms_p95",
        "restore_batch_ms_p99",
        "demote_mib_per_s_p50",
        "restore_mib_per_s_p50",
        "restore_effective_mib_per_s_p50",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# M3.10 Cold-Tier Adapter Benchmark Comparison",
        "",
        "| backend | profile | qd | rows | bytes | demote p95 ms | restore p95 ms | batch p95 ms | effective restore p50 MiB/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['cold_backend']}` | `{row['profile']}` | "
            f"{row['queue_depth']} | {row['rows']} | {row['bytes_total']} | "
            f"{row['demote_ms_p95']} | {row['restore_ms_p95']} | "
            f"{row['restore_batch_ms_p95']} | "
            f"{row['restore_effective_mib_per_s_p50']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument(
        "--result-dir",
        default="results/m3_10_cold_tier_adapter_bench/summary",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize_adapter_benches(
        [Path(item) for item in args.summary],
        result_dir=Path(args.result_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
