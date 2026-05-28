#!/usr/bin/env python3
"""Run M3.14-D packed-vs-per-layer cold restore comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.run_cold_tier_adapter_bench import (
    ColdTierAdapterBenchConfig,
    run_cold_tier_adapter_bench,
)


@dataclass(frozen=True)
class PackedVsPerLayerBenchConfig:
    result_dir: Path
    prefix_tokens: list[int] | None = None
    repeats: int = 3
    layers: int = 4
    block_size: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"
    queue_depth: int = 1
    profile: str = "qwen25_14b_tiny"


def run_packed_vs_per_layer_bench(
    config: PackedVsPerLayerBenchConfig,
) -> dict[str, Any]:
    result_dir = config.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix_tokens = config.prefix_tokens or [2048, 8192]
    backend_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    by_token_rows: list[dict[str, Any]] = []
    for backend in ["local_posix", "packed_v1"]:
        backend_result_dir = result_dir / backend
        cold_root = backend_result_dir / "cold"
        summary = run_cold_tier_adapter_bench(
            ColdTierAdapterBenchConfig(
                result_dir=backend_result_dir,
                cold_root=cold_root,
                cold_backend=backend,
                prefix_tokens=prefix_tokens,
                repeats=config.repeats,
                layers=config.layers,
                block_size=config.block_size,
                kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                dtype=config.dtype,
                queue_depth=config.queue_depth,
                profile=config.profile,
            )
        )
        backend_summaries.append(summary)
        row = _row_from_summary(summary)
        row["cold_file_count"] = _count_cold_files(cold_root)
        row["cold_data_file_count"] = _count_cold_data_files(cold_root)
        rows.append(row)
        by_token_rows.extend(
            _by_token_rows(
                adapter_csv=Path(str(summary["csv_path"])),
                cold_root=cold_root,
            )
        )

    csv_path = result_dir / "packed_vs_per_layer_comparison.csv"
    by_token_csv_path = result_dir / "packed_vs_per_layer_by_token.csv"
    report_path = result_dir / "packed_vs_per_layer_report.md"
    summary_path = result_dir / "packed_vs_per_layer_summary.json"
    _write_csv(csv_path, rows)
    _write_by_token_csv(by_token_csv_path, by_token_rows)
    result = {
        "status": "OK" if rows else "EMPTY",
        "rows": len(rows),
        "backends": [row["cold_backend"] for row in rows],
        "prefix_tokens": [int(item) for item in prefix_tokens],
        "repeats": config.repeats,
        "profile": config.profile,
        "queue_depth": config.queue_depth,
        "restore_p95_delta_ms": _delta(
            rows,
            field="restore_ms_p95",
            minuend="packed_v1",
            subtrahend="local_posix",
        ),
        "restore_batch_p95_delta_ms": _delta(
            rows,
            field="restore_batch_ms_p95",
            minuend="packed_v1",
            subtrahend="local_posix",
        ),
        "cold_data_file_count_delta": _delta(
            rows,
            field="cold_data_file_count",
            minuend="packed_v1",
            subtrahend="local_posix",
        ),
        "csv_path": str(csv_path),
        "by_token_csv_path": str(by_token_csv_path),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "backend_summaries": backend_summaries,
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, rows, by_token_rows, result)
    return result


def _row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "cold_backend": summary.get("cold_backend", ""),
        "profile": summary.get("profile", ""),
        "queue_depth": summary.get("queue_depth", 1),
        "rows": summary.get("rows", 0),
        "bytes_total": summary.get("bytes_total", 0),
        "demote_ms_p50": summary.get("demote_ms_p50", 0),
        "demote_ms_p95": summary.get("demote_ms_p95", 0),
        "demote_ms_p99": summary.get("demote_ms_p99", 0),
        "restore_ms_p50": summary.get("restore_ms_p50", 0),
        "restore_ms_p95": summary.get("restore_ms_p95", 0),
        "restore_ms_p99": summary.get("restore_ms_p99", 0),
        "restore_batch_ms_p50": summary.get("restore_batch_ms_p50", 0),
        "restore_batch_ms_p95": summary.get("restore_batch_ms_p95", 0),
        "restore_batch_ms_p99": summary.get("restore_batch_ms_p99", 0),
        "restore_effective_mib_per_s_p50": summary.get(
            "restore_effective_mib_per_s_p50",
            0,
        ),
        "summary_path": summary.get("summary_path", ""),
    }


def _count_cold_files(cold_root: Path) -> int:
    if not cold_root.exists():
        return 0
    return sum(1 for path in cold_root.rglob("*") if path.is_file())


def _count_cold_data_files(cold_root: Path) -> int:
    if not cold_root.exists():
        return 0
    return sum(
        1
        for path in cold_root.rglob("*")
        if path.is_file() and path.name != "packed_manifest.json"
    )


def _by_token_rows(
    *,
    adapter_csv: Path,
    cold_root: Path,
) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(adapter_csv.open(newline="", encoding="utf-8")))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["tokens"]), []).append(row)
    result: list[dict[str, Any]] = []
    for tokens, items in sorted(grouped.items()):
        backend = items[0]["cold_backend"]
        result.append(
            {
                "tokens": tokens,
                "cold_backend": backend,
                "rows": len(items),
                "bytes_total": sum(int(item["size_bytes"]) for item in items),
                "cold_file_count": _count_files_for_tokens(cold_root, tokens),
                "cold_data_file_count": _count_data_files_for_tokens(cold_root, tokens),
                "demote_ms_p50": _percentile(
                    [float(item["demote_ms"]) for item in items],
                    50,
                ),
                "demote_ms_p95": _percentile(
                    [float(item["demote_ms"]) for item in items],
                    95,
                ),
                "restore_ms_p50": _percentile(
                    [float(item["restore_ms"]) for item in items],
                    50,
                ),
                "restore_ms_p95": _percentile(
                    [float(item["restore_ms"]) for item in items],
                    95,
                ),
                "restore_ms_p99": _percentile(
                    [float(item["restore_ms"]) for item in items],
                    99,
                ),
                "restore_effective_mib_per_s_p50": _percentile(
                    [float(item["restore_effective_mib_per_s"]) for item in items],
                    50,
                ),
            }
        )
    return result


def _count_files_for_tokens(cold_root: Path, tokens: int) -> int:
    return _count_matching_files(cold_root, f"adapter-{tokens}-r")


def _count_data_files_for_tokens(cold_root: Path, tokens: int) -> int:
    return _count_matching_files(
        cold_root,
        f"adapter-{tokens}-r",
        exclude_names={"packed_manifest.json"},
    )


def _count_matching_files(
    cold_root: Path,
    prefix: str,
    *,
    exclude_names: set[str] | None = None,
) -> int:
    if not cold_root.exists():
        return 0
    excluded = exclude_names or set()
    count = 0
    for path in cold_root.rglob("*"):
        if path.is_file() and prefix in str(path.relative_to(cold_root)):
            if path.name not in excluded:
                count += 1
    return count


def _delta(
    rows: list[dict[str, Any]],
    *,
    field: str,
    minuend: str,
    subtrahend: str,
) -> float | None:
    by_backend = {str(row["cold_backend"]): row for row in rows}
    if minuend not in by_backend or subtrahend not in by_backend:
        return None
    return round(
        float(by_backend[minuend][field]) - float(by_backend[subtrahend][field]),
        3,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "cold_backend",
        "profile",
        "queue_depth",
        "rows",
        "bytes_total",
        "cold_file_count",
        "cold_data_file_count",
        "demote_ms_p50",
        "demote_ms_p95",
        "demote_ms_p99",
        "restore_ms_p50",
        "restore_ms_p95",
        "restore_ms_p99",
        "restore_batch_ms_p50",
        "restore_batch_ms_p95",
        "restore_batch_ms_p99",
        "restore_effective_mib_per_s_p50",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_by_token_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "tokens",
        "cold_backend",
        "rows",
        "bytes_total",
        "cold_file_count",
        "cold_data_file_count",
        "demote_ms_p50",
        "demote_ms_p95",
        "restore_ms_p50",
        "restore_ms_p95",
        "restore_ms_p99",
        "restore_effective_mib_per_s_p50",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    by_token_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# M3.14-D Packed vs Per-Layer Restore",
        "",
        f"- profile: `{summary['profile']}`",
        f"- prefix tokens: `{summary['prefix_tokens']}`",
        f"- repeats: `{summary['repeats']}`",
        f"- queue depth: `{summary['queue_depth']}`",
        f"- packed restore p95 delta ms: `{summary['restore_p95_delta_ms']}`",
        f"- packed cold data file count delta: `{summary['cold_data_file_count_delta']}`",
        "",
        "| backend | rows | bytes | cold data files | restore p50 ms | restore p95 ms | restore p99 ms | batch p95 ms | effective restore p50 MiB/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['cold_backend']}` | {row['rows']} | {row['bytes_total']} | "
            f"{row['cold_data_file_count']} | {row['restore_ms_p50']} | "
            f"{row['restore_ms_p95']} | {row['restore_ms_p99']} | "
            f"{row['restore_batch_ms_p95']} | "
            f"{row['restore_effective_mib_per_s_p50']} |"
        )
    lines.extend(
        [
            "",
            "## By Token Length",
            "",
            "| tokens | backend | rows | cold data files | restore p50 ms | restore p95 ms | restore p99 ms | effective restore p50 MiB/s |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_token_rows:
        lines.append(
            f"| {row['tokens']} | `{row['cold_backend']}` | {row['rows']} | "
            f"{row['cold_data_file_count']} | {row['restore_ms_p50']} | "
            f"{row['restore_ms_p95']} | {row['restore_ms_p99']} | "
            f"{row['restore_effective_mib_per_s_p50']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/m3_14_packed_vs_per_layer")
    parser.add_argument("--prefix-tokens", nargs="+", type=int, default=[2048, 8192])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=["synthetic", "qwen25_14b_tiny"],
        default="qwen25_14b_tiny",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_packed_vs_per_layer_bench(
        PackedVsPerLayerBenchConfig(
            result_dir=Path(args.result_dir),
            prefix_tokens=args.prefix_tokens,
            repeats=args.repeats,
            layers=args.layers,
            block_size=args.block_size,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=args.dtype,
            queue_depth=args.queue_depth,
            profile=args.profile,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
