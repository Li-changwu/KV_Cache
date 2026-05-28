#!/usr/bin/env python3
"""Run a deterministic multi-prefix prefetch queue smoke."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.prefetch_queue import PrefetchQueue, PrefetchResult


def run_prefetch_queue_smoke(
    *,
    tensor_store: Path,
    result_dir: Path,
    prefix_ids: list[str] | None = None,
    block_size: int = 16,
    kv_bytes_per_token: int = 196_608,
    storage_gbps: float = 8.8,
    deadline_ms: float = 12_000.0,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = prefix_ids or _discover_prefix_ids(tensor_store)
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=block_size,
        kv_bytes_per_token=kv_bytes_per_token,
        storage_gbps=storage_gbps,
        event_log_path=result_dir / "prefetch_queue_events.jsonl",
    )
    results = [
        queue.submit(prefix_id=prefix_id, deadline_ms=deadline_ms)
        for prefix_id in selected
    ]
    rows = [_result_to_row(result) for result in results]
    _write_csv(result_dir / "prefetch_queue_smoke.csv", rows)
    stats = queue.stats()
    summary = {
        "status": "OK" if results else "EMPTY",
        "tensor_store": str(tensor_store),
        "requests": int(stats["prefetch_queue_requests_total"]),
        "completed": int(stats["prefetch_queue_completed_total"]),
        "deadline_miss": int(stats["prefetch_queue_deadline_miss_total"]),
        "bytes_total": int(stats["prefetch_queue_bytes_total"]),
        "max_depth": int(stats["prefetch_queue_max_depth"]),
        "last_estimated_ready_ms": stats["prefetch_queue_last_estimated_ready_ms"],
        "csv_path": str(result_dir / "prefetch_queue_smoke.csv"),
        "report_path": str(result_dir / "prefetch_queue_report.md"),
    }
    (result_dir / "prefetch_queue_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_dir / "prefetch_queue_report.md", rows, summary)
    return summary


def _discover_prefix_ids(tensor_store: Path) -> list[str]:
    if not tensor_store.exists():
        return []
    return sorted(path.parent.name for path in tensor_store.glob("*/manifest.json"))


def _result_to_row(result: PrefetchResult) -> dict[str, Any]:
    return {
        "prefix_id": result.prefix_id,
        "status": result.status,
        "tokens": result.tokens,
        "bytes": result.bytes,
        "actual_bytes": result.actual_bytes,
        "queue_depth_before": result.queue_depth_before,
        "start_ms": result.start_ms,
        "duration_ms": result.duration_ms,
        "executor_elapsed_ms": result.executor_elapsed_ms,
        "estimated_ready_ms": result.estimated_ready_ms,
        "deadline_ms": result.deadline_ms,
        "deadline_miss": result.deadline_miss,
        "tier": result.tier,
        "ready": result.ready,
        "object_id": result.object_id,
        "checksum_status": result.checksum_status,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "status",
        "tokens",
        "bytes",
        "actual_bytes",
        "queue_depth_before",
        "start_ms",
        "duration_ms",
        "executor_elapsed_ms",
        "estimated_ready_ms",
        "deadline_ms",
        "deadline_miss",
        "tier",
        "ready",
        "object_id",
        "checksum_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# M3.9 预取队列多请求烟测报告",
        "",
        f"- 请求数：`{summary['requests']}`",
        f"- 完成数：`{summary['completed']}`",
        f"- 截止时间错过数：`{summary['deadline_miss']}`",
        f"- 总字节数：`{summary['bytes_total']}`",
        f"- 最大队列深度：`{summary['max_depth']}`",
        f"- 最近预计完成时间：`{summary['last_estimated_ready_ms']}` ms",
        "",
        "## 明细",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row['prefix_id']}`：`{row['status']}`，"
            f"tokens=`{row['tokens']}`，"
            f"actual_bytes=`{row['actual_bytes']}`，"
            f"start=`{row['start_ms']}`ms，"
            f"ready=`{row['estimated_ready_ms']}`ms，"
            f"executor=`{row['executor_elapsed_ms']}`ms，"
            f"checksum=`{row['checksum_status']}`，"
            f"deadline miss=`{row['deadline_miss']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-store",
        default="results/m3_8_full_restart_matrix/tensor_store",
    )
    parser.add_argument(
        "--result-dir",
        default="results/m3_9_prefetch_queue_smoke",
    )
    parser.add_argument("--prefix-ids", nargs="*")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-bytes-per-token", type=int, default=196_608)
    parser.add_argument("--storage-gbps", type=float, default=8.8)
    parser.add_argument("--deadline-ms", type=float, default=12_000.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_prefetch_queue_smoke(
        tensor_store=Path(args.tensor_store),
        result_dir=Path(args.result_dir),
        prefix_ids=args.prefix_ids,
        block_size=args.block_size,
        kv_bytes_per_token=args.kv_bytes_per_token,
        storage_gbps=args.storage_gbps,
        deadline_ms=args.deadline_ms,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
