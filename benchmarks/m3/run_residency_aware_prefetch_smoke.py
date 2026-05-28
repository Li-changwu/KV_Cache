#!/usr/bin/env python3
"""Run a residency-aware prefetch smoke over local tensor-store manifests."""

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
from benchmarks.m3.tensor_store import KVBlockManifest, KVTensorStore


def run_residency_aware_prefetch_smoke(
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
    store = KVTensorStore(tensor_store, block_size)
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=block_size,
        kv_bytes_per_token=kv_bytes_per_token,
        storage_gbps=storage_gbps,
        event_log_path=result_dir / "prefetch_queue_events.jsonl",
    )
    rows: list[dict[str, Any]] = []
    for prefix_id in selected:
        manifest = store.read_manifest(prefix_id)
        if _is_execution_ready(manifest):
            rows.append(
                _residency_hit_row(
                    manifest,
                    deadline_ms,
                    kv_bytes_per_token=kv_bytes_per_token,
                )
            )
            continue
        result = queue.submit(prefix_id=prefix_id, deadline_ms=deadline_ms)
        rows.append(_prefetch_row(result))

    _write_csv(result_dir / "residency_aware_prefetch_smoke.csv", rows)
    summary = _summary(
        tensor_store=tensor_store,
        result_dir=result_dir,
        rows=rows,
        queue_stats=queue.stats(),
    )
    (result_dir / "residency_aware_prefetch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_dir / "residency_aware_prefetch_report.md", rows, summary)
    return summary


def _discover_prefix_ids(tensor_store: Path) -> list[str]:
    if not tensor_store.exists():
        return []
    return sorted(path.parent.name for path in tensor_store.glob("*/manifest.json"))


def _is_execution_ready(manifest: KVBlockManifest) -> bool:
    return manifest.ready and manifest.tier in {"HBM", "DRAM"}


def _residency_hit_row(
    manifest: KVBlockManifest,
    deadline_ms: float,
    *,
    kv_bytes_per_token: int,
) -> dict[str, Any]:
    bytes_ready = int(manifest.tokens * kv_bytes_per_token)
    return {
        "prefix_id": manifest.prefix_id,
        "status": "RESIDENCY_HIT",
        "tokens": manifest.tokens,
        "bytes": bytes_ready,
        "queue_depth_before": "",
        "start_ms": 0.0,
        "duration_ms": 0.0,
        "estimated_ready_ms": 0.0,
        "deadline_ms": float(deadline_ms),
        "deadline_miss": False,
        "tier": manifest.tier,
        "ready": manifest.ready,
        "queued": False,
    }


def _prefetch_row(result: PrefetchResult) -> dict[str, Any]:
    status = (
        "PREFETCH_COMPLETED"
        if result.status == "COMPLETED"
        else result.status
    )
    return {
        "prefix_id": result.prefix_id,
        "status": status,
        "tokens": result.tokens,
        "bytes": result.bytes,
        "queue_depth_before": result.queue_depth_before,
        "start_ms": result.start_ms,
        "duration_ms": result.duration_ms,
        "estimated_ready_ms": result.estimated_ready_ms,
        "deadline_ms": result.deadline_ms,
        "deadline_miss": result.deadline_miss,
        "tier": result.tier,
        "ready": result.ready,
        "queued": True,
    }


def _summary(
    *,
    tensor_store: Path,
    result_dir: Path,
    rows: list[dict[str, Any]],
    queue_stats: dict[str, float | int],
) -> dict[str, Any]:
    residency_hits = sum(1 for row in rows if row["status"] == "RESIDENCY_HIT")
    completed = sum(1 for row in rows if row["status"] == "PREFETCH_COMPLETED")
    deadline_miss = sum(1 for row in rows if row["status"] == "DEADLINE_MISS")
    return {
        "status": "OK" if rows else "EMPTY",
        "tensor_store": str(tensor_store),
        "rows": len(rows),
        "residency_hits": residency_hits,
        "prefetch_completed": completed,
        "deadline_miss": deadline_miss,
        "queue_requests": int(queue_stats["prefetch_queue_requests_total"]),
        "queue_completed": int(queue_stats["prefetch_queue_completed_total"]),
        "queue_deadline_miss": int(
            queue_stats["prefetch_queue_deadline_miss_total"]
        ),
        "queue_bytes_total": int(queue_stats["prefetch_queue_bytes_total"]),
        "csv_path": str(result_dir / "residency_aware_prefetch_smoke.csv"),
        "report_path": str(result_dir / "residency_aware_prefetch_report.md"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "status",
        "tokens",
        "bytes",
        "queue_depth_before",
        "start_ms",
        "duration_ms",
        "estimated_ready_ms",
        "deadline_ms",
        "deadline_miss",
        "tier",
        "ready",
        "queued",
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
        "# M3.9 驻留感知预取烟测报告",
        "",
        f"- 行数：`{summary['rows']}`",
        f"- 驻留命中：`{summary['residency_hits']}`",
        f"- 预取完成：`{summary['prefetch_completed']}`",
        f"- 截止时间错过：`{summary['deadline_miss']}`",
        f"- 队列请求数：`{summary['queue_requests']}`",
        f"- 队列累计字节数：`{summary['queue_bytes_total']}`",
        "",
        "## 明细",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row['prefix_id']}`：`{row['status']}`，"
            f"tier=`{row['tier']}`，"
            f"ready=`{row['ready']}`，"
            f"queued=`{row['queued']}`，"
            f"estimated_ready=`{row['estimated_ready_ms']}`ms"
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
        default="results/m3_9_residency_aware_prefetch_smoke",
    )
    parser.add_argument("--prefix-ids", nargs="*")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-bytes-per-token", type=int, default=196_608)
    parser.add_argument("--storage-gbps", type=float, default=8.8)
    parser.add_argument("--deadline-ms", type=float, default=12_000.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_residency_aware_prefetch_smoke(
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
