#!/usr/bin/env python3
"""Run an offline DRAM/NVMe tier migration smoke over stored KV manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.cold_tier import build_cold_tier_adapter
from benchmarks.m3.tensor_store import KVTensorStore


def run_tier_migration_smoke(
    *,
    tensor_store: Path,
    result_dir: Path,
    cold_root: Path | None = None,
    cold_backend: str = "local_posix",
    prefix_ids: list[str] | None = None,
    block_size: int = 16,
) -> dict[str, Any]:
    store = KVTensorStore(tensor_store, block_size=block_size)
    result_dir.mkdir(parents=True, exist_ok=True)
    effective_cold_root = cold_root or (result_dir / "cold_objects")
    cold_adapter = build_cold_tier_adapter(cold_backend, effective_cold_root)
    selected = prefix_ids or _discover_prefix_ids(tensor_store)
    rows: list[dict[str, Any]] = []
    for prefix_id in selected:
        initial = store.read_manifest(prefix_id)
        demoted = store.demote_to_cold_object(
            prefix_id,
            cold_adapter=cold_adapter,
            tier="NVME",
        )
        prefetched = store.restore_from_cold_object(
            prefix_id,
            cold_adapter=cold_adapter,
        )
        demote_event, restore_event = _latest_cold_events(
            store.read_migration_events(prefix_id)
        )
        rows.append(
            {
                "prefix_id": prefix_id,
                "tokens": initial.tokens,
                "initial_tier": initial.tier,
                "initial_ready": initial.ready,
                "demoted_tier": demoted.tier,
                "demoted_ready": demoted.ready,
                "prefetched_tier": prefetched.tier,
                "prefetched_ready": prefetched.ready,
                "object_id": demoted.object_id or "",
                "cold_uri": demoted.cold_uri or "",
                "cold_backend": cold_backend,
                "size_bytes": demoted.size_bytes,
                "demote_elapsed_ms": demote_event.get("elapsed_ms", ""),
                "restore_elapsed_ms": restore_event.get("elapsed_ms", ""),
                "checksum_status": restore_event.get("checksum_status", ""),
                "migration_events": len(store.read_migration_events(prefix_id)),
            }
        )
    _write_csv(result_dir / "tier_migration_smoke.csv", rows)
    _write_report(
        result_dir / "tier_migration_report.md",
        rows,
        tensor_store,
        effective_cold_root,
        cold_backend,
    )
    summary = {
        "status": "OK" if rows else "EMPTY",
        "prefixes": len(rows),
        "tensor_store": str(tensor_store),
        "cold_root": str(effective_cold_root),
        "cold_backend": cold_backend,
        "bytes_total": sum(int(row["size_bytes"]) for row in rows),
        "csv_path": str(result_dir / "tier_migration_smoke.csv"),
        "report_path": str(result_dir / "tier_migration_report.md"),
    }
    (result_dir / "tier_migration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _discover_prefix_ids(tensor_store: Path) -> list[str]:
    if not tensor_store.exists():
        return []
    return sorted(path.parent.name for path in tensor_store.glob("*/manifest.json"))


def _latest_cold_events(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    demote: dict[str, Any] = {}
    restore: dict[str, Any] = {}
    for event in events:
        if event.get("event") == "demote_to_cold_object":
            demote = event
        if event.get("event") == "restore_from_cold_object":
            restore = event
    return demote, restore


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "initial_tier",
        "demoted_tier",
        "prefetched_tier",
        "tokens",
        "initial_ready",
        "demoted_ready",
        "prefetched_ready",
        "object_id",
        "cold_uri",
        "cold_backend",
        "size_bytes",
        "demote_elapsed_ms",
        "restore_elapsed_ms",
        "checksum_status",
        "migration_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    tensor_store: Path,
    cold_root: Path,
    cold_backend: str,
) -> None:
    lines = [
        "# M3.10-A 真实 Cold-Tier 迁移冒烟报告",
        "",
        f"- 张量存储：`{tensor_store}`",
        f"- 冷层目录：`{cold_root}`",
        f"- 冷层后端：`{cold_backend}`",
        f"- 前缀数量：`{len(rows)}`",
        f"- 迁移字节数：`{sum(int(row['size_bytes']) for row in rows)}`",
        "",
        "## 明细",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row['prefix_id']}`："
            f"{row['initial_tier']} -> {row['demoted_tier']} -> "
            f"{row['prefetched_tier']}，tokens=`{row['tokens']}`，"
            f"bytes=`{row['size_bytes']}`，"
            f"checksum=`{row['checksum_status']}`，"
            f"events=`{row['migration_events']}`"
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
        default="results/m3_10_cold_tier_migration_smoke",
    )
    parser.add_argument("--cold-root")
    parser.add_argument(
        "--cold-backend",
        default="local_posix",
        choices=["local_posix", "3fs_posix", "packed_v1"],
    )
    parser.add_argument("--prefix-ids", nargs="*")
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_tier_migration_smoke(
        tensor_store=Path(args.tensor_store),
        result_dir=Path(args.result_dir),
        cold_root=Path(args.cold_root) if args.cold_root else None,
        cold_backend=args.cold_backend,
        prefix_ids=args.prefix_ids,
        block_size=args.block_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
