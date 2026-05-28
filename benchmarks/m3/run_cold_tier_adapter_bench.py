#!/usr/bin/env python3
"""Benchmark M3 cold-tier adapters with synthetic KV objects."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.cold_tier import build_cold_tier_adapter
from benchmarks.m3.tensor_store import KVTensorStore, block_slot_mapping


@dataclass(frozen=True)
class ColdTierAdapterBenchConfig:
    result_dir: Path
    cold_root: Path
    cold_backend: str = "local_posix"
    prefix_tokens: list[int] | None = None
    repeats: int = 3
    layers: int = 4
    block_size: int = 16
    kv_heads: int = 2
    head_dim: int = 8
    dtype: str = "float32"
    queue_depth: int = 1
    profile: str = "synthetic"


def run_cold_tier_adapter_bench(
    config: ColdTierAdapterBenchConfig,
) -> dict[str, Any]:
    result_dir = config.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    tensor_store = result_dir / "tensor_store"
    cold_root = config.cold_root
    config = _apply_profile(config)
    prefixes = config.prefix_tokens or [16, 64, 128, 256]
    adapter = build_cold_tier_adapter(config.cold_backend, cold_root)
    store = KVTensorStore(tensor_store, block_size=config.block_size)
    rows: list[dict[str, Any]] = []

    work_items = [
        (int(token_count), repeat)
        for token_count in prefixes
        for repeat in range(config.repeats)
    ]
    batch_size = max(1, int(config.queue_depth))
    batch_id = 0
    for batch in _chunked(work_items, batch_size):
        pending_rows: list[dict[str, Any]] = []
        for token_count, repeat in batch:
            prefix_id = f"adapter-{token_count}-r{repeat}"
            _write_synthetic_prefix(store, config, prefix_id, token_count)
            demote_started = time.perf_counter()
            demoted = store.demote_to_cold_object(
                prefix_id,
                cold_adapter=adapter,
                tier="NVME",
            )
            demote_ms = _elapsed_ms(demote_started)
            pending_rows.append(
                {
                    "prefix_id": prefix_id,
                    "tokens": token_count,
                    "repeat": repeat,
                    "batch_id": batch_id,
                    "queue_depth": batch_size,
                    "cold_backend": config.cold_backend,
                    "cold_root": str(cold_root),
                    "cold_uri": demoted.cold_uri or "",
                    "object_id": demoted.object_id or "",
                    "size_bytes": demoted.size_bytes,
                    "demote_ms": demote_ms,
                    "demote_mib_per_s": _mib_per_s(demoted.size_bytes, demote_ms),
                }
            )

        batch_started = time.perf_counter()
        restored_by_prefix = _restore_batch(
            tensor_store=tensor_store,
            block_size=config.block_size,
            adapter=adapter,
            prefix_ids=[str(row["prefix_id"]) for row in pending_rows],
            max_workers=batch_size,
        )
        restore_batch_ms = _elapsed_ms(batch_started)
        batch_bytes = sum(int(row["size_bytes"]) for row in pending_rows)
        effective_mib_per_s = _mib_per_s(batch_bytes, restore_batch_ms)

        for row in pending_rows:
            restored_payload = restored_by_prefix[str(row["prefix_id"])]
            row.update(
                {
                    "restore_ms": restored_payload["restore_ms"],
                    "restore_batch_ms": restore_batch_ms,
                    "restore_mib_per_s": _mib_per_s(
                        int(row["size_bytes"]),
                        float(restored_payload["restore_ms"]),
                    ),
                    "restore_effective_mib_per_s": effective_mib_per_s,
                    "checksum_status": restored_payload["checksum_status"],
                    "hot_files_restored": restored_payload["hot_files_restored"],
                }
            )
            rows.append(row)
        batch_id += 1

    csv_path = result_dir / "adapter_bench.csv"
    summary_path = result_dir / "adapter_bench_summary.json"
    report_path = result_dir / "adapter_bench_report.md"
    _write_csv(csv_path, rows)
    summary = _summarize(rows, config, tensor_store, csv_path, summary_path, report_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, rows, summary)
    return summary


def _write_synthetic_prefix(
    store: KVTensorStore,
    config: ColdTierAdapterBenchConfig,
    prefix_id: str,
    token_count: int,
) -> None:
    _validate_block_aligned(token_count, config.block_size)
    num_blocks = token_count // config.block_size
    slot_mapping = block_slot_mapping(
        list(range(num_blocks)),
        block_size=config.block_size,
        num_tokens=token_count,
    )
    dtype = _torch_dtype(config.dtype)
    shape = (
        num_blocks,
        2,
        config.block_size,
        config.kv_heads,
        config.head_dim,
    )
    total = 1
    for dim in shape:
        total *= dim
    base = torch.arange(total, dtype=torch.float32).reshape(shape).to(dtype=dtype)
    correctness_key = {
        "model_fingerprint": "synthetic",
        "tokenizer_fingerprint": "synthetic",
        "rope_config": "native",
        "dtype": str(dtype).replace("torch.", ""),
        "kv_layout": "NHD",
    }
    for layer_idx in range(config.layers):
        store.save_layer(
            prefix_id=prefix_id,
            token_start=0,
            token_end=token_count,
            correctness_key=correctness_key,
            token_ids=list(range(token_count)),
            layer_name=f"layer.{layer_idx}",
            kv_layer=base + layer_idx,
            slot_mapping=slot_mapping,
            layout="NHD",
        )


def _summarize(
    rows: list[dict[str, Any]],
    config: ColdTierAdapterBenchConfig,
    tensor_store: Path,
    csv_path: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    demote = [float(row["demote_ms"]) for row in rows]
    restore = [float(row["restore_ms"]) for row in rows]
    restore_batch = [float(row["restore_batch_ms"]) for row in rows]
    demote_bw = [float(row["demote_mib_per_s"]) for row in rows]
    restore_bw = [float(row["restore_mib_per_s"]) for row in rows]
    restore_effective_bw = [
        float(row["restore_effective_mib_per_s"]) for row in rows
    ]
    return {
        "status": "OK" if rows else "EMPTY",
        "rows": len(rows),
        "profile": config.profile,
        "layers": config.layers,
        "block_size": config.block_size,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "dtype": config.dtype,
        "queue_depth": max(1, int(config.queue_depth)),
        "cold_backend": config.cold_backend,
        "cold_root": str(config.cold_root),
        "tensor_store": str(tensor_store),
        "bytes_total": sum(int(row["size_bytes"]) for row in rows),
        "demote_ms_p50": _percentile(demote, 50),
        "demote_ms_p95": _percentile(demote, 95),
        "demote_ms_p99": _percentile(demote, 99),
        "restore_ms_p50": _percentile(restore, 50),
        "restore_ms_p95": _percentile(restore, 95),
        "restore_ms_p99": _percentile(restore, 99),
        "restore_batch_ms_p50": _percentile(restore_batch, 50),
        "restore_batch_ms_p95": _percentile(restore_batch, 95),
        "restore_batch_ms_p99": _percentile(restore_batch, 99),
        "demote_mib_per_s_p50": _percentile(demote_bw, 50),
        "restore_mib_per_s_p50": _percentile(restore_bw, 50),
        "restore_effective_mib_per_s_p50": _percentile(restore_effective_bw, 50),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "tokens",
        "repeat",
        "batch_id",
        "queue_depth",
        "cold_backend",
        "cold_root",
        "cold_uri",
        "object_id",
        "size_bytes",
        "demote_ms",
        "restore_ms",
        "restore_batch_ms",
        "demote_mib_per_s",
        "restore_mib_per_s",
        "restore_effective_mib_per_s",
        "checksum_status",
        "hot_files_restored",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# M3.10 Cold-Tier Adapter Benchmark",
        "",
        f"- backend: `{summary['cold_backend']}`",
        f"- profile: `{summary['profile']}`",
        f"- queue depth: `{summary['queue_depth']}`",
        f"- cold root: `{summary['cold_root']}`",
        f"- rows: `{summary['rows']}`",
        f"- bytes total: `{summary['bytes_total']}`",
        f"- demote p50/p95/p99 ms: `{summary['demote_ms_p50']}` / "
        f"`{summary['demote_ms_p95']}` / `{summary['demote_ms_p99']}`",
        f"- restore p50/p95/p99 ms: `{summary['restore_ms_p50']}` / "
        f"`{summary['restore_ms_p95']}` / `{summary['restore_ms_p99']}`",
        f"- restore batch p50/p95/p99 ms: `{summary['restore_batch_ms_p50']}` / "
        f"`{summary['restore_batch_ms_p95']}` / `{summary['restore_batch_ms_p99']}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row['prefix_id']}` tokens=`{row['tokens']}` "
            f"bytes=`{row['size_bytes']}` "
            f"demote_ms=`{row['demote_ms']}` "
            f"restore_ms=`{row['restore_ms']}` "
            f"batch_ms=`{row['restore_batch_ms']}` "
            f"checksum=`{row['checksum_status']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _restore_batch(
    *,
    tensor_store: Path,
    block_size: int,
    adapter: Any,
    prefix_ids: list[str],
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _restore_one,
                tensor_store=tensor_store,
                block_size=block_size,
                adapter=adapter,
                prefix_id=prefix_id,
            ): prefix_id
            for prefix_id in prefix_ids
        }
        for future in as_completed(futures):
            prefix_id = futures[future]
            results[prefix_id] = future.result()
    return results


def _restore_one(
    *,
    tensor_store: Path,
    block_size: int,
    adapter: Any,
    prefix_id: str,
) -> dict[str, Any]:
    store = KVTensorStore(tensor_store, block_size=block_size)
    started = time.perf_counter()
    restored = store.restore_from_cold_object(prefix_id, cold_adapter=adapter)
    restore_ms = _elapsed_ms(started)
    _, restore_event = _latest_cold_events(store.read_migration_events(prefix_id))
    hot_files_restored = all(
        (tensor_store / prefix_id / record.file_name).exists()
        for record in restored.layers.values()
    )
    return {
        "restore_ms": restore_ms,
        "checksum_status": str(restore_event.get("checksum_status", "")),
        "hot_files_restored": "yes" if hot_files_restored else "no",
    }


def _chunked(items: list[tuple[int, int]], size: int) -> list[list[tuple[int, int]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _apply_profile(config: ColdTierAdapterBenchConfig) -> ColdTierAdapterBenchConfig:
    if config.profile == "synthetic":
        return config
    if config.profile == "qwen25_14b_tiny":
        return ColdTierAdapterBenchConfig(
            result_dir=config.result_dir,
            cold_root=config.cold_root,
            cold_backend=config.cold_backend,
            prefix_tokens=config.prefix_tokens,
            repeats=config.repeats,
            layers=4,
            block_size=config.block_size,
            kv_heads=8,
            head_dim=128,
            dtype="bfloat16",
            queue_depth=config.queue_depth,
            profile=config.profile,
        )
    raise ValueError(f"unsupported profile: {config.profile}")


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


def _mib_per_s(size_bytes: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return round((size_bytes / (1024 * 1024)) / (elapsed_ms / 1000.0), 3)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _validate_block_aligned(token_count: int, block_size: int) -> None:
    if token_count <= 0 or token_count % block_size != 0:
        raise ValueError(
            f"token count must be positive and block aligned: "
            f"tokens={token_count}, block_size={block_size}"
        )


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="results/m3_10_cold_tier_adapter_bench",
    )
    parser.add_argument("--cold-root", required=True)
    parser.add_argument(
        "--cold-backend",
        default="local_posix",
        choices=["local_posix", "3fs_posix", "packed_v1"],
    )
    parser.add_argument("--prefix-tokens", nargs="+", type=int, default=[16, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=["synthetic", "qwen25_14b_tiny"],
        default="synthetic",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_cold_tier_adapter_bench(
        ColdTierAdapterBenchConfig(
            result_dir=Path(args.result_dir),
            cold_root=Path(args.cold_root),
            cold_backend=args.cold_backend,
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
