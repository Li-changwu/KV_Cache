#!/usr/bin/env python3
"""Run a deterministic async sidecar prefetch smoke over a tensor store."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.http_sidecar import M3SidecarConfig, SidecarRuntime
from benchmarks.m3.tensor_store import KVTensorStore


DEFAULT_MODEL_ID = "/root/models/Qwen2.5-14B-Instruct"


def run_async_prefetch_smoke(
    *,
    tensor_store: Path,
    result_dir: Path,
    prefix_id: str,
    token_end: int | None = None,
    request_token_count: int | None = None,
    block_size: int = 16,
    kv_bytes_per_token: int = 196_608,
    hbm_capacity_tokens: int = 65_000,
    dram_capacity_tokens: int = 1_000_000,
    ssd_capacity_tokens: int = 8_000_000,
    h2d_gbps: float = 25.0,
    storage_gbps: float = 8.8,
    advance_ms: float = 12_000.0,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    store = KVTensorStore(tensor_store, block_size=block_size)
    manifest = store.read_manifest(prefix_id)
    effective_token_end = int(token_end or manifest.token_end)
    effective_token_count = int(request_token_count or effective_token_end + block_size)
    store.demote_to_nvme(prefix_id)

    runtime = SidecarRuntime(
        M3SidecarConfig(
            model_id=DEFAULT_MODEL_ID,
            hbm_capacity_tokens=hbm_capacity_tokens,
            dram_capacity_tokens=dram_capacity_tokens,
            ssd_capacity_tokens=ssd_capacity_tokens,
            kv_bytes_per_token=kv_bytes_per_token,
            h2d_gbps=h2d_gbps,
            storage_gbps=storage_gbps,
            decision_log_path=result_dir / "decisions.jsonl",
            tensor_store_path=tensor_store,
            tensor_store_block_size=block_size,
            async_prefetch=True,
        )
    )
    commit_response = runtime.admit_from_payload(
        _payload(
            request_id="async-prefetch-commit",
            token_count=effective_token_end,
            prefix_id=prefix_id,
            prefix_end=0,
        ),
        source="async_smoke_commit_seed",
    )
    runtime.control_plane.commit_request(
        runtime.responses[commit_response["request_id"]],
        prefix_id=prefix_id,
        token_end=effective_token_end,
        tier="SSD",
        ready=False,
    )

    first = runtime.admit_from_payload(
        _payload(
            request_id="async-prefetch-first",
            token_count=effective_token_count,
            prefix_id=prefix_id,
            prefix_end=effective_token_end,
        ),
        source="async_smoke_first_admit",
    )
    second = runtime.admit_from_payload(
        _payload(
            request_id="async-prefetch-second",
            token_count=effective_token_count,
            prefix_id=prefix_id,
            prefix_end=effective_token_end,
        ),
        source="async_smoke_second_before_advance",
    )
    advance = runtime.advance_prefetch_from_payload({"max_ready_ms": advance_ms})
    third = runtime.admit_from_payload(
        _payload(
            request_id="async-prefetch-third",
            token_count=effective_token_count,
            prefix_id=prefix_id,
            prefix_end=effective_token_end,
        ),
        source="async_smoke_third_after_advance",
    )
    final_manifest = store.read_manifest(prefix_id)
    queue_stats = runtime.prefetch_queue.stats() if runtime.prefetch_queue else {}
    queue_results = _prefetch_results(result_dir / "decisions.jsonl")
    queued_status = _first_status(queue_results, "QUEUED")
    completed = list(advance.get("completed") or [])
    completed_status = str(completed[0]["status"]) if completed else "NOT_COMPLETED"
    completed_result = completed[0] if completed else {}
    row = {
        "prefix_id": prefix_id,
        "queued_status": queued_status,
        "completed_status": completed_status,
        "actual_bytes": int(completed_result.get("actual_bytes", 0)),
        "executor_elapsed_ms": completed_result.get("executor_elapsed_ms", ""),
        "checksum_status": completed_result.get("checksum_status", ""),
        "object_id": completed_result.get("object_id", ""),
        "first_decision": first["decision"],
        "second_decision_before_advance": second["decision"],
        "third_decision_after_advance": third["decision"],
        "final_tier": final_manifest.tier,
        "final_ready": final_manifest.ready,
        "pending_after_advance": int(
            queue_stats.get("prefetch_queue_pending_total", 0)
        ),
        "token_end": effective_token_end,
        "request_token_count": effective_token_count,
    }
    _write_csv(result_dir / "async_prefetch_smoke.csv", [row])
    _write_report(result_dir / "async_prefetch_report.md", row, tensor_store)
    summary = {
        "status": _status(row),
        "prefix_id": prefix_id,
        "tensor_store": str(tensor_store),
        **row,
        "csv_path": str(result_dir / "async_prefetch_smoke.csv"),
        "report_path": str(result_dir / "async_prefetch_report.md"),
        "decision_log_path": str(result_dir / "decisions.jsonl"),
    }
    (result_dir / "async_prefetch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _payload(
    *,
    request_id: str,
    token_count: int,
    prefix_id: str,
    prefix_end: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if prefix_end:
        candidates.append(
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": prefix_end,
                "committed": True,
            }
        )
    return {
        "request_id": request_id,
        "model_id": DEFAULT_MODEL_ID,
        "token_count": token_count,
        "decode_sla_ms": 12_000,
        "admission_window_ms": 12_000,
        "correctness_key": {
            "model_fingerprint": DEFAULT_MODEL_ID,
            "tokenizer_fingerprint": "qwen2.5-tokenizer",
            "rope_config": "native-32768",
            "dtype": "bf16",
            "kv_layout": "vllm-paged",
        },
        "prefix_candidates": candidates,
    }


def _prefetch_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        results.extend(record.get("prefetch_results") or [])
    return results


def _first_status(results: list[dict[str, Any]], status: str) -> str:
    for result in results:
        if result.get("status") == status:
            return status
    return "NOT_QUEUED"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "queued_status",
        "completed_status",
        "actual_bytes",
        "executor_elapsed_ms",
        "checksum_status",
        "object_id",
        "first_decision",
        "second_decision_before_advance",
        "third_decision_after_advance",
        "final_tier",
        "final_ready",
        "pending_after_advance",
        "token_end",
        "request_token_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, row: dict[str, Any], tensor_store: Path) -> None:
    lines = [
        "# M3.9 异步预取烟测报告",
        "",
        f"- 张量存储：`{tensor_store}`",
        f"- 前缀：`{row['prefix_id']}`",
        f"- 入队状态：`{row['queued_status']}`",
        f"- 完成状态：`{row['completed_status']}`",
        f"- 真实迁移字节数：`{row['actual_bytes']}`",
        f"- 执行器耗时：`{row['executor_elapsed_ms']}` ms",
        f"- checksum 状态：`{row['checksum_status']}`",
        f"- 推进前第二次准入：`{row['second_decision_before_advance']}`",
        f"- 推进后第三次准入：`{row['third_decision_after_advance']}`",
        f"- 最终层级：`{row['final_tier']}`，ready=`{row['final_ready']}`",
        f"- 推进后待处理数：`{row['pending_after_advance']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status(row: dict[str, Any]) -> str:
    if (
        row["queued_status"] == "QUEUED"
        and row["completed_status"] == "COMPLETED"
        and row["first_decision"] == "DELAY"
        and row["second_decision_before_advance"] == "DELAY"
        and row["third_decision_after_advance"] == "ADMIT"
        and row["final_tier"] == "DRAM"
        and bool(row["final_ready"])
        and int(row["pending_after_advance"]) == 0
    ):
        return "OK"
    return "FAILED"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-store",
        default="results/m3_8_full_restart_matrix/tensor_store",
    )
    parser.add_argument(
        "--result-dir",
        default="results/m3_9_async_prefetch_smoke",
    )
    parser.add_argument("--prefix-id", default="m3-8-64")
    parser.add_argument("--token-end", type=int)
    parser.add_argument("--request-token-count", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--storage-gbps", type=float, default=8.8)
    parser.add_argument("--advance-ms", type=float, default=12_000.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_async_prefetch_smoke(
        tensor_store=Path(args.tensor_store),
        result_dir=Path(args.result_dir),
        prefix_id=args.prefix_id,
        token_end=args.token_end,
        request_token_count=args.request_token_count,
        block_size=args.block_size,
        storage_gbps=args.storage_gbps,
        advance_ms=args.advance_ms,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
