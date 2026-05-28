#!/usr/bin/env python3
"""Run an offline sidecar prefetch smoke over a local tensor-store manifest."""

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


def run_sidecar_prefetch_smoke(
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
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    store = KVTensorStore(tensor_store, block_size=block_size)
    manifest = store.read_manifest(prefix_id)
    effective_token_end = int(token_end or manifest.token_end)
    effective_token_count = int(request_token_count or effective_token_end + block_size)

    store.demote_to_nvme(prefix_id)
    config = M3SidecarConfig(
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
    )
    runtime = SidecarRuntime(config)
    commit_response = runtime.admit_from_payload(
        _payload(
            request_id="sidecar-prefetch-commit",
            token_count=effective_token_end,
            prefix_id=prefix_id,
            prefix_end=0,
        ),
        source="smoke_commit_seed",
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
            request_id="sidecar-prefetch-first",
            token_count=effective_token_count,
            prefix_id=prefix_id,
            prefix_end=effective_token_end,
        ),
        source="smoke_first_admit",
    )
    second = runtime.admit_from_payload(
        _payload(
            request_id="sidecar-prefetch-second",
            token_count=effective_token_count,
            prefix_id=prefix_id,
            prefix_end=effective_token_end,
        ),
        source="smoke_second_admit",
    )
    final_manifest = store.read_manifest(prefix_id)
    migration_events = store.read_migration_events(prefix_id)
    prefetch_result = _latest_prefetch_result(result_dir / "decisions.jsonl")
    prefetch_status = str(prefetch_result.get("status", "NOT_RUN"))
    prefetch_deadline_miss = bool(prefetch_result.get("deadline_miss", False))
    row = {
        "prefix_id": prefix_id,
        "first_decision": first["decision"],
        "first_reason": first["reason"],
        "second_decision": second["decision"],
        "second_reason": second["reason"],
        "prefetch_status": prefetch_status,
        "prefetch_deadline_miss": prefetch_deadline_miss,
        "estimated_ready_ms": prefetch_result.get("estimated_ready_ms", ""),
        "deadline_ms": prefetch_result.get("deadline_ms", ""),
        "final_tier": final_manifest.tier,
        "final_ready": final_manifest.ready,
        "token_end": effective_token_end,
        "request_token_count": effective_token_count,
        "migration_events": len(migration_events),
    }
    _write_csv(result_dir / "sidecar_prefetch_smoke.csv", [row])
    _write_report(result_dir / "sidecar_prefetch_report.md", row, tensor_store)
    status = _status(row)
    summary = {
        "status": status,
        "prefix_id": prefix_id,
        "tensor_store": str(tensor_store),
        "first_decision": first["decision"],
        "first_reason": first["reason"],
        "second_decision": second["decision"],
        "second_reason": second["reason"],
        "prefetch_status": prefetch_status,
        "prefetch_deadline_miss": prefetch_deadline_miss,
        "estimated_ready_ms": prefetch_result.get("estimated_ready_ms"),
        "deadline_ms": prefetch_result.get("deadline_ms"),
        "final_tier": final_manifest.tier,
        "final_ready": final_manifest.ready,
        "csv_path": str(result_dir / "sidecar_prefetch_smoke.csv"),
        "report_path": str(result_dir / "sidecar_prefetch_report.md"),
        "decision_log_path": str(result_dir / "decisions.jsonl"),
    }
    (result_dir / "sidecar_prefetch_summary.json").write_text(
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "first_decision",
        "first_reason",
        "second_decision",
        "second_reason",
        "prefetch_status",
        "prefetch_deadline_miss",
        "estimated_ready_ms",
        "deadline_ms",
        "final_tier",
        "final_ready",
        "token_end",
        "request_token_count",
        "migration_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, row: dict[str, Any], tensor_store: Path) -> None:
    lines = [
        "# M3.9 Sidecar 预取冒烟报告",
        "",
        f"- 张量存储：`{tensor_store}`",
        f"- 前缀：`{row['prefix_id']}`",
        f"- 第一次准入：`{row['first_decision']}` / `{row['first_reason']}`",
        f"- 第二次准入：`{row['second_decision']}` / `{row['second_reason']}`",
        f"- 预取状态：`{row['prefetch_status']}`，deadline miss=`{row['prefetch_deadline_miss']}`",
        f"- 预计完成时间：`{row['estimated_ready_ms']}` ms，截止时间：`{row['deadline_ms']}` ms",
        f"- 最终层级：`{row['final_tier']}`，ready=`{row['final_ready']}`",
        f"- 迁移事件数：`{row['migration_events']}`",
        "",
        "## 结论",
        "",
        _report_conclusion(row),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status(row: dict[str, Any]) -> str:
    if (
        row["first_decision"] == "DELAY"
        and row["second_decision"] == "ADMIT"
        and row["prefetch_status"] == "COMPLETED"
        and row["final_tier"] == "DRAM"
        and bool(row["final_ready"])
    ):
        return "OK"
    if bool(row["prefetch_deadline_miss"]):
        return "DEADLINE_MISS"
    return "FAILED"


def _report_conclusion(row: dict[str, Any]) -> str:
    if _status(row) == "OK":
        return (
            "第一次请求仍然被延迟，侧车在决策之后触发预取。"
            "第二次请求看到同一前缀已经回到 DRAM，因此可以准入。"
        )
    if _status(row) == "DEADLINE_MISS":
        return (
            "预取队列估算该前缀无法在截止时间前完成迁移，因此没有把前缀标记为 ready。"
            "后续请求仍然保持延迟。"
        )
    return "本次烟测没有形成预期的准入或延迟结果，需要查看 decisions.jsonl。"


def _latest_prefetch_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        results = record.get("prefetch_results") or []
        if results:
            latest = dict(results[-1])
    return latest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-store",
        default="results/m3_8_full_restart_matrix/tensor_store",
    )
    parser.add_argument(
        "--result-dir",
        default="results/m3_9_sidecar_prefetch_smoke",
    )
    parser.add_argument("--prefix-id", default="m3-8-64")
    parser.add_argument("--token-end", type=int)
    parser.add_argument("--request-token-count", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--storage-gbps", type=float, default=8.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_sidecar_prefetch_smoke(
        tensor_store=Path(args.tensor_store),
        result_dir=Path(args.result_dir),
        prefix_id=args.prefix_id,
        token_end=args.token_end,
        request_token_count=args.request_token_count,
        block_size=args.block_size,
        storage_gbps=args.storage_gbps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"OK", "DEADLINE_MISS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
