#!/usr/bin/env python3
"""Run a deterministic M3.12 PrePass restore smoke."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.http_sidecar import M3SidecarConfig, create_app


DEFAULT_MODEL = "/root/models/Qwen2.5-14B-Instruct"
KV_BYTES_PER_TOKEN = 196_608


def run_prepass_restore_smoke(
    *,
    result_dir: Path,
    prefix_tokens: int = 2048,
    suffix_tokens: int = 128,
    async_prefetch: bool = False,
    storage_gbps: float = 8.8,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    tensor_store = result_dir / "tensor_store"
    decision_log = result_dir / "decisions.jsonl"
    prefix_id = f"m3-prepass-{prefix_tokens}"
    _write_cold_manifest(
        tensor_store=tensor_store,
        prefix_id=prefix_id,
        token_end=prefix_tokens,
        model=model,
    )
    app = create_app(
        M3SidecarConfig(
            model_id=model,
            hbm_capacity_tokens=65_000,
            dram_capacity_tokens=1_000_000,
            ssd_capacity_tokens=8_000_000,
            kv_bytes_per_token=KV_BYTES_PER_TOKEN,
            h2d_gbps=25.0,
            storage_gbps=storage_gbps,
            decision_log_path=decision_log,
            tensor_store_path=tensor_store,
            async_prefetch=async_prefetch,
        )
    )
    runtime = app.state.runtime
    commit_response = runtime.admit_from_payload(
        _control_payload(
            request_id="prepass_commit",
            model=model,
            token_count=prefix_tokens,
            prefix_id=prefix_id,
            prefix_end=0,
        )
    )
    runtime.control_plane.commit_request(
        runtime.responses[commit_response["request_id"]],
        prefix_id=prefix_id,
        token_end=prefix_tokens,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    prepass_payload = _control_payload(
        request_id="prepass_probe",
        model=model,
        token_count=prefix_tokens + suffix_tokens,
        prefix_id=prefix_id,
        prefix_end=prefix_tokens,
    )
    prepass_started = time.perf_counter()
    prepass = client.post("/prepass", json=prepass_payload)
    prepass_elapsed_ms = _elapsed_ms(prepass_started)
    prepass_json = prepass.json()

    advance_elapsed_ms: float | str = ""
    advance_json: dict[str, Any] = {}
    if async_prefetch:
        advance_started = time.perf_counter()
        advance = client.post("/prefetch/advance", json={"max_ready_ms": 12_000.0})
        advance_elapsed_ms = _elapsed_ms(advance_started)
        advance_json = advance.json()

    online_payload = _control_payload(
        request_id="online_after_prepass",
        model=model,
        token_count=prefix_tokens + suffix_tokens,
        prefix_id=prefix_id,
        prefix_end=prefix_tokens,
    )
    online_started = time.perf_counter()
    online = client.post("/admit", json=online_payload)
    online_elapsed_ms = _elapsed_ms(online_started)
    online_json = online.json()

    row = _row(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        async_prefetch=async_prefetch,
        prepass_elapsed_ms=prepass_elapsed_ms,
        prepass_json=prepass_json,
        advance_elapsed_ms=advance_elapsed_ms,
        advance_json=advance_json,
        online_elapsed_ms=online_elapsed_ms,
        online_json=online_json,
    )
    _write_csv(result_dir / "prepass_restore_smoke.csv", [row])
    _write_report(result_dir / "prepass_restore_report.md", row)
    summary = {
        "status": "OK" if row["online_decision"] == "ADMIT" else "ERROR",
        "ready_before_request": row["ready_before_request"] == "true",
        "online_decision": row["online_decision"],
        "csv_path": str(result_dir / "prepass_restore_smoke.csv"),
        "report_path": str(result_dir / "prepass_restore_report.md"),
    }
    (result_dir / "prepass_restore_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _row(
    *,
    prefix_tokens: int,
    suffix_tokens: int,
    async_prefetch: bool,
    prepass_elapsed_ms: float,
    prepass_json: dict[str, Any],
    advance_elapsed_ms: float | str,
    advance_json: dict[str, Any],
    online_elapsed_ms: float,
    online_json: dict[str, Any],
) -> dict[str, Any]:
    restore_results = prepass_json.get("restore_results") or []
    restore = restore_results[0] if restore_results else {}
    completed = advance_json.get("completed") if isinstance(advance_json, dict) else []
    advanced = completed[0] if completed else {}
    barrier = online_json.get("ready_barrier") or {}
    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "async_prefetch": str(async_prefetch).lower(),
        "prepass_status": str(prepass_json.get("status", "")),
        "ready_before_request": str(
            bool(
                prepass_json.get("ready_before_request")
                or advanced.get("status") == "COMPLETED"
            )
        ).lower(),
        "prepass_elapsed_ms": prepass_elapsed_ms,
        "restore_status": str(restore.get("status") or advanced.get("status") or ""),
        "restore_estimated_ready_ms": restore.get(
            "estimated_ready_ms",
            advanced.get("estimated_ready_ms", ""),
        ),
        "restore_executor_elapsed_ms": restore.get(
            "executor_elapsed_ms",
            advanced.get("executor_elapsed_ms", ""),
        ),
        "advance_elapsed_ms": advance_elapsed_ms,
        "online_decision": str(online_json.get("decision", "")),
        "online_reason": str(online_json.get("reason", "")),
        "online_elapsed_ms": online_elapsed_ms,
        "online_reuse_tokens": int(online_json.get("reuse_tokens", 0)),
        "online_delta_prefill_tokens": int(online_json.get("delta_prefill_tokens", 0)),
        "online_ready_barrier": str(
            bool(barrier.get("all_required_blocks_ready", False))
        ).lower(),
        "sync_cold_miss_total": int(barrier.get("sync_ssd_miss_total", 0)),
    }


def _write_cold_manifest(
    *,
    tensor_store: Path,
    prefix_id: str,
    token_end: int,
    model: str,
) -> None:
    prefix_dir = tensor_store / prefix_id
    prefix_dir.mkdir(parents=True, exist_ok=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": token_end,
                "correctness_key": {
                    "model_fingerprint": model,
                    "tokenizer_fingerprint": "qwen2.5-tokenizer",
                    "rope_config": "native-32768",
                    "dtype": "bf16",
                    "kv_layout": "vllm-paged",
                },
                "token_ids": list(range(min(token_end, 16))),
                "block_size": 16,
                "layout": "NHD",
                "tier": "NVME",
                "ready": False,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )


def _control_payload(
    *,
    request_id: str,
    model: str,
    token_count: int,
    prefix_id: str,
    prefix_end: int,
) -> dict[str, Any]:
    candidates = []
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
        "model_id": model,
        "token_count": token_count,
        "decode_sla_ms": 12_000,
        "admission_window_ms": 12_000,
        "correctness_key": {
            "model_fingerprint": model,
            "tokenizer_fingerprint": "qwen2.5-tokenizer",
            "rope_config": "native-32768",
            "dtype": "bf16",
            "kv_layout": "vllm-paged",
        },
        "prefix_candidates": candidates,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_tokens",
        "suffix_tokens",
        "async_prefetch",
        "prepass_status",
        "ready_before_request",
        "prepass_elapsed_ms",
        "restore_status",
        "restore_estimated_ready_ms",
        "restore_executor_elapsed_ms",
        "advance_elapsed_ms",
        "online_decision",
        "online_reason",
        "online_elapsed_ms",
        "online_reuse_tokens",
        "online_delta_prefill_tokens",
        "online_ready_barrier",
        "sync_cold_miss_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "# M3.12 PrePass Restore Smoke",
        "",
        f"- prefix tokens: `{row['prefix_tokens']}`",
        f"- suffix tokens: `{row['suffix_tokens']}`",
        f"- prepass status: `{row['prepass_status']}`",
        f"- ready before request: `{row['ready_before_request']}`",
        f"- restore status: `{row['restore_status']}`",
        f"- online decision: `{row['online_decision']}`",
        f"- sync cold miss total: `{row['sync_cold_miss_total']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/m3_12_prepass_restore_smoke")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--storage-gbps", type=float, default=8.8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--async-prefetch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_prepass_restore_smoke(
        result_dir=Path(args.result_dir),
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        async_prefetch=args.async_prefetch,
        storage_gbps=args.storage_gbps,
        model=args.model,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
