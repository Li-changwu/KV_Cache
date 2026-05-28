#!/usr/bin/env python3
"""Run or dry-run a tiny M3.8 prefix reuse smoke matrix.

The online mode intentionally stays small. It sends a first request that stores a
prefix and then a second request that asks the sidecar to reuse that prefix.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.cold_tier import build_cold_tier_adapter
from benchmarks.m3.tensor_store import KVTensorStore


DEFAULT_MODEL = "/root/models/Qwen2.5-14B-Instruct"


@dataclass(frozen=True)
class MatrixConfig:
    prefix_tokens: list[int]
    output_tokens: int
    result_dir: Path
    suffix_tokens: int = 16
    phase: str = "both"
    sidecar_url: str = "http://127.0.0.1:8010"
    model: str = DEFAULT_MODEL
    dry_run: bool = False
    prompt_unit: str = "cache"
    connector_event_log: Path | None = None
    sidecar_decision_log: Path | None = None
    tensor_store: Path | None = None
    cold_root: Path | None = None
    cold_backend: str = "local_posix"
    cold_tier_restore: bool = False
    prepass_before_reuse: bool = False
    advance_ms: float = 12_000.0
    block_size: int = 16


@dataclass(frozen=True)
class MatrixRow:
    run_id: str
    prefix_tokens: int
    suffix_tokens: int
    output_tokens: int


def matrix_rows(config: MatrixConfig) -> list[MatrixRow]:
    return [
        MatrixRow(
            run_id=f"reuse_prefix_{prefix_tokens}",
            prefix_tokens=int(prefix_tokens),
            suffix_tokens=config.suffix_tokens,
            output_tokens=config.output_tokens,
        )
        for prefix_tokens in config.prefix_tokens
    ]


def run_matrix(config: MatrixConfig) -> dict[str, Any]:
    if config.phase not in {"both", "store", "reuse"}:
        raise ValueError(f"unsupported phase: {config.phase}")
    config.result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in matrix_rows(config):
        if config.dry_run:
            rows.append(_dry_run_row(row, config.phase))
            continue
        rows.append(_run_online_row(config, row))
    _write_csv(config.result_dir / "reuse_smoke_matrix.csv", rows)
    _write_report(config.result_dir / "report.md", rows, config)
    status = "DRY_RUN" if config.dry_run else _overall_status(rows)
    return {
        "status": status,
        "rows": len(rows),
        "csv_path": str(config.result_dir / "reuse_smoke_matrix.csv"),
        "report_path": str(config.result_dir / "report.md"),
    }


def _dry_run_row(row: MatrixRow, phase: str = "both") -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "prefix_tokens": row.prefix_tokens,
        "suffix_tokens": row.suffix_tokens,
        "output_tokens": row.output_tokens,
        "phase": phase,
        "status": "DRY_RUN",
        "first_status_code": "",
        "second_status_code": "",
        "first_ttft_ms": "",
        "second_ttft_ms": "",
        "reuse_tokens": "",
        "decision": "",
        "reason": "",
        "ready_barrier_all_ready": "",
        "sync_ssd_miss_total": "",
        "error": "",
        "manifest_bytes": "",
        "store_elapsed_ms": "",
        "load_elapsed_ms": "",
        "connector_store_elapsed_ms": "",
        "connector_load_elapsed_ms": "",
        "connector_total_elapsed_ms": "",
        "store_events": "",
        "load_events": "",
        "external_load_observed": "no",
        **empty_prepass_summary(),
        **empty_cold_tier_summary(),
        **empty_sidecar_metric_summary(),
    }


def _run_online_row(config: MatrixConfig, row: MatrixRow) -> dict[str, Any]:
    prefix_id = f"m3-8-{row.prefix_tokens}"
    first_payload, second_payload = build_store_and_reuse_payloads(config, row)
    final_reuse_payload = second_payload
    cold_summary = {
        **empty_prepass_summary(),
        **empty_cold_tier_summary(),
    }
    if config.cold_tier_restore:
        cold_summary["cold_tier_restore"] = "yes"
        cold_summary["cold_backend"] = config.cold_backend
    try:
        with httpx.Client(timeout=None, trust_env=False) as client:
            first = None
            second = None
            first_elapsed_ms: float | str = ""
            second_elapsed_ms: float | str = ""
            metrics_before = fetch_sidecar_metrics(client, config.sidecar_url)
            decision_start_line = sidecar_decision_log_line_count(
                config.sidecar_decision_log
            )
            connector_start_line = connector_event_log_line_count(
                config.connector_event_log
            )
            if config.phase in {"both", "store"}:
                first_started = time.perf_counter()
                first = client.post(
                    config.sidecar_url.rstrip("/") + "/v1/completions",
                    json=first_payload,
                )
                first_elapsed_ms = _elapsed_ms(first_started)
                if config.cold_tier_restore:
                    cold_summary.update(
                        demote_prefix_to_cold_object(config, prefix_id)
                    )
                    cold_summary.update(
                        mark_sidecar_prefix_cold(
                            client,
                            config.sidecar_url,
                            request_id=f"{row.run_id}_store",
                            prefix_id=prefix_id,
                            token_end=row.prefix_tokens,
                        )
                    )

            if config.phase in {"both", "reuse"}:
                if config.cold_tier_restore and config.prepass_before_reuse:
                    prepass_payload = clone_payload_with_request_id(
                        second_payload,
                        f"{row.run_id}_prepass",
                    )
                    prepass_started = time.perf_counter()
                    prepass = client.post(
                        config.sidecar_url.rstrip("/") + "/prepass",
                        json=prepass_payload["m3_control"],
                    )
                    cold_summary.update(
                        summarize_prepass_response(
                            _safe_json(prepass),
                            elapsed_ms=_elapsed_ms(prepass_started),
                        )
                    )
                    if cold_summary.get("prepass_status") == "QUEUED":
                        advance_started = time.perf_counter()
                        advance = client.post(
                            config.sidecar_url.rstrip("/") + "/prefetch/advance",
                            json={"max_ready_ms": config.advance_ms},
                        )
                        cold_summary["cold_advance_status_code"] = _status_code(advance)
                        cold_summary["cold_advance_elapsed_ms"] = _elapsed_ms(
                            advance_started
                        )
                        cold_summary.update(
                            summarize_prefetch_advance(_safe_json(advance), prefix_id)
                        )
                    final_reuse_payload = clone_payload_with_request_id(
                        second_payload,
                        f"{row.run_id}_reuse_after_prepass",
                    )
                elif config.cold_tier_restore:
                    probe_payload = clone_payload_with_request_id(
                        second_payload,
                        f"{row.run_id}_cold_probe",
                    )
                    probe_started = time.perf_counter()
                    probe = client.post(
                        config.sidecar_url.rstrip("/") + "/admit",
                        json=probe_payload["m3_control"],
                    )
                    cold_summary["cold_probe_status_code"] = _status_code(probe)
                    cold_summary["cold_probe_elapsed_ms"] = _elapsed_ms(probe_started)
                    probe_json = _safe_json(probe)
                    if isinstance(probe_json, dict):
                        cold_summary["cold_probe_decision"] = str(
                            probe_json.get("decision", "")
                        )
                        cold_summary["cold_probe_reason"] = str(
                            probe_json.get("reason", "")
                        )
                    advance_started = time.perf_counter()
                    advance = client.post(
                        config.sidecar_url.rstrip("/") + "/prefetch/advance",
                        json={"max_ready_ms": config.advance_ms},
                    )
                    cold_summary["cold_advance_status_code"] = _status_code(advance)
                    cold_summary["cold_advance_elapsed_ms"] = _elapsed_ms(
                        advance_started
                    )
                    cold_summary.update(
                        summarize_prefetch_advance(_safe_json(advance), prefix_id)
                    )
                    final_reuse_payload = clone_payload_with_request_id(
                        second_payload,
                        f"{row.run_id}_reuse_after_restore",
                    )
                second_started = time.perf_counter()
                second = client.post(
                    config.sidecar_url.rstrip("/") + "/v1/completions",
                    json=final_reuse_payload,
                )
                second_elapsed_ms = _elapsed_ms(second_started)
            metrics_after = fetch_sidecar_metrics(client, config.sidecar_url)
    except Exception as exc:  # pragma: no cover - exercised by online smoke failures
        return {
            **_dry_run_row(row, config.phase),
            "status": "ERROR",
            "error": str(exc),
        }

    manifest_response = second if second is not None else first
    manifest_json = _safe_json(manifest_response) if manifest_response is not None else {}
    kv_params = manifest_json.get("kv_transfer_params", {}) if isinstance(manifest_json, dict) else {}
    connector = kv_params.get("m3_noop_connector", {}) if isinstance(kv_params, dict) else {}
    manifest = connector.get("manifest", {}) if isinstance(connector, dict) else {}
    event_summary = summarize_connector_events(
        config.connector_event_log,
        prefix_id=prefix_id,
        start_line=connector_start_line,
    )
    decision_summary = summarize_sidecar_decision_delta(
        config.sidecar_decision_log,
        start_line=decision_start_line,
        request_id=(
            (
                (
                    f"{row.run_id}_reuse_after_prepass"
                    if config.prepass_before_reuse
                    else f"{row.run_id}_reuse_after_restore"
                )
                if config.cold_tier_restore
                else f"{row.run_id}_reuse"
            )
            if config.phase in {"both", "reuse"}
            else f"{row.run_id}_store"
        ),
    )
    success = _response_success(first) and _response_success(second)
    return {
        "run_id": row.run_id,
        "prefix_tokens": row.prefix_tokens,
        "suffix_tokens": row.suffix_tokens,
        "output_tokens": row.output_tokens,
        "phase": config.phase,
        "status": "OK" if success else "ERROR",
        "first_status_code": _status_code(first),
        "second_status_code": _status_code(second),
        "first_ttft_ms": first_elapsed_ms,
        "second_ttft_ms": second_elapsed_ms,
        "reuse_tokens": row.prefix_tokens,
        **decision_summary,
        "error": "" if success else _response_error(first, second),
        "manifest_bytes": len(json.dumps(manifest, sort_keys=True)),
        **event_summary,
        **cold_summary,
        **summarize_sidecar_metric_deltas(metrics_before, metrics_after),
    }


def build_store_and_reuse_payloads(
    config: MatrixConfig,
    row: MatrixRow,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix_id = f"m3-8-{row.prefix_tokens}"
    prefix_prompt = _prompt_for_tokens(config.prompt_unit, row.prefix_tokens)
    reuse_prompt = _prompt_for_tokens(
        config.prompt_unit,
        row.prefix_tokens + row.suffix_tokens,
    )
    base = {
        "model": config.model,
        "max_tokens": row.output_tokens,
        "temperature": 0,
    }
    first_control = _control_payload(
        request_id=f"{row.run_id}_store",
        model=config.model,
        token_count=row.prefix_tokens,
        prefix_id=prefix_id,
        prefix_end=0,
        store_prefix_id=prefix_id,
    )
    second_control = _control_payload(
        request_id=f"{row.run_id}_reuse",
        model=config.model,
        token_count=row.prefix_tokens + row.suffix_tokens,
        prefix_id=prefix_id,
        prefix_end=row.prefix_tokens,
    )
    return (
        dict(base, prompt=prefix_prompt, m3_control=first_control),
        dict(base, prompt=reuse_prompt, m3_control=second_control),
    )


def clone_payload_with_request_id(
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    cloned = copy.deepcopy(payload)
    control = cloned.get("m3_control")
    if not isinstance(control, dict):
        raise ValueError("payload has no m3_control object")
    control["request_id"] = request_id
    return cloned


def demote_prefix_to_cold_object(
    config: MatrixConfig,
    prefix_id: str,
) -> dict[str, Any]:
    if config.tensor_store is None:
        raise ValueError("--tensor-store is required for cold-tier restore mode")
    cold_root = config.cold_root or (config.result_dir / "cold_objects")
    store = KVTensorStore(config.tensor_store, block_size=config.block_size)
    manifest = store.demote_to_cold_object(
        prefix_id,
        cold_adapter=build_cold_tier_adapter(config.cold_backend, cold_root),
        tier="NVME",
    )
    hot_files_present = all(
        (config.tensor_store / prefix_id / record.file_name).exists()
        for record in manifest.layers.values()
    )
    return {
        "cold_tier_restore": "yes",
        "cold_object_id": manifest.object_id or "",
        "cold_uri": manifest.cold_uri or "",
        "cold_size_bytes": manifest.size_bytes,
        "cold_checksum": manifest.checksum,
        "cold_backend": config.cold_backend,
        "cold_hot_files_present_after_demote": "yes" if hot_files_present else "no",
    }


def mark_sidecar_prefix_cold(
    client: Any,
    sidecar_url: str,
    *,
    request_id: str,
    prefix_id: str,
    token_end: int,
) -> dict[str, Any]:
    response = client.post(
        sidecar_url.rstrip("/") + "/commit",
        json={
            "request_id": request_id,
            "prefix_id": prefix_id,
            "token_end": token_end,
            "tier": "SSD",
            "ready": False,
        },
    )
    payload = _safe_json(response)
    return {
        "cold_commit_status_code": _status_code(response),
        "cold_commit_status": str(payload.get("status", "")),
    }


def summarize_prefetch_advance(
    payload: dict[str, Any],
    prefix_id: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    completed = payload.get("completed") if isinstance(payload, dict) else None
    if not isinstance(completed, list):
        return {
            "cold_restore_status": "NOT_COMPLETED",
            "cold_restore_actual_bytes": 0,
            "cold_restore_checksum_status": "",
            "cold_restore_executor_elapsed_ms": "",
        }
    selected = None
    for item in completed:
        if isinstance(item, dict) and item.get("prefix_id") == prefix_id:
            selected = item
            break
    if selected is None:
        return {
            "cold_restore_status": "NOT_COMPLETED",
            "cold_restore_actual_bytes": 0,
            "cold_restore_checksum_status": "",
            "cold_restore_executor_elapsed_ms": "",
        }
    summary["cold_restore_status"] = str(selected.get("status", ""))
    summary["cold_restore_actual_bytes"] = int(selected.get("actual_bytes", 0))
    summary["cold_restore_checksum_status"] = str(
        selected.get("checksum_status", "")
    )
    summary["cold_restore_executor_elapsed_ms"] = selected.get(
        "executor_elapsed_ms",
        "",
    )
    return summary


def summarize_prepass_response(
    payload: dict[str, Any],
    *,
    elapsed_ms: float,
) -> dict[str, Any]:
    restore_results = payload.get("restore_results") if isinstance(payload, dict) else []
    restore = restore_results[0] if isinstance(restore_results, list) and restore_results else {}
    return {
        "prepass_status_code": 200 if payload else "",
        "prepass_status": str(payload.get("status", "")) if isinstance(payload, dict) else "",
        "ready_before_request": (
            str(bool(payload.get("ready_before_request"))).lower()
            if isinstance(payload, dict)
            else ""
        ),
        "prepass_elapsed_ms": elapsed_ms,
        "prepass_restore_status": str(restore.get("status", "")) if isinstance(restore, dict) else "",
        "prepass_restore_estimated_ready_ms": (
            restore.get("estimated_ready_ms", "") if isinstance(restore, dict) else ""
        ),
        "prepass_restore_actual_bytes": (
            int(restore.get("actual_bytes", 0)) if isinstance(restore, dict) else ""
        ),
        "prepass_restore_checksum_status": (
            str(restore.get("checksum_status", "")) if isinstance(restore, dict) else ""
        ),
        "prepass_restore_executor_elapsed_ms": (
            restore.get("executor_elapsed_ms", "") if isinstance(restore, dict) else ""
        ),
    }


def summarize_connector_events(
    event_log_path: Path | None,
    *,
    prefix_id: str,
    start_line: int = 0,
) -> dict[str, Any]:
    summary = {
        "store_elapsed_ms": "",
        "load_elapsed_ms": "",
        "store_events": 0,
        "load_events": 0,
        "external_load_observed": "no",
    }
    if event_log_path is None or not event_log_path.exists():
        return summary
    store_elapsed = 0.0
    load_elapsed = 0.0
    store_events = 0
    load_events = 0
    for line in event_log_path.read_text(encoding="utf-8").splitlines()[start_line:]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("prefix_id") != prefix_id or record.get("status") != "ok":
            continue
        event = record.get("event")
        elapsed_ms = float(record.get("elapsed_ms") or 0.0)
        if event == "store_layer":
            store_elapsed += elapsed_ms
            store_events += 1
        elif event == "load_request":
            load_elapsed += elapsed_ms
            load_events += 1
    summary["store_elapsed_ms"] = round(store_elapsed, 3) if store_events else ""
    summary["load_elapsed_ms"] = round(load_elapsed, 3) if load_events else ""
    summary["connector_store_elapsed_ms"] = summary["store_elapsed_ms"]
    summary["connector_load_elapsed_ms"] = summary["load_elapsed_ms"]
    summary["connector_total_elapsed_ms"] = round(store_elapsed + load_elapsed, 3)
    summary["store_events"] = store_events
    summary["load_events"] = load_events
    summary["external_load_observed"] = "yes" if load_events else "no"
    return summary


def sidecar_decision_log_line_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def connector_event_log_line_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def summarize_sidecar_decision_delta(
    path: Path | None,
    *,
    start_line: int,
    request_id: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "decision": "",
        "reason": "",
        "ready_barrier_all_ready": "",
        "sync_ssd_miss_total": "",
    }
    if path is None or not path.exists():
        return summary
    lines = path.read_text(encoding="utf-8").splitlines()[start_line:]
    selected: dict[str, Any] | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = record.get("response") if isinstance(record, dict) else None
        request = record.get("request") if isinstance(record, dict) else None
        if not isinstance(response, dict) or not isinstance(request, dict):
            continue
        if str(response.get("request_id") or request.get("request_id")) != request_id:
            continue
        if record.get("source") != "proxy":
            continue
        selected = record
    if selected is None:
        return summary
    response = selected.get("response", {})
    barrier = selected.get("ready_barrier", {})
    summary["decision"] = str(response.get("decision", ""))
    summary["reason"] = str(response.get("reason", ""))
    if isinstance(barrier, dict):
        if "all_required_blocks_ready" in barrier:
            summary["ready_barrier_all_ready"] = str(
                bool(barrier["all_required_blocks_ready"])
            ).lower()
        summary["sync_ssd_miss_total"] = int(barrier.get("sync_ssd_miss_total", 0))
    return summary


def fetch_sidecar_metrics(
    client: httpx.Client,
    sidecar_url: str,
) -> dict[str, float]:
    try:
        response = client.get(sidecar_url.rstrip("/") + "/metrics")
        response.raise_for_status()
    except httpx.HTTPError:
        return {}
    return parse_sidecar_metrics(response.text)


def parse_sidecar_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def summarize_sidecar_metric_deltas(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, Any]:
    metric_map = {
        "residency_hit_delta": "residency_hit_total",
        "prefetch_queued_delta": "prefetch_queued_total",
        "prefetch_deadline_miss_delta": "prefetch_deadline_miss_total",
        "prefetch_queue_requests_delta": "prefetch_queue_requests_total",
        "prefetch_queue_completed_delta": "prefetch_queue_completed_total",
        "prefetch_queue_deadline_miss_delta": "prefetch_queue_deadline_miss_total",
        "prefetch_queue_bytes_delta": "prefetch_queue_bytes_total",
    }
    summary: dict[str, Any] = {}
    for output_name, metric_name in metric_map.items():
        summary[output_name] = _metric_delta(before, after, metric_name)
    summary["prefetch_queue_pending"] = _format_metric(
        after.get("prefetch_queue_pending_total", 0.0)
    )
    return summary


def empty_sidecar_metric_summary() -> dict[str, str]:
    return {
        "residency_hit_delta": "",
        "prefetch_queued_delta": "",
        "prefetch_deadline_miss_delta": "",
        "prefetch_queue_requests_delta": "",
        "prefetch_queue_completed_delta": "",
        "prefetch_queue_deadline_miss_delta": "",
        "prefetch_queue_bytes_delta": "",
        "prefetch_queue_pending": "",
    }


def empty_cold_tier_summary() -> dict[str, Any]:
    return {
        "cold_tier_restore": "no",
        "cold_object_id": "",
        "cold_uri": "",
        "cold_size_bytes": "",
        "cold_checksum": "",
        "cold_backend": "",
        "cold_hot_files_present_after_demote": "",
        "cold_commit_status_code": "",
        "cold_commit_status": "",
        "cold_probe_status_code": "",
        "cold_probe_decision": "",
        "cold_probe_reason": "",
        "cold_probe_elapsed_ms": "",
        "cold_advance_status_code": "",
        "cold_advance_elapsed_ms": "",
        "cold_restore_status": "",
        "cold_restore_actual_bytes": "",
        "cold_restore_checksum_status": "",
        "cold_restore_executor_elapsed_ms": "",
    }


def empty_prepass_summary() -> dict[str, Any]:
    return {
        "prepass_status_code": "",
        "prepass_status": "",
        "ready_before_request": "",
        "prepass_elapsed_ms": "",
        "prepass_restore_status": "",
        "prepass_restore_estimated_ready_ms": "",
        "prepass_restore_actual_bytes": "",
        "prepass_restore_checksum_status": "",
        "prepass_restore_executor_elapsed_ms": "",
    }


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    metric_name: str,
) -> int | float:
    return _format_metric(after.get(metric_name, 0.0) - before.get(metric_name, 0.0))


def _format_metric(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return round(value, 6)


def _control_payload(
    *,
    request_id: str,
    model: str,
    token_count: int,
    prefix_id: str,
    prefix_end: int,
    store_prefix_id: str | None = None,
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
    payload = {
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
    if store_prefix_id is not None:
        payload["store_prefix_id"] = store_prefix_id
    return payload


def _prompt_for_tokens(unit: str, token_count: int) -> str:
    return " ".join([unit] * max(1, token_count))


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _response_success(response: httpx.Response | None) -> bool:
    return True if response is None else response.is_success


def _status_code(response: httpx.Response | None) -> int | str:
    return "" if response is None else response.status_code


def _response_error(*responses: httpx.Response | None) -> str:
    for response in responses:
        if response is not None and not response.is_success:
            return response.text[:500]
    return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "prefix_tokens",
        "suffix_tokens",
        "output_tokens",
        "phase",
        "status",
        "first_status_code",
        "second_status_code",
        "first_ttft_ms",
        "second_ttft_ms",
        "reuse_tokens",
        "decision",
        "reason",
        "ready_barrier_all_ready",
        "sync_ssd_miss_total",
        "error",
        "manifest_bytes",
        "store_elapsed_ms",
        "load_elapsed_ms",
        "connector_store_elapsed_ms",
        "connector_load_elapsed_ms",
        "connector_total_elapsed_ms",
        "store_events",
        "load_events",
        "external_load_observed",
        "prepass_status_code",
        "prepass_status",
        "ready_before_request",
        "prepass_elapsed_ms",
        "prepass_restore_status",
        "prepass_restore_estimated_ready_ms",
        "prepass_restore_actual_bytes",
        "prepass_restore_checksum_status",
        "prepass_restore_executor_elapsed_ms",
        "cold_tier_restore",
        "cold_object_id",
        "cold_uri",
        "cold_size_bytes",
        "cold_checksum",
        "cold_backend",
        "cold_hot_files_present_after_demote",
        "cold_commit_status_code",
        "cold_commit_status",
        "cold_probe_status_code",
        "cold_probe_decision",
        "cold_probe_reason",
        "cold_probe_elapsed_ms",
        "cold_advance_status_code",
        "cold_advance_elapsed_ms",
        "cold_restore_status",
        "cold_restore_actual_bytes",
        "cold_restore_checksum_status",
        "cold_restore_executor_elapsed_ms",
        "residency_hit_delta",
        "prefetch_queued_delta",
        "prefetch_deadline_miss_delta",
        "prefetch_queue_requests_delta",
        "prefetch_queue_completed_delta",
        "prefetch_queue_deadline_miss_delta",
        "prefetch_queue_bytes_delta",
        "prefetch_queue_pending",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path: Path, rows: list[dict[str, Any]], config: MatrixConfig) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# M3.8 小矩阵复用验证报告",
        "",
        f"- 模型：`{config.model}`",
        f"- 前置代理：`{config.sidecar_url}`",
        f"- 输出令牌数：`{config.output_tokens}`",
        f"- 状态统计：`{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "## 行定义",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['run_id']}`：前缀 {row['prefix_tokens']} tokens，"
            f"新增后缀 {row['suffix_tokens']} tokens，状态 `{row['status']}`，"
            f"驻留命中变化 `{row.get('residency_hit_delta', '')}`，"
            f"预取入队变化 `{row.get('prefetch_queued_delta', '')}`，"
            f"预取截止错过变化 `{row.get('prefetch_deadline_miss_delta', '')}`，"
            f"cold restore `{row.get('cold_restore_status', '')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if all(row["status"] == "OK" for row in rows):
        return "OK"
    if any(row["status"] == "OK" for row in rows):
        return "PARTIAL"
    return "ERROR"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/m3_8_reuse_matrix")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prefix-tokens", nargs="+", type=int, default=[16, 64, 128, 256])
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--phase", choices=["both", "store", "reuse"], default="both")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--prompt-unit", default="cache")
    parser.add_argument("--connector-event-log")
    parser.add_argument("--sidecar-decision-log")
    parser.add_argument("--tensor-store")
    parser.add_argument("--cold-root")
    parser.add_argument(
        "--cold-backend",
        default="local_posix",
        choices=["local_posix", "3fs_posix", "packed_v1"],
    )
    parser.add_argument("--cold-tier-restore", action="store_true")
    parser.add_argument("--prepass-before-reuse", action="store_true")
    parser.add_argument("--advance-ms", type=float, default=12_000.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_matrix(
        MatrixConfig(
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            output_tokens=args.output_tokens,
            result_dir=Path(args.result_dir),
            sidecar_url=args.sidecar_url,
            model=args.model,
            dry_run=args.dry_run,
            phase=args.phase,
            prompt_unit=args.prompt_unit,
            connector_event_log=(
                Path(args.connector_event_log) if args.connector_event_log else None
            ),
            sidecar_decision_log=(
                Path(args.sidecar_decision_log) if args.sidecar_decision_log else None
            ),
            tensor_store=Path(args.tensor_store) if args.tensor_store else None,
            cold_root=Path(args.cold_root) if args.cold_root else None,
            cold_backend=args.cold_backend,
            cold_tier_restore=args.cold_tier_restore,
            prepass_before_reuse=args.prepass_before_reuse,
            advance_ms=args.advance_ms,
            block_size=args.block_size,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"OK", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
