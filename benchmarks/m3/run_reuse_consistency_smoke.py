#!/usr/bin/env python3
"""Compare a full-prefill response with a restart-control reuse response."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.run_reuse_smoke_matrix import (
    DEFAULT_MODEL,
)


def extract_completion_text(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    text = first.get("text")
    return text if isinstance(text, str) else None


def assess_consistency(
    *,
    baseline_status_code: int,
    reuse_status_code: int,
    baseline_payload: dict[str, Any],
    reuse_payload: dict[str, Any],
    external_load_observed: bool,
) -> dict[str, Any]:
    baseline_text = extract_completion_text(baseline_payload)
    reuse_text = extract_completion_text(reuse_payload)
    if baseline_status_code != 200 or reuse_status_code != 200:
        status = "ERROR"
        reason = "http_error"
    elif not baseline_text or not reuse_text:
        status = "ERROR"
        reason = "empty_output"
    elif baseline_text != reuse_text:
        status = "ERROR"
        reason = "text_mismatch"
    elif not external_load_observed:
        status = "ERROR"
        reason = "external_load_missing"
    else:
        status = "OK"
        reason = "matching_text_with_external_load"
    return {
        "status": status,
        "reason": reason,
        "baseline_status_code": baseline_status_code,
        "reuse_status_code": reuse_status_code,
        "baseline_text": baseline_text or "",
        "reuse_text": reuse_text or "",
        "texts_match": bool(baseline_text) and baseline_text == reuse_text,
        "external_load_observed": external_load_observed,
    }


def build_consistency_payloads(
    *,
    model: str,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
    prompt_unit: str,
    request_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = " ".join([prompt_unit] * (prefix_tokens + suffix_tokens))
    base = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
    }
    reuse = dict(base)
    reuse["m3_control"] = _control_payload(
        request_id=request_id,
        model=model,
        token_count=prefix_tokens + suffix_tokens,
        prefix_id=f"m3-8-{prefix_tokens}",
        prefix_end=prefix_tokens,
    )
    return base, reuse


def save_baseline_record(
    result_dir: Path,
    *,
    status_code: int,
    elapsed_ms: float,
    payload: dict[str, Any],
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "baseline_response.json"
    path.write_text(
        json.dumps(
            {
                "status_code": status_code,
                "elapsed_ms": elapsed_ms,
                "payload": payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_baseline_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def connector_event_line_count(event_log_path: Path | None) -> int:
    if event_log_path is None or not event_log_path.exists():
        return 0
    return len(event_log_path.read_text(encoding="utf-8").splitlines())


def summarize_connector_event_delta(
    event_log_path: Path | None,
    *,
    prefix_id: str,
    start_line: int,
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
    lines = event_log_path.read_text(encoding="utf-8").splitlines()
    store_elapsed = 0.0
    load_elapsed = 0.0
    store_events = 0
    load_events = 0
    for line in lines[start_line:]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("prefix_id") != prefix_id or record.get("status") != "ok":
            continue
        elapsed_ms = float(record.get("elapsed_ms") or 0.0)
        if record.get("event") == "store_layer":
            store_elapsed += elapsed_ms
            store_events += 1
        elif record.get("event") == "load_request":
            load_elapsed += elapsed_ms
            load_events += 1
    summary["store_elapsed_ms"] = round(store_elapsed, 3) if store_events else ""
    summary["load_elapsed_ms"] = round(load_elapsed, 3) if load_events else ""
    summary["store_events"] = store_events
    summary["load_events"] = load_events
    summary["external_load_observed"] = "yes" if load_events else "no"
    return summary


def run_consistency_smoke(
    *,
    sidecar_url: str,
    model: str,
    result_dir: Path,
    connector_event_log: Path | None,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
    prompt_unit: str,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix_id = f"m3-8-{prefix_tokens}"
    baseline_payload, reuse_payload = build_consistency_payloads(
        model=model,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        output_tokens=output_tokens,
        prompt_unit=prompt_unit,
        request_id=f"consistency_reuse_{prefix_tokens}",
    )
    with httpx.Client(timeout=None, trust_env=False) as client:
        baseline_started = time.perf_counter()
        baseline_response = client.post(
            sidecar_url.rstrip("/") + "/v1/completions",
            json=baseline_payload,
        )
        baseline_elapsed_ms = _elapsed_ms(baseline_started)
        event_start_line = connector_event_line_count(connector_event_log)
        reuse_started = time.perf_counter()
        reuse_response = client.post(
            sidecar_url.rstrip("/") + "/v1/completions",
            json=reuse_payload,
        )
        reuse_elapsed_ms = _elapsed_ms(reuse_started)
    baseline_json = _safe_json(baseline_response)
    reuse_json = _safe_json(reuse_response)
    events = summarize_connector_event_delta(
        connector_event_log,
        prefix_id=prefix_id,
        start_line=event_start_line,
    )
    assessment = assess_consistency(
        baseline_status_code=baseline_response.status_code,
        reuse_status_code=reuse_response.status_code,
        baseline_payload=baseline_json,
        reuse_payload=reuse_json,
        external_load_observed=events["external_load_observed"] == "yes",
    )
    assessment.update(
        {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "output_tokens": output_tokens,
            "baseline_elapsed_ms": baseline_elapsed_ms,
            "reuse_elapsed_ms": reuse_elapsed_ms,
            "load_events": events["load_events"],
            "load_elapsed_ms": events["load_elapsed_ms"],
        }
    )
    (result_dir / "consistency_smoke.json").write_text(
        json.dumps(assessment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_dir / "consistency_smoke_report.md", assessment)
    return assessment


def run_baseline_phase(
    *,
    sidecar_url: str,
    model: str,
    result_dir: Path,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
    prompt_unit: str,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload, _ = build_consistency_payloads(
        model=model,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        output_tokens=output_tokens,
        prompt_unit=prompt_unit,
        request_id=f"consistency_reuse_{prefix_tokens}",
    )
    with httpx.Client(timeout=None, trust_env=False) as client:
        started = time.perf_counter()
        response = client.post(
            sidecar_url.rstrip("/") + "/v1/completions",
            json=baseline_payload,
        )
        elapsed_ms = _elapsed_ms(started)
    payload = _safe_json(response)
    save_baseline_record(
        result_dir,
        status_code=response.status_code,
        elapsed_ms=elapsed_ms,
        payload=payload,
    )
    result = {
        "status": "OK" if response.status_code == 200 else "ERROR",
        "phase": "baseline",
        "status_code": response.status_code,
        "elapsed_ms": elapsed_ms,
        "text": extract_completion_text(payload) or "",
    }
    (result_dir / "baseline_phase.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_reuse_phase(
    *,
    sidecar_url: str,
    model: str,
    result_dir: Path,
    connector_event_log: Path | None,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
    prompt_unit: str,
    baseline_record_path: Path | None = None,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    baseline_record = load_baseline_record(
        baseline_record_path or result_dir / "baseline_response.json"
    )
    _, reuse_payload = build_consistency_payloads(
        model=model,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        output_tokens=output_tokens,
        prompt_unit=prompt_unit,
        request_id=f"consistency_reuse_{prefix_tokens}",
    )
    prefix_id = f"m3-8-{prefix_tokens}"
    event_start_line = connector_event_line_count(connector_event_log)
    with httpx.Client(timeout=None, trust_env=False) as client:
        started = time.perf_counter()
        response = client.post(
            sidecar_url.rstrip("/") + "/v1/completions",
            json=reuse_payload,
        )
        elapsed_ms = _elapsed_ms(started)
    reuse_payload_json = _safe_json(response)
    events = summarize_connector_event_delta(
        connector_event_log,
        prefix_id=prefix_id,
        start_line=event_start_line,
    )
    assessment = assess_consistency(
        baseline_status_code=int(baseline_record["status_code"]),
        reuse_status_code=response.status_code,
        baseline_payload=dict(baseline_record["payload"]),
        reuse_payload=reuse_payload_json,
        external_load_observed=events["external_load_observed"] == "yes",
    )
    assessment.update(
        {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "output_tokens": output_tokens,
            "baseline_elapsed_ms": float(baseline_record["elapsed_ms"]),
            "reuse_elapsed_ms": elapsed_ms,
            "load_events": events["load_events"],
            "load_elapsed_ms": events["load_elapsed_ms"],
        }
    )
    (result_dir / "consistency_smoke.json").write_text(
        json.dumps(assessment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result_dir / "consistency_smoke_report.md", assessment)
    return assessment


def _control_payload(
    *,
    request_id: str,
    model: str,
    token_count: int,
    prefix_id: str,
    prefix_end: int,
) -> dict[str, Any]:
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
        "prefix_candidates": [
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": prefix_end,
                "committed": True,
            }
        ],
    }


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# M3.8 输出一致性冒烟报告",
        "",
        f"- 状态：`{result['status']}`",
        f"- 原因：`{result['reason']}`",
        f"- 前缀长度：`{result['prefix_tokens']}`",
        f"- 新增后缀：`{result['suffix_tokens']}`",
        f"- 外部加载：`{result['external_load_observed']}`",
        f"- 文本一致：`{result['texts_match']}`",
        f"- 普通路径耗时：`{result['baseline_elapsed_ms']} ms`",
        f"- 复用路径耗时：`{result['reuse_elapsed_ms']} ms`",
        f"- 加载事件数：`{result['load_events']}`",
        "",
        "## 输出",
        "",
        f"- 普通路径：`{result['baseline_text']}`",
        f"- 复用路径：`{result['reuse_text']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--result-dir", default="results/m3_8_full_restart_matrix")
    parser.add_argument("--connector-event-log")
    parser.add_argument("--phase", choices=["both", "baseline", "reuse"], default="both")
    parser.add_argument("--baseline-record")
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--prompt-unit", default="cache")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "sidecar_url": args.sidecar_url,
        "model": args.model,
        "result_dir": Path(args.result_dir),
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "output_tokens": args.output_tokens,
        "prompt_unit": args.prompt_unit,
    }
    if args.phase == "baseline":
        result = run_baseline_phase(**common)
    elif args.phase == "reuse":
        result = run_reuse_phase(
            **common,
            connector_event_log=(
                Path(args.connector_event_log) if args.connector_event_log else None
            ),
            baseline_record_path=Path(args.baseline_record) if args.baseline_record else None,
        )
    else:
        result = run_consistency_smoke(
            **common,
            connector_event_log=(
                Path(args.connector_event_log) if args.connector_event_log else None
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
