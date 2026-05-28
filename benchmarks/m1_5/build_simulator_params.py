#!/usr/bin/env python3
"""Build simulator parameters from measured M1.5 calibration outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED = [
    "vllm_latency.csv",
    "apc_prefix_reuse.csv",
    "hardware_io.csv",
    "h2d_bandwidth.csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _metric_triplet(row: dict[str, str], metric: str) -> dict[str, float | None]:
    return {
        "p50_ms": _float(row.get(f"{metric}_p50_ms")),
        "p90_ms": _float(row.get(f"{metric}_p90_ms")),
        "p99_ms": _float(row.get(f"{metric}_p99_ms")),
    }


def _load_env(result_dir: Path) -> dict[str, Any]:
    env_path = result_dir / "env.json"
    if not env_path.exists():
        return {}
    return json.loads(env_path.read_text(encoding="utf-8"))


def _dtype_bytes(dtype: str | None) -> int:
    normalized = (dtype or "").lower()
    if "float16" in normalized or "bfloat16" in normalized or "half" in normalized:
        return 2
    if "float32" in normalized or normalized in {"float", "fp32"}:
        return 4
    return 2


def _kv_bytes_per_token(model_config: dict[str, Any]) -> int | None:
    layers = model_config.get("num_hidden_layers")
    kv_heads = model_config.get("num_key_value_heads") or model_config.get("num_attention_heads")
    q_heads = model_config.get("num_attention_heads")
    head_dim = model_config.get("head_dim")
    hidden_size = model_config.get("hidden_size")
    if head_dim in (None, "") and hidden_size not in (None, "") and q_heads not in (None, ""):
        head_dim = int(hidden_size) // int(q_heads)
    if layers in (None, "") or kv_heads in (None, "") or head_dim in (None, ""):
        return None
    return (
        int(layers)
        * int(kv_heads)
        * int(head_dim)
        * 2
        * _dtype_bytes(str(model_config.get("torch_dtype") or "bfloat16"))
    )


def _note(env: dict[str, Any], key: str, default: str) -> str:
    notes = env.get("notes") or {}
    value = notes.get(key)
    return str(value) if value not in (None, "") else default


def _ttft_by_input(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float | None]] = {}
    for row in rows:
        if row.get("status") != "OK" or not row.get("input_len"):
            continue
        input_len = int(float(row["input_len"]))
        output[input_len] = _metric_triplet(row, "ttft")
    return output


def _tpot_by_context(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float | None]] = {}
    for row in rows:
        if row.get("status") != "OK" or not row.get("input_len"):
            continue
        input_len = int(float(row["input_len"]))
        output[input_len] = _metric_triplet(row, "tpot")
    return output


def _find_random_baseline(
    latency_rows: list[dict[str, str]], total_input_len: int, output_len: int
) -> dict[str, str] | None:
    candidates = []
    for row in latency_rows:
        if row.get("status") != "OK" or not row.get("input_len"):
            continue
        row_output_len = int(float(row.get("output_len") or 0))
        if row_output_len != output_len:
            continue
        distance = abs(int(float(row["input_len"])) - total_input_len)
        candidates.append((distance, row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _delta_ms(baseline: dict[str, str] | None, row: dict[str, str], metric: str) -> float | None:
    if baseline is None:
        return None
    baseline_value = _float(baseline.get(metric))
    row_value = _float(row.get(metric))
    if baseline_value is None or row_value is None:
        return None
    return baseline_value - row_value


def _apc_by_prefix(
    rows: list[dict[str, str]], latency_rows: list[dict[str, str]]
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "OK" or not row.get("prefix_len"):
            continue
        prefix_len = int(float(row["prefix_len"]))
        suffix_len = int(float(row["suffix_len"] or 0))
        output_len = int(float(row["output_len"] or 0))
        total_input_len = prefix_len + suffix_len
        baseline = _find_random_baseline(latency_rows, total_input_len, output_len)
        output[prefix_len] = {
            "observed_prefix_repetition_ttft": _metric_triplet(row, "ttft"),
            "ttft_delta_vs_nearest_random_ms": {
                "p50_ms": _delta_ms(baseline, row, "ttft_p50_ms"),
                "p90_ms": _delta_ms(baseline, row, "ttft_p90_ms"),
                "p99_ms": _delta_ms(baseline, row, "ttft_p99_ms"),
            },
            "nearest_random_input_len": int(float(baseline["input_len"]))
            if baseline is not None and baseline.get("input_len")
            else None,
            "num_prefixes": int(float(row["num_prefixes"] or 0)),
            "suffix_len": suffix_len,
        }
    return output


def _nvme_by_size(rows: list[dict[str, str]]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        label = row.get("block_size_label") or row.get("block_size") or "unknown"
        output[label] = {
            "bandwidth_gbps": _float(row.get("bandwidth_gbps")),
            "p50_ms": _float(row.get("latency_p50_ms")),
            "p90_ms": _float(row.get("latency_p90_ms")),
            "p99_ms": _float(row.get("latency_p99_ms")),
        }
    return output


def _h2d_by_size(rows: list[dict[str, str]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        label = row.get("size_label") or row.get("size_bytes") or "unknown"
        output[label] = _float(row.get("bandwidth_gbps"))
    return output


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _capacity_rows(result_dir: Path) -> list[dict[str, str]]:
    path = result_dir / "qwen3_capacity_probe.csv"
    if not path.exists():
        return []
    return _read_csv(path)


def _best_label_value(values: dict[str, float | None]) -> str:
    measured = {key: value for key, value in values.items() if value is not None}
    if not measured:
        return "no OK rows"
    label, value = max(measured.items(), key=lambda item: item[1])
    return f"{label}: {value:.3f} GB/s"


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_report(
    result_dir: Path,
    params: dict[str, Any],
    latency: list[dict[str, str]],
    apc: list[dict[str, str]],
    hardware_io: list[dict[str, str]],
    h2d: list[dict[str, str]],
) -> Path:
    report_path = result_dir / "report.md"
    hardware = params["hardware"]
    vllm = params["vllm"]
    notes = params["notes"]

    lines = [
        "# Benchmark Calibration Report",
        "",
        f"Verification Status: {params['verification_status']}",
        "",
        "## Environment",
        "",
        f"- Model: `{vllm['model_id'] or 'unknown'}`",
        f"- vLLM: `{vllm['version']}`",
        f"- GPU: {hardware['gpu_name']}",
        f"- HBM bytes: {hardware['hbm_total_bytes']}",
        f"- DRAM bytes: {hardware['dram_total_bytes']}",
        f"- Storage path: `{notes['storage_path']}`",
        f"- Dense-attention representativeness: `{notes['dense_attention_representativeness']}`",
        "",
        "## vLLM Baseline",
        "",
        f"- Random rows by status: `{_status_counts(latency)}`",
    ]

    for input_len, metrics in sorted(vllm["ttft_by_input_len"].items()):
        lines.append(
            "- "
            f"input_len={input_len}: "
            f"TTFT p50/p90/p99={_format_ms(metrics['p50_ms'])}/"
            f"{_format_ms(metrics['p90_ms'])}/{_format_ms(metrics['p99_ms'])} ms"
        )

    lines.extend(
        [
            "",
            "## APC Prefix Reuse",
            "",
            f"- Prefix rows by status: `{_status_counts(apc)}`",
        ]
    )
    for prefix_len, metrics in sorted(vllm["apc_hit_benefit_by_prefix_len"].items()):
        ttft = metrics["observed_prefix_repetition_ttft"]
        delta = metrics["ttft_delta_vs_nearest_random_ms"]
        lines.append(
            "- "
            f"prefix_len={prefix_len}, suffix_len={metrics['suffix_len']}, "
            f"num_prefixes={metrics['num_prefixes']}: "
            f"prefix TTFT p50={_format_ms(ttft['p50_ms'])} ms, "
            f"delta_vs_nearest_random_p50={_format_ms(delta['p50_ms'])} ms"
        )

    capacity_rows = _capacity_rows(result_dir)
    if capacity_rows:
        lines.extend(
            [
                "",
                "## Qwen3 Capacity Probe",
                "",
                f"- Capacity rows by status: `{_status_counts(capacity_rows)}`",
            ]
        )
        for row in capacity_rows:
            detail = (
                f"max_model_len={row.get('max_model_len')}: "
                f"status={row.get('status')}"
            )
            if row.get("gpu_kv_cache_tokens"):
                detail += f", gpu_kv_cache_tokens={row['gpu_kv_cache_tokens']}"
            if row.get("max_concurrency"):
                detail += f", max_concurrency={row['max_concurrency']}x"
            if row.get("cpu_offloaded_parameters_gb"):
                detail += f", cpu_offload={row['cpu_offloaded_parameters_gb']}GB"
            if row.get("needed_kv_cache_gib"):
                detail += (
                    f", needed={row['needed_kv_cache_gib']}GiB, "
                    f"available={row.get('available_kv_cache_gib')}GiB, "
                    f"estimated_max_model_len={row.get('estimated_max_model_len')}"
                )
            lines.append(f"- {detail}")

    lines.extend(
        [
            "",
            "## Hardware Calibration",
            "",
            f"- NVMe rows by status: `{_status_counts(hardware_io)}`",
            f"- Best H2D bandwidth: {_best_label_value(hardware['h2d_gbps'])}",
            "",
            "## Anomalies And Limitations",
            "",
            "- OOM/ERROR rows are preserved in CSV inputs and excluded from simulator scalar tables.",
            "- APC benefit is reported as a measured prefix-repetition observation plus nearest-random delta when a comparable random row exists.",
            f"- Dense-attention representativeness: `{notes['dense_attention_representativeness']}`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_params(result_dir: str | Path) -> dict[str, Any]:
    base = Path(result_dir)
    for filename in REQUIRED:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"required calibration file missing: {filename}")

    env = _load_env(base)
    gpu = (env.get("gpu") or [{}])[0]
    dram = env.get("dram") or {}
    vllm = env.get("vllm") or {}
    model_config = env.get("model_config") or {}

    latency = _read_csv(base / "vllm_latency.csv")
    apc = _read_csv(base / "apc_prefix_reuse.csv")
    hardware_io = _read_csv(base / "hardware_io.csv")
    h2d = _read_csv(base / "h2d_bandwidth.csv")

    model_id = ""
    for row in latency + apc:
        if row.get("model"):
            model_id = row["model"]
            break

    params = {
        "verification_status": "MEASURED",
        "hardware": {
            "gpu_name": gpu.get("name", "unknown"),
            "hbm_total_bytes": int(gpu.get("memory_total_mib", 0)) * 1024 * 1024,
            "dram_total_bytes": int(dram.get("total_bytes", 0) or 0),
            "nvme_read_latency_by_size": _nvme_by_size(hardware_io),
            "h2d_gbps": _h2d_by_size(h2d),
        },
        "vllm": {
            "version": vllm.get("version", "unknown"),
            "model_id": model_id,
            "kv_bytes_per_token": _kv_bytes_per_token(model_config),
            "ttft_by_input_len": _ttft_by_input(latency),
            "tpot_by_context_len": _tpot_by_context(latency),
            "apc_hit_benefit_by_prefix_len": _apc_by_prefix(apc, latency),
        },
        "notes": {
            "dense_attention_representativeness": _note(
                env,
                "dense_attention_representativeness",
                "unknown",
            ),
            "storage_path": _note(env, "storage_path", "local_nvme_lower_bound"),
        },
    }

    output_path = base / "simulator_params.yaml"
    output_path.write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_report(base, params, latency, apc, hardware_io, h2d)
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/benchmark_calibration")
    args = parser.parse_args(argv)
    build_params(args.result_dir)
    print(Path(args.result_dir) / "simulator_params.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
