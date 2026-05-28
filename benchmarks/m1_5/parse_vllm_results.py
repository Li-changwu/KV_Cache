#!/usr/bin/env python3
"""Parse vLLM benchmark JSON files into M1.5 calibration CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "run_id",
    "model",
    "dataset",
    "input_len",
    "output_len",
    "prefix_len",
    "suffix_len",
    "num_prefixes",
    "request_rate",
    "max_concurrency",
    "num_prompts",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p99_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p99_ms",
    "e2el_p50_ms",
    "e2el_p90_ms",
    "e2el_p99_ms",
    "request_throughput",
    "output_throughput",
    "total_errors",
    "status",
]


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return ""


def _metric(data: dict[str, Any], metric: str, percentile: str) -> Any:
    candidates = [
        f"p{percentile}_{metric}_ms",
        f"percentile_{percentile}_{metric}_ms",
        f"{metric}_p{percentile}_ms",
    ]
    if percentile == "50":
        candidates.extend([f"median_{metric}_ms", f"median_{metric}"])
    return _first(data, *candidates)


def _errors(data: dict[str, Any]) -> list[str]:
    return [str(error) for error in data.get("errors") or [] if str(error).strip()]


def _status(data: dict[str, Any]) -> str:
    errors = _errors(data)
    error_text = " ".join(str(error).lower() for error in errors)
    if "out of memory" in error_text or "oom" in error_text:
        return "OOM"
    if errors:
        return "ERROR"
    return "OK"


def _uniform_list_value(data: dict[str, Any], key: str) -> Any:
    values = data.get(key)
    if not isinstance(values, list) or not values:
        return ""
    first = values[0]
    if all(value == first for value in values):
        return first
    return ""


def parse_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset = str(_first(data, "dataset_name", "dataset", "metadata_dataset_name"))
    if not dataset:
        if "prefix_repetition_prefix_len" in data:
            dataset = "prefix_repetition"
        else:
            dataset = "random"

    errors = _errors(data)
    row = {
        "run_id": _first(data, "label", "run_id") or path.stem,
        "model": _first(data, "model_id", "model"),
        "dataset": dataset,
        "input_len": _first(
            data,
            "input_len",
            "random_input_len",
        )
        or _uniform_list_value(data, "input_lens"),
        "output_len": _first(
            data,
            "output_len",
            "random_output_len",
            "prefix_repetition_output_len",
        )
        or _uniform_list_value(data, "output_lens"),
        "prefix_len": _first(data, "prefix_len", "prefix_repetition_prefix_len"),
        "suffix_len": _first(data, "suffix_len", "prefix_repetition_suffix_len"),
        "num_prefixes": _first(
            data, "num_prefixes", "prefix_repetition_num_prefixes"
        ),
        "request_rate": _first(data, "request_rate"),
        "max_concurrency": _first(data, "max_concurrency"),
        "num_prompts": _first(data, "num_prompts"),
        "request_throughput": _first(data, "request_throughput"),
        "output_throughput": _first(data, "output_throughput"),
        "total_errors": len(errors),
        "status": _status(data),
    }
    for metric in ["ttft", "tpot", "itl", "e2el"]:
        for percentile in ["50", "90", "99"]:
            row[f"{metric}_p{percentile}_ms"] = _metric(data, metric, percentile)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_directory(raw_dir: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    rows = [parse_result(path) for path in sorted(raw_path.glob("*.json"))]

    latency_rows = [row for row in rows if row["dataset"] != "prefix_repetition"]
    apc_rows = [row for row in rows if row["dataset"] == "prefix_repetition"]

    latency_csv = output_path / "vllm_latency.csv"
    apc_csv = output_path / "apc_prefix_reuse.csv"
    _write_csv(latency_csv, latency_rows)
    _write_csv(apc_csv, apc_rows)
    return latency_csv, apc_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="results/benchmark_calibration/raw")
    parser.add_argument("--output-dir", default="results/benchmark_calibration")
    args = parser.parse_args(argv)

    latency_csv, apc_csv = parse_directory(args.raw_dir, args.output_dir)
    print(f"wrote {latency_csv}")
    print(f"wrote {apc_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
