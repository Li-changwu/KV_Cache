#!/usr/bin/env python3
"""Generate and optionally execute vLLM benchmark commands for M1.5."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchConfig:
    model: str
    base_url: str
    endpoint: str
    result_dir: Path
    raw_result_dir: Path
    suites: dict[str, dict[str, Any]]


RANDOM_SUITES = {"random_smoke", "latency_sweep", "serving_sweep"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_config(path: str | Path) -> BenchConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    required = ["model", "base_url", "endpoint", "result_dir", "raw_result_dir"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")

    model = str(data["model"])
    if not Path(model).exists():
        raise ValueError(f"model path does not exist: {model}")

    suites = {
        key: value
        for key, value in data.items()
        if key in RANDOM_SUITES or key == "apc_sweep"
    }
    if not suites:
        raise ValueError("config does not define any benchmark suites")

    return BenchConfig(
        model=model,
        base_url=str(data["base_url"]).rstrip("/"),
        endpoint=str(data["endpoint"]),
        result_dir=Path(data["result_dir"]),
        raw_result_dir=Path(data["raw_result_dir"]),
        suites=suites,
    )


def _common_args(config: BenchConfig, suite: str, metadata: dict[str, Any]) -> list[str]:
    metadata_args = [f"{key}={value}" for key, value in sorted(metadata.items())]
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        config.base_url,
        "--endpoint",
        config.endpoint,
        "--model",
        config.model,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(config.raw_result_dir),
        "--label",
        f"m1_5_{suite}",
        "--metadata",
        *metadata_args,
    ]


def _render_random_suite(config: BenchConfig, suite: str, spec: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = []
    for input_len in _as_list(spec.get("input_lens")):
        for output_len in _as_list(spec.get("output_lens")):
            for request_rate in _as_list(spec.get("request_rates")):
                for max_concurrency in _as_list(spec.get("max_concurrencies")):
                    metadata = {
                        "suite": suite,
                        "dataset_name": "random",
                        "input_len": input_len,
                        "output_len": output_len,
                    }
                    command = _common_args(config, suite, metadata) + [
                        "--dataset-name",
                        "random",
                        "--random-input-len",
                        str(input_len),
                        "--random-output-len",
                        str(output_len),
                        "--num-prompts",
                        str(spec["num_prompts"]),
                        "--request-rate",
                        str(request_rate),
                        "--max-concurrency",
                        str(max_concurrency),
                        "--result-filename",
                        (
                            f"{suite}-random-in{input_len}-out{output_len}"
                            f"-rps{request_rate}-c{max_concurrency}.json"
                        ),
                    ]
                    commands.append(command)
    return commands


def _render_apc_suite(config: BenchConfig, spec: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = []
    for num_prefixes in _as_list(spec.get("num_prefixes")):
        for prefix_len in _as_list(spec.get("prefix_lens")):
            for suffix_len in _as_list(spec.get("suffix_lens")):
                for output_len in _as_list(spec.get("output_lens")):
                    for request_rate in _as_list(spec.get("request_rates")):
                        for max_concurrency in _as_list(spec.get("max_concurrencies")):
                            metadata = {
                                "suite": "apc_sweep",
                                "dataset_name": "prefix_repetition",
                                "prefix_len": prefix_len,
                                "suffix_len": suffix_len,
                                "num_prefixes": num_prefixes,
                                "output_len": output_len,
                            }
                            command = _common_args(config, "apc_sweep", metadata) + [
                                "--dataset-name",
                                "prefix_repetition",
                                "--prefix-repetition-num-prefixes",
                                str(num_prefixes),
                                "--prefix-repetition-prefix-len",
                                str(prefix_len),
                                "--prefix-repetition-suffix-len",
                                str(suffix_len),
                                "--prefix-repetition-output-len",
                                str(output_len),
                                "--num-prompts",
                                str(spec["num_prompts"]),
                                "--request-rate",
                                str(request_rate),
                                "--max-concurrency",
                                str(max_concurrency),
                                "--result-filename",
                                (
                                    f"apc-prefix{prefix_len}-suffix{suffix_len}"
                                    f"-np{num_prefixes}-out{output_len}"
                                    f"-rps{request_rate}-c{max_concurrency}.json"
                                ),
                            ]
                            commands.append(command)
    return commands


def render_commands(config: BenchConfig, suites: list[str] | None = None) -> list[list[str]]:
    selected = suites or list(config.suites)
    commands: list[list[str]] = []
    for suite in selected:
        if suite not in config.suites:
            raise ValueError(f"unknown or disabled suite: {suite}")
        spec = config.suites[suite]
        if suite in RANDOM_SUITES:
            commands.extend(_render_random_suite(config, suite, spec))
        elif suite == "apc_sweep":
            commands.extend(_render_apc_suite(config, spec))
        else:
            raise ValueError(f"unsupported suite: {suite}")
    return commands


def _option_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError:
        return ""
    value_index = index + 1
    if value_index >= len(command):
        return ""
    return command[value_index]


def _metadata(command: list[str]) -> dict[str, str]:
    if "--metadata" not in command:
        return {}
    index = command.index("--metadata") + 1
    metadata: dict[str, str] = {}
    while index < len(command):
        value = command[index]
        if value.startswith("--"):
            break
        if "=" in value:
            key, item = value.split("=", 1)
            metadata[key] = item
        index += 1
    return metadata


def _write_failure_result(command: list[str], completed: subprocess.CompletedProcess[Any]) -> None:
    result_dir = Path(_option_value(command, "--result-dir") or ".")
    result_filename = _option_value(command, "--result-filename")
    if not result_filename:
        label = _option_value(command, "--label") or "failed"
        result_filename = f"{label}.json"
    result_path = result_dir / result_filename
    if result_path.exists():
        return

    metadata = _metadata(command)
    stderr = getattr(completed, "stderr", "") or ""
    stdout = getattr(completed, "stdout", "") or ""
    error_text = stderr.strip() or stdout.strip() or f"vllm bench exited {completed.returncode}"
    dataset = _option_value(command, "--dataset-name") or metadata.get("dataset_name", "")

    row: dict[str, Any] = {
        "date": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "backend": "openai",
        "label": _option_value(command, "--label"),
        "model_id": _option_value(command, "--model"),
        "num_prompts": _option_value(command, "--num-prompts"),
        "request_rate": _option_value(command, "--request-rate"),
        "max_concurrency": _option_value(command, "--max-concurrency"),
        "dataset_name": dataset,
        "request_throughput": 0.0,
        "output_throughput": 0.0,
        "errors": [error_text],
        "returncode": completed.returncode,
    }
    row.update({f"metadata_{key}": value for key, value in metadata.items()})
    for key, option in {
        "random_input_len": "--random-input-len",
        "random_output_len": "--random-output-len",
        "prefix_repetition_num_prefixes": "--prefix-repetition-num-prefixes",
        "prefix_repetition_prefix_len": "--prefix-repetition-prefix-len",
        "prefix_repetition_suffix_len": "--prefix-repetition-suffix-len",
        "prefix_repetition_output_len": "--prefix-repetition-output-len",
    }.items():
        value = _option_value(command, option)
        if value:
            row[key] = value
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")


def run_commands(commands: list[list[str]], dry_run: bool) -> int:
    for command in commands:
        printable = " ".join(command)
        if dry_run:
            print(printable)
            continue
        print(f"[m1.5] running: {printable}", flush=True)
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            print(f"[m1.5] command failed with exit {completed.returncode}", file=sys.stderr)
            _write_failure_result(command, completed)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_5_benchmark.yaml")
    parser.add_argument("--suite", action="append", dest="suites")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not args.dry_run:
        config.raw_result_dir.mkdir(parents=True, exist_ok=True)
    commands = render_commands(config, suites=args.suites)
    return run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
