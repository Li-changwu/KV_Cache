#!/usr/bin/env python3
"""Collect reproducibility metadata for M1.5 calibration runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _maybe_import_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import vllm

        versions["vllm"] = {"version": getattr(vllm, "__version__", "unknown")}
    except Exception as exc:  # pragma: no cover - environment dependent
        versions["vllm"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch

        versions["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        versions["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    return versions


def _gpu_info() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    code, stdout, _ = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return []
    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total_mib": int(float(parts[1])),
                    "driver_version": parts[2],
                }
            )
    return gpus


def _dram_info() -> dict[str, Any]:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal"}:
            values[key] = int(value.strip().split()[0]) * 1024
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
    }


def _block_devices() -> list[dict[str, str]]:
    if not shutil.which("lsblk"):
        return []
    code, stdout, _ = _run(["lsblk", "-J", "-o", "NAME,MODEL,SIZE,TYPE,MOUNTPOINTS"])
    if code != 0:
        return []
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    return data.get("blockdevices", [])


def _git_status() -> dict[str, Any]:
    code, stdout, stderr = _run(["git", "status", "--short"])
    return {"returncode": code, "stdout": stdout, "stderr": stderr}


def _model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {"path": str(config_path), "error": "config.json missing"}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    keys = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "use_sliding_window",
        "sliding_window",
        "torch_dtype",
        "quantization_config",
    ]
    return {key: data[key] for key in keys if key in data}


def _attention_representativeness(model_config: dict[str, Any]) -> str:
    if model_config.get("use_sliding_window") is False:
        return "full_attention_path_for_configured_context"
    if model_config.get("use_sliding_window") is True:
        return "limited_by_sliding_window_attention"
    sliding_window = model_config.get("sliding_window")
    if sliding_window not in (None, False, 0):
        return "limited_by_sliding_window_attention"
    return "full_attention_path_for_configured_context"


def collect(model: str | Path) -> dict[str, Any]:
    model_path = Path(model)
    model_config = _model_config(model_path)
    env = _maybe_import_versions()
    env.update(
        {
            "cwd": os.getcwd(),
            "model_path": str(model_path),
            "model_config": model_config,
            "gpu": _gpu_info(),
            "dram": _dram_info(),
            "block_devices": _block_devices(),
            "git_status": _git_status(),
            "notes": {
                "dense_attention_representativeness": _attention_representativeness(
                    model_config
                )
            },
        }
    )
    if "vllm" not in env:
        env["vllm"] = {"version": "unknown"}
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/models/gpt-oss-20b")
    parser.add_argument("--output", default="results/benchmark_calibration/env.json")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect(args.model), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
