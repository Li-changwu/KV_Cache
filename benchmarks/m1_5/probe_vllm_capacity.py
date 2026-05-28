#!/usr/bin/env python3
"""Probe and summarize vLLM serve capacity for long-context experiments."""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


CAPACITY_COLUMNS = [
    "max_model_len",
    "status",
    "gpu_kv_cache_tokens",
    "max_concurrency",
    "cpu_offloaded_parameters_gb",
    "needed_kv_cache_gib",
    "available_kv_cache_gib",
    "estimated_max_model_len",
    "startup_seconds",
    "exit_code",
    "log_path",
    "error_summary",
]


@dataclass
class CapacityProbeResult:
    max_model_len: int
    status: str
    gpu_kv_cache_tokens: int | None = None
    max_concurrency: float | None = None
    cpu_offloaded_parameters_gb: float | None = None
    needed_kv_cache_gib: float | None = None
    available_kv_cache_gib: float | None = None
    estimated_max_model_len: int | None = None
    startup_seconds: float | None = None
    exit_code: int | None = None
    log_path: str = ""
    error_summary: str = ""


def _int_from_group(match: re.Match[str] | None, group: int = 1) -> int | None:
    if not match:
        return None
    return int(match.group(group).replace(",", ""))


def _float_from_group(match: re.Match[str] | None, group: int = 1) -> float | None:
    if not match:
        return None
    return float(match.group(group))


def _short_error(text: str) -> str:
    marker_groups = [
        ["CUDA out of memory", "torch.OutOfMemoryError:"],
        ["To serve at least one request", "No available memory for the cache blocks"],
        ["ValueError:", "Failed to load model"],
    ]
    lines = [line.strip() for line in text.splitlines()]
    for markers in marker_groups:
        for stripped in lines:
            if any(marker in stripped for marker in markers):
                return stripped[:500]
    return ""


def parse_vllm_serve_log(text: str, max_model_len: int) -> CapacityProbeResult:
    """Parse a vLLM serve log into a stable capacity result row."""

    gpu_kv_cache_tokens = _int_from_group(
        re.search(r"GPU KV cache size:\s*([0-9,]+)\s+tokens", text)
    )
    max_concurrency = _float_from_group(
        re.search(r"Maximum concurrency for\s+[0-9,]+\s+tokens per request:\s*([0-9.]+)x", text)
    )
    cpu_offloaded_parameters_gb = _float_from_group(
        re.search(r"Total CPU offloaded parameters:\s*([0-9.]+)", text)
    )

    capacity_error = re.search(
        r"To serve at least one request with the models's max seq len "
        r"\([0-9,]+\).*?"
        r"\(([0-9.]+)\s+GiB KV cache is needed, which is larger than "
        r"the available KV cache memory\s*\(([0-9.]+)\s+GiB\).*?"
        r"estimated maximum model length is ([0-9,]+)",
        text,
        flags=re.DOTALL,
    )
    needed_kv_cache_gib = _float_from_group(capacity_error, 1)
    available_kv_cache_gib = _float_from_group(capacity_error, 2)
    estimated_max_model_len = _int_from_group(capacity_error, 3)

    if "Application startup complete" in text:
        status = "STARTED"
    elif capacity_error or "No available memory for the cache blocks" in text:
        status = "KV_CAPACITY_ERROR"
    elif "Failed to load model - not enough GPU memory" in text or "CUDA out of memory" in text:
        status = "WEIGHT_OOM"
    else:
        status = "UNKNOWN"

    error_summary = _short_error(text)
    if capacity_error:
        error_summary = (
            f"KV cache needed {needed_kv_cache_gib} GiB exceeds available "
            f"{available_kv_cache_gib} GiB; estimated max model length "
            f"{estimated_max_model_len}"
        )

    return CapacityProbeResult(
        max_model_len=max_model_len,
        status=status,
        gpu_kv_cache_tokens=gpu_kv_cache_tokens,
        max_concurrency=max_concurrency,
        cpu_offloaded_parameters_gb=cpu_offloaded_parameters_gb,
        needed_kv_cache_gib=needed_kv_cache_gib,
        available_kv_cache_gib=available_kv_cache_gib,
        estimated_max_model_len=estimated_max_model_len,
        error_summary=error_summary,
    )


def write_capacity_csv(path: str | Path, rows: Iterable[CapacityProbeResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CAPACITY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _wait_for_health(
    base_url: str, timeout_sec: float, process: subprocess.Popen[str] | None = None
) -> bool:
    deadline = time.monotonic() + timeout_sec
    health_url = f"{base_url.rstrip('/')}/health"
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with opener.open(health_url, timeout=3) as response:
                if 200 <= response.status < 500:
                    return True
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    return False


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def build_serve_command(args: argparse.Namespace, max_model_len: int) -> list[str]:
    command = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
    ]
    if args.cpu_offload_gb is not None:
        command.extend(["--cpu-offload-gb", str(args.cpu_offload_gb)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.extra_args:
        command.extend(args.extra_args)
    return command


def run_capacity_probe(args: argparse.Namespace, max_model_len: int) -> CapacityProbeResult:
    result_dir = Path(args.result_dir)
    log_dir = result_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"capacity_maxlen{max_model_len}.log"
    if args.log_prefix:
        log_path = log_dir / f"{args.log_prefix}_maxlen{max_model_len}.log"

    if _port_open(args.host, args.port):
        return CapacityProbeResult(
            max_model_len=max_model_len,
            status="PORT_IN_USE",
            log_path=str(log_path),
            error_summary=f"{args.host}:{args.port} is already open",
        )

    command = build_serve_command(args, max_model_len)
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = ""
    env["NO_PROXY"] = "127.0.0.1,localhost"
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        env.pop(key, None)

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("+ " + " ".join(command) + "\n")
        log_fh.flush()
        process = subprocess.Popen(
            command,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        started = _wait_for_health(
            f"http://{args.host}:{args.port}",
            timeout_sec=args.startup_timeout_sec,
            process=process,
        )
        elapsed = time.monotonic() - start
        if started:
            _terminate_process(process)
        else:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_process(process)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    result = parse_vllm_serve_log(text, max_model_len=max_model_len)
    result.startup_seconds = round(elapsed, 3)
    result.exit_code = process.returncode
    result.log_path = str(log_path)
    if not started and result.status == "UNKNOWN":
        result.status = "STARTUP_TIMEOUT"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", default="results/qwen3_32b_calibration")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-model-lens", nargs="+", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--cpu-offload-gb", type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--startup-timeout-sec", type=float, default=420)
    parser.add_argument("--log-prefix", default="")
    parser.add_argument("--parse-log", action="append", default=[])
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    rows: list[CapacityProbeResult] = []
    if args.parse_log:
        for item in args.parse_log:
            max_len_text, path_text = item.split(":", 1)
            max_len = int(max_len_text)
            path = Path(path_text)
            result = parse_vllm_serve_log(
                path.read_text(encoding="utf-8", errors="replace"),
                max_model_len=max_len,
            )
            result.log_path = str(path)
            rows.append(result)
    else:
        for max_model_len in args.max_model_lens:
            rows.append(run_capacity_probe(args, max_model_len))

    output = Path(args.output) if args.output else Path(args.result_dir) / "qwen3_capacity_probe.csv"
    write_capacity_csv(output, rows)
    print(f"wrote {output}")
    for row in rows:
        print(
            f"max_model_len={row.max_model_len} status={row.status} "
            f"gpu_kv_cache_tokens={row.gpu_kv_cache_tokens} "
            f"max_concurrency={row.max_concurrency} log={row.log_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
