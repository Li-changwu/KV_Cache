#!/usr/bin/env python3
"""Measure local NVMe/KV-shaped read bandwidth for M1.5 calibration."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import time
from pathlib import Path


SIZES = {
    "4K": 4 * 1024,
    "64K": 64 * 1024,
    "1M": 1024 * 1024,
    "16M": 16 * 1024 * 1024,
    "64M": 64 * 1024 * 1024,
}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def _ensure_file(path: Path, size_bytes: int) -> None:
    if path.exists() and path.stat().st_size >= size_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        chunk = os.urandom(min(1024 * 1024, size_bytes))
        written = 0
        while written < size_bytes:
            take = min(len(chunk), size_bytes - written)
            fh.write(chunk[:take])
            written += take


def _python_read_bench(path: Path, block_label: str, block_size: int, repeats: int) -> dict[str, str]:
    latencies: list[float] = []
    total_bytes = 0
    start_all = time.perf_counter()
    for _ in range(repeats):
        with path.open("rb") as fh:
            while True:
                start = time.perf_counter()
                data = fh.read(block_size)
                elapsed = time.perf_counter() - start
                if not data:
                    break
                total_bytes += len(data)
                latencies.append(elapsed * 1000)
    elapsed_all = time.perf_counter() - start_all
    bandwidth_gbps = (total_bytes / elapsed_all) / 1_000_000_000 if elapsed_all else 0.0
    return {
        "method": "python",
        "mode": "sequential_read",
        "block_size": str(block_size),
        "block_size_label": block_label,
        "bandwidth_gbps": f"{bandwidth_gbps:.6f}",
        "latency_p50_ms": f"{statistics.median(latencies):.6f}" if latencies else "0",
        "latency_p90_ms": f"{_percentile(latencies, 90):.6f}",
        "latency_p99_ms": f"{_percentile(latencies, 99):.6f}",
        "status": "OK",
    }


def _fio_available() -> bool:
    return shutil.which("fio") is not None


def _selected_sizes(labels: list[str] | None) -> dict[str, int]:
    if not labels:
        return SIZES
    unknown = [label for label in labels if label not in SIZES]
    if unknown:
        raise ValueError(f"unknown block size labels: {', '.join(unknown)}")
    return {label: SIZES[label] for label in labels}


def run_bench(
    output: str | Path,
    data_file: str | Path,
    size_mb: int,
    repeats: int,
    selected_sizes: list[str] | None = None,
) -> Path:
    output_path = Path(output)
    data_path = Path(data_file)
    _ensure_file(data_path, size_mb * 1024 * 1024)

    rows = []
    method_note = (
        "python_fallback_fio_available"
        if _fio_available()
        else "python_fallback_fio_missing"
    )
    for label, size in _selected_sizes(selected_sizes).items():
        row = _python_read_bench(data_path, label, size, repeats)
        row["method"] = method_note if row["method"] == "python" else row["method"]
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/benchmark_calibration/hardware_io.csv")
    parser.add_argument("--data-file", default="results/benchmark_calibration/io_probe.bin")
    parser.add_argument("--size-mb", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=sorted(SIZES),
        help="Optional block-size labels for quick smoke runs.",
    )
    args = parser.parse_args(argv)
    print(run_bench(args.output, args.data_file, args.size_mb, args.repeats, args.sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
