#!/usr/bin/env python3
"""Measure pinned host-to-device transfer bandwidth with torch."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path


SIZES = {
    "64MB": 64 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
    "1GB": 1024 * 1024 * 1024,
    "4GB": 4 * 1024 * 1024 * 1024,
}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def _selected_sizes(labels: list[str] | None) -> dict[str, int]:
    if not labels:
        return SIZES
    unknown = [label for label in labels if label not in SIZES]
    if unknown:
        raise ValueError(f"unknown H2D size labels: {', '.join(unknown)}")
    return {label: SIZES[label] for label in labels}


def run_bench(
    output: str | Path,
    warmups: int,
    repeats: int,
    selected_sizes: list[str] | None = None,
) -> Path:
    import torch

    output_path = Path(output)
    rows = []
    if not torch.cuda.is_available():
        rows.append(
            {
                "size_bytes": "0",
                "size_label": "cuda_unavailable",
                "bandwidth_gbps": "0",
                "latency_p50_ms": "0",
                "latency_p90_ms": "0",
                "latency_p99_ms": "0",
                "status": "CUDA_UNAVAILABLE",
            }
        )
    else:
        for label, size_bytes in _selected_sizes(selected_sizes).items():
            element_count = size_bytes // 4
            try:
                host = torch.empty(element_count, dtype=torch.float32, pin_memory=True)
                device = torch.empty(element_count, dtype=torch.float32, device="cuda")
                for _ in range(warmups):
                    device.copy_(host, non_blocking=True)
                    torch.cuda.synchronize()
                latencies = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    device.copy_(host, non_blocking=True)
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)
                median_ms = statistics.median(latencies)
                bandwidth_gbps = (size_bytes / (median_ms / 1000)) / 1_000_000_000
                rows.append(
                    {
                        "size_bytes": str(size_bytes),
                        "size_label": label,
                        "bandwidth_gbps": f"{bandwidth_gbps:.6f}",
                        "latency_p50_ms": f"{median_ms:.6f}",
                        "latency_p90_ms": f"{_percentile(latencies, 90):.6f}",
                        "latency_p99_ms": f"{_percentile(latencies, 99):.6f}",
                        "status": "OK",
                    }
                )
            except RuntimeError as exc:
                rows.append(
                    {
                        "size_bytes": str(size_bytes),
                        "size_label": label,
                        "bandwidth_gbps": "0",
                        "latency_p50_ms": "0",
                        "latency_p90_ms": "0",
                        "latency_p99_ms": "0",
                        "status": f"ERROR: {exc}",
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/benchmark_calibration/h2d_bandwidth.csv")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=sorted(SIZES),
        help="Optional tensor-size labels for quick smoke runs.",
    )
    args = parser.parse_args(argv)
    print(run_bench(args.output, args.warmups, args.repeats, args.sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
