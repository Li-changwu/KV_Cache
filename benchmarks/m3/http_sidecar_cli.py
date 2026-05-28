#!/usr/bin/env python3
"""Run the M3.5 HTTP sidecar or print the resolved configuration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m2.simulator import load_calibration
from benchmarks.m3.http_sidecar import M3SidecarConfig, create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-params", required=True)
    parser.add_argument("--capacity-csv")
    parser.add_argument(
        "--decision-log",
        default="results/m3_5_http_sidecar/decisions.jsonl",
    )
    parser.add_argument("--upstream-base-url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--inject-kv-transfer-params", action="store_true")
    parser.add_argument("--tensor-store-path")
    parser.add_argument("--tensor-store-block-size", type=int, default=16)
    parser.add_argument("--cold-root")
    parser.add_argument(
        "--cold-backend",
        default="local_posix",
        choices=["local_posix", "3fs_posix", "packed_v1"],
    )
    parser.add_argument("--async-prefetch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> M3SidecarConfig:
    params_path = Path(args.simulator_params)
    capacity_path = Path(args.capacity_csv) if args.capacity_csv else None
    params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    calibration = load_calibration(params_path, capacity_path)
    hbm_tokens = _best_capacity_tokens(capacity_path) or calibration.measured_gpu_kv_tokens
    return M3SidecarConfig(
        model_id=calibration.model_id,
        hbm_capacity_tokens=hbm_tokens or 0,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=calibration.kv_bytes_per_token,
        h2d_gbps=_best_h2d_gbps(params),
        storage_gbps=_best_storage_gbps(params),
        decision_log_path=Path(args.decision_log),
        upstream_base_url=args.upstream_base_url,
        inject_kv_transfer_params=args.inject_kv_transfer_params,
        tensor_store_path=Path(args.tensor_store_path) if args.tensor_store_path else None,
        tensor_store_block_size=args.tensor_store_block_size,
        cold_root=Path(args.cold_root) if args.cold_root else None,
        cold_backend=args.cold_backend,
        async_prefetch=args.async_prefetch,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config_from_args(args)
    if args.dry_run:
        payload = asdict(config)
        payload["decision_log_path"] = str(config.decision_log_path)
        if config.tensor_store_path is not None:
            payload["tensor_store_path"] = str(config.tensor_store_path)
        if config.cold_root is not None:
            payload["cold_root"] = str(config.cold_root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _best_capacity_tokens(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    started = [
        int(float(row["gpu_kv_cache_tokens"]))
        for row in rows
        if row.get("status") == "STARTED" and row.get("gpu_kv_cache_tokens")
    ]
    return max(started) if started else None


def _best_h2d_gbps(params: dict) -> float:
    values = params.get("hardware", {}).get("h2d_gbps", {})
    measured = [float(value) for value in values.values() if value not in (None, "")]
    return max(measured) if measured else 25.0


def _best_storage_gbps(params: dict) -> float:
    rows = params.get("hardware", {}).get("nvme_read_latency_by_size", {})
    measured = [
        float(value["bandwidth_gbps"])
        for value in rows.values()
        if isinstance(value, dict) and value.get("bandwidth_gbps") not in (None, "")
    ]
    return max(measured) if measured else 8.8


if __name__ == "__main__":
    raise SystemExit(main())
