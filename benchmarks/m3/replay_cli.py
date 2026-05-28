#!/usr/bin/env python3
"""Run the M3 sidecar replay using measured calibration inputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.control_plane import (
    CorrectnessKey,
    PrefixCandidate,
    SidecarRequest,
    TieredKVControlPlane,
)
from benchmarks.m3.replay import run_replay, write_replay_outputs
from benchmarks.m2.simulator import load_calibration


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def _correctness_key(model_id: str) -> CorrectnessKey:
    return CorrectnessKey(
        model_fingerprint=model_id,
        tokenizer_fingerprint="qwen2.5-tokenizer",
        rope_config="native-32768",
        dtype="bf16",
        kv_layout="vllm-paged",
    )


def _default_requests(model_id: str) -> list[SidecarRequest]:
    correctness_key = _correctness_key(model_id)
    return [
        SidecarRequest(
            request_id="session_a_turn_1",
            model_id=model_id,
            token_count=1_048_576,
            decode_sla_ms=12_000.0,
            admission_window_ms=12_000.0,
            correctness_key=correctness_key,
            prefix_candidates=[],
        ),
        SidecarRequest(
            request_id="session_a_turn_2",
            model_id=model_id,
            token_count=1_048_576,
            decode_sla_ms=12_000.0,
            admission_window_ms=12_000.0,
            correctness_key=correctness_key,
            prefix_candidates=[
                PrefixCandidate(
                    prefix_id="session-a",
                    token_start=0,
                    token_end=786_432,
                    committed=True,
                )
            ],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-params", required=True)
    parser.add_argument("--capacity-csv")
    parser.add_argument("--output-dir", default="results/m3_sidecar_replay")
    args = parser.parse_args(argv)

    params_path = Path(args.simulator_params)
    capacity_path = Path(args.capacity_csv) if args.capacity_csv else None
    params = _load_yaml(params_path)
    calibration = load_calibration(params_path, capacity_path)
    hbm_tokens = _best_capacity_tokens(capacity_path) or calibration.measured_gpu_kv_tokens or 0
    control_plane = TieredKVControlPlane(
        hbm_capacity_tokens=hbm_tokens,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=calibration.kv_bytes_per_token,
        h2d_gbps=_best_h2d_gbps(params),
        storage_gbps=_best_storage_gbps(params),
    )
    rows = run_replay(
        control_plane,
        _default_requests(calibration.model_id),
        commit_prefix_id="session-a",
    )
    write_replay_outputs(args.output_dir, rows)
    print(Path(args.output_dir) / "m3_replay_decisions.csv")
    print(Path(args.output_dir) / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
