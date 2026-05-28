#!/usr/bin/env python3
"""First-pass tiered KV cache simulator for exact 1M-token workloads."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m2.workloads import Request, build_workload


DEFAULT_KV_BYTES_PER_TOKEN = 64 * 8 * 128 * 2 * 2
TOKENS_1M = 1_048_576


@dataclass(frozen=True)
class Calibration:
    model_id: str
    kv_bytes_per_token: int
    hbm_total_bytes: int
    dram_total_bytes: int
    h2d_gbps: float
    nvme_gbps: float
    measured_gpu_kv_tokens: int | None
    measured_max_model_len: int | None
    vllm_version: str


@dataclass(frozen=True)
class SimulationConfig:
    hbm_capacity_tokens: int | None = None
    dram_capacity_tokens: int = 1_500_000
    nvme_capacity_tokens: int = 8_000_000
    request_deadline_ms: float = 250.0
    admission_prefetch_window_ms: float = 10_000.0
    num_requests: int = 8
    total_tokens: int = TOKENS_1M
    append_tokens: int = 262_144
    shared_prefix_tokens: int = 786_432
    hbm_resident_fraction: float = 0.125
    prefetch_lead_ms: float = 0.0
    h2d_parallelism: int = 1
    nvme_parallelism: int = 1
    reuse_prediction_error_rate: float = 0.0


@dataclass(frozen=True)
class RequestResult:
    workload: str
    request_id: str
    total_tokens: int
    reuse_tokens: int
    new_tokens: int
    hbm_hit_tokens: int
    dram_hit_tokens: int
    nvme_prefetch_tokens: int
    prefill_tokens: int
    prefill_tokens_saved: int
    prefetch_ms: float
    prefetch_critical_path_ms: float
    ttft_estimate_ms: float
    deadline_miss: bool
    admitted: bool
    delayed: bool
    rejected: bool
    sync_ssd_miss: bool


@dataclass(frozen=True)
class SummaryResult:
    workload: str
    requests: int
    admitted_requests: int
    rejected_or_delayed_requests: int
    effective_hit_rate: float
    prefill_tokens_saved: int
    prefetch_deadline_miss_rate: float
    sync_ssd_miss_rate: float
    nvme_read_gb: float
    h2d_gb: float
    max_ttft_ms: float
    p50_ttft_ms: float


@dataclass(frozen=True)
class SimulationResult:
    workload: str
    summary: SummaryResult
    requests: list[RequestResult]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_capacity(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _best_float(values: dict[str, Any]) -> float:
    candidates = [float(value) for value in values.values() if value not in (None, "")]
    if not candidates:
        return 1.0
    return max(candidates)


def _best_nvme_gbps(values: dict[str, Any]) -> float:
    candidates = []
    for metrics in values.values():
        if isinstance(metrics, dict) and metrics.get("bandwidth_gbps") not in (None, ""):
            candidates.append(float(metrics["bandwidth_gbps"]))
    return max(candidates) if candidates else 1.0


def _best_capacity(rows: list[dict[str, str]]) -> tuple[int | None, int | None]:
    started = []
    for row in rows:
        if row.get("status") != "STARTED":
            continue
        tokens = row.get("gpu_kv_cache_tokens")
        max_len = row.get("max_model_len")
        if tokens and max_len:
            started.append((int(float(tokens)), int(float(max_len))))
    if not started:
        return None, None
    return max(started, key=lambda item: item[0])


def load_calibration(
    simulator_params: str | Path,
    capacity_csv: str | Path | None = None,
) -> Calibration:
    params_path = Path(simulator_params)
    params = _read_yaml(params_path)
    hardware = params.get("hardware") or {}
    vllm = params.get("vllm") or {}
    capacity_path = Path(capacity_csv) if capacity_csv is not None else None
    measured_tokens, measured_max_len = _best_capacity(_read_capacity(capacity_path))

    return Calibration(
        model_id=str(vllm.get("model_id") or ""),
        kv_bytes_per_token=int(vllm.get("kv_bytes_per_token") or DEFAULT_KV_BYTES_PER_TOKEN),
        hbm_total_bytes=int(hardware.get("hbm_total_bytes") or 0),
        dram_total_bytes=int(hardware.get("dram_total_bytes") or 0),
        h2d_gbps=_best_float(hardware.get("h2d_gbps") or {}),
        nvme_gbps=_best_nvme_gbps(hardware.get("nvme_read_latency_by_size") or {}),
        measured_gpu_kv_tokens=measured_tokens,
        measured_max_model_len=measured_max_len,
        vllm_version=str(vllm.get("version") or "unknown"),
    )


def _transfer_ms(tokens: int, kv_bytes_per_token: int, gbps: float) -> float:
    if tokens <= 0:
        return 0.0
    bytes_to_transfer = tokens * kv_bytes_per_token
    return bytes_to_transfer / (gbps * 1_000_000_000) * 1000.0


def _parallelized(ms: float, workers: int) -> float:
    return ms / max(workers, 1)


def _prefill_ms(tokens: int, total_tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    # A conservative shape function: full 1M prefill is deliberately expensive.
    return 50.0 + (tokens / max(total_tokens, 1)) * 5000.0


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _simulate_request(
    request: Request,
    calibration: Calibration,
    config: SimulationConfig,
    hbm_capacity_tokens: int,
    committed_prefix_tokens: dict[str, int],
) -> RequestResult:
    reusable_tokens = min(
        request.reuse_tokens,
        committed_prefix_tokens.get(request.prefix_id, 0),
        request.total_tokens,
    )
    prediction_keep_rate = 1.0 - min(max(config.reuse_prediction_error_rate, 0.0), 1.0)
    reusable_tokens = int(reusable_tokens * prediction_keep_rate)
    hbm_hit_tokens = min(reusable_tokens, hbm_capacity_tokens)
    dram_hit_tokens = max(0, min(reusable_tokens - hbm_hit_tokens, config.dram_capacity_tokens))
    nvme_prefetch_tokens = max(0, reusable_tokens - hbm_hit_tokens - dram_hit_tokens)
    prefill_tokens = request.total_tokens - reusable_tokens
    prefill_tokens_saved = reusable_tokens

    nvme_ms = _transfer_ms(
        nvme_prefetch_tokens,
        calibration.kv_bytes_per_token,
        calibration.nvme_gbps,
    )
    h2d_ms = _transfer_ms(
        dram_hit_tokens + nvme_prefetch_tokens,
        calibration.kv_bytes_per_token,
        calibration.h2d_gbps,
    )
    prefetch_ms = _parallelized(nvme_ms, config.nvme_parallelism) + _parallelized(
        h2d_ms,
        config.h2d_parallelism,
    )
    prefetch_critical_path_ms = max(0.0, prefetch_ms - config.prefetch_lead_ms)
    prefill_ms = _prefill_ms(prefill_tokens, request.total_tokens)
    ttft_estimate_ms = prefetch_critical_path_ms + prefill_ms

    capacity_ok = request.total_tokens <= (
        hbm_capacity_tokens + config.dram_capacity_tokens + config.nvme_capacity_tokens
    )
    prefetch_ok = prefetch_ms <= config.admission_prefetch_window_ms
    reuse_required_for_sla = request.total_tokens >= TOKENS_1M and reusable_tokens == 0
    rejected = not capacity_ok
    deadline_miss = prefetch_critical_path_ms > config.request_deadline_ms
    admitted = capacity_ok and prefetch_ok and not deadline_miss and not reuse_required_for_sla
    delayed = capacity_ok and (not admitted)

    committed_prefix_tokens[request.prefix_id] = max(
        committed_prefix_tokens.get(request.prefix_id, 0),
        request.total_tokens,
    )

    return RequestResult(
        workload=request.workload,
        request_id=request.request_id,
        total_tokens=request.total_tokens,
        reuse_tokens=reusable_tokens,
        new_tokens=request.new_tokens,
        hbm_hit_tokens=hbm_hit_tokens,
        dram_hit_tokens=dram_hit_tokens,
        nvme_prefetch_tokens=nvme_prefetch_tokens,
        prefill_tokens=prefill_tokens,
        prefill_tokens_saved=prefill_tokens_saved,
        prefetch_ms=round(prefetch_ms, 3),
        prefetch_critical_path_ms=round(prefetch_critical_path_ms, 3),
        ttft_estimate_ms=round(ttft_estimate_ms, 3),
        deadline_miss=deadline_miss,
        admitted=admitted,
        delayed=delayed,
        rejected=rejected,
        sync_ssd_miss=False,
    )


def _summarize(workload: str, rows: list[RequestResult], calibration: Calibration) -> SummaryResult:
    total_tokens = sum(row.total_tokens for row in rows)
    saved_tokens = sum(row.prefill_tokens_saved for row in rows)
    nvme_tokens = sum(row.nvme_prefetch_tokens for row in rows)
    transfer_tokens = sum(row.dram_hit_tokens + row.nvme_prefetch_tokens for row in rows)
    ttfts = [row.ttft_estimate_ms for row in rows]
    requests = len(rows)
    rejected_or_delayed = sum(1 for row in rows if row.rejected or row.delayed)

    return SummaryResult(
        workload=workload,
        requests=requests,
        admitted_requests=sum(1 for row in rows if row.admitted),
        rejected_or_delayed_requests=rejected_or_delayed,
        effective_hit_rate=saved_tokens / total_tokens if total_tokens else 0.0,
        prefill_tokens_saved=saved_tokens,
        prefetch_deadline_miss_rate=sum(1 for row in rows if row.deadline_miss) / requests
        if requests
        else 0.0,
        sync_ssd_miss_rate=sum(1 for row in rows if row.sync_ssd_miss) / requests
        if requests
        else 0.0,
        nvme_read_gb=nvme_tokens * calibration.kv_bytes_per_token / 1_000_000_000,
        h2d_gb=transfer_tokens * calibration.kv_bytes_per_token / 1_000_000_000,
        max_ttft_ms=max(ttfts) if ttfts else 0.0,
        p50_ttft_ms=_p50(ttfts),
    )


def run_simulation(
    calibration: Calibration,
    workload: str,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    config = config or SimulationConfig()
    hbm_capacity_tokens = (
        config.hbm_capacity_tokens
        or calibration.measured_gpu_kv_tokens
        or int(calibration.hbm_total_bytes / calibration.kv_bytes_per_token * config.hbm_resident_fraction)
    )
    requests = build_workload(
        workload,
        num_requests=config.num_requests,
        total_tokens=config.total_tokens,
        append_tokens=config.append_tokens,
        shared_prefix_tokens=config.shared_prefix_tokens,
    )
    committed_prefix_tokens: dict[str, int] = {}
    rows = [
        _simulate_request(
            request,
            calibration,
            config,
            hbm_capacity_tokens,
            committed_prefix_tokens,
        )
        for request in requests
    ]
    return SimulationResult(
        workload=workload,
        summary=_summarize(workload, rows, calibration),
        requests=rows,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: str | Path, results: list[SimulationResult]) -> None:
    output = Path(output_dir)
    summary_rows = [asdict(result.summary) for result in results]
    request_rows = [
        asdict(request)
        for result in results
        for request in result.requests
    ]
    _write_csv(output / "m2_summary.csv", summary_rows)
    _write_csv(output / "m2_requests.csv", request_rows)

    lines = [
        "# M2 Tiered KV Simulation Report",
        "",
        "## Summary",
        "",
    ]
    for result in results:
        summary = result.summary
        lines.append(
            "- "
            f"{summary.workload}: hit_rate={summary.effective_hit_rate:.3f}, "
            f"saved_tokens={summary.prefill_tokens_saved}, "
            f"deadline_miss_rate={summary.prefetch_deadline_miss_rate:.3f}, "
            f"reject_or_delay={summary.rejected_or_delayed_requests}, "
            f"sync_ssd_miss_rate={summary.sync_ssd_miss_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- SSD/NVMe reads are modeled only as pre-decode prefetch. Synchronous decode-path SSD misses are not allowed.",
            "- Low-locality 1M requests are treated as negative controls and may be delayed or rejected by admission.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-params", required=True)
    parser.add_argument("--capacity-csv")
    parser.add_argument("--output-dir", default="results/m2_simulation")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["session_append", "shared_prefix", "low_locality"],
    )
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--hbm-capacity-tokens", type=int)
    parser.add_argument("--dram-capacity-tokens", type=int, default=1_500_000)
    parser.add_argument("--nvme-capacity-tokens", type=int, default=8_000_000)
    parser.add_argument("--request-deadline-ms", type=float, default=250.0)
    parser.add_argument("--admission-prefetch-window-ms", type=float, default=10_000.0)
    args = parser.parse_args(argv)

    calibration = load_calibration(args.simulator_params, args.capacity_csv)
    config = SimulationConfig(
        hbm_capacity_tokens=args.hbm_capacity_tokens,
        dram_capacity_tokens=args.dram_capacity_tokens,
        nvme_capacity_tokens=args.nvme_capacity_tokens,
        request_deadline_ms=args.request_deadline_ms,
        admission_prefetch_window_ms=args.admission_prefetch_window_ms,
        num_requests=args.num_requests,
    )
    results = [run_simulation(calibration, workload, config) for workload in args.workloads]
    write_outputs(args.output_dir, results)
    print(Path(args.output_dir) / "m2_summary.csv")
    print(Path(args.output_dir) / "m2_requests.csv")
    print(Path(args.output_dir) / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
