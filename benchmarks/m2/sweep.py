#!/usr/bin/env python3
"""Sensitivity sweep for the M2 tiered KV cache simulator."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m2.simulator import (
    Calibration,
    SimulationConfig,
    load_calibration,
    run_simulation,
)


@dataclass(frozen=True)
class SweepConfig:
    workloads: list[str] = field(
        default_factory=lambda: ["session_append", "shared_prefix", "low_locality"]
    )
    deadlines_ms: list[float] = field(default_factory=lambda: [250.0, 1000.0, 5000.0, 12_000.0])
    prefetch_leads_ms: list[float] = field(default_factory=lambda: [0.0])
    h2d_gbps: list[float] = field(default_factory=lambda: [25.0])
    nvme_gbps: list[float] = field(default_factory=lambda: [2.5])
    dram_capacity_tokens: list[int] = field(default_factory=lambda: [1_500_000])
    hbm_capacity_tokens: list[int] = field(default_factory=lambda: [44_128])
    h2d_parallelism: list[int] = field(default_factory=lambda: [1])
    nvme_parallelism: list[int] = field(default_factory=lambda: [1])
    reuse_prediction_error_rates: list[float] = field(default_factory=lambda: [0.0])
    target_admitted_rate: float = 0.875
    base_config: SimulationConfig = field(default_factory=SimulationConfig)


@dataclass(frozen=True)
class SweepRow:
    workload: str
    deadline_ms: float
    prefetch_lead_ms: float
    h2d_gbps: float
    nvme_gbps: float
    dram_capacity_tokens: int
    hbm_capacity_tokens: int
    h2d_parallelism: int
    nvme_parallelism: int
    reuse_prediction_error_rate: float
    requests: int
    admitted_requests: int
    admitted_rate: float
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
class BoundaryRow:
    workload: str
    status: str
    target_admitted_rate: float
    deadline_ms: float | None
    prefetch_lead_ms: float | None
    h2d_gbps: float | None
    nvme_gbps: float | None
    dram_capacity_tokens: int | None
    hbm_capacity_tokens: int | None
    h2d_parallelism: int | None
    nvme_parallelism: int | None
    reuse_prediction_error_rate: float | None
    admitted_rate: float
    effective_hit_rate: float
    prefetch_deadline_miss_rate: float
    sync_ssd_miss_rate: float


def run_sweep(calibration: Calibration, sweep: SweepConfig) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for (
        workload,
        deadline_ms,
        prefetch_lead_ms,
        h2d_gbps,
        nvme_gbps,
        dram_capacity_tokens,
        hbm_capacity_tokens,
        h2d_parallelism,
        nvme_parallelism,
        reuse_prediction_error_rate,
    ) in product(
        sweep.workloads,
        sweep.deadlines_ms,
        sweep.prefetch_leads_ms,
        sweep.h2d_gbps,
        sweep.nvme_gbps,
        sweep.dram_capacity_tokens,
        sweep.hbm_capacity_tokens,
        sweep.h2d_parallelism,
        sweep.nvme_parallelism,
        sweep.reuse_prediction_error_rates,
    ):
        case_calibration = replace(calibration, h2d_gbps=h2d_gbps, nvme_gbps=nvme_gbps)
        case_config = replace(
            sweep.base_config,
            request_deadline_ms=deadline_ms,
            prefetch_lead_ms=prefetch_lead_ms,
            hbm_capacity_tokens=hbm_capacity_tokens,
            dram_capacity_tokens=dram_capacity_tokens,
            h2d_parallelism=h2d_parallelism,
            nvme_parallelism=nvme_parallelism,
            reuse_prediction_error_rate=reuse_prediction_error_rate,
        )
        result = run_simulation(case_calibration, workload, case_config)
        summary = result.summary
        admitted_rate = summary.admitted_requests / summary.requests if summary.requests else 0.0
        rows.append(
            SweepRow(
                workload=workload,
                deadline_ms=deadline_ms,
                prefetch_lead_ms=prefetch_lead_ms,
                h2d_gbps=h2d_gbps,
                nvme_gbps=nvme_gbps,
                dram_capacity_tokens=dram_capacity_tokens,
                hbm_capacity_tokens=hbm_capacity_tokens,
                h2d_parallelism=h2d_parallelism,
                nvme_parallelism=nvme_parallelism,
                reuse_prediction_error_rate=reuse_prediction_error_rate,
                requests=summary.requests,
                admitted_requests=summary.admitted_requests,
                admitted_rate=admitted_rate,
                rejected_or_delayed_requests=summary.rejected_or_delayed_requests,
                effective_hit_rate=summary.effective_hit_rate,
                prefill_tokens_saved=summary.prefill_tokens_saved,
                prefetch_deadline_miss_rate=summary.prefetch_deadline_miss_rate,
                sync_ssd_miss_rate=summary.sync_ssd_miss_rate,
                nvme_read_gb=summary.nvme_read_gb,
                h2d_gb=summary.h2d_gb,
                max_ttft_ms=summary.max_ttft_ms,
                p50_ttft_ms=summary.p50_ttft_ms,
            )
        )
    return rows


def _boundary_sort_key(row: SweepRow) -> tuple[float, float, int, int, float, float, int, int, float]:
    return (
        row.reuse_prediction_error_rate,
        -row.effective_hit_rate,
        row.deadline_ms,
        row.prefetch_lead_ms,
        row.dram_capacity_tokens,
        row.hbm_capacity_tokens,
        row.h2d_gbps,
        row.nvme_gbps,
        row.h2d_parallelism,
        row.nvme_parallelism,
    )


def _boundary_from_row(row: SweepRow, target_admitted_rate: float) -> BoundaryRow:
    return BoundaryRow(
        workload=row.workload,
        status="MEETS_TARGET",
        target_admitted_rate=target_admitted_rate,
        deadline_ms=row.deadline_ms,
        prefetch_lead_ms=row.prefetch_lead_ms,
        h2d_gbps=row.h2d_gbps,
        nvme_gbps=row.nvme_gbps,
        dram_capacity_tokens=row.dram_capacity_tokens,
        hbm_capacity_tokens=row.hbm_capacity_tokens,
        h2d_parallelism=row.h2d_parallelism,
        nvme_parallelism=row.nvme_parallelism,
        reuse_prediction_error_rate=row.reuse_prediction_error_rate,
        admitted_rate=row.admitted_rate,
        effective_hit_rate=row.effective_hit_rate,
        prefetch_deadline_miss_rate=row.prefetch_deadline_miss_rate,
        sync_ssd_miss_rate=row.sync_ssd_miss_rate,
    )


def find_boundaries(rows: list[SweepRow], target_admitted_rate: float = 0.875) -> list[BoundaryRow]:
    boundaries: list[BoundaryRow] = []
    workloads = sorted({row.workload for row in rows})
    for workload in workloads:
        workload_rows = [row for row in rows if row.workload == workload]
        candidates = [
            row
            for row in workload_rows
            if row.admitted_rate >= target_admitted_rate and row.sync_ssd_miss_rate == 0.0
        ]
        if candidates:
            boundaries.append(_boundary_from_row(min(candidates, key=_boundary_sort_key), target_admitted_rate))
            continue
        best = max(workload_rows, key=lambda row: row.admitted_rate, default=None)
        boundaries.append(
            BoundaryRow(
                workload=workload,
                status="NO_SLA_REGION",
                target_admitted_rate=target_admitted_rate,
                deadline_ms=None,
                prefetch_lead_ms=None,
                h2d_gbps=None,
                nvme_gbps=None,
                dram_capacity_tokens=None,
                hbm_capacity_tokens=None,
                h2d_parallelism=None,
                nvme_parallelism=None,
                reuse_prediction_error_rate=None,
                admitted_rate=best.admitted_rate if best else 0.0,
                effective_hit_rate=best.effective_hit_rate if best else 0.0,
                prefetch_deadline_miss_rate=best.prefetch_deadline_miss_rate if best else 0.0,
                sync_ssd_miss_rate=best.sync_ssd_miss_rate if best else 0.0,
            )
        )
    return boundaries


def find_deadline_frontier(
    rows: list[SweepRow],
    target_admitted_rate: float = 0.875,
) -> list[BoundaryRow]:
    frontier: list[BoundaryRow] = []
    groups = sorted({(row.workload, row.deadline_ms) for row in rows})
    for workload, deadline_ms in groups:
        group_rows = [
            row for row in rows if row.workload == workload and row.deadline_ms == deadline_ms
        ]
        candidates = [
            row
            for row in group_rows
            if row.admitted_rate >= target_admitted_rate and row.sync_ssd_miss_rate == 0.0
        ]
        if candidates:
            frontier.append(_boundary_from_row(min(candidates, key=_boundary_sort_key), target_admitted_rate))
            continue
        best = max(group_rows, key=lambda row: row.admitted_rate, default=None)
        frontier.append(
            BoundaryRow(
                workload=workload,
                status="NO_SLA_REGION",
                target_admitted_rate=target_admitted_rate,
                deadline_ms=deadline_ms,
                prefetch_lead_ms=None,
                h2d_gbps=None,
                nvme_gbps=None,
                dram_capacity_tokens=None,
                hbm_capacity_tokens=None,
                h2d_parallelism=None,
                nvme_parallelism=None,
                reuse_prediction_error_rate=None,
                admitted_rate=best.admitted_rate if best else 0.0,
                effective_hit_rate=best.effective_hit_rate if best else 0.0,
                prefetch_deadline_miss_rate=best.prefetch_deadline_miss_rate if best else 0.0,
                sync_ssd_miss_rate=best.sync_ssd_miss_rate if best else 0.0,
            )
        )
    return frontier


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_outputs(
    output_dir: str | Path,
    rows: list[SweepRow],
    boundaries: list[BoundaryRow],
    sweep: SweepConfig,
) -> None:
    output = Path(output_dir)
    _write_csv(output / "m2_sweep_summary.csv", [asdict(row) for row in rows])
    _write_csv(output / "m2_sweep_boundaries.csv", [asdict(row) for row in boundaries])
    frontier = find_deadline_frontier(rows, target_admitted_rate=sweep.target_admitted_rate)
    _write_csv(output / "m2_sweep_deadline_frontier.csv", [asdict(row) for row in frontier])

    lines = [
        "# M2 Sensitivity Sweep Report",
        "",
        "## Summary",
        "",
        f"- Sweep rows: {len(rows)}",
        f"- Target admitted rate: {sweep.target_admitted_rate:.3f}",
        "- Synchronous SSD miss rate must remain 0 for a boundary to count.",
        "",
        "## Boundary Results",
        "",
    ]
    for row in boundaries:
        if row.status == "MEETS_TARGET":
            lines.append(
                "- "
                f"{row.workload}: enters SLA at deadline={row.deadline_ms:g}ms, "
                f"prefetch_lead={row.prefetch_lead_ms:g}ms, "
                f"H2D={row.h2d_gbps:g}GB/s x{row.h2d_parallelism}, "
                f"NVMe/3FS={row.nvme_gbps:g}GB/s x{row.nvme_parallelism}, "
                f"DRAM tokens={row.dram_capacity_tokens}, HBM tokens={row.hbm_capacity_tokens}, "
                f"prediction_error={row.reuse_prediction_error_rate:.3f}, "
                f"admitted_rate={row.admitted_rate:.3f}, hit_rate={row.effective_hit_rate:.3f}"
            )
        else:
            lines.append(
                "- "
                f"{row.workload}: no SLA region found; best admitted_rate={row.admitted_rate:.3f}, "
                f"hit_rate={row.effective_hit_rate:.3f}"
            )
    lines.extend(
        [
            "",
            "## Deadline Frontier",
            "",
        ]
    )
    for row in frontier:
        if row.status == "MEETS_TARGET":
            lines.append(
                "- "
                f"{row.workload} at {row.deadline_ms:g}ms: "
                f"prefetch_lead={row.prefetch_lead_ms:g}ms, "
                f"H2D={row.h2d_gbps:g}GB/s x{row.h2d_parallelism}, "
                f"NVMe/3FS={row.nvme_gbps:g}GB/s x{row.nvme_parallelism}, "
                f"DRAM tokens={row.dram_capacity_tokens}, HBM tokens={row.hbm_capacity_tokens}, "
                f"hit_rate={row.effective_hit_rate:.3f}"
            )
        else:
            lines.append(
                "- "
                f"{row.workload} at {row.deadline_ms:g}ms: no SLA region found; "
                f"best admitted_rate={row.admitted_rate:.3f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A meeting row means every required SSD/NVMe read can be completed before decode; it does not permit synchronous decode-path SSD misses.",
            "- Low-locality rows are negative controls. If they do not enter SLA, the admission policy is preserving the intended boundary.",
            "- Bandwidth values are sensitivity inputs derived from local NVMe/H2D calibration or hypothetical 3FS-style targets.",
            "",
        ]
    )
    (output / "sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_floats(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def _parse_ints(values: list[str]) -> list[int]:
    return [int(float(value)) for value in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-params", required=True)
    parser.add_argument("--capacity-csv")
    parser.add_argument("--output-dir", default="results/m2_sweep")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["session_append", "shared_prefix", "low_locality"],
    )
    parser.add_argument("--deadlines-ms", nargs="+", default=["250", "1000", "5000", "12000"])
    parser.add_argument("--prefetch-leads-ms", nargs="+", default=["0", "5000", "10000"])
    parser.add_argument("--h2d-gbps", nargs="+", default=["25", "50", "100", "200"])
    parser.add_argument("--nvme-gbps", nargs="+", default=["2.5", "8.8", "20", "50"])
    parser.add_argument(
        "--dram-capacity-tokens",
        nargs="+",
        default=["500000", "1000000", "1500000", "2500000"],
    )
    parser.add_argument("--hbm-capacity-tokens", nargs="+", default=["44128", "131072", "262144"])
    parser.add_argument("--h2d-parallelism", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--nvme-parallelism", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--reuse-prediction-error-rates", nargs="+", default=["0", "0.1", "0.25"])
    parser.add_argument("--target-admitted-rate", type=float, default=0.875)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--total-tokens", type=int, default=1_048_576)
    parser.add_argument("--append-tokens", type=int, default=262_144)
    parser.add_argument("--shared-prefix-tokens", type=int, default=786_432)
    parser.add_argument("--admission-prefetch-window-ms", type=float, default=20_000.0)
    args = parser.parse_args(argv)

    base_config = SimulationConfig(
        num_requests=args.num_requests,
        total_tokens=args.total_tokens,
        append_tokens=args.append_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
        admission_prefetch_window_ms=args.admission_prefetch_window_ms,
    )
    sweep = SweepConfig(
        workloads=args.workloads,
        deadlines_ms=_parse_floats(args.deadlines_ms),
        prefetch_leads_ms=_parse_floats(args.prefetch_leads_ms),
        h2d_gbps=_parse_floats(args.h2d_gbps),
        nvme_gbps=_parse_floats(args.nvme_gbps),
        dram_capacity_tokens=_parse_ints(args.dram_capacity_tokens),
        hbm_capacity_tokens=_parse_ints(args.hbm_capacity_tokens),
        h2d_parallelism=_parse_ints(args.h2d_parallelism),
        nvme_parallelism=_parse_ints(args.nvme_parallelism),
        reuse_prediction_error_rates=_parse_floats(args.reuse_prediction_error_rates),
        target_admitted_rate=args.target_admitted_rate,
        base_config=base_config,
    )
    calibration = load_calibration(args.simulator_params, args.capacity_csv)
    rows = run_sweep(calibration, sweep)
    boundaries = find_boundaries(rows, target_admitted_rate=sweep.target_admitted_rate)
    write_sweep_outputs(args.output_dir, rows, boundaries, sweep)
    print(Path(args.output_dir) / "m2_sweep_summary.csv")
    print(Path(args.output_dir) / "m2_sweep_boundaries.csv")
    print(Path(args.output_dir) / "sweep_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
