#!/usr/bin/env python3
"""Generate a PrePass lead-time experiment plan from measured M3 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.m3.prepass_lead_time_planner import (
    build_lead_time_plan,
    load_restore_profile,
    write_lead_time_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepass-csv", required=True, type=Path)
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--prefix-tokens",
        nargs="+",
        type=int,
        default=[512, 2048, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--available-leads-ms",
        nargs="+",
        type=float,
        default=[0.0, 1000.0, 5000.0, 12000.0],
    )
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--tail-multiplier", type=float, default=1.2)
    parser.add_argument("--safety-margin-ms", type=float, default=250.0)
    parser.add_argument("--packed-object-prefix-tokens", type=int, default=8192)
    parser.add_argument("--packed-object-lead-ms", type=float, default=5000.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = load_restore_profile(args.prepass_csv, args.baseline_csv)
    rows = build_lead_time_plan(
        profile,
        prefix_tokens=args.prefix_tokens,
        available_leads_ms=args.available_leads_ms,
        suffix_tokens=args.suffix_tokens,
        output_tokens=args.output_tokens,
        tail_multiplier=args.tail_multiplier,
        safety_margin_ms=args.safety_margin_ms,
        packed_object_prefix_tokens=args.packed_object_prefix_tokens,
        packed_object_lead_ms=args.packed_object_lead_ms,
    )
    summary = write_lead_time_outputs(
        result_dir=args.result_dir,
        profile=profile,
        rows=rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
