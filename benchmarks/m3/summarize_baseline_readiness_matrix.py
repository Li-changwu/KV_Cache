#!/usr/bin/env python3
"""Summarize the M3.11 baseline readiness matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BASELINE_SUMMARY_COLUMNS = [
    "baseline_id",
    "workload",
    "prefix_tokens",
    "suffix_tokens",
    "output_tokens",
    "concurrency",
    "ok_count",
    "error_count",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "itl_p95_ms",
    "tpot_p95_ms",
    "historical_kv_hit_rate_mean",
    "historical_byte_hit_rate_mean",
    "prefetch_deadline_miss_rate",
    "sync_cold_miss_rate",
    "external_load_observed_rate",
    "restore_elapsed_p50_ms",
    "restore_elapsed_p95_ms",
    "restore_bytes_per_useful_byte_mean",
    "restore_inclusive_ttft_p50_ms",
    "restore_wait_p50_ms",
    "connector_store_elapsed_p50_ms",
    "connector_load_elapsed_p50_ms",
    "connector_total_elapsed_p50_ms",
    "unattributed_ttft_p50_ms",
]

WORKLOAD_SUMMARY_COLUMNS = [
    "workload",
    "rows",
    "ok_count",
    "error_count",
    "b5_rows",
    "b5_ok_count",
    "b5_ttft_vs_full_prefill_median",
    "b5_historical_byte_hit_rate_mean",
    "b5_sync_cold_miss_rate",
    "b5_external_load_observed_rate",
]

RATIO_GROUP_COLUMNS = [
    "model_id",
    "workload",
    "prefix_tokens",
    "suffix_tokens",
    "output_tokens",
    "concurrency",
]


def summarize_baseline_readiness_matrix(result_dir: Path) -> dict[str, Any]:
    baseline_path = result_dir / "baseline_matrix.csv"
    rows = _read_rows(baseline_path)
    if not rows:
        summary_dir = result_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(summary_dir / "baseline_summary.csv", BASELINE_SUMMARY_COLUMNS, [])
        _write_csv(summary_dir / "workload_summary.csv", WORKLOAD_SUMMARY_COLUMNS, [])
        _write_report(summary_dir / "readiness_report.md", [], [], {})
        return {"status": "EMPTY", "rows": 0}

    _fill_ttft_ratios(rows)
    _fill_stage_breakdowns(rows)
    _write_csv(baseline_path, list(rows[0].keys()), rows)
    baseline_summary = _baseline_summary(rows)
    workload_summary = _workload_summary(rows)
    summary_dir = result_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        summary_dir / "baseline_summary.csv",
        BASELINE_SUMMARY_COLUMNS,
        baseline_summary,
    )
    _write_csv(
        summary_dir / "workload_summary.csv",
        WORKLOAD_SUMMARY_COLUMNS,
        workload_summary,
    )
    status = _overall_status(rows)
    report_summary = {
        "status": status,
        "rows": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "OK"),
        "dry_run_count": sum(1 for row in rows if row.get("status") == "DRY_RUN"),
        "error_count": sum(1 for row in rows if _is_error_status(row.get("status", ""))),
    }
    _write_report(
        summary_dir / "readiness_report.md",
        baseline_summary,
        workload_summary,
        report_summary,
    )
    return report_summary


def _fill_ttft_ratios(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_ratio_group_key(row)].append(row)
    for group_rows in grouped.values():
        medians = {
            baseline: _median(_float_values(row.get("ttft_ms") for row in group_rows if row.get("baseline_id") == baseline and row.get("status") == "OK"))
            for baseline in ["B0", "B1", "B3", "B5"]
        }
        for row in group_rows:
            if row.get("baseline_id") != "B5":
                continue
            b5 = medians.get("B5")
            if b5 is None:
                continue
            row["ttft_vs_full_prefill"] = _ratio_string(b5, medians.get("B0"))
            row["ttft_vs_dram_ready_reuse"] = _ratio_string(b5, medians.get("B3"))
            row["ttft_vs_apc_hot"] = _ratio_string(b5, medians.get("B1"))


def _fill_stage_breakdowns(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if not row.get("online_service_ttft_ms"):
            row["online_service_ttft_ms"] = row.get("ttft_ms", "")
        if not row.get("restore_wait_ms"):
            row["restore_wait_ms"] = row.get("restore_elapsed_ms", "") or "0.0"
        if not row.get("restore_inclusive_ttft_ms") and row.get("online_service_ttft_ms"):
            row["restore_inclusive_ttft_ms"] = _number_string(
                _float_or_zero(row.get("online_service_ttft_ms"))
                + _float_or_zero(row.get("restore_wait_ms"))
            )
        if not row.get("connector_store_elapsed_ms"):
            row["connector_store_elapsed_ms"] = ""
        if not row.get("connector_load_elapsed_ms"):
            row["connector_load_elapsed_ms"] = ""
        if not row.get("connector_total_elapsed_ms"):
            connector_total = (
                _float_or_zero(row.get("connector_store_elapsed_ms"))
                + _float_or_zero(row.get("connector_load_elapsed_ms"))
            )
            row["connector_total_elapsed_ms"] = (
                _number_string(connector_total)
                if connector_total
                else ""
            )
        if not row.get("unattributed_ttft_ms") and row.get("online_service_ttft_ms"):
            row["unattributed_ttft_ms"] = _number_string(
                _float_or_zero(row.get("online_service_ttft_ms"))
                - _float_or_zero(row.get("connector_load_elapsed_ms"))
            )


def _baseline_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("baseline_id", ""),
            row.get("workload", ""),
            row.get("prefix_tokens", ""),
            row.get("suffix_tokens", ""),
            row.get("output_tokens", ""),
            row.get("concurrency", ""),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        ok_rows = [row for row in group_rows if row.get("status") == "OK"]
        summary_rows.append(
            {
                "baseline_id": key[0],
                "workload": key[1],
                "prefix_tokens": key[2],
                "suffix_tokens": key[3],
                "output_tokens": key[4],
                "concurrency": key[5],
                "ok_count": len(ok_rows),
                "error_count": sum(
                    1 for row in group_rows if _is_error_status(row.get("status", ""))
                ),
                "ttft_p50_ms": _format_optional(_percentile(_column_floats(ok_rows, "ttft_ms"), 50)),
                "ttft_p95_ms": _format_optional(_percentile(_column_floats(ok_rows, "ttft_ms"), 95)),
                "ttft_p99_ms": _format_optional(_percentile(_column_floats(ok_rows, "ttft_ms"), 99)),
                "itl_p95_ms": _format_optional(_percentile(_column_floats(ok_rows, "itl_p95_ms"), 95)),
                "tpot_p95_ms": _format_optional(_percentile(_column_floats(ok_rows, "tpot_p95_ms"), 95)),
                "historical_kv_hit_rate_mean": _format_optional(_mean(_column_floats(ok_rows, "historical_kv_hit_rate"))),
                "historical_byte_hit_rate_mean": _format_optional(_mean(_column_floats(ok_rows, "historical_byte_hit_rate"))),
                "prefetch_deadline_miss_rate": _format_optional(_bool_rate(ok_rows, "prefetch_deadline_miss")),
                "sync_cold_miss_rate": _format_optional(_positive_rate(ok_rows, "sync_cold_miss_total")),
                "external_load_observed_rate": _format_optional(_bool_rate(ok_rows, "external_load_observed")),
                "restore_elapsed_p50_ms": _format_optional(_percentile(_column_floats(ok_rows, "restore_elapsed_ms"), 50)),
                "restore_elapsed_p95_ms": _format_optional(_percentile(_column_floats(ok_rows, "restore_elapsed_ms"), 95)),
                "restore_bytes_per_useful_byte_mean": _format_optional(_mean(_column_floats(ok_rows, "restore_bytes_per_useful_byte"))),
                "restore_inclusive_ttft_p50_ms": _format_optional(
                    _percentile(_column_floats(ok_rows, "restore_inclusive_ttft_ms"), 50)
                ),
                "restore_wait_p50_ms": _format_optional(
                    _percentile(_column_floats(ok_rows, "restore_wait_ms"), 50)
                ),
                "connector_store_elapsed_p50_ms": _format_optional(
                    _percentile(_column_floats(ok_rows, "connector_store_elapsed_ms"), 50)
                ),
                "connector_load_elapsed_p50_ms": _format_optional(
                    _percentile(_column_floats(ok_rows, "connector_load_elapsed_ms"), 50)
                ),
                "connector_total_elapsed_p50_ms": _format_optional(
                    _percentile(_column_floats(ok_rows, "connector_total_elapsed_ms"), 50)
                ),
                "unattributed_ttft_p50_ms": _format_optional(
                    _percentile(_column_floats(ok_rows, "unattributed_ttft_ms"), 50)
                ),
                "ttft_vs_full_prefill_median": _format_optional(
                    _median(_column_floats(ok_rows, "ttft_vs_full_prefill"))
                ),
                "ttft_vs_dram_ready_reuse_median": _format_optional(
                    _median(_column_floats(ok_rows, "ttft_vs_dram_ready_reuse"))
                ),
                "ttft_vs_apc_hot_median": _format_optional(
                    _median(_column_floats(ok_rows, "ttft_vs_apc_hot"))
                ),
            }
        )
    return summary_rows


def _workload_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("workload", "")].append(row)
    summary_rows: list[dict[str, Any]] = []
    for workload in sorted(grouped):
        group_rows = grouped[workload]
        b5_rows = [row for row in group_rows if row.get("baseline_id") == "B5"]
        b5_ok = [row for row in b5_rows if row.get("status") == "OK"]
        summary_rows.append(
            {
                "workload": workload,
                "rows": len(group_rows),
                "ok_count": sum(1 for row in group_rows if row.get("status") == "OK"),
                "error_count": sum(
                    1 for row in group_rows if _is_error_status(row.get("status", ""))
                ),
                "b5_rows": len(b5_rows),
                "b5_ok_count": len(b5_ok),
                "b5_ttft_vs_full_prefill_median": _format_optional(
                    _median(_column_floats(b5_ok, "ttft_vs_full_prefill"))
                ),
                "b5_historical_byte_hit_rate_mean": _format_optional(
                    _mean(_column_floats(b5_ok, "historical_byte_hit_rate"))
                ),
                "b5_sync_cold_miss_rate": _format_optional(
                    _positive_rate(b5_ok, "sync_cold_miss_total")
                ),
                "b5_external_load_observed_rate": _format_optional(
                    _bool_rate(b5_ok, "external_load_observed")
                ),
            }
        )
    return summary_rows


def _write_report(
    path: Path,
    baseline_summary: list[dict[str, Any]],
    workload_summary: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# M3.11 Baseline Readiness Report",
        "",
        f"- Status: `{summary.get('status', 'EMPTY')}`",
        f"- Rows: `{summary.get('rows', 0)}`",
        f"- OK rows: `{summary.get('ok_count', 0)}`",
        f"- Dry-run rows: `{summary.get('dry_run_count', 0)}`",
        f"- Error/boundary rows: `{summary.get('error_count', 0)}`",
        "",
        "## B5 vs B0",
        "",
    ]
    b5_rows = [row for row in baseline_summary if row.get("baseline_id") == "B5"]
    if not b5_rows:
        lines.append("- No B5 rows available yet.")
    else:
        selected_rows = _select_report_b5_rows(b5_rows)
        for row in selected_rows:
            ratio = _find_b5_ratio_text(row)
            lines.append(
                "- "
                f"workload=`{row.get('workload')}`, "
                f"prefix=`{row.get('prefix_tokens')}`, "
                f"suffix=`{row.get('suffix_tokens')}`, "
                f"output=`{row.get('output_tokens')}`, "
                f"ok=`{row.get('ok_count')}`, "
                f"sync_cold_miss_rate=`{row.get('sync_cold_miss_rate')}`, "
                f"external_load_rate=`{row.get('external_load_observed_rate')}`, "
                f"B5 vs B0 `{ratio}`, "
                f"restore_inclusive_ttft_p50_ms=`{row.get('restore_inclusive_ttft_p50_ms')}`, "
                f"connector_total_p50_ms=`{row.get('connector_total_elapsed_p50_ms')}`, "
                f"unattributed_p50_ms=`{row.get('unattributed_ttft_p50_ms')}`"
            )
        omitted = len(b5_rows) - len(selected_rows)
        if omitted > 0:
            lines.append(f"- Omitted `{omitted}` additional B5 shape rows from this short report.")
    lines.extend(["", "## Workloads", ""])
    for row in workload_summary:
        lines.append(
            "- "
            f"`{row.get('workload')}`: rows=`{row.get('rows')}`, "
            f"B5 ok=`{row.get('b5_ok_count')}/{row.get('b5_rows')}`, "
            f"B5 byte hit=`{row.get('b5_historical_byte_hit_rate_mean')}`, "
            f"B5 external load=`{row.get('b5_external_load_observed_rate')}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `DRY_RUN` rows validate schema and artifact layout only.",
            "- `ERROR_BOUNDARY` rows are preserved to avoid hiding context-limit cases.",
            "- Online execution must later prove connector external-load events for B3/B5.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_b5_ratio_text(row: dict[str, Any]) -> str:
    value = row.get("ttft_vs_full_prefill_median") or ""
    return str(value) if value else "pending"


def _select_report_b5_rows(rows: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    with_ratio = [row for row in rows if row.get("ttft_vs_full_prefill_median")]
    if with_ratio:
        return with_ratio[:limit]
    preferred_prefixes = {"8192", "16384", "32768"}
    preferred_workloads = {"session_append", "shared_prefix"}
    selected = [
        row
        for row in rows
        if str(row.get("prefix_tokens")) in preferred_prefixes
        and str(row.get("workload")) in preferred_workloads
    ]
    return (selected or rows)[:limit]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _ratio_group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in RATIO_GROUP_COLUMNS)


def _ratio_string(numerator: float, denominator: float | None) -> str:
    if denominator is None or denominator == 0:
        return ""
    return str(round(numerator / denominator, 6)).rstrip("0").rstrip(".")


def _overall_status(rows: list[dict[str, str]]) -> str:
    if all(row.get("status") == "DRY_RUN" for row in rows):
        return "DRY_RUN"
    if any(row.get("status") == "OK" for row in rows):
        return "OK"
    if any(_is_error_status(row.get("status", "")) for row in rows):
        return "PARTIAL"
    return "EMPTY"


def _is_error_status(status: str) -> bool:
    return status in {"ERROR", "OOM", "TIMEOUT", "ERROR_BOUNDARY"}


def _column_floats(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    return _float_values(row.get(column) for row in rows)


def _float_values(values: Iterable[str | None]) -> list[float]:
    floats: list[float] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            floats.append(float(value))
        except ValueError:
            continue
    return floats


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _bool_rate(rows: list[dict[str, str]], column: str) -> float | None:
    if not rows:
        return None
    positives = sum(1 for row in rows if _truthy(row.get(column, "")))
    return positives / len(rows)


def _positive_rate(rows: list[dict[str, str]], column: str) -> float | None:
    if not rows:
        return None
    positives = 0
    for row in rows:
        try:
            positives += 1 if float(row.get(column, "") or 0) > 0 else 0
        except ValueError:
            continue
    return positives / len(rows)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return str(round(float(value), 6)).rstrip("0").rstrip(".") + (
        ".0" if float(value).is_integer() else ""
    )


def _float_or_zero(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def _number_string(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded) == 0:
        rounded = 0.0
    text = f"{rounded:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/m3_11_baseline_readiness")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_baseline_readiness_matrix(Path(args.result_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"OK", "DRY_RUN", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
