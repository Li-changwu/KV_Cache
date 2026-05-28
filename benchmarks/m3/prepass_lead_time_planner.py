"""PrePass lead-time planner for sizing KV Anti-Caching experiments.

This module consumes measured M3.11/M3.12 rows and turns them into a small
risk-gated experiment plan. It does not run vLLM; it decides which expensive
online matrix rows are worth running next.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PACKED_OBJECT_PREFIX_TOKENS = 8192
DEFAULT_PACKED_OBJECT_LEAD_MS = 5000.0


@dataclass(frozen=True)
class RestoreProfile:
    source_prefix_tokens: int
    measured_restore_bytes: int
    measured_restore_ms: float
    kv_bytes_per_token: int
    prepass_overhead_ms: float
    connector_load_ms: float
    online_ttft_ms: float
    b0_ttft_ms: float | None = None
    b3_ttft_ms: float | None = None
    b5_reactive_ttft_ms: float | None = None
    source_prepass_csv: str = ""
    source_baseline_csv: str = ""

    @property
    def restore_bytes_per_ms(self) -> float:
        return self.measured_restore_bytes / self.measured_restore_ms

    @property
    def restore_mib_per_s(self) -> float:
        return self.restore_bytes_per_ms * 1000.0 / (1024.0 * 1024.0)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["restore_bytes_per_ms"] = self.restore_bytes_per_ms
        payload["restore_mib_per_s"] = self.restore_mib_per_s
        return payload


@dataclass(frozen=True)
class LeadTimePlanRow:
    experiment_order: int
    prefix_tokens: int
    suffix_tokens: int
    output_tokens: int
    available_lead_ms: float
    estimated_kv_bytes: int
    estimated_restore_ms: float
    prepass_overhead_ms: float
    safety_margin_ms: float
    required_lead_ms: float
    residual_online_wait_ms: float
    hide_restore: bool
    packed_object_required: bool
    priority_class: str
    risk_class: str
    reason: str

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "experiment_order": self.experiment_order,
            "prefix_tokens": self.prefix_tokens,
            "suffix_tokens": self.suffix_tokens,
            "output_tokens": self.output_tokens,
            "available_lead_ms": _number_string(self.available_lead_ms),
            "estimated_kv_bytes": self.estimated_kv_bytes,
            "estimated_restore_ms": _number_string(self.estimated_restore_ms),
            "prepass_overhead_ms": _number_string(self.prepass_overhead_ms),
            "safety_margin_ms": _number_string(self.safety_margin_ms),
            "required_lead_ms": _number_string(self.required_lead_ms),
            "residual_online_wait_ms": _number_string(self.residual_online_wait_ms),
            "hide_restore": _bool_string(self.hide_restore),
            "packed_object_required": _bool_string(self.packed_object_required),
            "priority_class": self.priority_class,
            "risk_class": self.risk_class,
            "reason": self.reason,
        }


PLAN_COLUMNS = [
    "experiment_order",
    "prefix_tokens",
    "suffix_tokens",
    "output_tokens",
    "available_lead_ms",
    "estimated_kv_bytes",
    "estimated_restore_ms",
    "prepass_overhead_ms",
    "safety_margin_ms",
    "required_lead_ms",
    "residual_online_wait_ms",
    "hide_restore",
    "packed_object_required",
    "priority_class",
    "risk_class",
    "reason",
]


def load_restore_profile(prepass_csv: Path, baseline_csv: Path) -> RestoreProfile:
    """Load the best measured true-cold PrePass profile from CSV artifacts."""
    prepass_rows = _read_csv(prepass_csv)
    baseline_rows = _read_csv(baseline_csv)
    row = _select_prepass_row(prepass_rows)
    source_prefix_tokens = _required_int(row, "prefix_tokens")
    measured_restore_bytes = (
        _int_or_none(row.get("cold_restore_actual_bytes"))
        or _int_or_none(row.get("prepass_restore_actual_bytes"))
        or 0
    )
    measured_restore_ms = (
        _float_or_none(row.get("cold_restore_executor_elapsed_ms"))
        or _float_or_none(row.get("prepass_restore_executor_elapsed_ms"))
        or 0.0
    )
    if measured_restore_bytes <= 0:
        raise ValueError(f"no restore byte count found in {prepass_csv}")
    if measured_restore_ms <= 0:
        raise ValueError(f"no positive restore elapsed ms found in {prepass_csv}")

    kv_bytes_per_token = _infer_kv_bytes_per_token(
        baseline_rows=baseline_rows,
        source_prefix_tokens=source_prefix_tokens,
        measured_restore_bytes=measured_restore_bytes,
    )
    return RestoreProfile(
        source_prefix_tokens=source_prefix_tokens,
        measured_restore_bytes=measured_restore_bytes,
        measured_restore_ms=measured_restore_ms,
        kv_bytes_per_token=kv_bytes_per_token,
        prepass_overhead_ms=_float_or_zero(row.get("prepass_elapsed_ms")),
        connector_load_ms=_float_or_zero(row.get("connector_load_elapsed_ms")),
        online_ttft_ms=_float_or_zero(row.get("second_ttft_ms")),
        b0_ttft_ms=_baseline_ttft(baseline_rows, "B0", source_prefix_tokens),
        b3_ttft_ms=_baseline_ttft(baseline_rows, "B3", source_prefix_tokens),
        b5_reactive_ttft_ms=_baseline_ttft(baseline_rows, "B5", source_prefix_tokens),
        source_prepass_csv=str(prepass_csv),
        source_baseline_csv=str(baseline_csv),
    )


def build_lead_time_plan(
    profile: RestoreProfile,
    *,
    prefix_tokens: Iterable[int],
    available_leads_ms: Iterable[float],
    suffix_tokens: int = 128,
    output_tokens: int = 1,
    tail_multiplier: float = 1.2,
    safety_margin_ms: float = 250.0,
    packed_object_prefix_tokens: int = DEFAULT_PACKED_OBJECT_PREFIX_TOKENS,
    packed_object_lead_ms: float = DEFAULT_PACKED_OBJECT_LEAD_MS,
) -> list[LeadTimePlanRow]:
    """Build rows ordered by calibration value before expensive vLLM runs."""
    ordered_prefixes = sorted({int(value) for value in prefix_tokens})
    ordered_leads = sorted({float(value) for value in available_leads_ms})
    if not ordered_prefixes:
        raise ValueError("prefix_tokens must not be empty")
    if not ordered_leads:
        raise ValueError("available_leads_ms must not be empty")

    prefix_orders = {
        prefix: index + 1 for index, prefix in enumerate(_prioritize_prefixes(ordered_prefixes, profile))
    }
    rows: list[LeadTimePlanRow] = []
    for prefix in sorted(ordered_prefixes, key=lambda value: prefix_orders[value]):
        estimated_kv_bytes = int(prefix * profile.kv_bytes_per_token)
        estimated_restore_ms = (
            profile.measured_restore_ms
            * (estimated_kv_bytes / profile.measured_restore_bytes)
            * tail_multiplier
        )
        required_lead_ms = (
            estimated_restore_ms + profile.prepass_overhead_ms + safety_margin_ms
        )
        packed_required = (
            prefix >= packed_object_prefix_tokens
            or required_lead_ms > packed_object_lead_ms
        )
        priority_class = _priority_class(prefix, profile.source_prefix_tokens)
        for lead_ms in ordered_leads:
            residual_wait = max(0.0, required_lead_ms - lead_ms)
            hide_restore = residual_wait == 0.0
            rows.append(
                LeadTimePlanRow(
                    experiment_order=prefix_orders[prefix],
                    prefix_tokens=prefix,
                    suffix_tokens=suffix_tokens,
                    output_tokens=output_tokens,
                    available_lead_ms=lead_ms,
                    estimated_kv_bytes=estimated_kv_bytes,
                    estimated_restore_ms=estimated_restore_ms,
                    prepass_overhead_ms=profile.prepass_overhead_ms,
                    safety_margin_ms=safety_margin_ms,
                    required_lead_ms=required_lead_ms,
                    residual_online_wait_ms=residual_wait,
                    hide_restore=hide_restore,
                    packed_object_required=packed_required,
                    priority_class=priority_class,
                    risk_class=_risk_class(hide_restore, packed_required),
                    reason=_reason(prefix, lead_ms, required_lead_ms, packed_required),
                )
            )
    return rows


def write_lead_time_outputs(
    *,
    result_dir: Path,
    profile: RestoreProfile,
    rows: list[LeadTimePlanRow],
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "prepass_lead_time_plan.csv"
    report_path = result_dir / "prepass_lead_time_report.md"
    profile_path = result_dir / "restore_profile.json"
    _write_plan_csv(csv_path, rows)
    profile_path.write_text(
        json.dumps(profile.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, profile, rows)
    return {
        "status": "OK",
        "rows": len(rows),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "profile_path": str(profile_path),
    }


def _select_prepass_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [row for row in rows if row.get("status") in {"OK", ""}]
    if not candidates:
        raise ValueError("no successful PrePass rows found")
    with_cold_restore = [
        row
        for row in candidates
        if _int_or_none(row.get("cold_restore_actual_bytes"))
        and _float_or_none(row.get("cold_restore_executor_elapsed_ms"))
    ]
    return (with_cold_restore or candidates)[0]


def _infer_kv_bytes_per_token(
    *,
    baseline_rows: list[dict[str, str]],
    source_prefix_tokens: int,
    measured_restore_bytes: int,
) -> int:
    for row in baseline_rows:
        if row.get("status") != "OK":
            continue
        value = _int_or_none(row.get("kv_bytes_per_token"))
        if value:
            return value
    for row in baseline_rows:
        prefix = _int_or_none(row.get("prefix_tokens"))
        historical_bytes = _int_or_none(row.get("historical_bytes_required"))
        if prefix and historical_bytes:
            return int(round(historical_bytes / prefix))
    return int(round(measured_restore_bytes / source_prefix_tokens))


def _baseline_ttft(
    rows: list[dict[str, str]],
    baseline_id: str,
    source_prefix_tokens: int,
) -> float | None:
    for row in rows:
        if row.get("baseline_id") != baseline_id or row.get("status") != "OK":
            continue
        if _int_or_none(row.get("prefix_tokens")) != source_prefix_tokens:
            continue
        value = _float_or_none(row.get("ttft_ms"))
        if value is not None:
            return value
    return None


def _prioritize_prefixes(prefixes: list[int], profile: RestoreProfile) -> list[int]:
    source = profile.source_prefix_tokens

    def key(prefix: int) -> tuple[int, int]:
        if prefix < source:
            return (0, prefix)
        if prefix == source:
            return (1, prefix)
        if prefix <= source * 4:
            return (2, prefix)
        if prefix < 32768:
            return (3, prefix)
        return (4, prefix)

    return sorted(prefixes, key=key)


def _priority_class(prefix: int, source_prefix_tokens: int) -> str:
    if prefix < source_prefix_tokens:
        return "downscale_calibration"
    if prefix == source_prefix_tokens:
        return "source_repeat"
    if prefix <= source_prefix_tokens * 4:
        return "first_scaling_probe"
    if prefix < 32768:
        return "packed_layout_gate"
    return "context_limit_boundary"


def _risk_class(hide_restore: bool, packed_required: bool) -> str:
    if not hide_restore:
        return "LEAD_TIME_RISK"
    if packed_required:
        return "LAYOUT_RISK"
    return "OK"


def _reason(
    prefix_tokens: int,
    available_lead_ms: float,
    required_lead_ms: float,
    packed_required: bool,
) -> str:
    if available_lead_ms < required_lead_ms:
        return (
            "available lead is below estimated restore requirement; online request "
            "would still wait for cold KV"
        )
    if packed_required:
        return (
            "restore can be hidden only if layout tail remains controlled; validate "
            "packed cold object before scaling further"
        )
    if prefix_tokens <= 2048:
        return "calibration row for validating the 2K restore-scaling model"
    return "lead time is sufficient under current linear restore model"


def _write_plan_csv(path: Path, rows: list[LeadTimePlanRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _write_report(path: Path, profile: RestoreProfile, rows: list[LeadTimePlanRow]) -> None:
    hidden = sum(1 for row in rows if row.hide_restore)
    lead_risk = sum(1 for row in rows if row.risk_class == "LEAD_TIME_RISK")
    packed_rows = sum(1 for row in rows if row.packed_object_required)
    first_rows = _first_lead_rows(rows)
    lines = [
        "# M3.13 PrePass Lead-Time Plan",
        "",
        "## Source Profile",
        "",
        f"- Source prefix tokens: `{profile.source_prefix_tokens}`",
        f"- Measured restore bytes: `{profile.measured_restore_bytes}`",
        f"- Measured restore elapsed: `{_number_string(profile.measured_restore_ms)} ms`",
        f"- Restore throughput: `{_number_string(profile.restore_mib_per_s)} MiB/s`",
        f"- KV bytes/token: `{profile.kv_bytes_per_token}`",
        f"- PrePass control overhead: `{_number_string(profile.prepass_overhead_ms)} ms`",
        "",
        "## Plan Summary",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Restore hidden rows: `{hidden}`",
        f"- Lead-time risk rows: `{lead_risk}`",
        f"- Rows requiring packed cold object attention: `{packed_rows}`",
        "",
        "## Recommended Experiment Order",
        "",
    ]
    for row in first_rows:
        lines.append(
            "- "
            f"#{row.experiment_order} prefix=`{row.prefix_tokens}` "
            f"class=`{row.priority_class}` "
            f"estimated_restore_ms=`{_number_string(row.estimated_restore_ms)}` "
            f"required_lead_ms=`{_number_string(row.required_lead_ms)}` "
            f"packed=`{_bool_string(row.packed_object_required)}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PrePass is effective only when the restore work finishes before the online request arrives.",
            "- `residual_online_wait_ms > 0` means cold restore would still leak into user-visible TTFT.",
            "- `packed_object_required=true` marks the point where per-layer files are likely to dominate restore tail; this is the trigger to validate a packed cold object / extent layout.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_lead_rows(rows: list[LeadTimePlanRow]) -> list[LeadTimePlanRow]:
    selected: dict[int, LeadTimePlanRow] = {}
    for row in rows:
        selected.setdefault(row.prefix_tokens, row)
    return [selected[prefix] for prefix in sorted(selected, key=lambda p: selected[p].experiment_order)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _required_int(row: dict[str, str], key: str) -> int:
    value = _int_or_none(row.get(key))
    if value is None:
        raise ValueError(f"missing required int column: {key}")
    return value


def _int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _float_or_zero(value: str | None) -> float:
    return _float_or_none(value) or 0.0


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def _number_string(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded) == 0:
        rounded = 0.0
    text = f"{rounded:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text
