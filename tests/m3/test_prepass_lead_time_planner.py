import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.m3.prepass_lead_time_planner import (
    build_lead_time_plan,
    load_restore_profile,
)


def test_restore_profile_scales_measured_2k_restore_to_8k(tmp_path):
    prepass_csv, baseline_csv = _write_sample_inputs(tmp_path)
    profile = load_restore_profile(prepass_csv, baseline_csv)

    assert profile.source_prefix_tokens == 2048
    assert profile.kv_bytes_per_token == 200
    assert profile.restore_bytes_per_ms == pytest.approx(512.0)

    rows = build_lead_time_plan(
        profile,
        prefix_tokens=[8192],
        available_leads_ms=[5000.0],
        suffix_tokens=128,
        output_tokens=1,
        tail_multiplier=1.0,
        safety_margin_ms=0.0,
    )

    row = rows[0]
    assert row.estimated_kv_bytes == 8192 * 200
    assert row.estimated_restore_ms == pytest.approx(3200.0)
    assert row.required_lead_ms == pytest.approx(3210.0)
    assert row.hide_restore is True
    assert row.residual_online_wait_ms == 0.0


def test_insufficient_lead_time_is_marked_as_user_visible_risk(tmp_path):
    prepass_csv, baseline_csv = _write_sample_inputs(tmp_path)
    profile = load_restore_profile(prepass_csv, baseline_csv)

    rows = build_lead_time_plan(
        profile,
        prefix_tokens=[8192],
        available_leads_ms=[1000.0],
        tail_multiplier=1.0,
        safety_margin_ms=0.0,
    )

    row = rows[0]
    assert row.hide_restore is False
    assert row.residual_online_wait_ms == pytest.approx(2210.0)
    assert row.risk_class == "LEAD_TIME_RISK"


def test_experiment_order_prioritizes_calibration_then_scaling_then_layout_gate(tmp_path):
    prepass_csv, baseline_csv = _write_sample_inputs(tmp_path)
    profile = load_restore_profile(prepass_csv, baseline_csv)

    rows = build_lead_time_plan(
        profile,
        prefix_tokens=[32768, 8192, 512, 16384, 2048],
        available_leads_ms=[0.0],
    )

    assert [row.prefix_tokens for row in rows] == [512, 2048, 8192, 16384, 32768]
    assert [row.experiment_order for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0].priority_class == "downscale_calibration"
    assert rows[1].priority_class == "source_repeat"
    assert rows[2].priority_class == "first_scaling_probe"
    assert rows[2].packed_object_required is True
    assert rows[-1].priority_class == "context_limit_boundary"


def test_cli_writes_plan_csv_report_and_profile(tmp_path):
    prepass_csv, baseline_csv = _write_sample_inputs(tmp_path)
    result_dir = tmp_path / "plan"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/plan_prepass_lead_time_matrix.py",
            "--prepass-csv",
            str(prepass_csv),
            "--baseline-csv",
            str(baseline_csv),
            "--result-dir",
            str(result_dir),
            "--prefix-tokens",
            "512",
            "2048",
            "8192",
            "--available-leads-ms",
            "0",
            "5000",
            "--tail-multiplier",
            "1.0",
            "--safety-margin-ms",
            "0",
        ],
        cwd="/root/KV",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "OK"
    assert summary["rows"] == 6

    csv_path = result_dir / "prepass_lead_time_plan.csv"
    report_path = result_dir / "prepass_lead_time_report.md"
    profile_path = result_dir / "restore_profile.json"
    assert csv_path.exists()
    assert report_path.exists()
    assert profile_path.exists()

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert rows[0]["prefix_tokens"] == "512"
    assert rows[0]["available_lead_ms"] == "0.0"
    assert rows[-1]["prefix_tokens"] == "8192"
    assert rows[-1]["available_lead_ms"] == "5000.0"
    assert rows[-1]["packed_object_required"] == "true"
    report = report_path.read_text(encoding="utf-8")
    assert "M3.13 PrePass Lead-Time Plan" in report
    assert "packed cold object" in report


def _write_sample_inputs(tmp_path: Path) -> tuple[Path, Path]:
    prepass_csv = tmp_path / "reuse_smoke_matrix.csv"
    _write_csv(
        prepass_csv,
        [
            {
                "run_id": "reuse_prefix_2048",
                "prefix_tokens": "2048",
                "suffix_tokens": "128",
                "output_tokens": "1",
                "phase": "reuse",
                "status": "OK",
                "second_ttft_ms": "270.0",
                "prepass_elapsed_ms": "10.0",
                "prepass_restore_actual_bytes": "409600",
                "prepass_restore_executor_elapsed_ms": "0.0",
                "cold_restore_actual_bytes": "409600",
                "cold_restore_executor_elapsed_ms": "800.0",
                "connector_load_elapsed_ms": "100.0",
            }
        ],
    )
    baseline_csv = tmp_path / "baseline_matrix.csv"
    _write_csv(
        baseline_csv,
        [
            {
                "baseline_id": "B0",
                "prefix_tokens": "2048",
                "status": "OK",
                "ttft_ms": "700.0",
                "kv_bytes_per_token": "200",
                "historical_bytes_required": "409600",
            },
            {
                "baseline_id": "B3",
                "prefix_tokens": "2048",
                "status": "OK",
                "ttft_ms": "280.0",
                "kv_bytes_per_token": "200",
                "connector_load_elapsed_ms": "95.0",
            },
            {
                "baseline_id": "B5",
                "prefix_tokens": "2048",
                "status": "OK",
                "ttft_ms": "330.0",
                "restore_elapsed_ms": "900.0",
                "kv_bytes_per_token": "200",
            },
        ],
    )
    return prepass_csv, baseline_csv


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
