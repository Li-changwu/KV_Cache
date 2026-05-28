import csv
import subprocess
import sys

from benchmarks.m3.run_prepass_restore_smoke import run_prepass_restore_smoke


def test_prepass_restore_smoke_writes_ready_before_request_artifacts(tmp_path):
    result = run_prepass_restore_smoke(
        result_dir=tmp_path / "result",
        prefix_tokens=64,
        suffix_tokens=16,
        async_prefetch=False,
    )

    assert result["status"] == "OK"
    assert result["ready_before_request"] is True
    assert result["online_decision"] == "ADMIT"
    rows = list(csv.DictReader((tmp_path / "result" / "prepass_restore_smoke.csv").open()))
    assert rows[0]["prepass_status"] == "READY"
    assert rows[0]["ready_before_request"] == "true"
    assert rows[0]["online_decision"] == "ADMIT"
    assert rows[0]["sync_cold_miss_total"] == "0"
    assert (tmp_path / "result" / "prepass_restore_report.md").exists()


def test_prepass_restore_smoke_help_runs():
    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_prepass_restore_smoke.py", "--help"],
        cwd="/root/KV",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--prefix-tokens" in completed.stdout
