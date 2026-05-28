import subprocess
import sys

from benchmarks.m3.run_reuse_consistency_smoke import (
    assess_consistency,
    build_consistency_payloads,
    connector_event_line_count,
    extract_completion_text,
    load_baseline_record,
    save_baseline_record,
    summarize_connector_event_delta,
)


def test_extract_completion_text_from_openai_completion_response():
    response = {"choices": [{"text": " same answer"}]}

    assert extract_completion_text(response) == " same answer"


def test_assess_consistency_accepts_matching_non_empty_outputs():
    result = assess_consistency(
        baseline_status_code=200,
        reuse_status_code=200,
        baseline_payload={"choices": [{"text": "stable"}]},
        reuse_payload={"choices": [{"text": "stable"}]},
        external_load_observed=True,
    )

    assert result["status"] == "OK"
    assert result["texts_match"] is True
    assert result["external_load_observed"] is True


def test_assess_consistency_rejects_empty_or_mismatched_outputs():
    empty = assess_consistency(
        baseline_status_code=200,
        reuse_status_code=200,
        baseline_payload={"choices": [{"text": ""}]},
        reuse_payload={"choices": [{"text": ""}]},
        external_load_observed=True,
    )
    mismatch = assess_consistency(
        baseline_status_code=200,
        reuse_status_code=200,
        baseline_payload={"choices": [{"text": "alpha"}]},
        reuse_payload={"choices": [{"text": "beta"}]},
        external_load_observed=True,
    )

    assert empty["status"] == "ERROR"
    assert empty["reason"] == "empty_output"
    assert mismatch["status"] == "ERROR"
    assert mismatch["reason"] == "text_mismatch"


def test_build_consistency_payloads_uses_same_logical_prompt():
    baseline, reuse = build_consistency_payloads(
        model="/model",
        prefix_tokens=64,
        suffix_tokens=16,
        output_tokens=4,
        prompt_unit="cache",
        request_id="consistency-1",
    )

    assert baseline["prompt"] == reuse["prompt"]
    assert len(baseline["prompt"].split()) == 80
    assert reuse["m3_control"]["prefix_candidates"][0]["token_end"] == 64
    assert reuse["m3_control"]["token_count"] == 80


def test_summarize_connector_event_delta_ignores_old_load_events(tmp_path):
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(
        '{"event":"load_request","prefix_id":"m3-8-64","status":"ok","elapsed_ms":10}\n',
        encoding="utf-8",
    )
    start_line = connector_event_line_count(event_log)
    with event_log.open("a", encoding="utf-8") as fh:
        fh.write(
            '{"event":"store_layer","prefix_id":"m3-8-64","status":"ok","elapsed_ms":2}\n'
        )
        fh.write(
            '{"event":"load_request","prefix_id":"m3-8-64","status":"ok","elapsed_ms":3}\n'
        )

    summary = summarize_connector_event_delta(
        event_log,
        prefix_id="m3-8-64",
        start_line=start_line,
    )

    assert summary["load_events"] == 1
    assert summary["load_elapsed_ms"] == 3.0
    assert summary["external_load_observed"] == "yes"


def test_save_and_load_baseline_record(tmp_path):
    path = save_baseline_record(
        tmp_path,
        status_code=200,
        elapsed_ms=12.5,
        payload={"choices": [{"text": "stable"}]},
    )

    record = load_baseline_record(path)

    assert record["status_code"] == 200
    assert record["elapsed_ms"] == 12.5
    assert record["payload"]["choices"][0]["text"] == "stable"


def test_cli_help_runs_when_script_is_executed_by_path():
    result = subprocess.run(
        [sys.executable, "benchmarks/m3/run_reuse_consistency_smoke.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--phase" in result.stdout
