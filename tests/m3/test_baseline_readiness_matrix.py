import csv
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.run_reuse_smoke_matrix import summarize_connector_events
from benchmarks.m3.run_baseline_readiness_matrix import (
    BASELINE_MATRIX_COLUMNS,
    BaselineReadinessConfig,
    CompletionTiming,
    exit_code_for_status,
    build_matrix_rows,
    run_baseline_readiness_matrix,
)
from benchmarks.m3.summarize_baseline_readiness_matrix import (
    summarize_baseline_readiness_matrix,
)


def test_build_matrix_rows_expands_baselines_and_shapes():
    rows = build_matrix_rows(
        BaselineReadinessConfig(
            result_dir=Path("results/m3_11_baseline_readiness"),
            baselines=["B0", "B5"],
            workloads=["session_append"],
            prefix_tokens=[512, 2048],
            suffix_tokens=[128],
            output_tokens=[1, 16],
            repeats=2,
            dry_run=True,
        )
    )

    assert len(rows) == 16
    assert rows[0].run_id == "m3_11_B0_session_append_p512_s128_o1_c1_r0"
    assert rows[-1].run_id == "m3_11_B5_session_append_p2048_s128_o16_c1_r1"
    assert {row.baseline_id for row in rows} == {"B0", "B5"}
    assert {row.prefix_tokens for row in rows} == {512, 2048}
    assert {row.output_tokens for row in rows} == {1, 16}


def test_dry_run_writes_required_outputs_and_schema(tmp_path):
    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B0", "B3", "B5"],
            workloads=["session_append", "low_locality"],
            prefix_tokens=[512],
            suffix_tokens=[128],
            output_tokens=[1],
            repeats=1,
            dry_run=True,
        )
    )

    assert result["status"] == "DRY_RUN"
    assert result["rows"] == 6
    for relative in [
        "env.json",
        "run_config.yaml",
        "baseline_matrix.csv",
        "request_metrics.csv",
        "sidecar_decisions.jsonl",
        "connector_events.jsonl",
        "restore_events.jsonl",
        "summary/baseline_summary.csv",
        "summary/workload_summary.csv",
        "summary/readiness_report.md",
    ]:
        assert (tmp_path / relative).exists()

    with (tmp_path / "baseline_matrix.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == BASELINE_MATRIX_COLUMNS
        rows = list(reader)

    assert len(rows) == 6
    assert {row["status"] for row in rows} == {"DRY_RUN"}
    assert rows[0]["kv_bytes_per_token"] == "196608"
    assert rows[0]["model_context_limit"] == "32768"
    assert rows[0]["backend"] == "simulated"
    assert "historical_byte_hit_rate" in rows[0]
    assert "sync_cold_miss_total" in rows[0]
    assert "ttft_vs_full_prefill" in rows[0]
    assert (tmp_path / "request_metrics.csv").read_text(encoding="utf-8").startswith(
        "run_id,timestamp_utc"
    )


def test_online_native_vllm_b0_b1_b2_uses_warm_and_cold_paths(tmp_path):
    client = FakeCompletionClient()

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B0", "B1", "B2"],
            workloads=["session_append"],
            prefix_tokens=[512],
            suffix_tokens=[128],
            output_tokens=[1],
            repeats=1,
            dry_run=False,
        ),
        completion_client=client,
    )

    assert result["status"] == "OK"
    rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    assert [row["baseline_id"] for row in rows] == ["B0", "B1", "B2"]
    assert {row["status"] for row in rows} == {"OK"}
    assert {row["backend"] for row in rows} == {"vllm_http"}
    assert [call["purpose"] for call in client.calls] == [
        "measure_B0",
        "warm_B1",
        "measure_B1",
        "measure_B2",
    ]
    assert len(client.calls[0]["prompt"].split()) == 640
    assert client.calls[1]["prompt"].split()[:512] == client.calls[2]["prompt"].split()[:512]
    assert client.calls[0]["prompt"].split()[:512] != client.calls[3]["prompt"].split()[:512]
    assert all("-" not in token for token in client.calls[0]["prompt"].split()[:512])
    assert all("-" not in token for token in client.calls[3]["prompt"].split()[:512])

    b0, b1, b2 = rows
    assert b0["reuse_tokens"] == "0"
    assert b0["historical_kv_hit_tokens"] == "0"
    assert b0["required_historical_tokens"] == "512"
    assert b1["reuse_tokens"] == "512"
    assert b1["historical_kv_hit_rate"] == "1.0"
    assert b1["prefill_tokens_saved"] == "512"
    assert b2["reuse_tokens"] == "0"
    assert b2["historical_kv_hit_rate"] == "0.0"
    assert b2["admission_reason"] == "apc_cold_cache_or_restart"


def test_strict_b2_restart_orchestrates_same_prefix_across_service_restart(tmp_path):
    events = []
    client = FakeCompletionClient(events=events)
    service_manager = FakeServiceManager(events=events)

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B2"],
            workloads=["session_append"],
            prefix_tokens=[512],
            suffix_tokens=[128],
            output_tokens=[1],
            repeats=1,
            dry_run=False,
            strict_b2_restart=True,
        ),
        completion_client=client,
        service_manager=service_manager,
    )

    assert result["status"] == "OK"
    assert events == [
        ("service", "start", "warm_B2_before_restart"),
        ("complete", "warm_B2_before_restart"),
        ("service", "stop", "warm_B2_before_restart"),
        ("service", "start", "measure_B2_after_restart"),
        ("complete", "measure_B2_after_restart"),
        ("service", "stop", "measure_B2_after_restart"),
    ]
    warm_prompt, measure_prompt = [call["prompt"] for call in client.calls]
    assert warm_prompt.split()[:512] == measure_prompt.split()[:512]
    assert len(warm_prompt.split()) == 512
    assert len(measure_prompt.split()) == 640

    rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    assert rows[0]["status"] == "OK"
    assert rows[0]["baseline_id"] == "B2"
    assert rows[0]["admission_reason"] == "apc_restart_cold_cache"
    assert rows[0]["reuse_tokens"] == "0"
    assert rows[0]["historical_kv_hit_rate"] == "0.0"


def test_strict_b2_boundary_does_not_start_service(tmp_path):
    events = []
    client = FakeCompletionClient(events=events)
    service_manager = FakeServiceManager(events=events)

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B2"],
            workloads=["session_append"],
            prefix_tokens=[32768],
            suffix_tokens=[2048],
            output_tokens=[16],
            repeats=1,
            dry_run=False,
            strict_b2_restart=True,
        ),
        completion_client=client,
        service_manager=service_manager,
    )

    assert result["status"] == "PARTIAL"
    assert events == []
    assert client.calls == []
    rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    assert rows[0]["status"] == "ERROR_BOUNDARY"
    assert rows[0]["admission_reason"] == "context_boundary"


def test_online_b3_b5_bridge_maps_reuse_executor_rows_to_baseline_schema(tmp_path):
    executor = FakeReuseExecutor(
        {
            "B3": {
                "status": "OK",
                "second_ttft_ms": "210.5",
                "reuse_tokens": "512",
                "decision": "ADMIT",
                "reason": "required_kv_ready_before_decode",
                "ready_barrier_all_ready": "true",
                "sync_ssd_miss_total": "0",
                "store_elapsed_ms": "12.5",
                "load_elapsed_ms": "3.25",
                "store_events": "96",
                "load_events": "1",
                "external_load_observed": "yes",
                "online_service_ttft_ms": "210.5",
                "restore_inclusive_ttft_ms": "210.5",
                "restore_wait_ms": "0.0",
                "connector_store_elapsed_ms": "12.5",
                "connector_load_elapsed_ms": "3.25",
                "cold_tier_restore": "no",
            },
            "B5": {
                "status": "OK",
                "second_ttft_ms": "260.0",
                "reuse_tokens": "512",
                "decision": "ADMIT",
                "reason": "required_kv_ready_before_decode",
                "ready_barrier_all_ready": "true",
                "sync_ssd_miss_total": "0",
                "store_elapsed_ms": "14.0",
                "load_elapsed_ms": "4.0",
                "store_events": "96",
                "load_events": "1",
                "external_load_observed": "yes",
                "online_service_ttft_ms": "260.0",
                "restore_inclusive_ttft_ms": "301.5",
                "restore_wait_ms": "41.5",
                "connector_store_elapsed_ms": "14.0",
                "connector_load_elapsed_ms": "4.0",
                "cold_tier_restore": "yes",
                "cold_backend": "local_posix",
                "cold_probe_decision": "DELAY",
                "cold_restore_status": "COMPLETED",
                "cold_restore_actual_bytes": str(512 * 196608),
                "cold_restore_checksum_status": "ok",
                "cold_restore_executor_elapsed_ms": "41.5",
                "prefetch_deadline_miss_delta": "0",
            },
        }
    )

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B3", "B5"],
            workloads=["session_append"],
            prefix_tokens=[512],
            suffix_tokens=[128],
            output_tokens=[1],
            repeats=1,
            dry_run=False,
        ),
        reuse_executor=executor,
    )

    assert result["status"] == "OK"
    assert executor.calls == [
        {
            "baseline_id": "B3",
            "prefix_tokens": 512,
            "suffix_tokens": 128,
            "output_tokens": 1,
            "cold_tier_restore": False,
        },
        {
            "baseline_id": "B5",
            "prefix_tokens": 512,
            "suffix_tokens": 128,
            "output_tokens": 1,
            "cold_tier_restore": True,
        },
    ]
    rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    b3, b5 = rows
    assert b3["baseline_id"] == "B3"
    assert b3["backend"] == "vllm_sidecar_connector"
    assert b3["storage_backend"] == "dram"
    assert b3["ttft_ms"] == "210.5"
    assert b3["historical_byte_hit_rate"] == "1.0"
    assert b3["external_load_observed"] == "true"
    assert b3["connector_load_events"] == "1"
    assert b3["online_service_ttft_ms"] == "210.5"
    assert b3["restore_inclusive_ttft_ms"] == "210.5"
    assert b3["restore_wait_ms"] == "0.0"
    assert b3["connector_store_elapsed_ms"] == "12.5"
    assert b3["connector_load_elapsed_ms"] == "3.25"
    assert b3["unattributed_ttft_ms"] == "207.25"
    assert b3["restore_status"] == "NA"
    assert b3["admission_reason"] == "required_kv_ready_before_decode"

    assert b5["baseline_id"] == "B5"
    assert b5["backend"] == "vllm_sidecar_connector"
    assert b5["storage_backend"] == "local_posix"
    assert b5["cold_probe_decision"] == "DELAY"
    assert b5["restore_status"] == "COMPLETED"
    assert b5["restore_actual_bytes"] == str(512 * 196608)
    assert b5["restore_useful_bytes"] == str(512 * 196608)
    assert b5["restore_bytes_per_useful_byte"] == "1.0"
    assert b5["restore_elapsed_ms"] == "41.5"
    assert b5["restore_checksum_status"] == "ok"
    assert b5["online_service_ttft_ms"] == "260.0"
    assert b5["restore_inclusive_ttft_ms"] == "301.5"
    assert b5["restore_wait_ms"] == "41.5"
    assert b5["connector_store_elapsed_ms"] == "14.0"
    assert b5["connector_load_elapsed_ms"] == "4.0"
    assert b5["unattributed_ttft_ms"] == "256.0"
    assert b5["prefill_reuse_ready"] == "true"
    assert b5["decode_execution_ready"] == "true"
    assert b5["sync_cold_miss_total"] == "0"


def test_online_b3_b5_boundary_does_not_call_reuse_executor(tmp_path):
    executor = FakeReuseExecutor({})

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B5"],
            workloads=["session_append"],
            prefix_tokens=[32768],
            suffix_tokens=[2048],
            output_tokens=[16],
            repeats=1,
            dry_run=False,
        ),
        reuse_executor=executor,
    )

    assert result["status"] == "PARTIAL"
    assert executor.calls == []
    rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    assert rows[0]["status"] == "ERROR_BOUNDARY"
    assert rows[0]["backend"] == "vllm_sidecar_connector"


def test_reuse_csv_import_executor_maps_existing_reuse_matrix(tmp_path):
    from benchmarks.m3.run_baseline_readiness_matrix import ReuseCsvImportExecutor

    source_csv = tmp_path / "reuse_smoke_matrix.csv"
    source_csv.write_text(
        (
            "run_id,prefix_tokens,suffix_tokens,output_tokens,status,second_ttft_ms,"
            "reuse_tokens,decision,reason,ready_barrier_all_ready,sync_ssd_miss_total,"
            "load_events,store_events,external_load_observed,cold_tier_restore,"
            "cold_restore_status,cold_restore_actual_bytes,cold_restore_checksum_status,"
            "cold_restore_executor_elapsed_ms,prefetch_deadline_miss_delta\n"
            "reuse_prefix_512,512,128,1,OK,301.25,512,ADMIT,"
            "required_kv_ready_before_decode,true,0,1,96,yes,yes,"
            "COMPLETED,100663296,ok,55.5,0\n"
        ),
        encoding="utf-8",
    )

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path / "result",
            baselines=["B5"],
            workloads=["session_append"],
            prefix_tokens=[512],
            suffix_tokens=[128],
            output_tokens=[1],
            repeats=1,
            dry_run=False,
        ),
        reuse_executor=ReuseCsvImportExecutor({("B5", 512, 128, 1): source_csv}),
    )

    assert result["status"] == "OK"
    rows = list(csv.DictReader((tmp_path / "result" / "baseline_matrix.csv").open()))
    assert rows[0]["baseline_id"] == "B5"
    assert rows[0]["ttft_ms"] == "301.25"
    assert rows[0]["restore_status"] == "COMPLETED"
    assert rows[0]["external_load_observed"] == "true"


def test_summarize_connector_events_can_scope_to_new_log_lines(tmp_path):
    log_path = tmp_path / "connector_events.jsonl"
    events = [
        {
            "event": "store_layer",
            "status": "ok",
            "prefix_id": "m3-8-2048",
            "elapsed_ms": 100.0,
        },
        {
            "event": "load_request",
            "status": "ok",
            "prefix_id": "m3-8-2048",
            "elapsed_ms": 7.5,
        },
        {
            "event": "store_layer",
            "status": "ok",
            "prefix_id": "m3-8-2048",
            "elapsed_ms": 9.0,
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    summary = summarize_connector_events(log_path, prefix_id="m3-8-2048", start_line=1)

    assert summary["store_elapsed_ms"] == 9.0
    assert summary["load_elapsed_ms"] == 7.5
    assert summary["store_events"] == 1
    assert summary["load_events"] == 1


def test_online_native_vllm_boundary_row_does_not_call_client(tmp_path):
    client = FakeCompletionClient()

    result = run_baseline_readiness_matrix(
        BaselineReadinessConfig(
            result_dir=tmp_path,
            baselines=["B0"],
            workloads=["session_append"],
            prefix_tokens=[32768],
            suffix_tokens=[2048],
            output_tokens=[16],
            repeats=1,
            dry_run=False,
        ),
        completion_client=client,
    )

    assert result["status"] == "PARTIAL"
    assert client.calls == []
    rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    assert rows[0]["status"] == "ERROR_BOUNDARY"
    assert rows[0]["error_type"] == "context_limit"
    assert "exceeds model_context_limit" in rows[0]["error_message"]


def test_summarizer_computes_ratios_and_readiness_report(tmp_path):
    config = BaselineReadinessConfig(
        result_dir=tmp_path,
        baselines=["B0", "B3", "B5"],
        workloads=["session_append"],
        prefix_tokens=[8192],
        suffix_tokens=[128],
        output_tokens=[1],
        repeats=1,
        dry_run=True,
    )
    rows = [
        _row(config, "B0", ttft_ms="1000", reuse_tokens="0"),
        _row(
            config,
            "B3",
            ttft_ms="200",
            reuse_tokens="8192",
            external_load_observed="true",
            connector_load_events="1",
        ),
        _row(
            config,
            "B5",
            ttft_ms="240",
            reuse_tokens="8192",
            historical_kv_hit_tokens="8192",
            historical_kv_hit_rate="1.0",
            historical_bytes_required=str(8192 * 196608),
            historical_bytes_hit=str(8192 * 196608),
            historical_byte_hit_rate="1.0",
            restore_status="COMPLETED",
            restore_actual_bytes=str(8192 * 196608),
            restore_useful_bytes=str(8192 * 196608),
            restore_bytes_per_useful_byte="1.0",
            sync_cold_miss_total="0",
            external_load_observed="true",
            connector_load_events="1",
            restore_wait_ms="40",
            online_service_ttft_ms="240",
            restore_inclusive_ttft_ms="280",
            connector_store_elapsed_ms="20",
            connector_load_elapsed_ms="5",
            connector_total_elapsed_ms="25",
            admission_decision="ADMIT",
        ),
    ]
    _write_baseline_matrix(tmp_path / "baseline_matrix.csv", rows)

    summary = summarize_baseline_readiness_matrix(tmp_path)

    assert summary["status"] == "OK"
    b5_rows = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    b5 = next(row for row in b5_rows if row["baseline_id"] == "B5")
    assert b5["ttft_vs_full_prefill"] == "0.24"
    assert b5["ttft_vs_dram_ready_reuse"] == "1.2"
    summary_rows = list(
        csv.DictReader((tmp_path / "summary" / "baseline_summary.csv").open())
    )
    b5_summary = next(row for row in summary_rows if row["baseline_id"] == "B5")
    assert b5_summary["ttft_p50_ms"] == "240.0"
    assert b5_summary["restore_inclusive_ttft_p50_ms"] == "280.0"
    assert b5_summary["connector_total_elapsed_p50_ms"] == "25.0"
    assert b5_summary["unattributed_ttft_p50_ms"] == "235.0"
    assert b5_summary["external_load_observed_rate"] == "1.0"
    report = (tmp_path / "summary" / "readiness_report.md").read_text(
        encoding="utf-8"
    )
    assert "M3.11 Baseline Readiness Report" in report
    assert "B5 vs B0" in report
    assert "B5 vs B0 `0.24`" in report
    assert "`8192`" in report


def test_summarizer_preserves_boundary_rows(tmp_path):
    config = BaselineReadinessConfig(
        result_dir=tmp_path,
        baselines=["B0", "B5"],
        workloads=["session_append"],
        prefix_tokens=[32768],
        suffix_tokens=[2048],
        output_tokens=[16],
        repeats=1,
        dry_run=True,
    )
    rows = [
        _row(config, "B0", status="ERROR_BOUNDARY", error_type="context_limit"),
        _row(config, "B5", status="ERROR_BOUNDARY", error_type="context_limit"),
    ]
    _write_baseline_matrix(tmp_path / "baseline_matrix.csv", rows)

    summary = summarize_baseline_readiness_matrix(tmp_path)

    assert summary["status"] == "PARTIAL"
    reloaded = list(csv.DictReader((tmp_path / "baseline_matrix.csv").open()))
    assert len(reloaded) == 2
    assert {row["status"] for row in reloaded} == {"ERROR_BOUNDARY"}
    summary_rows = list(
        csv.DictReader((tmp_path / "summary" / "baseline_summary.csv").open())
    )
    assert summary_rows[0]["error_count"] == "1"


def test_baseline_readiness_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_baseline_readiness_matrix.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--phase" in completed.stdout
    assert "--dry-run" in completed.stdout
    assert "--base-url" in completed.stdout


def test_runner_exit_code_treats_partial_as_completed_artifact():
    assert exit_code_for_status("DRY_RUN") == 0
    assert exit_code_for_status("OK") == 0
    assert exit_code_for_status("PARTIAL") == 0
    assert exit_code_for_status("ERROR") == 1


def test_baseline_summary_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/summarize_baseline_readiness_matrix.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--result-dir" in completed.stdout


def _row(
    config: BaselineReadinessConfig,
    baseline_id: str,
    *,
    status: str = "OK",
    ttft_ms: str = "",
    reuse_tokens: str = "0",
    error_type: str = "",
    historical_kv_hit_tokens: str = "0",
    historical_kv_hit_rate: str = "0.0",
    historical_bytes_required: str = "0",
    historical_bytes_hit: str = "0",
    historical_byte_hit_rate: str = "0.0",
    restore_status: str = "NA",
    restore_actual_bytes: str = "0",
    restore_useful_bytes: str = "0",
    restore_bytes_per_useful_byte: str = "",
    sync_cold_miss_total: str = "0",
    external_load_observed: str = "false",
    connector_load_events: str = "0",
    restore_wait_ms: str = "",
    online_service_ttft_ms: str = "",
    restore_inclusive_ttft_ms: str = "",
    connector_store_elapsed_ms: str = "",
    connector_load_elapsed_ms: str = "",
    connector_total_elapsed_ms: str = "",
    unattributed_ttft_ms: str = "",
    admission_decision: str = "ADMIT",
) -> dict[str, str]:
    row = {column: "" for column in BASELINE_MATRIX_COLUMNS}
    row.update(
        {
            "run_id": f"m3_11_{baseline_id}_session_append_p8192_s128_o1_c1_r0",
            "timestamp_utc": "2026-05-27T00:00:00Z",
            "model_id": config.model,
            "model_context_limit": str(config.model_context_limit),
            "kv_bytes_per_token": str(config.kv_bytes_per_token),
            "backend": "simulated",
            "baseline_id": baseline_id,
            "workload": "session_append",
            "prefix_tokens": "8192",
            "actual_prefix_tokens": "8192",
            "suffix_tokens": "128",
            "actual_suffix_tokens": "128",
            "output_tokens": "1",
            "concurrency": "1",
            "repeat_id": "0",
            "status": status,
            "error_type": error_type,
            "ttft_ms": ttft_ms,
            "reuse_tokens": reuse_tokens,
            "delta_prefill_tokens": "128" if baseline_id in {"B3", "B5"} else "8320",
            "prefill_tokens_saved": reuse_tokens,
            "required_historical_tokens": "8192" if baseline_id in {"B3", "B5"} else "0",
            "historical_kv_hit_tokens": historical_kv_hit_tokens,
            "historical_kv_hit_rate": historical_kv_hit_rate,
            "historical_bytes_required": historical_bytes_required,
            "historical_bytes_hit": historical_bytes_hit,
            "historical_byte_hit_rate": historical_byte_hit_rate,
            "restore_status": restore_status,
            "restore_actual_bytes": restore_actual_bytes,
            "restore_useful_bytes": restore_useful_bytes,
            "restore_bytes_per_useful_byte": restore_bytes_per_useful_byte,
            "storage_backend": "local_posix" if baseline_id == "B5" else "dram",
            "prefetch_deadline_miss": "false",
            "sync_cold_miss_total": sync_cold_miss_total,
            "admission_decision": admission_decision,
            "prefill_reuse_ready": "true",
            "decode_execution_ready": "true",
            "external_load_observed": external_load_observed,
            "connector_load_events": connector_load_events,
            "connector_store_events": "1" if baseline_id in {"B3", "B5"} else "0",
            "restore_wait_ms": restore_wait_ms,
            "online_service_ttft_ms": online_service_ttft_ms,
            "restore_inclusive_ttft_ms": restore_inclusive_ttft_ms,
            "connector_store_elapsed_ms": connector_store_elapsed_ms,
            "connector_load_elapsed_ms": connector_load_elapsed_ms,
            "connector_total_elapsed_ms": connector_total_elapsed_ms,
            "unattributed_ttft_ms": unattributed_ttft_ms,
        }
    )
    return row


def _write_baseline_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BASELINE_MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class FakeCompletionClient:
    def __init__(self, events=None):
        self.calls = []
        self.events = events

    def complete(self, *, prompt: str, output_tokens: int, run_id: str, purpose: str):
        if self.events is not None:
            self.events.append(("complete", purpose))
        self.calls.append(
            {
                "prompt": prompt,
                "output_tokens": output_tokens,
                "run_id": run_id,
                "purpose": purpose,
            }
        )
        index = len(self.calls)
        return CompletionTiming(
            status="OK",
            ttft_ms=100.0 + index,
            e2e_ms=120.0 + index,
            itl_p50_ms=1.0,
            itl_p95_ms=2.0,
            tpot_p50_ms=3.0,
            tpot_p95_ms=4.0,
            raw_path=f"raw/{run_id}_{purpose}.json",
        )


class FakeServiceManager:
    def __init__(self, events):
        self.events = events

    def start(self, *, run_id: str, purpose: str):
        self.events.append(("service", "start", purpose))

    def stop(self, *, run_id: str, purpose: str):
        self.events.append(("service", "stop", purpose))


class FakeReuseExecutor:
    def __init__(self, rows_by_baseline):
        self.rows_by_baseline = rows_by_baseline
        self.calls = []

    def run(self, *, config, row):
        source = self.rows_by_baseline[row.baseline_id]
        self.calls.append(
            {
                "baseline_id": row.baseline_id,
                "prefix_tokens": row.prefix_tokens,
                "suffix_tokens": row.suffix_tokens,
                "output_tokens": row.output_tokens,
                "cold_tier_restore": row.baseline_id == "B5",
            }
        )
        return {
            "run_id": f"reuse_prefix_{row.prefix_tokens}",
            "prefix_tokens": str(row.prefix_tokens),
            "suffix_tokens": str(row.suffix_tokens),
            "output_tokens": str(row.output_tokens),
            **source,
        }
