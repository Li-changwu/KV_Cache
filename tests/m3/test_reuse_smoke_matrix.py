from pathlib import Path
import subprocess
import sys

import torch
from fastapi.testclient import TestClient

from benchmarks.m3.http_sidecar import M3SidecarConfig, create_app
from benchmarks.m3.run_reuse_smoke_matrix import (
    MatrixConfig,
    build_store_and_reuse_payloads,
    clone_payload_with_request_id,
    demote_prefix_to_cold_object,
    matrix_rows,
    parse_sidecar_metrics,
    mark_sidecar_prefix_cold,
    run_matrix,
    summarize_prefetch_advance,
    summarize_prepass_response,
    sidecar_decision_log_line_count,
    summarize_sidecar_decision_delta,
    summarize_connector_events,
    summarize_sidecar_metric_deltas,
)
from benchmarks.m3.tensor_store import KVTensorStore, block_slot_mapping


def test_matrix_rows_expand_default_prefix_lengths():
    rows = matrix_rows(
        MatrixConfig(
            prefix_tokens=[16, 64, 128, 256],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=Path("results/m3_8_reuse_matrix"),
        )
    )

    assert [row.prefix_tokens for row in rows] == [16, 64, 128, 256]
    assert [row.run_id for row in rows] == [
        "reuse_prefix_16",
        "reuse_prefix_64",
        "reuse_prefix_128",
        "reuse_prefix_256",
    ]
    assert all(row.output_tokens == 1 for row in rows)
    assert all(row.suffix_tokens == 16 for row in rows)


def test_run_matrix_dry_run_writes_csv_and_report(tmp_path):
    result = run_matrix(
        MatrixConfig(
            prefix_tokens=[16, 64],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path,
            dry_run=True,
        )
    )

    assert result["status"] == "DRY_RUN"
    csv_path = tmp_path / "reuse_smoke_matrix.csv"
    report_path = tmp_path / "report.md"
    assert csv_path.exists()
    assert report_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "run_id,prefix_tokens,suffix_tokens,output_tokens,phase,status" in text
    assert "residency_hit_delta" in text.splitlines()[0]
    assert "prefetch_queued_delta" in text.splitlines()[0]
    assert "prefetch_queue_pending" in text.splitlines()[0]
    assert "cold_tier_restore" in text.splitlines()[0]
    assert "cold_restore_status" in text.splitlines()[0]
    assert "cold_restore_actual_bytes" in text.splitlines()[0]
    assert "connector_store_elapsed_ms" in text.splitlines()[0]
    assert "connector_load_elapsed_ms" in text.splitlines()[0]
    assert "connector_total_elapsed_ms" in text.splitlines()[0]
    assert "prepass_status" in text.splitlines()[0]
    assert "ready_before_request" in text.splitlines()[0]
    assert "prepass_restore_actual_bytes" in text.splitlines()[0]
    assert "prepass_restore_checksum_status" in text.splitlines()[0]
    assert "prepass_restore_executor_elapsed_ms" in text.splitlines()[0]
    assert "reason" in text.splitlines()[0]
    assert "ready_barrier_all_ready" in text.splitlines()[0]
    assert "reuse_prefix_16,16,16,1,both,DRY_RUN" in text
    assert "reuse_prefix_64,64,16,1,both,DRY_RUN" in text
    report_text = report_path.read_text(encoding="utf-8")
    assert "M3.8 小矩阵" in report_text
    assert "驻留命中变化" in report_text


def test_run_matrix_dry_run_supports_store_only_and_reuse_only(tmp_path):
    store_result = run_matrix(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "store",
            dry_run=True,
            phase="store",
        )
    )
    reuse_result = run_matrix(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "reuse",
            dry_run=True,
            phase="reuse",
        )
    )

    assert store_result["status"] == "DRY_RUN"
    assert reuse_result["status"] == "DRY_RUN"
    store_text = (tmp_path / "store" / "reuse_smoke_matrix.csv").read_text(
        encoding="utf-8"
    )
    reuse_text = (tmp_path / "reuse" / "reuse_smoke_matrix.csv").read_text(
        encoding="utf-8"
    )
    assert "phase" in store_text.splitlines()[0]
    assert "reuse_prefix_16,16,16,1,store,DRY_RUN" in store_text
    assert "reuse_prefix_16,16,16,1,reuse,DRY_RUN" in reuse_text


def test_reuse_smoke_matrix_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/run_reuse_smoke_matrix.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--cold-tier-restore" in completed.stdout
    assert "--prepass-before-reuse" in completed.stdout


def test_summarize_connector_events_by_prefix_id(tmp_path):
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(
        "\n".join(
            [
                '{"event":"store_layer","prefix_id":"m3-8-64","status":"ok","elapsed_ms":1.5}',
                '{"event":"store_layer","prefix_id":"m3-8-64","status":"ok","elapsed_ms":2.0}',
                '{"event":"load_request","prefix_id":"m3-8-64","status":"ok","elapsed_ms":3.25}',
                '{"event":"load_request","prefix_id":"m3-8-128","status":"ok","elapsed_ms":9.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_connector_events(event_log, prefix_id="m3-8-64")

    assert summary["store_elapsed_ms"] == 3.5
    assert summary["load_elapsed_ms"] == 3.25
    assert summary["connector_store_elapsed_ms"] == 3.5
    assert summary["connector_load_elapsed_ms"] == 3.25
    assert summary["connector_total_elapsed_ms"] == 6.75
    assert summary["store_events"] == 2
    assert summary["load_events"] == 1
    assert summary["external_load_observed"] == "yes"


def test_summarize_connector_events_marks_missing_load(tmp_path):
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(
        '{"event":"store_layer","prefix_id":"m3-8-64","status":"ok","elapsed_ms":1.5}\n',
        encoding="utf-8",
    )

    summary = summarize_connector_events(event_log, prefix_id="m3-8-64")

    assert summary["store_events"] == 1
    assert summary["load_events"] == 0
    assert summary["external_load_observed"] == "no"


def test_summarize_sidecar_decision_delta_reads_row_decision(tmp_path):
    decision_log = tmp_path / "decisions.jsonl"
    decision_log.write_text(
        "\n".join(
            [
                '{"source":"proxy","request":{"request_id":"old"},"response":{"decision":"ADMIT","reason":"old"},"ready_barrier":{"all_required_blocks_ready":true,"sync_ssd_miss_total":0}}',
                '{"source":"proxy","request":{"request_id":"reuse_prefix_16_reuse"},"response":{"decision":"ADMIT","reason":"required_kv_ready_before_decode"},"ready_barrier":{"all_required_blocks_ready":true,"sync_ssd_miss_total":0}}',
                '{"source":"prefetch_to_dram","request":{"request_id":"reuse_prefix_16_reuse"},"response":{"decision":"DELAY","reason":"required_kv_not_ready_before_decode"},"ready_barrier":{"all_required_blocks_ready":false,"sync_ssd_miss_total":0}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    start_line = 1

    summary = summarize_sidecar_decision_delta(
        decision_log,
        start_line=start_line,
        request_id="reuse_prefix_16_reuse",
    )

    assert summary["decision"] == "ADMIT"
    assert summary["reason"] == "required_kv_ready_before_decode"
    assert summary["ready_barrier_all_ready"] == "true"
    assert summary["sync_ssd_miss_total"] == 0


def test_sidecar_decision_log_line_count_handles_missing_file(tmp_path):
    assert sidecar_decision_log_line_count(tmp_path / "missing.jsonl") == 0


def test_parse_sidecar_metrics_handles_prometheus_text():
    metrics = parse_sidecar_metrics(
        """
        # TYPE admission_decision_total counter
        admission_decision_total{decision="ADMIT",reason="ready"} 2
        admission_decision_total{decision="DELAY",reason="not_ready"} 3
        # TYPE residency_hit_total counter
        residency_hit_total 5
        prefetch_queued_total 7
        prefetch_deadline_miss_total 11
        prefetch_queue_completed_total 13
        prefetch_queue_deadline_miss_total 17
        prefetch_queue_pending_total 19
        """
    )

    assert metrics["admission_decision_total"] == 5
    assert metrics["residency_hit_total"] == 5
    assert metrics["prefetch_queued_total"] == 7
    assert metrics["prefetch_deadline_miss_total"] == 11
    assert metrics["prefetch_queue_completed_total"] == 13
    assert metrics["prefetch_queue_deadline_miss_total"] == 17
    assert metrics["prefetch_queue_pending_total"] == 19


def test_summarize_sidecar_metric_deltas_tracks_residency_and_prefetch():
    before = {
        "residency_hit_total": 1,
        "prefetch_queued_total": 2,
        "prefetch_deadline_miss_total": 3,
        "prefetch_queue_requests_total": 4,
        "prefetch_queue_completed_total": 5,
        "prefetch_queue_deadline_miss_total": 6,
        "prefetch_queue_bytes_total": 1024,
        "prefetch_queue_pending_total": 7,
    }
    after = {
        "residency_hit_total": 4,
        "prefetch_queued_total": 6,
        "prefetch_deadline_miss_total": 8,
        "prefetch_queue_requests_total": 10,
        "prefetch_queue_completed_total": 12,
        "prefetch_queue_deadline_miss_total": 14,
        "prefetch_queue_bytes_total": 4096,
        "prefetch_queue_pending_total": 9,
    }

    summary = summarize_sidecar_metric_deltas(before, after)

    assert summary["residency_hit_delta"] == 3
    assert summary["prefetch_queued_delta"] == 4
    assert summary["prefetch_deadline_miss_delta"] == 5
    assert summary["prefetch_queue_requests_delta"] == 6
    assert summary["prefetch_queue_completed_delta"] == 7
    assert summary["prefetch_queue_deadline_miss_delta"] == 8
    assert summary["prefetch_queue_bytes_delta"] == 3072
    assert summary["prefetch_queue_pending"] == 9


def test_online_payloads_use_same_prefix_id_for_store_and_reuse():
    row = matrix_rows(
        MatrixConfig(
            prefix_tokens=[64],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=Path("results/m3_8_reuse_matrix"),
        )
    )[0]

    store_payload, reuse_payload = build_store_and_reuse_payloads(
        MatrixConfig(
            prefix_tokens=[64],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=Path("results/m3_8_reuse_matrix"),
        ),
        row,
    )

    store_control = store_payload["m3_control"]
    reuse_control = reuse_payload["m3_control"]
    assert store_control["store_prefix_id"] == "m3-8-64"
    assert store_control["token_count"] == 64
    assert reuse_control["token_count"] == 80
    assert len(store_payload["prompt"].split()) == 64
    assert len(reuse_payload["prompt"].split()) == 80
    assert reuse_control["prefix_candidates"][0]["prefix_id"] == "m3-8-64"
    assert reuse_control["prefix_candidates"][0]["token_end"] == 64


def test_clone_payload_with_request_id_preserves_original_payload():
    _, reuse_payload = build_store_and_reuse_payloads(
        MatrixConfig(
            prefix_tokens=[64],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=Path("results/m3_8_reuse_matrix"),
        ),
        matrix_rows(
            MatrixConfig(
                prefix_tokens=[64],
                suffix_tokens=16,
                output_tokens=1,
                result_dir=Path("results/m3_8_reuse_matrix"),
            )
        )[0],
    )

    cloned = clone_payload_with_request_id(reuse_payload, "cold-probe")

    assert cloned["m3_control"]["request_id"] == "cold-probe"
    assert reuse_payload["m3_control"]["request_id"] == "reuse_prefix_64_reuse"
    assert cloned["m3_control"]["prefix_candidates"] == reuse_payload["m3_control"][
        "prefix_candidates"
    ]


def test_demote_prefix_to_cold_object_removes_hot_files(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = KVTensorStore(tensor_store, block_size=16)
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    store.save_layer(
        prefix_id="m3-8-16",
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "model"},
        token_ids=list(range(16)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1], block_size=16, num_tokens=16),
        layout="NHD",
    )

    summary = demote_prefix_to_cold_object(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "result",
            tensor_store=tensor_store,
            cold_root=tmp_path / "cold",
            cold_tier_restore=True,
        ),
        "m3-8-16",
    )

    assert summary["cold_tier_restore"] == "yes"
    assert summary["cold_size_bytes"] > 0
    assert summary["cold_hot_files_present_after_demote"] == "no"
    assert (tmp_path / "cold" / "m3-8-16" / "layer.0.safetensors").exists()
    assert not (tensor_store / "m3-8-16" / "layer.0.safetensors").exists()


def test_demote_prefix_to_cold_object_reports_saved_tokens(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = KVTensorStore(tensor_store, block_size=16)
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    store.save_layer(
        prefix_id="m3-8-64",
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "model"},
        token_ids=list(range(16)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1], block_size=16, num_tokens=16),
        layout="NHD",
    )

    summary = demote_prefix_to_cold_object(
        MatrixConfig(
            prefix_tokens=[64],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "result",
            tensor_store=tensor_store,
            cold_root=tmp_path / "cold",
            cold_tier_restore=True,
        ),
        "m3-8-64",
    )

    assert summary["cold_saved_tokens"] == 16
    assert summary["cold_expected_prefix_tokens"] == 64
    assert summary["cold_saved_token_mismatch"] == "yes"


def test_demote_prefix_to_cold_object_can_use_3fs_posix_backend(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = KVTensorStore(tensor_store, block_size=16)
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    store.save_layer(
        prefix_id="m3-8-16",
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "model"},
        token_ids=list(range(16)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1], block_size=16, num_tokens=16),
        layout="NHD",
    )

    summary = demote_prefix_to_cold_object(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "result",
            tensor_store=tensor_store,
            cold_root=tmp_path / "threefs_mount",
            cold_backend="3fs_posix",
            cold_tier_restore=True,
        ),
        "m3-8-16",
    )

    assert summary["cold_backend"] == "3fs_posix"
    assert (tmp_path / "threefs_mount" / "m3-8-16" / "layer.0.safetensors").exists()
    events = store.read_migration_events("m3-8-16")
    assert events[-1]["cold_backend"] == "3fs_posix"


def test_summarize_prefetch_advance_extracts_real_restore_result():
    summary = summarize_prefetch_advance(
        {
            "completed": [
                {
                    "prefix_id": "m3-8-16",
                    "status": "COMPLETED",
                    "actual_bytes": 123,
                    "checksum_status": "ok",
                    "executor_elapsed_ms": 1.25,
                }
            ]
        },
        "m3-8-16",
    )

    assert summary["cold_restore_status"] == "COMPLETED"
    assert summary["cold_restore_actual_bytes"] == 123
    assert summary["cold_restore_checksum_status"] == "ok"
    assert summary["cold_restore_executor_elapsed_ms"] == 1.25


def test_summarize_prepass_response_extracts_ready_and_restore_status():
    summary = summarize_prepass_response(
        {
            "status": "READY",
            "ready_before_request": True,
            "restore_results": [
                {
                    "status": "COMPLETED",
                    "estimated_ready_ms": 45.0,
                    "actual_bytes": 123,
                    "checksum_status": "ok",
                    "executor_elapsed_ms": 1.25,
                }
            ],
        },
        elapsed_ms=12.5,
    )

    assert summary["prepass_status"] == "READY"
    assert summary["ready_before_request"] == "true"
    assert summary["prepass_elapsed_ms"] == 12.5
    assert summary["prepass_restore_status"] == "COMPLETED"
    assert summary["prepass_restore_estimated_ready_ms"] == 45.0
    assert summary["prepass_restore_actual_bytes"] == 123
    assert summary["prepass_restore_checksum_status"] == "ok"
    assert summary["prepass_restore_executor_elapsed_ms"] == 1.25


def test_run_matrix_dry_run_includes_cold_backend_columns(tmp_path):
    result = run_matrix(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path,
            dry_run=True,
            cold_backend="3fs_posix",
            cold_tier_restore=True,
        )
    )

    assert result["status"] == "DRY_RUN"
    text = (tmp_path / "reuse_smoke_matrix.csv").read_text(encoding="utf-8")
    assert "cold_backend" in text.splitlines()[0]


def test_mark_sidecar_prefix_cold_posts_unready_commit():
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "committed"}

    class FakeClient:
        def __init__(self):
            self.calls = []

        def post(self, url, json):
            self.calls.append((url, json))
            return FakeResponse()

    client = FakeClient()

    summary = mark_sidecar_prefix_cold(
        client,
        "http://sidecar",
        request_id="reuse_prefix_64_store",
        prefix_id="m3-8-64",
        token_end=64,
    )

    assert client.calls == [
        (
            "http://sidecar/commit",
            {
                "request_id": "reuse_prefix_64_store",
                "prefix_id": "m3-8-64",
                "token_end": 64,
                "tier": "SSD",
                "ready": False,
            },
        )
    ]
    assert summary["cold_commit_status_code"] == 200
    assert summary["cold_commit_status"] == "committed"


def test_cold_tier_probe_advance_then_admit_with_sidecar_control_plane(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = KVTensorStore(tensor_store, block_size=16)
    correctness_key = {
        "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
        "tokenizer_fingerprint": "qwen2.5-tokenizer",
        "rope_config": "native-32768",
        "dtype": "bf16",
        "kv_layout": "vllm-paged",
    }
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    store.save_layer(
        prefix_id="m3-8-16",
        token_start=0,
        token_end=16,
        correctness_key=correctness_key,
        token_ids=list(range(16)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1], block_size=16, num_tokens=16),
        layout="NHD",
    )
    app = create_app(
        M3SidecarConfig(
            model_id="/root/models/Qwen2.5-14B-Instruct",
            hbm_capacity_tokens=65_000,
            dram_capacity_tokens=1_000_000,
            ssd_capacity_tokens=8_000_000,
            kv_bytes_per_token=196_608,
            h2d_gbps=25.0,
            storage_gbps=8.8,
            decision_log_path=tmp_path / "decisions.jsonl",
            tensor_store_path=tensor_store,
            async_prefetch=True,
        )
    )
    client = TestClient(app)
    row = matrix_rows(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "result",
        )
    )[0]
    store_payload, reuse_payload = build_store_and_reuse_payloads(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "result",
        ),
        row,
    )

    client.post("/admit", json=store_payload["m3_control"])
    cold_summary = demote_prefix_to_cold_object(
        MatrixConfig(
            prefix_tokens=[16],
            suffix_tokens=16,
            output_tokens=1,
            result_dir=tmp_path / "result",
            tensor_store=tensor_store,
            cold_root=tmp_path / "cold",
            cold_tier_restore=True,
        ),
        "m3-8-16",
    )
    commit_summary = mark_sidecar_prefix_cold(
        client,
        "http://testserver",
        request_id="reuse_prefix_16_store",
        prefix_id="m3-8-16",
        token_end=16,
    )
    probe = client.post("/admit", json=reuse_payload["m3_control"])
    advanced = client.post("/prefetch/advance", json={"max_ready_ms": 12_000.0})
    final = client.post("/admit", json=reuse_payload["m3_control"])

    assert cold_summary["cold_hot_files_present_after_demote"] == "no"
    assert commit_summary["cold_commit_status"] == "committed"
    assert probe.json()["decision"] == "DELAY"
    assert probe.json()["reason"] == "required_kv_not_ready_before_decode"
    assert advanced.json()["completed"][0]["status"] == "COMPLETED"
    assert advanced.json()["completed"][0]["checksum_status"] == "ok"
    assert final.json()["decision"] == "ADMIT"
