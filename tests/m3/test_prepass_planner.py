import json
from pathlib import Path

from fastapi.testclient import TestClient

from benchmarks.m3.http_sidecar import M3SidecarConfig, create_app


def _payload(
    request_id: str,
    *,
    token_count: int = 80,
    prefix_id: str = "session-a",
    prefix_end: int = 64,
    prepass_deadline_ms: float | None = None,
) -> dict:
    payload = {
        "request_id": request_id,
        "model_id": "/root/models/Qwen2.5-14B-Instruct",
        "token_count": token_count,
        "decode_sla_ms": 12_000,
        "admission_window_ms": 12_000,
        "correctness_key": {
            "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
            "tokenizer_fingerprint": "qwen2.5-tokenizer",
            "rope_config": "native-32768",
            "dtype": "bf16",
            "kv_layout": "vllm-paged",
        },
        "prefix_candidates": [
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": prefix_end,
                "committed": True,
            }
        ],
    }
    if prepass_deadline_ms is not None:
        payload["prepass_deadline_ms"] = prepass_deadline_ms
    return payload


def _write_cold_manifest(tensor_store: Path, prefix_id: str = "session-a") -> None:
    prefix_dir = tensor_store / prefix_id
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": 64,
                "correctness_key": {
                    "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
                    "tokenizer_fingerprint": "qwen2.5-tokenizer",
                    "rope_config": "native-32768",
                    "dtype": "bf16",
                    "kv_layout": "vllm-paged",
                },
                "token_ids": list(range(64)),
                "block_size": 16,
                "layout": "NHD",
                "tier": "NVME",
                "ready": False,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )


def _create_client_with_cold_prefix(
    tmp_path: Path,
    *,
    async_prefetch: bool = False,
) -> TestClient:
    tensor_store = tmp_path / "tensor_store"
    _write_cold_manifest(tensor_store)
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
            async_prefetch=async_prefetch,
        )
    )
    runtime = app.state.runtime
    response = runtime.admit_from_payload(
        {
            "request_id": "commit-r1",
            "model_id": "/root/models/Qwen2.5-14B-Instruct",
            "token_count": 64,
            "decode_sla_ms": 12_000,
            "admission_window_ms": 12_000,
            "correctness_key": _payload("unused")["correctness_key"],
            "prefix_candidates": [],
        }
    )
    runtime.control_plane.commit_request(
        runtime.responses[response["request_id"]],
        prefix_id="session-a",
        token_end=64,
        tier="SSD",
        ready=False,
    )
    return TestClient(app)


def test_prepass_restores_cold_prefix_before_online_admit(tmp_path):
    client = _create_client_with_cold_prefix(tmp_path)

    metrics_before = client.get("/metrics").text
    prepass = client.post("/prepass", json=_payload("prepass-r1"))
    metrics_after_prepass = client.get("/metrics").text
    admitted = client.post("/admit", json=_payload("online-r1"))

    assert prepass.status_code == 200
    body = prepass.json()
    assert body["status"] == "READY"
    assert body["ready_before_request"] is True
    assert body["restore_results"][0]["status"] == "COMPLETED"
    assert admitted.json()["decision"] == "ADMIT"
    assert admitted.json()["ready_barrier"]["all_required_blocks_ready"] is True
    assert _metric(metrics_after_prepass, "prefill_tokens_saved_total") == _metric(
        metrics_before,
        "prefill_tokens_saved_total",
    )

    manifest = json.loads(
        (tmp_path / "tensor_store" / "session-a" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["tier"] == "DRAM"
    assert manifest["ready"] is True

    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["source"] == "prepass_result" for record in records)


def _metric(text: str, metric_name: str) -> float:
    for line in text.splitlines():
        if line.startswith(metric_name + " "):
            return float(line.split()[1])
    raise AssertionError(f"missing metric {metric_name}")


def test_async_prepass_queues_restore_without_marking_ready_until_advanced(tmp_path):
    client = _create_client_with_cold_prefix(tmp_path, async_prefetch=True)

    prepass = client.post("/prepass", json=_payload("prepass-r1"))
    before_advance = client.post("/admit", json=_payload("online-before"))
    advanced = client.post("/prefetch/advance", json={"max_ready_ms": 12_000.0})
    after_advance = client.post("/admit", json=_payload("online-after"))

    assert prepass.json()["status"] == "QUEUED"
    assert prepass.json()["ready_before_request"] is False
    assert prepass.json()["restore_results"][0]["status"] == "QUEUED"
    assert before_advance.json()["decision"] == "DELAY"
    assert advanced.json()["completed"][0]["status"] == "COMPLETED"
    assert after_advance.json()["decision"] == "ADMIT"
