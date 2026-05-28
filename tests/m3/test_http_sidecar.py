import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

from fastapi.testclient import TestClient

from benchmarks.m3.http_sidecar import (
    M3SidecarConfig,
    build_kv_transfer_params,
    create_app,
)


def _client(tmp_path: Path) -> TestClient:
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
        )
    )
    return TestClient(app)


def _payload(
    request_id: str,
    token_count: int = 1_048_576,
    prefix_end: int = 0,
) -> dict:
    candidates = []
    if prefix_end:
        candidates.append(
            {
                "prefix_id": "session-a",
                "token_start": 0,
                "token_end": prefix_end,
                "committed": True,
            }
        )
    return {
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
        "prefix_candidates": candidates,
    }


def test_admit_logs_decision_and_metrics(tmp_path):
    client = _client(tmp_path)

    response = client.post("/admit", json=_payload("r1"))

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "FULL_PREFILL_FALLBACK"
    assert body["sync_ssd_miss_allowed"] is False
    assert body["ready_barrier"]["sync_ssd_miss_total"] == 0
    assert (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")

    metrics = client.get("/metrics").text
    assert 'admission_decision_total{decision="FULL_PREFILL_FALLBACK"' in metrics
    assert "sync_ssd_miss_total 0" in metrics


def test_commit_then_admit_reuses_prefix(tmp_path):
    client = _client(tmp_path)

    first = client.post("/admit", json=_payload("r1"))
    client.post(
        "/commit",
        json={
            "request_id": "r1",
            "prefix_id": "session-a",
            "token_end": 1_048_576,
            "tier": "DRAM",
        },
    )
    second = client.post("/admit", json=_payload("r2", prefix_end=786_432))

    assert first.json()["decision"] == "FULL_PREFILL_FALLBACK"
    assert second.json()["decision"] == "ADMIT"
    assert second.json()["reuse_tokens"] == 786_432
    assert second.json()["delta_prefill_tokens"] == 262_144


def test_commit_can_record_unready_capacity_tier_prefix(tmp_path):
    client = _client(tmp_path)

    client.post("/admit", json=_payload("r1", token_count=256))
    commit = client.post(
        "/commit",
        json={
            "request_id": "r1",
            "prefix_id": "session-a",
            "token_end": 256,
            "tier": "SSD",
            "ready": False,
        },
    )
    second = client.post("/admit", json=_payload("r2", token_count=272, prefix_end=256))

    assert commit.status_code == 200
    runtime = client.app.state.runtime
    entry = runtime.control_plane.manifest._ranges["session-a"][-1]
    assert entry.ready is False
    assert second.status_code == 200
    assert second.json()["decision"] == "DELAY"
    assert second.json()["reason"] == "required_kv_not_ready_before_decode"
    assert second.json()["ready_barrier"]["all_required_blocks_ready"] is False


def test_proxy_without_upstream_records_sidecar_decision(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/v1/completions",
        json={
            "model": "/root/models/Qwen2.5-14B-Instruct",
            "prompt": "hello",
            "max_tokens": 1,
            "m3_control": _payload("proxy-r1", token_count=128),
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["sidecar"]["decision"] == "ADMIT"
    assert body["proxied"] is False
    assert body["reason"] == "upstream_not_configured"


def test_proxy_forwards_request_without_sidecar_metadata(tmp_path):
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured["path"] = self.path
            captured["body"] = body.decode("utf-8")
            payload = json.dumps({"id": "cmpl-1", "choices": [{"text": "ok"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
        decision_log_path=tmp_path / "decisions.jsonl",
        upstream_base_url=f"http://127.0.0.1:{server.server_port}",
    )
    app = create_app(config)
    client = TestClient(app)

    try:
        response = client.post(
            "/v1/completions",
            json={
                "model": "/root/models/Qwen2.5-14B-Instruct",
                "prompt": "hello",
                "max_tokens": 1,
                "m3_control": _payload("proxy-r1", token_count=128),
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.json()["id"] == "cmpl-1"
    assert captured["path"] == "/v1/completions"
    assert "m3_control" not in captured["body"]


def test_build_kv_transfer_params_carries_sidecar_plan(tmp_path):
    client = _client(tmp_path)
    client.post("/admit", json=_payload("r1"))
    client.post(
        "/commit",
        json={
            "request_id": "r1",
            "prefix_id": "session-a",
            "token_end": 1_048_576,
            "tier": "DRAM",
        },
    )
    response = client.post("/admit", json=_payload("r2", prefix_end=786_432))

    params = build_kv_transfer_params(
        sidecar_request=_payload("r2", prefix_end=786_432),
        sidecar_response=response.json(),
    )

    assert params["m3_connector_version"] == 1
    assert params["request_id"] == "r2"
    assert params["decision"] == "ADMIT"
    assert params["reuse_tokens"] == 786_432
    assert params["delta_prefill_tokens"] == 262_144
    assert params["store_policy"] == "load_only"
    assert "store_prefix_id" not in params
    assert "store_token_end" not in params
    assert params["sync_ssd_miss_allowed"] is False
    assert params["correctness_key"]["model_fingerprint"].endswith(
        "Qwen2.5-14B-Instruct"
    )
    assert params["required_ranges"] == [
        {
            "prefix_id": "session-a",
            "token_start": 0,
            "token_end": 786_432,
            "required_tier": "DRAM_OR_HBM",
            "deadline_ms": 12_000.0,
        }
    ]


def test_build_kv_transfer_params_marks_explicit_store_request():
    params = build_kv_transfer_params(
        sidecar_request={
            **_payload("r1", token_count=128),
            "store_prefix_id": "session-a",
        },
        sidecar_response={
            "decision": "ADMIT",
            "reason": "no_reuse_required",
            "reuse_tokens": 0,
            "delta_prefill_tokens": 128,
            "sync_ssd_miss_allowed": False,
            "required_ranges": [],
        },
    )

    assert params["store_policy"] == "store_prefix"
    assert params["store_prefix_id"] == "session-a"
    assert params["store_token_end"] == 128


def test_proxy_forwards_kv_transfer_params_for_vllm_connector(tmp_path):
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured["body"] = body.decode("utf-8")
            payload = json.dumps({"id": "cmpl-1", "choices": [{"text": "ok"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
        decision_log_path=tmp_path / "decisions.jsonl",
        upstream_base_url=f"http://127.0.0.1:{server.server_port}",
        inject_kv_transfer_params=True,
    )
    app = create_app(config)
    client = TestClient(app)

    try:
        response = client.post(
            "/v1/completions",
            json={
                "model": "/root/models/Qwen2.5-14B-Instruct",
                "prompt": "hello",
                "max_tokens": 1,
                "m3_control": _payload("proxy-r2", token_count=128),
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response.status_code == 200
    forwarded = json.loads(captured["body"])
    assert "m3_control" not in forwarded
    assert forwarded["kv_transfer_params"]["m3_connector_version"] == 1
    assert forwarded["kv_transfer_params"]["request_id"] == "proxy-r2"
    assert forwarded["kv_transfer_params"]["decision"] == "ADMIT"


def test_proxy_auto_commits_manifest_returned_by_connector(tmp_path):
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured["body"] = body.decode("utf-8")
            payload = json.dumps(
                {
                    "id": "cmpl-1",
                    "choices": [{"text": "ok"}],
                    "kv_transfer_params": {
                        "m3_noop_connector": {
                            "request_id": "vllm-internal-id",
                            "manifest": {
                                "prefix_id": "session-a",
                                "token_start": 0,
                                "token_end": 64,
                                "block_ids": [1, 2, 3, 4],
                            },
                        }
                    },
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
        decision_log_path=tmp_path / "decisions.jsonl",
        upstream_base_url=f"http://127.0.0.1:{server.server_port}",
        inject_kv_transfer_params=True,
    )
    app = create_app(config)
    client = TestClient(app)

    try:
        first = client.post(
            "/v1/completions",
            json={
                "model": "/root/models/Qwen2.5-14B-Instruct",
                "prompt": "hello",
                "max_tokens": 1,
                "m3_control": _payload("proxy-r1", token_count=64),
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert first.status_code == 200
    second = client.post("/admit", json=_payload("proxy-r2", token_count=80, prefix_end=64))
    assert second.status_code == 200
    assert second.json()["decision"] == "ADMIT"
    assert second.json()["reuse_tokens"] == 64

    log_records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["source"] == "proxy_auto_commit" for record in log_records)


def test_sidecar_prefetches_tensor_store_when_required_kv_not_ready(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    prefix_dir = tensor_store / "session-a"
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": "session-a",
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
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
        decision_log_path=tmp_path / "decisions.jsonl",
        tensor_store_path=tensor_store,
    )
    app = create_app(config)
    runtime = app.state.runtime
    response = runtime.admit_from_payload(_payload("commit-r1", token_count=64))
    runtime.control_plane.commit_request(
        runtime.responses["commit-r1"],
        prefix_id="session-a",
        token_end=64,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    delayed = client.post("/admit", json=_payload("r1", token_count=80, prefix_end=64))
    admitted = client.post("/admit", json=_payload("r2", token_count=80, prefix_end=64))

    assert delayed.json()["decision"] == "DELAY"
    assert delayed.json()["ready_barrier"]["all_required_blocks_ready"] is False
    assert admitted.json()["decision"] == "ADMIT"
    manifest = json.loads((prefix_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "DRAM"
    assert manifest["ready"] is True
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["source"] == "prefetch_to_dram" for record in records)


def test_sidecar_prefetch_queue_preserves_delay_when_deadline_missed(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    prefix_dir = tensor_store / "session-a"
    prefix_dir.mkdir(parents=True)
    token_end = 1_000_000
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": "session-a",
                "token_start": 0,
                "token_end": token_end,
                "correctness_key": {
                    "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
                    "tokenizer_fingerprint": "qwen2.5-tokenizer",
                    "rope_config": "native-32768",
                    "dtype": "bf16",
                    "kv_layout": "vllm-paged",
                },
                "token_ids": list(range(16)),
                "block_size": 16,
                "layout": "NHD",
                "tier": "NVME",
                "ready": False,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=0.002,
        decision_log_path=tmp_path / "decisions.jsonl",
        tensor_store_path=tensor_store,
    )
    app = create_app(config)
    runtime = app.state.runtime
    response = runtime.admit_from_payload(_payload("commit-r1", token_count=token_end))
    runtime.control_plane.commit_request(
        runtime.responses[response["request_id"]],
        prefix_id="session-a",
        token_end=token_end,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    first = client.post(
        "/admit",
        json=_payload("r1", token_count=token_end + 16, prefix_end=token_end),
    )
    second = client.post(
        "/admit",
        json=_payload("r2", token_count=token_end + 16, prefix_end=token_end),
    )

    assert first.json()["decision"] == "DELAY"
    assert second.json()["decision"] == "DELAY"
    manifest = json.loads((prefix_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "NVME"
    assert manifest["ready"] is False
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prefetch_records = [
        record for record in records if record["source"] == "prefetch_to_dram"
    ]
    assert prefetch_records
    assert prefetch_records[-1]["prefetch_results"][0]["status"] == "DEADLINE_MISS"


def test_sidecar_metrics_include_prefetch_queue_stats(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    for prefix_id, token_end in [("session-a", 64), ("session-b", 1_000_000)]:
        prefix_dir = tensor_store / prefix_id
        prefix_dir.mkdir(parents=True)
        (prefix_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "prefix_id": prefix_id,
                    "token_start": 0,
                    "token_end": token_end,
                    "correctness_key": {
                        "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
                        "tokenizer_fingerprint": "qwen2.5-tokenizer",
                        "rope_config": "native-32768",
                        "dtype": "bf16",
                        "kv_layout": "vllm-paged",
                    },
                    "token_ids": list(range(min(token_end, 16))),
                    "block_size": 16,
                    "layout": "NHD",
                    "tier": "NVME",
                    "ready": False,
                    "layers": {},
                }
            ),
            encoding="utf-8",
        )
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=0.002,
        decision_log_path=tmp_path / "decisions.jsonl",
        tensor_store_path=tensor_store,
    )
    app = create_app(config)
    runtime = app.state.runtime
    for prefix_id, token_end in [("session-a", 64), ("session-b", 1_000_000)]:
        response = runtime.admit_from_payload(
            _payload(f"commit-{prefix_id}", token_count=token_end)
        )
        runtime.control_plane.commit_request(
            runtime.responses[response["request_id"]],
            prefix_id=prefix_id,
            token_end=token_end,
            tier="SSD",
            ready=False,
        )
    client = TestClient(app)

    client.post("/admit", json=_payload("r1", token_count=80, prefix_end=64))
    client.post(
        "/admit",
        json={
            **_payload("r2", token_count=1_000_016, prefix_end=1_000_000),
            "prefix_candidates": [
                {
                    "prefix_id": "session-b",
                    "token_start": 0,
                    "token_end": 1_000_000,
                    "committed": True,
                }
            ],
        },
    )

    metrics = client.get("/metrics").text

    assert "prefetch_queue_requests_total 2" in metrics
    assert "prefetch_queue_completed_total 1" in metrics
    assert "prefetch_queue_deadline_miss_total 1" in metrics
    assert "prefetch_queue_bytes_total 196620582912" in metrics
    assert "prefetch_queue_last_estimated_ready_ms" in metrics


def test_sidecar_treats_dram_ready_tensor_store_as_residency_hit(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    prefix_dir = tensor_store / "session-a"
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": "session-a",
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
                "tier": "DRAM",
                "ready": True,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=0.000001,
        decision_log_path=tmp_path / "decisions.jsonl",
        tensor_store_path=tensor_store,
    )
    app = create_app(config)
    runtime = app.state.runtime
    response = runtime.admit_from_payload(_payload("commit-r1", token_count=64))
    runtime.control_plane.commit_request(
        runtime.responses[response["request_id"]],
        prefix_id="session-a",
        token_end=64,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    first = client.post("/admit", json=_payload("r1", token_count=80, prefix_end=64))
    second = client.post("/admit", json=_payload("r2", token_count=80, prefix_end=64))

    assert first.json()["decision"] == "ADMIT"
    assert first.json()["reason"] == "required_kv_ready_before_decode"
    assert second.json()["decision"] == "ADMIT"
    assert second.json()["reason"] == "required_kv_ready_before_decode"
    metrics = client.get("/metrics").text
    assert "residency_hit_total 1" in metrics
    assert "prefetch_queue_requests_total 0" in metrics
    assert "prefetch_queue_deadline_miss_total 0" in metrics
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    hit_records = [
        record for record in records if record["source"] == "residency_hit"
    ]
    assert hit_records
    assert hit_records[-1]["hits"][0]["prefix_id"] == "session-a"


def test_sidecar_records_prefetch_error_when_tensor_manifest_missing(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    tensor_store.mkdir(parents=True)
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
        decision_log_path=tmp_path / "decisions.jsonl",
        tensor_store_path=tensor_store,
    )
    app = create_app(config)
    runtime = app.state.runtime
    response = runtime.admit_from_payload(_payload("commit-r1", token_count=64))
    runtime.control_plane.commit_request(
        runtime.responses[response["request_id"]],
        prefix_id="session-a",
        token_end=64,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    delayed = client.post("/admit", json=_payload("r1", token_count=80, prefix_end=64))

    assert delayed.status_code == 200
    assert delayed.json()["decision"] == "DELAY"
    metrics = client.get("/metrics").text
    assert "residency_hit_total 0" in metrics
    assert "prefetch_queued_total 0" in metrics
    assert "prefetch_queue_requests_total 0" in metrics
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prefetch_records = [
        record for record in records if record["source"] == "prefetch_to_dram"
    ]
    assert prefetch_records
    assert prefetch_records[-1]["errors"][0]["prefix_id"] == "session-a"
    assert "no manifest" in prefetch_records[-1]["errors"][0]["error"]


def test_sidecar_does_not_prefetch_tensor_manifest_with_wrong_correctness_key(
    tmp_path,
):
    tensor_store = tmp_path / "tensor_store"
    prefix_dir = tensor_store / "session-a"
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": "session-a",
                "token_start": 0,
                "token_end": 64,
                "correctness_key": {
                    "model_fingerprint": "wrong-model",
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
    config = M3SidecarConfig(
        model_id="/root/models/Qwen2.5-14B-Instruct",
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
        decision_log_path=tmp_path / "decisions.jsonl",
        tensor_store_path=tensor_store,
    )
    app = create_app(config)
    runtime = app.state.runtime
    response = runtime.admit_from_payload(_payload("commit-r1", token_count=64))
    runtime.control_plane.commit_request(
        runtime.responses[response["request_id"]],
        prefix_id="session-a",
        token_end=64,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    first = client.post("/admit", json=_payload("r1", token_count=80, prefix_end=64))
    second = client.post("/admit", json=_payload("r2", token_count=80, prefix_end=64))

    assert first.json()["decision"] == "DELAY"
    assert second.json()["decision"] == "DELAY"
    metrics = client.get("/metrics").text
    assert "residency_hit_total 0" in metrics
    assert "prefetch_queued_total 0" in metrics
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prefetch_records = [
        record for record in records if record["source"] == "prefetch_to_dram"
    ]
    assert prefetch_records
    assert prefetch_records[-1]["errors"][0]["prefix_id"] == "session-a"
    assert "correctness key mismatch" in prefetch_records[-1]["errors"][0]["error"]


def test_sidecar_async_prefetch_delays_until_background_queue_advances(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    prefix_dir = tensor_store / "session-a"
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": "session-a",
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
    config = M3SidecarConfig(
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
    app = create_app(config)
    runtime = app.state.runtime
    response = runtime.admit_from_payload(_payload("commit-r1", token_count=64))
    runtime.control_plane.commit_request(
        runtime.responses[response["request_id"]],
        prefix_id="session-a",
        token_end=64,
        tier="SSD",
        ready=False,
    )
    client = TestClient(app)

    first = client.post("/admit", json=_payload("r1", token_count=80, prefix_end=64))
    second = client.post("/admit", json=_payload("r2", token_count=80, prefix_end=64))
    completed = client.post("/prefetch/advance", json={"max_ready_ms": 12_000.0})
    third = client.post("/admit", json=_payload("r3", token_count=80, prefix_end=64))

    assert first.json()["decision"] == "DELAY"
    assert second.json()["decision"] == "DELAY"
    assert completed.json()["completed"][0]["prefix_id"] == "session-a"
    assert third.json()["decision"] == "ADMIT"
    metrics = client.get("/metrics").text
    assert "prefetch_queued_total 1" in metrics
    assert "prefetch_queue_pending_total 0" in metrics
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prefetch_records = [
        record for record in records if record["source"] == "prefetch_to_dram"
    ]
    assert prefetch_records
    prefetch_statuses = [
        result["status"]
        for record in prefetch_records
        for result in record["prefetch_results"]
    ]
    assert "QUEUED" in prefetch_statuses
    assert "ALREADY_QUEUED" in prefetch_statuses
