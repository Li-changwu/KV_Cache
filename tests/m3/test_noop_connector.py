import json
from types import SimpleNamespace

import torch

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from benchmarks.m3.noop_connector import (
    M3NoOpConnector,
    M3NoOpConnectorMetadata,
    M3NoOpConnectorWorkerMetadata,
    M3NoOpRequestMeta,
    align_down_to_block_size,
)


class FakeBlocks:
    def __init__(self, block_ids):
        self._block_ids = block_ids

    def get_block_ids(self):
        return [self._block_ids]


def fake_request(request_id="r1", reuse_tokens=128, decision="ADMIT"):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(200)),
        kv_transfer_params={
            "m3_connector_version": 1,
            "request_id": request_id,
            "decision": decision,
            "reuse_tokens": reuse_tokens,
            "delta_prefill_tokens": 200 - reuse_tokens,
            "sync_ssd_miss_allowed": False,
            "correctness_key": {"model_fingerprint": "model"},
            "store_prefix_id": "session-a",
            "store_token_end": 192,
            "required_ranges": [
                {
                    "prefix_id": "session-a",
                    "token_start": 0,
                    "token_end": reuse_tokens,
                    "required_tier": "DRAM_OR_HBM",
                    "deadline_ms": 12_000.0,
                }
            ],
        },
    )


def test_align_down_to_block_size():
    assert align_down_to_block_size(0, 16) == 0
    assert align_down_to_block_size(15, 16) == 0
    assert align_down_to_block_size(16, 16) == 16
    assert align_down_to_block_size(31, 16) == 16


def test_get_num_new_matched_tokens_uses_admitted_plan_and_block_alignment():
    connector = M3NoOpConnector.for_testing(block_size=16)
    request = fake_request(reuse_tokens=130)

    matched, async_load = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=32,
    )

    assert matched == 96
    assert async_load is False
    assert connector.observed_requests["r1"].aligned_reuse_tokens == 128


def test_get_num_new_matched_tokens_ignores_non_admitted_or_sync_ssd_plans():
    connector = M3NoOpConnector.for_testing(block_size=16)
    rejected = fake_request(request_id="reject", reuse_tokens=128, decision="REJECT")
    bad = fake_request(request_id="bad", reuse_tokens=128)
    bad.kv_transfer_params["sync_ssd_miss_allowed"] = True

    assert connector.get_num_new_matched_tokens(rejected, 0) == (0, False)
    assert connector.get_num_new_matched_tokens(bad, 0) == (0, False)


def test_update_state_after_alloc_and_build_connector_meta():
    connector = M3NoOpConnector.for_testing(block_size=16)
    request = fake_request(reuse_tokens=128)

    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
    connector.update_state_after_alloc(request, FakeBlocks([10, 11, 12]), 128)
    metadata = connector.build_connector_meta(
        SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=None)
    )

    assert isinstance(metadata, M3NoOpConnectorMetadata)
    assert len(metadata.requests) == 1
    item = metadata.requests[0]
    assert item.request_id == "r1"
    assert item.reuse_tokens == 128
    assert item.local_block_ids == [10, 11, 12]
    assert item.correctness_key == {"model_fingerprint": "model"}
    assert item.prefix_id == "session-a"
    assert item.token_start == 0
    assert item.token_ids == list(range(128))
    assert item.required_ranges == [
        {
            "prefix_id": "session-a",
            "token_start": 0,
            "token_end": 128,
            "required_tier": "DRAM_OR_HBM",
            "deadline_ms": 12_000.0,
        }
    ]
    assert connector.pending_loads == {}


def test_worker_methods_record_metadata_without_tensor_store():
    connector = M3NoOpConnector.for_testing(block_size=16)
    metadata = M3NoOpConnectorMetadata(
        requests=[
            M3NoOpRequestMeta(
                request_id="r1",
                reuse_tokens=128,
                local_block_ids=[1, 2],
                operation="load",
                correctness_key={"model_fingerprint": "model"},
                prefix_id="session-a",
                token_start=0,
                token_ids=list(range(128)),
                required_ranges=[],
            )
        ]
    )

    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(SimpleNamespace())
    connector.wait_for_layer_load("layer.0")
    connector.save_kv_layer("layer.0", kv_layer=None, attn_metadata=None)
    connector.wait_for_save()
    worker_meta = connector.build_connector_worker_meta()

    assert connector.loaded_requests == ["r1"]
    assert isinstance(worker_meta, M3NoOpConnectorWorkerMetadata)
    assert worker_meta.loaded_request_ids == ["r1"]
    assert connector.get_block_ids_with_load_errors() == set()


def test_build_connector_meta_records_store_request_when_tensor_store_enabled(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    request = fake_request(reuse_tokens=0)
    request.kv_transfer_params["reuse_tokens"] = 0
    request.kv_transfer_params["store_token_end"] = 32

    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
    metadata = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="r1", prompt_token_ids=list(range(32)), block_ids=([7, 8],))
            ],
            scheduled_cached_reqs=None,
        )
    )

    stores = [item for item in metadata.requests if item.operation == "store"]
    assert len(stores) == 1
    assert stores[0].prefix_id == "session-a"
    assert stores[0].reuse_tokens == 32
    assert stores[0].local_block_ids == [7, 8]


def test_build_connector_meta_does_not_store_load_only_reuse_by_default(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    request = fake_request(reuse_tokens=128)
    request.kv_transfer_params.pop("store_prefix_id")
    request.kv_transfer_params["store_token_end"] = 192

    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
    connector.update_state_after_alloc(request, FakeBlocks([10, 11, 12, 13]), 128)
    metadata = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="r1",
                    prompt_token_ids=list(range(200)),
                    block_ids=([10, 11, 12, 13, 14, 15, 16, 17],),
                )
            ],
            scheduled_cached_reqs=None,
        )
    )

    assert [item.operation for item in metadata.requests] == ["load"]


def test_request_finished_manifest_prefers_store_prefix_id(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    request = fake_request(request_id="r1", reuse_tokens=0)
    request.kv_transfer_params["store_prefix_id"] = "session-a"
    request.kv_transfer_params["store_token_end"] = 32
    request.kv_transfer_params["required_ranges"] = []

    _, params = connector.request_finished(request, block_ids=[7, 8])

    assert params is not None
    assert params["m3_noop_connector"]["manifest"]["prefix_id"] == "session-a"


def test_request_finished_suppresses_manifest_for_load_only_reuse(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    request = fake_request(request_id="r1", reuse_tokens=128)
    request.kv_transfer_params.pop("store_prefix_id")
    request.kv_transfer_params["store_policy"] = "load_only"
    request.kv_transfer_params["store_token_end"] = 192

    _, params = connector.request_finished(request, block_ids=[7, 8, 9, 10])

    assert params is None


def test_store_request_clamps_to_actual_prompt_tokens_and_allocated_blocks(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    request = fake_request(reuse_tokens=0)
    request.prompt_token_ids = list(range(40))
    request.kv_transfer_params["store_token_end"] = 128
    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    metadata = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="r1", prompt_token_ids=list(range(40)), block_ids=([7],))
            ],
            scheduled_cached_reqs=None,
        )
    )

    stores = [item for item in metadata.requests if item.operation == "store"]
    assert len(stores) == 1
    assert stores[0].reuse_tokens == 16
    assert stores[0].local_block_ids == [7]


def test_start_load_kv_reports_load_error_on_correctness_key_mismatch(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    slot_mapping = torch.arange(0, 16)
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    connector.tensor_store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "old"},
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=slot_mapping,
        layout="NHD",
    )
    metadata = M3NoOpConnectorMetadata(
        requests=[
            M3NoOpRequestMeta(
                request_id="r1",
                reuse_tokens=16,
                local_block_ids=[0],
                operation="load",
                correctness_key={"model_fingerprint": "new"},
                prefix_id="session-a",
                token_start=0,
                token_ids=list(range(16)),
                required_ranges=[],
            )
        ]
    )
    connector.bind_connector_metadata(metadata)
    connector.register_kv_caches({"layer.0": torch.zeros_like(source)})

    connector.start_load_kv(SimpleNamespace())

    assert connector.loaded_requests == []
    assert connector.get_block_ids_with_load_errors() == {0}


def test_connector_saves_then_loads_tensor_blocks(tmp_path):
    connector = M3NoOpConnector.for_testing(block_size=16, storage_path=tmp_path)
    source = torch.arange(4 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        4, 2, 16, 1, 1
    )
    request = fake_request(reuse_tokens=0)
    request.kv_transfer_params["store_token_end"] = 32
    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
    store_meta = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="r1", prompt_token_ids=list(range(32)), block_ids=([1, 2],))
            ],
            scheduled_cached_reqs=None,
        )
    )

    connector.bind_connector_metadata(store_meta)
    connector.save_kv_layer("layer.0", source, attn_metadata=None)
    connector.wait_for_save()

    target = torch.zeros_like(source)
    connector.register_kv_caches({"layer.0": target})
    load_meta = M3NoOpConnectorMetadata(
        requests=[
            M3NoOpRequestMeta(
                request_id="r2",
                reuse_tokens=32,
                local_block_ids=[0, 3],
                operation="load",
                correctness_key={"model_fingerprint": "model"},
                prefix_id="session-a",
                token_start=0,
                token_ids=list(range(32)),
                required_ranges=[],
            )
        ]
    )
    connector.bind_connector_metadata(load_meta)

    connector.start_load_kv(SimpleNamespace())

    assert connector.loaded_requests == ["r2"]
    assert torch.equal(target[0], source[1])
    assert torch.equal(target[3], source[2])


def test_connector_writes_structured_events_and_metrics(tmp_path):
    event_log = tmp_path / "events.jsonl"
    connector = M3NoOpConnector.for_testing(
        block_size=16,
        storage_path=tmp_path / "store",
        event_log_path=event_log,
    )
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    request = fake_request(reuse_tokens=0)
    request.kv_transfer_params["store_token_end"] = 16
    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
    store_meta = connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="r1", prompt_token_ids=list(range(16)), block_ids=([1],))
            ],
            scheduled_cached_reqs=None,
        )
    )

    connector.bind_connector_metadata(store_meta)
    connector.save_kv_layer("layer.0", source, attn_metadata=None)
    connector.wait_for_save()

    target = torch.zeros_like(source)
    connector.register_kv_caches({"layer.0": target})
    load_meta = M3NoOpConnectorMetadata(
        requests=[
            M3NoOpRequestMeta(
                request_id="r2",
                reuse_tokens=16,
                local_block_ids=[0],
                operation="load",
                correctness_key={"model_fingerprint": "model"},
                prefix_id="session-a",
                token_start=0,
                token_ids=list(range(16)),
                required_ranges=[],
            )
        ]
    )
    connector.bind_connector_metadata(load_meta)

    connector.start_load_kv(SimpleNamespace())

    records = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == [
        "store_layer",
        "load_request",
    ]
    assert records[0]["status"] == "ok"
    assert records[0]["prefix_id"] == "session-a"
    assert records[0]["tokens"] == 16
    assert records[0]["layer_name"] == "layer.0"
    assert records[0]["elapsed_ms"] >= 0
    assert records[1]["status"] == "ok"
    assert records[1]["layers"] == 1
    assert records[1]["tokens"] == 16
    assert connector.metrics["store_layer_ok_total"] == 1
    assert connector.metrics["load_request_ok_total"] == 1
    assert connector.metrics["store_tokens_total"] == 16
    assert connector.metrics["load_tokens_total"] == 16


def test_connector_writes_load_error_event(tmp_path):
    event_log = tmp_path / "events.jsonl"
    connector = M3NoOpConnector.for_testing(
        block_size=16,
        storage_path=tmp_path / "store",
        event_log_path=event_log,
    )
    slot_mapping = torch.arange(0, 16)
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    connector.tensor_store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "old"},
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=slot_mapping,
        layout="NHD",
    )
    connector.register_kv_caches({"layer.0": torch.zeros_like(source)})
    connector.bind_connector_metadata(
        M3NoOpConnectorMetadata(
            requests=[
                M3NoOpRequestMeta(
                    request_id="r1",
                    reuse_tokens=16,
                    local_block_ids=[0],
                    operation="load",
                    correctness_key={"model_fingerprint": "new"},
                    prefix_id="session-a",
                    token_start=0,
                    token_ids=list(range(16)),
                    required_ranges=[],
                )
            ]
        )
    )

    connector.start_load_kv(SimpleNamespace())

    records = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["event"] == "load_request"
    assert records[0]["status"] == "error"
    assert "correctness key mismatch" in records[0]["error"]
    assert records[0]["block_count"] == 1
    assert connector.metrics["load_request_error_total"] == 1


def test_connector_refuses_nvme_load_until_prefetched(tmp_path):
    event_log = tmp_path / "events.jsonl"
    connector = M3NoOpConnector.for_testing(
        block_size=16,
        storage_path=tmp_path / "store",
        event_log_path=event_log,
    )
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    connector.tensor_store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "model"},
        token_ids=list(range(16)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=torch.arange(0, 16),
        layout="NHD",
    )
    connector.tensor_store.demote_to_nvme("session-a")
    target = torch.zeros_like(source)
    connector.register_kv_caches({"layer.0": target})
    connector.bind_connector_metadata(
        M3NoOpConnectorMetadata(
            requests=[
                M3NoOpRequestMeta(
                    request_id="r1",
                    reuse_tokens=16,
                    local_block_ids=[0],
                    operation="load",
                    correctness_key={"model_fingerprint": "model"},
                    prefix_id="session-a",
                    token_start=0,
                    token_ids=list(range(16)),
                    required_ranges=[],
                )
            ]
        )
    )

    connector.start_load_kv(SimpleNamespace())

    assert connector.loaded_requests == []
    assert connector.get_block_ids_with_load_errors() == {0}
    error_record = json.loads(event_log.read_text(encoding="utf-8").splitlines()[-1])
    assert error_record["event"] == "load_request"
    assert error_record["status"] == "error"
    assert "prefetch to DRAM" in error_record["error"]

    connector.tensor_store.prefetch_to_dram("session-a")
    connector.load_error_block_ids.clear()
    connector.bind_connector_metadata(
        M3NoOpConnectorMetadata(
            requests=[
                M3NoOpRequestMeta(
                    request_id="r2",
                    reuse_tokens=16,
                    local_block_ids=[0],
                    operation="load",
                    correctness_key={"model_fingerprint": "model"},
                    prefix_id="session-a",
                    token_start=0,
                    token_ids=list(range(16)),
                    required_ranges=[],
                )
            ]
        )
    )

    connector.start_load_kv(SimpleNamespace())

    assert connector.loaded_requests == ["r2"]
    assert connector.get_block_ids_with_load_errors() == set()


def test_vllm_factory_can_load_noop_connector_by_module_path():
    config = KVTransferConfig(
        kv_connector="M3NoOpConnector",
        kv_role="kv_both",
        kv_connector_module_path="benchmarks.m3.noop_connector",
    )

    cls = KVConnectorFactory.get_connector_class(config)

    assert cls is M3NoOpConnector
