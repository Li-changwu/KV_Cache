import json
from pathlib import Path

import torch

from benchmarks.m3.prefetch_queue import AsyncPrefetchQueue, PrefetchQueue
from benchmarks.m3.tensor_store import KVTensorStore, block_slot_mapping


def _write_manifest(path: Path, prefix_id: str, token_end: int) -> None:
    prefix_dir = path / prefix_id
    prefix_dir.mkdir(parents=True)
    (prefix_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prefix_id": prefix_id,
                "token_start": 0,
                "token_end": token_end,
                "correctness_key": {"model_fingerprint": "model"},
                "token_ids": list(range(token_end)),
                "block_size": 16,
                "layout": "NHD",
                "tier": "NVME",
                "ready": False,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )


def _write_tensor_layer(path: Path, prefix_id: str = "session-a") -> KVTensorStore:
    store = KVTensorStore(path, block_size=16)
    source = torch.arange(2 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        2, 2, 16, 1, 1
    )
    store.save_layer(
        prefix_id=prefix_id,
        token_start=0,
        token_end=16,
        correctness_key={"model_fingerprint": "model"},
        token_ids=list(range(16)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1], block_size=16, num_tokens=16),
        layout="NHD",
    )
    return store


def test_prefetch_queue_prefetches_when_transfer_fits_deadline(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", token_end=64)
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=8.8,
    )

    result = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert result.status == "COMPLETED"
    assert result.deadline_miss is False
    assert result.estimated_ready_ms <= 12_000.0
    manifest = json.loads(
        (tensor_store / "session-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tier"] == "DRAM"
    assert manifest["ready"] is True
    events = queue.read_events()
    assert events[-1]["status"] == "COMPLETED"
    assert events[-1]["queue_depth_before"] == 0


def test_prefetch_queue_restores_real_cold_object_files(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = _write_tensor_layer(tensor_store)
    demoted = store.demote_to_cold_object(
        "session-a",
        cold_root=tmp_path / "cold",
        tier="LOCAL_NVME",
    )
    hot_tensor = tensor_store / "session-a" / "layer.0.safetensors"
    assert not hot_tensor.exists()
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=8.8,
    )

    result = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert result.status == "COMPLETED"
    assert result.tier == "DRAM"
    assert result.ready is True
    assert result.actual_bytes == demoted.size_bytes
    assert result.executor_elapsed_ms >= 0
    assert result.object_id == demoted.object_id
    assert result.cold_uri == demoted.cold_uri
    assert result.checksum_status == "ok"
    assert hot_tensor.exists()
    manifest = json.loads(
        (tensor_store / "session-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["cold_uri"] == demoted.cold_uri
    assert manifest["checksum"] == demoted.checksum
    events = store.read_migration_events("session-a")
    assert events[-1]["event"] == "prefetch_to_dram"
    assert events[-1]["bytes"] == demoted.size_bytes
    assert events[-1]["checksum_status"] == "ok"
    queue_events = queue.read_events()
    assert queue_events[-1]["actual_bytes"] == demoted.size_bytes
    assert queue_events[-1]["checksum_status"] == "ok"


def test_prefetch_queue_can_restore_from_3fs_posix_backend(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = _write_tensor_layer(tensor_store)
    demoted = store.demote_to_cold_object(
        "session-a",
        cold_root=tmp_path / "threefs_mount",
        tier="NVME",
    )
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=8.8,
        cold_root=tmp_path / "threefs_mount",
        cold_backend="3fs_posix",
    )

    result = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert result.status == "COMPLETED"
    assert result.checksum_status == "ok"
    assert result.cold_uri == demoted.cold_uri
    events = store.read_migration_events("session-a")
    assert events[-1]["cold_backend"] == "3fs_posix"


def test_prefetch_queue_preserves_nvme_when_deadline_would_be_missed(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", token_end=1_000_000)
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=0.001,
    )

    result = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert result.status == "DEADLINE_MISS"
    assert result.deadline_miss is True
    assert result.estimated_ready_ms > 12_000.0
    manifest = json.loads(
        (tensor_store / "session-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tier"] == "NVME"
    assert manifest["ready"] is False
    events = queue.read_events()
    assert events[-1]["status"] == "DEADLINE_MISS"
    assert events[-1]["bytes"] == 196_608_000_000


def test_prefetch_queue_stats_track_multiple_requests_and_virtual_queue(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", token_end=64)
    _write_manifest(tensor_store, "session-b", token_end=128)
    queue = PrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=0.001,
    )

    first = queue.submit(prefix_id="session-a", deadline_ms=60_000.0)
    second = queue.submit(prefix_id="session-b", deadline_ms=60_000.0)

    assert first.status == "COMPLETED"
    assert second.status == "COMPLETED"
    assert second.start_ms == first.estimated_ready_ms
    assert second.estimated_ready_ms > first.estimated_ready_ms

    stats = queue.stats()
    assert stats["prefetch_queue_requests_total"] == 2
    assert stats["prefetch_queue_completed_total"] == 2
    assert stats["prefetch_queue_deadline_miss_total"] == 0
    assert stats["prefetch_queue_bytes_total"] == (64 + 128) * 196_608
    assert stats["prefetch_queue_max_depth"] == 1
    assert stats["prefetch_queue_last_estimated_ready_ms"] == second.estimated_ready_ms


def test_async_prefetch_queue_keeps_manifest_not_ready_until_advanced(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", token_end=64)
    queue = AsyncPrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=8.8,
    )

    queued = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert queued.status == "QUEUED"
    assert queued.deadline_miss is False
    manifest = json.loads(
        (tensor_store / "session-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tier"] == "NVME"
    assert manifest["ready"] is False
    stats = queue.stats()
    assert stats["prefetch_queue_requests_total"] == 1
    assert stats["prefetch_queue_completed_total"] == 0
    assert stats["prefetch_queue_pending_total"] == 1

    completed = queue.advance_ready(max_ready_ms=queued.estimated_ready_ms)

    assert [result.status for result in completed] == ["COMPLETED"]
    manifest = json.loads(
        (tensor_store / "session-a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tier"] == "DRAM"
    assert manifest["ready"] is True
    stats = queue.stats()
    assert stats["prefetch_queue_completed_total"] == 1
    assert stats["prefetch_queue_pending_total"] == 0


def test_async_prefetch_queue_restores_real_cold_object_only_when_advanced(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    store = _write_tensor_layer(tensor_store)
    demoted = store.demote_to_cold_object(
        "session-a",
        cold_root=tmp_path / "cold",
        tier="LOCAL_NVME",
    )
    hot_tensor = tensor_store / "session-a" / "layer.0.safetensors"
    queue = AsyncPrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=8.8,
    )

    queued = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert queued.status == "QUEUED"
    assert queued.actual_bytes == demoted.size_bytes
    assert queued.checksum_status == "not_run"
    assert not hot_tensor.exists()

    completed = queue.advance_ready(max_ready_ms=queued.estimated_ready_ms)

    assert [result.status for result in completed] == ["COMPLETED"]
    assert completed[0].actual_bytes == demoted.size_bytes
    assert completed[0].checksum_status == "ok"
    assert completed[0].executor_elapsed_ms >= 0
    assert hot_tensor.exists()


def test_async_prefetch_queue_deadline_miss_is_not_pending(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_manifest(tensor_store, "session-a", token_end=1_000_000)
    queue = AsyncPrefetchQueue(
        tensor_store=tensor_store,
        block_size=16,
        kv_bytes_per_token=196_608,
        storage_gbps=0.001,
    )

    result = queue.submit(prefix_id="session-a", deadline_ms=12_000.0)

    assert result.status == "DEADLINE_MISS"
    assert queue.advance_all() == []
    stats = queue.stats()
    assert stats["prefetch_queue_deadline_miss_total"] == 1
    assert stats["prefetch_queue_pending_total"] == 0
