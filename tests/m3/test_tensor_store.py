from pathlib import Path

import pytest
import torch

from benchmarks.m3.tensor_store import (
    CorrectnessKeyMismatch,
    KVBlockManifest,
    KVTensorStore,
    LayoutMismatch,
    ThreeFSColdTierAdapter,
    TierNotReady,
    TokenIdsMismatch,
    block_slot_mapping,
    copy_blocks_into_layer,
    extract_blocks_from_layer,
)
from benchmarks.m3.cold_tier import (
    LocalColdTierAdapter,
    ThreeFSColdTierAdapter,
    build_cold_tier_adapter,
)


def correctness_key(model: str = "qwen2.5") -> dict[str, str]:
    return {
        "model_fingerprint": model,
        "tokenizer_fingerprint": "qwen2.5-tokenizer",
        "rope_config": "native-32768",
        "dtype": "bf16",
        "kv_layout": "NHD",
    }


def layer_tensor(num_blocks: int = 4, block_size: int = 4) -> torch.Tensor:
    values = torch.arange(num_blocks * 2 * block_size * 2 * 3, dtype=torch.float32)
    return values.reshape(num_blocks, 2, block_size, 2, 3)


def test_block_slot_mapping_requires_block_aligned_token_count():
    assert block_slot_mapping([3, 4], block_size=4, num_tokens=8).tolist() == [
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]

    with pytest.raises(ValueError, match="block aligned"):
        block_slot_mapping([3, 4], block_size=4, num_tokens=7)


def test_extract_and_copy_blocks_round_trip_nhd_layout():
    source = layer_tensor()
    slot_mapping = block_slot_mapping([1, 2], block_size=4, num_tokens=8)
    extracted = extract_blocks_from_layer(source, slot_mapping, block_size=4)

    target = torch.zeros_like(source)
    copy_blocks_into_layer(target, extracted, slot_mapping, block_size=4)

    assert torch.equal(target[1:3], source[1:3])
    assert torch.equal(target[0], torch.zeros_like(target[0]))


def test_tensor_store_writes_manifest_and_loads_matching_key(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    slot_mapping = block_slot_mapping([1, 2], block_size=4, num_tokens=8)

    manifest = store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=8,
        correctness_key=correctness_key(),
        token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=slot_mapping,
        layout="NHD",
    )

    assert isinstance(manifest, KVBlockManifest)
    assert manifest.tokens == 8
    assert (tmp_path / "session-a" / "manifest.json").exists()

    target = torch.zeros_like(source)
    loaded = store.load_layer(
        prefix_id="session-a",
        expected_correctness_key=correctness_key(),
        expected_token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=target,
        slot_mapping=slot_mapping,
        layout="NHD",
    )

    assert loaded.token_end == 8
    assert torch.equal(target[1:3], source[1:3])


def test_tensor_store_tracks_residency_and_blocks_sync_nvme_load(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=8,
        correctness_key=correctness_key(),
        token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1, 2], block_size=4, num_tokens=8),
        layout="NHD",
    )

    assert store.read_manifest("session-a").tier == "DRAM"

    demoted = store.demote_to_nvme("session-a")

    assert demoted.tier == "NVME"
    assert demoted.ready is False
    with pytest.raises(TierNotReady, match="prefetch"):
        store.load_layer(
            prefix_id="session-a",
            expected_correctness_key=correctness_key(),
            expected_token_ids=list(range(8)),
            layer_name="layer.0",
            kv_layer=torch.zeros_like(source),
            slot_mapping=block_slot_mapping([0, 1], block_size=4, num_tokens=8),
            layout="NHD",
        )

    prefetched = store.prefetch_to_dram("session-a")

    assert prefetched.tier == "DRAM"
    assert prefetched.ready is True
    target = torch.zeros_like(source)
    store.load_layer(
        prefix_id="session-a",
        expected_correctness_key=correctness_key(),
        expected_token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=target,
        slot_mapping=block_slot_mapping([0, 1], block_size=4, num_tokens=8),
        layout="NHD",
    )
    assert torch.equal(target[0:2], source[1:3])


def test_tensor_store_moves_layer_files_to_cold_object_and_restores(
    tmp_path: Path,
):
    store = KVTensorStore(tmp_path / "hot", block_size=4)
    source = layer_tensor()
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=8,
        correctness_key=correctness_key(),
        token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1, 2], block_size=4, num_tokens=8),
        layout="NHD",
    )
    hot_tensor_path = tmp_path / "hot" / "session-a" / "layer.0.safetensors"

    demoted = store.demote_to_cold_object(
        "session-a",
        cold_root=tmp_path / "cold",
        tier="LOCAL_NVME",
    )

    assert demoted.tier == "LOCAL_NVME"
    assert demoted.ready is False
    assert demoted.cold_uri
    assert demoted.object_id
    assert demoted.size_bytes > 0
    assert demoted.checksum.startswith("sha256:")
    assert demoted.offset_table["layer.0"]["file_name"] == "layer.0.safetensors"
    assert not hot_tensor_path.exists()
    assert (
        tmp_path
        / "cold"
        / demoted.object_id
        / demoted.layers["layer.0"].file_name
    ).exists()
    with pytest.raises(TierNotReady):
        store.load_layer(
            prefix_id="session-a",
            expected_correctness_key=correctness_key(),
            expected_token_ids=list(range(8)),
            layer_name="layer.0",
            kv_layer=torch.zeros_like(source),
            slot_mapping=block_slot_mapping([0, 1], block_size=4, num_tokens=8),
            layout="NHD",
        )

    restored = store.restore_from_cold_object("session-a")

    assert restored.tier == "DRAM"
    assert restored.ready is True
    assert hot_tensor_path.exists()
    target = torch.zeros_like(source)
    store.load_layer(
        prefix_id="session-a",
        expected_correctness_key=correctness_key(),
        expected_token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=target,
        slot_mapping=block_slot_mapping([0, 1], block_size=4, num_tokens=8),
        layout="NHD",
    )
    assert torch.equal(target[0:2], source[1:3])
    events = store.read_migration_events("session-a")
    assert [event["event"] for event in events] == [
        "demote_to_cold_object",
        "restore_from_cold_object",
    ]
    assert events[0]["bytes"] == demoted.size_bytes
    assert events[1]["checksum_status"] == "ok"


def test_tensor_store_can_use_3fs_posix_cold_tier_adapter(tmp_path: Path):
    store = KVTensorStore(tmp_path / "hot", block_size=4)
    source = layer_tensor()
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=8,
        correctness_key=correctness_key(),
        token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1, 2], block_size=4, num_tokens=8),
        layout="NHD",
    )
    adapter = ThreeFSColdTierAdapter(tmp_path / "threefs_mount")

    demoted = store.demote_to_cold_object(
        "session-a",
        cold_adapter=adapter,
        tier="THREE_FS",
    )
    restored = store.restore_from_cold_object("session-a", cold_adapter=adapter)

    assert demoted.tier == "THREE_FS"
    assert demoted.cold_uri.startswith("file://")
    assert demoted.size_bytes > 0
    assert (tmp_path / "threefs_mount" / "session-a" / "layer.0.safetensors").exists()
    assert restored.tier == "DRAM"
    events = store.read_migration_events("session-a")
    assert events[0]["cold_backend"] == "3fs_posix"
    assert events[1]["cold_backend"] == "3fs_posix"


def test_cold_tier_adapter_factory_selects_local_and_3fs_backends(tmp_path: Path):
    local = build_cold_tier_adapter("local_posix", tmp_path / "local")
    threefs = build_cold_tier_adapter("3fs_posix", tmp_path / "threefs")

    assert isinstance(local, LocalColdTierAdapter)
    assert local.backend == "local_posix"
    assert isinstance(threefs, ThreeFSColdTierAdapter)
    assert threefs.backend == "3fs_posix"


def test_tensor_store_records_migration_events(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=4,
        correctness_key=correctness_key(),
        token_ids=list(range(4)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1], block_size=4, num_tokens=4),
        layout="NHD",
    )

    store.demote_to_nvme("session-a")
    store.prefetch_to_dram("session-a")

    events = store.read_migration_events("session-a")
    assert [event["event"] for event in events] == [
        "demote_to_nvme",
        "prefetch_to_dram",
    ]
    assert events[0]["from_tier"] == "DRAM"
    assert events[0]["to_tier"] == "NVME"
    assert events[0]["tokens"] == 4


def test_tensor_store_loads_prefix_subset_of_saved_tokens(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=8,
        correctness_key=correctness_key(),
        token_ids=list(range(8)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1, 2], block_size=4, num_tokens=8),
        layout="NHD",
    )

    target = torch.zeros_like(source)
    store.load_layer(
        prefix_id="session-a",
        expected_correctness_key=correctness_key(),
        expected_token_ids=list(range(4)),
        layer_name="layer.0",
        kv_layer=target,
        slot_mapping=block_slot_mapping([3], block_size=4, num_tokens=4),
        layout="NHD",
    )

    assert torch.equal(target[3], source[1])


def test_tensor_store_rejects_correctness_key_mismatch(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    slot_mapping = block_slot_mapping([1], block_size=4, num_tokens=4)
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=4,
        correctness_key=correctness_key(model="old-model"),
        token_ids=list(range(4)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=slot_mapping,
        layout="NHD",
    )

    with pytest.raises(CorrectnessKeyMismatch):
        store.load_layer(
            prefix_id="session-a",
            expected_correctness_key=correctness_key(model="new-model"),
            expected_token_ids=list(range(4)),
            layer_name="layer.0",
            kv_layer=torch.zeros_like(source),
            slot_mapping=slot_mapping,
            layout="NHD",
        )


def test_tensor_store_rejects_layout_mismatch(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    slot_mapping = block_slot_mapping([1], block_size=4, num_tokens=4)
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=4,
        correctness_key=correctness_key(),
        token_ids=list(range(4)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=slot_mapping,
        layout="NHD",
    )

    with pytest.raises(LayoutMismatch):
        store.load_layer(
            prefix_id="session-a",
            expected_correctness_key=correctness_key(),
            expected_token_ids=list(range(4)),
            layer_name="layer.0",
            kv_layer=torch.zeros_like(source),
            slot_mapping=slot_mapping,
            layout="HND",
        )


def test_tensor_store_rejects_token_ids_mismatch(tmp_path: Path):
    store = KVTensorStore(tmp_path, block_size=4)
    source = layer_tensor()
    slot_mapping = block_slot_mapping([1], block_size=4, num_tokens=4)
    store.save_layer(
        prefix_id="session-a",
        token_start=0,
        token_end=4,
        correctness_key=correctness_key(),
        token_ids=[1, 2, 3, 4],
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=slot_mapping,
        layout="NHD",
    )

    with pytest.raises(TokenIdsMismatch):
        store.load_layer(
            prefix_id="session-a",
            expected_correctness_key=correctness_key(),
            expected_token_ids=[1, 2, 3, 5],
            layer_name="layer.0",
            kv_layer=torch.zeros_like(source),
            slot_mapping=slot_mapping,
            layout="NHD",
        )
