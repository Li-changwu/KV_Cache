import json
from pathlib import Path

import pytest
import torch

from benchmarks.m3.cold_tier import PackedColdTierAdapter, build_cold_tier_adapter
from benchmarks.m3.tensor_store import (
    ChecksumMismatch,
    KVTensorStore,
    block_slot_mapping,
)


def correctness_key() -> dict[str, str]:
    return {
        "model_fingerprint": "qwen2.5",
        "tokenizer_fingerprint": "qwen2.5-tokenizer",
        "rope_config": "native-32768",
        "dtype": "bf16",
        "kv_layout": "NHD",
    }


def layer_tensor(
    *,
    num_blocks: int = 2,
    block_size: int = 4,
    offset: int = 0,
) -> torch.Tensor:
    values = torch.arange(num_blocks * 2 * block_size * 2 * 3, dtype=torch.float32)
    return (values + offset).reshape(num_blocks, 2, block_size, 2, 3)


def write_two_layer_prefix(root: Path) -> KVTensorStore:
    store = KVTensorStore(root, block_size=4)
    slot_mapping = block_slot_mapping([0, 1], block_size=4, num_tokens=8)
    for layer_idx in range(2):
        store.save_layer(
            prefix_id="session-a",
            token_start=0,
            token_end=8,
            correctness_key=correctness_key(),
            token_ids=list(range(8)),
            layer_name=f"layer.{layer_idx}",
            kv_layer=layer_tensor(offset=layer_idx * 1000),
            slot_mapping=slot_mapping,
            layout="NHD",
        )
    return store


def test_packed_adapter_demotes_layers_into_single_object_and_restores(tmp_path):
    store = write_two_layer_prefix(tmp_path / "hot")
    adapter = PackedColdTierAdapter(tmp_path / "cold")

    demoted = store.demote_to_cold_object(
        "session-a",
        cold_adapter=adapter,
        tier="LOCAL_NVME",
    )

    object_dir = tmp_path / "cold" / demoted.object_id
    assert demoted.cold_uri == (object_dir / "packed_object.bin").resolve().as_uri()
    assert (object_dir / "packed_object.bin").exists()
    assert (object_dir / "packed_manifest.json").exists()
    assert not (object_dir / "layer.0.safetensors").exists()
    assert not (object_dir / "layer.1.safetensors").exists()
    assert not (tmp_path / "hot" / "session-a" / "layer.0.safetensors").exists()
    assert not (tmp_path / "hot" / "session-a" / "layer.1.safetensors").exists()
    assert demoted.offset_table["layer.0"]["file_name"] == "packed_object.bin"
    assert demoted.offset_table["layer.1"]["file_name"] == "packed_object.bin"
    assert demoted.offset_table["layer.0"]["offset"] == 0
    assert demoted.offset_table["layer.1"]["offset"] > 0
    assert demoted.offset_table["layer.0"]["packed_layout"] == "packed_v1"
    assert demoted.offset_table["layer.0"]["copy_mode"] == "streaming"

    packed_manifest = json.loads(
        (object_dir / "packed_manifest.json").read_text(encoding="utf-8")
    )
    assert packed_manifest["layout"] == "packed_v1_layer_major"
    assert packed_manifest["object_id"] == demoted.object_id
    assert packed_manifest["size_bytes"] == demoted.size_bytes
    assert packed_manifest["copy_mode"] == "streaming"

    summarized = adapter.summarize(
        demoted.cold_uri,
        demoted.layers,
        object_id=demoted.object_id or "session-a",
    )
    assert summarized.checksum == demoted.checksum
    assert summarized.offset_table["layer.0"]["checksum_source"] == "packed_manifest"

    restored = store.restore_from_cold_object("session-a", cold_adapter=adapter)

    assert restored.tier == "DRAM"
    assert restored.ready is True
    assert (tmp_path / "hot" / "session-a" / "layer.0.safetensors").exists()
    assert (tmp_path / "hot" / "session-a" / "layer.1.safetensors").exists()
    for layer_idx in range(2):
        target = torch.zeros_like(layer_tensor())
        store.load_layer(
            prefix_id="session-a",
            expected_correctness_key=correctness_key(),
            expected_token_ids=list(range(8)),
            layer_name=f"layer.{layer_idx}",
            kv_layer=target,
            slot_mapping=block_slot_mapping([0, 1], block_size=4, num_tokens=8),
            layout="NHD",
        )
        assert torch.equal(target[0:2], layer_tensor(offset=layer_idx * 1000)[0:2])


def test_packed_adapter_detects_object_checksum_mismatch(tmp_path):
    store = write_two_layer_prefix(tmp_path / "hot")
    adapter = PackedColdTierAdapter(tmp_path / "cold")
    demoted = store.demote_to_cold_object(
        "session-a",
        cold_adapter=adapter,
        tier="LOCAL_NVME",
    )
    packed_path = Path(demoted.cold_uri.removeprefix("file://"))
    packed_path.write_bytes(packed_path.read_bytes() + b"corrupt")

    with pytest.raises(ChecksumMismatch):
        store.restore_from_cold_object("session-a", cold_adapter=adapter)

    manifest = store.read_manifest("session-a")
    assert manifest.tier == "LOCAL_NVME"
    assert manifest.ready is False


def test_packed_adapter_builds_manifest_checksum_from_extent_metadata(
    tmp_path,
    monkeypatch,
):
    store = write_two_layer_prefix(tmp_path / "hot")
    adapter = PackedColdTierAdapter(tmp_path / "cold")
    original_open = Path.open

    def open_without_packed_reread(path, mode="r", *args, **kwargs):
        if Path(path).name == "packed_object.bin" and "r" in mode:
            raise AssertionError(
                "packed object should not be reread after streaming append"
            )
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_without_packed_reread)

    demoted = store.demote_to_cold_object(
        "session-a",
        cold_adapter=adapter,
        tier="LOCAL_NVME",
    )

    assert demoted.checksum.startswith("sha256:")


def test_cold_tier_adapter_factory_selects_packed_v1(tmp_path):
    adapter = build_cold_tier_adapter("packed_v1", tmp_path / "cold")

    assert isinstance(adapter, PackedColdTierAdapter)
    assert adapter.backend == "packed_v1"
