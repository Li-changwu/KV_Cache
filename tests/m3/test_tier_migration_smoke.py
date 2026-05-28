import json
from pathlib import Path

import torch

from benchmarks.m3.run_tier_migration_smoke import run_tier_migration_smoke
from benchmarks.m3.tensor_store import KVTensorStore, block_slot_mapping


def _write_tensor_layer(path: Path, prefix_id: str) -> None:
    store = KVTensorStore(path, block_size=16)
    source = torch.arange(4 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(
        4, 2, 16, 1, 1
    )
    store.save_layer(
        prefix_id=prefix_id,
        token_start=0,
        token_end=32,
        correctness_key={"model_fingerprint": "model"},
        token_ids=list(range(32)),
        layer_name="layer.0",
        kv_layer=source,
        slot_mapping=block_slot_mapping([1, 2], block_size=16, num_tokens=32),
        layout="NHD",
    )


def test_tier_migration_smoke_demotes_and_prefetches_selected_prefixes(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_tensor_layer(tensor_store, "m3-8-16")
    _write_tensor_layer(tensor_store, "m3-8-64")

    result = run_tier_migration_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        cold_root=tmp_path / "cold",
        prefix_ids=["m3-8-16", "m3-8-64"],
    )

    assert result["status"] == "OK"
    assert result["prefixes"] == 2
    assert result["bytes_total"] > 0
    csv_text = (tmp_path / "result" / "tier_migration_smoke.csv").read_text(
        encoding="utf-8"
    )
    assert "prefix_id,initial_tier,demoted_tier,prefetched_tier" in csv_text
    assert "m3-8-16,DRAM,NVME,DRAM" in csv_text
    rows = list(
        json.loads(line)
        for line in (tensor_store / "m3-8-16" / "migration_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert rows[0]["event"] == "demote_to_cold_object"
    assert rows[0]["bytes"] > 0
    assert rows[1]["event"] == "restore_from_cold_object"
    assert rows[1]["checksum_status"] == "ok"
    assert (tmp_path / "cold" / "m3-8-16" / "layer.0.safetensors").exists()
    assert (tensor_store / "m3-8-16" / "layer.0.safetensors").exists()
    assert (tmp_path / "result" / "tier_migration_report.md").exists()


def test_tier_migration_smoke_can_target_3fs_posix_backend(tmp_path):
    tensor_store = tmp_path / "tensor_store"
    _write_tensor_layer(tensor_store, "m3-8-16")

    result = run_tier_migration_smoke(
        tensor_store=tensor_store,
        result_dir=tmp_path / "result",
        cold_root=tmp_path / "threefs_mount",
        cold_backend="3fs_posix",
        prefix_ids=["m3-8-16"],
    )

    assert result["status"] == "OK"
    assert result["cold_backend"] == "3fs_posix"
    assert (tmp_path / "threefs_mount" / "m3-8-16" / "layer.0.safetensors").exists()
    rows = list(
        json.loads(line)
        for line in (tensor_store / "m3-8-16" / "migration_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert rows[0]["cold_backend"] == "3fs_posix"
    assert rows[1]["cold_backend"] == "3fs_posix"
