from benchmarks.m3.kv_evicted_index import (
    KVEvictedIndex,
    KVRangeStatus,
)
from benchmarks.m3.tensor_store import KVBlockManifest, KVLayerRecord


def correctness_key(model: str = "qwen2.5") -> dict[str, str]:
    return {
        "model_fingerprint": model,
        "tokenizer_fingerprint": "qwen2.5-tokenizer",
        "rope_config": "native-32768",
        "dtype": "bf16",
        "kv_layout": "vllm-paged",
    }


def manifest(
    *,
    prefix_id: str = "session-a",
    token_start: int = 0,
    token_end: int = 64,
    tier: str = "DRAM",
    ready: bool = True,
    key: dict[str, str] | None = None,
    object_id: str | None = None,
    cold_uri: str | None = None,
    offset_table: dict[str, dict[str, object]] | None = None,
) -> KVBlockManifest:
    return KVBlockManifest(
        prefix_id=prefix_id,
        token_start=token_start,
        token_end=token_end,
        correctness_key=key or correctness_key(),
        token_ids=list(range(token_start, token_end)),
        block_size=16,
        layout="NHD",
        tier=tier,
        ready=ready,
        cold_uri=cold_uri,
        object_id=object_id,
        offset_table=offset_table or {},
        checksum="sha256:object",
        size_bytes=4096,
        layers={
            "layer.0": KVLayerRecord(
                layer_name="layer.0",
                file_name="layer.0.safetensors",
                shape=[1, 2, 16, 8, 128],
                dtype="bfloat16",
            )
        },
    )


def test_ready_cpu_or_hbm_range_is_classified_ready():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(tier="DRAM", ready=True))

    result = index.lookup_required(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )

    assert result.status == KVRangeStatus.CPU_READY
    assert result.entry is not None
    assert result.entry.tier == "DRAM"
    assert result.entry.ready is True
    assert result.cold_extents == []


def test_gpu_cpu_and_ssd_are_classified_as_distinct_tiers():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(prefix_id="gpu", tier="HBM", ready=True))
    index.upsert_from_manifest(manifest(prefix_id="cpu", tier="DRAM", ready=True))
    index.upsert_from_manifest(manifest(prefix_id="ssd", tier="SSD", ready=False))

    gpu = index.lookup_required(
        prefix_id="gpu",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )
    cpu = index.lookup_required(
        prefix_id="cpu",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )
    ssd = index.lookup_required(
        prefix_id="ssd",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )

    assert gpu.status == KVRangeStatus.GPU_READY
    assert cpu.status == KVRangeStatus.CPU_READY
    assert ssd.status == KVRangeStatus.SSD_COLD


def test_ssd_not_ready_range_is_classified_cold_with_extent_locations():
    index = KVEvictedIndex()
    index.upsert_from_manifest(
        manifest(
            tier="NVME",
            ready=False,
            object_id="obj-session-a",
            cold_uri="file:///cold/obj-session-a",
            offset_table={
                "layer.0": {
                    "file_name": "layer.0.safetensors",
                    "offset": 128,
                    "size_bytes": 2048,
                    "checksum": "sha256:layer0",
                }
            },
        )
    )

    result = index.lookup_required(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )

    assert result.status == KVRangeStatus.SSD_COLD
    assert result.entry is not None
    assert result.entry.object_id == "obj-session-a"
    assert result.entry.cold_uri == "file:///cold/obj-session-a"
    assert result.cold_extents[0].layer_name == "layer.0"
    assert result.cold_extents[0].offset == 128
    assert result.cold_extents[0].length == 2048


def test_fetching_range_is_not_requeued_as_cold():
    index = KVEvictedIndex()
    index.upsert_from_manifest(
        manifest(
            tier="SSD",
            ready=False,
            object_id="obj-session-a",
            cold_uri="file:///cold/obj-session-a",
        )
    )

    before = index.mark_fetching(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        request_id="restore-r1",
    )
    after = index.lookup_required(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )

    assert before.status == KVRangeStatus.FETCHING
    assert after.status == KVRangeStatus.FETCHING
    assert after.entry is not None
    assert after.entry.fetching_request_id == "restore-r1"


def test_correctness_key_mismatch_is_distinct_from_missing():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(key=correctness_key(model="old-model")))

    result = index.lookup_required(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(model="new-model"),
    )

    assert result.status == KVRangeStatus.MISMATCH
    assert result.entry is not None
    assert result.entry.correctness_key["model_fingerprint"] == "old-model"


def test_unknown_or_uncovered_range_is_missing():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(prefix_id="session-a", token_end=64))

    unknown = index.lookup_required(
        prefix_id="session-b",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
    )
    uncovered = index.lookup_required(
        prefix_id="session-a",
        token_start=0,
        token_end=128,
        correctness_key=correctness_key(),
    )

    assert unknown.status == KVRangeStatus.MISSING
    assert uncovered.status == KVRangeStatus.MISSING


def test_mark_ready_promotes_cold_or_fetching_range_to_ready_cpu_tier():
    index = KVEvictedIndex()
    index.upsert_from_manifest(
        manifest(
            tier="SSD",
            ready=False,
            object_id="obj-session-a",
            cold_uri="file:///cold/obj-session-a",
        )
    )
    index.mark_fetching(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        request_id="restore-r1",
    )

    result = index.mark_ready(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        target_tier="CPU",
    )

    assert result.status == KVRangeStatus.CPU_READY
    assert result.entry is not None
    assert result.entry.tier == "CPU"
    assert result.entry.ready is True
    assert result.entry.fetching_request_id is None


def test_mark_loading_tracks_cpu_to_gpu_transfer_without_reclassifying_as_cold():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(tier="DRAM", ready=True))

    loading = index.mark_loading(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        request_id="h2d-r1",
    )

    assert loading.status == KVRangeStatus.LOADING
    assert loading.entry is not None
    assert loading.entry.loading_request_id == "h2d-r1"
    assert loading.entry.tier == "LOADING"


def test_mark_ready_after_loading_promotes_range_to_gpu_ready():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(tier="DRAM", ready=True))
    index.mark_loading(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        request_id="h2d-r1",
    )

    ready = index.mark_ready(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        target_tier="HBM",
    )

    assert ready.status == KVRangeStatus.GPU_READY
    assert ready.entry is not None
    assert ready.entry.tier == "HBM"
    assert ready.entry.ready is True
    assert ready.entry.loading_request_id is None


def test_classify_required_groups_gpu_cpu_ssd_fetching_loading_missing_and_mismatch():
    index = KVEvictedIndex()
    index.upsert_from_manifest(manifest(prefix_id="gpu", tier="HBM", ready=True))
    index.upsert_from_manifest(manifest(prefix_id="cpu", tier="DRAM", ready=True))
    index.upsert_from_manifest(manifest(prefix_id="cold", tier="SSD", ready=False))
    index.upsert_from_manifest(manifest(prefix_id="fetching", tier="SSD", ready=False))
    index.mark_fetching(
        prefix_id="fetching",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        request_id="restore-r1",
    )
    index.upsert_from_manifest(manifest(prefix_id="loading", tier="DRAM", ready=True))
    index.mark_loading(
        prefix_id="loading",
        token_start=0,
        token_end=64,
        correctness_key=correctness_key(),
        request_id="h2d-r1",
    )
    index.upsert_from_manifest(
        manifest(prefix_id="mismatch", key=correctness_key(model="old-model"))
    )

    grouped = index.classify_required(
        [
            ("gpu", 0, 64, correctness_key()),
            ("cpu", 0, 64, correctness_key()),
            ("cold", 0, 64, correctness_key()),
            ("fetching", 0, 64, correctness_key()),
            ("loading", 0, 64, correctness_key()),
            ("missing", 0, 64, correctness_key()),
            ("mismatch", 0, 64, correctness_key(model="new-model")),
        ]
    )

    assert [item.prefix_id for item in grouped.gpu_ready] == ["gpu"]
    assert [item.prefix_id for item in grouped.cpu_ready] == ["cpu"]
    assert [item.prefix_id for item in grouped.ssd_cold] == ["cold"]
    assert [item.prefix_id for item in grouped.fetching] == ["fetching"]
    assert [item.prefix_id for item in grouped.loading] == ["loading"]
    assert [item.prefix_id for item in grouped.missing] == ["missing"]
    assert [item.prefix_id for item in grouped.mismatch] == ["mismatch"]
    assert [item.prefix_id for item in grouped.ready] == ["gpu", "cpu"]
    assert [item.prefix_id for item in grouped.cold] == ["cold"]
