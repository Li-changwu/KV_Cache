"""Minimal block-level KV tensor store for the M3.7 connector prototype."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from benchmarks.m3.cold_tier import (
    ColdTierAdapter,
    LocalColdTierAdapter,
    ThreeFSColdTierAdapter,
)


class KVTensorStoreError(RuntimeError):
    """Base error for local KV tensor store failures."""


class CorrectnessKeyMismatch(KVTensorStoreError):
    """Raised when a stored prefix belongs to a different correctness key."""


class LayoutMismatch(KVTensorStoreError):
    """Raised when stored KV layout differs from the requested layout."""


class TokenIdsMismatch(KVTensorStoreError):
    """Raised when stored token ids differ from the requested prefix."""


class TierNotReady(KVTensorStoreError):
    """Raised when requested KV is not in an execution-ready tier."""


class ChecksumMismatch(KVTensorStoreError):
    """Raised when a cold object fails checksum verification."""


@dataclass(frozen=True)
class KVLayerRecord:
    layer_name: str
    file_name: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class KVBlockManifest:
    prefix_id: str
    token_start: int
    token_end: int
    correctness_key: dict[str, Any]
    token_ids: list[int]
    block_size: int
    layout: str
    tier: str = "DRAM"
    ready: bool = True
    cold_uri: str | None = None
    object_id: str | None = None
    offset_table: dict[str, dict[str, Any]] = field(default_factory=dict)
    checksum: str = ""
    size_bytes: int = 0
    layers: dict[str, KVLayerRecord] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return max(0, self.token_end - self.token_start)

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "layers": {
                name: asdict(record) for name, record in sorted(self.layers.items())
            },
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "KVBlockManifest":
        return cls(
            prefix_id=str(payload["prefix_id"]),
            token_start=int(payload["token_start"]),
            token_end=int(payload["token_end"]),
            correctness_key=dict(payload["correctness_key"]),
            token_ids=[int(token_id) for token_id in payload.get("token_ids", [])],
            block_size=int(payload["block_size"]),
            layout=str(payload["layout"]),
            tier=str(payload.get("tier", "DRAM")),
            ready=bool(payload.get("ready", True)),
            cold_uri=(
                str(payload["cold_uri"])
                if payload.get("cold_uri") is not None
                else None
            ),
            object_id=(
                str(payload["object_id"])
                if payload.get("object_id") is not None
                else None
            ),
            offset_table={
                str(name): dict(record)
                for name, record in dict(payload.get("offset_table") or {}).items()
            },
            checksum=str(payload.get("checksum", "")),
            size_bytes=int(payload.get("size_bytes", 0)),
            layers={
                name: KVLayerRecord(
                    layer_name=str(record["layer_name"]),
                    file_name=str(record["file_name"]),
                    shape=[int(dim) for dim in record["shape"]],
                    dtype=str(record["dtype"]),
                )
                for name, record in dict(payload.get("layers") or {}).items()
            },
        )


class KVTensorStore:
    def __init__(self, root: str | Path, block_size: int) -> None:
        self.root = Path(root)
        self.block_size = int(block_size)

    def save_layer(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
        token_ids: list[int] | None = None,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        layout: str,
    ) -> KVBlockManifest:
        num_tokens = int(token_end) - int(token_start)
        _validate_block_aligned(num_tokens, self.block_size)
        prefix_dir = self._prefix_dir(prefix_id)
        prefix_dir.mkdir(parents=True, exist_ok=True)

        existing = self._read_manifest(prefix_id)
        layers = dict(existing.layers) if existing is not None else {}
        kv_cache = extract_blocks_from_layer(kv_layer, slot_mapping, self.block_size)
        file_name = f"{_safe_name(layer_name)}.safetensors"
        safetensors.torch.save_file(
            {"kv_cache": kv_cache.detach().cpu().contiguous()},
            prefix_dir / file_name,
        )
        layers[layer_name] = KVLayerRecord(
            layer_name=layer_name,
            file_name=file_name,
            shape=list(kv_cache.shape),
            dtype=str(kv_cache.dtype).replace("torch.", ""),
        )
        manifest = KVBlockManifest(
            prefix_id=prefix_id,
            token_start=int(token_start),
            token_end=int(token_end),
            correctness_key=dict(correctness_key),
            token_ids=[int(token_id) for token_id in (token_ids or [])],
            block_size=self.block_size,
            layout=layout,
            tier="DRAM",
            ready=True,
            cold_uri=None,
            object_id=None,
            offset_table={},
            checksum="",
            size_bytes=0,
            layers=layers,
        )
        self._write_manifest(manifest)
        return manifest

    def load_layer(
        self,
        *,
        prefix_id: str,
        expected_correctness_key: dict[str, Any],
        expected_token_ids: list[int] | None = None,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        layout: str,
    ) -> KVBlockManifest:
        manifest = self.read_manifest(prefix_id)
        self._ensure_execution_ready(manifest)
        if manifest.correctness_key != dict(expected_correctness_key):
            raise CorrectnessKeyMismatch(
                f"correctness key mismatch for prefix {prefix_id}"
            )
        if manifest.layout != layout:
            raise LayoutMismatch(
                f"layout mismatch for prefix {prefix_id}: "
                f"stored={manifest.layout}, requested={layout}"
            )
        if expected_token_ids is not None and manifest.token_ids:
            expected = [int(token_id) for token_id in expected_token_ids]
            if manifest.token_ids[: len(expected)] != expected:
                raise TokenIdsMismatch(f"token ids mismatch for prefix {prefix_id}")
        record = manifest.layers.get(layer_name)
        if record is None:
            raise FileNotFoundError(
                f"stored prefix {prefix_id} has no KV layer {layer_name}"
            )
        tensor_path = self._prefix_dir(prefix_id) / record.file_name
        src = safetensors.torch.load_file(tensor_path)["kv_cache"].to(
            device=kv_layer.device,
            dtype=kv_layer.dtype,
        )
        src = _truncate_to_slot_count(src, int(slot_mapping.numel()))
        copy_blocks_into_layer(kv_layer, src, slot_mapping, self.block_size)
        return manifest

    def demote_to_nvme(self, prefix_id: str) -> KVBlockManifest:
        manifest = self.read_manifest(prefix_id)
        if manifest.layers and self._all_hot_layer_files_exist(manifest):
            return self.demote_to_cold_object(
                prefix_id,
                cold_root=self._default_cold_root(),
                tier="NVME",
                event_name="demote_to_nvme",
            )
        updated = self._replace_residency(manifest, tier="NVME", ready=False)
        self._write_manifest(updated)
        self._append_migration_event(
            prefix_id=prefix_id,
            event="demote_to_nvme",
            from_tier=manifest.tier,
            to_tier=updated.tier,
            tokens=updated.tokens,
        )
        return updated

    def prefetch_to_dram(self, prefix_id: str) -> KVBlockManifest:
        manifest = self.read_manifest(prefix_id)
        if manifest.cold_uri and manifest.layers:
            return self.restore_from_cold_object(
                prefix_id,
                target_tier="DRAM",
                event_name="prefetch_to_dram",
            )
        updated = self._replace_residency(manifest, tier="DRAM", ready=True)
        self._write_manifest(updated)
        self._append_migration_event(
            prefix_id=prefix_id,
            event="prefetch_to_dram",
            from_tier=manifest.tier,
            to_tier=updated.tier,
            tokens=updated.tokens,
        )
        return updated

    def demote_to_cold_object(
        self,
        prefix_id: str,
        *,
        cold_root: str | Path | None = None,
        cold_adapter: ColdTierAdapter | None = None,
        tier: str = "LOCAL_NVME",
        object_id: str | None = None,
        event_name: str = "demote_to_cold_object",
    ) -> KVBlockManifest:
        manifest = self.read_manifest(prefix_id)
        prefix_dir = self._prefix_dir(prefix_id)
        object_id = object_id or _safe_name(prefix_id)
        adapter = cold_adapter or LocalColdTierAdapter(
            cold_root or self._default_cold_root()
        )
        start = time.perf_counter()

        if manifest.cold_uri and not self._all_hot_layer_files_exist(manifest):
            if adapter.exists(manifest.cold_uri, manifest.layers):
                updated = self._replace_residency(manifest, tier=tier, ready=False)
                self._write_manifest(updated)
                self._append_migration_event(
                    prefix_id=prefix_id,
                    event=event_name,
                    from_tier=manifest.tier,
                    to_tier=updated.tier,
                    tokens=updated.tokens,
                    bytes=updated.size_bytes,
                    cold_uri=updated.cold_uri,
                    object_id=updated.object_id,
                    elapsed_ms=_elapsed_ms(start),
                    checksum=updated.checksum,
                    checksum_status="ok",
                    cold_backend=adapter.backend,
                    reused_cold_object=True,
                )
                return updated

        summary = adapter.demote(
            prefix_dir=prefix_dir,
            object_id=object_id,
            layers=manifest.layers,
        )
        updated = KVBlockManifest(
            prefix_id=manifest.prefix_id,
            token_start=manifest.token_start,
            token_end=manifest.token_end,
            correctness_key=dict(manifest.correctness_key),
            token_ids=list(manifest.token_ids),
            block_size=manifest.block_size,
            layout=manifest.layout,
            tier=tier,
            ready=False,
            cold_uri=summary.uri,
            object_id=summary.object_id,
            offset_table=summary.offset_table,
            checksum=summary.checksum,
            size_bytes=summary.size_bytes,
            layers=dict(manifest.layers),
        )
        self._write_manifest(updated)
        self._append_migration_event(
            prefix_id=prefix_id,
            event=event_name,
            from_tier=manifest.tier,
            to_tier=updated.tier,
            tokens=updated.tokens,
            bytes=updated.size_bytes,
            cold_uri=summary.uri,
            object_id=summary.object_id,
            elapsed_ms=_elapsed_ms(start),
            checksum=updated.checksum,
            checksum_status="ok",
            cold_backend=summary.backend,
        )
        return updated

    def restore_from_cold_object(
        self,
        prefix_id: str,
        *,
        cold_adapter: ColdTierAdapter | None = None,
        target_tier: str = "DRAM",
        event_name: str = "restore_from_cold_object",
    ) -> KVBlockManifest:
        manifest = self.read_manifest(prefix_id)
        if not manifest.cold_uri:
            raise FileNotFoundError(f"prefix {prefix_id} has no cold object URI")
        adapter = cold_adapter or LocalColdTierAdapter(self._default_cold_root())
        start = time.perf_counter()
        try:
            summary = adapter.summarize(
                manifest.cold_uri,
                manifest.layers,
                object_id=manifest.object_id or _safe_name(prefix_id),
            )
            checksum = summary.checksum
            if manifest.checksum and checksum != manifest.checksum:
                raise ChecksumMismatch(
                    f"checksum mismatch for prefix {prefix_id}: "
                    f"stored={manifest.checksum}, actual={checksum}"
                )
            prefix_dir = self._prefix_dir(prefix_id)
            adapter.restore(
                uri=manifest.cold_uri,
                prefix_dir=prefix_dir,
                layers=manifest.layers,
            )
        except Exception as exc:
            self._append_migration_event(
                prefix_id=prefix_id,
                event=event_name,
                from_tier=manifest.tier,
                to_tier=manifest.tier,
                tokens=manifest.tokens,
                bytes=manifest.size_bytes,
                cold_uri=manifest.cold_uri,
                object_id=manifest.object_id,
                elapsed_ms=_elapsed_ms(start),
                checksum=manifest.checksum,
                checksum_status="failed",
                cold_backend=adapter.backend,
                error=str(exc),
            )
            raise

        updated = self._replace_residency(manifest, tier=target_tier, ready=True)
        self._write_manifest(updated)
        self._append_migration_event(
            prefix_id=prefix_id,
            event=event_name,
            from_tier=manifest.tier,
            to_tier=updated.tier,
            tokens=updated.tokens,
            bytes=summary.size_bytes,
            cold_uri=manifest.cold_uri,
            object_id=manifest.object_id,
            elapsed_ms=_elapsed_ms(start),
            checksum=checksum,
            checksum_status="ok",
            cold_backend=summary.backend,
        )
        return updated

    def read_migration_events(self, prefix_id: str) -> list[dict[str, Any]]:
        path = self._prefix_dir(prefix_id) / "migration_events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line))
        return events

    def read_manifest(self, prefix_id: str) -> KVBlockManifest:
        manifest = self._read_manifest(prefix_id)
        if manifest is None:
            raise FileNotFoundError(f"no manifest for prefix {prefix_id}")
        return manifest

    def _read_manifest(self, prefix_id: str) -> KVBlockManifest | None:
        path = self._prefix_dir(prefix_id) / "manifest.json"
        if not path.exists():
            return None
        return KVBlockManifest.from_json(json.loads(path.read_text(encoding="utf-8")))

    def _write_manifest(self, manifest: KVBlockManifest) -> None:
        path = self._prefix_dir(manifest.prefix_id) / "manifest.json"
        path.write_text(
            json.dumps(manifest.to_json(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def _prefix_dir(self, prefix_id: str) -> Path:
        return self.root / _safe_name(prefix_id)

    def _default_cold_root(self) -> Path:
        return self.root.parent / f"{self.root.name}_cold"

    def _all_hot_layer_files_exist(self, manifest: KVBlockManifest) -> bool:
        prefix_dir = self._prefix_dir(manifest.prefix_id)
        return all(
            (prefix_dir / record.file_name).exists()
            for record in manifest.layers.values()
        )

    def _replace_residency(
        self,
        manifest: KVBlockManifest,
        *,
        tier: str,
        ready: bool,
    ) -> KVBlockManifest:
        return KVBlockManifest(
            prefix_id=manifest.prefix_id,
            token_start=manifest.token_start,
            token_end=manifest.token_end,
            correctness_key=dict(manifest.correctness_key),
            token_ids=list(manifest.token_ids),
            block_size=manifest.block_size,
            layout=manifest.layout,
            tier=tier,
            ready=ready,
            cold_uri=manifest.cold_uri,
            object_id=manifest.object_id,
            offset_table=dict(manifest.offset_table),
            checksum=manifest.checksum,
            size_bytes=manifest.size_bytes,
            layers=dict(manifest.layers),
        )

    def _ensure_execution_ready(self, manifest: KVBlockManifest) -> None:
        if manifest.ready and manifest.tier in {"HBM", "DRAM"}:
            return
        raise TierNotReady(
            f"prefix {manifest.prefix_id} is in tier {manifest.tier}; "
            "prefetch to DRAM before loading"
        )

    def _append_migration_event(
        self,
        *,
        prefix_id: str,
        event: str,
        from_tier: str,
        to_tier: str,
        tokens: int,
        **extra: Any,
    ) -> None:
        path = self._prefix_dir(prefix_id) / "migration_events.jsonl"
        record = {
            "event": event,
            "prefix_id": prefix_id,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "tokens": int(tokens),
            **extra,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def block_slot_mapping(
    block_ids: list[int],
    *,
    block_size: int,
    num_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_block_aligned(num_tokens, block_size)
    required_blocks = num_tokens // block_size
    if required_blocks > len(block_ids):
        raise ValueError(
            f"need {required_blocks} blocks for {num_tokens} tokens, "
            f"got {len(block_ids)}"
        )
    block_ids_tensor = torch.tensor(
        block_ids[:required_blocks],
        dtype=torch.long,
        device=device,
    )
    offsets = torch.arange(0, block_size, dtype=torch.long, device=device)
    return (
        block_ids_tensor.reshape(required_blocks, 1) * block_size
        + offsets.reshape(1, block_size)
    ).flatten()


def extract_blocks_from_layer(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    slot_mapping = slot_mapping.to(device=kv_layer.device, dtype=torch.long)
    if _is_block_major_kv_layout(kv_layer):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        return kv_layer[block_idxs, :, offsets, ...].detach().clone()
    if kv_layer.dim() >= 5 and kv_layer.shape[0] == 2:
        num_blocks = kv_layer.shape[1]
        page_size = kv_layer.shape[2]
        return (
            kv_layer.reshape(2, num_blocks * page_size, -1)[:, slot_mapping, :]
            .detach()
            .clone()
        )
    num_blocks = kv_layer.shape[0]
    page_size = kv_layer.shape[1]
    return (
        kv_layer.reshape(num_blocks * page_size, -1)[slot_mapping, :]
        .detach()
        .clone()
    )


def copy_blocks_into_layer(
    kv_layer: torch.Tensor,
    src_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    slot_mapping = slot_mapping.to(device=kv_layer.device, dtype=torch.long)
    src_kv_cache = src_kv_cache.to(device=kv_layer.device, dtype=kv_layer.dtype)
    if _is_block_major_kv_layout(kv_layer):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        kv_layer[block_idxs, :, offsets, ...] = src_kv_cache
        return
    if kv_layer.dim() >= 5 and kv_layer.shape[0] == 2:
        num_blocks = kv_layer.shape[1]
        page_size = kv_layer.shape[2]
        flat = kv_layer.reshape(2, num_blocks * page_size, -1)
        flat[:, slot_mapping, :] = src_kv_cache.reshape(2, len(slot_mapping), -1)
        return
    num_blocks = kv_layer.shape[0]
    page_size = kv_layer.shape[1]
    flat = kv_layer.reshape(num_blocks * page_size, -1)
    flat[slot_mapping, :] = src_kv_cache.reshape(len(slot_mapping), -1)


def _is_block_major_kv_layout(kv_layer: torch.Tensor) -> bool:
    return kv_layer.dim() >= 5 and kv_layer.shape[1] == 2


def _validate_block_aligned(num_tokens: int, block_size: int) -> None:
    if num_tokens <= 0 or num_tokens % block_size != 0:
        raise ValueError(
            f"token count must be positive and block aligned: "
            f"num_tokens={num_tokens}, block_size={block_size}"
        )


def _truncate_to_slot_count(src: torch.Tensor, slot_count: int) -> torch.Tensor:
    if _is_saved_block_major_layout(src):
        return src[:slot_count]
    if src.dim() >= 3 and src.shape[0] == 2:
        return src[:, :slot_count, ...]
    return src[:slot_count]


def _is_saved_block_major_layout(src: torch.Tensor) -> bool:
    return src.dim() >= 4 and src.shape[1] == 2


def _summarize_cold_object(
    cold_dir: Path,
    layers: dict[str, KVLayerRecord],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    offset_table: dict[str, dict[str, Any]] = {}
    total = 0
    for layer_name, record in sorted(layers.items()):
        path = cold_dir / record.file_name
        size = path.stat().st_size
        file_digest = _sha256_file(path)
        offset_table[layer_name] = {
            "file_name": record.file_name,
            "offset": total,
            "size_bytes": size,
            "checksum": f"sha256:{file_digest}",
        }
        digest.update(record.file_name.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(file_digest.encode("ascii"))
        total += size
    return {
        "offset_table": offset_table,
        "checksum": f"sha256:{digest.hexdigest()}",
        "size_bytes": total,
    }


def _all_cold_layer_files_exist(
    cold_dir: Path,
    layers: dict[str, KVLayerRecord],
) -> bool:
    return all((cold_dir / record.file_name).exists() for record in layers.values())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"unsupported cold object URI scheme: {parsed.scheme}")
    return Path(uri)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _safe_name(value: str) -> str:
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9._-]+", text):
        return text
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "unnamed"
