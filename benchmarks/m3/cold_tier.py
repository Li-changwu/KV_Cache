"""Cold-tier storage adapters for M3 KV objects."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from benchmarks.m3.tensor_store import KVLayerRecord


@dataclass(frozen=True)
class ColdObjectSummary:
    uri: str
    object_id: str
    offset_table: dict[str, dict[str, object]]
    checksum: str
    size_bytes: int
    backend: str


class ColdTierAdapter(Protocol):
    backend: str

    def object_path(self, object_id: str) -> Path:
        ...

    def uri_to_path(self, uri: str) -> Path:
        ...

    def exists(self, uri: str, layers: dict[str, KVLayerRecord]) -> bool:
        ...

    def demote(
        self,
        *,
        prefix_dir: Path,
        object_id: str,
        layers: dict[str, KVLayerRecord],
    ) -> ColdObjectSummary:
        ...

    def summarize(
        self,
        uri: str,
        layers: dict[str, KVLayerRecord],
        *,
        object_id: str,
    ) -> ColdObjectSummary:
        ...

    def restore(
        self,
        *,
        uri: str,
        prefix_dir: Path,
        layers: dict[str, KVLayerRecord],
    ) -> None:
        ...


class LocalColdTierAdapter:
    backend = "local_posix"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def object_path(self, object_id: str) -> Path:
        return self.root / object_id

    def uri_to_path(self, uri: str) -> Path:
        return path_from_uri(uri)

    def exists(self, uri: str, layers: dict[str, KVLayerRecord]) -> bool:
        cold_dir = self.uri_to_path(uri)
        return all((cold_dir / record.file_name).exists() for record in layers.values())

    def demote(
        self,
        *,
        prefix_dir: Path,
        object_id: str,
        layers: dict[str, KVLayerRecord],
    ) -> ColdObjectSummary:
        cold_dir = self.object_path(object_id)
        if cold_dir.exists():
            shutil.rmtree(cold_dir)
        cold_dir.mkdir(parents=True, exist_ok=True)
        for record in layers.values():
            src = prefix_dir / record.file_name
            if not src.exists():
                raise FileNotFoundError(f"missing hot KV layer file: {src}")
            shutil.move(str(src), str(cold_dir / record.file_name))
        return summarize_cold_object(
            cold_dir,
            layers,
            object_id=object_id,
            backend=self.backend,
        )

    def summarize(
        self,
        uri: str,
        layers: dict[str, KVLayerRecord],
        *,
        object_id: str,
    ) -> ColdObjectSummary:
        return summarize_cold_object(
            self.uri_to_path(uri),
            layers,
            object_id=object_id,
            backend=self.backend,
        )

    def restore(
        self,
        *,
        uri: str,
        prefix_dir: Path,
        layers: dict[str, KVLayerRecord],
    ) -> None:
        cold_dir = self.uri_to_path(uri)
        prefix_dir.mkdir(parents=True, exist_ok=True)
        for record in layers.values():
            src = cold_dir / record.file_name
            if not src.exists():
                raise FileNotFoundError(f"missing cold KV layer file: {src}")
            shutil.copy2(src, prefix_dir / record.file_name)


class ThreeFSColdTierAdapter(LocalColdTierAdapter):
    """3FS mount adapter using POSIX file semantics as the first integration step."""

    backend = "3fs_posix"


class PackedColdTierAdapter(LocalColdTierAdapter):
    """Local POSIX packed cold object layout for sequential restore experiments."""

    backend = "packed_v1"
    object_file_name = "packed_object.bin"
    manifest_file_name = "packed_manifest.json"
    copy_chunk_bytes = 1024 * 1024

    def exists(self, uri: str, layers: dict[str, KVLayerRecord]) -> bool:
        packed_path = self.uri_to_path(uri)
        manifest_path = packed_path.parent / self.manifest_file_name
        if not packed_path.exists() or not manifest_path.exists():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return all(layer_name in payload.get("offset_table", {}) for layer_name in layers)

    def demote(
        self,
        *,
        prefix_dir: Path,
        object_id: str,
        layers: dict[str, KVLayerRecord],
    ) -> ColdObjectSummary:
        cold_dir = self.object_path(object_id)
        if cold_dir.exists():
            shutil.rmtree(cold_dir)
        cold_dir.mkdir(parents=True, exist_ok=True)
        packed_path = cold_dir / self.object_file_name
        offset_table: dict[str, dict[str, object]] = {}
        total = 0
        with packed_path.open("wb") as out:
            for layer_name, record in sorted(layers.items()):
                src = prefix_dir / record.file_name
                if not src.exists():
                    raise FileNotFoundError(f"missing hot KV layer file: {src}")
                size, file_digest = self._append_file_streaming(src, out)
                offset_table[layer_name] = {
                    "file_name": self.object_file_name,
                    "original_file_name": record.file_name,
                    "offset": total,
                    "size_bytes": size,
                    "length": size,
                    "checksum": f"sha256:{file_digest}",
                    "packed_layout": "packed_v1",
                    "copy_mode": "streaming",
                }
                total += size
        for record in layers.values():
            (prefix_dir / record.file_name).unlink()
        return self._write_and_summarize(
            cold_dir=cold_dir,
            object_id=object_id,
            layers=layers,
            offset_table=offset_table,
        )

    def summarize(
        self,
        uri: str,
        layers: dict[str, KVLayerRecord],
        *,
        object_id: str,
    ) -> ColdObjectSummary:
        packed_path = self.uri_to_path(uri)
        manifest_path = packed_path.parent / self.manifest_file_name
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        offset_table = {
            str(name): dict(record)
            for name, record in dict(payload.get("offset_table") or {}).items()
        }
        expected_size = int(payload.get("size_bytes", 0))
        actual_size = packed_path.stat().st_size
        if expected_size != actual_size:
            return ColdObjectSummary(
                uri=packed_path.resolve().as_uri(),
                object_id=object_id,
                offset_table=offset_table,
                checksum=f"sha256:size-mismatch:{actual_size}",
                size_bytes=actual_size,
                backend=self.backend,
            )
        manifest_offsets = {
            layer_name: {
                **record,
                "checksum_source": "packed_manifest",
            }
            for layer_name, record in offset_table.items()
        }
        return ColdObjectSummary(
            uri=packed_path.resolve().as_uri(),
            object_id=object_id,
            offset_table=manifest_offsets,
            checksum=str(payload.get("checksum", "")),
            size_bytes=actual_size,
            backend=self.backend,
        )

    def restore(
        self,
        *,
        uri: str,
        prefix_dir: Path,
        layers: dict[str, KVLayerRecord],
    ) -> None:
        packed_path = self.uri_to_path(uri)
        manifest_path = packed_path.parent / self.manifest_file_name
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        offset_table = dict(payload.get("offset_table") or {})
        prefix_dir.mkdir(parents=True, exist_ok=True)
        with packed_path.open("rb") as packed:
            for layer_name, record in sorted(layers.items()):
                extent = dict(offset_table[layer_name])
                digest = self._restore_extent_streaming(
                    packed=packed,
                    target=prefix_dir / record.file_name,
                    offset=int(extent["offset"]),
                    length=int(extent["size_bytes"]),
                )
                if f"sha256:{digest}" != str(extent.get("checksum", "")):
                    raise ValueError(f"packed extent checksum mismatch: {layer_name}")

    def _write_and_summarize(
        self,
        *,
        cold_dir: Path,
        object_id: str,
        layers: dict[str, KVLayerRecord],
        offset_table: dict[str, dict[str, object]],
    ) -> ColdObjectSummary:
        packed_path = cold_dir / self.object_file_name
        summary = self._summarize_from_offsets(
            packed_path=packed_path,
            object_id=object_id,
            offset_table=offset_table,
        )
        manifest_payload = {
            "layout": "packed_v1_layer_major",
            "copy_mode": "streaming",
            "object_id": object_id,
            "uri": summary.uri,
            "size_bytes": summary.size_bytes,
            "checksum": summary.checksum,
            "layers": sorted(layers),
            "offset_table": summary.offset_table,
        }
        (cold_dir / self.manifest_file_name).write_text(
            json.dumps(manifest_payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return summary

    def _summarize_from_offsets(
        self,
        *,
        packed_path: Path,
        object_id: str,
        offset_table: dict[str, dict[str, object]],
    ) -> ColdObjectSummary:
        digest = hashlib.sha256()
        size_bytes = packed_path.stat().st_size
        for layer_name, record in sorted(offset_table.items()):
            offset = int(record.get("offset", 0))
            length = int(record.get("size_bytes", record.get("length", 0)))
            file_digest = str(record.get("checksum", "")).removeprefix("sha256:")
            digest.update(str(layer_name).encode("utf-8"))
            digest.update(str(offset).encode("ascii"))
            digest.update(str(length).encode("ascii"))
            digest.update(file_digest.encode("ascii"))
        normalized_offsets = {
            str(name): dict(record) for name, record in sorted(offset_table.items())
        }
        return ColdObjectSummary(
            uri=packed_path.resolve().as_uri(),
            object_id=object_id,
            offset_table=normalized_offsets,
            checksum=f"sha256:{digest.hexdigest()}",
            size_bytes=size_bytes,
            backend=self.backend,
        )

    def _append_file_streaming(self, src: Path, out) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with src.open("rb") as fh:
            for chunk in iter(lambda: fh.read(self.copy_chunk_bytes), b""):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        return size, digest.hexdigest()

    def _restore_extent_streaming(
        self,
        *,
        packed,
        target: Path,
        offset: int,
        length: int,
    ) -> str:
        digest = hashlib.sha256()
        remaining = int(length)
        packed.seek(int(offset))
        with target.open("wb") as out:
            while remaining:
                chunk = packed.read(min(self.copy_chunk_bytes, remaining))
                if not chunk:
                    raise EOFError(
                        f"packed object ended before extent restore completed: {target}"
                    )
                out.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()


def build_cold_tier_adapter(
    backend: str,
    root: str | Path,
) -> ColdTierAdapter:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in {"local", "local_posix", "posix"}:
        return LocalColdTierAdapter(root)
    if normalized in {"3fs", "3fs_posix", "threefs", "threefs_posix"}:
        return ThreeFSColdTierAdapter(root)
    if normalized in {"packed", "packed_v1", "local_packed", "local_packed_v1"}:
        return PackedColdTierAdapter(root)
    raise ValueError(f"unsupported cold-tier backend: {backend}")


def summarize_cold_object(
    cold_dir: Path,
    layers: dict[str, KVLayerRecord],
    *,
    object_id: str,
    backend: str,
) -> ColdObjectSummary:
    digest = hashlib.sha256()
    offset_table: dict[str, dict[str, object]] = {}
    total = 0
    for layer_name, record in sorted(layers.items()):
        path = cold_dir / record.file_name
        size = path.stat().st_size
        file_digest = sha256_file(path)
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
    return ColdObjectSummary(
        uri=cold_dir.resolve().as_uri(),
        object_id=object_id,
        offset_table=offset_table,
        checksum=f"sha256:{digest.hexdigest()}",
        size_bytes=total,
        backend=backend,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"unsupported cold object URI scheme: {parsed.scheme}")
    return Path(uri)
