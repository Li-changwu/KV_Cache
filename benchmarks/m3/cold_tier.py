"""Cold-tier storage adapters for M3 KV objects."""

from __future__ import annotations

import hashlib
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


def build_cold_tier_adapter(
    backend: str,
    root: str | Path,
) -> ColdTierAdapter:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in {"local", "local_posix", "posix"}:
        return LocalColdTierAdapter(root)
    if normalized in {"3fs", "3fs_posix", "threefs", "threefs_posix"}:
        return ThreeFSColdTierAdapter(root)
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
