"""CPU-resident KV Anti-Cache index for M3.14."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Iterable

from benchmarks.m3.tensor_store import KVBlockManifest


class KVRangeStatus(StrEnum):
    READY = "READY"
    COLD = "COLD"
    FETCHING = "FETCHING"
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"


READY_TIERS = {"HBM", "DRAM", "CPU"}
COLD_TIERS = {"SSD", "NVME", "LOCAL_NVME", "THREE_FS", "3FS", "3FS_POSIX"}


@dataclass(frozen=True)
class KVExtentRef:
    layer_name: str
    token_start: int
    token_end: int
    offset: int
    length: int
    checksum: str = ""
    file_name: str = ""


@dataclass(frozen=True)
class KVIndexEntry:
    prefix_id: str
    token_start: int
    token_end: int
    correctness_key: dict[str, Any]
    tier: str
    ready: bool
    object_id: str | None = None
    cold_uri: str | None = None
    extents: tuple[KVExtentRef, ...] = ()
    checksum: str = ""
    size_bytes: int = 0
    last_access_epoch: int = 0
    access_count: int = 0
    fetching_request_id: str | None = None

    @property
    def tokens(self) -> int:
        return max(0, self.token_end - self.token_start)

    def covers(self, token_start: int, token_end: int) -> bool:
        return self.token_start <= int(token_start) and self.token_end >= int(token_end)


@dataclass(frozen=True)
class KVLookupResult:
    status: KVRangeStatus
    prefix_id: str
    token_start: int
    token_end: int
    entry: KVIndexEntry | None = None
    cold_extents: list[KVExtentRef] = field(default_factory=list)


@dataclass(frozen=True)
class KVClassification:
    ready: tuple[KVLookupResult, ...]
    cold: tuple[KVLookupResult, ...]
    fetching: tuple[KVLookupResult, ...]
    missing: tuple[KVLookupResult, ...]
    mismatch: tuple[KVLookupResult, ...]

    def to_json(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "ready": [_lookup_to_json(item) for item in self.ready],
            "cold": [_lookup_to_json(item) for item in self.cold],
            "fetching": [_lookup_to_json(item) for item in self.fetching],
            "missing": [_lookup_to_json(item) for item in self.missing],
            "mismatch": [_lookup_to_json(item) for item in self.mismatch],
        }


class KVEvictedIndex:
    """In-memory index that classifies historical KV residency before GPU entry."""

    def __init__(self) -> None:
        self._entries: dict[str, list[KVIndexEntry]] = {}
        self._epoch = 0

    def upsert_from_manifest(self, manifest: KVBlockManifest) -> KVIndexEntry:
        return self.upsert(
            prefix_id=manifest.prefix_id,
            token_start=manifest.token_start,
            token_end=manifest.token_end,
            correctness_key=manifest.correctness_key,
            tier=manifest.tier,
            ready=manifest.ready,
            object_id=manifest.object_id,
            cold_uri=manifest.cold_uri,
            extents=_extents_from_manifest(manifest),
            checksum=manifest.checksum,
            size_bytes=manifest.size_bytes,
        )

    def upsert(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
        tier: str,
        ready: bool,
        object_id: str | None = None,
        cold_uri: str | None = None,
        extents: Iterable[KVExtentRef | dict[str, Any]] = (),
        checksum: str = "",
        size_bytes: int = 0,
    ) -> KVIndexEntry:
        entry = KVIndexEntry(
            prefix_id=str(prefix_id),
            token_start=int(token_start),
            token_end=int(token_end),
            correctness_key=dict(correctness_key),
            tier=_normalize_tier(tier),
            ready=bool(ready),
            object_id=object_id,
            cold_uri=cold_uri,
            extents=tuple(_coerce_extent(extent) for extent in extents),
            checksum=str(checksum),
            size_bytes=int(size_bytes),
            last_access_epoch=self._next_epoch(),
        )
        self._append(entry)
        return entry

    def lookup_required(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
    ) -> KVLookupResult:
        token_start = int(token_start)
        token_end = int(token_end)
        entry = self._best_entry(prefix_id, token_start, token_end)
        if entry is None:
            return KVLookupResult(
                status=KVRangeStatus.MISSING,
                prefix_id=prefix_id,
                token_start=token_start,
                token_end=token_end,
            )
        if entry.correctness_key != dict(correctness_key):
            return self._touch_result(
                KVLookupResult(
                    status=KVRangeStatus.MISMATCH,
                    prefix_id=prefix_id,
                    token_start=token_start,
                    token_end=token_end,
                    entry=entry,
                )
            )
        status = _status_for_entry(entry)
        cold_extents = (
            _select_extents(entry, token_start, token_end)
            if status in {KVRangeStatus.COLD, KVRangeStatus.FETCHING}
            else []
        )
        return self._touch_result(
            KVLookupResult(
                status=status,
                prefix_id=prefix_id,
                token_start=token_start,
                token_end=token_end,
                entry=entry,
                cold_extents=cold_extents,
            )
        )

    def classify_required(
        self,
        required: Iterable[
            tuple[str, int, int, dict[str, Any]] | dict[str, Any]
        ],
    ) -> KVClassification:
        buckets: dict[KVRangeStatus, list[KVLookupResult]] = {
            status: [] for status in KVRangeStatus
        }
        for item in required:
            prefix_id, token_start, token_end, correctness_key = _parse_required(item)
            result = self.lookup_required(
                prefix_id=prefix_id,
                token_start=token_start,
                token_end=token_end,
                correctness_key=correctness_key,
            )
            buckets[result.status].append(result)
        return KVClassification(
            ready=tuple(buckets[KVRangeStatus.READY]),
            cold=tuple(buckets[KVRangeStatus.COLD]),
            fetching=tuple(buckets[KVRangeStatus.FETCHING]),
            missing=tuple(buckets[KVRangeStatus.MISSING]),
            mismatch=tuple(buckets[KVRangeStatus.MISMATCH]),
        )

    def mark_fetching(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
        request_id: str,
    ) -> KVLookupResult:
        result = self.lookup_required(
            prefix_id=prefix_id,
            token_start=token_start,
            token_end=token_end,
            correctness_key=correctness_key,
        )
        if result.entry is None or result.status in {
            KVRangeStatus.MISSING,
            KVRangeStatus.MISMATCH,
            KVRangeStatus.READY,
        }:
            return result
        updated = replace(
            result.entry,
            tier="FETCHING",
            ready=False,
            fetching_request_id=str(request_id),
            last_access_epoch=self._next_epoch(),
        )
        self._append(updated)
        return KVLookupResult(
            status=KVRangeStatus.FETCHING,
            prefix_id=prefix_id,
            token_start=int(token_start),
            token_end=int(token_end),
            entry=updated,
            cold_extents=_select_extents(updated, int(token_start), int(token_end)),
        )

    def mark_ready(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
        target_tier: str = "CPU",
    ) -> KVLookupResult:
        result = self.lookup_required(
            prefix_id=prefix_id,
            token_start=token_start,
            token_end=token_end,
            correctness_key=correctness_key,
        )
        if result.entry is None or result.status in {
            KVRangeStatus.MISSING,
            KVRangeStatus.MISMATCH,
        }:
            return result
        updated = replace(
            result.entry,
            tier=_normalize_tier(target_tier),
            ready=True,
            fetching_request_id=None,
            last_access_epoch=self._next_epoch(),
        )
        self._append(updated)
        return KVLookupResult(
            status=KVRangeStatus.READY,
            prefix_id=prefix_id,
            token_start=int(token_start),
            token_end=int(token_end),
            entry=updated,
        )

    def mark_evicted(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
        object_id: str,
        cold_uri: str,
        extents: Iterable[KVExtentRef | dict[str, Any]],
        tier: str = "SSD",
        checksum: str = "",
        size_bytes: int = 0,
    ) -> KVLookupResult:
        current = self.lookup_required(
            prefix_id=prefix_id,
            token_start=token_start,
            token_end=token_end,
            correctness_key=correctness_key,
        )
        base = current.entry
        entry = KVIndexEntry(
            prefix_id=prefix_id,
            token_start=int(token_start),
            token_end=int(token_end),
            correctness_key=dict(correctness_key),
            tier=_normalize_tier(tier),
            ready=False,
            object_id=str(object_id),
            cold_uri=str(cold_uri),
            extents=tuple(_coerce_extent(item) for item in extents),
            checksum=checksum or (base.checksum if base is not None else ""),
            size_bytes=int(size_bytes or (base.size_bytes if base is not None else 0)),
            last_access_epoch=self._next_epoch(),
            access_count=base.access_count if base is not None else 0,
        )
        self._append(entry)
        return KVLookupResult(
            status=KVRangeStatus.COLD,
            prefix_id=prefix_id,
            token_start=int(token_start),
            token_end=int(token_end),
            entry=entry,
            cold_extents=_select_extents(entry, int(token_start), int(token_end)),
        )

    def touch(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
    ) -> KVLookupResult:
        return self.lookup_required(
            prefix_id=prefix_id,
            token_start=token_start,
            token_end=token_end,
            correctness_key=correctness_key,
        )

    def entries_for_prefix(self, prefix_id: str) -> tuple[KVIndexEntry, ...]:
        return tuple(self._entries.get(prefix_id, ()))

    def _append(self, entry: KVIndexEntry) -> None:
        self._entries.setdefault(entry.prefix_id, []).append(entry)

    def _best_entry(
        self,
        prefix_id: str,
        token_start: int,
        token_end: int,
    ) -> KVIndexEntry | None:
        eligible = [
            entry
            for entry in self._entries.get(prefix_id, [])
            if entry.covers(token_start, token_end)
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda entry: (
                entry.last_access_epoch,
                entry.tokens,
                int(entry.ready),
                _tier_rank(entry.tier),
            ),
        )

    def _touch_result(self, result: KVLookupResult) -> KVLookupResult:
        entry = result.entry
        if entry is None:
            return result
        updated = replace(
            entry,
            access_count=entry.access_count + 1,
            last_access_epoch=self._next_epoch(),
        )
        self._append(updated)
        return replace(result, entry=updated)

    def _next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch


def _extents_from_manifest(manifest: KVBlockManifest) -> list[KVExtentRef]:
    extents: list[KVExtentRef] = []
    for layer_name, record in sorted(manifest.offset_table.items()):
        extents.append(
            KVExtentRef(
                layer_name=str(layer_name),
                token_start=manifest.token_start,
                token_end=manifest.token_end,
                offset=int(record.get("offset", 0)),
                length=int(record.get("size_bytes", record.get("length", 0))),
                checksum=str(record.get("checksum", "")),
                file_name=str(record.get("file_name", "")),
            )
        )
    return extents


def _select_extents(
    entry: KVIndexEntry,
    token_start: int,
    token_end: int,
) -> list[KVExtentRef]:
    selected = [
        extent
        for extent in entry.extents
        if extent.token_start <= int(token_start) and extent.token_end >= int(token_end)
    ]
    return list(selected if selected else entry.extents)


def _status_for_entry(entry: KVIndexEntry) -> KVRangeStatus:
    tier = _normalize_tier(entry.tier)
    if tier == "FETCHING" or entry.fetching_request_id:
        return KVRangeStatus.FETCHING
    if entry.ready and tier in READY_TIERS:
        return KVRangeStatus.READY
    if tier in COLD_TIERS or entry.cold_uri or entry.object_id or entry.extents:
        return KVRangeStatus.COLD
    return KVRangeStatus.MISSING


def _parse_required(
    item: tuple[str, int, int, dict[str, Any]] | dict[str, Any],
) -> tuple[str, int, int, dict[str, Any]]:
    if isinstance(item, dict):
        return (
            str(item["prefix_id"]),
            int(item["token_start"]),
            int(item["token_end"]),
            dict(item["correctness_key"]),
        )
    prefix_id, token_start, token_end, correctness_key = item
    return str(prefix_id), int(token_start), int(token_end), dict(correctness_key)


def _coerce_extent(item: KVExtentRef | dict[str, Any]) -> KVExtentRef:
    if isinstance(item, KVExtentRef):
        return item
    return KVExtentRef(
        layer_name=str(item["layer_name"]),
        token_start=int(item["token_start"]),
        token_end=int(item["token_end"]),
        offset=int(item.get("offset", 0)),
        length=int(item.get("length", item.get("size_bytes", 0))),
        checksum=str(item.get("checksum", "")),
        file_name=str(item.get("file_name", "")),
    )


def _lookup_to_json(result: KVLookupResult) -> dict[str, Any]:
    entry = result.entry
    return {
        "status": result.status.value,
        "prefix_id": result.prefix_id,
        "token_start": result.token_start,
        "token_end": result.token_end,
        "tier": entry.tier if entry is not None else "",
        "ready": entry.ready if entry is not None else False,
        "object_id": entry.object_id if entry is not None else "",
        "cold_uri": entry.cold_uri if entry is not None else "",
        "extents": [
            {
                "layer_name": extent.layer_name,
                "token_start": extent.token_start,
                "token_end": extent.token_end,
                "offset": extent.offset,
                "length": extent.length,
                "checksum": extent.checksum,
                "file_name": extent.file_name,
            }
            for extent in result.cold_extents
        ],
    }


def _normalize_tier(tier: str) -> str:
    normalized = str(tier).strip().upper().replace("-", "_")
    if normalized == "LOCAL_SSD":
        return "SSD"
    return normalized


def _tier_rank(tier: str) -> int:
    normalized = _normalize_tier(tier)
    if normalized == "HBM":
        return 5
    if normalized in {"DRAM", "CPU"}:
        return 4
    if normalized == "FETCHING":
        return 3
    if normalized in COLD_TIERS:
        return 2
    return 1
