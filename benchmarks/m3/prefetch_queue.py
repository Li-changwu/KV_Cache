"""Bandwidth and deadline aware cold-tier restore queue for M3.10-A."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.m3.cold_tier import ColdTierAdapter, build_cold_tier_adapter
from benchmarks.m3.tensor_store import KVTensorStore


@dataclass(frozen=True)
class PrefetchResult:
    prefix_id: str
    status: str
    tokens: int
    bytes: int
    deadline_ms: float
    queue_depth_before: int
    start_ms: float
    duration_ms: float
    estimated_ready_ms: float
    deadline_miss: bool
    tier: str
    ready: bool
    actual_bytes: int = 0
    executor_elapsed_ms: float = 0.0
    object_id: str = ""
    cold_uri: str = ""
    checksum_status: str = ""


@dataclass(frozen=True)
class _PendingPrefetch:
    result: PrefetchResult


class PrefetchQueue:
    """A deterministic first-fit queue for NVMe-to-DRAM restore."""

    def __init__(
        self,
        *,
        tensor_store: str | Path,
        block_size: int,
        kv_bytes_per_token: int,
        storage_gbps: float,
        event_log_path: str | Path | None = None,
        cold_root: str | Path | None = None,
        cold_backend: str = "local_posix",
        cold_adapter: ColdTierAdapter | None = None,
    ) -> None:
        self.store = KVTensorStore(tensor_store, block_size)
        self.kv_bytes_per_token = int(kv_bytes_per_token)
        self.storage_gbps = float(storage_gbps)
        effective_cold_root = (
            Path(cold_root)
            if cold_root is not None
            else self.store.root.parent / f"{self.store.root.name}_cold"
        )
        self.cold_adapter = cold_adapter or build_cold_tier_adapter(
            cold_backend,
            effective_cold_root,
        )
        self.event_log_path = (
            Path(event_log_path)
            if event_log_path is not None
            else Path(tensor_store) / "prefetch_queue_events.jsonl"
        )
        self._available_at_ms = 0.0
        self._queue_depth = 0
        self._requests_total = 0
        self._completed_total = 0
        self._deadline_miss_total = 0
        self._bytes_total = 0
        self._max_depth = 0
        self._last_estimated_ready_ms = 0.0

    def submit(self, *, prefix_id: str, deadline_ms: float) -> PrefetchResult:
        manifest = self.store.read_manifest(prefix_id)
        bytes_to_transfer = int(manifest.tokens * self.kv_bytes_per_token)
        duration_ms = _transfer_ms(bytes_to_transfer, self.storage_gbps)
        queue_depth_before = self._queue_depth
        start_ms = self._available_at_ms
        estimated_ready_ms = start_ms + duration_ms
        deadline_miss = estimated_ready_ms > float(deadline_ms)
        self._requests_total += 1
        self._bytes_total += bytes_to_transfer
        self._last_estimated_ready_ms = round(estimated_ready_ms, 3)

        if deadline_miss:
            self._deadline_miss_total += 1
            result = PrefetchResult(
                prefix_id=prefix_id,
                status="DEADLINE_MISS",
                tokens=manifest.tokens,
                bytes=bytes_to_transfer,
                actual_bytes=manifest.size_bytes,
                deadline_ms=float(deadline_ms),
                queue_depth_before=queue_depth_before,
                start_ms=round(start_ms, 3),
                duration_ms=round(duration_ms, 3),
                executor_elapsed_ms=0.0,
                estimated_ready_ms=round(estimated_ready_ms, 3),
                deadline_miss=True,
                tier=manifest.tier,
                ready=manifest.ready,
                object_id=manifest.object_id or "",
                cold_uri=manifest.cold_uri or "",
                checksum_status="not_run",
            )
            self._append_event(result)
            return result

        self._queue_depth += 1
        self._max_depth = max(self._max_depth, self._queue_depth)
        event_count_before = len(self.store.read_migration_events(prefix_id))
        prefetched = self._restore_or_metadata_prefetch(prefix_id)
        executor_event = _latest_new_event(
            self.store.read_migration_events(prefix_id),
            event_count_before,
        )
        self._available_at_ms = estimated_ready_ms
        self._queue_depth -= 1
        self._completed_total += 1
        result = PrefetchResult(
            prefix_id=prefix_id,
            status="COMPLETED",
            tokens=manifest.tokens,
            bytes=bytes_to_transfer,
            actual_bytes=int(executor_event.get("bytes", prefetched.size_bytes)),
            deadline_ms=float(deadline_ms),
            queue_depth_before=queue_depth_before,
            start_ms=round(start_ms, 3),
            duration_ms=round(duration_ms, 3),
            executor_elapsed_ms=float(executor_event.get("elapsed_ms", 0.0)),
            estimated_ready_ms=round(estimated_ready_ms, 3),
            deadline_miss=False,
            tier=prefetched.tier,
            ready=prefetched.ready,
            object_id=prefetched.object_id or "",
            cold_uri=prefetched.cold_uri or "",
            checksum_status=str(executor_event.get("checksum_status", "")),
        )
        self._append_event(result)
        return result

    def read_events(self) -> list[dict[str, Any]]:
        if not self.event_log_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.event_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def stats(self) -> dict[str, float | int]:
        return {
            "prefetch_queue_requests_total": self._requests_total,
            "prefetch_queue_completed_total": self._completed_total,
            "prefetch_queue_deadline_miss_total": self._deadline_miss_total,
            "prefetch_queue_bytes_total": self._bytes_total,
            "prefetch_queue_max_depth": self._max_depth,
            "prefetch_queue_last_estimated_ready_ms": self._last_estimated_ready_ms,
            "prefetch_queue_available_at_ms": round(self._available_at_ms, 3),
        }

    def _append_event(self, result: PrefetchResult) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    def _restore_or_metadata_prefetch(self, prefix_id: str):
        manifest = self.store.read_manifest(prefix_id)
        if manifest.cold_uri and manifest.layers:
            return self.store.restore_from_cold_object(
                prefix_id,
                cold_adapter=self.cold_adapter,
                event_name="prefetch_to_dram",
            )
        return self.store.prefetch_to_dram(prefix_id)


class AsyncPrefetchQueue(PrefetchQueue):
    """A deterministic background queue that completes only when advanced."""

    def __init__(
        self,
        *,
        tensor_store: str | Path,
        block_size: int,
        kv_bytes_per_token: int,
        storage_gbps: float,
        event_log_path: str | Path | None = None,
        cold_root: str | Path | None = None,
        cold_backend: str = "local_posix",
        cold_adapter: ColdTierAdapter | None = None,
    ) -> None:
        super().__init__(
            tensor_store=tensor_store,
            block_size=block_size,
            kv_bytes_per_token=kv_bytes_per_token,
            storage_gbps=storage_gbps,
            event_log_path=event_log_path,
            cold_root=cold_root,
            cold_backend=cold_backend,
            cold_adapter=cold_adapter,
        )
        self._pending: list[_PendingPrefetch] = []

    def submit(self, *, prefix_id: str, deadline_ms: float) -> PrefetchResult:
        pending = self._pending_for_prefix(prefix_id)
        if pending is not None:
            result = PrefetchResult(
                prefix_id=pending.result.prefix_id,
                status="ALREADY_QUEUED",
                tokens=pending.result.tokens,
                bytes=pending.result.bytes,
                actual_bytes=pending.result.actual_bytes,
                deadline_ms=float(deadline_ms),
                queue_depth_before=len(self._pending),
                start_ms=pending.result.start_ms,
                duration_ms=pending.result.duration_ms,
                executor_elapsed_ms=0.0,
                estimated_ready_ms=pending.result.estimated_ready_ms,
                deadline_miss=False,
                tier=pending.result.tier,
                ready=pending.result.ready,
                object_id=pending.result.object_id,
                cold_uri=pending.result.cold_uri,
                checksum_status="not_run",
            )
            self._append_event(result)
            return result
        manifest = self.store.read_manifest(prefix_id)
        bytes_to_transfer = int(manifest.tokens * self.kv_bytes_per_token)
        duration_ms = _transfer_ms(bytes_to_transfer, self.storage_gbps)
        queue_depth_before = len(self._pending)
        start_ms = self._available_at_ms
        estimated_ready_ms = start_ms + duration_ms
        deadline_miss = estimated_ready_ms > float(deadline_ms)
        self._requests_total += 1
        self._bytes_total += bytes_to_transfer
        self._last_estimated_ready_ms = round(estimated_ready_ms, 3)

        if deadline_miss:
            self._deadline_miss_total += 1
            result = PrefetchResult(
                prefix_id=prefix_id,
                status="DEADLINE_MISS",
                tokens=manifest.tokens,
                bytes=bytes_to_transfer,
                actual_bytes=manifest.size_bytes,
                deadline_ms=float(deadline_ms),
                queue_depth_before=queue_depth_before,
                start_ms=round(start_ms, 3),
                duration_ms=round(duration_ms, 3),
                executor_elapsed_ms=0.0,
                estimated_ready_ms=round(estimated_ready_ms, 3),
                deadline_miss=True,
                tier=manifest.tier,
                ready=manifest.ready,
                object_id=manifest.object_id or "",
                cold_uri=manifest.cold_uri or "",
                checksum_status="not_run",
            )
            self._append_event(result)
            return result

        result = PrefetchResult(
            prefix_id=prefix_id,
            status="QUEUED",
            tokens=manifest.tokens,
            bytes=bytes_to_transfer,
            actual_bytes=manifest.size_bytes,
            deadline_ms=float(deadline_ms),
            queue_depth_before=queue_depth_before,
            start_ms=round(start_ms, 3),
            duration_ms=round(duration_ms, 3),
            executor_elapsed_ms=0.0,
            estimated_ready_ms=round(estimated_ready_ms, 3),
            deadline_miss=False,
            tier=manifest.tier,
            ready=manifest.ready,
            object_id=manifest.object_id or "",
            cold_uri=manifest.cold_uri or "",
            checksum_status="not_run",
        )
        self._pending.append(_PendingPrefetch(result=result))
        self._available_at_ms = estimated_ready_ms
        self._max_depth = max(self._max_depth, len(self._pending))
        self._append_event(result)
        return result

    def advance_ready(self, *, max_ready_ms: float) -> list[PrefetchResult]:
        completed: list[PrefetchResult] = []
        remaining: list[_PendingPrefetch] = []
        for pending in self._pending:
            if pending.result.estimated_ready_ms <= float(max_ready_ms):
                completed.append(self._complete_pending(pending))
            else:
                remaining.append(pending)
        self._pending = remaining
        return completed

    def advance_all(self) -> list[PrefetchResult]:
        return self.advance_ready(max_ready_ms=float("inf"))

    def stats(self) -> dict[str, float | int]:
        stats = super().stats()
        stats["prefetch_queue_pending_total"] = len(self._pending)
        return stats

    def _pending_for_prefix(self, prefix_id: str) -> _PendingPrefetch | None:
        for pending in self._pending:
            if pending.result.prefix_id == prefix_id:
                return pending
        return None

    def _complete_pending(self, pending: _PendingPrefetch) -> PrefetchResult:
        event_count_before = len(
            self.store.read_migration_events(pending.result.prefix_id)
        )
        prefetched = self._restore_or_metadata_prefetch(pending.result.prefix_id)
        executor_event = _latest_new_event(
            self.store.read_migration_events(pending.result.prefix_id),
            event_count_before,
        )
        self._completed_total += 1
        result = PrefetchResult(
            prefix_id=pending.result.prefix_id,
            status="COMPLETED",
            tokens=pending.result.tokens,
            bytes=pending.result.bytes,
            actual_bytes=int(executor_event.get("bytes", prefetched.size_bytes)),
            deadline_ms=pending.result.deadline_ms,
            queue_depth_before=pending.result.queue_depth_before,
            start_ms=pending.result.start_ms,
            duration_ms=pending.result.duration_ms,
            executor_elapsed_ms=float(executor_event.get("elapsed_ms", 0.0)),
            estimated_ready_ms=pending.result.estimated_ready_ms,
            deadline_miss=False,
            tier=prefetched.tier,
            ready=prefetched.ready,
            object_id=prefetched.object_id or "",
            cold_uri=prefetched.cold_uri or "",
            checksum_status=str(executor_event.get("checksum_status", "")),
        )
        self._append_event(result)
        return result


def _latest_new_event(
    events: list[dict[str, Any]],
    event_count_before: int,
) -> dict[str, Any]:
    if len(events) <= event_count_before:
        return {}
    return events[-1]


def _transfer_ms(bytes_to_transfer: int, gbps: float) -> float:
    if gbps <= 0:
        return float("inf")
    return bytes_to_transfer / (gbps * 1_000_000_000) * 1000.0
