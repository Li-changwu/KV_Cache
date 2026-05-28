"""M3 vLLM KV connector prototype.

M3.6 used this connector as a no-op metadata smoke. M3.7 keeps that safe path
while adding an optional local tensor store for short block-level copy tests.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)

from benchmarks.m3.tensor_store import (
    KVTensorStore,
    KVTensorStoreError,
    block_slot_mapping,
)


@dataclass(frozen=True)
class M3NoOpRequestMeta:
    request_id: str
    reuse_tokens: int
    local_block_ids: list[int]
    operation: str = "load"
    required_ranges: list[dict[str, Any]] = field(default_factory=list)
    correctness_key: dict[str, Any] = field(default_factory=dict)
    prefix_id: str | None = None
    token_start: int = 0
    token_ids: list[int] = field(default_factory=list)


@dataclass
class M3NoOpConnectorMetadata(KVConnectorMetadata):
    requests: list[M3NoOpRequestMeta] = field(default_factory=list)


@dataclass
class M3NoOpConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    loaded_request_ids: list[str] = field(default_factory=list)

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> "M3NoOpConnectorWorkerMetadata":
        assert isinstance(other, M3NoOpConnectorWorkerMetadata)
        return M3NoOpConnectorWorkerMetadata(
            loaded_request_ids=self.loaded_request_ids + other.loaded_request_ids
        )


@dataclass(frozen=True)
class M3ObservedPlan:
    request_id: str
    aligned_reuse_tokens: int
    save_tokens: int
    store_policy: str
    correctness_key: dict[str, Any]
    prefix_id: str | None
    token_start: int
    token_ids: list[int]
    required_ranges: list[dict[str, Any]]


class M3NoOpConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config,
        role: KVConnectorRole,
        kv_cache_config=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.block_size = int(getattr(vllm_config.cache_config, "block_size", 16))
        storage_path = self._kv_transfer_config.get_from_extra_config(
            "m3_tensor_store_path",
            None,
        )
        self.tensor_store = (
            KVTensorStore(storage_path, self.block_size) if storage_path else None
        )
        event_log_path = self._kv_transfer_config.get_from_extra_config(
            "m3_event_log_path",
            None,
        )
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self.kv_layout = self._kv_transfer_config.get_from_extra_config(
            "m3_kv_layout",
            "NHD",
        )
        self.kv_caches: dict[str, Any] = {}
        self.observed_requests: dict[str, M3ObservedPlan] = {}
        self.pending_loads: dict[str, M3NoOpRequestMeta] = {}
        self.pending_saves: dict[str, M3NoOpRequestMeta] = {}
        self.loaded_requests: list[str] = []
        self.load_error_block_ids: set[int] = set()
        self.metrics: Counter[str] = Counter()

    @classmethod
    def for_testing(
        cls,
        block_size: int = 16,
        storage_path: str | Path | None = None,
        event_log_path: str | Path | None = None,
    ) -> "M3NoOpConnector":
        connector = cls.__new__(cls)
        connector._connector_metadata = None
        connector._vllm_config = None
        connector._kv_transfer_config = None
        connector._kv_cache_config = None
        connector._role = KVConnectorRole.SCHEDULER
        connector.block_size = block_size
        connector.tensor_store = (
            KVTensorStore(storage_path, block_size) if storage_path else None
        )
        connector.event_log_path = Path(event_log_path) if event_log_path else None
        connector.kv_layout = "NHD"
        connector.kv_caches = {}
        connector.observed_requests = {}
        connector.pending_loads = {}
        connector.pending_saves = {}
        connector.loaded_requests = []
        connector.load_error_block_ids = set()
        connector.metrics = Counter()
        return connector

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        params = getattr(request, "kv_transfer_params", None) or {}
        if params.get("m3_connector_version") == 1:
            self._observe_request_plan(request, params)
        if not _is_usable_plan(params):
            return 0, False
        reuse_tokens = align_down_to_block_size(
            int(params.get("reuse_tokens", 0)),
            self.block_size,
        )
        if reuse_tokens <= num_computed_tokens:
            return 0, False
        return reuse_tokens - num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        request_id = str(getattr(request, "request_id"))
        plan = self.observed_requests.get(request_id)
        if plan is None:
            return
        block_ids = _extract_block_ids(blocks)
        self.pending_loads[request_id] = M3NoOpRequestMeta(
            request_id=request_id,
            reuse_tokens=num_external_tokens,
            local_block_ids=block_ids,
            operation="load",
            correctness_key=plan.correctness_key,
            prefix_id=plan.prefix_id,
            token_start=plan.token_start,
            token_ids=plan.token_ids[:num_external_tokens],
            required_ranges=plan.required_ranges,
        )

    def build_connector_meta(self, scheduler_output) -> M3NoOpConnectorMetadata:
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", []):
            request_id = str(getattr(new_req, "req_id"))
            plan = self.observed_requests.get(request_id)
            if plan is None or plan.save_tokens <= 0 or not plan.prefix_id:
                continue
            block_ids = _extract_block_ids(getattr(new_req, "block_ids", ()))
            if not block_ids:
                continue
            prompt_token_ids = list(getattr(new_req, "prompt_token_ids", []) or [])
            if not prompt_token_ids:
                prompt_token_ids = plan.token_ids
            save_tokens = min(
                plan.save_tokens,
                align_down_to_block_size(len(prompt_token_ids), self.block_size),
                len(block_ids) * self.block_size,
            )
            if save_tokens <= 0:
                continue
            self.pending_saves[request_id] = M3NoOpRequestMeta(
                request_id=request_id,
                reuse_tokens=save_tokens,
                local_block_ids=block_ids[: save_tokens // self.block_size],
                operation="store",
                correctness_key=plan.correctness_key,
                prefix_id=plan.prefix_id,
                token_start=plan.token_start,
                token_ids=prompt_token_ids[:save_tokens],
                required_ranges=plan.required_ranges,
            )
        metadata = M3NoOpConnectorMetadata(
            requests=list(self.pending_loads.values()) + list(self.pending_saves.values())
        )
        self.pending_loads.clear()
        self.pending_saves.clear()
        return metadata

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.kv_caches = dict(kv_caches)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, M3NoOpConnectorMetadata)
        for item in metadata.requests:
            if item.operation != "load":
                continue
            start = time.perf_counter()
            if self.tensor_store is None or not item.prefix_id:
                self.loaded_requests.append(item.request_id)
                elapsed_ms = _elapsed_ms(start)
                self._record_metric("load_request_ok_total")
                self._record_metric("load_tokens_total", item.reuse_tokens)
                self._emit_event(
                    "load_request",
                    request_id=item.request_id,
                    prefix_id=item.prefix_id,
                    tokens=item.reuse_tokens,
                    block_count=len(item.local_block_ids),
                    layers=0,
                    status="ok",
                    elapsed_ms=elapsed_ms,
                )
                continue
            try:
                layer_count = self._load_request_layers(item)
            except (KVTensorStoreError, OSError, ValueError) as exc:
                self.load_error_block_ids.update(item.local_block_ids)
                elapsed_ms = _elapsed_ms(start)
                self._record_metric("load_request_error_total")
                self._emit_event(
                    "load_request",
                    request_id=item.request_id,
                    prefix_id=item.prefix_id,
                    tokens=item.reuse_tokens,
                    block_count=len(item.local_block_ids),
                    layers=len(self.kv_caches),
                    status="error",
                    error=str(exc),
                    elapsed_ms=elapsed_ms,
                )
                print(
                    f"M3 connector load failed for request {item.request_id}: {exc}",
                    flush=True,
                )
            else:
                self.loaded_requests.append(item.request_id)
                elapsed_ms = _elapsed_ms(start)
                self._record_metric("load_request_ok_total")
                self._record_metric("load_tokens_total", item.reuse_tokens)
                self._emit_event(
                    "load_request",
                    request_id=item.request_id,
                    prefix_id=item.prefix_id,
                    tokens=item.reuse_tokens,
                    block_count=len(item.local_block_ids),
                    layers=layer_count,
                    status="ok",
                    elapsed_ms=elapsed_ms,
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        if self.tensor_store is None:
            return None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, M3NoOpConnectorMetadata)
        for item in metadata.requests:
            if item.operation != "store":
                continue
            if not item.prefix_id:
                continue
            num_tokens = item.reuse_tokens
            slot_mapping = block_slot_mapping(
                item.local_block_ids,
                block_size=self.block_size,
                num_tokens=num_tokens,
                device=kv_layer.device,
            )
            start = time.perf_counter()
            try:
                self.tensor_store.save_layer(
                    prefix_id=item.prefix_id,
                    token_start=item.token_start,
                    token_end=item.token_start + num_tokens,
                    correctness_key=item.correctness_key,
                    token_ids=item.token_ids[:num_tokens],
                    layer_name=layer_name,
                    kv_layer=kv_layer,
                    slot_mapping=slot_mapping,
                    layout=self.kv_layout,
                )
            except (KVTensorStoreError, OSError, ValueError) as exc:
                elapsed_ms = _elapsed_ms(start)
                self._record_metric("store_layer_error_total")
                self._emit_event(
                    "store_layer",
                    request_id=item.request_id,
                    prefix_id=item.prefix_id,
                    layer_name=layer_name,
                    tokens=num_tokens,
                    block_count=len(item.local_block_ids),
                    status="error",
                    error=str(exc),
                    elapsed_ms=elapsed_ms,
                )
                raise
            else:
                elapsed_ms = _elapsed_ms(start)
                self._record_metric("store_layer_ok_total")
                self._record_metric("store_tokens_total", num_tokens)
                self._emit_event(
                    "store_layer",
                    request_id=item.request_id,
                    prefix_id=item.prefix_id,
                    layer_name=layer_name,
                    tokens=num_tokens,
                    block_count=len(item.local_block_ids),
                    status="ok",
                    elapsed_ms=elapsed_ms,
                )
        return None

    def wait_for_save(self) -> None:
        return None

    def build_connector_worker_meta(self) -> M3NoOpConnectorWorkerMetadata:
        return M3NoOpConnectorWorkerMetadata(
            loaded_request_ids=list(self.loaded_requests)
        )

    def request_finished(
        self,
        request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        params = getattr(request, "kv_transfer_params", None)
        if not params:
            return False, None
        if not _store_policy_allows_store(params):
            return False, None
        correctness_key = dict(params.get("correctness_key") or {})
        required_ranges = list(params.get("required_ranges") or [])
        primary_range = _primary_required_range(required_ranges)
        token_ids = list(getattr(request, "prompt_token_ids", []) or [])
        token_end = int(primary_range.get("token_end", len(token_ids))) if primary_range else len(token_ids)
        token_start = int(primary_range.get("token_start", 0)) if primary_range else 0
        save_tokens = align_down_to_block_size(token_end - token_start, self.block_size)
        prefix_id = str(
            params.get("store_prefix_id")
            or (primary_range.get("prefix_id") if primary_range else None)
            or getattr(request, "request_id")
        )
        return False, {
            "m3_noop_connector": {
                "request_id": str(getattr(request, "request_id")),
                "saved_block_ids": list(block_ids),
                "manifest": {
                    "prefix_id": prefix_id,
                    "token_start": token_start,
                    "token_end": token_start + save_tokens,
                    "block_ids": list(block_ids)[: save_tokens // self.block_size],
                },
            }
        }

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self.load_error_block_ids)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config) -> str | None:
        return "NHD"

    def _load_request_layers(self, item: M3NoOpRequestMeta) -> int:
        if self.tensor_store is None:
            return 0
        assert item.prefix_id is not None
        layer_count = 0
        for layer_name, kv_layer in self.kv_caches.items():
            slot_mapping = block_slot_mapping(
                item.local_block_ids,
                block_size=self.block_size,
                num_tokens=item.reuse_tokens,
                device=kv_layer.device,
            )
            self.tensor_store.load_layer(
                prefix_id=item.prefix_id,
                expected_correctness_key=item.correctness_key,
                expected_token_ids=item.token_ids[: item.reuse_tokens],
                layer_name=layer_name,
                kv_layer=kv_layer,
                slot_mapping=slot_mapping,
                layout=self.kv_layout,
            )
            layer_count += 1
        return layer_count

    def _observe_request_plan(self, request, params: dict[str, Any]) -> None:
        request_id = str(getattr(request, "request_id"))
        prompt_token_ids = list(getattr(request, "prompt_token_ids", []) or [])
        required_ranges = list(params.get("required_ranges") or [])
        primary_range = _primary_required_range(required_ranges)
        reuse_tokens = align_down_to_block_size(
            int(params.get("reuse_tokens", 0)),
            self.block_size,
        )
        store_policy = str(params.get("store_policy") or "")
        store_prefix_id = params.get("store_prefix_id")
        if _store_policy_allows_store(params):
            save_tokens = align_down_to_block_size(
                int(params.get("store_token_end", len(prompt_token_ids))),
                self.block_size,
            )
            prefix_id = store_prefix_id
        else:
            save_tokens = 0
            prefix_id = None
        if prefix_id is None and primary_range:
            prefix_id = primary_range.get("prefix_id")
        if prefix_id is None:
            prefix_id = request_id
        effective_store_policy = store_policy or (
            "implicit_store" if save_tokens else "load_only"
        )
        self.observed_requests[request_id] = M3ObservedPlan(
            request_id=request_id,
            aligned_reuse_tokens=reuse_tokens,
            save_tokens=save_tokens,
            store_policy=effective_store_policy,
            correctness_key=dict(params.get("correctness_key") or {}),
            prefix_id=str(prefix_id),
            token_start=0,
            token_ids=prompt_token_ids[: max(reuse_tokens, save_tokens)],
            required_ranges=required_ranges,
        )

    def _record_metric(self, name: str, value: int = 1) -> None:
        self.metrics[name] += int(value)

    def _emit_event(self, event: str, **payload: Any) -> None:
        if self.event_log_path is None:
            return
        record = {
            "event": event,
            "connector": "M3NoOpConnector",
            **payload,
        }
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def align_down_to_block_size(num_tokens: int, block_size: int) -> int:
    if num_tokens <= 0:
        return 0
    return (num_tokens // block_size) * block_size


def _extract_block_ids(blocks) -> list[int]:
    if hasattr(blocks, "get_block_ids"):
        groups = blocks.get_block_ids()
        if groups:
            return list(groups[0])
    if isinstance(blocks, (list, tuple)):
        if blocks and isinstance(blocks[0], (list, tuple)):
            return list(blocks[0])
        return list(blocks)
    return []


def _is_usable_plan(params: dict[str, Any]) -> bool:
    if params.get("m3_connector_version") != 1:
        return False
    if params.get("decision") != "ADMIT":
        return False
    if params.get("sync_ssd_miss_allowed") is not False:
        return False
    return int(params.get("reuse_tokens", 0)) > 0


def _store_policy_allows_store(params: dict[str, Any]) -> bool:
    if not params.get("store_prefix_id"):
        return False
    policy = str(params.get("store_policy") or "store_prefix")
    return policy in {"store_prefix", "writeback"}


def _primary_required_range(ranges: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ranges:
        return None
    return max(
        ranges,
        key=lambda item: int(item.get("token_end", 0)) - int(item.get("token_start", 0)),
    )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
