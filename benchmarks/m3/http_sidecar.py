"""HTTP sidecar and optional OpenAI-compatible proxy for M3.5."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from benchmarks.m3.control_plane import (
    ConnectorState,
    CorrectnessKey,
    Decision,
    PrefixCandidate,
    ReadyBarrierResult,
    RequiredRange,
    SidecarRequest,
    SidecarResponse,
    TieredKVControlPlane,
)
from benchmarks.m3.kv_evicted_index import (
    KVEvictedIndex,
    KVClassification,
    KVRangeStatus,
)
from benchmarks.m3.prefetch_queue import (
    AsyncPrefetchQueue,
    PrefetchQueue,
    PrefetchResult,
)
from benchmarks.m3.prepass_planner import build_prepass_plan
from benchmarks.m3.tensor_store import KVTensorStore, KVTensorStoreError


@dataclass(frozen=True)
class M3SidecarConfig:
    model_id: str
    hbm_capacity_tokens: int
    dram_capacity_tokens: int
    ssd_capacity_tokens: int
    kv_bytes_per_token: int
    h2d_gbps: float
    storage_gbps: float
    decision_log_path: Path
    upstream_base_url: str | None = None
    default_decode_sla_ms: float = 12_000.0
    default_admission_window_ms: float = 12_000.0
    tokenizer_fingerprint: str = "qwen2.5-tokenizer"
    rope_config: str = "native-32768"
    dtype: str = "bf16"
    kv_layout: str = "vllm-paged"
    inject_kv_transfer_params: bool = False
    tensor_store_path: Path | None = None
    tensor_store_block_size: int = 16
    cold_root: Path | None = None
    cold_backend: str = "local_posix"
    async_prefetch: bool = False


class DecisionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        request: SidecarRequest,
        response: SidecarResponse,
        barrier: ReadyBarrierResult,
        source: str,
    ) -> None:
        record = {
            "source": source,
            "request": _request_to_json(request),
            "response": _response_to_json(response),
            "ready_barrier": _barrier_to_json(barrier),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


class SidecarRuntime:
    def __init__(self, config: M3SidecarConfig) -> None:
        self.config = config
        self.control_plane = TieredKVControlPlane(
            hbm_capacity_tokens=config.hbm_capacity_tokens,
            dram_capacity_tokens=config.dram_capacity_tokens,
            ssd_capacity_tokens=config.ssd_capacity_tokens,
            kv_bytes_per_token=config.kv_bytes_per_token,
            h2d_gbps=config.h2d_gbps,
            storage_gbps=config.storage_gbps,
        )
        self.log = DecisionLog(config.decision_log_path)
        self.responses: dict[str, SidecarResponse] = {}
        self.requests: dict[str, SidecarRequest] = {}
        self.decisions: Counter[tuple[str, str]] = Counter()
        self.prefill_tokens_saved_total = 0
        self.sync_ssd_miss_total = 0
        self.residency_hit_total = 0
        self.prefetch_queued_total = 0
        self.prefetch_deadline_miss_total = 0
        self.prepass_requests_total = 0
        self.prepass_ready_total = 0
        self.tensor_store = (
            KVTensorStore(config.tensor_store_path, config.tensor_store_block_size)
            if config.tensor_store_path
            else None
        )
        self.kv_index = KVEvictedIndex()
        self.prefetch_queue = (
            (AsyncPrefetchQueue if config.async_prefetch else PrefetchQueue)(
                tensor_store=config.tensor_store_path,
                block_size=config.tensor_store_block_size,
                kv_bytes_per_token=config.kv_bytes_per_token,
                storage_gbps=config.storage_gbps,
                event_log_path=config.decision_log_path.parent
                / "prefetch_queue_events.jsonl",
                cold_root=config.cold_root,
                cold_backend=config.cold_backend,
            )
            if config.tensor_store_path
            else None
        )

    def admit_from_payload(
        self,
        payload: dict[str, Any],
        source: str = "admit",
    ) -> dict[str, Any]:
        request = _parse_sidecar_request(payload, self.config)
        response = self.control_plane.admit(request)
        residency_hits = self._apply_residency_hits(request, response)
        if residency_hits:
            response = self.control_plane.admit(request)
        barrier = self.control_plane.ready_barrier(response)
        self.responses[response.request_id] = response
        self.requests[response.request_id] = request
        self._record(request, response, barrier, source)
        if residency_hits:
            self._record_residency_hit_event(
                request=request,
                response=response,
                hits=residency_hits,
            )
        self._maybe_prefetch_to_dram(request, response)
        body = _response_to_json(response)
        body["ready_barrier"] = _barrier_to_json(barrier)
        return body

    def commit_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        response = self.responses.get(request_id)
        if response is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown request_id for commit: {request_id}",
            )
        prefix_id = str(payload["prefix_id"])
        token_end = int(payload["token_end"])
        tier = ConnectorState(str(payload.get("tier", ConnectorState.DRAM.value)))
        ready = bool(payload.get("ready", True))
        self.control_plane.commit_request(
            response,
            prefix_id=prefix_id,
            token_end=token_end,
            tier=tier,
            ready=ready,
        )
        original_request = self.requests.get(request_id)
        if original_request is not None:
            self.kv_index.upsert(
                prefix_id=prefix_id,
                token_start=0,
                token_end=token_end,
                correctness_key=asdict(original_request.correctness_key),
                tier=tier.value,
                ready=ready,
            )
        self._sync_index_from_tensor_store(prefix_id)
        return {"status": "committed", "request_id": request_id}

    def auto_commit_from_connector_manifest(
        self,
        *,
        sidecar_request: dict[str, Any],
        sidecar_response: dict[str, Any],
        upstream_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        manifest = _extract_connector_manifest(upstream_payload)
        if manifest is None:
            return None
        request_id = str(sidecar_response["request_id"])
        prefix_id = str(manifest["prefix_id"])
        token_end = int(manifest["token_end"])
        response = self.responses.get(request_id)
        if response is None:
            return None
        self.control_plane.commit_request(
            response,
            prefix_id=prefix_id,
            token_end=token_end,
            tier=ConnectorState.DRAM,
        )
        original_request = self.requests.get(request_id)
        if original_request is not None:
            self.kv_index.upsert(
                prefix_id=prefix_id,
                token_start=0,
                token_end=token_end,
                correctness_key=asdict(original_request.correctness_key),
                tier=ConnectorState.DRAM.value,
                ready=True,
            )
        self._sync_index_from_tensor_store(prefix_id)
        self._record_commit_event(
            sidecar_request=sidecar_request,
            sidecar_response=sidecar_response,
            manifest=manifest,
            source="proxy_auto_commit",
        )
        return {
            "status": "committed",
            "request_id": request_id,
            "prefix_id": prefix_id,
            "token_end": token_end,
        }

    def ready_barrier_for_request(self, request_id: str) -> dict[str, Any]:
        response = self.responses.get(request_id)
        if response is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown request_id for ready barrier: {request_id}",
            )
        return _barrier_to_json(self.control_plane.ready_barrier(response))

    def prepass_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = _parse_sidecar_request(payload, self.config)
        response = self.control_plane.admit(request)
        residency_hits = self._apply_residency_hits(request, response)
        if residency_hits:
            response = self.control_plane.admit(request)
        classification = self._classify_prepass_request(request)
        restore_results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        if response.decision == Decision.DELAY:
            for required in response.required_ranges:
                hit = self._try_residency_hit(request, required, count_metric=False)
                if hit is not None:
                    residency_hits.append(hit)
                    continue
                source_error = self._prefetch_source_error(request, required)
                if source_error is not None:
                    errors.append(
                        {"prefix_id": required.prefix_id, "error": source_error}
                    )
                    continue
                if self.prefetch_queue is None:
                    errors.append(
                        {
                            "prefix_id": required.prefix_id,
                            "error": "prepass requires tensor_store_path",
                        }
                    )
                    continue
                try:
                    result = self.prefetch_queue.submit(
                        prefix_id=required.prefix_id,
                        deadline_ms=float(
                            payload.get("prepass_deadline_ms", required.deadline_ms)
                        ),
                    )
                except (KVTensorStoreError, OSError, ValueError) as exc:
                    errors.append({"prefix_id": required.prefix_id, "error": str(exc)})
                    continue
                result_json = _prefetch_result_to_json(result)
                restore_results.append(result_json)
                if result.status in {"COMPLETED", "QUEUED"}:
                    self.prefetch_queued_total += 1
                if result.deadline_miss:
                    self.prefetch_deadline_miss_total += 1
                if result.status == "COMPLETED":
                    self.control_plane.mark_prefetched([required])
                    self._sync_index_from_tensor_store(required.prefix_id)
                elif result.status in {"QUEUED", "ALREADY_QUEUED"}:
                    self.kv_index.mark_fetching(
                        prefix_id=required.prefix_id,
                        token_start=required.token_start,
                        token_end=required.token_end,
                        correctness_key=asdict(request.correctness_key),
                        request_id=request.request_id,
                    )
        final_response = self.control_plane.admit(request)
        barrier = self.control_plane.ready_barrier(final_response)
        self.responses[final_response.request_id] = final_response
        self.requests[final_response.request_id] = request
        plan = build_prepass_plan(
            required_count=len(response.required_ranges),
            ready_count=sum(1 for required in response.required_ranges if self.control_plane.connector.is_ready(required)),
            restore_results=restore_results,
            errors=errors,
        )
        self.prepass_requests_total += 1
        if plan.ready_before_request:
            self.prepass_ready_total += 1
        body = {
            **plan.to_json(),
            "request_id": request.request_id,
            "decision_after_prepass": final_response.decision.value,
            "reason_after_prepass": final_response.reason,
            "reuse_tokens": final_response.reuse_tokens,
            "delta_prefill_tokens": final_response.delta_prefill_tokens,
            "required_ranges": [
                _range_to_json(item) for item in response.required_ranges
            ],
            "classification": classification.to_json(),
            "classification_counts": _classification_counts(classification),
            "restore_results": restore_results,
            "errors": errors,
            "ready_barrier": _barrier_to_json(barrier),
        }
        self._record_prepass_event(
            request=request,
            response=final_response,
            prepass_body=body,
        )
        return body

    def advance_prefetch_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(self.prefetch_queue, AsyncPrefetchQueue):
            return {"completed": [], "async_prefetch": False}
        max_ready_ms = float(payload.get("max_ready_ms", float("inf")))
        completed = self.prefetch_queue.advance_ready(max_ready_ms=max_ready_ms)
        completed_json = [_prefetch_result_to_json(result) for result in completed]
        required = [
            RequiredRange(
                prefix_id=result.prefix_id,
                token_start=0,
                token_end=result.tokens,
                required_tier="DRAM_OR_HBM",
                deadline_ms=result.deadline_ms,
            )
            for result in completed
        ]
        self.control_plane.mark_prefetched(required)
        for result in completed:
            self._sync_index_from_tensor_store(result.prefix_id)
        self._record_async_prefetch_advance(completed_json)
        return {"completed": completed_json, "async_prefetch": True}

    def metrics_text(self) -> str:
        lines = [
            "# TYPE admission_decision_total counter",
        ]
        for (decision, reason), count in sorted(self.decisions.items()):
            lines.append(
                'admission_decision_total'
                f'{{decision="{decision}",reason="{reason}"}} {count}'
            )
        lines.extend(
            [
                "# TYPE prefill_tokens_saved_total counter",
                f"prefill_tokens_saved_total {self.prefill_tokens_saved_total}",
                "# TYPE sync_ssd_miss_total counter",
                f"sync_ssd_miss_total {self.sync_ssd_miss_total}",
                "# TYPE residency_hit_total counter",
                f"residency_hit_total {self.residency_hit_total}",
                "# TYPE prefetch_queued_total counter",
                f"prefetch_queued_total {self.prefetch_queued_total}",
                "# TYPE prefetch_deadline_miss_total counter",
                f"prefetch_deadline_miss_total {self.prefetch_deadline_miss_total}",
                "# TYPE prepass_requests_total counter",
                f"prepass_requests_total {self.prepass_requests_total}",
                "# TYPE prepass_ready_total counter",
                f"prepass_ready_total {self.prepass_ready_total}",
            ]
        )
        if self.prefetch_queue is not None:
            for name, value in self.prefetch_queue.stats().items():
                metric_type = "counter" if name.endswith("_total") else "gauge"
                lines.append(f"# TYPE {name} {metric_type}")
                lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"

    def _record(
        self,
        request: SidecarRequest,
        response: SidecarResponse,
        barrier: ReadyBarrierResult,
        source: str,
    ) -> None:
        self.decisions[(response.decision.value, response.reason)] += 1
        self.prefill_tokens_saved_total += response.reuse_tokens
        self.sync_ssd_miss_total += barrier.sync_ssd_miss_total
        self.log.append(request, response, barrier, source)

    def _classify_prepass_request(
        self,
        request: SidecarRequest,
    ) -> KVClassification:
        required = []
        for candidate in request.prefix_candidates:
            if not candidate.committed:
                continue
            self._sync_index_from_control_plane(
                prefix_id=candidate.prefix_id,
                token_start=candidate.token_start,
                token_end=candidate.token_end,
            )
            self._sync_index_from_tensor_store(candidate.prefix_id)
            required.append(
                (
                    candidate.prefix_id,
                    candidate.token_start,
                    candidate.token_end,
                    asdict(request.correctness_key),
                )
            )
        return self.kv_index.classify_required(required)

    def _sync_index_from_control_plane(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
    ) -> None:
        entry = self.control_plane.manifest.lookup(
            PrefixCandidate(
                prefix_id=prefix_id,
                token_start=token_start,
                token_end=token_end,
                committed=True,
            )
        )
        if entry is None:
            return
        if not entry.ready and self._index_is_fetching(
            prefix_id=entry.prefix_id,
            token_start=entry.token_start,
            token_end=entry.token_end,
            correctness_key=asdict(entry.correctness_key),
        ):
            return
        self.kv_index.upsert(
            prefix_id=entry.prefix_id,
            token_start=entry.token_start,
            token_end=entry.token_end,
            correctness_key=asdict(entry.correctness_key),
            tier=entry.tier.value,
            ready=entry.ready,
        )

    def _sync_index_from_tensor_store(self, prefix_id: str) -> None:
        if self.tensor_store is None:
            return
        try:
            manifest = self.tensor_store.read_manifest(prefix_id)
        except FileNotFoundError:
            return
        if not manifest.ready and self._index_is_fetching(
            prefix_id=manifest.prefix_id,
            token_start=manifest.token_start,
            token_end=manifest.token_end,
            correctness_key=manifest.correctness_key,
        ):
            return
        self.kv_index.upsert_from_manifest(manifest)

    def _index_is_fetching(
        self,
        *,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: dict[str, Any],
    ) -> bool:
        result = self.kv_index.lookup_required(
            prefix_id=prefix_id,
            token_start=token_start,
            token_end=token_end,
            correctness_key=correctness_key,
        )
        return result.status == KVRangeStatus.FETCHING

    def _record_commit_event(
        self,
        *,
        sidecar_request: dict[str, Any],
        sidecar_response: dict[str, Any],
        manifest: dict[str, Any],
        source: str,
    ) -> None:
        record = {
            "source": source,
            "request": dict(sidecar_request),
            "response": dict(sidecar_response),
            "connector_manifest": dict(manifest),
        }
        self.log.path.parent.mkdir(parents=True, exist_ok=True)
        with self.log.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _maybe_prefetch_to_dram(
        self,
        request: SidecarRequest,
        response: SidecarResponse,
    ) -> None:
        if self.prefetch_queue is None:
            return
        if response.decision != Decision.DELAY:
            return
        if response.reason not in {
            "required_kv_not_ready_before_decode",
            "prefetch_misses_admission_window",
        }:
            return
        prefetched: list[dict[str, Any]] = []
        residency_hits: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        prefetch_results: list[dict[str, Any]] = []
        for required in response.required_ranges:
            hit = self._try_residency_hit(request, required, count_metric=False)
            if hit is not None:
                residency_hits.append(hit)
                continue
            source_error = self._prefetch_source_error(request, required)
            if source_error is not None:
                errors.append({"prefix_id": required.prefix_id, "error": source_error})
                continue
            try:
                result = self.prefetch_queue.submit(
                    prefix_id=required.prefix_id,
                    deadline_ms=required.deadline_ms,
                )
            except (KVTensorStoreError, OSError, ValueError) as exc:
                errors.append({"prefix_id": required.prefix_id, "error": str(exc)})
                continue
            prefetch_results.append(_prefetch_result_to_json(result))
            if result.status in {"COMPLETED", "QUEUED"}:
                self.prefetch_queued_total += 1
            if result.deadline_miss:
                self.prefetch_deadline_miss_total += 1
            if result.status != "COMPLETED":
                continue
            self.control_plane.mark_prefetched([required])
            prefetched.append(
                {
                    "prefix_id": required.prefix_id,
                    "token_start": required.token_start,
                    "token_end": required.token_end,
                    "tier": result.tier,
                    "ready": result.ready,
                }
            )
        if residency_hits:
            self._record_residency_hit_event(
                request=request,
                response=response,
                hits=residency_hits,
            )
        if prefetched or errors or prefetch_results:
            self._record_prefetch_event(
                request=request,
                response=response,
                prefetched=prefetched,
                errors=errors,
                prefetch_results=prefetch_results,
            )

    def _apply_residency_hits(
        self,
        request: SidecarRequest,
        response: SidecarResponse,
    ) -> list[dict[str, Any]]:
        if response.decision != Decision.DELAY:
            return []
        if response.reason not in {
            "required_kv_not_ready_before_decode",
            "prefetch_misses_admission_window",
        }:
            return []
        hits: list[dict[str, Any]] = []
        for required in response.required_ranges:
            hit = self._try_residency_hit(request, required)
            if hit is not None:
                hits.append(hit)
        return hits

    def _try_residency_hit(
        self,
        request: SidecarRequest,
        required: RequiredRange,
        *,
        count_metric: bool = True,
    ) -> dict[str, Any] | None:
        if self.tensor_store is None:
            return None
        try:
            manifest = self.tensor_store.read_manifest(required.prefix_id)
        except FileNotFoundError:
            return None
        if manifest.token_start > required.token_start:
            return None
        if manifest.token_end < required.token_end:
            return None
        if manifest.correctness_key != asdict(request.correctness_key):
            return None
        if not manifest.ready or manifest.tier not in {"HBM", "DRAM"}:
            return None
        self.control_plane.mark_prefetched([required])
        if count_metric:
            self.residency_hit_total += 1
        return {
            "prefix_id": required.prefix_id,
            "token_start": required.token_start,
            "token_end": required.token_end,
            "tier": manifest.tier,
            "ready": manifest.ready,
        }

    def _prefetch_source_error(
        self,
        request: SidecarRequest,
        required: RequiredRange,
    ) -> str | None:
        if self.tensor_store is None:
            return None
        try:
            manifest = self.tensor_store.read_manifest(required.prefix_id)
        except FileNotFoundError:
            return None
        if manifest.correctness_key != asdict(request.correctness_key):
            return f"correctness key mismatch for prefix {required.prefix_id}"
        if (
            manifest.token_start > required.token_start
            or manifest.token_end < required.token_end
        ):
            return (
                f"tensor manifest for prefix {required.prefix_id} does not cover "
                f"required range {required.token_start}:{required.token_end}"
            )
        return None

    def _record_residency_hit_event(
        self,
        *,
        request: SidecarRequest,
        response: SidecarResponse,
        hits: list[dict[str, Any]],
    ) -> None:
        record = {
            "source": "residency_hit",
            "request": _request_to_json(request),
            "response": _response_to_json(response),
            "hits": hits,
        }
        self.log.path.parent.mkdir(parents=True, exist_ok=True)
        with self.log.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_prefetch_event(
        self,
        *,
        request: SidecarRequest,
        response: SidecarResponse,
        prefetched: list[dict[str, Any]],
        errors: list[dict[str, str]],
        prefetch_results: list[dict[str, Any]],
    ) -> None:
        record = {
            "source": "prefetch_to_dram",
            "request": _request_to_json(request),
            "response": _response_to_json(response),
            "prefetched": prefetched,
            "prefetch_results": prefetch_results,
            "errors": errors,
        }
        self.log.path.parent.mkdir(parents=True, exist_ok=True)
        with self.log.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_async_prefetch_advance(
        self,
        completed: list[dict[str, Any]],
    ) -> None:
        if not completed:
            return
        record = {
            "source": "async_prefetch_advance",
            "completed": completed,
        }
        self.log.path.parent.mkdir(parents=True, exist_ok=True)
        with self.log.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_prepass_event(
        self,
        *,
        request: SidecarRequest,
        response: SidecarResponse,
        prepass_body: dict[str, Any],
    ) -> None:
        record = {
            "source": "prepass_result",
            "request": _request_to_json(request),
            "response": _response_to_json(response),
            "prepass": dict(prepass_body),
        }
        self.log.path.parent.mkdir(parents=True, exist_ok=True)
        with self.log.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def create_app(config: M3SidecarConfig) -> FastAPI:
    runtime = SidecarRuntime(config)
    app = FastAPI(title="M3 Tiered KV Sidecar", version="0.1")
    app.state.runtime = runtime

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_id": config.model_id}

    @app.post("/admit")
    async def admit(request: Request) -> dict[str, Any]:
        return runtime.admit_from_payload(await request.json(), source="admit")

    @app.post("/commit")
    async def commit(request: Request) -> dict[str, Any]:
        return runtime.commit_from_payload(await request.json())

    @app.get("/ready_barrier/{request_id}")
    def ready_barrier(request_id: str) -> dict[str, Any]:
        return runtime.ready_barrier_for_request(request_id)

    @app.post("/prepass")
    async def prepass(request: Request) -> dict[str, Any]:
        return runtime.prepass_from_payload(await request.json())

    @app.post("/prefetch/advance")
    async def advance_prefetch(request: Request) -> dict[str, Any]:
        return runtime.advance_prefetch_from_payload(await request.json())

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(runtime.metrics_text(), media_type="text/plain")

    @app.post("/v1/completions", response_model=None)
    async def proxy_completions(request: Request) -> Any:
        body = await request.json()
        sidecar_body = body.get("m3_control")
        if not isinstance(sidecar_body, dict):
            sidecar_body = _infer_control_payload(body, config)
        sidecar_result = runtime.admit_from_payload(sidecar_body, source="proxy")
        if not config.upstream_base_url:
            payload = {
                "proxied": False,
                "reason": "upstream_not_configured",
                "sidecar": sidecar_result,
            }
            return Response(
                content=json.dumps(payload),
                status_code=202,
                media_type="application/json",
            )
        forwarded = dict(body)
        forwarded.pop("m3_control", None)
        if config.inject_kv_transfer_params:
            forwarded["kv_transfer_params"] = build_kv_transfer_params(
                sidecar_request=sidecar_body,
                sidecar_response=sidecar_result,
            )
        upstream = config.upstream_base_url.rstrip("/") + "/v1/completions"
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            upstream_response = await client.post(
                upstream,
                json=forwarded,
                headers=_forward_headers(request),
            )
        if upstream_response.headers.get("content-type", "").startswith(
            "application/json"
        ):
            try:
                upstream_payload = upstream_response.json()
            except ValueError:
                upstream_payload = None
            if isinstance(upstream_payload, dict):
                runtime.auto_commit_from_connector_manifest(
                    sidecar_request=sidecar_body,
                    sidecar_response=sidecar_result,
                    upstream_payload=upstream_payload,
                )
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get(
                "content-type",
                "application/json",
            ),
        )

    return app


def _parse_sidecar_request(
    payload: dict[str, Any],
    config: M3SidecarConfig,
) -> SidecarRequest:
    key_payload = payload.get("correctness_key") or {}
    correctness_key = CorrectnessKey(
        model_fingerprint=str(
            key_payload.get("model_fingerprint", payload.get("model_id", config.model_id))
        ),
        tokenizer_fingerprint=str(
            key_payload.get("tokenizer_fingerprint", config.tokenizer_fingerprint)
        ),
        rope_config=str(key_payload.get("rope_config", config.rope_config)),
        dtype=str(key_payload.get("dtype", config.dtype)),
        kv_layout=str(key_payload.get("kv_layout", config.kv_layout)),
    )
    candidates = [
        PrefixCandidate(
            prefix_id=str(candidate["prefix_id"]),
            token_start=int(candidate.get("token_start", 0)),
            token_end=int(candidate["token_end"]),
            committed=bool(candidate.get("committed", True)),
        )
        for candidate in payload.get("prefix_candidates", [])
    ]
    return SidecarRequest(
        request_id=str(payload.get("request_id") or uuid4()),
        model_id=str(payload.get("model_id", config.model_id)),
        token_count=int(payload["token_count"]),
        decode_sla_ms=float(payload.get("decode_sla_ms", config.default_decode_sla_ms)),
        admission_window_ms=float(
            payload.get("admission_window_ms", config.default_admission_window_ms)
        ),
        correctness_key=correctness_key,
        prefix_candidates=candidates,
    )


def _infer_control_payload(body: dict[str, Any], config: M3SidecarConfig) -> dict[str, Any]:
    prompt = body.get("prompt", "")
    token_count = int(body.get("m3_token_count") or _rough_token_count(prompt))
    return {
        "request_id": str(body.get("request_id") or uuid4()),
        "model_id": str(body.get("model", config.model_id)),
        "token_count": token_count,
        "decode_sla_ms": config.default_decode_sla_ms,
        "admission_window_ms": config.default_admission_window_ms,
        "correctness_key": {
            "model_fingerprint": str(body.get("model", config.model_id)),
            "tokenizer_fingerprint": config.tokenizer_fingerprint,
            "rope_config": config.rope_config,
            "dtype": config.dtype,
            "kv_layout": config.kv_layout,
        },
        "prefix_candidates": body.get("m3_prefix_candidates", []),
    }


def _rough_token_count(prompt: Any) -> int:
    if isinstance(prompt, list):
        return max(1, sum(_rough_token_count(item) for item in prompt))
    text = str(prompt)
    return max(1, len(text.split()))


def _forward_headers(request: Request) -> dict[str, str]:
    blocked = {"host", "content-length", "connection"}
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked
    }


def build_kv_transfer_params(
    sidecar_request: dict[str, Any],
    sidecar_response: dict[str, Any],
) -> dict[str, Any]:
    """Translate sidecar admission output into vLLM kv_transfer_params."""
    explicit_store_prefix_id = sidecar_request.get("store_prefix_id")
    store_policy = "store_prefix" if explicit_store_prefix_id else "load_only"
    params = {
        "m3_connector_version": 1,
        "request_id": str(sidecar_request["request_id"]),
        "decision": str(sidecar_response["decision"]),
        "reason": str(sidecar_response["reason"]),
        "reuse_tokens": int(sidecar_response["reuse_tokens"]),
        "delta_prefill_tokens": int(sidecar_response["delta_prefill_tokens"]),
        "store_policy": store_policy,
        "sync_ssd_miss_allowed": bool(sidecar_response["sync_ssd_miss_allowed"]),
        "correctness_key": dict(sidecar_request.get("correctness_key") or {}),
        "required_ranges": list(sidecar_response.get("required_ranges") or []),
    }
    if explicit_store_prefix_id:
        params["store_prefix_id"] = str(explicit_store_prefix_id)
        params["store_token_end"] = int(sidecar_request["token_count"])
    return params


def _prefetch_result_to_json(result: PrefetchResult) -> dict[str, Any]:
    return asdict(result)


def _request_to_json(request: SidecarRequest) -> dict[str, Any]:
    return {
        **asdict(request),
        "correctness_key": asdict(request.correctness_key),
        "prefix_candidates": [asdict(candidate) for candidate in request.prefix_candidates],
    }


def _response_to_json(response: SidecarResponse) -> dict[str, Any]:
    return {
        **asdict(response),
        "decision": response.decision.value,
        "required_ranges": [_range_to_json(item) for item in response.required_ranges],
    }


def _barrier_to_json(barrier: ReadyBarrierResult) -> dict[str, Any]:
    return {
        "request_id": barrier.request_id,
        "all_required_blocks_ready": barrier.all_required_blocks_ready,
        "missing_blocks": [_range_to_json(item) for item in barrier.missing_blocks],
        "sync_ssd_miss_total": barrier.sync_ssd_miss_total,
    }


def _classification_counts(classification: KVClassification) -> dict[str, int]:
    return {
        "ready": len(classification.ready),
        "cold": len(classification.cold),
        "fetching": len(classification.fetching),
        "missing": len(classification.missing),
        "mismatch": len(classification.mismatch),
    }


def _range_to_json(item: RequiredRange) -> dict[str, Any]:
    return asdict(item)


def _extract_connector_manifest(payload: dict[str, Any]) -> dict[str, Any] | None:
    params = payload.get("kv_transfer_params")
    if not isinstance(params, dict):
        return None
    connector = params.get("m3_noop_connector")
    if not isinstance(connector, dict):
        return None
    manifest = connector.get("manifest")
    if not isinstance(manifest, dict):
        return None
    if "prefix_id" not in manifest or "token_end" not in manifest:
        return None
    return manifest
