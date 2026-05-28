"""Minimal sidecar control plane and mock KV connector for M3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class Decision(StrEnum):
    ADMIT = "ADMIT"
    DELAY = "DELAY"
    REJECT = "REJECT"
    FULL_PREFILL_FALLBACK = "FULL_PREFILL_FALLBACK"


class ConnectorState(StrEnum):
    HBM = "HBM"
    DRAM = "DRAM"
    SSD = "SSD"
    FETCHING = "FETCHING"


@dataclass(frozen=True)
class CorrectnessKey:
    model_fingerprint: str
    tokenizer_fingerprint: str
    rope_config: str
    dtype: str
    kv_layout: str


@dataclass(frozen=True)
class PrefixCandidate:
    prefix_id: str
    token_start: int
    token_end: int
    committed: bool


@dataclass(frozen=True)
class SidecarRequest:
    request_id: str
    model_id: str
    token_count: int
    decode_sla_ms: float
    admission_window_ms: float
    correctness_key: CorrectnessKey
    prefix_candidates: list[PrefixCandidate]


@dataclass(frozen=True)
class KVRange:
    prefix_id: str
    token_start: int
    token_end: int
    tier: ConnectorState
    ready: bool
    correctness_key: CorrectnessKey
    epoch: int = 0

    @property
    def tokens(self) -> int:
        return max(0, self.token_end - self.token_start)


@dataclass(frozen=True)
class RequiredRange:
    prefix_id: str
    token_start: int
    token_end: int
    required_tier: str
    deadline_ms: float


@dataclass(frozen=True)
class SidecarResponse:
    request_id: str
    decision: Decision
    reason: str
    reuse_tokens: int
    delta_prefill_tokens: int
    estimated_prefetch_ms: float
    estimated_ttft_ms: float
    sync_ssd_miss_allowed: bool
    required_ranges: list[RequiredRange]


@dataclass(frozen=True)
class ReadyBarrierResult:
    request_id: str
    all_required_blocks_ready: bool
    missing_blocks: list[RequiredRange]
    sync_ssd_miss_total: int


class KVManifest:
    def __init__(self) -> None:
        self._ranges: dict[str, list[KVRange]] = {}
        self._epoch = 0

    def commit(
        self,
        prefix_id: str,
        token_start: int,
        token_end: int,
        correctness_key: CorrectnessKey,
        tier: ConnectorState = ConnectorState.DRAM,
        ready: bool = True,
    ) -> None:
        self._epoch += 1
        entry = KVRange(
            prefix_id=prefix_id,
            token_start=token_start,
            token_end=token_end,
            correctness_key=correctness_key,
            tier=tier,
            ready=ready,
            epoch=self._epoch,
        )
        self._ranges.setdefault(prefix_id, []).append(entry)

    def lookup(self, candidate: PrefixCandidate) -> KVRange | None:
        entries = self._ranges.get(candidate.prefix_id, [])
        eligible = [
            entry
            for entry in entries
            if entry.token_start <= candidate.token_start
            and entry.token_end >= candidate.token_end
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda entry: (
                entry.epoch,
                entry.tokens,
                int(entry.ready),
                _tier_priority(entry.tier),
            ),
        )

    def mark_prefetched(
        self,
        required: RequiredRange,
        *,
        tier: ConnectorState = ConnectorState.DRAM,
    ) -> KVRange | None:
        entry = self.lookup(
            PrefixCandidate(
                prefix_id=required.prefix_id,
                token_start=required.token_start,
                token_end=required.token_end,
                committed=True,
            )
        )
        if entry is None:
            return None
        self._epoch += 1
        updated = KVRange(
            prefix_id=entry.prefix_id,
            token_start=entry.token_start,
            token_end=entry.token_end,
            correctness_key=entry.correctness_key,
            tier=tier,
            ready=True,
            epoch=self._epoch,
        )
        self._ranges.setdefault(entry.prefix_id, []).append(updated)
        return updated


class MockKVConnector:
    def __init__(self, manifest: KVManifest) -> None:
        self.manifest = manifest
        self.prefetches: list[RequiredRange] = []
        self.pinned: list[RequiredRange] = []

    def lookup(self, candidate: PrefixCandidate) -> KVRange | None:
        return self.manifest.lookup(candidate)

    def prefetch(self, ranges: list[RequiredRange]) -> None:
        self.prefetches.extend(ranges)

    def pin(self, ranges: list[RequiredRange]) -> bool:
        self.pinned.extend(ranges)
        return True

    def is_ready(self, required: RequiredRange) -> bool:
        candidate = PrefixCandidate(
            prefix_id=required.prefix_id,
            token_start=required.token_start,
            token_end=required.token_end,
            committed=True,
        )
        entry = self.lookup(candidate)
        return entry is not None and entry.ready and entry.tier in {
            ConnectorState.HBM,
            ConnectorState.DRAM,
        }


class TieredKVControlPlane:
    def __init__(
        self,
        hbm_capacity_tokens: int,
        dram_capacity_tokens: int,
        ssd_capacity_tokens: int,
        kv_bytes_per_token: int,
        h2d_gbps: float,
        storage_gbps: float,
    ) -> None:
        self.hbm_capacity_tokens = hbm_capacity_tokens
        self.dram_capacity_tokens = dram_capacity_tokens
        self.ssd_capacity_tokens = ssd_capacity_tokens
        self.kv_bytes_per_token = kv_bytes_per_token
        self.h2d_gbps = h2d_gbps
        self.storage_gbps = storage_gbps
        self.manifest = KVManifest()
        self.connector = MockKVConnector(self.manifest)

    def admit(self, request: SidecarRequest) -> SidecarResponse:
        match = self._best_reuse_match(request)
        if match == "correctness_key_mismatch":
            return self._response(
                request,
                Decision.REJECT,
                reason="correctness_key_mismatch",
                reuse_tokens=0,
                required_ranges=[],
                estimated_prefetch_ms=0.0,
            )
        if match is None:
            decision = (
                Decision.FULL_PREFILL_FALLBACK
                if request.token_count >= 1_048_576
                else Decision.ADMIT
            )
            reason = (
                "no_reusable_prefix_full_prefill"
                if decision == Decision.FULL_PREFILL_FALLBACK
                else "no_reusable_prefix"
            )
            return self._response(
                request,
                decision,
                reason=reason,
                reuse_tokens=0,
                required_ranges=[],
                estimated_prefetch_ms=0.0,
            )

        candidate, entry = match
        reuse_tokens = min(candidate.token_end - candidate.token_start, request.token_count)
        required = [
            RequiredRange(
                prefix_id=candidate.prefix_id,
                token_start=candidate.token_start,
                token_end=candidate.token_end,
                required_tier="DRAM_OR_HBM",
                deadline_ms=request.admission_window_ms,
            )
        ]
        estimated_prefetch_ms = self._estimate_prefetch_ms(entry)
        capacity_ok = request.token_count <= (
            self.hbm_capacity_tokens + self.dram_capacity_tokens + self.ssd_capacity_tokens
        )
        ready = self.connector.is_ready(required[0])
        if not capacity_ok:
            decision = Decision.REJECT
            reason = "capacity_exceeded"
        elif not ready:
            decision = Decision.DELAY
            reason = "required_kv_not_ready_before_decode"
            self.connector.prefetch(required)
        elif estimated_prefetch_ms > request.admission_window_ms:
            decision = Decision.DELAY
            reason = "prefetch_misses_admission_window"
            self.connector.prefetch(required)
        else:
            decision = Decision.ADMIT
            reason = "required_kv_ready_before_decode"
            self.connector.pin(required)

        return self._response(
            request,
            decision,
            reason=reason,
            reuse_tokens=reuse_tokens,
            required_ranges=required,
            estimated_prefetch_ms=estimated_prefetch_ms,
        )

    def commit_request(
        self,
        response: SidecarResponse,
        prefix_id: str,
        token_end: int,
        tier: ConnectorState | str = ConnectorState.DRAM,
        ready: bool = True,
    ) -> None:
        # The response does not carry the correctness key; replay calls this only for
        # requests admitted or executed as full-prefill fallback, so use the request key
        # recorded by _last_request_keys.
        correctness_key = self._last_request_keys[response.request_id]
        self.manifest.commit(
            prefix_id=prefix_id,
            token_start=0,
            token_end=token_end,
            correctness_key=correctness_key,
            tier=ConnectorState(str(tier)),
            ready=ready,
        )

    def mark_prefetched(self, ranges: list[RequiredRange]) -> None:
        for required in ranges:
            self.manifest.mark_prefetched(required, tier=ConnectorState.DRAM)

    def ready_barrier(self, response: SidecarResponse) -> ReadyBarrierResult:
        missing = [
            required
            for required in response.required_ranges
            if not self.connector.is_ready(required)
        ]
        return ReadyBarrierResult(
            request_id=response.request_id,
            all_required_blocks_ready=not missing,
            missing_blocks=missing,
            sync_ssd_miss_total=0,
        )

    def _best_reuse_match(
        self,
        request: SidecarRequest,
    ) -> tuple[PrefixCandidate, KVRange] | str | None:
        self._last_request_keys = getattr(self, "_last_request_keys", {})
        self._last_request_keys[request.request_id] = request.correctness_key
        mismatched = False
        matches: list[tuple[PrefixCandidate, KVRange]] = []
        for candidate in request.prefix_candidates:
            if not candidate.committed:
                continue
            entry = self.connector.lookup(candidate)
            if entry is None:
                continue
            if entry.correctness_key != request.correctness_key:
                mismatched = True
                continue
            matches.append((candidate, entry))
        if matches:
            return max(matches, key=lambda item: item[0].token_end - item[0].token_start)
        if mismatched:
            return "correctness_key_mismatch"
        return None

    def _estimate_prefetch_ms(self, entry: KVRange) -> float:
        if entry.tier == ConnectorState.HBM:
            return 0.0
        h2d_ms = self._transfer_ms(entry.tokens, self.h2d_gbps)
        if entry.tier == ConnectorState.DRAM:
            return h2d_ms
        storage_ms = self._transfer_ms(entry.tokens, self.storage_gbps)
        return storage_ms + h2d_ms

    def _transfer_ms(self, tokens: int, gbps: float) -> float:
        bytes_to_transfer = tokens * self.kv_bytes_per_token
        return bytes_to_transfer / (gbps * 1_000_000_000) * 1000.0

    def _response(
        self,
        request: SidecarRequest,
        decision: Decision,
        reason: str,
        reuse_tokens: int,
        required_ranges: list[RequiredRange],
        estimated_prefetch_ms: float,
    ) -> SidecarResponse:
        delta_prefill_tokens = request.token_count - reuse_tokens
        estimated_ttft_ms = estimated_prefetch_ms + (
            50.0 + 5000.0 * (delta_prefill_tokens / max(request.token_count, 1))
            if delta_prefill_tokens
            else estimated_prefetch_ms
        )
        return SidecarResponse(
            request_id=request.request_id,
            decision=decision,
            reason=reason,
            reuse_tokens=reuse_tokens,
            delta_prefill_tokens=delta_prefill_tokens,
            estimated_prefetch_ms=round(estimated_prefetch_ms, 3),
            estimated_ttft_ms=round(estimated_ttft_ms, 3),
            sync_ssd_miss_allowed=False,
            required_ranges=required_ranges,
        )


def response_to_row(response: SidecarResponse, barrier: ReadyBarrierResult) -> dict[str, str]:
    row = asdict(response)
    row["decision"] = response.decision.value
    row["all_required_blocks_ready"] = str(barrier.all_required_blocks_ready)
    row["missing_blocks"] = str(len(barrier.missing_blocks))
    row["sync_ssd_miss_total"] = str(barrier.sync_ssd_miss_total)
    row.pop("required_ranges")
    return {key: str(value) for key, value in row.items()}


def _tier_priority(tier: ConnectorState) -> int:
    if tier == ConnectorState.HBM:
        return 3
    if tier == ConnectorState.DRAM:
        return 2
    if tier == ConnectorState.FETCHING:
        return 1
    return 0
