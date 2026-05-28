"""Synthetic workloads for the M2 tiered KV cache simulator."""

from __future__ import annotations

from dataclasses import dataclass


ONE_M_TOKENS = 1_048_576


@dataclass(frozen=True)
class Request:
    request_id: str
    workload: str
    total_tokens: int
    reuse_tokens: int
    new_tokens: int
    prefix_id: str
    arrival_ms: float


def _request(
    workload: str,
    index: int,
    total_tokens: int,
    reuse_tokens: int,
    prefix_id: str,
    spacing_ms: float,
) -> Request:
    reuse_tokens = min(reuse_tokens, total_tokens)
    return Request(
        request_id=f"{workload}_{index:04d}",
        workload=workload,
        total_tokens=total_tokens,
        reuse_tokens=reuse_tokens,
        new_tokens=total_tokens - reuse_tokens,
        prefix_id=prefix_id,
        arrival_ms=index * spacing_ms,
    )


def build_workload(
    name: str,
    num_requests: int = 8,
    total_tokens: int = ONE_M_TOKENS,
    append_tokens: int = 262_144,
    shared_prefix_tokens: int = 786_432,
    spacing_ms: float = 100.0,
) -> list[Request]:
    """Build one of the first three M2 workload classes."""

    if num_requests <= 0:
        return []

    if name == "session_append":
        requests = []
        for index in range(num_requests):
            reuse_tokens = min(index * append_tokens, total_tokens)
            requests.append(
                _request(
                    workload=name,
                    index=index,
                    total_tokens=total_tokens,
                    reuse_tokens=reuse_tokens,
                    prefix_id="session_a",
                    spacing_ms=spacing_ms,
                )
            )
        return requests

    if name == "shared_prefix":
        return [
            _request(
                workload=name,
                index=index,
                total_tokens=total_tokens,
                reuse_tokens=shared_prefix_tokens if index > 0 else 0,
                prefix_id="shared_doc_a",
                spacing_ms=spacing_ms,
            )
            for index in range(num_requests)
        ]

    if name == "low_locality":
        return [
            _request(
                workload=name,
                index=index,
                total_tokens=total_tokens,
                reuse_tokens=0,
                prefix_id=f"random_doc_{index:04d}",
                spacing_ms=spacing_ms,
            )
            for index in range(num_requests)
        ]

    raise ValueError(f"unknown workload: {name}")
