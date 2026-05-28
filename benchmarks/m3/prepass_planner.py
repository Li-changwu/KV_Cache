"""Minimal KV PrePass status planner for M3.12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrePassPlan:
    status: str
    ready_before_request: bool
    required_count: int
    ready_count: int
    restore_count: int
    deadline_miss_count: int
    error_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready_before_request": self.ready_before_request,
            "required_count": self.required_count,
            "ready_count": self.ready_count,
            "restore_count": self.restore_count,
            "deadline_miss_count": self.deadline_miss_count,
            "error_count": self.error_count,
        }


def build_prepass_plan(
    *,
    required_count: int,
    ready_count: int,
    restore_results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> PrePassPlan:
    """Classify a prepass after residency checks and restore scheduling."""
    deadline_miss_count = sum(
        1 for result in restore_results if bool(result.get("deadline_miss"))
    )
    queued_count = sum(
        1
        for result in restore_results
        if result.get("status") in {"QUEUED", "ALREADY_QUEUED"}
    )
    total_ready = ready_count
    error_count = len(errors)

    if error_count:
        status = "ERROR"
    elif deadline_miss_count:
        status = "MISSED"
    elif required_count and total_ready >= required_count:
        status = "READY"
    elif queued_count:
        status = "QUEUED"
    elif required_count == 0:
        status = "READY"
    else:
        status = "PENDING"

    return PrePassPlan(
        status=status,
        ready_before_request=(required_count == 0 or total_ready >= required_count),
        required_count=required_count,
        ready_count=total_ready,
        restore_count=len(restore_results),
        deadline_miss_count=deadline_miss_count,
        error_count=error_count,
    )
