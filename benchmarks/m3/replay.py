"""Offline replay helpers for the M3 sidecar prototype."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.m3.control_plane import (
    Decision,
    SidecarRequest,
    TieredKVControlPlane,
    response_to_row,
)


def run_replay(
    control_plane: TieredKVControlPlane,
    requests: list[SidecarRequest],
    commit_prefix_id: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for request in requests:
        response = control_plane.admit(request)
        barrier = control_plane.ready_barrier(response)
        rows.append(response_to_row(response, barrier))
        if response.decision in {Decision.ADMIT, Decision.FULL_PREFILL_FALLBACK}:
            control_plane.commit_request(
                response,
                prefix_id=commit_prefix_id,
                token_end=request.token_count,
            )
    return rows


def write_replay_outputs(output_dir: str | Path, rows: list[dict[str, str]]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "m3_replay_decisions.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    sync_misses = sum(int(row["sync_ssd_miss_total"]) for row in rows)
    lines = [
        "# M3 Sidecar Replay Report",
        "",
        f"- requests={len(rows)}",
        f"- sync_ssd_miss_total={sync_misses}",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['request_id']}: decision={row['decision']}, "
            f"reuse_tokens={row['reuse_tokens']}, "
            f"delta_prefill_tokens={row['delta_prefill_tokens']}, "
            f"sync_ssd_miss_total={row['sync_ssd_miss_total']}"
        )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
