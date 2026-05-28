#!/usr/bin/env python3
"""Summarize an M3.8 restart-control reuse matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def summarize_restart_matrix(result_dir: Path) -> dict[str, Any]:
    store_rows = _read_rows(result_dir / "store_phase" / "reuse_smoke_matrix.csv")
    reuse_rows = _read_rows(result_dir / "reuse_phase" / "reuse_smoke_matrix.csv")
    external_load_rows = [
        row for row in reuse_rows if row.get("external_load_observed") == "yes"
    ]
    ok_store_rows = [row for row in store_rows if row.get("status") == "OK"]
    ok_reuse_rows = [row for row in reuse_rows if row.get("status") == "OK"]
    all_external = bool(reuse_rows) and len(external_load_rows) == len(reuse_rows)
    all_ok = len(ok_store_rows) == len(store_rows) and len(ok_reuse_rows) == len(reuse_rows)
    status = "OK" if all_ok and all_external else "PARTIAL" if ok_reuse_rows else "ERROR"
    summary = {
        "status": status,
        "store_rows": len(store_rows),
        "reuse_rows": len(reuse_rows),
        "ok_store_rows": len(ok_store_rows),
        "ok_reuse_rows": len(ok_reuse_rows),
        "external_load_rows": len(external_load_rows),
        "all_external_load_observed": all_external,
        "total_load_elapsed_ms": round(_sum_float(reuse_rows, "load_elapsed_ms"), 3),
        "total_store_elapsed_ms": round(_sum_float(store_rows, "store_elapsed_ms"), 3),
    }
    _write_report(result_dir / "restart_matrix_report.md", summary, store_rows, reuse_rows)
    (result_dir / "restart_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sum_float(rows: list[dict[str, str]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            total += float(value)
    return total


def _write_report(
    path: Path,
    summary: dict[str, Any],
    store_rows: list[dict[str, str]],
    reuse_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# M3.8 完整重启对照矩阵报告",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 保存阶段成功：`{summary['ok_store_rows']}/{summary['store_rows']}`",
        f"- 复用阶段成功：`{summary['ok_reuse_rows']}/{summary['reuse_rows']}`",
        f"- 外部加载覆盖：`{summary['external_load_rows']}/{summary['reuse_rows']}`",
        f"- 保存事件总耗时：`{summary['total_store_elapsed_ms']} ms`",
        f"- 加载事件总耗时：`{summary['total_load_elapsed_ms']} ms`",
        "",
        "## 复用阶段明细",
        "",
    ]
    for row in reuse_rows:
        lines.append(
            "- "
            f"`{row.get('run_id')}`：status=`{row.get('status')}`，"
            f"prefix=`{row.get('prefix_tokens')}`，"
            f"second_ttft_ms=`{row.get('second_ttft_ms')}`，"
            f"load_elapsed_ms=`{row.get('load_elapsed_ms')}`，"
            f"external_load=`{row.get('external_load_observed')}`"
        )
    if store_rows:
        lines.extend(["", "## 保存阶段明细", ""])
        for row in store_rows:
            lines.append(
                "- "
                f"`{row.get('run_id')}`：status=`{row.get('status')}`，"
                f"prefix=`{row.get('prefix_tokens')}`，"
                f"first_ttft_ms=`{row.get('first_ttft_ms')}`，"
                f"store_elapsed_ms=`{row.get('store_elapsed_ms')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/m3_8_full_restart_matrix")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_restart_matrix(Path(args.result_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
