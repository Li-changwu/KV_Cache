#!/usr/bin/env python3
"""Build M3 workload manifests from LongMemEval records.

The adapter treats LongMemEval chat history as reusable historical KV prefix
and the current question as the online suffix. It is intentionally offline:
online runners can consume the generated JSONL/CSV without knowing the source
dataset format.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_DATA_PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT_DIR = Path("results/m3_16_longmemeval_workload")
DEFAULT_TARGET_PREFIX_TOKENS = [2048, 8192, 16_384]

JSONL_COLUMNS = [
    "dataset_name",
    "dataset_split",
    "question_id",
    "question_type",
    "question_date",
    "answer_session_ids",
    "target_prefix_tokens",
    "actual_prefix_tokens",
    "full_history_tokens",
    "suffix_tokens",
    "reuse_prompt_tokens",
    "history_truncated",
    "suffix_truncated",
    "truncation_policy",
    "m3_prefix_id",
    "prefix_prompt",
    "suffix_prompt",
    "reuse_prompt",
]

CSV_COLUMNS = [
    "dataset_name",
    "dataset_split",
    "question_id",
    "question_type",
    "question_date",
    "answer_session_count",
    "target_prefix_tokens",
    "actual_prefix_tokens",
    "full_history_tokens",
    "suffix_tokens",
    "reuse_prompt_tokens",
    "history_truncated",
    "suffix_truncated",
    "truncation_policy",
    "m3_prefix_id",
]


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


@dataclass(frozen=True)
class WhitespaceTokenizer:
    """Small deterministic fallback used by unit tests and dry-run fixtures."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(_split_words(text))))

    def decode(self, token_ids: list[int]) -> str:  # pragma: no cover - unused
        raise NotImplementedError("WhitespaceTokenizer cannot decode ids")

    def trim_left(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        words = _split_words(text)
        truncated = len(words) > max_tokens
        kept = words[-max_tokens:] if truncated else words
        return " ".join(kept), len(kept), truncated

    def trim_right(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        words = _split_words(text)
        truncated = len(words) > max_tokens
        kept = words[:max_tokens] if truncated else words
        return " ".join(kept), len(kept), truncated


@dataclass
class HuggingFaceTokenizer:
    tokenizer: Any

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    def trim_left(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        ids = self.encode(text)
        truncated = len(ids) > max_tokens
        kept = ids[-max_tokens:] if truncated else ids
        return self.decode(kept), len(kept), truncated

    def trim_right(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        ids = self.encode(text)
        truncated = len(ids) > max_tokens
        kept = ids[:max_tokens] if truncated else ids
        return self.decode(kept), len(kept), truncated


@dataclass(frozen=True)
class LongMemEvalAdapterConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_samples: int = 8
    target_prefix_tokens: list[int] = field(
        default_factory=lambda: list(DEFAULT_TARGET_PREFIX_TOKENS)
    )
    suffix_token_budget: int = 256
    tokenizer: str | None = "/root/models/Qwen2.5-14B-Instruct"
    include_answer: bool = False
    skip_abstention: bool = False
    dataset_name: str = "LongMemEval"
    dataset_split: str | None = None


def load_longmemeval_records(path: Path) -> list[dict[str, Any]]:
    integrity = ensure_complete_json_file(path)
    if not integrity["ok"]:
        raise ValueError(
            f"invalid LongMemEval JSON file {path}: "
            f"{integrity['error_type']}: {integrity['error']}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    else:
        raise ValueError(f"unsupported LongMemEval JSON shape in {path}")
    return [record for record in records if isinstance(record, dict)]


def ensure_complete_json_file(
    path: Path,
    *,
    expected_min_bytes: int = 2,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "ok": False,
            "error_type": "file_not_found",
            "error": str(path),
            "size_bytes": 0,
        }
    size_bytes = path.stat().st_size
    if size_bytes < expected_min_bytes:
        return {
            "ok": False,
            "error_type": "file_too_small",
            "error": f"size_bytes={size_bytes} < expected_min_bytes={expected_min_bytes}",
            "size_bytes": size_bytes,
        }
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error_type": "json_decode_error",
            "error": str(exc),
            "size_bytes": size_bytes,
        }
    return {
        "ok": True,
        "error_type": "",
        "error": "",
        "size_bytes": size_bytes,
    }


def format_history_prefix(record: dict[str, Any]) -> str:
    session_ids = _list_value(record.get("haystack_session_ids"))
    dates = _list_value(record.get("haystack_dates"))
    sessions = _list_value(record.get("haystack_sessions"))
    lines: list[str] = []
    for index, session in enumerate(sessions):
        session_id = str(session_ids[index]) if index < len(session_ids) else ""
        date = str(dates[index]) if index < len(dates) else ""
        header = f"[Session {index + 1}"
        if session_id:
            header += f" | id={session_id}"
        if date:
            header += f" | date={date}"
        header += "]"
        lines.append(header)
        if not isinstance(session, list):
            continue
        for turn in session:
            if not isinstance(turn, dict):
                continue
            role = _role_label(str(turn.get("role", "")))
            content = _clean_text(str(turn.get("content", "")))
            if content:
                lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def format_question_suffix(
    record: dict[str, Any],
    *,
    include_answer: bool = False,
) -> str:
    date = _clean_text(str(record.get("question_date", "")))
    question = _clean_text(str(record.get("question", "")))
    lines = [f"[Current Question | date={date}]" if date else "[Current Question]"]
    lines.append(f"User: {question}")
    if include_answer and record.get("answer") not in (None, ""):
        lines.append(f"Expected answer: {_clean_text(str(record['answer']))}")
    return "\n".join(lines).strip()


def build_longmemeval_manifest(config: LongMemEvalAdapterConfig) -> dict[str, Any]:
    records = load_longmemeval_records(config.data_path)
    tokenizer = _build_tokenizer(config.tokenizer)
    selected = _select_records(records, config.max_samples, config.skip_abstention)
    rows: list[dict[str, Any]] = []
    for record in selected:
        full_prefix = format_history_prefix(record)
        suffix_text = format_question_suffix(
            record,
            include_answer=config.include_answer,
        )
        full_history_tokens = _count_tokens(tokenizer, full_prefix)
        suffix_prompt, suffix_tokens, suffix_truncated = _trim_right(
            tokenizer,
            suffix_text,
            config.suffix_token_budget,
        )
        for target in config.target_prefix_tokens:
            prefix_prompt, actual_prefix_tokens, history_truncated = _trim_left(
                tokenizer,
                full_prefix,
                int(target),
            )
            reuse_prompt = _build_reuse_prompt(tokenizer, prefix_prompt, suffix_prompt)
            rows.append(
                {
                    "dataset_name": config.dataset_name,
                    "dataset_split": config.dataset_split
                    or _dataset_split_from_path(config.data_path),
                    "question_id": str(record.get("question_id", "")),
                    "question_type": str(record.get("question_type", "")),
                    "question_date": str(record.get("question_date", "")),
                    "answer_session_ids": [
                        str(item) for item in _list_value(record.get("answer_session_ids"))
                    ],
                    "target_prefix_tokens": int(target),
                    "actual_prefix_tokens": actual_prefix_tokens,
                    "full_history_tokens": full_history_tokens,
                    "suffix_tokens": suffix_tokens,
                    "reuse_prompt_tokens": actual_prefix_tokens + suffix_tokens,
                    "history_truncated": history_truncated,
                    "suffix_truncated": suffix_truncated,
                    "truncation_policy": "history_keep_most_recent_suffix_keep_question_start",
                    "m3_prefix_id": _prefix_id(record.get("question_id", ""), int(target)),
                    "prefix_prompt": prefix_prompt,
                    "suffix_prompt": suffix_prompt,
                    "reuse_prompt": reuse_prompt,
                }
            )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = config.output_dir / "longmemeval_workload.jsonl"
    csv_path = config.output_dir / "longmemeval_workload.csv"
    summary_path = config.output_dir / "longmemeval_workload_summary.json"
    report_path = config.output_dir / "longmemeval_workload_report.md"
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    summary = _summary(rows, config)
    _write_json(summary_path, summary)
    _write_report(report_path, summary, config)
    return {
        "status": "OK" if rows else "EMPTY",
        "rows": len(rows),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def _build_tokenizer(tokenizer_name: str | None) -> Any:
    if tokenizer_name in (None, "", "whitespace"):
        return WhitespaceTokenizer()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return HuggingFaceTokenizer(tokenizer)


def _count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def _trim_left(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, int, bool]:
    if hasattr(tokenizer, "trim_left"):
        return tokenizer.trim_left(text, max(1, int(max_tokens)))
    ids = tokenizer.encode(text)
    truncated = len(ids) > max_tokens
    kept = ids[-max_tokens:] if truncated else ids
    return tokenizer.decode(kept), len(kept), truncated


def _trim_right(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, int, bool]:
    if hasattr(tokenizer, "trim_right"):
        return tokenizer.trim_right(text, max(1, int(max_tokens)))
    ids = tokenizer.encode(text)
    truncated = len(ids) > max_tokens
    kept = ids[:max_tokens] if truncated else ids
    return tokenizer.decode(kept), len(kept), truncated


def _build_reuse_prompt(tokenizer: Any, prefix_prompt: str, suffix_prompt: str) -> str:
    if not prefix_prompt:
        return suffix_prompt
    if not suffix_prompt:
        return prefix_prompt
    prefix_ids = tokenizer.encode(prefix_prompt)
    candidates = [
        f"{prefix_prompt}\n{suffix_prompt}",
        f"{prefix_prompt}\n\n{suffix_prompt}",
        f"{prefix_prompt} {suffix_prompt}",
        f"{prefix_prompt}{suffix_prompt}",
    ]
    for candidate in candidates:
        if tokenizer.encode(candidate)[: len(prefix_ids)] == prefix_ids:
            return candidate
    raise ValueError(
        "unable to build token-stable reuse prompt: "
        f"prefix_tokens={len(prefix_ids)} suffix_chars={len(suffix_prompt)}"
    )


def _select_records(
    records: list[dict[str, Any]],
    max_samples: int,
    skip_abstention: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        question_id = str(record.get("question_id", ""))
        if skip_abstention and question_id.endswith("_abs"):
            continue
        selected.append(record)
        if len(selected) >= max_samples:
            break
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{column: row.get(column, "") for column in CSV_COLUMNS},
                    "answer_session_count": len(row.get("answer_session_ids", [])),
                }
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    summary: dict[str, Any],
    config: LongMemEvalAdapterConfig,
) -> None:
    lines = [
        "# LongMemEval Workload Manifest",
        "",
        f"- data_path: `{config.data_path}`",
        f"- tokenizer: `{config.tokenizer or 'whitespace'}`",
        f"- rows: `{summary['rows']}`",
        f"- samples: `{summary['samples']}`",
        f"- target_prefix_tokens: `{summary['target_prefix_tokens']}`",
        f"- actual_prefix_tokens_min: `{summary['actual_prefix_tokens_min']}`",
        f"- actual_prefix_tokens_max: `{summary['actual_prefix_tokens_max']}`",
        f"- truncated_rows: `{summary['truncated_rows']}`",
        "",
        "This manifest maps LongMemEval history sessions to reusable historical KV prefixes and the current question to the online suffix.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(
    rows: list[dict[str, Any]],
    config: LongMemEvalAdapterConfig,
) -> dict[str, Any]:
    prefix_values = [int(row["actual_prefix_tokens"]) for row in rows]
    samples = {str(row["question_id"]) for row in rows}
    return {
        "status": "OK" if rows else "EMPTY",
        "rows": len(rows),
        "samples": len(samples),
        "dataset_split": config.dataset_split or _dataset_split_from_path(config.data_path),
        "target_prefix_tokens": [int(item) for item in config.target_prefix_tokens],
        "actual_prefix_tokens_min": min(prefix_values) if prefix_values else 0,
        "actual_prefix_tokens_max": max(prefix_values) if prefix_values else 0,
        "suffix_token_budget": int(config.suffix_token_budget),
        "truncated_rows": sum(1 for row in rows if row["history_truncated"]),
        "suffix_truncated_rows": sum(1 for row in rows if row["suffix_truncated"]),
        "tokenizer": config.tokenizer or "whitespace",
    }


def _dataset_split_from_path(path: Path) -> str:
    return path.stem


def _prefix_id(question_id: Any, target_tokens: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(question_id)).strip("-")
    return f"lme-{safe or 'unknown'}-p{target_tokens}"


def _role_label(role: str) -> str:
    lowered = role.strip().lower()
    if lowered == "user":
        return "User"
    if lowered == "assistant":
        return "Assistant"
    if lowered == "system":
        return "System"
    return role.strip().title() or "Turn"


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _split_words(text: str) -> list[str]:
    return text.split()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--target-prefix-tokens",
        nargs="+",
        type=int,
        default=list(DEFAULT_TARGET_PREFIX_TOKENS),
    )
    parser.add_argument("--suffix-token-budget", type=int, default=256)
    parser.add_argument(
        "--tokenizer",
        default="/root/models/Qwen2.5-14B-Instruct",
        help="Tokenizer path/name. Use 'whitespace' for deterministic fixture tests.",
    )
    parser.add_argument("--include-answer", action="store_true")
    parser.add_argument("--skip-abstention", action="store_true")
    parser.add_argument("--dataset-name", default="LongMemEval")
    parser.add_argument("--dataset-split")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_longmemeval_manifest(
        LongMemEvalAdapterConfig(
            data_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            max_samples=int(args.max_samples),
            target_prefix_tokens=[int(item) for item in args.target_prefix_tokens],
            suffix_token_budget=int(args.suffix_token_budget),
            tokenizer=None if args.tokenizer == "whitespace" else str(args.tokenizer),
            include_answer=bool(args.include_answer),
            skip_abstention=bool(args.skip_abstention),
            dataset_name=str(args.dataset_name),
            dataset_split=args.dataset_split,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
