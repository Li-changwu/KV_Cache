import csv
import json
import subprocess
import sys
from pathlib import Path

import benchmarks.m3.longmemeval_workload as lmw
from benchmarks.m3.longmemeval_workload import (
    LongMemEvalAdapterConfig,
    build_longmemeval_manifest,
    ensure_complete_json_file,
    format_history_prefix,
    format_question_suffix,
    load_longmemeval_records,
)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

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


def _write_fixture(path: Path) -> None:
    records = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "What city did I decide to visit?",
            "answer": "Kyoto",
            "question_date": "2025-01-03",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2025-01-01", "2025-01-02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I am comparing Kyoto and Osaka."},
                    {"role": "assistant", "content": "Kyoto sounds calmer."},
                ],
                [
                    {"role": "user", "content": "I decided to visit Kyoto in April."},
                    {"role": "assistant", "content": "I will remember Kyoto for April."},
                ],
            ],
            "answer_session_ids": ["s2"],
        },
        {
            "question_id": "q2_abs",
            "question_type": "single-session-user",
            "question": "What did I say about Mars?",
            "answer": "No answer",
            "question_date": "2025-01-04",
            "haystack_session_ids": ["s3"],
            "haystack_dates": ["2025-01-03"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I like tea."},
                    {"role": "assistant", "content": "Tea preference noted."},
                ]
            ],
            "answer_session_ids": [],
        },
    ]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def test_format_history_prefix_preserves_sessions_and_roles():
    prefix = format_history_prefix(
        {
            "question_id": "q1",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2025-01-01"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            ],
        }
    )

    assert "[Session 1 | id=s1 | date=2025-01-01]" in prefix
    assert "User: hello" in prefix
    assert "Assistant: hi" in prefix


def test_format_question_suffix_uses_current_turn_and_answer_hint():
    suffix = format_question_suffix(
        {
            "question_id": "q1",
            "question_date": "2025-01-03",
            "question": "Where am I going?",
            "answer": "Kyoto",
        },
        include_answer=True,
    )

    assert "[Current Question | date=2025-01-03]" in suffix
    assert "User: Where am I going?" in suffix
    assert "Expected answer: Kyoto" in suffix


def test_load_longmemeval_records_reads_list_json(tmp_path):
    fixture = tmp_path / "longmemeval_s_cleaned.json"
    _write_fixture(fixture)

    records = load_longmemeval_records(fixture)

    assert [record["question_id"] for record in records] == ["q1", "q2_abs"]


def test_ensure_complete_json_file_rejects_truncated_download(tmp_path):
    partial = tmp_path / "longmemeval_oracle.json"
    partial.write_text('[{"question_id": "q1"}', encoding="utf-8")

    result = ensure_complete_json_file(partial, expected_min_bytes=32)

    assert result["ok"] is False
    assert result["error_type"] in {"file_too_small", "json_decode_error"}


def test_build_manifest_truncates_prefix_to_token_budget(tmp_path):
    fixture = tmp_path / "longmemeval_s_cleaned.json"
    _write_fixture(fixture)
    output_dir = tmp_path / "manifest"

    result = build_longmemeval_manifest(
        LongMemEvalAdapterConfig(
            data_path=fixture,
            output_dir=output_dir,
            max_samples=1,
            target_prefix_tokens=[24],
            suffix_token_budget=12,
            tokenizer=None,
        )
    )

    assert result["status"] == "OK"
    manifest_path = output_dir / "longmemeval_workload.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["question_id"] == "q1"
    assert row["dataset_split"] == "longmemeval_s_cleaned"
    assert row["target_prefix_tokens"] == 24
    assert row["actual_prefix_tokens"] <= 24
    assert row["suffix_tokens"] <= 12
    assert row["reuse_prompt"].startswith(row["prefix_prompt"])
    assert row["m3_prefix_id"] == "lme-q1-p24"

    csv_rows = list(csv.DictReader((output_dir / "longmemeval_workload.csv").open()))
    assert csv_rows[0]["question_id"] == "q1"
    assert csv_rows[0]["actual_prefix_tokens"] == str(row["actual_prefix_tokens"])


def test_build_manifest_reuse_prompt_preserves_prefix_token_boundary(
    tmp_path,
    monkeypatch,
):
    record = {
        "question_id": "q_boundary",
        "question_type": "multi-session",
        "question": "What should be reused?",
        "answer": "The tail prefix.",
        "question_date": "2025-01-03",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2025-01-01"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma delta epsilon zeta eta theta",
                }
            ],
        ],
        "answer_session_ids": ["s1"],
    }
    full_prefix = format_history_prefix(record)
    target = len(full_prefix) - full_prefix.rfind(" ")
    assert full_prefix[-target:].startswith(" ")
    fixture = tmp_path / "longmemeval_s_cleaned.json"
    fixture.write_text(json.dumps([record]), encoding="utf-8")
    output_dir = tmp_path / "manifest"
    tokenizer = _CharTokenizer()
    monkeypatch.setattr(lmw, "_build_tokenizer", lambda _name: tokenizer)

    build_longmemeval_manifest(
        LongMemEvalAdapterConfig(
            data_path=fixture,
            output_dir=output_dir,
            max_samples=1,
            target_prefix_tokens=[target],
            suffix_token_budget=64,
            tokenizer="char",
        )
    )

    [row] = [
        json.loads(line)
        for line in (output_dir / "longmemeval_workload.jsonl").read_text().splitlines()
    ]
    prefix_ids = tokenizer.encode(row["prefix_prompt"])
    reuse_ids = tokenizer.encode(row["reuse_prompt"])
    assert row["prefix_prompt"].startswith(" ")
    assert row["reuse_prompt"].startswith(row["prefix_prompt"])
    assert reuse_ids[: len(prefix_ids)] == prefix_ids


def test_longmemeval_workload_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "benchmarks/m3/longmemeval_workload.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--target-prefix-tokens" in completed.stdout
