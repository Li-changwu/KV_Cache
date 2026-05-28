import csv
from pathlib import Path

from benchmarks.m1_5.parse_vllm_results import parse_directory
from benchmarks.m1_5.parse_vllm_results import parse_result


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_parse_directory_splits_random_and_prefix_results(tmp_path):
    parse_directory(FIXTURES, tmp_path)

    latency_rows = read_csv(tmp_path / "vllm_latency.csv")
    apc_rows = read_csv(tmp_path / "apc_prefix_reuse.csv")

    assert {row["dataset"] for row in latency_rows} == {"random"}
    assert {row["dataset"] for row in apc_rows} == {"prefix_repetition"}
    by_run_id = {row["run_id"]: row for row in latency_rows}
    assert by_run_id["m1_5_latency_512_16"]["input_len"] == "512"
    assert by_run_id["m1_5_latency_512_16"]["ttft_p50_ms"] == "90.0"
    assert apc_rows[0]["prefix_len"] == "1024"
    assert apc_rows[0]["suffix_len"] == "128"


def test_parse_directory_marks_oom_and_missing_percentiles(tmp_path):
    parse_directory(FIXTURES, tmp_path)

    rows = read_csv(tmp_path / "vllm_latency.csv")
    by_run_id = {row["run_id"]: row for row in rows}

    oom = by_run_id["m1_5_latency_32768_128"]
    assert oom["status"] == "OOM"
    assert oom["total_errors"] == "1"

    missing = by_run_id["m1_5_latency_missing"]
    assert missing["status"] == "OK"
    assert missing["ttft_p50_ms"] == ""
    assert missing["tpot_p99_ms"] == ""


def test_parse_result_ignores_empty_error_slots(tmp_path):
    result = tmp_path / "vllm_empty_errors.json"
    result.write_text(
        """
{
  "label": "m1_5_empty_errors",
  "model_id": "/root/models/gpt-oss-20b",
  "request_rate": 1.0,
  "max_concurrency": 1,
  "num_prompts": 2,
  "completed": 2,
  "failed": 0,
  "input_lens": [128, 128],
  "output_lens": [8, 8],
  "errors": ["", ""],
  "p50_ttft_ms": 12.0
}
""",
        encoding="utf-8",
    )

    row = parse_result(result)

    assert row["status"] == "OK"
    assert row["total_errors"] == 0
    assert row["input_len"] == 128
    assert row["output_len"] == 8
