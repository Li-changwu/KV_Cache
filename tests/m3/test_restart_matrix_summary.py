from pathlib import Path

from benchmarks.m3.summarize_restart_matrix import summarize_restart_matrix


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "run_id,prefix_tokens,suffix_tokens,output_tokens,phase,status,"
        "first_status_code,second_status_code,first_ttft_ms,second_ttft_ms,"
        "reuse_tokens,decision,error,manifest_bytes,store_elapsed_ms,"
        "load_elapsed_ms,store_events,load_events,external_load_observed\n"
    )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_summarize_restart_matrix_requires_external_load_for_every_reuse_row(tmp_path):
    _write_csv(
        tmp_path / "store_phase" / "reuse_smoke_matrix.csv",
        [
            "reuse_prefix_16,16,16,1,store,OK,200,,100,,16,,,77,10,,48,0,no",
            "reuse_prefix_64,64,16,1,store,OK,200,,100,,64,,,86,20,,48,0,no",
        ],
    )
    _write_csv(
        tmp_path / "reuse_phase" / "reuse_smoke_matrix.csv",
        [
            "reuse_prefix_16,16,16,1,reuse,OK,,200,,120,16,,,77,20,3,96,1,yes",
            "reuse_prefix_64,64,16,1,reuse,OK,,200,,140,64,,,86,40,4,96,1,yes",
        ],
    )

    summary = summarize_restart_matrix(tmp_path)

    assert summary["status"] == "OK"
    assert summary["reuse_rows"] == 2
    assert summary["external_load_rows"] == 2
    assert summary["all_external_load_observed"] is True
    report = (tmp_path / "restart_matrix_report.md").read_text(encoding="utf-8")
    assert "完整重启对照矩阵" in report
    assert "外部加载覆盖：`2/2`" in report
    assert "reuse_prefix_64" in report


def test_summarize_restart_matrix_marks_missing_external_load_as_partial(tmp_path):
    _write_csv(
        tmp_path / "store_phase" / "reuse_smoke_matrix.csv",
        ["reuse_prefix_16,16,16,1,store,OK,200,,100,,16,,,77,10,,48,0,no"],
    )
    _write_csv(
        tmp_path / "reuse_phase" / "reuse_smoke_matrix.csv",
        ["reuse_prefix_16,16,16,1,reuse,OK,,200,,120,16,,,77,20,,96,0,no"],
    )

    summary = summarize_restart_matrix(tmp_path)

    assert summary["status"] == "PARTIAL"
    assert summary["all_external_load_observed"] is False
