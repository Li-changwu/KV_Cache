import csv
import sys
from types import SimpleNamespace

from benchmarks.m1_5.run_h2d_microbench import run_bench as run_h2d_bench
from benchmarks.m1_5.run_io_microbench import run_bench as run_io_bench


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_io_microbench_accepts_size_subset(tmp_path):
    output = tmp_path / "hardware_io.csv"
    data_file = tmp_path / "io_probe.bin"

    run_io_bench(output, data_file, size_mb=1, repeats=1, selected_sizes=["4K"])

    rows = read_csv(output)
    assert [row["block_size_label"] for row in rows] == ["4K"]
    assert rows[0]["status"] == "OK"
    assert rows[0]["method"].startswith("python_fallback")


def test_h2d_microbench_accepts_size_subset(tmp_path, monkeypatch):
    class FakeTensor:
        def copy_(self, other, non_blocking=False):
            return self

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        float32="float32",
        empty=lambda *args, **kwargs: FakeTensor(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    output = tmp_path / "h2d_bandwidth.csv"
    run_h2d_bench(output, warmups=1, repeats=1, selected_sizes=["64MB"])

    rows = read_csv(output)
    assert [row["size_label"] for row in rows] == ["64MB"]
    assert rows[0]["status"] == "OK"
