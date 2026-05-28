import csv

from benchmarks.m1_5.probe_vllm_capacity import (
    CapacityProbeResult,
    parse_vllm_serve_log,
    write_capacity_csv,
)


def test_parse_vllm_serve_log_detects_started_capacity():
    text = """
INFO [uva.py:61] Total CPU offloaded parameters: 24.28
INFO [kv_cache_utils.py:1319] GPU KV cache size: 10,720 tokens
INFO [kv_cache_utils.py:1324] Maximum concurrency for 2,048 tokens per request: 5.23x
INFO:     Application startup complete.
"""

    result = parse_vllm_serve_log(text, max_model_len=2048)

    assert result.status == "STARTED"
    assert result.gpu_kv_cache_tokens == 10720
    assert result.max_concurrency == 5.23
    assert result.cpu_offloaded_parameters_gb == 24.28


def test_parse_vllm_serve_log_detects_kv_capacity_failure():
    text = """
ValueError: To serve at least one request with the models's max seq len (16,384),
(4.00 GiB KV cache is needed, which is larger than the available KV cache memory
(2.50 GiB). Based on the available memory, the estimated maximum model length is 10,240.
Try increasing `gpu_memory_utilization` or decreasing `max_model_len`
"""

    result = parse_vllm_serve_log(text, max_model_len=16384)

    assert result.status == "KV_CAPACITY_ERROR"
    assert result.needed_kv_cache_gib == 4.0
    assert result.available_kv_cache_gib == 2.5
    assert result.estimated_max_model_len == 10240
    assert "needed" in result.error_summary


def test_parse_vllm_serve_log_detects_weight_oom():
    text = """
Failed to load model - not enough GPU memory.
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 100.00 MiB.
"""

    result = parse_vllm_serve_log(text, max_model_len=2048)

    assert result.status == "WEIGHT_OOM"
    assert "out of memory" in result.error_summary.lower()


def test_write_capacity_csv_preserves_rows(tmp_path):
    output = tmp_path / "capacity.csv"
    write_capacity_csv(
        output,
        [
            CapacityProbeResult(
                max_model_len=2048,
                status="STARTED",
                gpu_kv_cache_tokens=10720,
                max_concurrency=5.23,
            ),
            CapacityProbeResult(max_model_len=16384, status="KV_CAPACITY_ERROR"),
        ],
    )

    with output.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert [row["max_model_len"] for row in rows] == ["2048", "16384"]
    assert [row["status"] for row in rows] == ["STARTED", "KV_CAPACITY_ERROR"]
