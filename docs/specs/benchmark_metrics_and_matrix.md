# M1.5 Benchmark Metrics and Matrix

## 目标

M1.5 在仿真前建立真实测量输入。第一轮只做 vLLM/APC baseline characterization 和本地 NVMe/H2D 校准，不修改 vLLM 内核、connector 或 KV cache manager。

## 固定默认

| 项 | 默认 |
| --- | --- |
| 模型 | `/root/models/gpt-oss-20b` |
| vLLM endpoint | `http://127.0.0.1:8000/v1/completions` |
| 结果目录 | `results/benchmark_calibration/` |
| 存储路径 | 本地 NVMe lower-bound |
| 代表性限制 | `gpt-oss-20b` 有 `sliding_window=128`，不代表最终 dense 1M attention |

## 指标

| 指标 | 来源 | 用途 |
| --- | --- | --- |
| TTFT P50/P90/P99 | `vllm bench serve` | admission baseline |
| TPOT/ITL P50/P90/P99 | `vllm bench serve` | decode cost model |
| E2E P50/P90/P99 | `vllm bench serve` | 用户可感知延迟 |
| request/output throughput | `vllm bench serve` | queueing model |
| APC prefix reuse benefit | `prefix_repetition` dataset | `PrefixIndex` / `KVSharingPlan` 动机 |
| NVMe read p50/p90/p99 | `run_io_microbench.py` | restore time model |
| H2D bandwidth | `run_h2d_microbench.py` | DRAM -> HBM promotion model |

## Benchmark Matrix

### B0 Environment

```bash
python benchmarks/m1_5/collect_env.py \
  --model /root/models/gpt-oss-20b \
  --output results/benchmark_calibration/env.json
```

### B1/B2 vLLM random workload

Use:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  VLLM_PLUGINS= \
  python benchmarks/m1_5/run_vllm_bench.py \
  --config configs/m1_5_benchmark.yaml \
  --suite random_smoke \
  --dry-run
```

Remove `--dry-run` only after a vLLM server is already running.
Use `VLLM_PLUGINS=` to avoid local stale plugin entry points and `NO_PROXY` to keep localhost traffic off the machine-wide socks proxy.

### B3 APC prefix repetition

Use:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  VLLM_PLUGINS= \
  python benchmarks/m1_5/run_vllm_bench.py \
  --config configs/m1_5_benchmark.yaml \
  --suite apc_sweep \
  --dry-run
```

Compare `prefix_repetition` rows against random rows with similar total input length. The first report should avoid claiming full APC causality unless prefix caching on/off server configs are explicitly controlled.

### B5 NVMe and H2D

```bash
python benchmarks/m1_5/run_io_microbench.py
python benchmarks/m1_5/run_h2d_microbench.py
```

For smoke runs that avoid large files or 4GB pinned allocations:

```bash
python benchmarks/m1_5/run_io_microbench.py \
  --output /tmp/m1_5_hardware_io.csv \
  --data-file /tmp/m1_5_io_probe.bin \
  --size-mb 1 \
  --repeats 1 \
  --sizes 4K

python benchmarks/m1_5/run_h2d_microbench.py \
  --output /tmp/m1_5_h2d_bandwidth.csv \
  --warmups 1 \
  --repeats 1 \
  --sizes 64MB
```

## Result Pipeline

```bash
python benchmarks/m1_5/parse_vllm_results.py \
  --raw-dir results/benchmark_calibration/raw \
  --output-dir results/benchmark_calibration

python benchmarks/m1_5/build_simulator_params.py \
  --result-dir results/benchmark_calibration
```

## Acceptance Criteria

- `env.json` records vLLM, torch, CUDA, GPU, DRAM, block devices, git status, and model config.
- `vllm_latency.csv` and `apc_prefix_reuse.csv` preserve failed/OOM rows with `status`, not by omission.
- If a `vllm bench serve` command exits non-zero before writing JSON, `run_vllm_bench.py` writes a failure JSON row with the command metadata and error text.
- `hardware_io.csv` records whether it used `python_fallback_fio_available` or `python_fallback_fio_missing`.
- `h2d_bandwidth.csv` records CUDA unavailable or per-size errors rather than crashing without output.
- `simulator_params.yaml` is generated only from measured CSV files.
- `report.md` summarizes environment, vLLM baseline rows, APC prefix rows, hardware calibration, anomalies, and the `gpt-oss-20b` sliding-window limitation.
