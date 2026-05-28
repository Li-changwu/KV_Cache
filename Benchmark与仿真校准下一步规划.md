# Benchmark 与仿真校准下一步规划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-05-24
- Verification Status: UNVERIFIED
- Version Label: benchmark_calibration_plan_v1

## 1. 结论先行

最紧要的下一步不是直接进入完整仿真，也不是马上改 vLLM connector，而是插入一个 **M1.5：Benchmark 与硬件校准阶段**。

原计划中的“仿真先行”需要修正为：

```text
M1 指标/schema 固化
  -> M1.5 vLLM Benchmark + 硬件校准
  -> M2 硬件参数驱动的仿真
  -> M3 vLLM 最小原型接入
```

原因很简单：仿真如果没有真实硬件参数，会很容易把 3FS/NVMe/DRAM/H2D 传输、vLLM prefill/decode 行为、APC 命中收益和 queueing tail latency 估错。那样仿真结果即使漂亮，也无法支撑论文或工程决策。

## 2. 为什么 Benchmark 要排在仿真前

当前方案的关键约束都是数值型约束：

- `TTFT_offload <= 1.2 * TTFT_DRAM_baseline`
- `effective_hit_rate >= 90%`
- `synchronous_3fs_miss_rate == 0`
- storage/network sustained utilization `<= 70%`
- committed ranges 不默认 full prefill 重算

这些约束依赖真实系统数据：

| 仿真参数 | 必须由 Benchmark 给出 |
| --- | --- |
| `TTFT_full_prefill_baseline` | vLLM 长 prompt prefill 实测 |
| `TTFT_delta_prefill_baseline` | prefix/session 复用或短 delta prompt 实测 |
| `first_decode_step_time` / TPOT | vLLM decode 实测 |
| `prefix_cache_hit_benefit` | vLLM APC / prefix repetition 实测 |
| `kv_bytes_per_token` | 模型 config + KV layout |
| HBM 可用容量与 block 容量 | vLLM KV cache usage / GPU memory |
| DRAM pinned buffer 能力 | 本机内存与 pinned H2D microbenchmark |
| SSD/3FS restore bandwidth 与 tail latency | fio / KV-shaped read microbenchmark |
| queueing 模型 | `vllm bench serve` 的 request rate / concurrency sweep |

所以，Benchmark 不是附属实验，而是仿真的输入层。

## 3. 当前可用实验环境初判

本机已具备第一轮基线条件：

| 项 | 当前观测 |
| --- | --- |
| vLLM | `0.19.0` |
| GPU | NVIDIA RTX A6000, 46068 MiB |
| CPU | Intel Xeon Platinum 8368Q, 2 sockets, 152 logical CPUs |
| DRAM | 503 GiB |
| 本地 NVMe | Samsung SSD 990 PRO 2TB |
| 本地模型 | `/root/models/gpt-oss-20b`、`/root/models/Qwen3-32B`、`/root/models/gemma-4-26B-A4B-it` 等 |

限制也很明确：A6000 48GB 不适合直接跑真实 `1M tokens` 推理，但足够做长上下文趋势、APC/prefix reuse、vLLM queueing、KV 容量模型和本地 NVMe/DRAM/H2D 校准。

## 4. M1.5 的核心研究问题

### RQ-B1：vLLM 原生长上下文性能基线是什么？

要回答：

- 在当前 GPU 和模型上，input length 增长时 TTFT 如何变化？
- output length 增长时 TPOT/ITL 如何变化？
- concurrency / request rate 对 TTFT P50/P90/P99 有多大影响？
- chunked prefill、prefix caching、max batched tokens 等配置对结果有什么影响？

### RQ-B2：vLLM APC 对共享 prefix 的实际收益是多少？

要回答：

- 相同 prefix + 不同 suffix 的请求，TTFT 是否显著下降？
- prefix length 越长，收益是否线性增加？
- prefix repetition dataset 的结果能否解释 `PrefixIndex` 和 `KVSharingPlan` 的必要性？
- APC 命中收益能否作为 `session_prefill_hit_rate` / `prefix_prefill_hit_rate` 的现实上界？

### RQ-B3：硬件 restore path 的真实参数是什么？

要回答：

- 本地 NVMe 顺序读、随机读、不同 block size 下的 tail latency 是多少？
- 从 page cache / direct I/O 到 DRAM 的差异有多大？
- pinned DRAM -> GPU H2D 带宽是多少？
- KV-shaped block 批量读取时，restore time 是否能在 deadline 前完成？

### RQ-B4：仿真器最少需要哪些硬件参数？

最终 M2 仿真器至少要吃这些输入：

```yaml
hardware:
  gpu_name: NVIDIA RTX A6000
  hbm_total_bytes: measured
  hbm_usable_kv_bytes: measured_or_estimated
  dram_total_bytes: measured
  dram_pinned_budget_bytes: measured_or_configured
  nvme_seq_read_gbps: measured
  nvme_random_read_iops_by_block: measured
  nvme_read_p50_p90_p99_ms_by_size: measured
  h2d_gbps: measured
vllm:
  version: 0.19.0
  model_id: selected
  kv_block_tokens: measured_or_configured
  ttft_by_input_len: measured
  tpot_by_context_len: measured
  apc_hit_benefit_by_prefix_len: measured
  kv_cache_usage_by_context_len: measured
```

## 5. Benchmark 指标表

### 第一优先级指标

| 指标 | 来源 | 用途 |
| --- | --- | --- |
| TTFT P50/P90/P99 | `vllm bench serve` / Prometheus metrics | admission baseline |
| TPOT/ITL P50/P90/P99 | `vllm bench serve` / metrics | decode cost model |
| E2E latency | `vllm bench serve` | 用户可感知延迟 |
| request throughput | `vllm bench serve` | queueing / serving capacity |
| prompt throughput | vLLM benchmark output / metrics | prefill throughput |
| generation throughput | vLLM benchmark output / metrics | decode throughput |
| KV cache usage | vLLM metrics | HBM KV pressure |
| GPU memory used | `nvidia-smi` | HBM capacity model |
| prefix hit benefit | prefix repetition workload | Prefill reuse model |
| NVMe read p50/p90/p99 | fio / microbench | restore time model |
| H2D bandwidth | torch microbench | DRAM -> HBM promotion model |

### 第二优先级指标

| 指标 | 用途 |
| --- | --- |
| preemption / swapped requests | 判断 HBM 压力和调度副作用 |
| CPU memory / pinned memory usage | DRAM tier 可用性 |
| page cache hit/miss 影响 | 区分真实 SSD 与缓存读 |
| block size 对读放大影响 | 决定 `32/64/128/256 tokens` 消融范围 |
| short request P99 under mixed load | 服务隔离目标 |

## 6. 第一轮 Benchmark 矩阵

### B0：环境与容量探针

目标：记录可复现实验环境。

输出：

- `results/benchmark_calibration/env.json`
- `results/benchmark_calibration/model_kv_capacity.json`
- `results/benchmark_calibration/hardware_summary.md`

### B1：vLLM 单请求 latency sweep

目标：测 input length 对 TTFT 和 first decode 的影响。

建议 sweep：

| 变量 | 值 |
| --- | --- |
| model | 先用本地可跑的 `gpt-oss-20b` 或更小模型 |
| input_len | `512 / 1K / 2K / 4K / 8K / 16K / 24K / 32K`，以显存可承受为准 |
| output_len | `1 / 16 / 128` |
| batch/concurrency | `1` |
| prefix caching | off / on |

### B2：vLLM online serving sweep

目标：测 queueing 和混部服务行为。

建议 sweep：

| 变量 | 值 |
| --- | --- |
| request_rate | `1 / 2 / 4 / 8 req/s` 或根据 smoke 结果调整 |
| max_concurrency | `1 / 2 / 4 / 8` |
| input_len | `2K / 8K / 16K / 32K` |
| output_len | `128` |
| dataset | random |

输出 TTFT、TPOT、ITL、E2E 的 P50/P90/P99。

### B3：prefix repetition / APC sweep

目标：测共享 prefix 的实际收益，直接服务 `PrefixIndex` 和 `KVSharingPlan` 论证。

建议使用 vLLM `prefix_repetition` dataset：

| 变量 | 值 |
| --- | --- |
| num_prefixes | `1 / 4 / 16` |
| prefix_len | `1K / 4K / 8K / 16K` |
| suffix_len | `128 / 512 / 2K` |
| output_len | `128` |
| prefix caching | off / on |

关键比较：

```text
TTFT(prefix_caching_on) vs TTFT(prefix_caching_off)
TTFT(shared_prefix) vs TTFT(random_prompt_same_len)
```

### B4：Mixed serving 干扰实验

目标：测长 prompt prefill 对短请求 P99 的干扰。

建议两类流量混合：

- short realtime：`input_len=512`，`output_len=128`
- long prefill：`input_len=16K/32K`，`output_len=128`

输出：

- short-only P99 TTFT
- mixed-load short P99 TTFT
- long request TTFT
- GPU KV usage 和 preemption

### B5：硬件 I/O 与 restore microbench

目标：为 3FS/SSD restore 仿真提供参数。

任务：

- fio 测 NVMe 顺序读、随机读、不同 block size tail latency。
- 生成 KV-shaped segment 文件，测批量读取到 DRAM 的耗时。
- 用 torch 测 pinned DRAM -> GPU H2D 带宽。
- 如有 3FS 环境，再替换 NVMe 为 3FS 客户端路径重复 B5。

## 7. 仿真前置门槛

只有拿到以下文件，才进入 M2 仿真：

| 文件 | 内容 |
| --- | --- |
| `results/benchmark_calibration/vllm_latency.csv` | TTFT/TPOT/ITL/E2E by input/output/concurrency |
| `results/benchmark_calibration/apc_prefix_reuse.csv` | prefix caching 收益 by prefix_len/suffix_len |
| `results/benchmark_calibration/hardware_io.csv` | NVMe/3FS read latency 和 bandwidth |
| `results/benchmark_calibration/h2d_bandwidth.csv` | DRAM -> HBM 传输带宽 |
| `results/benchmark_calibration/simulator_params.yaml` | 仿真器输入参数 |
| `results/benchmark_calibration/report.md` | 对 vLLM 现状和仿真参数的解释 |

如果这些文件没有形成，M2 的仿真只允许作为 toy model，不应用来支撑论文 claim。

## 8. 我建议马上执行的 3 个动作

1. **写 `docs/specs/benchmark_metrics_and_matrix.md`**
   把上述 B0-B5 的指标、命令、结果路径和成功标准固化。

2. **写最小 benchmark harness**
   先封装环境采集、vLLM bench serve 参数矩阵、结果 JSON 汇总，不碰 connector。

3. **跑一个 smoke benchmark**
   用本地 `gpt-oss-20b` 或更稳的小模型跑 `input_len=512/2048`、`output_len=16`、`concurrency=1`，验证 vLLM bench 输出、Prometheus metrics、GPU 监控和结果落盘链路。

## 9. 需要讨论确认的问题

1. 第一轮 benchmark 模型用哪个？我倾向先用 `/root/models/gpt-oss-20b`，因为它在本机 A6000 上最可能跑通。
2. 我们是否接受第一轮只测到 `32K` 或显存允许的最大上下文，然后用 KV 容量模型外推 `1M`？
3. 是否有真实 3FS 环境？如果没有，B5 先用本地 NVMe 作为 lower-bound / shape calibration。
4. 你更想先证明哪条主张：vLLM 当前长上下文瓶颈、APC/prefix reuse 收益，还是 3FS restore deadline 可行性？
