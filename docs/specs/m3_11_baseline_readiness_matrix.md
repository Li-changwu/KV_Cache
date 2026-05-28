# M3.11 Baseline Readiness Matrix

日期：2026-05-27

## 0. 目标

M3.11 的目标不是继续扩 3FS adapter，也不是直接冲 1M。它要回答一个更基础的问题：

> 当前 Persistent KV Anti-Caching 路径，在 512-32K 的真实在线范围内，是否已经比 full prefill 或 naive cold restore 更有价值？

如果 M3.11 不能在高复用 workload 上证明收益，就不应继续投入真实 3FS executor、128K/1M 大实验或更复杂的数据面优化。

## 1. 问题边界

当前项目主线是 **Persistent KV Anti-Caching for Multi-Turn Long-Context Serving**。

本阶段只验证 exact path：

- 不引入 sparse attention。
- 不引入 lossy KV compression。
- 不引入 semantic retrieval replacement。
- 不把 SSD/3FS 当透明 HBM 扩展。
- 不允许 Prefill / Decode 执行期同步读取 SSD/3FS。

KVDrive、LMCache、Mooncake、Tutti、DualPath、CacheFlow 都是后续强 baseline。M3.11 第一轮先建立自有 B0-B5 矩阵，避免在系统自身口径未固定时过早接入外部系统。

## 2. 第一轮 Baseline

| 编号 | 名称 | 目的 | 必须回答的问题 |
|---|---|---|---|
| B0 | vLLM full prefill, APC off | 重算下界 | 不复用历史 KV 时 TTFT 有多高？ |
| B1 | vLLM APC hot-cache | 最强本地热前缀 baseline | 同进程热前缀下，原生 APC 有多强？ |
| B2 | vLLM APC restart / cold-cache | APC 失效区域 | vLLM 重启或 cache 失效后，APC 是否还能复用？ |
| B3 | external DRAM-ready KV reuse | offload-free reuse denominator | KV 已在执行可达层时，外部复用路径额外开销是多少？ |
| B4 | naive cold restore / synchronous offload | 反例 baseline | 如果不做 admission barrier，cold restore 如何污染 TTFT/tail？ |
| B5 | current KV Anti-Caching: `DELAY -> restore -> ADMIT` | 当前方法 | ready barrier + restore queue 是否降低风险并保留收益？ |

第一轮不要求 B5 赢 B1 hot-cache。B1 是本地热缓存上界；B5 的目标场景是 restart、capacity pressure、idle session restore、跨 engine/session 的 committed historical KV。

## 3. 模型与运行范围

### 3.1 默认模型

第一轮默认模型仍使用：

```text
/root/models/Qwen2.5-14B-Instruct
```

原因：

- 当前 vLLM connector / tensor store / restart cold-tier 路径已经在该模型上验证。
- 单卡 RTX A6000 48GB 上可启动到 32768。
- 适合先固定 benchmark schema 和系统口径。

### 3.2 候选长上下文模型

`gradientai/Llama-3-8B-Instruct-Gradient-1048k` 暂列为候选长上下文基座，但不进入上午第一轮 B0-B5。

进入矩阵前必须先完成：

- 下载与 license / disk 检查。
- 32K / 64K / 128K vLLM capacity probe。
- connector 16/64/256 restart cold-tier smoke。
- `kv_bytes_per_token=131072` 写入 calibration。
- correctness key 覆盖 `model_id`、tokenizer hash、`rope_theta=3580165449.0`、layout、block size。

### 3.3 Prefix / Suffix Matrix

第一轮真实在线矩阵：

| 维度 | 取值 |
|---|---|
| prefix tokens | 512, 2048, 8192, 16384, 32768 |
| suffix tokens | 128, 512, 2048 |
| output tokens | 1, 16 |
| concurrency | 1 first; 2/4 after smoke |
| repeats | smoke=1; report=3; paper>=5 |

如果 32768 + suffix + output 超过 `max_model_len`，保留 `ERROR_BOUNDARY` 行，不静默丢弃。

## 4. Workloads

### 4.1 `session_append`

模拟同一个 agent session 多轮追加：

```text
turn_0: long history prefix
turn_1: reuse committed history + append suffix
turn_2: reuse previous committed history + append suffix
```

期望：

- B0 会重算历史 prompt。
- B3/B5 应只对新增 suffix 做 delta prefill。
- historical KV hit rate 应接近 1，除非 correctness mismatch 或 capacity miss。

### 4.2 `shared_prefix`

模拟多个请求共享长文档 / 长系统提示：

```text
request_i = shared_long_prefix + request_specific_suffix_i
```

期望：

- B1 hot-cache 应很强。
- B2 restart 后应暴露 APC 失效边界。
- B5 应在 cold restore 后恢复 shared prefix reuse。

### 4.3 `low_locality`

随机或近似随机长上下文，无可复用历史：

```text
request_i = unique_long_context_i
```

期望：

- B5 不应强行 `ADMIT` 到 SLA 快路径。
- 正确行为是 `FULL_PREFILL_FALLBACK`、`RELAXED_QUEUE`、`DELAY` 或 `REJECT`。
- 该 workload 是负对照，不纳入 admitted high-reuse hit-rate 目标。

### 4.4 `mixed_short_long`

短请求与长请求混部：

```text
short requests: 128-512 tokens
long reusable requests: 8K-32K prefix + suffix
```

期望：

- 观察 short-request p95/p99 TTFT 是否被 cold restore 污染。
- B5 应通过 admission / queue 将 cold I/O 移出短请求 critical path。

第一轮如果时间有限，先做 `session_append` 和 `shared_prefix`，`low_locality` 至少保留 smoke，`mixed_short_long` 可放到第二轮。

## 5. 输出目录

统一结果目录：

```text
results/m3_11_baseline_readiness/
  env.json
  run_config.yaml
  raw/
  baseline_matrix.csv
  request_metrics.csv
  sidecar_decisions.jsonl
  connector_events.jsonl
  restore_events.jsonl
  summary/
    baseline_summary.csv
    workload_summary.csv
    readiness_report.md
```

每个实验行必须带 `run_id`，便于 raw JSON、sidecar log、connector log 和 CSV 对齐。

## 6. CSV Schema

### 6.1 `baseline_matrix.csv`

必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `run_id` | string | 全局唯一实验行 ID |
| `timestamp_utc` | string | ISO 时间 |
| `model_id` | string | 模型路径或 HF repo |
| `model_context_limit` | int | 配置上下文长度 |
| `kv_bytes_per_token` | int | 模型 KV bytes/token |
| `backend` | string | `vllm`, `sidecar_connector`, `simulated` |
| `baseline_id` | enum | `B0` 到 `B5` |
| `workload` | enum | `session_append`, `shared_prefix`, `low_locality`, `mixed_short_long` |
| `prefix_tokens` | int | 目标 prefix token 数 |
| `actual_prefix_tokens` | int | tokenizer 后实际 prefix token 数 |
| `suffix_tokens` | int | 目标 suffix token 数 |
| `actual_suffix_tokens` | int | tokenizer 后实际 suffix token 数 |
| `output_tokens` | int | 目标输出 token 数 |
| `concurrency` | int | 并发 |
| `repeat_id` | int | 重复编号 |
| `status` | enum | `OK`, `ERROR`, `OOM`, `TIMEOUT`, `ERROR_BOUNDARY` |
| `error_type` | string | 错误类型 |
| `error_message` | string | 截断错误消息 |

延迟字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ttft_ms` | float | 单请求 TTFT |
| `e2e_ms` | float | 端到端耗时 |
| `itl_p50_ms` | float | ITL p50 |
| `itl_p95_ms` | float | ITL p95 |
| `tpot_p50_ms` | float | TPOT p50 |
| `tpot_p95_ms` | float | TPOT p95 |
| `queue_wait_ms` | float | sidecar / restore queue 等待 |
| `ready_barrier_wait_ms` | float | Prefill/Decode barrier 等待 |

复用与 prefill 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `reuse_tokens` | int | 实际复用 token 数 |
| `delta_prefill_tokens` | int | 新增 prefill token 数 |
| `prefill_tokens_saved` | int | 相对 full prefill 节省 token 数 |
| `required_historical_tokens` | int | 本轮需要的历史 KV token 数 |
| `historical_kv_hit_tokens` | int | 历史 KV 命中 token 数 |
| `historical_kv_hit_rate` | float | `historical_kv_hit_tokens / required_historical_tokens` |
| `historical_bytes_required` | int | 历史 KV 所需字节 |
| `historical_bytes_hit` | int | 历史 KV 命中字节 |
| `historical_byte_hit_rate` | float | `historical_bytes_hit / historical_bytes_required` |

restore 与 cold-tier 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `cold_probe_decision` | string | 首次 cold probe 决策 |
| `restore_status` | string | `COMPLETED`, `DEADLINE_MISS`, `FAILED`, `NA` |
| `restore_actual_bytes` | int | 实际 restore 字节 |
| `restore_useful_bytes` | int | 对本轮有用的 restore 字节 |
| `restore_bytes_per_useful_byte` | float | `actual / useful` |
| `restore_elapsed_ms` | float | restore 执行耗时 |
| `restore_batch_ms` | float | batch restore wall time |
| `restore_checksum_status` | string | `ok`, `failed`, `NA` |
| `storage_backend` | string | `dram`, `local_posix`, `3fs_posix`, `none` |
| `prefetch_deadline_miss` | bool | 是否 deadline miss |
| `sync_cold_miss_total` | int | 执行期同步冷层 miss，SLA 队列必须为 0 |

决策字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `admission_decision` | string | `ADMIT`, `DELAY_RESTORE`, `FULL_PREFILL_FALLBACK`, `RELAXED_QUEUE`, `REJECT` |
| `admission_reason` | string | 决策原因 |
| `prefill_reuse_ready` | bool | Prefill 复用 barrier |
| `decode_execution_ready` | bool | Decode 执行 barrier |
| `external_load_observed` | bool | connector 是否观测到外部 load |
| `connector_load_events` | int | load event 数量 |
| `connector_store_events` | int | store event 数量 |

ratio 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ttft_vs_full_prefill` | float | `B5 ttft / B0 ttft`，按同 workload/shape 对齐 |
| `ttft_vs_dram_ready_reuse` | float | `B5 ttft / B3 ttft` |
| `ttft_vs_apc_hot` | float | `B5 ttft / B1 ttft`，只作参考，不作为第一轮硬门槛 |

### 6.2 `baseline_summary.csv`

按 `(baseline_id, workload, prefix_tokens, suffix_tokens, output_tokens, concurrency)` 聚合：

- `ok_count`
- `error_count`
- `ttft_p50_ms`
- `ttft_p95_ms`
- `ttft_p99_ms`
- `itl_p95_ms`
- `tpot_p95_ms`
- `historical_kv_hit_rate_mean`
- `historical_byte_hit_rate_mean`
- `prefetch_deadline_miss_rate`
- `sync_cold_miss_rate`
- `external_load_observed_rate`
- `restore_elapsed_p50_ms`
- `restore_elapsed_p95_ms`
- `restore_bytes_per_useful_byte_mean`

## 7. Ratio 计算规则

所有 ratio 必须在同一个 group 内计算：

```text
group = model_id, workload, prefix_tokens, suffix_tokens, output_tokens, concurrency
```

重复实验先取中位数，再计算 ratio，避免单个长尾污染。

定义：

```text
ttft_vs_full_prefill = median_ttft(B5) / median_ttft(B0)
ttft_vs_dram_ready_reuse = median_ttft(B5) / median_ttft(B3)
ttft_vs_apc_hot = median_ttft(B5) / median_ttft(B1)
```

如果 baseline 缺失，ratio 写空，并在 report 中列为 `MISSING_BASELINE`。

## 8. 判定标准

### 8.1 继续推进条件

若满足以下条件，进入 packed object / restore executor 优化：

- `session_append` 或 `shared_prefix` 在 8K/16K/32K 至少两个规模上，B5 相对 B0 有明显 TTFT 或 prefill token 节省。
- B5 相对 B4 降低 sync cold miss 风险，并且 `sync_cold_miss_total == 0`。
- B5 的 `external_load_observed_rate > 0`，不能被 vLLM APC 热缓存遮蔽。
- B5 在 admitted high-reuse class 的 `historical_byte_hit_rate >= 0.9`，或能解释未达标原因。
- low_locality 不被错误计入 high-reuse hit-rate。

### 8.2 暂停推进条件

若出现以下情况，应暂停真实 3FS executor 投入，先修系统设计：

- B5 在 8K/16K/32K 均不优于 B0。
- B5 与 B4 的 tail latency / sync miss 风险没有差异。
- B5 复用请求没有 connector external load event，说明实验被 APC hot-cache 污染。
- restore bytes/useful bytes 长期过高，说明缺少 packed object / partial restore。
- correctness key mismatch、layout mismatch、checksum failure 被吞掉，没有形成结构化错误行。

## 9. 实验污染控制

必须控制：

- `VLLM_PLUGINS=`，避免本机残留插件污染。
- `NO_PROXY=127.0.0.1,localhost`，避免 localhost 走 socks proxy。
- B1 hot-cache 和 B2 restart 必须分开记录。
- B5 reuse 阶段必须能证明 restart 后外部 connector load 发生。
- 所有失败/OOM/超长边界行必须写入 CSV。
- 每个 run 记录 git diff stat、vLLM version、CUDA、GPU memory、model config。

## 10. B0-B5 最小执行形态

### B0: full prefill

- 启动 vLLM，关闭或绕开 APC 影响。
- 对每个 prompt 做完整 prefill。
- 记录 TTFT / ITL / TPOT。

### B1: APC hot-cache

- 同一 vLLM 进程中，先 warm shared prefix。
- 再发送复用请求。
- 记录 hot-cache 下的 TTFT。

### B2: APC restart / cold-cache

- warm shared prefix 后重启 vLLM。
- 发送同样复用请求。
- 若无外部 KV，预期退化为 full prefill 或接近 full prefill。

### B3: external DRAM-ready reuse

- 通过当前 connector 保存 prefix KV。
- 不 demote 到 cold tier，保持 `DRAM/ready=true`。
- 重启 vLLM 后通过外部 connector load。
- 这是 B5 的 offload-free denominator。

### B4: naive cold restore

- 将 KV demote 到 cold tier。
- 不使用 admission/ready barrier，或在离线模拟中把 cold restore 计入执行期等待。
- 目标是构造反例：cold miss 污染 TTFT/tail。
- 若不安全或不方便在真实 vLLM 中实现，可先用 simulator / replay 生成 B4，并在 report 中标注。

### B5: current Anti-Caching

- store prefix KV。
- demote 到 cold tier。
- restart vLLM。
- cold probe 得到 `DELAY_RESTORE` 或 `DELAY`。
- restore/checksum ok。
- 再 admit，ready barrier 通过。
- 复用请求必须观测到 connector `load_request ok`。

## 11. Runner 设计建议

建议新增：

```text
benchmarks/m3/run_baseline_readiness_matrix.py
benchmarks/m3/summarize_baseline_readiness_matrix.py
tests/m3/test_baseline_readiness_matrix.py
```

runner 第一版只需要支持 dry-run 和已有 M3.10 reuse smoke matrix 的复用：

- `--phase {b0_full_prefill,b1_apc_hot,b2_apc_restart,b3_dram_ready,b4_naive_cold,b5_anti_caching}`
- `--prefix-tokens 512 2048 8192 16384 32768`
- `--suffix-tokens 128 512 2048`
- `--output-tokens 1 16`
- `--workload session_append shared_prefix low_locality`
- `--result-dir results/m3_11_baseline_readiness`
- `--dry-run`

先实现 schema 和 dry-run，再接真实 vLLM。

## 12. 产出物

本阶段完成后必须有：

- `results/m3_11_baseline_readiness/baseline_matrix.csv`
- `results/m3_11_baseline_readiness/summary/baseline_summary.csv`
- `results/m3_11_baseline_readiness/summary/readiness_report.md`
- `findings.md` 中的结论更新
- `task_plan.md` 中 P0 项状态更新

## 13. 明确不做

M3.11 第一轮不做：

- 不接真实 3FS executor。
- 不接 LMCache / Mooncake / DualPath / KVDrive-style 外部系统。
- 不跑 128K/1M。
- 不改 attention kernel。
- 不引入 sparse attention 或近似复用。
- 不把 low-locality 计入 high-reuse hit-rate。
