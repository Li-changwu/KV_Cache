# KV Anti-Caching 有效性评估与 Baseline 对比计划

日期：2026-05-26

## 0. 先给结论

当前方法**有用，但还没有被证明比强 baseline 更有效**。

更精确地说：

- 在**语义和系统路径**层面，当前 M3.10 原型已经有效：它能把 cold-tier KV 从“执行时才发现的 miss”变成“admission 前可检测、可延迟、可恢复、可校验、ready 后再执行”的状态。
- 相比**朴素 offloading / 透明 SSD miss**，当前方法已经有明确价值：它显式阻止 Prefill/Decode 执行期同步冷层 I/O，能保证 `sync_ssd_miss_total=0`，并把 miss 暴露为 `DELAY` 或 restore queue，而不是藏进 TTFT/ITL 尾延迟。
- 相比**vLLM APC、LMCache、Mooncake、Tutti、DualPath、CacheFlow、KVDrive**等强 baseline，当前还不能声称性能胜出：我们缺少 512-32K 在线矩阵、真实 3FS/生产 SSD、强 baseline 同机对比、p95/p99 TTFT/ITL、以及 admitted workload 上的 90% hit-rate 证据。

因此，当前最稳妥的科研表述是：

> KV Anti-Caching 当前已经验证了必要的 correctness/control-plane invariant，但尚未验证 end-to-end performance superiority。它是一个有希望的系统设计，不是已经完成的性能结论。

## 1. “有没有用”要拆成四个层次

| 层次 | 当前判断 | 证据强度 | 说明 |
|---|---|---:|---|
| 语义正确性 | 有用 | 强 | correctness key、manifest、tier state、ready barrier、checksum、restart 后外部 load 已走通。 |
| 容量层路径 | 有用但小规模 | 中 | 真实 cold object 文件迁移已实现，但只覆盖 16/64/128/256 小前缀。 |
| SLA 性能收益 | 未证明 | 弱 | M2 仿真显示高复用 workload 有希望，M3 在线还没有 512-32K/1M 性能矩阵。 |
| 强 baseline 胜出 | 未证明 | 弱 | 尚未同机对比 vLLM APC、LMCache、Mooncake/Tutti/DualPath-style 策略。 |

这个拆分很重要。我们不能把“路径跑通”说成“方法已经赢了”，但也不能因为还没跑强 baseline 就否定当前实现的价值。

## 2. 当前已经证明了什么

### 2.1 Cold-tier 不是透明执行内存

M3.10 的关键进步是：cold tier 被严格建模为 capacity tier，而不是 HBM/DRAM 的透明扩展。

当前请求流程是：

1. sidecar `/admit` 发现 required KV 在 cold tier 且 `ready=false`；
2. 返回 `DELAY`，原因是 `required_kv_not_ready_before_decode`；
3. `/prefetch/advance` 执行真实 cold object restore；
4. checksum 通过后，manifest 更新为 `DRAM/ready=true`；
5. 再次 admission 才返回 `ADMIT`；
6. vLLM connector 在复用阶段出现真实 `load_request ok`；
7. `sync_ssd_miss_total=0`。

这说明当前方法已经把 cold miss 从执行路径移到了控制路径。对 Prefill 来说，它避免把历史 KV cold miss 退化成隐藏的全量重算；对 Decode 来说，它避免同步 SSD/3FS I/O 进入 ITL/TPOT critical path。

### 2.2 Restart 后外部 KV load 确实发生

`results/m3_10_cold_tier_online/restart_matrix/reuse_phase/reuse_smoke_matrix.csv` 中，16/64/128/256 四个 prefix 均满足：

- `status=OK`
- `cold_probe_decision=DELAY`
- `cold_restore_status=COMPLETED`
- `cold_restore_checksum_status=ok`
- `decision=ADMIT`
- `ready_barrier_all_ready=true`
- `sync_ssd_miss_total=0`
- `external_load_observed=yes`
- `load_events=1`

对应 restore 字节数约为：

- 16 tokens：`3,149,952` bytes
- 64 tokens：`12,587,136` bytes
- 128 tokens：`25,170,048` bytes
- 256 tokens：`50,335,872` bytes

这不是单纯 mock metadata，而是真实文件迁移、真实 checksum、重启 vLLM 后真实 connector load 的端到端闭环。

### 2.3 当前 adapter benchmark 固定了评测格式，但不是 3FS 性能结论

`results/m3_10_cold_tier_adapter_bench_qd/summary/adapter_bench_comparison.md` 显示 queue-depth=4、`qwen25_14b_tiny` profile 下：

| backend | restore p95 | batch p95 | effective restore p50 |
|---|---:|---:|---:|
| `local_posix` | `11.334ms` | `13.584ms` | `545.648MiB/s` |
| `3fs_posix` | `45.969ms` | `68.175ms` | `202.162MiB/s` |

但这里的 `3fs_posix` 仍是本地目录模拟 3FS mount 的 POSIX adapter，不是 DeepSeek 3FS 集群/RDMA/GDS/io_uring 数据路径。它的价值是固定 backend 抽象、CSV schema 和报告格式；不能用来声称真实 3FS 性能。

### 2.4 M2 仿真支持“高复用有希望、低局部性应拒绝”的方向

Qwen2.5-14B sweep 显示：

- `low_locality` 在 250ms、1000ms、5000ms、12000ms、30000ms 下都没有 SLA region，最佳 admitted rate 仍为 0。
- `session_append` 在严格 250ms 下需要极高 H2D 并行能力，但在 5000ms/12000ms 等较宽松 deadline 下进入更现实区间。
- `shared_prefix` 同样需要高带宽和足够 HBM/DRAM residency。
- 当前全局 hit rate 约 `0.656-0.688`，低于目标 `>=90%`。

这说明我们的 admission boundary 是合理的：低局部性 1M random 不该进入精确 SLA 快路径。但这也说明当前 hit-rate 证据还不满足最终目标。

## 3. 当前没有证明什么

### 3.1 没有证明 1M SLA

当前真实在线矩阵只到 256 tokens。它证明了机制，不证明百万级长上下文。M2 的 1M 是控制面仿真，不是 vLLM 内核级真实执行。

必须补齐：

- 512 / 2K / 8K / 16K / 32K 在线矩阵；
- 128K / 1M 仿真与真实环境分层验证；
- packed cold object，避免每层小文件导致 metadata 和 syscall 放大；
- partial promote，避免恢复远大于本轮需要的 KV；
- 多资源调度，覆盖 storage、network/RDMA、DRAM、H2D、GPU slack。

### 3.2 没有证明 TTFT 增幅 <=20%

当前 CSV 中的 `second_ttft_ms` 很噪声化，小前缀 restore/load 的固定开销也很明显。我们还没有一个严肃定义的 denominator。

后续必须同时报告两个 ratio：

1. `TTFT_tiered / TTFT_full_prefill_no_reuse`：回答“相比完全重算是否节省 TTFT”。
2. `TTFT_tiered / TTFT_dram_ready_reuse`：回答“相比没有 cold offload、KV 已在 DRAM ready 的理想复用路径，offload 增幅是否 <=20%”。

第二个才最接近“较无卸载方案增幅 <=20%”这个指标。

### 3.3 没有证明 hit rate >=90%

当前 M2 中 `session_append` / `shared_prefix` 的 effective hit rate 约 65%-69%。这不能直接解释为失败，但必须重定义指标口径：

- `all_tokens_hit_rate`：所有 prompt tokens 的复用比例，会被新增 delta 拉低；
- `historical_kv_hit_rate`：required historical KV 中命中的比例；
- `byte_hit_rate`：required historical KV bytes 中命中的比例；
- `admitted_sla_class_hit_rate`：只统计被准入的高复用 workload，不把 low-locality negative control 混进去。

最终 `>=90%` 应该约束 `admitted_sla_class_historical_byte_hit_rate`，而不是所有请求、所有 tokens 的平均值。

### 3.4 没有证明强 baseline 对比优势

目前还没有同机、同模型、同 prompt、同并发的直接对比：

- vLLM full prefill / no APC；
- vLLM APC hot-cache；
- vLLM APC cold-cache / restart / capacity pressure；
- external DRAM KV reuse；
- naive cold offload / synchronous restore；
- LMCache；
- Mooncake / Mooncake Store；
- Tutti-style local SSD fast path；
- DualPath-style storage-to-prefill / storage-to-decode；
- CacheFlow-style 3D KV restoration；
- KVDrive-style holistic multi-tier GPU/DRAM/SSD orchestration。

所以目前不能写“我们的系统显著优于 baseline”。最多能写“现有 baseline 没有同时覆盖我们要求的 exact semantics + cold-tier admission + prefill/decode readiness invariant；性能优势需要下一阶段验证”。

## 4. 与主要 baseline 的当前判断

### 4.1 vLLM APC

vLLM APC 是最直接的强 baseline。APC 会对共享前缀复用 KV，从而跳过共享部分 prefill。对同一进程内的 hot prefix，它很可能比我们当前路径更快，因为不需要 cold restore 和外部 connector load。

我们的机会不在 APC hot-cache 场景，而在：

- vLLM 进程重启后；
- cache capacity 不足后；
- 跨 engine / 跨节点复用；
- 历史 KV 已经冷却到 DRAM/SSD/3FS 后；
- 需要 admission guard 避免 cold miss 污染 SLA 队列时。

因此实验必须分成 `APC-hot` 和 `APC-cold/restart/capacity-pressure` 两类。

### 4.2 LMCache

LMCache 已经把 KV cache 做成 engine 外的 cache layer，并强调可跨 engine 复用、可复用非 prefix 文本，官方文档声称在多轮 QA/RAG 等场景有显著 TTFT/GPU cycle 节省。

这对我们压力很大。我们不能只说“我们也能外部 KV 复用”。我们的差异必须是：

- strict correctness key；
- Prefill / Decode ready barrier；
- cold-tier admission，而不是执行期 miss；
- 3FS/SSD capacity tier 被纳入 SLA 预算；
- low-locality negative control 明确 delay/fallback/reject；
- exact 1M 长上下文目标。

### 4.3 Mooncake

Mooncake 是 KVCache-centric disaggregated architecture，是系统层强 baseline。它已经覆盖 disaggregated prefill/decode、KV transfer、KV-centric scheduling 等方向，并且 Mooncake Store 已经进入开源生态。

如果我们只做“外部 KV cache + scheduler”，很容易被 Mooncake 覆盖。我们的论文空间必须集中在：

- exact 1M context；
- 3FS-aware cold capacity；
- anti-caching 风格的 pre-pass / delay / partial promote；
- Prefill/Decode 双阶段 readiness；
- 对业务 SLA 的 admission boundary 和负对照。

### 4.4 Tutti

Tutti 直接面向 SSD-backed KV cache practical long-context serving，重点是减少 HBM 和 SSD 之间的 CPU 介入、加速 local SSD data path。它很可能在本地 SSD raw restore 性能上强于我们当前 POSIX 小文件路径。

我们的差异不应是“比 Tutti 更会读 SSD”，而应是：

- Tutti 更像 data plane / fast path；
- KV Anti-Caching 更像 control plane / policy semantics；
- 我们要决定哪些 KV 该冷却、何时 restore、何时准入、何时拒绝；
- 后续可以把 Tutti-style data plane 作为一种 executor 或 baseline。

### 4.5 DualPath

DualPath 针对 agentic LLM inference 的 storage bandwidth bottleneck，提出 storage-to-prefill 与 storage-to-decode 双路径，并用全局调度缓解 prefill 侧 storage NIC 饱和。

这和我们的目标高度重叠。它应该进入后续两种方案对比中的第二种策略：

- 方案 A：Admission-window Offloading，cold tier -> DRAM -> HBM；
- 方案 B：DualPath-style Restore，storage-to-prefill + storage-to-decode + RDMA transfer。

### 4.6 CacheFlow

CacheFlow 把 KV restoration 建模为 token、layer、GPU 三维并行问题，目标是降低 TTFT。它提示我们当前单资源 restore queue 太粗糙。若我们不做 packed object、layer/token chunk、并行 restore，性能上很可能输给这类 restoration scheduler。

### 4.7 KVDrive

KVDrive 是近期非常贴近的 multi-tier KV baseline。它同样把 GPU memory、host DRAM 和 SSD 作为整体系统来管理，并强调 placement、pipeline scheduling 与 cross-tier coordination。它对我们的压力是：单纯说“多级 KV 管理”已经不够新。

我们的差异应进一步收窄到：

- exact committed KV reuse，不依赖稀疏/近似替代；
- Anti-Caching 式 pre-pass admission，把 cold miss 变成 delay/requeue，而不是执行期 stall；
- Prefill reuse ready + Decode execution ready 的双阶段 invariant；
- 3FS/shared cold tier 与跨 engine/session 语义；
- workload boundary 和 negative control 的生产准入策略。

## 5. 我们的方法什么时候可能赢

当前方法最可能有效的区域是：

1. **高复用历史上下文**：例如 1M 历史 + 1K-32K 新增 delta；历史部分可以复用，新 token 少。
2. **同一 session 多轮 append**：上轮 committed KV 可直接作为下一轮历史。
3. **共享长文档 / 长系统提示**：多个请求共用大 prefix，且 prefix 大到 recompute 明显更贵。
4. **idle session restore**：session 冷却后再次活跃，restore 能在 admission window 内完成。
5. **容量压力 / 重启 / 跨 engine**：vLLM APC hot-cache 不再有效时，外部 committed KV 有价值。
6. **SLA 队列保护**：即使不能立即服务，系统能 delay/requeue，而不是让冷层 I/O 混进短请求和 decode tail。

最可能无效或不占优的区域是：

1. **低局部性 random 1M**：没有可复用历史，应该 fallback/reject，不应承诺收益。
2. **小 prefix**：固定 connector/sidecar/restore 开销可能超过重算成本。
3. **vLLM APC hot-cache**：同进程热前缀缓存通常更快。
4. **没有真实高带宽 cold tier**：如果 3FS/SSD/H2D 带宽不足，restore 会吃掉 TTFT。
5. **每层小文件布局**：当前 layout 在大规模下会被 metadata/syscall/tail latency 放大。
6. **强 data plane baseline**：Tutti/CacheFlow/DualPath 类系统可能在 raw restore 与 overlap 上领先。

## 6. 下一阶段必须做的 baseline 矩阵

### 6.1 Baseline 定义

| 编号 | Baseline | 目的 |
|---|---|---|
| B0 | vLLM full prefill, APC off | 重算下界，回答“不复用有多慢”。 |
| B1 | vLLM APC hot-cache | 最强本地热前缀 baseline。 |
| B2 | vLLM APC after restart / capacity pressure | 验证 APC 失效区域。 |
| B3 | external DRAM-ready KV reuse | 测 offload-free reuse denominator。 |
| B4 | naive cold restore / synchronous offload | 证明 ready barrier 和 admission 的价值。 |
| B5 | current KV Anti-Caching: DELAY -> restore -> ADMIT | 当前方法主线。 |
| B6 | LMCache + vLLM | 强 external KV baseline。 |
| B7 | DualPath/Tutti/CacheFlow-style executor or simulation | 强 data-plane / scheduler baseline。 |
| B8 | KVDrive-style multi-tier orchestration | 强多级协同 baseline，检验我们的 Anti-Caching admission 语义是否有额外价值。 |

### 6.2 第一轮规模

用 Qwen2.5-14B-Instruct，在单卡 A6000 可承载范围内先做：

- prefix tokens：512、2K、8K、16K、32K；
- suffix tokens：128、512、2K；
- output tokens：1、16；
- workloads：`session_append`、`shared_prefix`、`low_locality`、`mixed_short_long`；
- repeats：至少 3，论文级至少 5-10；
- 并发：1、2、4；后续再扩。

### 6.3 必须报告的指标

核心指标：

- TTFT p50/p95/p99；
- ITL / TPOT p50/p95/p99；
- `TTFT_tiered / TTFT_full_prefill_no_reuse`；
- `TTFT_tiered / TTFT_dram_ready_reuse`；
- `prefill_tokens_saved`；
- `delta_prefill_tokens`；
- `historical_kv_hit_rate`；
- `historical_byte_hit_rate`；
- `admitted_sla_class_hit_rate`；
- `restore_bytes / useful_bytes`；
- `prefetch_deadline_miss_rate`；
- `sync_3fs_miss_rate`；
- storage / H2D / network utilization；
- mixed serving 下 short-request p99 TTFT。

正确性指标：

- correctness key mismatch 拒绝数；
- checksum failure 数；
- manifest coverage failure 数；
- wrong reuse / output anomaly smoke；
- cold-tier ready=false 被 connector load 的次数，必须为 0。

### 6.4 判定标准

第一轮不是立刻证明 1M，而是判断方法是否值得继续扩大：

- 对 `session_append` 和 `shared_prefix`，B5 至少应在 8K/16K/32K 上相对 B0 明显降低 TTFT 或 prefill compute。
- B5 相对 B3 的 TTFT 增幅应随着 prefix 变大收敛，目标是接近 `<=20%`，但第一轮可先看趋势。
- B5 相对 B4 应显著降低 tail latency 和 `sync cold miss` 风险，哪怕平均 TTFT 不一定更低。
- B5 不要求赢 B1 `APC-hot`；如果 B5 在 APC restart/capacity-pressure 下有效，就是合理价值区间。
- `low_locality` 不应被硬塞进 SLA 快路径；正确行为是 delay/fallback/reject。

## 7. 本项目当前的最诚实定位

如果现在写论文/汇报，应这样表述：

> 我们已经实现了一个 exact KV Anti-Caching 原型，它把冷 KV 从不可控的执行期 miss 转换成 admission 前可调度的 restore，并在 vLLM restart 小矩阵中验证了真实 cold object 迁移、checksum、ready barrier 和 external KV load。当前结果证明了系统 invariant 的可行性，但性能有效性仍需在 512-32K/真实 3FS/强 baseline 矩阵中验证。

不要这样表述：

> 我们已经证明了 1M tokens 下比现有方法更快。

这个结论目前没有证据支持。

## 8. 直接下一步

下一步不应该继续只扩 adapter feature，而应该先跑 **M3.11 Baseline Readiness Matrix**：

1. 写统一 benchmark runner，把 B0-B5 放进同一个 CSV schema。
2. 先覆盖 512/2K/8K/16K/32K，生成 Qwen2.5-14B 单卡对比报告。
3. 把当前 M3.10 cold-tier path 接入该 runner，输出 TTFT ratio、hit-rate、deadline miss、sync miss。
4. 如果 B5 在 8K-32K 高复用 workload 上仍不优于 B0 或 B4，就暂停 3FS 深挖，先优化 packed object、partial promote 和 restore overlap。
5. 如果 B5 相对 B0/B4 有明显收益，再接真实 3FS mount、LMCache、DualPath/Tutti-style baseline。

## 9. 参考基线资料

- vLLM Automatic Prefix Caching: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- LMCache documentation: https://docs.lmcache.ai/
- Mooncake repository: https://github.com/kvcache-ai/Mooncake
- Mooncake paper: https://arxiv.org/abs/2407.00079
- Tutti paper: https://arxiv.org/abs/2605.03375
- DualPath paper: https://arxiv.org/abs/2602.21548
- CacheFlow paper: https://arxiv.org/abs/2604.25080
- KVDrive paper: https://arxiv.org/abs/2605.18071
