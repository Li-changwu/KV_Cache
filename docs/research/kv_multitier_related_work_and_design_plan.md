# KV 多级管理系统相关工作与设计规划

日期：2026-05-25

## 0. 执行摘要

当前项目已经完成到 M3.9：有 vLLM sidecar / connector 原型、短前缀真实 KV tensor 保存与加载、元数据级 DRAM/NVMe 分层、deadline-aware 预取队列和在线小矩阵指标。下一步若要满足目标指标，系统设计必须从“可证明路径正确”推进到“真实 1M tokens 服务闭环”：真实 cold-tier I/O、后台迁移执行器、SLA admission、命中率控制、带宽保护和两类方案对比。

本文的核心建议是将系统定位为 **SLA-governed exact long-context KV service**，而不是单纯的 KV offload。论文叙事应围绕四个硬贡献：

1. 精确语义：严格 correctness key、版本化 manifest、ready barrier，避免错误复用。
2. SLA 准入：所有 cold-tier 读取必须在 decode 前完成；不能把 SSD/3FS miss 放进 decode critical path。
3. 多级执行层：HBM/DRAM 是 execution-reachable，3FS/SSD 是 capacity tier；通过 background restore 和 admission window 把 cold miss 转换成 warm hit。
4. 面向 agent 长上下文的 workload 边界：高复用、多轮 append、共享长文档、idle session restore 是目标场景；low-locality 1M random 是负对照，应 delay / fallback / reject。

如果能在真实 3FS/NVMe 环境下稳定支持 1M tokens，证明 `TTFT_tiered / TTFT_baseline <= 1.2`，适用场景命中率 >=90%，并与 LMCache / vLLM APC / disaggregated prefill / DualPath-style 策略做强对比，则具备冲击 CCF-A 系统会议的潜力。若只停在当前 M3.9 元数据迁移和小前缀 smoke，论文强度不足。

## 1. 研究问题

### 主研究问题

在保持精确 attention 语义的前提下，能否通过 SLA 感知的多级 KV Cache 控制面，使长上下文 LLM 推理稳定承载 1M tokens 请求，并将 1M 极限长度下的 TTFT 增幅控制在业务 SLA 阈值内？

### 子问题

1. 对 agent 多轮与科学发现 workload，哪些 KV ranges 可安全复用，如何定义 correctness key 与 committed ranges？
2. 在 HBM/DRAM/3FS 三层之间，如何做 admission、placement、prefetch、eviction，使 decode path 同步 SSD/3FS miss 恒为 0？
3. 如何把 1M tokens 的目标拆成可评测指标：TTFT ratio、effective hit rate、bandwidth utilization、deadline miss、tail latency、短请求干扰？
4. 相比 vLLM APC、SGLang RadixAttention、LMCache、Mooncake、Tutti、DualPath、CacheFlow 等系统，本文贡献到底在哪里？

### FINER 评估

| 维度 | 评分 | 说明 |
|---|---:|---|
| Feasible | 4 | 已有 vLLM connector 原型和本机/仿真数据；仍需真实 cold-tier 与 1M 集群环境。 |
| Interesting | 5 | agent 长上下文、多轮推理和科学发现工作流正在把 KV 容量变成服务瓶颈。 |
| Novel | 3-4 | KV offload / reuse 竞争非常强；必须突出 SLA 准入、3FS-aware、精确 1M、negative-control 边界。 |
| Ethical | 5 | 系统性能研究，风险主要是错误复用导致数据泄漏或错误输出，可通过 correctness key 和隔离控制。 |
| Relevant | 5 | 直接服务长程记忆、复杂决策和生产推理成本。 |

## 2. 相关工作地图

### 2.1 KV 内存虚拟化与 prefix reuse

- **vLLM / PagedAttention** 是基本盘。PagedAttention 把 KV cache 管理做成类似虚拟内存 paging，减少碎片和重复复制，并支持跨请求 KV 共享；论文报告吞吐相对 FasterTransformer / Orca 提升 2-4x。局限是它主要解决 GPU 内 KV 管理和 serving throughput，不直接解决 1M 级 cold-tier restore 与 SLA admission。来源：arXiv 2309.06180。
- **vLLM Automatic Prefix Caching** 使用 block hash 做前缀缓存，适合作为 baseline。它是精确 prefix reuse 的强基线，但缓存容量仍主要受 HBM/host 内存约束，且没有把 3FS/NVMe 冷层作为 SLA 受控资源。
- **SGLang / RadixAttention** 针对多调用、结构化程序、多轮对话做 KV reuse，前缀树式复用很适合 agent workload。它强调 runtime reuse，但不是以 1M exact context + 3FS capacity tier 为中心。

### 2.2 Disaggregated prefill / decode 与跨实例 KV transfer

- **DistServe** 将 prefill 与 decode 分离，核心洞察是二者混部会造成干扰和资源耦合。它为我们提供了系统层拆分 prefill restore、decode execution 的理由。
- **vLLM disaggregated prefill / NIXL connector** 已经提供工程接入点。当前项目选择 sidecar + KV connector 是对的，因为 vLLM 官方路径也把 KV transfer 和 connector 作为 disaggregated serving 的扩展面。
- **Mooncake** 是 KVCache-centric disaggregated architecture：分离 prefill / decode cluster，利用 CPU、DRAM、SSD 做 disaggregated KV cache，并用 scheduler 满足 SLO。它是非常强的系统对标。我们的差异必须更明确：精确 1M SLA、3FS-aware cold tier、decode-path no-sync-miss 约束、开源 vLLM connector 原型与负对照 admission。

### 2.3 KV cache layer / persistent KV / non-prefix reuse

- **LMCache** 把 KV cache 做成 engine 外的可复用 cache layer，支持跨 query / engine reuse、offloading 和 prefill-decode disaggregation。它是必须 benchmark 的强 baseline。
- **CacheBlend** 处理 RAG 中 reused chunks 不总是 prefix 的问题，通过 cached knowledge fusion 和选择性 recompute 来降低 TTFT。它启发我们考虑 chunk-level reuse，但主线若要求精确 attention，第一版应只做严格 prefix / committed range，非 prefix blend 可作为 optional path 或 ablation。
- **CacheGen** 压缩/流式传输 KV cache，以减少网络传输延迟。它适合作为 bandwidth-reduction 相关工作，但如果项目第一版坚持无损精确语义，压缩应放在后续可插拔扩展。

### 2.4 多级存储、SSD-backed KV 与 restoration

- **FlexGen** 证明 GPU/CPU/disk tensor offload 可以让有限 GPU 运行大模型，但目标偏 throughput 和资源受限 batch，不是在线 1M SLA。
- **Tutti** 直接把 SSD-backed KV cache 做成 long-context serving 的实用系统，重点包括 GPU-centric KV object store、I/O pipeline 和 slack-aware scheduling。这是近期最接近的竞争方向之一。
- **CacheFlow** 把 KV restoration 看成 tokens/layers/GPUs 三维并行问题，用 batch-aware scheduler 降低 TTFT。它提示我们不能只做单流 NVMe->DRAM 估算，必须建模 layer parallel、token chunk parallel、batch contention。
- **DualPath** 针对 agentic LLM inference 的 storage bandwidth bottleneck，提出 storage-to-prefill 与 storage-to-decode 双路径。它和本文目标高度接近，至少要作为第二方案实现或仿真对比。
- **Infinite-LLM / DistKV-LLM** 使用 distributed KVCache 支持到百万级上下文，是“分布式 GPU/CPU 内存池”路线。我们的 3FS/SSD capacity tier 应和它区分：成本结构、共享存储、cold restore 与 SLA admission。

### 2.5 近似 / 压缩 / 驱逐类工作

H2O、Scissorhands、CacheGen、TTKV 等都说明 KV 体积可以通过 heavy-hitter、时序重要性、压缩、精度分层减少。但本项目第一版目标是 exact attention，不应把这些当主线。它们适合作为“未来可插拔 compression tier”或 ablation，而不是指标达成的基础。

## 3. 竞品与差异化定位

| 系统/方向 | 强项 | 对我们构成的压力 | 我们必须强调的差异 |
|---|---|---|---|
| vLLM APC / PagedAttention | 成熟、强基线、精确 prefix reuse | 任何小规模 prefix reuse 都会被认为已解决 | 1M、cold tier、SLA admission、ready barrier、3FS 带宽控制 |
| SGLang RadixAttention | agent / structured workload 复用强 | 多轮场景叙事相近 | 与 3FS/NVMe 多级迁移和 production SLA 结合 |
| LMCache | engine 外 KV cache layer、跨 engine reuse | 很接近 persistent KV cache 方向 | 1M exact target、decode no-sync-miss、3FS-aware admission、强负对照 |
| Mooncake | 工业级 KV-centric disaggregation | SLO + KV scheduler 叙事非常强 | 开源 vLLM connector、3FS 冷层、1M exact SLA、细粒度 correctness |
| Tutti | SSD-backed KV cache practical | 直接覆盖 SSD-backed 长上下文 | 3FS/多级共享存储、agent workload、DualPath/offload 对比、精确服务边界 |
| CacheFlow | restoration 并行调度 | 对单流预取模型形成挑战 | 引入 layer/token/device 并行恢复，并用 admission 控制何时可 decode |
| DualPath | agentic KV storage 双路径 | 方案名已经非常接近 | 必须把 DualPath 作为比较方案，或吸收为系统中的策略族 |

## 4. 推荐系统设计：SLA-KV

### 4.1 总体架构

```
Gateway
  -> Workload Classifier / SLA Classifier
  -> CorrectnessKey + PrefixIndex + ManifestIndex
  -> Reuse Planner
  -> Residency Planner
  -> Admission Guard
  -> Prefetch / Restore Scheduler
  -> vLLM KV Connector
  -> Decode Ready Barrier
  -> Eviction / Writeback / Metrics
```

其中 Gateway / sidecar 不直接做 attention kernel 变更；vLLM connector 只执行保存、加载、pin、release。这样保持低侵入，并和当前 M3.5-M3.9 原型兼容。

### 4.2 数据模型

**KVCorrectnessKey** 必须覆盖：

- model id、revision、weight hash、tokenizer hash；
- RoPE / YaRN / position scaling 参数；
- dtype、KV layout、attention backend、block size；
- LoRA / adapter / prompt adapter；
- tenant / salt；
- prompt token ids hash、position range、committed range version；
- sampling 不应影响 KV，但要记录 request class 以便审计。

**KVManifest** 应从当前 `prefix_id, token_start, token_end, layers, tier, ready` 扩展到：

- `object_group_id`：避免每层小文件过多；
- `tier_state`: HBM | DRAM | LOCAL_NVME | 3FS | FETCHING | CORRUPT；
- `ready_epoch` / `lease_expire_ts`；
- `checksum` / `size_bytes` / `layout_version`；
- `source_path` / `cold_path` / `restore_path`；
- `access_stats`: last_access, reuse_count, prefetch_success, miss_reason；
- `policy_tags`: pinned, protected, evictable, speculative。

### 4.3 多级语义

- **HBM**：vLLM PagedAttention 执行层；最小粒度 block。
- **DRAM**：执行可达 staging tier；decode 前必须能从 DRAM/HBM 完成必要加载。
- **LOCAL_NVME**：单机 cold/warm tier，可用于 M3.10 真实设备级迁移。
- **3FS**：共享 cold capacity tier，跨节点 / session restore；不允许 decode path 同步读。

关键不变量：

1. `ready=false` 的 KV 不能被 connector load。
2. `tier in {LOCAL_NVME, 3FS}` 的 KV 即使文件存在，也不能直接进入 decode。
3. Admission 只能在 required ranges 都已经 ready，或其预取能在 admission window 内完成时返回 `ADMIT`。
4. 若预测赶不上 deadline，只能 `DELAY`、`FULL_PREFILL_FALLBACK`、`RELAXED_QUEUE` 或 `REJECT`。

### 4.4 Reuse Planner

当前 M2 命中率约 65%-69%，低于目标 90%。这里首先要澄清指标定义：

- `effective_hit_rate_all_tokens = reused_tokens / total_prompt_tokens`，它会被新增 delta 拉低；
- `historical_kv_hit_rate = reused_historical_tokens / required_historical_tokens`，更适合衡量多轮 session restore；
- `byte_hit_rate = bytes_loaded_from_reusable_KV / bytes_required_for_historical_KV`，更适合带宽和 cache 设计；
- `admitted_hit_rate = hit_rate over admitted SLA class only`，应排除 low-locality negative-control。

要达成 >=90%，设计上需要：

1. 工作负载定义为高局部性：例如 1M 历史上下文 + 1K-32K 新增 delta，而不是每轮追加 256K。
2. committed range 采用分段提交：system prompt、长文档、tool trace、历史 turns 分别 manifest 化。
3. PrefixIndex 支持 session lineage：`session_id -> committed ranges -> prefix DAG`。
4. Admission 对低命中率请求直接退出 SLA 快路径，避免拉低整体指标。

### 4.5 Admission Guard

Admission 估计：

```
TTFT_est =
  max(restore_critical_path_ms, h2d_stage_ms, queue_wait_ms)
  + delta_prefill_compute_ms
  + connector_overhead_ms
  + decode_start_overhead_ms
```

准入条件：

```
hit_rate_est >= target_hit_rate
TTFT_est <= SLA_ms
TTFT_est / baseline_ttft <= 1.2
sync_3fs_miss_est == 0
storage_util_pred <= 0.7
h2d_util_pred <= 0.7
short_request_p99_guard_ok
```

如果不满足，返回：

- `DELAY_PREFETCH`: 有可复用 KV，但赶不上当前 deadline；
- `FULL_PREFILL_FALLBACK`: 低局部性或 correctness mismatch；
- `RELAXED_QUEUE`: 用户允许宽松 SLA；
- `REJECT`: 系统保护或租户配额。

### 4.6 Restore / Prefetch Scheduler

当前 M3.9 的 `PrefetchQueue` 是单资源估算。下一版必须变成多资源约束调度：

- storage read: 3FS / NVMe bandwidth and IOPS；
- network/RDMA: 3FS 到 compute 节点；
- CPU/DRAM copy；
- H2D copy；
- GPU compute overlap；
- per-layer / per-token chunk dependency。

推荐策略：

1. 先用 EDF + slack-aware SRPT 作为 deterministic baseline。
2. 加 primal-dual budget controller 保证 storage/H2D/network utilization 不超过 70%。
3. 用 contextual bandit 学习不同 workload class 的 prefetch lead、tier placement、chunk size。
4. 对 1M restore 引入 CacheFlow 式三维并行：token chunk x layer group x device。

### 4.7 DualPath 作为策略族

为了满足“两种方案 benchmark”，建议实现两个策略：

**方案 A：Admission-window Offloading**

- 3FS/SSD -> DRAM -> HBM；
- 当前请求不等待 cold miss；
- 适合 idle session restore、提前预取、重复文档。

**方案 B：DualPath Restore**

- storage-to-prefill：prefill worker 提前拉取历史 KV，并与 delta prefill overlap；
- storage-to-decode：decode worker 只接收已 ready 的 critical ranges，缺失则 block admission；
- 全局 scheduler 根据 bandwidth skew 和 queue slack 选择路径。

两方案共用 correctness key、manifest、metrics 和 ready barrier，差异仅在 restore path 和 scheduler。

## 5. Level 2 优化闭环设计

### Inner loop

当前控制面可以抽象为：

```
observe request + cache state
  -> propose reuse / placement / prefetch plan
  -> evaluate TTFT, hit rate, bandwidth, deadline risk
  -> admit / delay / fallback
  -> observe actual latency and update policy
```

### 当前瓶颈

- M2 sweep 是离线网格搜索，不能在线自适应。
- M3.9 只有确定性队列，未学习 workload class 和真实设备 tail latency。
- 当前 hit rate 未达 90%，需要准入策略和 workload classifier 联动，而不是让所有请求进同一指标池。

### 候选机制

| 机制 | 来源 | 作用 | 复杂度 | 风险 |
|---|---|---|---|---|
| Primal-dual budget controller | 在线优化 | 保证 3FS/H2D/network utilization <=70% | 中 | 参数不稳会过度 reject |
| Contextual bandit | 在线学习 | 学习 prefetch lead、tier placement、chunk size | 中 | 早期探索可能伤 SLA |
| EDF + SRPT + slack scheduler | 实时调度 | 控制 deadline miss 与 tail latency | 低 | 可能偏向短 restore，饿死长请求 |
| MAP-Elites / DOE | 离线探索 | 系统性搜索策略族，避免只调少数参数 | 中 | 离线收益不一定在线迁移 |
| MPC | 控制理论 | 多步预测 queue / bandwidth / hit rate | 高 | 模型误差大时复杂度不值 |

### 推荐机制

先采用 **Budgeted Slack Scheduler**：

- admission 层用 primal-dual 维护 storage/H2D/network 三个预算价格；
- restore queue 用 EDF 排 deadline，同 deadline 下用 highest marginal saved-prefill-per-byte；
- policy 层用 contextual bandit 在离散动作中选择 `prefetch_lead_ms, chunk_size, target_tier, path=A/B`；
- 所有学习策略只影响 `DELAY` 和 future prefetch，不允许绕过 ready barrier。

接口草案：

```python
class BudgetedSlackScheduler:
    def propose(request, cache_state, budgets) -> RestorePlan: ...
    def evaluate(plan, calibration, queue_state) -> AdmissionEstimate: ...
    def decide(estimate, sla) -> AdmissionDecision: ...
    def observe(actual_metrics) -> None: ...
```

暂不建议直接生成代码；应先在 M2 simulator 中实现这套策略并与现有 fixed policy 对比。

## 6. 实验计划

### 阶段 E1：真实 cold-tier M3.10

- 把当前 `NVME` manifest 状态替换为真实目录/设备级迁移。
- safetensors 对象从 DRAM-store 目录搬到 cold-store 目录。
- 后台 executor 执行 async copy / mmap / direct I/O / io_uring 可选。
- 记录真实 bytes、latency、p50/p95/p99、failure、checksum。
- 保持单测中确定性 fake executor，在线 smoke 使用真实 executor。

成功标准：

- 未 ready 时 connector load 必须失败。
- executor 完成后当前或后续请求能 ADMIT。
- `sync_3fs_miss_total == 0`。

### 阶段 E2：32K-128K 在线矩阵

模型：

- 默认 Qwen2.5-14B，原生 32K；
- 128K 作为 YaRN/RoPE extension run，单独标记；
- Qwen3-32B 只做压力上界。

矩阵：

- prefix: 512, 1K, 2K, 4K, 8K, 16K, 32K；
- suffix: 128, 512, 2K；
- paths: baseline full prefill, vLLM APC, external DRAM hit, cold-tier restore, DualPath；
- load: single request, batch mixed short/long, multi-session restore。

指标：

- TTFT p50/p90/p99；
- `TTFT_tiered / TTFT_baseline`；
- external_load_observed；
- hit rate 四种定义；
- storage/H2D utilization；
- deadline miss；
- short request P99。

### 阶段 E3：1M 仿真 + 实机闭环

1M tokens 对 Qwen2.5-14B KV 约 183GiB，Qwen3-32B 约 244GiB。单卡 A6000 无法直接承载，需要 DRAM/3FS capacity tier。

实机路线：

- 单机：DRAM 作为 1M capacity，NVMe/3FS 做 cold restore，验证控制语义和 TTFT 下界。
- 多机：3FS + RDMA / high bandwidth network，验证共享 cold tier 和 admission。
- 真实 1M online 如果没有足够模型上下文支持，可用 YaRN/支持 1M 的模型做专门 run；否则用 “1M KV object restore + decode-ready validation + shorter decode model run” 分离验证，但论文主结论必须如实标注。

### 阶段 E4：论文级对比

强 baseline：

- vLLM native APC；
- SGLang/RadixAttention 思路的 prefix tree；
- LMCache；
- Mooncake-style disaggregated KV cache，如无法复现则用公开配置和自实现近似；
- Tutti/SSD-backed KV，如代码可用则直接对比，否则做 design-level baseline；
- CacheFlow/DualPath 策略仿真或实现。

Ablation：

- no admission；
- no ready barrier；
- no budget controller；
- no correctness key strict check；
- no 3D restore parallelism；
- no workload classifier；
- DRAM-only vs NVMe vs 3FS。

## 7. 论文可发表性判断

### 可以冲 CCF-A 的条件

需要满足以下至少 5 项：

1. 真实系统而非纯仿真：vLLM connector + 真实 cold-tier I/O + 真实 3FS/NVMe。
2. 明确 1M exact context 支持：不靠近似压缩或语义检索替代。
3. 强 baseline：至少 vLLM APC、LMCache、disaggregated prefill、DualPath/offload 对比。
4. 清晰 novel mechanism：SLA admission + ready barrier + multi-resource restore scheduler + correctness key。
5. 生产或真实 trace：agent 多轮、工具调用、科学发现、idle restore 或共享长文档 workload。
6. 端到端指标：TTFT ratio、P99、hit rate、bandwidth、short-job interference。
7. 开源或可复现实验：脚本、配置、failure rows、artifact schema。

### 论文标题方向

- `SLA-KV: Admission-Controlled Multi-Tier KV Cache for Exact Million-Token LLM Serving`
- `No Cold Miss on the Decode Path: SLA-Governed KV Restoration for Long-Context LLM Inference`
- `Making Million-Token KV Cache Serving Predictable with Multi-Tier Storage and Ready Barriers`

### 目标会议

如果系统实做足够强，偏 systems / storage / serving，可考虑 OSDI、SOSP、USENIX ATC、ASPLOS 等系统方向会议；若重点落在网络/RDMA/3FS traffic scheduling，可考虑 SIGCOMM/NSDI 方向。具体 CCF 分类需以单位当年 CCF 推荐目录为准。

### 最大审稿风险

1. “这不就是 LMCache / Mooncake / Tutti 吗？”
   必须用 1M exact SLA、3FS-aware admission、negative control、ready barrier 不变量区分。

2. “只在小模型/小上下文上验证，不能支持论文主张。”
   必须有 1M 或至少 128K->1M 的 scaling experiment。

3. “命中率 90% 是 workload 选择出来的。”
   需要公开 workload 分类规则，并报告 low-locality 负对照。

4. “TTFT baseline 不公平。”
   需要定义 DRAM baseline、full prefill baseline、native APC baseline，并报告不可承载时的 capacity failure。

5. “系统实现太浅。”
   M3.9 元数据迁移不够，必须进入真实 I/O、后台 executor、故障恢复。

## 8. 下一步工程路线

### 未来 1-2 周

1. 完成 M3.10：真实 cold-tier 目录/设备迁移和后台 executor。
2. 在 M2 simulator 实现 Budgeted Slack Scheduler。
3. 在线小矩阵扩展到 512-32K prefix。
4. 建立固定报告脚本：自动输出 TTFT ratio、hit rate、bandwidth、deadline miss。

### 未来 3-5 周

1. 接入 3FS 或至少模拟 3FS mount 的共享路径。
2. 加入 DualPath 策略族。
3. 跑 Qwen2.5 32K/128K、Qwen3 压力上界。
4. 引入真实 agent trace 或合成但可解释的 session append workload。

### 未来 6-10 周

1. 1M end-to-end 或分解式实验证明。
2. 强 baseline 对比。
3. 完成论文骨架：问题、设计、不变量、调度、实现、评测、局限。
4. 做 artifact cleanup：所有当前 untracked 文件应进入干净分支、CI 和 reproducibility package。

## 9. 本轮资料来源

- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
- vLLM Automatic Prefix Caching: https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM Disaggregated Prefill: https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM NIXL connector: https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
- SGLang / RadixAttention: https://arxiv.org/abs/2312.07104
- DistServe: https://arxiv.org/abs/2401.09670
- Infinite-LLM / DistKV-LLM: https://arxiv.org/abs/2401.02669
- Mooncake: https://arxiv.org/abs/2407.00079
- LMCache: https://arxiv.org/abs/2510.09665
- LMCache GitHub: https://github.com/LMCache/LMCache
- CacheGen: https://arxiv.org/abs/2310.07240
- CacheBlend: https://arxiv.org/abs/2405.16444
- FlexGen: https://arxiv.org/abs/2303.06865
- H2O: https://arxiv.org/abs/2306.14048
- Scissorhands: https://arxiv.org/abs/2305.17118
- Tutti: https://arxiv.org/abs/2605.03375
- DualPath: https://arxiv.org/abs/2602.21548
- CacheFlow: https://arxiv.org/abs/2604.25080
- DeepSeek 3FS: https://github.com/deepseek-ai/3FS
