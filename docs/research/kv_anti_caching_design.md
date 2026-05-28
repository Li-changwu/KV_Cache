# KV Anti-Caching 设计方案

日期：2026-05-25

## 0. 结论

KV Anti-Caching 方向具备继续推进的价值。它不是把 KV Cache 简单放到 SSD/3FS 上，而是把长上下文推理中的 KV 管理重新定义为一个 memory-primary 的反向缓存系统：

- HBM/DRAM 是执行主层，只有这里的 KV 才能进入 decode ready set。
- 3FS/SSD 是 cold anti-cache capacity tier，只保存可恢复、可验证、可批量调度的冷 KV。
- 请求进入 vLLM 前必须完成 KV pre-pass，枚举本轮需要的历史 KV ranges。
- 若 required KV 仍在 cold tier，本轮请求不允许同步读 3FS/SSD，只能 delay、requeue、fallback 或 reject。
- restore 完成后只 partial promote 本轮真正需要的 KV ranges，避免把整块冷 KV 全部重新变热。

可行性评分：4/5。当前 M3.9 已经有 sidecar、manifest、tier state、ready barrier、deadline-aware prefetch queue 和在线小矩阵指标，足够作为起点。主要缺口是 M3.10 的真实 cold-tier I/O、冷块对象布局、后台执行器和 1M scaling。

创新性评分：4/5。单独的 offload、prefix caching、SSD-backed KV 已经很拥挤；但“Anti-Caching semantics for exact long-context KV serving”仍有清晰空间。关键是把 DeBrabant 等人的 Anti-Caching 思想迁移成 KV 专用的 pre-pass admission、cold-block construction、partial promotion 和 SLA-governed non-blocking restore，而不是只做一个更快的存储插件。

论文潜力判断：有条件可冲 CCF-A 系统会议。前提是实现真实 3FS/NVMe cold tier、证明 exact 1M、给出强 baseline，并把 low-locality negative control 如实报告。

## 1. 当前基础与缺口

### 1.1 已有基础

当前项目已完成到 M3.9，具备以下基础：

- `benchmarks/m3/control_plane.py`：已有 `ADMIT / DELAY / REJECT / FULL_PREFILL_FALLBACK` 决策、correctness key、manifest lookup 和 ready barrier。
- `benchmarks/m3/tensor_store.py`：已有 `KVBlockManifest`，包含 `tier` 和 `ready`；`TierNotReady` 已阻止未预取 NVMe KV 被 connector 同步加载。
- `benchmarks/m3/prefetch_queue.py`：已有 deterministic deadline-aware prefetch queue 和 async queue 原型。
- `docs/specs/m3_connector_sidecar_minimal_interface.md`：已经明确 decode critical path 不允许同步读 SSD/3FS。
- M2 simulator 和 sweep 已证明：高复用 session/shared-prefix workload 有进入 SLA 的可能；low-locality 1M workload 应该作为负对照被 delay/fallback/reject。

这些已经覆盖了 KV Anti-Caching 的控制面雏形，但还没有形成完整的 anti-cache 系统语义。

### 1.2 主要缺口

当前缺口集中在五个方面：

1. **没有真实 cold-tier 数据路径**：M3.9 仍是元数据级 `DRAM -> NVME -> DRAM`，没有真实目录/设备搬运、checksum、tail latency、3FS queue depth。
2. **没有冷块组织策略**：现有 manifest 以 prefix/layer 文件为主，还没有把多个细粒度 KV ranges 组织成 3FS/NVMe 友好的 cold object。
3. **没有 partial promotion**：当前 `prefetch_to_dram()` 会把整个 prefix 标记为 ready，不能表达只恢复某些 token range、layer group 或 block group。
4. **调度器仍是单资源估算**：`PrefetchQueue` 只按 storage GB/s 估算，尚未同时建模 3FS/NVMe、RDMA/network、DRAM、H2D、GPU slack 和短请求保护。
5. **缺少在线策略闭环**：没有把实际 TTFT、deadline miss、hit rate、queue tail、bandwidth utilization 回写到 admission/prefetch/eviction policy。

## 2. 与 Tutti 的关系

Tutti 的贡献在于 SSD-backed KV 的 fast data path：GPU-centric object store、GPU io_uring、SGL、slack-aware I/O scheduling。它证明了本地 NVMe 到 HBM 的路径如果绕开 CPU critical path，可以显著降低 GPU stall。

KV Anti-Caching 与 Tutti 的差异应明确写成：

| 维度 | Tutti | KV Anti-Caching |
|---|---|---|
| 核心问题 | 本地 SSD KV restore/store 怎么更快 | 哪些 KV 应该冷却、何时 restore、何时准入 |
| 层级视角 | HBM-SSD GPU-centric fast path | HBM/DRAM 执行主层 + 3FS/SSD anti-cache |
| 请求处理 | 加速已决定要加载的 KV | pre-pass 枚举 required KV，未 ready 则 delay/requeue |
| 调度目标 | 减少 GPU stall，利用 slack | TTFT SLA、hit rate、no sync cold miss、带宽预算、短请求隔离 |
| 跨节点 | 当前 remote path 仍有 CPU/RDMA 开销 | 3FS/shared cold tier 是核心目标之一 |
| 论文差异 | GPU storage stack co-design | KV anti-cache policy and admission semantics |

因此，Tutti 可以作为可选 data plane 或强 baseline；我们的创新必须在 control plane 和 policy semantics 上成立。

## 3. Anti-Caching 到 KV 的迁移

DeBrabant 等人的 Anti-Caching 对 KV 设计的关键启发如下：

| 数据库 Anti-Caching | KV Anti-Caching 对应设计 |
|---|---|
| Main memory is primary storage | HBM/DRAM 是执行主层，3FS/SSD 只是 cold capacity tier |
| Cold tuples are evicted to disk | 冷 KV ranges 被 demote 到 cold objects |
| Tuple-level eviction, block-level disk write | 细粒度选择 KV range，粗粒度打包成 3FS/NVMe 友好对象 |
| Evicted Table | `KVResidencyTable`：range -> tier/object/offset/ready/version |
| Block Table | `ColdBlockTable`：cold object -> layout/checksum/valid map |
| LRU Chain / aLRU sampling | `SampledHotnessIndex`：低开销采样更新热度 |
| Pre-pass execution | `KVPrePass`：admission 前枚举 required historical KV |
| Abort and requeue | `DELAY_RESTORE` / `REQUEUE_AFTER_READY` |
| Tuple merge | `PartialPromote`：只恢复本轮需要的 KV range/layer group |
| Lazy block compaction | cold object valid bitmap + 后台 compaction |
| Snapshot / command log | manifest commit log + cold object version/checksum |

这个迁移的本质是：不要把 cold miss 留给执行时发现，而是在 admission 前把未来会访问的 KV 集合静态化、批量化、可调度化。

## 4. 系统目标与非目标

### 4.1 目标

1. 在 exact attention 语义下支持最长 1M tokens 上下文。
2. SLA 队列中 `sync_3fs_miss_total == 0`。
3. 对适用 workload，admitted historical KV hit rate 或 byte hit rate >= 90%。
4. 1M 极限长度下 `TTFT_tiered / TTFT_baseline <= 1.2`，baseline 需明确区分 DRAM-ready、full-prefill、native APC。
5. 3FS/NVMe、network/RDMA、H2D sustained utilization 低于警戒线，例如 70%。
6. 低局部性 1M 请求不污染指标池，必须进入 relaxed queue、full-prefill fallback、delay 或 reject。

### 4.2 非目标

第一版不做以下事情：

- 不引入有损 KV 压缩、稀疏注意力、语义检索替代。
- 不修改 attention kernel。
- 不承诺随机 low-locality 1M 请求的低 TTFT。
- 不把 3FS/SSD 当透明 HBM 扩展。
- 不允许 connector 在 decode critical path 上同步从 3FS/SSD 读取。

## 5. 总体架构

```
Gateway / Sidecar
  -> Workload Classifier
  -> CorrectnessKey Builder
  -> Prefix / Session Lineage Index
  -> KVPrePass
  -> KV Anti-Cache Planner
      -> Reuse Plan
      -> Cold Restore Plan
      -> Partial Promote Plan
      -> Eviction / Demotion Plan
  -> Budgeted Anti-Cache Scheduler
  -> Admission Guard
  -> Background Restore Executor
  -> vLLM KV Connector
  -> Decode Ready Barrier
  -> Metrics / Feedback Controller
```

其中 sidecar/control plane 做策略，vLLM connector 做受控的数据搬运和 ready 检查。这个边界继承 M3 的低侵入路线。

## 6. 核心数据结构

### 6.1 KVCorrectnessKey

必须覆盖：

- model id、revision、weight hash；
- tokenizer hash；
- RoPE / YaRN / position scaling；
- dtype、KV layout、attention backend、block size；
- LoRA / adapter / prompt adapter；
- tenant id 或 salt；
- prompt token ids hash；
- committed range version；
- security domain。

只有 correctness key 完全相同的 KV 才能复用。

### 6.2 KVRange

逻辑访问单位：

```text
KVRange = {
  tenant_id,
  session_id,
  prefix_id,
  token_start,
  token_end,
  layer_start,
  layer_end,
  block_ids,
  correctness_key,
  committed_epoch
}
```

第一版可以先把 layer range 设为 all layers，后续再做 layer group 并行 restore。

### 6.3 KVResidencyTable

内存常驻索引：

```text
KVResidencyEntry = {
  range_id,
  tier: HBM | DRAM | LOCAL_NVME | 3FS | FETCHING | STAGED | CORRUPT,
  ready: bool,
  cold_object_id,
  object_offset,
  byte_length,
  valid: bool,
  refcount,
  pin_until,
  last_access_ts,
  sampled_access_count,
  reuse_score,
  checksum,
  version
}
```

它相当于 Anti-Caching 的 Evicted Table，但扩展为多级 residency 和 correctness-aware manifest。

### 6.4 ColdBlockTable

3FS/NVMe 上的对象表：

```text
ColdObject = {
  cold_object_id,
  storage_uri,
  object_size_bytes,
  layout_version,
  correctness_key_hash,
  ranges: [range_id],
  offset_table,
  valid_bitmap,
  compression: none,
  checksum,
  created_epoch,
  sealed: bool
}
```

第一版对象大小建议采用 32MB 到 128MB 区间作为 sweep 参数。它比小文件更适合 3FS/NVMe 带宽，又不会大到 partial promote 完全失效。

### 6.5 SampledHotnessIndex

热度更新不能发生在每次 KV 访问上。建议使用采样和分段热度：

```text
reuse_score =
  w1 * sampled_recent_hits
  + w2 * predicted_session_reuse_prob
  + w3 * saved_prefill_ms
  - w4 * restore_bytes
  - w5 * age
```

其中 `sampled_recent_hits` 只对部分请求更新，类似 Anti-Caching 的 aLRU。这样可以避免控制面热度维护本身吃掉收益。

## 7. 请求生命周期

### 7.1 Commit

请求成功执行后，控制面把可复用 KV 写成 committed ranges：

1. connector 保存 layer KV 到 DRAM object 或本地 tensor store；
2. control plane 写入 `KVResidencyTable`；
3. correctness key、token ids、layout、checksum 完整记录；
4. 后台 demotion 策略决定是否把较冷 ranges 打包到 cold objects。

### 7.2 Demotion

当 HBM/DRAM 水位超过阈值：

1. 在 evictable ranges 中按最低 reuse score 选候选；
2. 将细粒度 ranges 组合成 cold object；
3. 顺序写入 LOCAL_NVME 或 3FS；
4. fsync/checksum 成功后把 residency 标为 cold tier、ready=false；
5. 释放 DRAM/HBM 副本，保留内存索引。

注意：第一版应避免 duplicate source of truth。一个 committed range 的执行副本可以在 HBM/DRAM，容量副本可以在 cold tier，但 ready 状态必须明确，不能让 connector 随机选择 cold path。

### 7.3 KVPrePass

每个请求进入 vLLM 前：

1. tokenize 并计算 correctness key；
2. 查 prefix/session lineage；
3. 枚举本轮 required historical KV ranges；
4. 查询 `KVResidencyTable`；
5. 生成三类集合：
   - `ready_set`: HBM/DRAM 且 ready；
   - `cold_set`: LOCAL_NVME/3FS 或 ready=false；
   - `missing_set`: 无 manifest 或 correctness mismatch。

若 `missing_set` 非空，不能进入 exact reuse 快路径。若 `cold_set` 非空，交给 scheduler 判断能否在 admission window 内恢复。

### 7.4 Restore

restore 不直接服务当前 decode，而是生成后台任务：

```text
RestoreTask = {
  request_id,
  range_ids,
  source_tier,
  target_tier: DRAM,
  bytes,
  saved_prefill_ms,
  deadline_ms,
  priority,
  dependency: cold_object_id + offsets,
  path: local_nvme | 3fs | dualpath_prefill | dualpath_decode
}
```

任务完成后，range 状态变为 `DRAM/ready=true` 或 `STAGED/ready=true`，当前或后续请求才能重新 admission。

### 7.5 Partial Promote

如果一个 cold object 内只有部分 ranges 被访问，只 promote 对应 ranges：

- 更新被访问 ranges 为 DRAM ready；
- cold object 内原位置通过 valid bitmap 保持可读或标为 stale；
- 后台 compaction 合并空洞；
- 未访问 ranges 不进入热端 LRU 尾部。

这是区别于普通 object restore 的关键创新点，可减少读放大和冷热抖动。

### 7.6 Decode Ready Barrier

vLLM decode 前必须检查：

```text
required_ranges all in {HBM, DRAM} and ready=true
sync_3fs_miss_total == 0
```

如果检查失败，当前 sequence 只能暂停、重排或降级，不能同步读 3FS/SSD。

## 8. Budgeted Anti-Cache Scheduler

### 8.1 Inner Loop

调度闭环：

```text
observe(request, residency, queue, metrics)
  -> propose(reuse, restore, demote, promote plan)
  -> evaluate(TTFT, hit rate, bytes, bandwidth, deadline, interference)
  -> decide(ADMIT, DELAY_RESTORE, REQUEUE, FALLBACK, REJECT)
  -> observe(actual metrics)
  -> update budgets and hotness
```

当前 M2/M3 的瓶颈是规则固定：M2 sweep 是离线网格，M3 queue 是 first-fit storage-only。下一步应升级为多资源预算调度。

### 8.2 候选机制评估

| 机制 | 作用 | 复杂度 | 风险 | 结论 |
|---|---|---:|---|---|
| EDF + SRPT | 先满足 deadline，短 restore 优先 | 低 | 长任务饥饿 | 作为 baseline |
| Primal-dual budget controller | 控制 3FS/H2D/network/GPU slack 使用率 | 中 | 参数过紧导致过度 delay | 推荐 |
| Contextual bandit | 学习 chunk size、prefetch lead、target tier | 中 | 探索影响 SLA | 只用于非关键路径 |
| MAP-Elites / DOE | 离线探索策略空间 | 中 | 在线迁移不稳 | 用于实验设计 |
| MPC | 多步预测队列和资源 | 高 | 模型误差大 | 后续再考虑 |

推荐第一版：**Budgeted EDF + Marginal Benefit Scheduler**。

### 8.3 调度目标

对每个 RestoreTask 计算：

```text
benefit = saved_prefill_ms + reuse_future_value
cost =
  storage_price * storage_bytes
  + network_price * network_bytes
  + h2d_price * h2d_bytes
  + gpu_slack_price * io_kernel_time
score = benefit / max(cost, epsilon)
```

准入条件：

```text
historical_kv_hit_rate_est >= 0.90
TTFT_est <= SLA_ms
TTFT_est / baseline_ttft <= 1.20
storage_util_pred <= 0.70
network_util_pred <= 0.70
h2d_util_pred <= 0.70
short_request_p99_guard_ok
sync_3fs_miss_est == 0
```

### 8.4 预算价格更新

每个周期根据资源使用率更新价格：

```text
price_r = max(0, price_r + eta * (observed_util_r - target_util_r))
```

资源越拥塞，相关 restore/demotion 任务分数越低，admission 越保守。学习策略只影响 future restore 和 demotion，不允许绕过 ready barrier。

### 8.5 防饥饿

为避免 EDF/SRPT 只服务短任务：

- 对等待时间加入 aging bonus；
- 对同一 tenant/session 设置 restore share；
- 长任务可拆成 range chunks 分批完成；
- 1M 请求默认必须有 prefetch lead 或 relaxed SLA，不允许挤占短请求 critical path。

## 9. 三种策略族

为了满足至少两种超长上下文方案的 benchmark，建议把 KV Anti-Caching 与现有方案组织成三类策略：

### 9.1 Policy A：Admission-Window Anti-Caching

路径：3FS/SSD -> DRAM -> HBM。

特点：

- pre-pass 后发现 cold ranges；
- 若 admission window 足够，后台 restore 到 DRAM；
- restore 完成后 requeue/admit；
- decode 不碰 cold tier。

适合：idle session restore、多轮 agent 追加、共享长文档。

### 9.2 Policy B：DualPath Anti-Caching

路径：

- storage-to-prefill：prefill worker 提前拉历史 KV，并与 delta prefill overlap；
- storage-to-decode：decode worker 只接收 ready critical ranges。

特点：

- scheduler 根据 prefill/decode 侧资源 slack 选择路径；
- 两条路径仍共享 correctness key、manifest 和 ready barrier；
- 与 DualPath 论文形成直接对比或融合。

适合：prefill/decode 分离部署、agentic workload、存储带宽瓶颈明显场景。

### 9.3 Policy C：Tutti-Compatible Local Fast Path

路径：local NVMe -> HBM，尽量绕开 CPU critical path。

特点：

- 作为本地 NVMe fast path 插件；
- control plane 仍由 Anti-Caching scheduler 决定加载什么；
- data plane 可替换为普通 io_uring、GDS、Tutti-like GPU I/O。

适合：本地 NVMe 丰富、GPU/SSD 拓扑明确的单机或单节点场景。

## 10. 实施路线

### M3.10-A：真实 cold-tier executor

目标：把 M3.9 的元数据迁移替换成真实目录/设备级迁移。

任务：

- `KVBlockManifest` 增加 `cold_uri`、`object_id`、`offset_table`、`checksum`、`size_bytes`。
- 实现 `demote_to_cold_object()` 和 `restore_from_cold_object()`。
- 支持 DRAM store dir、local NVMe cold dir、3FS mount dir 三种路径。
- 后台 executor 记录 copy latency、tail latency、bytes、checksum failure。
- 保留 fake executor 供单元测试。

验收：

- cold object 存在但 `ready=false` 时 connector 必须失败。
- executor 完成后 same request re-admission 或下一请求可 ADMIT。
- `sync_3fs_miss_total == 0`。

### M3.10-B：KVPrePass 与 partial promote

任务：

- 将 prefix/session lineage 拆成多个 `KVRange`。
- admission 前输出 `ready_set/cold_set/missing_set`。
- cold object 支持 offset table 和 valid bitmap。
- restore 只 mark selected ranges ready。

验收：

- 同一 cold object 内只恢复一部分 ranges，不会把整个 object 计为 hit。
- 未恢复 ranges 仍不能被 connector load。
- 读放大、promote bytes、useful bytes 可观测。

### M3.11：Budgeted Anti-Cache Scheduler

任务：

- 把 `PrefetchQueue` 升级为多资源队列。
- 建模 storage、network、DRAM、H2D、GPU slack。
- 实现 EDF + benefit/cost 排序。
- 增加 primal-dual resource price。
- 输出 per-task decision trace。

验收：

- resource utilization 超阈值时 admission 自动收紧。
- short-request P99 guard 生效。
- low-locality 负对照不进入 SLA 快路径。

### M4：策略族与强 baseline

任务：

- 实现 Policy A/B/C 的统一接口。
- 与 vLLM APC、LMCache、DualPath-style、自研 naive offload 对比。
- 扩展在线矩阵到 512、1K、4K、16K、32K、128K。
- 设计 1M 分解式实机实验和 scaling 实验。

验收：

- 至少两种方案完成同 workload benchmark。
- 有 TTFT、ITL、hit rate、deadline miss、bandwidth、short P99、cost 指标。

## 11. 实验设计

### 11.1 Workload

必须包含：

- `session_append`: 1M 历史上下文，小 delta 追加。
- `shared_prefix`: 多请求共享长文档/system prompt。
- `agent_trace`: tool result 和多轮调用形成 session lineage。
- `idle_restore`: 会话长时间 idle 后恢复。
- `mixed_short_long`: 短请求和 1M 请求混部。
- `low_locality`: 随机 1M，作为负对照。

### 11.2 Baseline

至少包括：

- vLLM native PagedAttention/APC。
- LMCache DRAM。
- LMCache SSD/GDS，如果环境支持。
- naive per-request offload。
- DualPath-style restore。
- Tutti 或 Tutti-like local NVMe fast path，如代码/环境不可用则做 design-level baseline 和可复现实验替代。

### 11.3 指标

核心指标：

- TTFT p50/p90/p99。
- ITL p50/p90/p99。
- `TTFT_tiered / TTFT_baseline`。
- `historical_kv_hit_rate`。
- `byte_hit_rate`。
- `admitted_sla_hit_rate`。
- `sync_3fs_miss_total`。
- prefetch deadline miss。
- restore useful bytes / restored bytes。
- read amplification。
- 3FS/NVMe/network/H2D utilization。
- short-request P99 interference。
- cost per 1M generated tokens。

### 11.4 1M 证明方式

优先级从高到低：

1. 真实 1M-capable model + 真实 1M context + 真实 cold-tier restore。
2. 128K/256K/512K/1M scaling curve + 1M KV object restore + shorter decode validation。
3. 仿真外推只能作为补充，不能单独支撑主结论。

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 3FS tail latency 抖动 | deadline miss | budget controller + admission 收紧 + prefetch lead |
| cold object 粒度不合适 | 读放大或元数据膨胀 | 32MB-128MB sweep，报告 useful/restored bytes |
| partial promote 复杂 | manifest 状态错误 | version、valid bitmap、checksum、单测覆盖 |
| hit rate 90% 被质疑 | 论文风险 | 明确定义 admitted workload，报告 low-locality 负对照 |
| baseline 不公平 | 审稿风险 | 分别报告 full prefill、DRAM-ready、APC、LMCache、DualPath |
| online learning 伤 SLA | 生产风险 | 学习只影响 future prefetch，不影响 ready barrier |
| 1M 实机资源不足 | 主张不足 | 做分解式证明，但主论文需如实标注限制 |

## 13. 论文定位

建议论文主张：

> KV Anti-Caching turns long-context KV reuse from an opportunistic cache hit into an admission-controlled, memory-primary serving protocol. By combining request pre-pass, partial KV promotion, cold-object construction, and budgeted non-blocking restore, it supports exact million-token serving without synchronous cold-tier misses on the decode path.

可用标题：

- `KV Anti-Caching: Admission-Controlled Cold KV Management for Exact Million-Token LLM Serving`
- `No Cold Miss on the Decode Path: Anti-Caching KV Restoration for Long-Context LLM Inference`
- `Memory-Primary KV Serving: Anti-Caching for Exact Long-Context LLMs`

论文贡献应写成：

1. KV Anti-Caching 抽象：memory-primary、多级 residency、cold objects、partial promotion。
2. KVPrePass admission：执行前枚举 required KV，非阻塞 cold restore。
3. Budgeted Anti-Cache Scheduler：用多资源预算同时控制 TTFT、hit rate、带宽和短请求干扰。
4. vLLM 原型与 3FS/NVMe 实验：exact semantics、no sync cold miss、1M scaling、强 baseline。

如果只完成 M3.10-A 和小矩阵，适合作 workshop 或技术报告；若完成 M3.10-B、M3.11、M4 的 1M/强 baseline，则具备投 CCF-A 系统会议的基本形态。

## 14. 参考资料

- 本地项目设计：`/root/KV/1M_Tokens_KV_Cache_分层管理技术方案.md`
- M3 connector/sidecar 接口：`/root/KV/docs/specs/m3_connector_sidecar_minimal_interface.md`
- 当前相关工作规划：`/root/KV/docs/research/kv_multitier_related_work_and_design_plan.md`
- DeBrabant et al., "Anti-Caching: A New Approach to Database Management System Architecture", PVLDB 2013: https://www.vldb.org/pvldb/vol6/p1942-debrabant.pdf
- Tutti: https://arxiv.org/abs/2605.03375
- DualPath: https://arxiv.org/abs/2602.21548
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- DeepSeek 3FS: https://github.com/deepseek-ai/3FS
