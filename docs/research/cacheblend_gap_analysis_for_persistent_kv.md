# CacheBlend Gap Analysis for Persistent KV Anti-Caching

日期：2026-05-29

## 结论先行

CacheBlend 已经很好地覆盖了一个重要问题：RAG 输入包含多个可复用文本块时，非前缀 chunk 的 KV 不能简单拼接，因为它缺少与前文的 cross-attention。CacheBlend 用 selective KV recompute 修复这个质量问题，并把重算与 KV loading pipeline 起来。

但 CacheBlend 没有完全覆盖我们的目标问题。它更像是 **within-request RAG chunk fusion**，而不是 **cross-request persistent historical KV anti-caching**。最可写成论文的点是：

> CacheBlend already makes cached RAG chunk fusion practical. However, in multi-turn long-context serving, reusable KV often becomes cold persistent state across idle periods, restarts, and engine boundaries. In this cold-resume scenario, a cached KV object on SSD/3FS is not execution-ready: loading it at request arrival can dominate TTFT, and opportunistic reuse lacks an admission invariant. We introduce Persistent KV Anti-Caching, which indexes committed historical KV, runs a PrePass before admission, restores and verifies cold KV ahead of online execution, and admits only GPU/CPU-ready KV ranges.

最推荐主打场景：**idle multi-turn session resume with a long committed history and a short new turn**。例如企业助手、代码助手、科研 agent、LongMemEval-style multi-session memory。用户离开后历史 KV 从 HBM/DRAM 冷却到 SSD/3FS；下一轮只追加几十到几百 token。CacheBlend 的 chunk fusion 思路不能保证 cold KV 在请求进入 prefill/decode 前 ready，也不处理 restore deadline、ready barrier、restart/cross-engine persistence 和 SLA admission。

## CacheBlend 做得好的地方

CacheBlend 的问题定义很清楚：RAG 请求往往包含多个 retrieved chunks，只有第一个 chunk 是 prefix；prefix caching 只能复用第一段，full KV reuse 又忽略跨 chunk 的 cross-attention。它的核心方法是：

1. 将每个文本 chunk 的 KV 预计算并缓存。
2. 在线请求中把多个 chunk 的 KV 组合起来。
3. 选择性重算一小部分 high-KV-deviation tokens，修复 cross-attention。
4. 将 selective recompute 与下一层 KV loading 重叠，使慢存储上的 KV loading 尽量不增加 TTFT。

因此，不能把论文叙事写成“我们首次把 KV 存到 SSD 并跨请求复用”。这个空间已经被 CacheBlend/LMCache/RAGCache/AttentionStore/KVDrive/Tutti 等工作挤得很满。

## 对我们目标而言的粗糙处

这些不是 CacheBlend 自身的失败，而是它没有试图解决我们的目标场景。

### 1. 目标是 RAG chunk fusion，不是 committed historical KV lifecycle

CacheBlend 处理的是请求内多个 retrieved text chunks 如何融合。它没有定义 session lineage、committed KV ranges、turn append、idle demotion、restart recovery、cross-engine reuse 这些生命周期语义。

我们的机会：把历史 KV 从“可复用文本片段的中间状态”提升为“多轮服务状态对象”。对象有 correctness key、token range、tier state、checksum、object id、restore/load state。

### 2. 选择性重算是近似质量路径，不是 exact state reuse 路径

CacheBlend 为了支持非前缀 chunk，接受 partial recompute 和质量近似，评估的是 F1/Rouge-L 是否接近 full recompute。我们的主线可以反过来收窄：只处理 exact prefix / committed history，但保证语义严格、可验证、可审计。

适合场景：企业记录、代码仓库上下文、医疗/法律/科研 agent traces，系统不能依赖“质量差不多”的 KV 融合路径。

### 3. 存储层设计仍是简化的 cache store

CacheBlend 论文中 KV store 的系统假设较轻：单层 storage device、hash table、LRU eviction、`torch.load()`/`torch.save()`、layer-wise fetch。它关注的是 load/recompute pipeline，而不是 cold-tier object layout、3FS/SSD tail latency、restore queue、CPU_READY/GPU_READY 区分和 packed extent。

我们的机会：把数据库 anti-caching 的 Evicted Table / Block Table / PrePass 映射到 KV：

- KV Evicted Index 常驻内存；
- SSD/3FS 上是 packed cold object；
- PrePass 先分类 `GPU_READY/CPU_READY/SSD_COLD/FETCHING/LOADING/MISSING/MISMATCH`；
- admission guard 不允许执行期同步冷层 miss。

### 4. Pipeline 假设不等于 SLA admission

CacheBlend 的关键优化是让 per-layer KV loading 隐藏 selective recompute。但它的控制器主要用平均 recompute/load delay 和 storage cost 做选择。真实多轮服务中更难的是：请求到达时 KV 可能完全在冷层，且多个 session 同时恢复，tail latency 和 queueing delay 会主导 TTFT。

我们的机会：把问题从“本 job 内如何 hide recompute”改成“请求到达前能否使 cold history 变 ready”。这需要 lead-time planner、restore deadline、queue depth、bandwidth admission、delay/fallback/reject。

### 5. 评测边界偏小而干净

CacheBlend 评测主要是 RAG/QA/summarization，典型 top chunks、512-token chunks、单节点 NVMe、少量模型和数据集。它没有覆盖：

- 8K/16K/32K/128K/1M committed history cold resume；
- vLLM restart 后 exact external KV load；
- 多租户 restore storm；
- 3FS/shared cold tier；
- negative-control low-locality workload；
- p95/p99 deadline miss；
- restore-inclusive vs online-only TTFT 分解。

这些正好可以成为我们的系统论文评测边界。

## 最推荐论文切入点

### 场景

**Cold-resume multi-turn long-context serving**：

1. 一个 session 已经有长历史，例如 8K/16K/32K/128K tokens。
2. 历史上下文已经完成 prefill，并以 exact KV 形式 committed。
3. session idle、engine restart、capacity pressure 或 cross-engine migration 使 KV 从 HBM/DRAM demote 到 SSD/3FS。
4. 用户或 agent 下一轮只追加很短 suffix。
5. 如果请求到达后才 restore，TTFT 被 SSD/3FS restore 主导；如果直接 full prefill，则重复计算长历史；如果把 SSD 当慢主存，则 decode/prefill 可能同步 miss。

### 可以写成的 thesis

> CacheBlend shows that precomputed KV can be fused for RAG chunks without large quality loss. However, it assumes the system can load and repair cached chunks within a request. In cold-resume multi-turn serving, reusable KV is a persistent cold object, not a ready cache entry. The missing abstraction is admission-controlled KV anti-caching: the system must know before execution which historical KV ranges are required, where they reside, and whether they can become ready within the online deadline.

### 方法

Persistent KV Anti-Caching:

1. **Session lineage + committed KV ranges**：只复用已经 committed 的历史 KV，避免 arbitrary non-prefix fusion 的语义风险。
2. **KV Evicted Index**：内存中维护 range -> tier/object/checksum/state。
3. **KV PrePass**：请求进入 vLLM 前枚举 required historical KV ranges。
4. **Packed cold object**：SSD/3FS 存大对象和 extent metadata，避免 per-layer 小文件。
5. **Two-phase readiness**：SSD->CPU restore 和 CPU->GPU load 分开建模，`CPU_READY` 不等于 `GPU_READY`。
6. **Admission guard**：ready 则 admit；可按 deadline 恢复则 delay；不可恢复则 full-prefill fallback 或 reject；执行路径不允许同步 SSD/3FS miss。

## 实验矩阵建议

### Workloads

1. **Synthetic session append**：2K/8K/16K/32K/128K prefix + 32/128/512 suffix。
2. **LongMemEval-S cold resume**：真实 multi-session memory prefix + question suffix。
3. **Agent idle window**：tool call 或 retrieval 阶段给出 0/1/3/5/10s lead time。
4. **Resume storm**：N 个 idle sessions 同时恢复，测 p95/p99 deadline miss。
5. **Low-locality negative control**：随机历史不复用，系统应 delay/fallback/reject，而不是污染 cold-tier scheduler。

### Baselines

- B0 full prefill。
- B1 vLLM APC hot cache。
- B2 vLLM APC restart/cold cache。
- B3 DRAM-ready external KV reuse。
- B4 naive request-arrival cold restore。
- B5 Persistent KV Anti-Caching PrePass。
- B6 LMCache/CacheBlend-style external KV reuse，如果可集成。
- B7 KVDrive/Tutti-style multi-tier or data-plane baseline，如果可获得 artifact。

### Metrics

- online TTFT；
- restore-inclusive TTFT；
- P95/P99 TTFT；
- prepass lead time required；
- `sync_cold_miss_total`；
- `ready_barrier_all_ready`；
- deadline miss rate；
- historical KV byte hit rate；
- restored bytes / useful bytes；
- connector load time；
- storage and H2D bandwidth utilization。

## 风险与 reviewer 可能攻击

1. **AttentionStore/LMCache 比 CacheBlend 更接近 multi-turn reuse**。回应：承认它们是强 baseline；我们的区别必须落在 cold-tier anti-caching semantics、PrePass admission、ready barrier 和 restore-inclusive SLA，而不是“能跨请求复用 KV”。
2. **如果 restore 不能隐藏，系统比 full prefill 慢**。回应：这不是隐藏，而是论文核心边界。必须报告 online TTFT 和 restore-inclusive TTFT，并证明哪些 workload 有 lead time。
3. **exact prefix/committed history 比 CacheBlend 的 arbitrary chunk reuse 窄**。回应：这是有意收窄，换取 exactness、auditability 和 persistent lifecycle；非前缀 fusion 可作为未来与 CacheBlend 结合的扩展。
4. **没有真实 3FS/生产 SSD 不够系统论文**。回应：当前原型只能是方向性证据；完整投稿需要真实 cold-tier executor、p95/p99 和强 baseline。

## 推荐一句话定位

CacheBlend solves **how to reuse and repair multiple cached RAG chunks inside one request**. Persistent KV Anti-Caching solves **how to turn cold historical KV from previous requests into exact, verified, admission-ready serving state before the next request enters the LLM engine**.
