# KV Anti-Caching 相关工作目录

日期：2026-05-28

本文记录当前项目已经调研、讨论或纳入 baseline 视野的 KV cache / 长上下文推理 / anti-caching 相关工作。写作目标不是做最终论文综述，而是形成一个可持续维护的“相关工作台账”：每篇工作说明它解决什么问题、具体怎么做、对 Persistent KV Anti-Caching 的启发，以及为什么仍没有完全覆盖我们的研究问题。

我们的研究问题是：在多轮长上下文服务中，把已经生成并提交的历史 KV 当作长期服务状态管理；HBM/CPU 是主存，SSD/3FS 是冷区；请求进入 prefill/decode 前，系统必须知道 required KV 在哪里、是否 ready、能否在 SLA 内恢复。

## 1. 快速索引

| 类别 | 工作 | 主要作用 | 与本项目关系 |
|---|---|---|---|
| 数据库反缓存 | Anti-Caching | 主存为主、冷数据落盘、pre-pass 找冷数据 | 本项目方法论原型 |
| GPU KV 管理 | PagedAttention / vLLM | KV 分页、减少碎片、共享 block | GPU 内 KV 管理基线 |
| Prefix reuse | vLLM APC | block hash 前缀复用 | 精确 prefix reuse 基线 |
| Prefix tree reuse | SGLang / RadixAttention | radix tree 管理共享前缀 | agent 多轮复用基线 |
| PD 分离 | DistServe | prefill/decode 分离到不同 GPU | 两阶段资源隔离基线 |
| PD 分离 | Splitwise | 按阶段拆分 LLM inference 资源 | PD 资源解耦相关工作 |
| KV 中心服务 | Mooncake | KVCache-centric disaggregated serving | 强 baseline |
| 外部 KV cache | LMCache | engine 外 KV cache layer | 强 baseline |
| RAG KV 复用 | RAGCache | knowledge tree + GPU/host 层级缓存 | 证明文档 KV 复用价值 |
| 非前缀复用 | CacheBlend | cached chunks + selective recompute | 非前缀复用启发 |
| 压缩传输 | CacheGen | KV 压缩与流式传输 | 带宽优化启发 |
| Offloading | FlexGen | GPU/CPU/disk tensor offload | 早期 offload 基线 |
| SSD KV | Tutti | SSD-backed KV fast path | 强 data-plane baseline |
| 双路径恢复 | DualPath | storage-to-prefill / storage-to-decode | 强 scheduler baseline |
| KV 恢复调度 | CacheFlow | token/layer/GPU 三维恢复 | restore 并行调度启发 |
| 多级 KV | KVDrive | GPU/DRAM/SSD 协同管理 | 主竞争 baseline |
| 动态 KV 管理 | InfiniGen | 按需加载长期上下文 KV | 长上下文 offload 相关 |
| 分布式 KV | Infinite-LLM / DistKV-LLM | DistAttention + Distributed KVCache | 百万上下文内存池路线 |
| KV 剪枝 | H2O | heavy hitter oracle 保留关键 KV | 近似/压缩方向 |
| KV 剪枝 | Scissorhands | persistence of importance 驱逐 | 近似/压缩方向 |
| Query-aware sparse | Quest | query-aware 选择关键 KV | decode-time sparse KV 方向 |
| KV offload | ShadowKV | 长上下文 KV offloading | 候选强 baseline |
| KV offload | MagicPIG | KV offloading / prefetch 优化 | 候选强 baseline |
| KV 服务层 | KVServe | KV cache serving / 压缩优化 | 候选相关工作 |
| 动态虚拟内存 | vAttention | OS virtual memory 管理 KV | PagedAttention 替代路线 |

## 2. 数据库 Anti-Caching 原型

### 2.1 Anti-Caching: A New Approach to Database Management System Architecture

- 来源：DeBrabant et al., PVLDB 2013, `/root/KV/p1942-debrabant.pdf`；公开 PDF：https://www.vldb.org/pvldb/vol6/p1942-debrabant.pdf
- 核心问题：主存数据库性能很好，但要求所有数据都放进内存；一旦依赖 OS paging，事务会在 page fault 上阻塞。
- 具体做法：
  1. 反转传统 buffer pool 视角：主存是 primary storage，磁盘只存放冷数据，而不是磁盘为主、内存为缓存。
  2. 维护内存中的 Evicted Table，记录每个被驱逐 tuple 所在的 disk block id 和 offset；索引项可以指向内存 tuple，也可以指向 Evicted Table。
  3. 当内存水位超过阈值时，把 LRU 链上的冷 tuple 打包成固定大小 anti-cache block，写到磁盘 Block Table，并更新索引与 Evicted Table。
  4. 当事务访问到 evicted tuple 时，不在执行路径同步等待磁盘；系统进入 pre-pass，继续试跑事务以收集它需要的所有 evicted blocks。
  5. pre-pass 后 abort/rollback 该事务，后台批量读取需要的 block，把 tuple merge 回内存，然后重新执行事务。
  6. 提供 block-merge 和 tuple-merge 等恢复策略；tuple-merge 避免把未请求的冷 tuple 全部带回内存，减少读放大和反复驱逐。
- 对本项目的启发：
  - KV 版 Anti-Caching 也应维护内存中的 Residency Table，记录每个 KV range 当前在 GPU、CPU、SSD/3FS、fetching 还是 missing。
  - KVPrePass 对应数据库 pre-pass：请求进入 vLLM 前先枚举 required KV ranges，发现 cold ranges 后批量 restore，而不是让 prefill/decode 执行期遇到同步 cold miss。
  - packed cold object 对应 anti-cache block；partial promote 对应 tuple-merge，而不是把整个大对象恢复回主存。
- 不能直接解决的问题：
  - 数据库 tuple 的访问在事务逻辑中显式出现，而 LLM attention 对历史 KV 的访问更像按 position/layer/head 的批量张量访问。
  - 数据库可 abort/retry 事务；LLM 请求若已经进入 decode，再 abort 会直接破坏用户可见 SLA，所以 KV 必须在 admission 前完成 ready 判断。

## 3. GPU KV 管理与 Prefix Reuse

### 3.1 Efficient Memory Management for Large Language Model Serving with PagedAttention / vLLM

- 来源：https://arxiv.org/abs/2309.06180
- 核心问题：LLM serving 中 KV cache 很大且动态增长，传统连续内存分配会产生严重碎片和重复复制，限制 batch size。
- 具体做法：
  1. 借鉴操作系统虚拟内存，把每个请求的 KV cache 切成固定大小 block/page。
  2. 通过 block table 将逻辑 token block 映射到物理 KV block，使请求的 KV 不需要连续存放。
  3. 支持 copy-on-write 和 block 共享，让并行采样、beam search 或共享前缀请求复用同一份物理 KV。
  4. 在 vLLM serving engine 中结合 continuous batching，让 GPU 可以同时服务更多请求。
- 对本项目的启发：
  - KV range / cold object 的基本单位应尽量与 vLLM block 对齐，否则 connector 和 restore 后装载会产生额外转换。
  - PagedAttention 是 GPU 内 execution layer，适合作为 HBM 层的数据结构基础。
- 为什么没有解决我们的场景：
  - 它解决的是 GPU 内 KV 内存利用率和共享，不解决 KV 被驱逐到 SSD/3FS 后如何持久化、恢复和准入。
  - 它默认请求已进入执行系统；缺少 persistent historical KV 的请求前 PrePass 和 cold-tier ready barrier。

### 3.2 vLLM Automatic Prefix Caching

- 来源：https://docs.vllm.ai/en/latest/design/prefix_caching/
- 核心问题：许多请求共享相同 system prompt、few-shot examples、文档前缀或对话历史，重复 prefill 会浪费 GPU 计算。
- 具体做法：
  1. 将 prompt tokens 按 block 切分，为每个 block 计算 hash；hash 通常包含前序 block hash、当前 block token ids 和其他影响 KV 正确性的额外信息。
  2. 如果新请求的前缀 block hash 已存在，则直接复用对应 KV block，跳过该部分 prefill。
  3. 通过 block-level cache 管理 eviction 和 reuse，使完全相同的前缀能被精确复用。
- 对本项目的启发：
  - correctness key 应至少包含 token ids、position/range、model/tokenizer/layout 等影响 KV 语义的字段。
  - APC 是我们必须比较的 B1/B2 baseline：hot-cache 时它应非常强，restart/capacity-pressure 后它会暴露持久化不足。
- 为什么没有解决我们的场景：
  - APC 是 opportunistic cache，缓存命中依赖 KV 仍在可用 cache 中。
  - 当历史 KV 因容量压力、重启或跨实例迁移变冷时，APC 不负责从 SSD/3FS 恢复，也不提供 restore-inclusive admission。

### 3.3 Efficiently Programming Large Language Models using SGLang / RadixAttention

- 来源：https://arxiv.org/abs/2312.07104
- 核心问题：agent、工具调用、多轮对话、树状搜索等程序化 LLM 应用会产生大量共享前缀和分支，传统 request-at-a-time serving 无法充分复用。
- 具体做法：
  1. 提供 SGLang 编程接口表达复杂 LLM workflow，包括多轮调用、分支、约束解码等。
  2. 使用 RadixAttention，把请求前缀组织成 radix tree；共享 token prefix 的节点对应可复用 KV。
  3. runtime 在执行新请求时查找最长共享前缀，复用已有 KV，并管理分支节点的生命周期。
  4. 结合 batching、parallelism、structured output 等优化复杂 workflow 的吞吐。
- 对本项目的启发：
  - 多轮 agent workload 确实存在大量 prefix/tree reuse，KVPrePass 可以借鉴 prefix tree / session lineage 来枚举 required ranges。
  - shared prefix、session append、tool trace 是我们的核心 workload 类型。
- 为什么没有解决我们的场景：
  - RadixAttention 关注运行时前缀复用，不以 SSD/3FS cold tier 和 1M persistent historical KV 为核心。
  - 它没有定义 cold history 的持久 manifest、跨重启恢复、restore deadline 和准入边界。

## 4. Prefill/Decode 分离与 KV 传输

### 4.1 DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving

- 来源：https://arxiv.org/abs/2401.09670
- 核心问题：prefill 与 decode 的计算特征不同；混部在同一 GPU 上会造成干扰，并让 TTFT 与 TPOT 难以同时优化。
- 具体做法：
  1. 将 prefill 和 decoding 分配到不同 GPU 或 GPU 组，消除两阶段直接资源干扰。
  2. 根据 TTFT 和 TPOT SLO，为 prefill/decode 分别选择资源数量和并行策略。
  3. 考虑集群带宽，将两个阶段放置在通信成本可接受的位置。
  4. 用 goodput 作为优化目标，即在满足 latency SLO 的请求比例下最大化吞吐。
- 对本项目的启发：
  - 我们的 ready barrier 也必须区分 prefill reuse ready 和 decode execution ready。
  - restore 调度要考虑 prefill/decode 阶段的不同 deadline 和资源竞争。
- 为什么没有解决我们的场景：
  - DistServe 解决计算阶段解耦，不解决历史 KV 已经在 SSD/3FS 冷区时的持久化恢复。
  - 如果 required KV 不 ready，PD 分离本身不会告诉系统是 delay、fallback、restore 还是 reject。

### 4.2 Splitwise: Efficient Generative LLM Inference Using Phase Splitting

- 来源：https://www.microsoft.com/en-us/research/uploads/prodnew/2023/12/Splitwise_ISCA24.pdf
- 核心问题：LLM inference 的 prompt processing 和 token generation 两个阶段在计算/内存/通信需求上不同，单一资源池调度会浪费资源。
- 具体做法：
  1. 把 LLM inference 分成不同 phase，给不同 phase 匹配更合适的硬件资源和并行方式。
  2. 将 prompt-heavy 和 generation-heavy 的部分分开调度，减少互相阻塞。
  3. 通过阶段间通信转移必要状态，使系统获得更高整体效率。
- 对本项目的启发：
  - KV Anti-Caching 也应把 restore 放在请求 critical path 之外，并根据阶段 deadline 做调度。
  - 可以作为 prefill/decode phase separation 的背景工作。
- 为什么没有解决我们的场景：
  - 它主要关注 phase splitting 的 compute scheduling，不维护 persistent KV object 的 residency / correctness / restore manifest。

### 4.3 Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- 来源：https://arxiv.org/abs/2407.00079；项目文档：https://kvcache-ai.github.io/Mooncake/
- 核心问题：长上下文请求中，KVCache 已经成为 serving 系统的中心资源；传统以 GPU compute 为中心的架构无法充分利用 CPU/DRAM/SSD，也难以满足 SLO。
- 具体做法：
  1. 将 prefill cluster 和 decoding cluster 解耦，围绕 KVCache transfer 组织 serving pipeline。
  2. 利用 GPU 集群中未充分使用的 CPU、DRAM 和 SSD 形成 disaggregated KVCache。
  3. 设计 KVCache-centric scheduler，在吞吐和 latency SLO 之间做平衡。
  4. 面对高负载场景，使用预测式 early rejection，避免系统接受注定无法满足 SLO 的请求。
  5. 在真实 Kimi 服务 workload 上评估长上下文场景的请求处理能力。
- 对本项目的启发：
  - 它证明 KVCache-centric serving 是正确系统方向，是非常强的 baseline。
  - 我们的 admission/reject 叙事不能停留在概念层面，必须有 metrics 和 workload boundary。
- 为什么没有完全覆盖我们的场景：
  - Mooncake 的中心是 disaggregated serving 与 KVCache 调度，不是从数据库 anti-caching 启发的 persistent cold-history recovery protocol。
  - 我们必须突出 committed historical KV 的 exact correctness key、KVPrePass、packed cold object、restore-inclusive admission 和 prefill/decode 双 ready invariant。

### 4.4 vLLM NIXL Connector / KV Transfer

- 来源：https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
- 核心问题：prefill/decode 解耦或跨实例 serving 需要把 KV cache 从 producer 传给 consumer。
- 具体做法：
  1. 通过 vLLM 的 KV transfer connector 抽象，让不同后端负责 KV 的发送、接收和缓存。
  2. NIXL connector 支持 disaggregated serving 场景中的 KV 传输。
  3. connector 与 vLLM scheduler/worker 结合，在请求生命周期中触发 save/load。
- 对本项目的启发：
  - 我们当前 sidecar + connector 原型应尽量贴近 vLLM 官方 connector 边界。
  - control plane 负责 admission/restore；connector 只加载已经 ready 的 KV。
- 为什么没有解决我们的场景：
  - connector 是机制接口，不是完整 cold-tier policy；它不决定哪些 KV 应被恢复，也不提供 SLA 准入。

## 5. 外部 KV Cache、RAG KV 复用与非前缀复用

### 5.1 LMCache

- 来源：https://docs.lmcache.ai/；项目记录中的论文入口：https://arxiv.org/abs/2510.09665
- 核心问题：LLM serving 中许多 prompt prefix 或文本片段会跨请求、跨 engine 重复出现，应把 KV cache 作为 engine 外的可复用服务层。
- 具体做法：
  1. 在 vLLM/SGLang 等 engine 外维护 KV cache layer，提供 lookup、store、move、pin 等能力。
  2. 支持 GPU、CPU、local storage、remote storage 等多级后端。
  3. 在多轮 QA、RAG、shared prompt 等场景中复用已计算 KV，减少 prefill 计算和 TTFT。
  4. 通过 connector 与 serving engine 集成，使 KV 可以跨请求甚至跨实例复用。
- 对本项目的启发：
  - LMCache 是最接近的 persistent KV cache baseline，必须纳入后续 B6 或强 baseline 组。
  - 我们可以借鉴其 cache layer 接口和后端抽象。
- 为什么没有完全覆盖我们的场景：
  - LMCache 关注 cache layer 能力；我们的论文差异要落在 anti-caching semantics：request PrePass、cold-tier restore deadline、two-phase ready barrier、低局部性 delay/reject。
  - 如果我们只做“外部 KV cache”，很容易被 LMCache 覆盖。

### 5.2 RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation

- 来源：https://arxiv.org/abs/2404.12457；ACM DOI：https://dl.acm.org/doi/10.1145/3768628
- 核心问题：RAG 会把检索到的知识注入 prompt，导致长序列 prefill 成本高；但热门知识片段会被反复检索，存在缓存机会。
- 具体做法：
  1. 针对 RAG workload 分析瓶颈，发现知识注入导致长序列计算和内存成本高。
  2. 将检索知识的中间状态组织成 knowledge tree，利用文档/片段之间的公共结构复用 KV。
  3. 在 GPU 和 host memory hierarchy 中缓存这些中间状态。
  4. 设计考虑 LLM inference 特征和 RAG retrieval pattern 的 replacement policy。
  5. 动态重叠 retrieval 与 inference，减少端到端 latency。
  6. 基于 vLLM 和 Faiss 实现，并报告 TTFT 和 throughput 改进。
- 对本项目的启发：
  - 企业 RAG 中“热门文档片段 KV 复用”是真实而强的应用场景。
  - knowledge tree 可启发我们对 shared document prefix / codebase chunk 建索引。
- 为什么没有完全覆盖我们的场景：
  - RAGCache 主要处理检索知识片段，而不是多轮会话历史的 persistent committed KV。
  - 它没有把 SSD/3FS cold tier、restore-inclusive admission、prefill/decode ready barrier 作为核心不变量。

### 5.3 CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

- 来源：https://arxiv.org/abs/2405.16444
- 核心问题：RAG 中可复用内容不总是严格 prefix；直接拼接不同 cached chunks 的 KV 会破坏 attention 语义，因为 chunk 之间缺少交互。
- 具体做法：
  1. 将可复用文档 chunk 的 KV 预先缓存。
  2. 在线请求到来后，把多个 cached chunks 与当前 query 融合，而不是从头 prefill 所有文本。
  3. 通过选择性 recompute 少量 token，让跨 chunk 和 query 的 attention 关系得到修正。
  4. 在保证质量的前提下降低 RAG 请求的 TTFT。
- 对本项目的启发：
  - 它指出“非前缀 KV 复用”并非简单拼接，必须处理 attention correctness。
  - 我们第一阶段应坚持 strict prefix / committed range exact reuse，避免把语义修复问题和 cold restore 问题混在一起。
- 为什么没有完全覆盖我们的场景：
  - CacheBlend 接受选择性重算和融合近似路径；我们当前目标是 exact persistent historical KV。
  - 它没有负责多级 residency、SSD/3FS restore budget 和 admission guard。

### 5.4 CacheGen: KV Cache Compression and Streaming for Fast LLM Serving

- 来源：https://arxiv.org/abs/2310.07240
- 核心问题：KV cache 跨节点或跨阶段传输时体积巨大，网络传输会成为服务瓶颈。
- 具体做法：
  1. 对 KV cache 做压缩，减少传输字节数。
  2. 利用 KV 的数值分布和 token/layer 特征选择编码方式。
  3. 支持流式传输，使接收端可以边接收边恢复或使用。
  4. 在分布式 serving 场景中减少 TTFT/传输延迟。
- 对本项目的启发：
  - packed cold object 后续可以叠加压缩 tier，降低 3FS/SSD 带宽压力。
  - 需要报告 useful/restored bytes 和压缩后的 effective bandwidth。
- 为什么没有完全覆盖我们的场景：
  - CacheGen 优化的是 KV 传输体积，不决定哪些 KV 需要恢复、何时准入、恢复不完如何处理。
  - 压缩可能引入精度/语义风险，第一版 exact path 不应依赖有损压缩达成指标。

## 6. 多级存储、SSD-backed KV 与恢复调度

### 6.1 FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

- 来源：https://arxiv.org/abs/2303.06865
- 核心问题：超大模型权重、activation 和 KV cache 可能超过单 GPU 内存，如何用 GPU/CPU/disk offloading 在资源受限环境中跑推理。
- 具体做法：
  1. 将模型权重、KV cache、activation 在 GPU、CPU、disk 之间分层放置。
  2. 使用线性规划/搜索选择 offloading 策略，平衡内存容量和 I/O。
  3. 通过大 batch 和流水化，提高吞吐。
  4. 目标更偏 offline / throughput-oriented generation，而不是在线低 TTFT。
- 对本项目的启发：
  - 证明 CPU/disk 可以作为 LLM inference 的容量层，但必须显式建模数据搬运。
  - 可以作为 naive offload 的历史背景。
- 为什么没有完全覆盖我们的场景：
  - FlexGen 不是面向多轮 long-context serving 的 persistent KV reuse。
  - 它不提供 request PrePass、ready barrier 和低尾延迟 SLA admission。

### 6.2 Tutti: SSD-backed KV Cache for Practical Long-context LLM Serving

- 来源：https://arxiv.org/abs/2605.03375
- 核心问题：长上下文 serving 中 KV cache 超过 HBM，SSD 可以提供容量，但 CPU 参与和普通 I/O path 会造成高延迟。
- 具体做法：
  1. 构建 SSD-backed KV cache，让冷 KV 存在本地 NVMe/SSD。
  2. 设计 GPU-centric object store，减少 CPU 在 HBM 与 SSD 数据通路中的介入。
  3. 使用 GPU io_uring、scatter-gather list 等机制优化 SSD 到 GPU 的数据面。
  4. 设计 slack-aware I/O scheduling，在 GPU 有空隙时恢复 KV，降低 stall。
  5. 目标是让 local SSD fast path 更接近可用于在线长上下文 serving。
- 对本项目的启发：
  - Tutti 是强 data-plane baseline；我们不应把“更快读 SSD”当作唯一创新。
  - 我们可以把 Tutti-like executor 当作 cold-tier restore 后端。
- 为什么没有完全覆盖我们的场景：
  - Tutti 重点是 SSD fast path；我们的问题还包括哪些 KV 该变冷、何时恢复、是否准入、如何保证 prefill/decode 无同步 cold miss。
  - 它不直接处理 3FS/shared cold tier 的全局 admission 和多轮 committed KV 的 correctness manifest。

### 6.3 DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

- 来源：https://arxiv.org/abs/2602.21548
- 核心问题：agentic LLM inference 中 KV storage I/O 会成为瓶颈；单一路径从 storage 到 prefill 或 decode 可能造成带宽拥塞。
- 具体做法：
  1. 提出 storage-to-prefill 和 storage-to-decode 两条数据路径。
  2. 当 prefill 侧 storage/NIC 拥塞时，可以让部分数据通过 decode 侧路径进入系统。
  3. 通过全局调度平衡两条路径的带宽，减少某一侧空转或拥塞。
  4. 面向 agentic workload 设计，关注多轮和工具调用带来的大 KV 数据搬运。
- 对本项目的启发：
  - DualPath 应作为后续强 baseline 或系统策略族：我们的 scheduler 可以选择 storage-to-prefill、storage-to-decode 或 delay。
  - 它提醒我们 restore path 不应只建模单个 SSD->CPU->GPU 流。
- 为什么没有完全覆盖我们的场景：
  - DualPath 解决的是“从哪条路径搬更快”；它不决定哪些历史 KV 应提前恢复，也不维护 persistent historical KV 的 exact readiness。
  - 如果请求到达后才开始 restore，restore-inclusive TTFT 仍可能高于 full prefill baseline。

### 6.4 CacheFlow: Token/Layer/GPU-aware KV Restoration

- 来源：https://arxiv.org/abs/2604.25080
- 核心问题：KV restoration 不是单一 I/O 操作，而是跨 token、layer、GPU 的复杂并行恢复问题；粗粒度 restore queue 会浪费并行度并放大 TTFT。
- 具体做法：
  1. 将 KV restore 分解到 token chunk、layer group、GPU/device 三个维度。
  2. 针对不同维度的并行度和依赖关系设计 batch-aware scheduler。
  3. 通过调度恢复顺序，让最影响当前请求的 KV 优先 ready。
  4. 目标是降低 restore 进入 TTFT 的部分，并提高多请求共享下的 restore 吞吐。
- 对本项目的启发：
  - packed cold object 不能只是一个大文件；需要 extent / layer group / token range 元数据，才能支持 partial promote。
  - M3.14-E 以后应避免 read_bytes + 二次切片，转向顺序读、pread 或批量读。
- 为什么没有完全覆盖我们的场景：
  - CacheFlow 更偏 restoration scheduler；我们还要处理 admission semantic：恢复不完是否 delay/fallback/reject。
  - 它不一定覆盖 3FS/shared cold tier 的带宽水位和多租户 SLA 边界。

### 6.5 KVDrive: Holistic Multi-tier KV Cache Management for Long-context LLM Inference

- 来源：https://arxiv.org/abs/2605.18071
- 核心问题：长上下文 inference 中，KV cache 超过 HBM 后，GPU/DRAM/SSD 三层之间的数据放置和搬运成为主要瓶颈。
- 具体做法：
  1. 将 GPU memory、host DRAM 和 SSD 作为整体 KV cache 层级管理。
  2. 识别 decode 中更 critical 的 KV working set，围绕 sliding window / critical KV 管理热数据。
  3. 设计跨层 placement 与 pipeline scheduling，重叠计算和数据搬运。
  4. 管理 GPU/DRAM/SSD 之间的带宽和容量，使 active long-context decoding 能持续推进。
- 对本项目的启发：
  - KVDrive 是我们的主竞争 baseline，说明“多级 KV 管理”本身已经不新。
  - 它的数据面思想，如 packed layout、attention-informed placement、pipeline overlap，都可以吸收。
- 为什么没有完全覆盖我们的场景：
  - KVDrive 的强中心是 active long-context decode 时 critical KV working set 如何搬运。
  - 我们聚焦的是多轮请求之间已经冷却的 committed historical KV，如何 exact、可验证、可准入地恢复为 prefill/decode 可复用状态。

### 6.6 InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management

- 来源：项目已检索为 long-context dynamic KV cache management 方向；后续需要以最终论文版本为准。
- 核心问题：长上下文 generation 中，不是所有历史 KV 对每一步 decode 都同等重要；如果全部放在 GPU，会超出容量。
- 具体做法：
  1. 将长上下文 KV 放在 CPU/外部内存中，只把当前 decode 需要的部分搬到 GPU。
  2. 利用 attention/query 信息估计哪些 KV 更可能被访问。
  3. 通过动态预取和 eviction 控制 GPU 中的 KV working set。
  4. 尽量把 offload/prefetch 与计算重叠。
- 对本项目的启发：
  - attention-informed hotness sketch 可以作为 P1 优化方向。
  - 但这种策略应服从 ready barrier，不能绕过 exact readiness。
- 为什么没有完全覆盖我们的场景：
  - 重点是 decode-time working set 管理，不是多轮 committed historical KV 的持久化和 admission。

### 6.7 Infinite-LLM / DistKV-LLM

- 来源：https://arxiv.org/abs/2401.02669
- 核心问题：百万级上下文需要远超单卡 HBM 的 KV 容量；可以通过分布式 KVCache 和分布式 attention 扩展上下文长度。
- 具体做法：
  1. 使用 DistAttention 将 attention 计算和 KV 存储分布到多个节点/设备。
  2. 建立 Distributed KVCache，避免单 GPU 存储完整 1M 上下文 KV。
  3. 通过通信协议和并行执行支持超长上下文。
- 对本项目的启发：
  - 这是“分布式 GPU/CPU 内存池”路线，可作为 1M context 的另一类 baseline。
  - 我们的 SSD/3FS cold tier 路线应强调成本结构、冷数据持久化和 restore admission。
- 为什么没有完全覆盖我们的场景：
  - 它更像扩展执行内存池，不是把冷历史 KV 持久化到 capacity tier 后再按请求恢复。
  - 对多轮 enterprise workflow 中的 cold/warm lifecycle 和 exact reuse 没有专门建模。

## 7. 近似、压缩、剪枝与 Query-aware KV 选择

### 7.1 H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

- 来源：https://arxiv.org/abs/2306.14048
- 核心问题：KV cache 随上下文增长线性扩大，但 attention 中少量 heavy hitter token 对生成影响更大。
- 具体做法：
  1. 观察 attention pattern，识别被频繁关注的 heavy hitter tokens。
  2. 在 KV cache 中保留 heavy hitters 和最近 tokens，驱逐较不重要的 KV。
  3. 在有限 KV budget 下保持模型质量，同时降低显存占用。
  4. 将 KV eviction 与 decoding 过程结合，使长生成可以在较小 cache 下进行。
- 对本项目的启发：
  - 可用于后续 approximate tier 或 admission fallback：当 exact path 无法满足 SLA 时，是否允许质量受控降级。
- 为什么没有完全覆盖我们的场景：
  - H2O 改变了完整 attention 的 KV 可见集合，不满足第一版 exact attention 目标。
  - 它不管理 SSD/3FS cold object，也不提供 persistent historical KV 恢复协议。

### 7.2 Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression

- 来源：https://arxiv.org/abs/2305.17118
- 核心问题：长序列推理中 KV cache 太大，但 token 的重要性具有持久性，可以根据早期重要性做保留/驱逐。
- 具体做法：
  1. 提出 persistence of importance hypothesis：过去重要的 token 往往未来仍重要。
  2. 根据 attention 重要性选择保留关键 KV，丢弃低重要性 KV。
  3. 在较小 KV cache budget 下进行长上下文推理。
- 对本项目的启发：
  - 可作为 attention-informed hotness sketch 的思想来源。
  - 能帮助估计哪些 KV range 更适合留在 HBM/CPU，哪些可以降到 SSD/3FS。
- 为什么没有完全覆盖我们的场景：
  - 它是近似驱逐方法，不保证 exact attention 语义。
  - 不处理请求前 cold-tier restore 和 admission。

### 7.3 Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- 来源：本轮检索为 query-aware sparsity / long-context inference 方向，需后续以正式论文入口核验。
- 核心问题：decode 时每个 query 不需要完整关注全部历史 KV，可以根据 query 选择更相关的 KV。
- 具体做法：
  1. 根据当前 query 或 attention 相关信号选择候选 KV block/token。
  2. 对未被选中的历史 KV 跳过或降级处理，降低注意力计算和 KV 读取成本。
  3. 重点优化 long-context decode 的计算和内存访问。
- 对本项目的启发：
  - 它代表“query-aware critical KV selection”路线，与 KVDrive/InfiniGen 有共同点。
- 为什么没有完全覆盖我们的场景：
  - 它偏执行期稀疏 attention，不是 exact committed historical KV 的持久化复用。
  - 如果使用近似选择，论文问题定义会从 exact serving 变成 approximate serving。

### 7.4 ShadowKV

- 来源：本轮检索为 long-context KV cache offloading 方向，需后续深读最终论文版本。
- 核心问题：长上下文 decode 时无法把全部 KV 放入 HBM，需要把一部分 KV 放到 CPU/外部内存并按需取回。
- 具体做法：
  1. 将大部分历史 KV 放在 CPU 或外部存储，GPU 只保留当前需要的 working set。
  2. 利用预测/预取减少 decode 时等待。
  3. 通过异步搬运和缓存策略降低 HBM 压力。
- 对本项目的启发：
  - 可作为 decode-time offload baseline。
- 为什么没有完全覆盖我们的场景：
  - 重点是在线 decode working set；不定义多轮 persistent KV 对象、checksum、manifest、admission deadline。

### 7.5 MagicPIG

- 来源：本轮检索为 KV cache offloading / prefetch 方向，需后续深读最终论文版本。
- 核心问题：长上下文 KV offloading 的 I/O 与 GPU 计算重叠不足，会导致 decode stall。
- 具体做法：
  1. 设计 KV block 的预取和替换策略，使 GPU 在需要 KV 前尽量完成搬运。
  2. 通过预测访问顺序、流水化 I/O 和可能的多级缓存降低等待。
  3. 面向长上下文或 memory-constrained serving 优化。
- 对本项目的启发：
  - 可为 P1 的 prefetch lead time、bandit 调参和流水化恢复提供参考。
- 为什么没有完全覆盖我们的场景：
  - 与 ShadowKV 类似，它偏执行期 offload；我们的核心是请求前 PrePass 和 SLA admission。

### 7.6 KVServe

- 来源：本项目讨论中作为 KV 压缩/服务层相关工作出现，需后续核验具体论文版本。
- 核心问题：KV cache 体积与传输成本限制 LLM serving throughput。
- 具体做法：
  1. 将 KV cache 作为可服务、可传输、可压缩的对象管理。
  2. 通过压缩、量化或服务层调度降低 KV 资源占用。
  3. 目标是提升 high-throughput serving 下的 KV 利用率。
- 对本项目的启发：
  - 适合作为压缩/服务层 related work，而不是第一阶段 exact path baseline。
- 为什么没有完全覆盖我们的场景：
  - 当前未确认其是否覆盖 persistent cold-tier admission；在论文中引用前必须进一步核验。

### 7.7 vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention

- 来源：https://arxiv.org/abs/2405.04437
- 核心问题：PagedAttention 需要专门的 attention kernel 和 block 管理；是否可以利用 OS virtual memory 获得更简单、可移植的 KV 动态内存管理。
- 具体做法：
  1. 使用虚拟内存管理 KV cache，把动态增长的 KV 交给 OS page table / memory mapping 机制处理。
  2. 避免对每个 attention backend 都实现 paged kernel。
  3. 尝试在保持性能的同时提高 portability。
- 对本项目的启发：
  - 提醒我们数据结构选择会影响 connector 和 attention backend 的侵入性。
  - 但 OS paging 不能直接暴露 SLA-aware cold miss 控制。
- 为什么没有完全覆盖我们的场景：
  - 它仍是 execution memory management，不是 SSD/3FS cold-tier persistent KV recovery。
  - 如果 cold miss 交给 OS 或透明机制，正好违背我们“不让 cold miss 进入 critical path”的目标。

## 8. 我们项目需要如何组织 Related Work

论文/PPT 中不建议把所有工作平铺。更合理的组织方式是：

1. **GPU 内 KV 管理与 prefix reuse**：PagedAttention、vLLM APC、SGLang/RadixAttention。说明它们解决 hot KV reuse，但没有 persistent cold-tier recovery。
2. **Prefill/decode disaggregation 与 KV transfer**：DistServe、Splitwise、Mooncake、NIXL。说明它们解决阶段干扰和跨实例传输，但不是 anti-caching lifecycle。
3. **外部 KV cache 与 RAG KV 复用**：LMCache、RAGCache、CacheBlend、CacheGen。说明它们证明复用价值，但没有完全覆盖多轮 committed historical KV 的 exact restore/admission。
4. **SSD/multi-tier KV restore**：Tutti、DualPath、CacheFlow、KVDrive、InfiniGen、Infinite-LLM。说明这是真正强相关竞争方向；我们的差异必须是 Persistent KV Anti-Caching semantics。
5. **近似/压缩/剪枝**：H2O、Scissorhands、Quest、ShadowKV、MagicPIG、KVServe。说明它们可作为后续优化或 fallback，但第一版不以 approximate attention 作为核心。
6. **数据库 Anti-Caching**：DeBrabant et al.。说明我们的思想来源：memory-primary、cold object、in-memory index、pre-pass、non-blocking fetch、partial merge。

## 9. 对我们方法设计的直接约束

1. 不能再声称“首次提出多级 KV 管理”。KVDrive、Mooncake、LMCache、Tutti、DualPath 已经覆盖了大量空间。
2. 必须把研究问题收窄到 **Persistent KV Anti-Caching for Multi-Turn Long-Context Serving**。
3. 必须证明当前方法不只是 data plane 更快，而是多了以下系统语义：
   - committed historical KV object；
   - strict correctness key；
   - GPU/CPU/SSD/3FS residency table；
   - KVPrePass；
   - packed cold object；
   - partial promote；
   - restore-inclusive admission；
   - prefill/decode two-phase ready barrier；
   - no synchronous cold miss in admitted execution。
4. baseline 至少要覆盖：
   - vLLM full prefill；
   - vLLM APC hot/cold/restart；
   - external DRAM-ready KV reuse；
   - naive cold restore/offload；
   - LMCache；
   - DualPath/Tutti/CacheFlow-style restore；
   - KVDrive-style multi-tier orchestration。

## 10. 后续需要继续核验的条目

以下工作本轮已经进入视野，但在正式论文引用前需要进一步做 source verification：

- LMCache 的具体论文版本、arXiv 元数据与最新系统接口。
- InfiniGen、Quest、ShadowKV、MagicPIG、KVServe 的正式题名、版本、实验设置和可复现代码。
- CacheFlow、DualPath、Tutti、KVDrive 这些 2026 近期工作的最终会议版本、代码状态与 artifact 可用性。
- Mooncake / LMCache / vLLM connector 生态在 2026 年的接口变化。
