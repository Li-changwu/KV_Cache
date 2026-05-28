# 1M Tokens KV Cache 分层管理实施规划

日期：2026-05-24
输入方案：`1M_Tokens_KV_Cache_分层管理技术方案.md` v0.5
规划状态：草案，供项目讨论和下一步任务拆解使用

## 1. 我对项目研究问题的理解

这个项目要解决的不是“把 KV Cache 放到 SSD 上”这个单点工程问题，而是：在精确 dense attention 语义不变的前提下，让 `1M tokens` 级别长上下文请求成为可预测、可准入、可观测、可降级的在线服务能力。

核心矛盾有三层：

1. `1M tokens` 的 KV 容量远超单卡 HBM，必须引入 DRAM 和 3FS/SSD 作为多级容量层。
2. 精确 decode 仍可能访问历史 KV，因此 3FS/SSD 不能进入 SLA decode 同步读路径。
3. 多轮 Agent / 长会话 / 共享前缀场景中，真正应优化的成本不是每轮重新 prefill 1M 历史 token，而是复用已提交历史 KV，只对新增 token 做 delta prefill。

所以第一版系统必须同时完成两件事：

- **Prefill 重算规避**：通过 `KVManifest`、`PrefixIndex`、`KVCorrectnessKey` 证明历史 KV 可复用，生成 `PrefillReusePlan`，只计算新增 token。
- **Decode miss 规避**：通过 pre-pass、prefetch 和 ready barrier，保证被准入请求的 required KV 已在 HBM/DRAM 可达集合中，SLA 队列内 `sync_3fs_miss_total == 0`。

项目的可承诺场景应限定在有复用或局部性的 workload：同一 session 多轮追加、共享长文档/system prompt、Agent trace/tool result 连续追加、短时间恢复的 idle session。低局部性随机 1M 请求应进入宽松队列、full prefill 路径、延迟/迁移，或被拒绝精确 SLA。

## 2. 整体架构理解

推荐把系统拆成五个边界清晰的层：

| 层 | 主要职责 | 第一版边界 |
| --- | --- | --- |
| 服务入口与准入层 | 接收请求，计算 correctness key、reuse plan、TTFT/hit-rate/bandwidth 预测，决定准入/延迟/迁移/拒绝 | 先在 vLLM 外部 gateway 或 sidecar 实现 |
| KV 控制平面 | 维护 `KVManifest`、`PrefixIndex`、`KVIndex`、`KVSharingPlan`、`PrefetchPlan`、`SchedulerDecision` | 独立进程或 scheduler 前置模块 |
| 多级存储与连接器 | 管理 HBM / pinned DRAM / 3FS 的 residency 转换、异步 prefetch、store、release | 通过 `Tiered3FSConnector` 接入 vLLM |
| vLLM 执行层 | 复用 PagedAttention、APC、KV connector、disaggregated prefill 等现有能力 | 不改 attention kernel，不改 GPU block allocator 核心逻辑 |
| 监控与 QoS 层 | 采集 TTFT、prefill reuse、effective hit rate、3FS tail latency、RDMA/带宽、短请求干扰，驱动 admission 收紧 | 先做指标闭环，再做复杂优化 |

关键数据流如下：

```text
Gateway
  -> SLA Admission Guard
  -> KVCorrectnessKey / PrefixIndex / KVManifest lookup
  -> KVSharingPlan across admission window
  -> PrefillReusePlan
  -> KVIndex residency lookup
  -> PrefetchPlan / 3FS restore
  -> delta prefill
  -> decode ready barrier
  -> vLLM decode scheduler
  -> async writeback / eviction / metrics
```

这个架构里，3FS/SSD 是容量层，不是透明显存扩展；HBM/DRAM 是执行可达层；调度器负责在请求进入 vLLM 执行前确认 required KV 是否 ready。

## 3. 推荐实施路线

我建议采用“仿真先行 + vLLM 低侵入原型并行收敛”的路线。

| 路线 | 优点 | 风险 | 结论 |
| --- | --- | --- | --- |
| 直接改 vLLM | 最快触达真实系统 | 早期容易陷入 connector/API/版本细节，难以判断策略本身是否有效 | 不作为第一步 |
| 先做纯仿真 | 快速验证 hit rate、prefetch、admission、sharing planner 策略 | 不能证明 vLLM 集成可行性 | 必须做，但不能单独做 |
| 仿真 + 最小 connector 双轨 | 策略收益和工程可行性同时验证，风险可拆开 | 需要清楚定义接口和阶段门槛 | 推荐 |

第一版目标不是一口气做完整生产系统，而是完成一个最小可信闭环：

1. 能用 trace/simulator 证明 `PrefillReusePlan + KVSharingPlan + PrefetchPlan + Admission` 相比基线有收益。
2. 能在 vLLM 最小原型里证明同一 1M session 第二轮请求不 full prefill，只 delta prefill 新增 token。
3. 能在 decode 前执行 required blocks ready 检查，SLA 队列中不发生同步 3FS miss。
4. 能输出支撑论文/报告的核心指标：TTFT ratio、prefill reuse rate、effective hit rate、sync miss、prefetch deadline miss、3FS/RDMA 带宽、短请求 P99 干扰。

## 4. 阶段规划

### M0：项目上下文与边界固化

目标：把研究边界、非目标、术语和后续 agent 工作上下文固定下来。

交付物：

- `AGENTS.md`：项目规则文件，说明精确 attention、禁止 decode 同步读 SSD、优先 vLLM 低侵入、评估口径等。
- `docs/context/project_map.md`：模块地图，说明 control plane、connector、simulator、benchmark、metrics 的职责。
- `docs/context/open_questions.md`：记录需要人工确认的问题。

验收标准：

- 后续新会话能仅凭方案文档、`AGENTS.md` 和规划文件恢复项目上下文。
- 所有实现讨论都能明确落到“prefill 复用”“decode ready”“admission/QoS”“vLLM connector”中的某一类。

### M1：指标、schema 与基线口径固化

目标：先固定评估口径，避免系统做出来后无法判断是否成功。

任务：

- 定义 `TTFT_full_prefill_baseline` 与 `TTFT_delta_prefill_baseline` 的测量方法。
- 固化 `KVCorrectnessKey`、`KVBlockId`、`KVManifest`、`KVIndexEntry`、`PrefillReusePlan`、`KVSharingPlan`、`PrefetchPlan`、`SchedulerDecision` 的 schema。
- 明确 workload 分类：`critical_1m_exact`、`normal_long`、`short_latency`、`background_recovery`。
- 明确对比基线：DRAM 扩容基线、naive SSD offload、independent per-request cache、DualPath only、APC/session reuse only、本方案。

交付物：

- `docs/specs/metrics_and_sla.md`
- `docs/specs/kv_metadata_schema.md`
- `docs/specs/workload_and_baselines.md`

验收标准：

- 所有核心指标都有公式、采集点和失败解释。
- `TTFT_offload <= 1.2 * TTFT_DRAM_baseline` 的 baseline 类型不再混淆 full prefill 和 delta prefill。
- `effective_hit_rate` 不把“理论上在 3FS 可读”的 block 误算为命中。

### M2：离线仿真与策略验证

目标：在不依赖真实 vLLM 改造的情况下，先验证调度策略是否能达成命中率、带宽和准入目标。

建议实现模块：

| 模块 | 职责 |
| --- | --- |
| `trace_loader` | 读取或生成 Single-1M、Multi-turn Agent、Shared Prefix、Shared Subrange、Mixed Serving、Low Locality trace |
| `kv_block_model` | 根据模型 KV layout、block size、context length 估算 block 和字节规模 |
| `residency_simulator` | 模拟 HBM/DRAM/3FS residency、waterline、pin/refcount、eviction |
| `reuse_planner` | 生成 session/prefix 的 `PrefillReusePlan` |
| `sharing_planner` | 支持 `off / merge_only / merge+decomposition / window_aware` |
| `prefetch_simulator` | 根据 3FS 带宽、queue depth、tail latency 和 deadline 估算 ready blocks |
| `admission_simulator` | 预测 TTFT、effective hit rate、bandwidth utilization，输出准入/延迟/拒绝 |

实验矩阵：

- block size：`32 / 64 / 128 / 256 tokens`
- sharing planner：`off / merge_only / merge+decomposition / window_aware`
- prefetch window：`critical only / +1 turn / +2 turns`
- QoS：`off / static cap / adaptive cap`
- eviction：`LRU / aLRU / predicted_next_use`

交付物：

- `sim/` 离线仿真器。
- `traces/` 小规模可复现实验 trace。
- `results/m2_simulation_report.md`：包含命中率、prefill tokens saved、3FS read amplification、deadline miss、admission reject/delay。

验收标准：

- Multi-turn Agent 与 Shared Prefix 场景中，预测 `effective_hit_rate >= 90%`，且 `synchronous_3fs_miss_predicted == 0`。
- Low Locality 场景被 admission 明确识别为非适用，而不是污染 90% hit-rate 目标。
- `KVSharingPlan` 相比 independent per-request cache 至少在 shared prefix 或 common subrange workload 上展示收益。

### M3：vLLM 最小原型接入

目标：证明方案能接入真实推理服务路径，同时保持低侵入。

实现范围：

- 外部 `Tiered KV Control Plane`：维护 session manifest、prefix index、in-memory KVIndex。
- `SLA Admission Guard`：在请求进入 vLLM 前做 correctness/reuse/residency/admission 判断。
- `Tiered3FSConnector` 最小接口：`lookup_prefix`、`plan_delta_prefill`、`lookup`、`prefetch`、`store`、`load_for_decode`、`release`。
- 本地文件系统或 3FS 客户端封装为 secondary tier；第一版只要求批量异步读写、checksum、manifest commit、指标。
- vLLM 侧复用已有 PagedAttention/APC/KV connector 能力，不修改 attention kernel。

最小状态机：

```text
RECEIVED
  -> ADMISSION_CHECK
  -> PREFETCHING
  -> READY_FOR_VLLM
  -> DELTA_PREFILL
  -> DECODE_READY_BARRIER
  -> DECODING
  -> COMMIT_KV
  -> IDLE_OR_CLOSED
```

交付物：

- `control_plane/` 最小控制平面。
- `connector/tiered_3fs_connector.py` 或等价 vLLM connector 实现。
- `examples/multi_turn_1m_session/` 端到端示例。
- `results/m3_vllm_mvp_report.md`。

验收标准：

- 同一 session 第二轮请求能复用 committed ranges，只对新增 token 做 delta prefill。
- correctness key 不匹配时，系统明确 full prefill fallback 或拒绝精确复用路径。
- `load_for_decode()` 不同步读 3FS；未 ready block 返回 `NOT_READY`，由调度层暂停或重排。
- `sync_3fs_miss_total == 0`，并且该指标可观测。

### M4：分离式 prefill/decode 与 DualPath

目标：从单实例 MVP 进入更接近生产的分离式推理架构。

任务：

- 接入 vLLM disaggregated prefill。
- 区分 `storage_to_prefill` 与 `storage_to_decode` 路径。
- 将 3FS read/write utilization、queue depth、tail latency、RDMA KV transfer utilization 纳入调度输入。
- 加入 DualPath 选择策略：prefill engine storage NIC 空闲时走 prefill；prefill 存储侧拥塞且 decode 侧有余量时切换路径；两者均拥塞时延迟 admission。
- 对 KV transfer 与 tensor/pipeline parallel 通信做优先级保护。

交付物：

- `docs/specs/dualpath_scheduling.md`
- `results/m4_dualpath_report.md`

验收标准：

- `DualPath only` 与本方案在同一 workload 上有可复现对比。
- 3FS/RDMA 超过 70% sustained utilization 时，admission 能自动收紧。
- Mixed Serving 中短请求 P99 TTFT 不被 long_exact_prefetching 持续拖高。

### M5：在线 SLA 闭环与生产化门槛

目标：把策略从“能跑”提升到“能治理、能告警、能回滚”。

任务：

- 接入完整 metrics exporter。
- 建立 dashboard：TTFT ratio、prefill reuse、session/prefix hit、sharing candidate hit、effective hit、sync miss、deadline miss、3FS/RDMA、HBM/DRAM residency、admission decision。
- 实现在线反馈：hit rate 连续低于 90% 时暂停新 critical 1M admission，扩大 P0/P1 prefetch，降低长请求并发，迁出低局部性请求。
- 补齐故障恢复：manifest commit 失败、checksum 失败、部分 block 缺失、3FS jitter、connector crash。
- 输出生产集成报告。

交付物：

- `docs/ops/metrics_and_alerts.md`
- `docs/ops/failure_recovery.md`
- `results/m5_online_sla_report.md`

验收标准：

- 覆盖 Single-1M、Multi-turn Agent、Shared Prefix、Shared Subrange、Mixed Serving、Bandwidth Stress、3FS jitter、Low Locality。
- 对多轮长会话，committed ranges 默认不 full prefill 重算。
- SLA 队列 `sync_3fs_miss_total == 0`。
- 带宽超限时 admission 自动收紧，并保护短请求。

## 5. 第一轮建议排期

如果以研究原型为目标，我建议第一轮只做到 M1-M3：

| 周期 | 重点 | 输出 |
| --- | --- | --- |
| 第 1 周 | M1 指标/schema/trace 口径 | 指标文档、schema 文档、workload 定义 |
| 第 2-3 周 | M2 仿真器和策略消融 | simulation report，明确 block size、waterline、sharing planner 初始参数 |
| 第 4-6 周 | M3 vLLM 最小原型 | connector/control plane MVP，多轮 session delta prefill demo |

M4/M5 不建议在 M2 结果出来前提前深做。DualPath 和生产 QoS 需要真实瓶颈数据，否则容易把工程复杂度加到错误位置。

## 6. 近期我建议马上做的 5 件事

1. **补齐项目上下文文件**：创建 `AGENTS.md` 和 `docs/context/project_map.md`，让后续 agent 或工程同学不会偏离“精确语义 + 低侵入 vLLM + 禁止 decode 同步读 3FS”的主线。
2. **先写 schema，不先写 connector**：把 correctness key、manifest、index、reuse plan、sharing plan、prefetch plan 的字段固定到可版本化格式。
3. **先做 trace 和仿真**：用 6 类 workload 建立策略可验证环境，优先证明 90% hit rate 的适用条件和拒绝条件。
4. **vLLM 只做最小闭环**：第一版只证明 session 级 delta prefill、3FS/DRAM restore、ready barrier 和指标，不追求完整 DualPath。
5. **把低局部性作为负样本**：明确系统会拒绝或降级这类请求，避免评估时把不可服务化场景混进目标承诺。

## 7. 待你确认的问题

这些问题会影响实施优先级：

1. 第一阶段是否以 vLLM 为唯一真实系统目标，还是同时保留 SGLang/LMCache 兼容性探索？
2. 可用硬件是什么：GPU 型号/数量、CPU DRAM、SSD 或 3FS 集群、RDMA/NIC 拓扑？
3. 目标模型是谁：Dense Transformer、GQA/MQA/MLA、KV dtype 是否包含 FP8？
4. `1M tokens` 主要是单 session 多轮追加，还是多用户共享长文档前缀？
5. 论文/项目创新更偏系统服务化，还是更偏调度算法与共享规划？
6. M3 原型是否允许先用本地 NVMe/文件系统模拟 3FS，再替换为真实 3FS 客户端？

## 8. 风险与规避

| 风险 | 表现 | 规避 |
| --- | --- | --- |
| 把 3FS 当透明显存扩展 | decode tail latency 被 SSD I/O 放大 | 强制 `load_for_decode()` 不同步读 3FS，ready barrier 前置 |
| baseline 口径混乱 | 1.2x TTFT 无法解释 | 分开 full prefill baseline 与 delta prefill baseline |
| correctness key 不稳定 | full prefill fallback 飙升 | 固定 tokenizer/model/template/adapter/attention mask digest |
| 命中率承诺过宽 | Low Locality 拉低结果 | admission 先筛 workload，非适用场景单独报告 |
| 过早深改 vLLM | 工程成本吞掉研究验证 | M3 前不改 attention kernel，不改核心 block allocator |
| 共享规划过复杂 | optimizer 自身增加 TTFT | 第一版固定 time budget，先 `merge_only`，再扩展 decomposition |
| 3FS tail latency 抖动 | prefetch deadline miss | 使用实时 queue depth/tail latency 估算，触发 admission 收紧 |

## 9. 成功标准

第一轮项目成功，不是“所有 1M 请求都快”，而是满足以下判断：

- 对适用 workload，系统能预测哪些请求可进 `critical_1m_exact`，并把不可满足 SLA 的请求挡在执行前。
- 多轮长会话的后续轮次主要成本是 KV restore、delta prefill 和 first decode，而不是 full prefill 1M 历史 token。
- decode 阶段不发生同步 3FS 读取；意外 miss 被观测、暂停、重排，并触发 admission 收紧。
- `KVSharingPlan` 至少在共享前缀或公共子段 workload 中展示超过 per-request cache 的收益。
- 短请求不会被长上下文 prefetch/writeback 持续拖高尾延迟。

## 10. 参考资料

- 本地方案：`/root/KV/1M_Tokens_KV_Cache_分层管理技术方案.md`
- 3FS 官方仓库：https://github.com/deepseek-ai/3fs
- vLLM disaggregated prefill：https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM prefix caching：https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM NIXL connector：https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
- DualPath arXiv：https://arxiv.org/abs/2602.21548
