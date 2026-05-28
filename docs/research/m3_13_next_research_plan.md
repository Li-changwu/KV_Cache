# M3.13 下一步研究方向优化：PrePass Lead-Time 与 Packed Restore Gate

## 研究方向结论

当前项目不应该继续把“能否从 cold tier 恢复 KV”作为主问题，因为 M3.12 已经证明真实 vLLM + true cold object 路径可以完成：

`store -> demote true cold object -> restart vLLM -> /prepass -> advance restore -> reuse`

下一步的核心研究问题应优化为：

> 对多轮长上下文服务而言，Persistent KV Anti-Caching 是否能在请求真正进入 Prefill/Decode 之前，用足够短且可预测的提前量完成 cold historical KV 的 restore、verification、admission，使在线 TTFT 只承担 DRAM/HBM-ready KV load 和 delta prefill？

换句话说，M3.13 之后要证明的不是“PrePass 是一个接口”，而是：

- 需要多少 lead time 才能隐藏 restore？
- restore tail 是否随 prefix 长度线性增长，还是被 per-layer 文件布局放大？
- 什么时候必须切换到 packed cold object / extent layout？
- 哪些请求应该被 `ADMIT`，哪些应该 `DELAY` 或 fallback，而不是让 cold miss 泄漏进在线 TTFT？

## Level2 诊断

当前实验内环是：

`选择 prefix/layout/lead-time -> 运行 vLLM -> 观察 TTFT/restore hiding -> 决定继续扩展或回退`

原来的瓶颈是实验选择偏手工，容易直接跳到 8K/16K/32K 真机矩阵，但这样会把三个因素混在一起：

1. PrePass 是否足够早；
2. cold restore executor 是否足够快；
3. per-layer safetensors 布局是否造成 tail latency。

因此本轮采用的 Level2 机制是 **space-filling + risk-gated PrePass lead-time planner**：

- 先用 2K 真实 restore 样本估算 KV bytes/token 和 restore throughput；
- 对 512/2K/8K/16K/32K 做 lead-time 风险外推；
- 输出 `residual_online_wait_ms`，显式说明如果请求来了才 restore，会泄漏多少等待；
- 输出 `packed_object_required`，把布局优化变成准入闸门，而不是事后解释。

## 已实现产物

新增 planner 与 CLI：

- `benchmarks/m3/prepass_lead_time_planner.py`
- `benchmarks/m3/plan_prepass_lead_time_matrix.py`
- `tests/m3/test_prepass_lead_time_planner.py`

用真实 M3.12/M3.11 数据生成的结果：

- `results/m3_13_prepass_lead_time_plan/prepass_lead_time_plan.csv`
- `results/m3_13_prepass_lead_time_plan/prepass_lead_time_report.md`
- `results/m3_13_prepass_lead_time_plan/restore_profile.json`

输入数据：

- PrePass true cold sample: `results/m3_12_2k_prepass_true/b5_prepass_true/reuse_phase/reuse_smoke_matrix.csv`
- B0/B1/B2/B3/B5 baseline: `results/m3_11_2k_load_only_rerun/final/baseline_matrix.csv`

## 关键数字

当前 2K 真实样本给出的 restore profile：

- Source prefix: `2048 tokens`
- KV bytes/token: `196608`
- True cold restore bytes: `402657408`
- True cold restore executor elapsed: `839.601 ms`
- Effective restore throughput: `457.365 MiB/s`
- PrePass control overhead: `11.070 ms`
- PrePass online TTFT: `271.782 ms`
- B0 full prefill TTFT: `717.061 ms`

在保守 `tail_multiplier=1.2`、`safety_margin=250ms` 下，planner 估算：

| Prefix | Estimated KV bytes | Estimated restore | Required lead | 判断 |
|---:|---:|---:|---:|---|
| 512 | 100663296 | 251.878 ms | 512.948 ms | 1s lead 足够，校准模型 |
| 2048 | 402653184 | 1007.511 ms | 1268.581 ms | 5s lead 足够，重复验证 2K |
| 8192 | 1610612736 | 4030.043 ms | 4291.113 ms | 5s lead 勉强足够，但必须关注 packed object |
| 16384 | 3221225472 | 8060.085 ms | 8321.155 ms | 12s lead 才足够，布局风险高 |
| 32768 | 6442450944 | 16120.170 ms | 16381.240 ms | 12s lead 仍不够，不能盲跑性能胜出结论 |

这说明：PrePass 的价值成立，但它依赖可用提前量和 restore tail。若没有提前量，B5 的 restore-inclusive TTFT 仍会显著超过 B0；若有提前量，online TTFT 可以接近 DRAM-ready reuse。

## 下一步详细规划

### M3.13-A：Lead-Time Planner 固化与回归

状态：已完成第一版。

验收标准：

- 能从真实 M3.12 PrePass CSV 和 M3.11 baseline CSV 生成 lead-time plan；
- 输出 `hide_restore`、`residual_online_wait_ms`、`packed_object_required`；
- 512/2K/8K/16K/32K 排序为校准、源重复、第一次扩展、布局闸门、上下文边界；
- 有单元测试和 CLI 测试。

### M3.13-B：512/2K/8K True PrePass 小矩阵

优先级：P0。

目标：

- 先跑 512、2K、8K，而不是直接跑 16K/32K；
- 每个 prefix 至少记录 `no prepass/reaction` 与 `prepass+advance` 两个口径；
- 对每一行都报告 online TTFT、restore-inclusive TTFT、restore executor、connector load、store/load events、sync cold miss。

推荐执行顺序：

1. 512 tokens：验证 downscale calibration，检查 restore 模型是否过于保守；
2. 2K tokens：重复 M3.12-B，估算方差；
3. 8K tokens：第一次 scaling probe，同时验证 per-layer layout 是否开始放大 tail。

进入 16K 的条件：

- 8K true restore 没有出现非线性 tail 爆炸；
- 8K PrePass 在 5s lead 下稳定 `hide_restore=true`；
- `store_events=0`、`load_events=1`、`sync_ssd_miss_total=0` 仍保持；
- connector load 不随 prefix 出现异常级增长。

### M3.13-C：Packed Cold Object / Extent Layout 原型

优先级：P0，应该在 16K/32K 性能主张之前完成。

原因：

- 当前 cold object 虽然已经是真实目录级迁移，但 hot tensor store 仍是 per-layer safetensors；
- 8K 后恢复数据量达到约 1.5GiB，16K 达到约 3GiB，32K 达到约 6GiB；
- 如果仍按每层文件和 Python 层恢复，tail latency 会成为主要变量，无法公平评价 Anti-Caching 策略本身。

最小原型目标：

- 一个 prefix 对应一个 packed cold object 或少量 extent；
- manifest 记录每层/每 block 的 offset、length、checksum；
- restore 支持 sequential read 到 hot tensor store；
- 保留 correctness key、checksum、ready barrier 语义；
- 输出 packed vs per-layer 的 restore p50/p95/p99、effective MiB/s、CPU time。

### M3.13-D：16K/32K Gate Matrix

优先级：P0，但必须在 M3.13-B/C 之后。

目标：

- 只在 packed layout 或 8K 结果足够稳定后跑；
- 对 16K/32K 分别测 `available_lead_ms=5000/12000/20000` 的风险边界；
- 把 32768 作为 Qwen2.5-14B native context boundary，不把它过度外推为 1M 结论。

验收标准：

- 能明确说出 16K/32K 下请求需要提前多久 PrePass；
- 能解释如果没有足够提前量，系统应该 DELAY/fallback，而不是同步 cold miss；
- 能比较 per-layer 与 packed layout 的 tail。

### M3.13-E：研究叙事与 baseline 加强

优先级：P1，跟随 P0 证据。

需要补齐：

- B4 naive cold restore/offload baseline；
- LMCache / KVDrive-style / DualPath-style 对照或仿真映射；
- shared system prompt、hot document、multi-turn append、low locality negative control 四类 workload；
- paper 中必须区分 online TTFT、restore-inclusive TTFT、lead-time required、deadline miss rate。

## 当前必须避免的误判

- 不能说“B5 已经端到端满足 SLA”。当前只能说：如果 restore 被 PrePass 提前隐藏，2K online TTFT 已经优于 B0。
- 不能把 0ms lead 下的 B5 restore-inclusive 当作系统最终失败。Anti-Caching 的设计点就是不要让 cold restore 进入在线请求路径。
- 不能直接从 2K 线性外推到 32K 后声称有效。planner 只是排序工具，8K/16K/32K 必须用真实路径验证。
- 不能在 per-layer layout 上跑出很差 tail 后否定方法本身。那可能是布局实现问题，而不是 Anti-Caching 调度问题。

## 下一次最自然操作

1. 用 planner 结果驱动 512/2K/8K true PrePass 小矩阵；
2. 同时设计 packed cold object manifest 和 restore executor；
3. 先比较 8K per-layer restore 是否已经触发非线性 tail；
4. 决定是否先改 packed layout，再跑 16K/32K。
