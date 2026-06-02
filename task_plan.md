# 任务计划：1M Tokens KV Cache 分层管理项目理解与实施规划

## 目标
充分理解 `1M_Tokens_KV_Cache_分层管理技术方案.md` 的研究问题、系统架构、关键机制与实现约束，并产出可沟通、可迭代的实施规划。

## 当前阶段
阶段 M3.15 正在推进：项目主线已重新收敛为 **Persistent KV Anti-Caching for Multi-Turn Long-Context Serving**。M3.12 已经证明真实 vLLM + true cold object + PrePass 可以把 cold restore 移出在线请求路径，M3.13 已经量化了不同 prefix 长度所需的 restore lead time，M3.14 已完成三级 KV 状态索引与 `packed_v1` 冷对象数据面优化。最新 8K packed PrePass gate 已证明真实 8192-token packed cold object 可以通过 `SSD_COLD -> restore -> online external load` 跑通，但 restore executor 仍约 3.05s，restore-inclusive 约 3.80s。当前 P0 要转向“8K/16K 可解释性能证据”：补同配置 8K baseline，压低 packed restore/load 数据面，并在进入 16K/32K 前明确 CPU_READY 到 GPU_READY 的调度边界。

## 当前优先级把关

| 优先级 | 任务 | 为什么必须这样排 |
|---|---|---|
| P0 | M3.11 Baseline Readiness Matrix：B0-B5 同机对比 | 不先证明相对 full prefill / APC / naive cold restore 有收益，继续扩系统没有科研判断依据 |
| P0 | 512/2K/8K/16K/32K 在线矩阵 | 当前 16/64/128/256 只能证明机制，不能证明长上下文服务有效 |
| P0 | Persistent session lineage + committed KV ranges | 框架核心是多轮历史 KV 作为持久服务状态；没有它就只是 prefix cache/offload |
| P0 | KV PrePass：枚举 required historical KV ranges | Anti-Caching 的根本是执行前知道 cold miss，而不是执行期发现 miss |
| P0 | M3.14-A KV Evicted Index：内存常驻 KV 状态索引 | 借鉴 Anti-Cache Evicted Table，请求进入 GPU 前必须先知道所需 KV 是 GPU_READY、CPU_READY、SSD_COLD、FETCHING、LOADING、MISSING 还是 MISMATCH |
| P0 | M3.13 PrePass lead-time planner 与 512/2K/8K 小矩阵 | B5 online TTFT 好看只在 restore 被提前隐藏时成立；必须量化需要多少提前量 |
| P0 | M3.14-B PrePass 分类集合 | PrePass 必须返回 gpu_ready/cpu_ready/ssd_cold/fetching/loading/missing/mismatch sets，而不是只返回计数和 QUEUED/READY |
| P0 | M3.14-B2 CPU->GPU load readiness 边界 | CPU_READY 不等于 GPU_READY；必须显式建模 H2D load 预算，否则会低估在线 TTFT/Decode ready 风险 |
| P0 | M3.14-C Packed Cold Object / extent layout | 对应 Anti-Cache Block Table；每层小文件无法支撑 32K/128K/1M，也无法与 KVDrive/3FS 做严肃对比 |
| P0 | M3.15 8K packed PrePass gate 与同配置 baseline | 8K 链路已跑通，但必须用同一 vLLM 配置对比 B0/B2，才能判断 online TTFT 和 restore-inclusive 的实际收益 |
| P0 | M3.15 packed restore/load 数据面继续优化 | 8K restore executor 约 3.05s、connector load 约 469ms，是进入 16K 前必须解释和压低的关键路径 |
| P1 | 生产级 restore executor：QD、pinned staging、H2D overlap、tail metrics | P0 证明有收益后，才值得优化真实数据面 |
| P1 | 真实 3FS mount / 生产 SSD 校准 | 重要，但应服务于已被 P0 证明的系统路径 |
| P1 | Partial promote + attention-informed hotness sketch | 强化 Anti-Caching 策略，吸收 KVDrive 启发，但不是第一证明点 |
| P1 | LMCache / DualPath / Tutti / CacheFlow / KVDrive-style 强 baseline | P0 先跑自有 B0-B5，随后补强系统 baseline |
| P2 | 128K/1M、多节点、租户隔离、配额、公平性、生产监控 | 论文/生产增强，在核心证据链成立后推进 |

## 各阶段

### 阶段 1：上下文恢复与资料盘点
- [x] 确认用户指定技能与工作流
- [x] 确认项目目录与已有文件
- [x] 阅读项目方案文档结构
- [x] 将初始发现记录到 `findings.md`
- **状态：** complete

### 阶段 2：研究问题与架构理解
- [x] 提炼核心研究问题与目标约束
- [x] 梳理整体架构、模块边界和数据流
- [x] 识别性能瓶颈、风险点和验证指标
- [x] 更新 `findings.md`
- **状态：** complete

### 阶段 3：实施路径设计
- [x] 拆分 MVP、实验验证、工程集成阶段
- [x] 定义每阶段输入、输出、验收标准
- [x] 标出需要进一步确认的问题
- **状态：** complete

### 阶段 4：Markdown 交付与沟通
- [x] 写入正式实施规划文档
- [x] 更新 `progress.md` 和计划状态
- [x] 向用户说明规划位置与核心建议
- **状态：** complete

### 阶段 5：Benchmark 与仿真校准讨论
- [x] 评估原方案与实施规划中的 benchmark / 仿真顺序
- [x] 检查本机 vLLM、GPU、DRAM、NVMe 基础环境
- [x] 形成 M1.5 Benchmark 与硬件校准规划
- [x] 与用户确认第一轮 benchmark 模型、上下文上限和 3FS/NVMe 路径
- [x] 实现 M1.5 benchmark/calibration harness、解析器、硬件微基准、simulator 参数与报告生成
- [x] 完成单元测试、dry-run、硬件轻量 smoke、vLLM runner integration smoke
- **状态：** complete

### 阶段 6：Benchmark 正式采样与仿真输入冻结
- [x] 启动正式 vLLM server，使用 `VLLM_PLUGINS=` 和 `NO_PROXY=127.0.0.1,localhost`
- [x] 跑 B0-B5 完整矩阵，记录 OOM/ERROR 行
- [x] 生成 `results/benchmark_calibration/{env.json,vllm_latency.csv,apc_prefix_reuse.csv,hardware_io.csv,h2d_bandwidth.csv,simulator_params.yaml,report.md}`
- [x] 基于真实结果冻结 M2 simulator 第一版参数
- **状态：** complete

### 阶段 7：M1.6 Qwen3-32B 真实基线与长上下文上探
- [x] 更新项目上下文，明确 `/root/models/Qwen3-32B` 是 M1.6 校准模型
- [x] 采集 Qwen3-32B 环境与模型配置到独立结果目录
- [x] 验证 Qwen3-32B 在 RTX A6000 48GB 上的加载可行性
- [x] 运行最小 random / prefix repetition 冒烟
- [x] 执行 Qwen3-32B 服务容量探测：2048 无卸载 OOM，24GB 主存卸载可到 8192 但 16384 因 KV 容量失败
- [x] 执行 Qwen3-32B 调参容量探测：32GB 主存卸载可启动 32768 与 40960，但最大并发仅约 1.35x / 1.08x
- [x] 由于主存卸载推理极慢，停止继续跑 Qwen3 长上下文性能矩阵；仅保留容量边界与极小请求作为本机事实
- [ ] 对 65536、131072 仅在启用 YaRN/RoPE scaling 后单独标记为扩展上下文实验
- [x] 生成 Qwen3-32B 专属 `simulator_params.yaml`，不覆盖 M1.5 gpt-oss 结果
- **状态：** complete

### 阶段 8：M2 仿真器第一版
- [x] 消费 Qwen3-32B 专属 `simulator_params.yaml` 与硬件校准数据
- [x] 将 Qwen3-32B 每 token KV 约 256KiB、24GB/32GB 卸载容量边界和启动成本纳入 simulator constraints
- [x] 建立 session/prefix-reuse/low-locality negative-control workload generator
- [x] 建立 HBM/DRAM/NVMe 三级 KV residency 与 prefetch/admission 离散事件模拟
- [x] 输出 TTFT、effective hit rate、deadline miss、storage utilization、admission reject/delay 等指标
- [x] 生成严格 250ms 与宽松 12000ms 两组对照结果
- **状态：** complete

### 阶段 9：M2 策略增强与敏感性分析
- [x] 扩展 simulator schema，显式建模预取提前量、并发 I/O、HBM/DRAM 驻留比例、复用预测误差
- [x] 做参数 sweep：deadline、DRAM 容量、H2D 带宽、NVMe/3FS 带宽、HBM resident tokens
- [x] 输出收益边界 CSV：在哪些 deadline / bandwidth / reuse rate 下进入 SLA
- [x] 为 M3 vLLM connector / sidecar 原型定义最小接口
- **状态：** complete

### 阶段 M1.7：Qwen2.5-14B 默认在线基座校准
- [x] 将 `/root/models/Qwen2.5-14B-Instruct` 设为后续默认在线实验模型
- [x] 修正环境采集：`use_sliding_window=false` 时不因 `sliding_window` 字段误判为滑动窗口模型
- [x] 修正 simulator 参数：按模型配置写出 `kv_bytes_per_token`
- [x] 修正 M2 仿真器：从 `simulator_params.yaml` 读取模型专属 KV/token，不再写死 Qwen3
- [x] 采集 Qwen2.5-14B 环境与模型配置
- [x] 无 CPU 卸载容量探测：2048、8192、16384、32768 均可启动
- [x] 运行 Qwen2.5-14B 极小 random / prefix repetition 冒烟
- [x] 生成 `results/qwen25_14b_calibration/simulator_params.yaml`
- [x] 用 Qwen2.5-14B 参数重跑 M2 sweep
- **状态：** complete

### 阶段 10：M3 vLLM connector / sidecar 最小闭环
- [x] 基于 `docs/specs/m3_connector_sidecar_minimal_interface.md` 固化 sidecar API 和 mock connector
- [x] 实现 correctness key / manifest / prefix lookup 的最小内存版本
- [x] 实现 admission decision：`ADMIT`、`DELAY`、`REJECT`、`FULL_PREFILL_FALLBACK`
- [x] 实现 ready barrier mock，证明 decode 前缺失 KV 会被阻止而不是同步读 SSD
- [x] 接入离线请求 replay，验证同一 session 第二轮只做 delta prefill 规划
- **状态：** complete

### 阶段 M3.5：真实 vLLM 前置代理 / connector 对接实验
- [x] 将 `benchmarks/m3/control_plane.py` 包装为 HTTP sidecar 或轻量前置代理
- [x] 对接 Qwen2.5-14B 在线 vLLM 服务，先只做 admission/replay 旁路记录，不拦截真实 KV
- [x] 记录真实请求的 correctness key、prefix candidates、decision 和 ready barrier 指标
- [x] 评估 vLLM KV connector 接入点，决定 mock connector 到真实 connector 的最小替换面
- **状态：** complete

### 阶段 M3.6：vLLM 自定义 KV connector 最小 smoke
- [x] 实现自定义 `KVConnectorBase_V1` no-op connector，先只验证 scheduler 外部 KV token 路径
- [x] sidecar 将 `m3_control` 决策翻译为 vLLM 原生 `kv_transfer_params`
- [x] 用 Qwen2.5-14B 短 prompt 验证 `get_num_new_matched_tokens()` / `build_connector_meta()` / worker metadata 流程
- [x] 将 no-op load 替换为真实 KV tensor copy 前，补齐 correctness key 与 block alignment 测试
- **状态：** complete

### 阶段 M3.7：真实 KV tensor copy 原型
- [x] 扩展 no-op connector，保存短 prompt 的 token ids、block ids、correctness key 和 KV tensor manifest
- [x] 在 worker 侧实现最小 layer-by-layer KV tensor extract/load，先只覆盖 Qwen2.5 + FlashAttention/NHD 路径
- [x] 设计并验证 block alignment、layout、load failure、correctness key mismatch 的失败测试
- [x] 做两轮短 prompt prefix reuse smoke，观察请求返回、sidecar 决策、manifest 和 load failure 状态
- **状态：** complete

### 阶段 M3.8：真实复用路径稳定化与小矩阵验证
- [x] 增加 connector 保存/加载的结构化日志和指标
- [x] 让 sidecar commit 与真实 connector manifest 自动联动，减少手动 commit
- [x] 实现 16/64/128/256 前缀复用小矩阵脚本和 dry-run 输出
- [x] 跑 16/64 前缀加 16 后缀在线小矩阵，记录请求耗时、manifest、自动提交和保存事件
- [x] 构造重启 vLLM 对照，绕开内置前缀缓存并观测到 connector `load_request`
- [x] 跑完整 16/64/128/256 前缀复用在线小矩阵，记录 TTFT、错误、manifest 大小和保存/加载耗时
- [x] 增加输出一致性 smoke：同一提示词全量 prefill 与复用路径返回不应出现明显异常
- [x] 根据小矩阵结果决定是否进入 DRAM/NVMe 分层迁移原型
- **状态：** complete

### 阶段 M3.9：DRAM/NVMe 分层迁移原型
- [x] 为 `KVBlockManifest` 增加 `tier` 与 `ready` 层级状态
- [x] 实现 `demote_to_nvme()` 与 `prefetch_to_dram()` 元数据迁移
- [x] 禁止 connector 在未预取时从 `NVME` 前缀同步加载
- [x] 记录每个前缀的 `migration_events.jsonl`
- [x] 新增离线迁移冒烟脚本和报告
- [x] 将 sidecar 的 `DELAY / prefetch` 决策自动绑定到 `prefetch_to_dram()`
- [x] 增加带宽和 deadline 感知的迁移队列
- [x] 增加多请求排队压力烟测
- [x] 将队列指标接入 `/metrics`
- [x] 区分已驻留 DRAM 命中与需要 NVMe 新预取的队列入口
- [x] 增加后台异步队列
- [x] 将真实在线小矩阵接入 residency / prefetch 指标
- **状态：** complete

### 阶段 M3.10：真实冷层与后台迁移原型
- [x] 将当前 `NVME` 元数据状态替换为真实 cold-tier 目录/设备级迁移
- [x] 实现确定性后台迁移执行器入口，异步队列推进时执行真实 restore
- [x] 将异步队列完成事件与 sidecar 控制面、tensor store manifest 闭环
- [x] 增加真实文件搬运耗时、真实字节数和 checksum 状态记录
- [x] 将真实 cold-tier 迁移接入在线小矩阵并生成生产路径样本
- [x] 扩展 16/64/128/256 完整 cold-tier restart 小矩阵
- [x] 准备后续 3FS 接入适配层：完成 `local_posix` / `3fs_posix` adapter 选择入口
- [x] 增加 cold-tier adapter benchmark，输出 local/3fs_posix demote/restore p50/p95/p99 与带宽对比
- [x] 扩展 adapter benchmark 支持 `queue_depth`、并发 restore batch 统计和 `qwen25_14b_tiny` profile
- [ ] 将 3FS adapter 从 POSIX mount 原型推进到真实 3FS 环境校准与生产级 executor
- **状态：** in_progress

### 阶段 M3.11：Baseline Readiness Matrix 与有效性验证
- [x] 复盘当前实现是否真的有效，区分语义路径、容量路径、性能收益和强 baseline 胜出四类证据
- [x] 形成当前不能声称性能胜出的风险判断，并写入研究文档
- [x] 将 KVDrive 提升为主竞争 baseline，明确本项目差异不是泛化多级 KV 管理，而是 exact persistent historical KV serving
- [x] 将后续任务划分为 P0/P1/P2，并明确 P0 必须优先于继续 3FS adapter 功能扩展
- [x] P0：定义统一 benchmark schema：TTFT、ITL/TPOT、TTFT ratio、historical byte hit rate、deadline miss、sync cold miss、restore bytes/useful bytes
- [x] P0：实现 M3.11 B0-B5 dry-run runner、summary 聚合器和 schema 回归测试，生成完整结果目录骨架
- [x] P0：生成 512/2K/8K/16K/32K prefix 与 128/512/2K suffix 的 dry-run 矩阵，保留 32768 边界错误行
- [x] P0：接入 B0/B1/B2 原生 vLLM HTTP 在线测量路径，并完成 512/128/1 的小矩阵 smoke
- [x] P0：实现严格 B2 restart/cold-cache orchestration，而不是仅用 cold unique prefix 近似
- [x] P0：接入 B3/B5 sidecar+connector reuse bridge，并支持导入已有 `reuse_smoke_matrix.csv` 进入统一 baseline schema
- [x] P0：实现 B0/B1/B2/B3/B5 第一轮 2K 同机对比：vLLM full prefill、APC hot/restart、DRAM-ready reuse、KV Anti-Caching
- [x] P0：补充 TTFT breakdown 观测字段，区分在线服务 TTFT、restore wait、connector load/store 求和诊断和未解释开销
- [x] P0：修复复用请求仍触发全量 store/writeback 的路径，避免 reuse 阶段重写 48 层 KV
- [x] P0：重跑 load-only 后 2K B3/B5 真机样本，并重建 B0/B1/B2/B3/B5 统一汇总
- [ ] P0：扩展 512/2K/8K/16K/32K prefix 与 128/512/2K suffix 矩阵
- [ ] P0：实现 persistent session lineage + committed KV range 生命周期，用于多轮 append / idle restore
- [ ] P0：实现 KV PrePass，枚举本轮 required historical KV ranges，并输出 gpu_ready/cpu_ready/ssd_cold/fetching/loading/missing/mismatch sets
- [ ] P0：实现 packed cold object / extent layout 原型，替代每层小文件路径用于 32K 级 restore
- [ ] P1：接入 LMCache 与 DualPath/Tutti/CacheFlow/KVDrive-style baseline 或仿真对比
- [ ] 生成有效性对比报告，决定是否继续投入真实 3FS executor、packed cold object 和 partial promote
- **状态：** in_progress

### 阶段 M3.12：KV PrePass 与 restore hiding 验证
- [x] P0：实现最小 `/prepass` 控制面入口，执行前枚举 required historical KV，并调度 cold restore
- [x] P0：PrePass 不污染在线 admission 指标；单独暴露 `prepass_requests_total` 与 `prepass_ready_total`
- [x] P0：新增 deterministic PrePass restore smoke，证明 cold prefix 可在 online request 前变成 ready
- [x] P0：为 cold-tier reuse matrix 增加 `--prepass-before-reuse` 开关和 CSV 字段
- [x] P0：用真实 vLLM 跑 2K B5 PrePass-before-reuse 在线样本，对比 reactive B5 restore-inclusive 与 PrePass B5 online TTFT
- [x] P0：将 PrePass 结果导入 M3.11 baseline schema，新增 B5-prepass 或阶段字段
- [ ] P0：扩展到 512/2K/8K 小矩阵，验证 restore hiding 是否随上下文增长仍成立
- [ ] P0：补强 PrePass 报告口径，在统一 schema 中显式区分 online TTFT、prepass wait、restore-inclusive TTFT
- **状态：** in_progress

### 阶段 M3.13：PrePass Lead-Time 与 Packed Restore Gate
- [x] 用 Level2 视角重新诊断下一步研究内环：`选择 prefix/layout/lead-time -> 运行 vLLM -> 观察 TTFT/restore hiding -> 决定扩展或回退`
- [x] 实现 `prepass_lead_time_planner`，从真实 M3.12 PrePass CSV 和 M3.11 baseline CSV 推导 restore profile
- [x] 输出 512/2K/8K/16K/32K × available lead 的 `hide_restore`、`residual_online_wait_ms`、`packed_object_required`
- [x] 生成 `results/m3_13_prepass_lead_time_plan/` 与研究计划文档
- [ ] 按 planner 顺序跑 512/2K/8K true PrePass 小矩阵，验证 restore scaling 和 lead-time 模型
- [x] 设计 packed cold object / extent layout 原型，在 16K/32K 前验证 per-layer layout tail
- [ ] 在 packed layout 或 8K 稳定后，再跑 16K/32K gate matrix
- **状态：** in_progress

### 阶段 M3.14：KV Evicted Index 与 Packed Cold Object
- [x] 将数据库 Anti-Cache 的 Evicted Table / Block Table / pre-pass / requeue 思想整理为 KV 多级调度设计
- [x] 明确 GPU/CPU/SSD 三层语义：HBM 是在线执行层，CPU 是可准入准备层，SSD 是冷区和持久化层
- [x] 产出设计规格 `docs/specs/m3_14_kv_evicted_index_and_packed_object.md`
- [x] M3.14-A：实现 `KV Evicted Index` 独立模块，支持基础 residency 分类
- [x] M3.14-B：将 `/prepass` 输出升级为分类集合，而不只是计数和状态
- [x] M3.14-B2：修正三级存储语义，将 PrePass/Index 细分为 GPU_READY/CPU_READY/SSD_COLD/FETCHING/LOADING/MISSING/MISMATCH，并保留 ready/cold 兼容汇总
- [x] M3.14-B3：建立 CPU_READY -> GPU_READY load readiness 观测口径，区分 SSD->CPU restore 与 CPU->GPU connector load 的时间预算
- [x] M3.14-C：实现 `Packed Cold Object v1`，先支持 local_posix packed layout，再接 3fs_posix
- [x] M3.14-D：跑 2K/8K packed vs per-layer restore 对比，决定是否进入 16K/32K gate matrix
- [x] M3.14-E：优化 packed restore 数据面，避免当前 Python bytes 拼接/切片路径抵消文件数收益
- [ ] 在 packed restore 优化或 8K 结果稳定后，再回到 M3.13 的 16K/32K gate matrix
- **状态：** in_progress

### 阶段 M3.15：8K Packed PrePass Gate 与性能归因
- [x] 修复 8K store 静默截断风险：`run_reuse_smoke_matrix.py` 在 demote 后记录 `cold_saved_tokens`、`cold_expected_prefix_tokens` 和 `cold_saved_token_mismatch`
- [x] 使用 `--max-num-batched-tokens 16384` 重跑 8K store，确认 connector 保存 `8192` tokens、`512` blocks、`48` layers
- [x] 生成真实 `packed_v1` cold object：`1610616960` bytes，冷区仅保留 `packed_object.bin` 与 `packed_manifest.json`
- [x] 重启 vLLM 后运行 8K `--prepass-before-reuse`，验证 `SSD_COLD -> packed restore -> ADMIT -> external load`
- [x] 输出研究报告 `docs/research/m3_15_8k_packed_prepass_gate.md`
- [x] 补同配置 8K B0/B2 baseline：同样使用 `--max-model-len 16384 --max-num-batched-tokens 16384`
- [x] 按用户要求跳过 512，完成 2K/16K packed PrePass gate，并与既有 8K 形成 2K/8K/16K 趋势表
- [x] 输出趋势报告 `docs/research/m3_15_2k16k_packed_prepass_trend.md`
- [ ] 优化 packed restore/load 数据面，降低 8K/16K restore executor 与 connector load 时间
- [ ] 建立 2K/8K/16K lead-time matrix，量化提前多久 PrePass 才能把 restore 稳定隐藏
- [ ] 数据面归因后再进入 32K gate
- **状态：** in_progress

### 阶段 M3.16：LongMemEval-S 公共多轮长记忆 Workload 接入
- [ ] 确认 LongMemEval cleaned 官方入口、许可与 S/oracle 文件格式
- [ ] 实现 LongMemEval workload adapter：把 `haystack_sessions` 转成历史 KV prefix，把 `question` 转成本轮 suffix
- [ ] 支持 Qwen tokenizer 计数与 2K/8K/16K token-budget 裁剪，输出可复现 JSONL/CSV manifest
- [ ] 将 manifest 接入现有 `run_reuse_smoke_matrix.py`，允许真实数据 prompt 复用当前 B5 PrePass/cold-tier 路径
- [ ] 先生成小样本 dry-run，再选择 1 个 2K LongMemEval-S/Oracle 样本跑真实端到端验证
- [ ] 在研究报告中区分 synthetic microbenchmark 与 public long-memory workload benchmark
- **状态：** in_progress

### 阶段 M3.17：CCF-A 系统论文初稿
- [x] 盘点本地设计文档、实验结果和 LongMemEval-S 样本证据
- [x] 核验 Anti-Caching、PagedAttention/vLLM、LMCache、Mooncake、Tutti、KVDrive、CacheBlend、LongMemEval 等相关工作基本信息
- [x] 在 `docs/paper/` 创建论文证据表、BibTeX 和英文论文草稿
- [x] 重点完成 Abstract、Introduction、Related Work、Design
- [x] 在正文中显式区分 prototype evidence、design target 和 future production claim
- [ ] 下一步把 Markdown 草稿转成 LaTeX conference skeleton，并补 Evaluation/Methodology/Limitations 成完整投稿形态
- **状态：** in_progress

### 研究阶段 R2：KV Anti-Caching 方案深化
- [x] 复盘 Tutti 未覆盖的调度与系统语义问题
- [x] 将数据库 Anti-Caching 的 pre-pass、Evicted Table、Block Table、tuple-merge 等机制映射到 KV 管理
- [x] 评估 KV Anti-Caching 的可行性、创新性和论文潜力
- [x] 形成可落地设计文档
- [x] 仿照 MoE serving PPT 逻辑，产出 KV Anti-Caching 项目叙事 slides
- **状态：** complete

### 研究阶段 R3：CacheBlend Gap Analysis 与选题重构
- [x] 阅读本地 `CacheBlend.pdf`，提取其问题定义、系统假设、评测边界和 limitations
- [x] 对比当前 Persistent KV Anti-Caching 主线，区分 CacheBlend 的 RAG chunk fusion 与本项目的 committed historical KV cold-resume
- [x] 找出可 defend 的论文切入点、目标场景、方法轮廓和实验矩阵
- [x] 输出研究备忘录 `docs/research/cacheblend_gap_analysis_for_persistent_kv.md`
- **状态：** complete

## 关键问题
1. 方案面向的第一落地平台是否以 vLLM 为主，还是先做独立模拟器/原型后再接入 vLLM？
2. 1M tokens 场景的主目标是单请求超长上下文、批量长上下文服务，还是二者都要覆盖？
3. 评估基线需要覆盖哪些系统：原生 vLLM/PagedAttention、LMCache、KV offload、或者自研静态策略？
4. GPU/CPU/SSD 的目标硬件配置与可用实验环境是什么？

## 已做决策
| 决策 | 理由 |
|------|------|
| 在 `/root/KV` 创建规划文件 | `planning-with-files-zh` 要求规划文件放在项目目录，且该目录是本项目根目录 |
| 先读本地方案文档，再决定是否补充外部资料 | 用户明确要求先理解给定方案；避免用外部资料稀释当前架构意图 |
| 采用“仿真先行 + vLLM 低侵入原型”作为推荐路线 | 可以先验证策略收益，再最小化 vLLM 集成风险 |
| 创建 `AGENTS.md` 作为未来会话上下文规则 | `context-engineering` 建议为新项目建立持久规则文件，避免后续偏离精确语义和低侵入边界 |
| 将下一步调整为 M1.5 Benchmark 与硬件校准 | 用户指出仿真必须基于真实硬件和 vLLM baseline；这是当前最紧要的实验前置条件 |
| M1.5 默认模型固定为 `/root/models/gpt-oss-20b`，存储先用本地 NVMe | 用户已锁定；该模型 `sliding_window=128`，只用于系统路径校准，不代表 dense 1M attention |
| 正式 vLLM benchmark 运行时设置 `VLLM_PLUGINS=` 与 `NO_PROXY=127.0.0.1,localhost` | 避免本机残留 vLLM 插件和 socks5 代理污染 localhost benchmark |
| M2 仿真器第一版以 `results/benchmark_calibration/simulator_params.yaml` 作为冻结输入 | 该文件已由真实 vLLM/APC/NVMe/H2D 结果生成，状态为 `MEASURED` |
| 在 M2 前插入 M1.6 Qwen3-32B 校准 | Qwen3-32B 无滑动窗口，更接近完整 attention 路径；上一轮 gpt-oss 数据只作为系统路径校准，不宜直接代表 1M dense attention |
| Qwen3-32B 上探先做加载与短上下文冒烟 | 32B bf16 模型在单张 48GB A6000 上显存压力很高，直接跑大矩阵可能只得到 OOM 而缺少可解释中间数据 |
| Qwen3-32B 长上下文上探改为“容量探测优先，性能矩阵后置” | 当前失败首先来自权重/显存/KV 容量，而不是 benchmark 指标不足；必须先知道服务能声明并实际承载的上下文上限 |
| M1.6 完成后进入 M2 仿真，而不是继续扩 Qwen3 CPU-offload 矩阵 | 32GB 主存卸载可启动 32K/40K，但极小请求已证明推理很慢；继续测吞吐不如把容量边界和 KV 尺寸用于 1M 分层策略仿真 |
| M2 第一版采用控制平面级仿真，不追求内核级 vLLM 精确复现 | 当前要回答准入、复用、预取 deadline 和低局部性降级问题；内核细节应在策略边界明确后再接入 |
| 阶段 9 边界选择优先低预测误差和高有效命中率 | 只因预测少复用而少搬 KV 的行虽然更容易准入，但不能代表分层 KV 复用收益 |
| M3 先做 sidecar / connector 最小闭环，不直接改 vLLM 内核 | 阶段 9 已说明收益依赖准入、预取、ready barrier 和负载边界；应先验证控制面契约 |
| Qwen2.5-14B-Instruct 成为默认在线实验模型 | 它可在单张 A6000 上无 CPU 卸载启动到 32768，保留完整注意力路径，且 KV/token 比 Qwen3-32B 低 25% |
| Qwen3-32B 保留为压力上界和历史对照 | 它需要 CPU offload，路径过慢，不适合继续作为当前单卡开发主线 |
| M3 第一版先实现离线 sidecar replay，不直接触碰真实 vLLM KV 内部 | 可以先验证 correctness key、manifest、admission、ready barrier 和 sync SSD miss=0 的控制面语义 |
| M3.5 采用 HTTP sidecar + 可选 OpenAI completions 前置代理 | 先把真实请求旁路记录到控制面日志，不修改 vLLM KV 内部；后续再将 mock connector 替换为真实 connector |
| M3.6 的真实接入点是自定义 vLLM `KVConnectorBase_V1`，不是 HTTP sidecar 本身 | HTTP sidecar 只能记录和转发；真正影响 KV 复用需要 scheduler/worker connector 读取 `kv_transfer_params` 并操作 vLLM block |
| M3.6 第一版 connector 只做 no-op load | 先验证 sidecar plan、block 对齐、scheduler metadata 与 worker metadata 链路；真实 KV tensor copy 放到下一步，避免同时引入 layout 风险 |
| M3.7 才进入真实 KV tensor copy | M3.6 已证明 sidecar -> kv_transfer_params -> vLLM connector -> request_finished 输出路径可走通；下一步风险集中在 KV tensor layout 和正确性键 |
| M3.7 先做本地文件张量存储，不直接接 3FS/SSD 分层策略 | 当前阶段要先证明 vLLM 分页 KV 的块级保存和加载语义；三层淘汰、预取和远端存储应在真实两轮 smoke 稳定后再接 |
| M3.8 先补可观测性和自动提交，再跑在线小矩阵 | M3.7 的主要人工步骤是手动 commit，且保存/加载耗时不可结构化分析；先稳定这些基础，矩阵结果才可解释 |
| M3.8 在线矩阵必须区分 HTTP 成功和外部 KV 加载成功 | 本次 16/64 成功请求没有观测到 `load_request` 事件，可能由 vLLM 内置前缀缓存覆盖；后续必须显式验证 connector load |
| M3.8 用重启 vLLM 对照强制进入外部加载路径 | 保留 sidecar 控制面和 tensor store，只重启 vLLM 后，本地前缀缓存被清空，复用阶段观测到 connector `load_request` |
| M3.8 完整小矩阵采用“保存阶段、重启 vLLM、复用阶段”的实验形态 | 这样四个前缀长度都能绕开 vLLM 内置前缀缓存，直接验证外部 KV 加载路径 |
| M3.8 输出一致性 smoke 也必须分阶段运行 | 若普通路径和复用路径连续跑在同一个 vLLM 进程中，普通路径可能先污染内置前缀缓存，导致外部加载观测失真 |
| 进入 M3.9 DRAM/NVMe 分层迁移原型 | M3.8 已证明短前缀真实外部加载路径成立；下一步应验证容量层 KV 必须先预取到执行层，不能在 decode 关键路径同步读 NVMe |
| M3.9 sidecar 预取采用“本轮延迟、后续准入” | 保持 decode 关键路径不同步读 NVMe；第一次请求只触发预取，第二次看到 DRAM/ready 后才能 `ADMIT` |
| M3.9 第一版迁移队列只做确定性元数据调度 | 当前目标是先证明 deadline miss 不会被误标 ready；真实异步线程、3FS、H2D 和并发 I/O 放到下一步 |
| M3.9 队列指标先接入 sidecar `/metrics` | 让队列请求数、完成数、deadline miss、累计字节和预计完成时间可被监控，再推进后台异步化 |
| M3.9 驻留命中必须优先于预取队列 | 如果 tensor store 已经是 `DRAM/ready` 且 correctness key 完全匹配，本次请求应直接准入，不应被延迟或入队 |
| M3.9 后台异步队列先采用确定性推进 | 当前目标是验证“入队不等于 ready、推进后才 ready”的控制面语义，真实线程、真实 I/O 并发和 3FS 后置 |
| KV Anti-Caching 作为下一阶段核心设计抽象 | 它把 M3.9 的 ready barrier、tier state、prefetch queue 提升为 memory-primary、pre-pass、partial promote、non-blocking restore 的完整调度语义 |
| 当前 M3.10 只能证明 invariant，不能证明性能胜出 | restart 小矩阵和 qd4 adapter benchmark 已证明路径成立，但缺少 512-32K/1M、真实 3FS、强 baseline 和 TTFT ratio 证据 |
| 下一阶段优先做 baseline readiness matrix | 如果不先回答相对 vLLM APC、LMCache、Tutti、DualPath、CacheFlow、KVDrive 是否有效，继续堆 3FS adapter 功能会缺少科研判断依据 |
| 项目主线重命名为 Persistent KV Anti-Caching for Multi-Turn Long-Context Serving | KVDrive 已覆盖泛化多级 KV 管理；本项目必须聚焦多轮 persistent historical KV 的 exact、可验证、可准入复用 |
| P0 任务优先级高于真实 3FS executor | 真实 3FS 是重要数据面，但若没有 session lineage、PrePass、packed object 和 baseline matrix，无法证明核心创新 |
| 2K B0/B1/B2/B3/B5 同机矩阵只证明语义链路，不证明性能收益 | 结果显示 B5 cold probe/restore/load 语义成立，但 TTFT 是 B0 的约 9.19 倍；下一步必须优先降低 connector/cold-object 数据面开销 |
| TTFT breakdown 显示优先优化 reuse 阶段写回/connector 生命周期 | B5 restore 约 874ms、load 约 184ms，但在线服务 TTFT 约 6591ms，且 connector store_layer 求和约 6484ms；复用请求仍在重写 KV，是当前 P0 性能嫌疑点 |
| 复用请求默认应是 load-only，只有显式 `store_policy=store_prefix` 才写回 | M3.11 load-only smoke 中 store 阶段 `store_layer=48`，重启后 reuse 阶段 `store_events=0`、`load_events=1`；这说明 P0 store/writeback 误触发已修正 |
| load-only 后的 2K 重跑把 B3/B5 从“性能反例”修正为“有希望的在线路径” | B3 online TTFT 从约 6055ms 降到 283.027ms，B5 online TTFT 从约 6591ms 降到 327.512ms，说明旧结果主要被复用阶段写回污染 |
| B5 必须同时报告 online TTFT 与 restore-inclusive TTFT | 若 cold restore 已由 PrePass/idle window 提前隐藏，则 online TTFT 可对比 SLA；若请求到达后才同步等待 restore，则应看 `327.512 + 1035.707 = 1363.219ms`，仍高于 B0 的 1.2x 阈值 |
| 下一轮 P0 优先级调整为小矩阵扩展 + PrePass/async restore + packed cold object | 当前 B5 online/B0 为 0.456742，但 restore-inclusive/B0 约 1.90；要证明 Persistent KV Anti-Caching 的系统价值，必须把 restore 移出请求关键路径并降低 8K+ 文件布局开销 |
| M3.12-A PrePass 第一版只证明控制面 restore hiding，不等价于真实 vLLM 性能胜出 | deterministic smoke 已证明 `PREPASS -> ready_before_request=true -> online ADMIT`，但下一步必须跑真实 vLLM B5 PrePass-before-reuse 样本，才能把它纳入 TTFT 对比 |
| M3.13 优先做 lead-time planner，而不是盲跑 16K/32K | M3.12-B 显示 PrePass online TTFT 有优势，但 restore-inclusive 仍高于 B0；先量化提前量和 packed layout 触发点，才能让后续矩阵结果可解释 |
| 8K 是下一轮真实扩展的关键 gate | planner 估算 8K 需要约 4.29s lead，5s lead 可隐藏 restore 但已触发 packed object 风险；16K/32K 应等 8K 结果和 packed layout 后再跑 |
| M3.14 将 Anti-Cache 的 Evicted Table 映射为 KV Evicted Index | 我们不需要 100% 提前预测请求是否踩冷 KV，但必须在请求进入 GPU 前通过内存索引快速判断所需 KV 是 GPU_READY、CPU_READY、SSD_COLD、FETCHING、LOADING、MISSING 还是 correctness mismatch |
| M3.14 将 Anti-Cache 的 Block Table 映射为 Packed Cold Object | SSD 上的 KV 应以 object/extent 组织，并把 object id、offset、length 常驻内存索引；PrePass 只查索引即可收集恢复任务，不应扫描磁盘目录 |
| SSD 不是慢主存，而是冷区和持久化层 | SSD 上的 KV 永远不是 ready；只有恢复到 CPU/HBM 并校验后，才允许 vLLM connector 加载 |
| M3.14-B2 修正 ready/cold 语义为三级状态 | `CPU_READY` 只表示 SSD->CPU restore 已完成，不表示 decode 可直接执行；`GPU_READY` 才是 HBM 执行 ready；`LOADING` 用于表示 CPU->GPU load 进行中 |
| M3.14-E 修正 packed_v1 数据面后，packed layout 可以继续进入 8K/16K gate | 初始 packed_v1 虽减少文件数但 restore 明显变慢；streaming restore、fast manifest summarize、demote summary 不再重读 packed object 后，r5 2K/8K 小矩阵中 packed restore p95 比 per-layer 低约 58.521ms，冷区数据文件数从 40 降到 10。但这仍是 local POSIX + qwen25 tiny profile，不是 3FS/生产结论 |
| 8K packed PrePass gate 必须校验实际保存 token 数 | 首次 8K 尝试因 vLLM chunked prefill/scheduler token budget 只保存 2048 tokens；后续长上下文 store 必须要求 `cold_saved_tokens == requested_prefix_tokens`，否则样本无效 |
| M3.15 8K packed PrePass 链路成立但还不是性能终局 | 当前 8K 样本 online TTFT 为 710.835ms、restore executor 为 3046.794ms、restore-inclusive 为 3797.230ms；说明 PrePass 可以隐藏 restore，但若提前量不足仍不满足严格 SLA |
| 同配置 8K B0/B2 baseline 已补齐 | B0 full prefill 为 2409.141ms，严格 B2 restart 为 2418.796ms，B2/B0 为 1.004008；8K packed PrePass online 为 B0 的 0.295057x，但 restore-inclusive 为 B0 的 1.576176x |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `vllm serve --disable-log-requests` 参数不存在 | 1 | 改用默认关闭请求日志，并使用 `--disable-log-stats --disable-uvicorn-access-log` |
| 本机 `HTTP_PROXY/HTTPS_PROXY=socks5://11.11.11.7:1080` 导致 localhost benchmark 503 | 1 | benchmark 命令清理代理变量并设置 `NO_PROXY=127.0.0.1,localhost` |
| vLLM raw JSON 中 `errors` 可能为 `["", ""]` | 1 | parser 改为只统计非空错误槽 |
| `input_len=32768` 且 `output_len>0` 超出 `--max-model-len 32768` | 3 | 作为预期边界 `ERROR` 行保留在 `vllm_latency.csv`，不静默丢弃 |
| 本机未安装 `fio` | 1 | NVMe 校准使用 `python_fallback_fio_missing`，报告中记录为 fallback 而非 fio 结果 |
| Qwen3-32B 不带主存卸载时 `--max-model-len 2048` 仍无法加载 | 1 | 记录为权重加载 OOM；改用 `--cpu-offload-gb 24` |
| Qwen3-32B 主存卸载 + 默认图捕获启动过慢 | 1 | 记录图捕获耗时异常；改用 `--enforce-eager` 完成短上下文冒烟 |
| Qwen3-32B 主存卸载下 512/16/4 请求基准过慢 | 1 | 中断大于当前阶段需求的基准，改用 128/1/1 极小基准拿到可解释样本 |
| Qwen3-32B 24GB 主存卸载下 `--max-model-len 16384` KV 容量不足 | 1 | 记录为 `KV_CAPACITY_ERROR`；随后调高到 32GB 主存卸载，验证 32768/40960 可启动 |
| M2 脚本直接执行时找不到 `benchmarks` 包 | 1 | 补充 CLI 回归测试，并在脚本入口按 `__file__` 注入仓库根路径 |
| M2 sweep 初版边界排序偏向高预测误差行 | 1 | 增加回归测试，改为优先低预测误差和高有效命中率，再比较 deadline 与资源成本 |
| Qwen2.5 配置有 `sliding_window=131072` 但 `use_sliding_window=false` | 1 | 修正 `collect_env.py`，优先尊重 `use_sliding_window=false`，标记为完整注意力路径 |
| M2 仿真器写死 Qwen3 KV/token | 1 | 增加测试，`build_simulator_params.py` 写出模型专属 `kv_bytes_per_token`，M2 从 YAML 读取 |
| 运行 Qwen2.5 benchmark 时误用 `--suites` 参数 | 1 | 按 `run_vllm_bench.py --help` 确认接口，改用重复 `--suite` |
| FastAPI 对 `Response | dict` 返回类型生成响应模型失败 | 1 | 将 `/v1/completions` 路由显式设为 `response_model=None` 并返回 `Any` |
| HTTP sidecar 转发时被系统 socks proxy 污染 | 1 | `httpx.AsyncClient(..., trust_env=False)`，避免 localhost 转发走代理 |
| vLLM 退出时出现 `resource_tracker` leaked semaphore warning | 1 | 记录为关闭阶段 Python 资源追踪警告；本次在线 smoke 已完成且进程、显存均清理 |
| M3 no-op connector 首次实现误把模块函数当实例方法调用 | 1 | 改为调用模块级 `_is_usable_plan(params)`，并通过 connector 单元测试 |
| M3.7 初版把保存动作放在 `request_finished()` 后安排 | 1 | 改为在 `build_connector_meta()` 中为本轮 forward 同时生成 store/load 元数据，确保 `save_kv_layer()` 能在正确生命周期保存每层 KV |
| M3.8 首次在线矩阵全提示词外部命中触发 vLLM `assert num_new_tokens > 0` | 2 | 改为前缀复用加新增后缀，并在矩阵 CSV 中加入 `external_load_observed` |
| 一致性 smoke 脚本直接按路径运行时找不到 `benchmarks` 包 | 1 | 按已有 CLI 模式在脚本入口注入仓库根路径，并增加 `--help` 回归测试 |
| M3.9 迁移冒烟 CSV 字段顺序与测试预期不一致 | 1 | 调整 CSV 字段顺序，让 `prefix_id,initial_tier,demoted_tier,prefetched_tier` 成为前四列，便于人工扫读 |
| M3.9 sidecar CLI 初始无法配置 tensor store | 1 | 增加 `--tensor-store-path` 和 `--tensor-store-block-size`，并在 dry-run 输出中展开路径 |
| M3.9 sidecar deadline miss 初始仍把前缀直接标记 ready | 1 | 引入 `PrefetchQueue`，只有估算能赶上 `deadline_ms` 才调用 `prefetch_to_dram()` |
| M3.9 sidecar 预取烟测 CLI 初始不支持调低 storage 带宽 | 1 | 增加 `--storage-gbps`，并把 `DEADLINE_MISS` 作为有效边界结果返回 0 |
| M3.9 metrics 测试初始选用带宽过低导致小前缀也 deadline miss | 1 | 将测试带宽调到能完成 64 token、但 1M token 仍 miss 的区间 |
| M3.9 驻留命中初版只让下一次请求准入 | 1 | 调整 sidecar 准入顺序，先同步 `DRAM/ready` tensor store 状态，再重新准入当前请求 |
| M3.9 correctness key 不匹配的 tensor manifest 初版可能被队列预取成 ready | 1 | 在提交 `PrefetchQueue` 前校验 tensor manifest correctness key 和覆盖范围，不匹配只记录错误，不预取 |
| M3.9 异步去重逻辑初版误放进同步队列 | 1 | 将 `ALREADY_QUEUED` 去重移动到 `AsyncPrefetchQueue`，同步 `PrefetchQueue` 保持原行为 |
| M3.9 缺少 tensor manifest 初版被当作入队 | 1 | 修正为预取错误记录，不增加 `prefetch_queued_total` 或队列请求数 |
| B3 初始服务在后台 shell 退出后端口关闭 | 1 | 改用持久执行会话分别管理 vLLM 和 sidecar 生命周期 |
| B5 首次 2K store 阶段未带 `--cold-tier-restore`，导致没有真实 demote 到 cold tier | 1 | 保留原目录作为不完整样本，另起 `b5_anti_caching_true` 干净重跑 store+demote+mark-cold+restart+restore |
| 本机缺少 `column` 命令 | 1 | 改用 `sed` 查看 CSV 原始内容；后续脚本和报告不依赖 `column` |
| 8K store 首次只保存 2048 tokens 而不是 8192 tokens | 1 | 增加 `cold_saved_token_mismatch` guard，并用 `--max-num-batched-tokens 16384` 重跑有效 8K 样本 |
| B2 strict restart 首次传递 `--vllm-extra-arg --max-num-batched-tokens` 被 argparse 误解析 | 1 | 改用 `--vllm-extra-arg=--max-num-batched-tokens --vllm-extra-arg=16384` |

## 备注
- `task_plan.md` 记录阶段和决策。
- `findings.md` 记录方案理解、研究发现、风险和待确认点。
- `progress.md` 记录本次会话操作。
