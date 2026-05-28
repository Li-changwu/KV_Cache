# 进度日志

## 会话：2026-05-24

### 阶段 1：上下文恢复与资料盘点
- **状态：** complete
- **开始时间：** 2026-05-24T00:00:00Z
- 执行的操作：
  - 读取用户指定技能说明。
  - 确认项目目录 `/root/KV` 中当前仅有方案文档和 LICENSE。
  - 创建 `task_plan.md`、`findings.md`、`progress.md` 作为持久化工作记忆。
  - 阅读 `1M_Tokens_KV_Cache_分层管理技术方案.md` 的完整结构与核心章节。
  - 快速校验外部主资料：3FS、vLLM disaggregated prefill、vLLM prefix caching、vLLM NIXL connector、DualPath。
- 创建/修改的文件：
  - `/root/KV/task_plan.md`
  - `/root/KV/findings.md`
  - `/root/KV/progress.md`
  - `/root/KV/1M_Tokens_KV_Cache_实施规划.md`
  - `/root/KV/AGENTS.md`

### 阶段 2：研究问题与架构理解
- **状态：** complete
- 执行的操作：
  - 提炼项目研究问题：1M 精确长上下文的服务化准入与多级 KV 管理，而非透明 SSD 扩显存。
  - 梳理架构边界：服务入口准入层、KV 控制平面、多级存储连接器、vLLM 执行层、监控与 QoS 层。
  - 梳理关键生命周期：admission、prefill 复用、pre-pass、prefetch、decode ready barrier、eviction/writeback、recovery。

### 阶段 3：实施路径设计
- **状态：** complete
- 执行的操作：
  - 形成“仿真先行 + vLLM 低侵入原型并行收敛”的推荐路线。
  - 拆分 M0-M5：上下文固化、指标/schema、离线仿真、vLLM MVP、DualPath、在线 SLA 闭环。
  - 为每个阶段定义交付物、验收标准、风险和待确认问题。

### 阶段 4：Markdown 交付与沟通
- **状态：** complete
- 执行的操作：
  - 写入 `1M_Tokens_KV_Cache_实施规划.md`。
  - 写入 `AGENTS.md` 作为未来 agent 的项目上下文规则。
  - 检查新建 Markdown 的占位符、行数和阶段状态。

### 阶段 5：Benchmark 与仿真校准讨论
- **状态：** complete
- 执行的操作：
  - 使用 academic-research-suite 的 experiment-agent plan 思路评估下一步实验设计。
  - 重新读取方案文档和实施规划中 benchmark、baseline、仿真相关段落。
  - 检查本机 vLLM/GPU/DRAM/NVMe 环境：vLLM 0.19.0、RTX A6000、503GiB DRAM、Samsung 990 PRO NVMe。
  - 确认 vLLM `bench serve` 支持 TTFT/TPOT/ITL/E2E percentile metrics、prefix repetition dataset、JSON 保存和 detailed request 信息。
  - 写入 `Benchmark与仿真校准下一步规划.md`。
  - 实现 M1.5 benchmark/calibration harness：环境采集、vLLM 命令生成、结果解析、NVMe/H2D 微基准、simulator 参数与报告生成。
  - 为 parser、runner、simulator 参数生成器和硬件微基准补充单元测试。
  - 启动 vLLM 0.19.0 `/root/models/gpt-oss-20b` 最小 server，并完成 runner-based random/APC integration smoke。
  - 关闭 vLLM server，确认端口 8000 关闭且 GPU 显存回落。
- 创建/修改的文件：
  - `/root/KV/Benchmark与仿真校准下一步规划.md`
  - `/root/KV/findings.md`
  - `/root/KV/task_plan.md`
  - `/root/KV/progress.md`
  - `/root/KV/docs/specs/benchmark_metrics_and_matrix.md`
  - `/root/KV/configs/m1_5_benchmark.yaml`
  - `/root/KV/benchmarks/m1_5/collect_env.py`
  - `/root/KV/benchmarks/m1_5/run_vllm_bench.py`
  - `/root/KV/benchmarks/m1_5/parse_vllm_results.py`
  - `/root/KV/benchmarks/m1_5/run_io_microbench.py`
  - `/root/KV/benchmarks/m1_5/run_h2d_microbench.py`
  - `/root/KV/benchmarks/m1_5/build_simulator_params.py`
  - `/root/KV/tests/m1_5/test_parse_vllm_results.py`
  - `/root/KV/tests/m1_5/test_run_vllm_bench.py`
  - `/root/KV/tests/m1_5/test_build_simulator_params.py`
  - `/root/KV/tests/m1_5/test_hardware_microbench.py`

### 阶段 6：Benchmark 正式采样与仿真输入冻结
- **状态：** complete
- 执行的操作：
  - 采集正式环境信息到 `results/benchmark_calibration/env.json`。
  - 使用 `VLLM_PLUGINS=` 启动 vLLM server，并设置 benchmark 侧 `NO_PROXY=127.0.0.1,localhost` 避免代理污染。
  - 完成 B0-B5 vLLM benchmark 矩阵，共写出 92 个 raw JSON。
  - 关闭 vLLM server，并确认端口 8000 关闭、GPU 显存回落到 11 MiB。
  - 运行 NVMe 校准：`python benchmarks/m1_5/run_io_microbench.py --output results/benchmark_calibration/hardware_io.csv --data-file results/benchmark_calibration/io_probe.bin --size-mb 1024 --repeats 3`。
  - 运行 H2D 校准：`python benchmarks/m1_5/run_h2d_microbench.py --output results/benchmark_calibration/h2d_bandwidth.csv --warmups 3 --repeats 10`。
  - 解析 raw JSON：`python benchmarks/m1_5/parse_vllm_results.py --raw-dir results/benchmark_calibration/raw --output-dir results/benchmark_calibration`。
  - 生成仿真输入与报告：`python benchmarks/m1_5/build_simulator_params.py --result-dir results/benchmark_calibration`。
- 输出文件：
  - `/root/KV/results/benchmark_calibration/env.json`
  - `/root/KV/results/benchmark_calibration/raw/*.json`（92 个）
  - `/root/KV/results/benchmark_calibration/vllm_latency.csv`
  - `/root/KV/results/benchmark_calibration/apc_prefix_reuse.csv`
  - `/root/KV/results/benchmark_calibration/hardware_io.csv`
  - `/root/KV/results/benchmark_calibration/h2d_bandwidth.csv`
  - `/root/KV/results/benchmark_calibration/simulator_params.yaml`
  - `/root/KV/results/benchmark_calibration/report.md`
- 关键结果：
  - `vllm_latency.csv`：44 行，状态分布 `{'OK': 41, 'ERROR': 3}`。
  - `apc_prefix_reuse.csv`：48 行，状态分布 `{'OK': 48}`。
  - 3 个 `ERROR` 均为 `input_len=32768` 加 `output_len>0` 超出 `--max-model-len 32768` 的边界 Bad Request，不是 OOM。
  - NVMe 校准使用 `python_fallback_fio_missing`：4K/64K/1M/16M/64M 顺序读带宽分别约 2.735/8.766/8.472/7.274/2.413 GB/s。
  - H2D pinned copy：64MB/256MB/1GB/4GB 带宽约 24.797/25.111/25.076/24.974 GB/s。
  - `simulator_params.yaml` 的 `verification_status` 为 `MEASURED`。

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 单元测试 | `pytest tests/m1_5 -q` | 全部通过 | `10 passed in 0.05s` | pass |
| runner dry-run | `python benchmarks/m1_5/run_vllm_bench.py --config configs/m1_5_benchmark.yaml --suite random_smoke --dry-run \| wc -l` | 生成 2 条 smoke 命令 | 输出 `2` | pass |
| fixture parser | `python benchmarks/m1_5/parse_vllm_results.py --raw-dir tests/fixtures/vllm_results --output-dir /tmp/m1_5_parse_smoke_verify` | 生成 latency/APC CSV | 两个 CSV 均写出 | pass |
| IO smoke | `python benchmarks/m1_5/run_io_microbench.py --output /tmp/m1_5_hardware_io_verify.csv --data-file /tmp/m1_5_io_probe_verify.bin --size-mb 1 --repeats 1 --sizes 4K` | 生成硬件 IO CSV | `python_fallback_fio_missing`, 4K bandwidth 约 1.75 GB/s | pass |
| H2D smoke | `python benchmarks/m1_5/run_h2d_microbench.py --output /tmp/m1_5_h2d_bandwidth_verify.csv --warmups 1 --repeats 1 --sizes 64MB` | 生成 H2D CSV | 64MB bandwidth 约 24.85 GB/s | pass |
| simulator/report smoke | `python benchmarks/m1_5/build_simulator_params.py --result-dir /tmp/m1_5_build_verify` | 生成 YAML 和 report | `simulator_params.yaml` 与 `report.md` 非空 | pass |
| vLLM runner integration smoke | `run_vllm_bench.py` 临时 128-token random + 256/64 prefix repetition 配置 | raw JSON 可解析为 random/APC CSV | random/APC 各 2/2 completed、0 failed、parser 输出 `status=OK` | pass |
| 阶段 6 单元测试 | `pytest tests/m1_5 -q` | 全部通过 | `10 passed in 0.06s` | pass |
| 阶段 6 产物完整性 | `test -s results/benchmark_calibration/{env.json,vllm_latency.csv,apc_prefix_reuse.csv,hardware_io.csv,h2d_bandwidth.csv,simulator_params.yaml,report.md}` | 全部非空 | 七个正式产物均非空 | pass |
| 阶段 6 raw/CSV 计数 | Python CSV/JSON 统计 | raw=92；latency 保留 ERROR；APC 全 OK | raw=92，latency 41 OK + 3 ERROR，APC 48 OK | pass |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-05-24T12:41:00Z | `vllm serve --disable-log-requests` 不被 vLLM 0.19.0 接受 | 1 | 查询 help 后改用 `--disable-log-stats --disable-uvicorn-access-log` |
| 2026-05-24T12:43:00Z | 首次手写 random smoke 全部 503 `Service Unavailable` | 1 | 发现本机代理变量影响 localhost；清理代理并设置 `NO_PROXY=127.0.0.1,localhost` 后通过 |
| 2026-05-24T12:45:00Z | vLLM raw JSON 的 `errors` 字段为 `["", ""]` 时 parser 误判 ERROR | 1 | parser 改为只统计非空错误槽，并补单元测试 |
| 2026-05-24T15:30:00Z | `input_len=32768` 与非零输出长度超出 server max context | 3 | 作为边界 `ERROR` 保存在 `vllm_latency.csv`，不纳入 simulator scalar 表 |
| 2026-05-24T15:34:00Z | 正式 NVMe 校准环境缺少 `fio` | 1 | 使用 Python fallback 顺序读，并在 CSV `method` 字段记录 `python_fallback_fio_missing` |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 7 正在进行，已完成 Qwen3-32B 加载可行性、主存卸载极小冒烟和专属校准产物 |
| 我要去哪里？ | 先完成 Qwen3-32B 服务容量探测，再决定是否进入 M2 仿真或更换量化/小模型/多卡策略 |
| 目标是什么？ | 用更接近完整 attention 的 Qwen3-32B 明确当前硬件边界，避免把 gpt-oss 滑动窗口曲线误当成 1M dense baseline |
| 我学到了什么？ | 见 `findings.md` |
| 我做了什么？ | 完成 M1.5 正式采样；随后切换到 Qwen3-32B，证明单卡无卸载 OOM，24GB 主存卸载可启动但很慢 |

---
*每个阶段完成后或遇到错误时更新此文件。*

## 会话：2026-05-25 相关工作调研与系统/论文规划
- **状态：** complete
- 使用技能：
  - `academic-research-suite`：采用 deep-research / academic-pipeline / experiment-agent 的研究问题、方法蓝图、实验规划框架。
  - `level2-research`：把 KV admission / prefetch / placement 抽象为优化闭环，并提出 Budgeted Slack Scheduler。
  - `vibe-research`：只借用需求验证、竞争格局和差异化定位框架；未按 app idea 问卷执行。
- 执行的操作：
  - 重新读取 `task_plan.md`、`findings.md`、`AGENTS.md` 和 M3.9 当前状态。
  - 调研相关工作并按类别整理：PagedAttention/vLLM APC、SGLang/RadixAttention、DistServe/NIXL、LMCache、Mooncake、Tutti、CacheFlow、DualPath、FlexGen、H2O/Scissorhands/CacheGen、3FS。
  - 从论文可发表性的角度重新定义系统定位：`SLA-governed exact long-context KV service`。
  - 设计推荐架构 `SLA-KV`：correctness key、manifest、reuse planner、residency planner、admission guard、multi-resource restore scheduler、ready barrier、metrics。
  - 明确 90% hit rate 的指标口径风险，并建议区分 all-token、historical KV、byte hit rate 和 admitted SLA class hit rate。
  - 形成 CCF-A 论文化判断：当前 M3.9 不够，必须补真实 cold-tier I/O、1M/128K scaling、强 baseline 和生产/真实 workload。
- 创建/修改的文件：
  - `/root/KV/docs/research/kv_multitier_related_work_and_design_plan.md`
  - `/root/KV/findings.md`
  - `/root/KV/progress.md`
- 后续建议：
  - M3.10 先实现真实 cold-tier 目录/设备级迁移和后台 executor。
  - M2 simulator 增加 Budgeted Slack Scheduler 策略。
  - 在线矩阵扩展到 512-32K prefix，并准备 128K/1M scaling 实验。

## 会话：2026-05-25 KV Anti-Caching 方案深化
- **状态：** complete
- 使用技能：
  - `planning-with-files-zh`：恢复 `task_plan.md`、`findings.md`、`progress.md`，并将本轮产出同步回项目工作记忆。
  - `academic-research-suite`：采用 deep-research / experiment-agent 的研究问题、方法与实验规划框架。
  - `level2-research`：把 KV restore/admission 视作 propose -> evaluate -> decide -> observe 的优化闭环，形成 Budgeted Anti-Cache Scheduler。
  - `vibe-research`：只借用可行性、差异化定位和竞品压力检查视角；未执行 app idea 问卷。
- 执行的操作：
  - 读取当前 M2 simulator、M2 sweep、M3 prefetch queue、M3 control plane、M3 tensor store 和 M3 connector/sidecar 接口。
  - 复盘 Anti-Caching PDF 中的 Evicted Table、Block Table、LRU Chain、pre-pass、abort/requeue、tuple-merge 和 lazy compaction 机制。
  - 评估 KV Anti-Caching 的可行性和创新性：可行性 4/5，创新性 4/5，成立条件是从 offload 转向 memory-primary KV anti-cache semantics。
  - 完成设计文档，明确 KVPrePass、KVResidencyTable、ColdBlockTable、SampledHotnessIndex、PartialPromote、Budgeted Anti-Cache Scheduler、三类策略族和实验路线。
- 创建/修改的文件：
  - `/root/KV/docs/research/kv_anti_caching_design.md`
  - `/root/KV/task_plan.md`
  - `/root/KV/findings.md`
  - `/root/KV/progress.md`
- 后续建议：
  - M3.10-A 先实现真实 cold-tier executor，将当前元数据迁移替换为真实文件/目录级搬运。
  - M3.10-B 增加 KVPrePass 和 partial promote，避免整块 cold object 读放大。
  - M3.11 将 `PrefetchQueue` 升级为多资源 Budgeted Anti-Cache Scheduler。

## 会话：2026-05-25

### 阶段 7：M1.6 Qwen3-32B 真实基线与长上下文上探
- **状态：** in_progress
- 执行的操作：
  - 依据用户要求，将下一步从“直接 M2 仿真”调整为“先用 Qwen3-32B 重新做真实基线校准，再进入 M2 仿真”。
  - 使用 context-engineering 更新项目规则文件 `AGENTS.md`，加入 Qwen3-32B 当前实验顺序、上下文边界和显存风险。
  - 使用 planning-with-files-zh 更新 `task_plan.md`，插入 M1.6 阶段，并将 M2 仿真顺延为阶段 8。
  - 更新 `findings.md`，记录 Qwen3-32B 本地配置、README 上下文说明、YaRN/RoPE scaling 条件和 KV cache 粗略估算。
- 当前设计：
  - 先运行 Qwen3-32B 环境采集和加载可行性。
  - 若加载成功，再运行最小 random / prefix repetition 冒烟。
  - 只在冒烟成功后上探 8192、16384、32768、40960。
  - 65536、131072 只作为启用 YaRN/RoPE scaling 后的扩展上下文实验。
- 追加执行：
  - 新增 `configs/m1_6_qwen3_32b_benchmark.yaml`，定义 Qwen3-32B 专用小矩阵，不覆盖 M1.5 结果。
  - 新增 `configs/m1_6_qwen3_32b_smoke_tiny.yaml`，用于主存卸载路径下的极小可服务性冒烟。
  - 修正 `benchmarks/m1_5/collect_env.py`，按模型配置判断是否滑动窗口，不再把所有模型标成 gpt-oss 限制。
  - 修正 `benchmarks/m1_5/build_simulator_params.py`，报告和 YAML 从 `env.json` 继承模型代表性说明。
  - 补充 `tests/m1_5/test_collect_env.py`，并更新 simulator 参数测试。
  - 运行 `pytest tests/m1_5 -q`，结果 `12 passed in 2.10s`；收尾复验为 `12 passed in 2.01s`。
  - 采集 Qwen3-32B 环境到 `results/qwen3_32b_calibration/env.json`。
  - 加载实验 1：`vllm serve /root/models/Qwen3-32B --max-model-len 2048 --gpu-memory-utilization 0.90`，失败，原因是权重加载阶段 CUDA OOM。
  - 加载实验 2：加入 `--cpu-offload-gb 24`，模型权重可加载，日志显示 CPU offloaded parameters 约 24.28GB，模型加载占用显存约 36.76GiB；但默认图捕获预计耗时过长，中断。
  - 加载实验 3：加入 `--cpu-offload-gb 24 --enforce-eager`，服务成功启动，最大模型长度 2048，GPU KV cache size 10720 tokens，2K 请求最大并发约 5.23x。
  - 手动极小请求：`Hello` 输入、1 token 输出成功，耗时约 6.682 秒。
  - 中断 `random_smoke` 中 512 输入、16 输出、4 请求配置，因为主存卸载路径下运行过慢，不适合作为当前阶段冒烟。
  - 使用极小配置完成 Qwen3-32B random 与 prefix repetition 各 1 条可解析结果。
  - 复制 M1.5 的硬件 IO/H2D CSV 到 Qwen3-32B 结果目录，作为同机硬件校准复用。
  - 生成 `results/qwen3_32b_calibration/{vllm_latency.csv,apc_prefix_reuse.csv,simulator_params.yaml,report.md}`。
- Qwen3-32B 极小结果：
  - random 128 输入、1 输出、1 请求：TTFT p50 约 4372.88ms，E2E p50 约 4372.88ms，状态 OK。
  - prefix repetition 128 前缀 + 16 后缀、1 输出、1 请求：TTFT p50 约 24090.90ms，E2E p50 约 24090.90ms，状态 OK。
  - 这些结果说明服务可用，但 24GB 主存卸载路径极慢；不能把它当作理想 Qwen3-32B GPU-only 基线。
- 收尾验证：
  - `raw_json_count=2`。
  - `vllm_latency.csv` 1 行 OK。
  - `apc_prefix_reuse.csv` 1 行 OK。
  - `hardware_io.csv` 5 行 OK。
  - `h2d_bandwidth.csv` 4 行 OK。
  - Qwen3 服务已关闭，端口 8000 关闭，GPU 显存回落到 11MiB。

### 阶段 7 续：Qwen3-32B 容量探测优先顺序
- **状态：** complete
- 本轮调整：
  - 先更新上下文与规划，把“直接跑长上下文性能矩阵”改为“先确认服务容量边界”。
  - 新增 `benchmarks/m1_5/probe_vllm_capacity.py`，解析 vLLM serve 日志中的启动状态、GPU KV cache tokens、最大并发、CPU offload 参数和 KV 容量错误。
  - 先为该脚本补充测试，确认 STARTED、WEIGHT_OOM、KV_CAPACITY_ERROR 和 CSV 写出行为。
- 执行的容量实验：
  - 解析既有日志：2048 无卸载为 `WEIGHT_OOM`；2048 + 24GB 主存卸载为 `STARTED`，GPU KV cache size 10720 tokens，最大并发约 5.23x。
  - 运行 8192 + 24GB 主存卸载：`STARTED`，GPU KV cache size 10720 tokens，最大并发约 1.31x。
  - 运行 16384 + 24GB 主存卸载：`KV_CAPACITY_ERROR`，需要 4.0GiB KV cache，可用 2.62GiB，vLLM 估计最大模型长度 10720。
  - 运行 32768 + 32GB 主存卸载：`STARTED`，CPU offloaded parameters 约 32.45GB，GPU KV cache size 44128 tokens，最大并发约 1.35x。
  - 运行 40960 + 32GB 主存卸载：`STARTED`，GPU KV cache size 44128 tokens，最大并发约 1.08x。
- 产物：
  - `results/qwen3_32b_calibration/qwen3_capacity_probe.csv`
  - `results/qwen3_32b_calibration/qwen3_capacity_probe_8192.csv`
  - `results/qwen3_32b_calibration/qwen3_capacity_probe_16384.csv`
  - `results/qwen3_32b_calibration/qwen3_capacity_probe_32768_offload32.csv`
  - `results/qwen3_32b_calibration/qwen3_capacity_probe_40960_offload32.csv`
  - `results/qwen3_32b_calibration/logs/capacity_cpu_offload24_eager_maxlen8192.log`
  - `results/qwen3_32b_calibration/logs/capacity_cpu_offload24_eager_maxlen16384.log`
  - `results/qwen3_32b_calibration/logs/capacity_cpu_offload32_eager_maxlen32768.log`
  - `results/qwen3_32b_calibration/logs/capacity_cpu_offload32_eager_maxlen40960.log`
- 报告更新：
  - 更新 `benchmarks/m1_5/build_simulator_params.py`，当结果目录存在 `qwen3_capacity_probe.csv` 时，在 `report.md` 中加入 Qwen3 Capacity Probe 章节。
  - 重建 `results/qwen3_32b_calibration/simulator_params.yaml` 和 `report.md`。
- 验证：
  - `pytest tests/m1_5 -q` 输出 `16 passed in 1.84s`。
  - Qwen3 服务已关闭，无残留 vLLM/EngineCore 进程。
  - GPU 显存回落到 11MiB。
- 结论：
  - Qwen3-32B 在单张 A6000 上并非不能声明 32768/40960，但需要 32GB 级主存卸载，且最大并发只剩约 1.35x/1.08x。
  - 该路径不适合继续跑完整性能矩阵；下一步应进入 M2 仿真，把 Qwen3 KV 尺寸和容量边界作为约束。

## 五问重启检查（2026-05-25 更新）
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 7 已完成，Qwen3-32B 加载、极小冒烟和容量边界均已记录 |
| 我要去哪里？ | 阶段 8：M2 仿真器第一版 |
| 目标是什么？ | 用 Qwen3 的真实 KV 尺寸/容量边界和已测硬件数据，模拟 1M 精确 KV 分层管理是否有收益 |
| 我学到了什么？ | Qwen3-32B 单卡无卸载 OOM；24GB 卸载最多约 10720 tokens；32GB 卸载可启动 32768/40960 但不是理想性能基线 |
| 我做了什么？ | 生成容量探测脚本、CSV、报告章节，并通过 `pytest tests/m1_5 -q` 验证 |

### 阶段 8：M2 仿真器第一版
- **状态：** complete
- 执行的操作：
  - 重新读取 `task_plan.md`、`findings.md`、`progress.md`、Qwen3 `simulator_params.yaml` 和 `qwen3_capacity_probe.csv`。
  - 使用测试先行方式新增 `tests/m2/test_simulator.py`，覆盖 calibration 加载、工作负载生成、复用/低局部性区分、deadline miss 准入语义、CSV/report 输出、CLI 直接执行。
  - 新增 `benchmarks/m2/workloads.py`，生成 `session_append`、`shared_prefix`、`low_locality` 三类 1M tokens 工作负载。
  - 新增 `benchmarks/m2/simulator.py`，加载 Qwen3/硬件校准数据，使用 Qwen3 每 token KV `262144 bytes`，模拟 HBM/DRAM/NVMe 三级 residency、prefetch、admission 和 delay/reject。
  - 修复 CLI 直接执行时 `ModuleNotFoundError: No module named 'benchmarks'`，通过回归测试锁定。
  - 修正 admission 语义：只有容量满足、预取窗口满足且不超过 request deadline 的请求才算 admitted；deadline miss 但容量足够的请求标为 delayed。
- 生成产物：
  - `results/m2_simulation/m2_summary.csv`
  - `results/m2_simulation/m2_requests.csv`
  - `results/m2_simulation/report.md`
  - `results/m2_simulation/strict_250ms/{m2_summary.csv,m2_requests.csv,report.md}`
  - `results/m2_simulation/relaxed_12000ms/{m2_summary.csv,m2_requests.csv,report.md}`
  - `results/m2_simulation/experiment_summary.md`
- 关键结果：
  - 严格 250ms deadline：`session_append` 命中率约 0.688，`shared_prefix` 命中率约 0.656，但二者 deadline miss rate 约 0.875，全部进入 delay；`low_locality` 命中率 0，全部 delay。
  - 宽松 12000ms deadline：`session_append` 和 `shared_prefix` 各 8 个请求中 7 个 admitted，首个无复用请求 delayed；`low_locality` 仍全部 delayed。
  - 两组实验 `sync_ssd_miss_rate` 均为 0，符合 decode path 不同步读 SSD 的硬边界。
- 验证：
  - `pytest tests -q` 输出 `20 passed in 2.13s`。

## 五问重启检查（阶段 8 后）
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 8 已完成，M2 仿真器第一版和默认实验结果已生成 |
| 我要去哪里？ | 阶段 9：M2 策略增强与敏感性分析 |
| 目标是什么？ | 找出 1M 精确 KV 分层管理在 deadline、带宽、DRAM、复用率变化下的收益边界 |
| 我学到了什么？ | 本机 250ms deadline 下即使有复用也赶不上大段 KV 搬运；放宽到 12s 后复用型负载可准入，低局部性仍应延迟/降级 |
| 我做了什么？ | 实现仿真器、三类负载、严格/宽松两组实验、报告和测试 |

### 阶段 9：M2 策略增强与敏感性分析
- **状态：** complete
- 执行的操作：
  - 重新读取 `task_plan.md`、`findings.md`、`progress.md`、`AGENTS.md`、M2 仿真器代码和测试，恢复阶段 9 上下文。
  - 使用测试先行方式扩展 `tests/m2/test_simulator.py`，覆盖预取提前量、H2D 并行度、复用预测误差对准入和命中率的影响。
  - 新增 `tests/m2/test_sweep.py`，覆盖参数扫描网格、收益边界、deadline frontier、CLI 输出和边界排序。
  - 扩展 `benchmarks/m2/simulator.py`，新增 `prefetch_lead_ms`、`h2d_parallelism`、`nvme_parallelism`、`reuse_prediction_error_rate`，并将 deadline miss 判断改为扣除提前预取后的 critical path。
  - 新增 `benchmarks/m2/sweep.py`，支持 deadline、预取提前量、H2D/NVMe 带宽、DRAM/HBM tokens、I/O 并行度、预测误差的笛卡尔扫描。
  - 新增 `docs/specs/m3_connector_sidecar_minimal_interface.md`，定义 M3 sidecar / connector 的最小控制面接口。
  - 发现 sweep 初版边界排序会偏向高预测误差行，因为预测少复用会少搬 KV；补充回归测试后改为优先低预测误差和高有效命中率。
- 生成产物：
  - `results/m2_sweep/m2_sweep_summary.csv`：311040 行完整扫描明细。
  - `results/m2_sweep/m2_sweep_boundaries.csv`：每类 workload 的总体收益边界。
  - `results/m2_sweep/m2_sweep_deadline_frontier.csv`：每个 deadline 下的前沿条件。
  - `results/m2_sweep/sweep_report.md`：人读报告。
  - `docs/specs/m3_connector_sidecar_minimal_interface.md`：M3 最小接口说明。
- 关键结果：
  - `low_locality` 在 250ms、1000ms、5000ms、12000ms、30000ms 下均没有 SLA 区域，最佳 admitted rate 为 0。
  - `session_append` 在 250ms 下进入目标准入率需要约 5000ms 提前预取、H2D 25GB/s x4、NVMe/3FS 20GB/s x4、DRAM 500000 tokens、HBM 44128 tokens；admitted rate 0.875，effective hit rate 0.6875。
  - `shared_prefix` 在 250ms 下需要 H2D 200GB/s x4、NVMe/3FS 50GB/s x2、DRAM 500000 tokens、HBM 262144 tokens；admitted rate 0.875，effective hit rate 0.65625。
  - 放宽到 5000ms 到 12000ms 后，复用型负载对本机 25GB/s H2D 和较低 NVMe/3FS 带宽更友好；这说明 admission window 和提前预取是阶段 10 的关键机制。
- 验证：
  - 新增测试初始红灯：缺少 `benchmarks.m2.sweep`、缺少 `find_deadline_frontier`、边界排序偏向高预测误差，均已按原因修复。
  - `pytest tests/m2 -q` 输出 `13 passed in 0.26s`。

## 五问重启检查（阶段 9 后）
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 9 已完成，M2 sweep 和收益边界报告已生成 |
| 我要去哪里？ | 阶段 10：M3 vLLM connector / sidecar 最小闭环 |
| 目标是什么？ | 把仿真中的准入、预取、ready barrier、低局部性降级语义落到一个可对接 vLLM 的最小控制面原型 |
| 我学到了什么？ | 1M 精确分层 KV 的收益强依赖复用型负载、提前预取和搬运带宽；随机低局部性负载不应进入 SLA 快路径 |
| 我做了什么？ | 实现参数扫描器、deadline frontier、M3 最小接口文档，并通过 M2 测试验证 |

### 模型切换评估：Qwen2.5-14B-Instruct
- **状态：** decision_needed
- 背景：
  - 用户指出 Qwen3-32B 太大，需要 CPU offload，给本机实验造成很大负担。
  - 新下载模型路径为 `/root/models/Qwen2.5-14B-Instruct`。
- 执行的操作：
  - 读取 Qwen2.5-14B 的 `config.json`、`tokenizer_config.json` 和 `README.md`。
  - 对比 Qwen2.5-14B 与 Qwen3-32B 的本地模型大小、上下文配置、滑动窗口标志和 KV/token 成本。
- 关键发现：
  - Qwen2.5-14B 模型文件约 28GB，Qwen3-32B 约 62GB。
  - Qwen2.5-14B 默认 `max_position_embeddings=32768`，`use_sliding_window=false`，`rope_scaling=null`；默认是完整注意力路径，不是滑动窗口路径。
  - Qwen2.5-14B README 说明 128K 需要 YaRN/RoPE scaling；因此 131072 应作为扩展上下文实验标记。
  - Qwen2.5-14B 每 token KV 约 196608 bytes，即 192KiB/token；Qwen3-32B 为 262144 bytes，即 256KiB/token。
  - 1M tokens 下，Qwen2.5-14B 的 KV 约 192GiB，Qwen3-32B 约 256GiB。Qwen2.5-14B 能降低约 25% KV 压力。
- 初步建议：
  - 有必要将后续默认在线 benchmark 和 M3 原型基座切换为 Qwen2.5-14B-Instruct。
  - Qwen3-32B 保留为压力上界和历史对照，不再作为当前单卡开发主线。

### 阶段 M1.7：Qwen2.5-14B 默认在线基座校准
- **状态：** complete
- 执行的操作：
  - 按用户确认，将 `/root/models/Qwen2.5-14B-Instruct` 正式设为默认在线实验模型。
  - 使用测试先行修正 `benchmarks/m1_5/collect_env.py`：当 `use_sliding_window=false` 时，即使配置存在 `sliding_window` 字段，也标记为完整注意力路径。
  - 使用测试先行修正 `benchmarks/m1_5/build_simulator_params.py`：从 `env.json` 的模型配置推导 `kv_bytes_per_token` 并写入 `simulator_params.yaml`。
  - 使用测试先行修正 `benchmarks/m2/simulator.py`：从 `simulator_params.yaml` 读取模型专属 KV/token，不再写死 Qwen3 的 262144 bytes/token。
  - 新增 `configs/m1_7_qwen25_14b_smoke.yaml`，用于 Qwen2.5 的极小 random / prefix repetition 冒烟。
  - 采集环境到 `results/qwen25_14b_calibration/env.json`，复用同机硬件 IO/H2D 校准。
  - 运行无 CPU offload 容量探测：2048、8192、16384、32768。
  - 启动 Qwen2.5-14B 32768 服务并运行极小 random 与 APC 冒烟；完成后关闭服务，GPU 显存回落到 11MiB。
  - 解析 raw JSON，生成 `vllm_latency.csv`、`apc_prefix_reuse.csv`、`simulator_params.yaml` 和 `report.md`。
  - 用 Qwen2.5 参数重跑 M2 sweep，输出到 `results/m2_sweep_qwen25_14b/`。
- 关键结果：
  - Qwen2.5-14B 模型加载占用约 27.57GiB 显存；可用 KV cache memory 约 11.86GiB。
  - GPU KV cache size 为 64752 tokens。
  - `--max-model-len 2048/8192/16384/32768` 全部 `STARTED`，最大并发约 31.62x / 7.90x / 3.95x / 1.98x。
  - random 128 输入、1 输出、1 请求：TTFT p50 约 216.28ms，状态 OK。
  - prefix repetition 128 前缀 + 16 后缀、1 输出、1 请求：TTFT p50 约 85.22ms，状态 OK。
  - Qwen2.5 专属 `simulator_params.yaml` 中 `kv_bytes_per_token=196608`，完整注意力代表性为 `full_attention_path_for_configured_context`。
  - Qwen2.5 M2 sweep 中，`low_locality` 仍无 SLA 区域；复用型负载进入 SLA 的条件比 Qwen3 部分放松，但 250ms 级 1M 精确服务仍强依赖高 H2D 带宽和复用形态。
- 遇到的问题：
  - 首次运行 benchmark 使用了错误参数 `--suites`，脚本实际接口是可重复 `--suite`；读取 `--help` 后改为 `--suite random_smoke --suite apc_sweep` 并成功。
- 产物：
  - `configs/m1_7_qwen25_14b_smoke.yaml`
  - `results/qwen25_14b_calibration/env.json`
  - `results/qwen25_14b_calibration/qwen25_capacity_probe.csv`
  - `results/qwen25_14b_calibration/vllm_latency.csv`
  - `results/qwen25_14b_calibration/apc_prefix_reuse.csv`
  - `results/qwen25_14b_calibration/simulator_params.yaml`
  - `results/qwen25_14b_calibration/report.md`
  - `results/m2_sweep_qwen25_14b/m2_sweep_summary.csv`
  - `results/m2_sweep_qwen25_14b/m2_sweep_boundaries.csv`
  - `results/m2_sweep_qwen25_14b/m2_sweep_deadline_frontier.csv`
  - `results/m2_sweep_qwen25_14b/sweep_report.md`
- 验证：
  - 局部修正验证：`pytest tests/m1_5/test_collect_env.py tests/m1_5/test_build_simulator_params.py tests/m2/test_simulator.py -q` 输出 `15 passed in 2.25s`。
  - 中途回归验证：`pytest tests/m1_5 tests/m2 -q` 输出 `32 passed in 2.47s`。

### 阶段 10：M3 vLLM connector / sidecar 最小闭环
- **状态：** complete
- 执行的操作：
  - 重新读取 `task_plan.md`、`findings.md`、`progress.md`、`AGENTS.md` 和 `docs/specs/m3_connector_sidecar_minimal_interface.md`，确认阶段 10 范围。
  - 使用测试先行新增 `tests/m3/test_control_plane.py`，覆盖：
    - 同一 session 第二轮复用已 committed prefix，只 delta prefill 新增 token。
    - ready barrier 阻止未就绪 SSD KV 进入同步读取路径。
    - 低局部性 1M 请求进入 `FULL_PREFILL_FALLBACK`，不声明分层 KV 快路径收益。
    - correctness key mismatch 返回 `REJECT`。
    - 离线 replay 写出决策 CSV 和 report。
  - 新增 `benchmarks/m3/control_plane.py`，实现 `CorrectnessKey`、`KVManifest`、`MockKVConnector`、`TieredKVControlPlane`、admission decision 和 ready barrier。
  - 新增 `benchmarks/m3/replay.py` 和 `benchmarks/m3/replay_cli.py`，支持用 Qwen2.5 校准参数跑固定两轮 session replay。
- 生成产物：
  - `results/m3_sidecar_replay/m3_replay_decisions.csv`
  - `results/m3_sidecar_replay/report.md`
- 关键结果：
  - `session_a_turn_1`：无可复用前缀，decision 为 `FULL_PREFILL_FALLBACK`，`delta_prefill_tokens=1048576`，`sync_ssd_miss_total=0`。
  - `session_a_turn_2`：复用 `786432` tokens，`delta_prefill_tokens=262144`，decision 为 `ADMIT`，`sync_ssd_miss_total=0`。
  - ready barrier 对未就绪 SSD range 返回 `all_required_blocks_ready=false`，但同步 SSD miss 仍为 0，符合硬边界。
- 验证：
  - M3 初始红灯：`ModuleNotFoundError: No module named 'benchmarks.m3'`，随后实现最小模块。
  - CLI 初始红灯：缺少 `benchmarks/m3/replay_cli.py`，随后实现并通过。
  - `pytest tests/m3 -q` 输出 `6 passed in 0.13s`。

## 五问重启检查（阶段 10 后）
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 10 已完成 M3 sidecar/control-plane 最小闭环 |
| 我要去哪里？ | 阶段 M3.5：真实 vLLM 前置代理或 connector 对接实验 |
| 目标是什么？ | 把离线 replay 的准入、复用、ready barrier 语义接到真实在线请求路径上 |
| 我学到了什么？ | 控制面最小语义可以独立验证：正确性键、manifest、delta prefill、ready barrier 和 sync SSD miss=0 都可测试 |
| 我做了什么？ | 实现 M3 控制面模块、mock connector、replay CLI 和结果报告 |

### 阶段 M3.5：HTTP sidecar / 前置代理第一版
- **状态：** in_progress
- 执行的操作：
  - 恢复 `task_plan.md`、`findings.md`、`progress.md`、`AGENTS.md`，确认阶段 M3.5 目标是先做真实请求旁路记录，不拦截真实 KV。
  - 基线验证：`pytest tests -q` 输出 `38 passed in 2.23s`；未发现残留 `vllm serve` / `EngineCore` 进程；GPU 显存约 11MiB 已释放。
  - 使用测试先行新增 `tests/m3/test_http_sidecar.py`，覆盖 `/admit` 日志与指标、`/commit` 后前缀复用、无上游旁路返回、配置上游时剥离 `m3_control` 后转发。
  - 使用测试先行新增 `tests/m3/test_http_sidecar_cli.py`，覆盖从 Qwen2.5 校准参数生成 sidecar 配置，以及 CLI `--dry-run` 输出。
  - 新增 `benchmarks/m3/http_sidecar.py`，实现 FastAPI app、最小 Prometheus 指标、JSONL 决策日志和 OpenAI completions 前置代理。
  - 新增 `benchmarks/m3/http_sidecar_cli.py`，支持从 `simulator_params.yaml` 与 capacity CSV 推导 Qwen2.5 默认配置并启动 sidecar。
  - 新增 `docs/specs/m3_5_http_sidecar_proxy.md`，记录启动命令、请求示例、输出路径和当前限制。
- 遇到的问题：
  - FastAPI 不能为 `Response | dict[str, Any]` 自动生成响应模型；改为 `response_model=None` 并返回 `Any`。
  - httpx 默认读取系统 socks proxy，导致本机 localhost 转发需要 `socksio`；改为 `httpx.AsyncClient(trust_env=False)`。
- 当前验证：
  - `pytest tests/m3 -q` 输出 `12 passed in 1.22s`。
  - `python benchmarks/m3/http_sidecar_cli.py --simulator-params results/qwen25_14b_calibration/simulator_params.yaml --capacity-csv results/qwen25_14b_calibration/qwen25_capacity_probe.csv --decision-log results/m3_5_http_sidecar/decisions.jsonl --dry-run` 成功输出 Qwen2.5 配置：`kv_bytes_per_token=196608`、`hbm_capacity_tokens=64752`、`h2d_gbps=25.111298`、`storage_gbps=8.76583`。
  - `pytest tests -q` 输出 `44 passed in 3.41s`。
- 在线 smoke：
  - 启动 Qwen2.5-14B vLLM：`--max-model-len 2048 --enforce-eager --gpu-memory-utilization 0.90`，服务端日志显示 GPU KV cache size 为 `64,752 tokens`。
  - 启动 M3.5 sidecar 到 `127.0.0.1:8010`，上游为 `http://127.0.0.1:8000`。
  - 通过 sidecar 调用 `/v1/completions`，真实 vLLM 返回正常 completion，`usage.prompt_tokens=1`、`completion_tokens=1`。
  - sidecar `/metrics` 显示 `admission_decision_total{decision="ADMIT",reason="no_reusable_prefix"} 1`、`sync_ssd_miss_total 0`。
  - `results/m3_5_http_sidecar/online_smoke_decisions.jsonl` 记录 `online-smoke-1` 的 correctness key、token_count、decision、ready barrier。
  - 关闭 sidecar 与 vLLM 后确认没有残留 `vllm serve`、`EngineCore` 或 `http_sidecar_cli` 进程，GPU 显存回落到约 11MiB。
- connector 接入点评估：
  - 读取本机 vLLM `KVConnectorBase_V1`、`KVConnectorFactory`、`ExampleConnector`、`SimpleCPUOffloadConnector` 和 `MoRIIOConnector` 相关源码。
  - 确认 OpenAI completions/chat 协议支持 `kv_transfer_params`，并会进入 `Request.kv_transfer_params`。
  - 新增 `docs/specs/m3_6_vllm_connector_integration_notes.md`，定义下一阶段自定义 connector 的最小替换面和风险。
- 遇到的问题：
  - 关闭 vLLM 时出现 `resource_tracker: leaked semaphore` 警告；本次记录为关闭阶段资源追踪警告，未影响请求结果或资源清理。
- 最终验证：
  - `pytest tests -q` 输出 `44 passed in 3.84s`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现残留服务进程。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`，显存已回落。
  - `results/m3_5_http_sidecar/online_smoke_decisions.jsonl` 有 1 行，记录 `online-smoke-1` 的 `ADMIT/no_reusable_prefix` 和 `sync_ssd_miss_total=0`。

## 五问重启检查（阶段 M3.5 后）
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 M3.5 已完成：HTTP sidecar/前置代理、在线 Qwen2.5 smoke、connector 接入点评估 |
| 我要去哪里？ | 阶段 M3.6：实现 vLLM 自定义 KV connector 的 no-op smoke |
| 目标是什么？ | 让 sidecar 决策通过 `kv_transfer_params` 进入 vLLM scheduler/worker connector 路径 |
| 我学到了什么？ | HTTP sidecar 能稳定记录真实请求，但真正复用 KV 必须实现 `KVConnectorBase_V1` 并处理 block 对齐和 KV tensor layout |
| 我做了什么？ | 新增 HTTP sidecar、CLI、测试、M3.5 文档、M3.6 接入点评估，并完成真实 vLLM 转发 smoke |

### 阶段 M3.6：vLLM 自定义 KV connector no-op smoke
- **状态：** in_progress
- 执行的操作：
  - 重新读取 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md` 和 `docs/specs/m3_6_vllm_connector_integration_notes.md`，确认 M3.6 范围：只做 no-op connector，暂不做真实 KV tensor copy。
  - 阅读 `benchmarks/m3/http_sidecar.py`、M3 测试、vLLM `KVConnectorBase_V1`、`KVTransferConfig` 和 OpenAI completions protocol 中的 `kv_transfer_params` 流向。
  - 使用测试先行扩展 `tests/m3/test_http_sidecar.py`，要求 sidecar 能把 `m3_control` 翻译为 vLLM 原生 `kv_transfer_params`，并在转发时注入该字段。
  - 实现 `build_kv_transfer_params()` 和 `M3SidecarConfig.inject_kv_transfer_params`，并在 `http_sidecar_cli.py` 增加 `--inject-kv-transfer-params`。
  - 使用测试先行新增 `tests/m3/test_noop_connector.py`，覆盖 block 对齐、只接受 `ADMIT` 且不允许同步 SSD miss 的计划、scheduler metadata、worker no-op metadata，以及 vLLM `KVConnectorFactory` 动态加载。
  - 新增 `benchmarks/m3/noop_connector.py`，实现 `M3NoOpConnector`、`M3NoOpConnectorMetadata`、`M3NoOpConnectorWorkerMetadata`。
  - 更新 `docs/specs/m3_5_http_sidecar_proxy.md` 和 `docs/specs/m3_6_vllm_connector_integration_notes.md`，补充 `--inject-kv-transfer-params`、`--kv-transfer-config` 示例和当前限制。
- 遇到的问题：
  - `build_kv_transfer_params` 初始不存在，测试红灯后补实现。
  - CLI 初始不支持 `--inject-kv-transfer-params`，测试红灯后补参数并传入配置。
  - no-op connector 初版误调用不存在的实例方法 `_is_usable_plan`，修正为模块函数。
- 当前验证：
  - `pytest tests/m3/test_http_sidecar.py -q` 输出 `6 passed in 1.60s`。
  - `pytest tests/m3/test_noop_connector.py -q` 输出 `6 passed, 2 warnings in 3.21s`。
  - `pytest tests/m3/test_http_sidecar_cli.py -q` 输出 `2 passed in 1.03s`。
  - `pytest tests/m3 -q` 输出 `20 passed, 2 warnings in 6.31s`。
  - `pytest tests -q` 输出 `52 passed, 2 warnings in 5.95s`。
  - sidecar dry-run 带 `--inject-kv-transfer-params` 成功输出配置，其中 `inject_kv_transfer_params=true`。
- 真实在线 smoke：
  - 以 `PYTHONPATH=/root/KV` 启动 Qwen2.5-14B vLLM，附带 `--kv-transfer-config '{"kv_connector":"M3NoOpConnector","kv_role":"kv_both","kv_connector_module_path":"benchmarks.m3.noop_connector","kv_load_failure_policy":"fail"}'`。
  - vLLM 日志显示 `Creating v1 connector with name: M3NoOpConnector`，并成功启动到 `http://127.0.0.1:8000`。
  - 启动 sidecar 到 `127.0.0.1:8010`，带 `--inject-kv-transfer-params`。
  - 通过 sidecar 发送短 `/v1/completions` 请求，vLLM 返回正常 completion，响应包含 `kv_transfer_params: {"m3_noop_connector": {"saved_block_ids": [1]}}`。
  - sidecar 指标显示 `admission_decision_total{decision="ADMIT",reason="no_reusable_prefix"} 1` 和 `sync_ssd_miss_total 0`。
  - `results/m3_6_noop_connector/online_smoke_decisions.jsonl` 记录了 `m3-6-online-smoke-1` 的 sidecar 决策。
  - 关闭 sidecar 与 vLLM 后，未发现残留服务进程，GPU 显存回落到约 11MiB。
- 仍需注意：
  - vLLM 关闭时仍出现一次 `resource_tracker: leaked semaphore` 警告；这与上一阶段相同，暂记为关闭阶段资源追踪警告。
- 最终验证：
  - `pytest tests -q` 输出 `52 passed, 2 warnings in 5.86s`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现残留服务进程。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`，显存已回落。
  - `results/m3_6_noop_connector/online_smoke_decisions.jsonl` 有 1 行，记录 `m3-6-online-smoke-1` 的 `ADMIT/no_reusable_prefix` 和 `sync_ssd_miss_total=0`。

## 五问重启检查（阶段 M3.6 后）
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 M3.6 已完成：sidecar 注入 `kv_transfer_params`，vLLM 成功加载 no-op connector，真实短请求进入 connector 返回路径 |
| 我要去哪里？ | 阶段 M3.7：真实 KV tensor copy 原型 |
| 目标是什么？ | 从 no-op metadata 链路升级为真实 KV extract/load，同时继续保持 correctness key 和 sync SSD miss=0 约束 |
| 我学到了什么？ | vLLM 0.19.0 的 `kv_connector_module_path` 可以加载项目内 connector；OpenAI 请求的 `kv_transfer_params` 能穿透到 connector；真实 tensor copy 风险尚未处理 |
| 我做了什么？ | 实现 `M3NoOpConnector`、sidecar 注入开关、相关测试和真实 Qwen2.5 在线 connector smoke |

### 阶段 M3.7：真实 KV tensor copy 原型
- **状态：** complete
- 执行的操作：
  - 重新读取 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`、M3.6 文档和 vLLM `ExampleConnector` / `KVConnectorBase_V1` 源码。
  - 使用测试先行新增 `tests/m3/test_tensor_store.py`，覆盖块对齐、分页 KV 抽取、复制回写、正确性键不匹配、布局不匹配。
  - 新增 `benchmarks/m3/tensor_store.py`，实现本地块级 KV 张量存储、清单文件、slot mapping、extract/load 工具函数。
  - 扩展 `benchmarks/m3/noop_connector.py`：在配置 `m3_tensor_store_path` 时启用本地张量存储；metadata 携带 operation、correctness key、prefix id、token ids 和 block ids；worker 侧可保存和加载每层 KV。
  - 扩展 `benchmarks/m3/http_sidecar.py`：`kv_transfer_params` 增加 `store_prefix_id` 与 `store_token_end`。
  - 新增 `docs/specs/m3_7_tensor_copy_prototype.md`，说明当前能力、限制和下一步在线冒烟顺序。
- 遇到的问题：
  - 新增数据类字段时一度出现默认值顺序错误，修正字段顺序后通过。
  - 初版把 store 请求放在 `request_finished()` 后安排，发现不符合 vLLM 生命周期；修正为 `build_connector_meta()` 同时生成 store/load metadata。
  - 补丁合并时 `_load_request_layers()` 缩进错误，按行号定位并修正。
- 当前验证：
  - `pytest tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py -q` 输出 `13 passed, 2 warnings in 3.41s`。
  - `pytest tests/m3/test_http_sidecar.py::test_build_kv_transfer_params_carries_sidecar_plan tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py -q` 输出 `14 passed, 2 warnings in 4.27s`。
  - `pytest tests/m3/test_noop_connector.py::test_connector_saves_then_loads_tensor_blocks -q` 输出 `1 passed, 2 warnings in 3.81s`。
  - 补充 token ids manifest 与前缀子集加载校验后，最终 `pytest tests -q` 输出 `64 passed, 2 warnings in 6.94s`。
- 在线 smoke：
  - 启动 Qwen2.5-14B vLLM：`--max-model-len 512 --enforce-eager`，启用 `M3NoOpConnector`，并配置 `m3_tensor_store_path=results/m3_7_tensor_copy/tensor_store`、`m3_kv_layout=NHD`。
  - 启动 sidecar 到 `127.0.0.1:8010`，上游为 `http://127.0.0.1:8000`，带 `--inject-kv-transfer-params`。
  - 第一轮请求 `m3-7-online-store-1` 正常返回，connector 在 `results/m3_7_tensor_copy/tensor_store/session-a/` 写出 manifest 和 48 层 safetensors。
  - 手动调用 `/commit` 将 `session-a` 的 64 tokens 标记为可复用。
  - 第二轮请求 `m3-7-online-load-1` 正常返回，sidecar 决策为 `ADMIT/required_kv_ready_before_decode`，`reuse_tokens=64`，`delta_prefill_tokens=48`，`sync_ssd_miss_allowed=false`。
  - `results/m3_7_tensor_copy/tensor_store/session-a/manifest.json` 最终记录 `token_end=112`、`layers=48`、`layout=NHD`、`token_ids_len=112`。
  - `results/m3_7_tensor_copy/online_smoke_decisions.jsonl` 有 2 行。
- 收尾验证：
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现残留服务进程。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
- 当前限制：
  - 当前只证明真实服务路径可以保存和加载短前缀 KV；还没有证明 TTFT 稳定收益。
  - sidecar commit 仍是手动调用，下一阶段需要与真实 manifest 自动联动。

### 阶段 M3.8：真实复用路径稳定化与小矩阵验证
- **状态：** in_progress
- 执行的操作：
  - 重新读取 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`、M3.7 连接器、张量存储、HTTP sidecar 和相关测试。
  - 使用测试先行新增连接器事件日志测试，要求保存/加载写入结构化 JSONL，并维护保存/加载成功、失败、令牌数计数。
  - 使用测试先行新增前置代理自动提交测试，要求上游 vLLM 响应中的 connector manifest 能自动进入控制面，第二轮无需手动 `/commit` 即可复用。
  - 使用测试先行新增 `tests/m3/test_reuse_smoke_matrix.py`，覆盖 16/64/128/256 小矩阵 dry-run、CSV/report 输出和连接器事件日志耗时汇总。
  - 扩展 `benchmarks/m3/noop_connector.py`：新增 `m3_event_log_path`，输出 `store_layer` / `load_request` 事件，统计 `store_layer_ok_total`、`load_request_ok_total` 等指标。
  - 扩展 `benchmarks/m3/http_sidecar.py`：转发上游响应后解析 `kv_transfer_params.m3_noop_connector.manifest`，使用 sidecar 请求编号自动提交 manifest，并记录 `proxy_auto_commit` 日志。
  - 新增 `benchmarks/m3/run_reuse_smoke_matrix.py`，默认展开 `16/64/128/256` tokens 前缀复用小矩阵，支持 dry-run 与在线请求入口。
  - 新增 `docs/specs/m3_8_reuse_stabilization.md`，说明结构化日志、自动提交和小矩阵运行方式。
- 当前验证：
  - 新测试红灯确认：连接器初始不支持 `event_log_path`，前置代理初始无法自动复用，上述矩阵脚本初始不存在。
  - 修复后局部测试：连接器事件日志测试 `2 passed`，前置代理自动提交测试 `1 passed`，小矩阵测试 `3 passed`。
  - `pytest tests/m3 -q` 输出 `37 passed, 2 warnings in 6.06s`。
  - `python benchmarks/m3/run_reuse_smoke_matrix.py --dry-run --result-dir /tmp/m3_8_matrix_dry --prefix-tokens 16 64 --output-tokens 1` 输出 `status=DRY_RUN`，并生成 CSV 与报告。
- 追加修正：
  - 首次在线矩阵把第二轮请求做成“全提示词均由外部 KV 覆盖”，触发 vLLM scheduler `assert num_new_tokens > 0`。根因是 vLLM 外部 KV 命中后仍要求本轮至少有新增提示词令牌进入调度。
  - 将矩阵语义修正为“保存前缀，再用同一前缀加 16 个新增后缀复用”，并补测试保证第二轮真实 prompt 长度等于 `prefix_tokens + suffix_tokens`。
  - 进一步给矩阵 CSV 增加 `external_load_observed` 字段，避免把 HTTP 200 误读为真实外部 KV 加载成功。
- 在线验证：
  - 启动 Qwen2.5-14B vLLM，启用 `M3NoOpConnector`、`m3_tensor_store_path=results/m3_8_reuse_matrix_online/tensor_store`、`m3_event_log_path=results/m3_8_reuse_matrix_online/connector_events.jsonl`。
  - 启动 sidecar，带 `--inject-kv-transfer-params`，决策日志写入 `results/m3_8_reuse_matrix_online/online_decisions.jsonl`。
  - `prefix_tokens=16, suffix_tokens=16, output_tokens=1` 在线矩阵返回 `status=OK`，两轮 HTTP 状态均为 200。
  - `prefix_tokens=64, suffix_tokens=16, output_tokens=1` 在线矩阵返回 `status=OK`，两轮 HTTP 状态均为 200。
  - sidecar 日志显示第二轮均为 `ADMIT/required_kv_ready_before_decode`，并发生 `proxy_auto_commit`。
  - connector 事件日志显示 `m3-8-16` 和 `m3-8-64` 各有 48 层保存事件；manifest 分别记录 48 层 safetensors。
  - 重要限制：本次在线矩阵没有观测到 `load_request` 事件，说明请求成功很可能被 vLLM 内置前缀缓存覆盖，不能把本次在线结果解释为“外部 KV 加载已完成”。下一步需要构造绕开/关闭 vLLM 内置前缀缓存的对照，或用不同请求形态强制进入 connector load 路径。
- 收尾验证：
  - `pytest tests -q` 输出 `71 passed, 2 warnings in 6.15s`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现残留服务进程。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
  - vLLM 关闭时仍出现一次 `resource_tracker` leaked semaphore warning，与 M3.5/M3.6 现象一致。
- 尚未执行：
  - 尚未完成能观测到真实 connector `load_request` 的在线小矩阵。
  - 尚未做输出一致性 smoke。

### 阶段 M3.8 续：重启 vLLM 后强制观测外部加载
- **状态：** complete
- 执行的操作：
  - 使用测试先行扩展 `tests/m3/test_reuse_smoke_matrix.py`，要求小矩阵脚本支持 `phase=store` 和 `phase=reuse` 两个分阶段模式。
  - 扩展 `benchmarks/m3/run_reuse_smoke_matrix.py`：新增 `MatrixConfig.phase` 与 CLI 参数 `--phase {both,store,reuse}`，CSV 增加 `phase` 字段。
  - 执行重启 vLLM 对照实验：先启动 vLLM 和 sidecar，运行 `phase=store` 保存 `m3-8-64`；然后只关闭并重启 vLLM，保留 sidecar 控制面和 tensor store，再运行 `phase=reuse`。
- 在线结果：
  - 保存阶段：`results/m3_8_restart_load_probe/store_phase/reuse_smoke_matrix.csv`，`phase=store`，HTTP 200，manifest 存在，`m3-8-64` 写出 48 层 safetensors。
  - 复用阶段：`results/m3_8_restart_load_probe/reuse_phase/reuse_smoke_matrix.csv`，`phase=reuse`，HTTP 200，`external_load_observed=yes`。
  - `results/m3_8_restart_load_probe/connector_events.jsonl` 统计：`load_request ok` 1 次，耗时约 24.635ms；`store_layer ok` 96 次。
  - sidecar 日志显示复用阶段为 `ADMIT/required_kv_ready_before_decode`，`reuse_tokens=64`，`delta_prefill_tokens=16`，并再次自动提交 manifest。
- 结论：
  - 重启 vLLM 后，本地内置前缀缓存被清空，请求真实进入 connector 外部 KV 加载路径。
  - M3.8 现在已证明：本地文件 KV tensor store、sidecar 自动提交、vLLM 重启后的外部 KV load 可以在短前缀场景闭环。
- 收尾验证：
  - `pytest tests -q` 输出 `73 passed, 2 warnings in 6.62s`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现残留服务进程。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
  - vLLM 关闭时仍出现一次 `resource_tracker` leaked semaphore warning，与之前阶段一致。
- 尚未执行：
  - 尚未跑完整 16/64/128/256 重启对照矩阵。
  - 尚未做输出一致性 smoke。

### 阶段 M3.8 续：完整重启矩阵与输出一致性冒烟
- **状态：** complete
- 执行的操作：
  - 恢复 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md` 上下文，并确认当前阶段为 M3.8 真实复用路径稳定化。
  - 运行前检查无残留 vLLM/EngineCore/sidecar 进程，GPU 显存为 `11MiB used / 45606MiB free`。
  - 启动 Qwen2.5-14B vLLM，启用 `M3NoOpConnector`，张量存储和事件日志指向 `results/m3_8_full_restart_matrix/`。
  - 启动 HTTP sidecar，保持 `--inject-kv-transfer-params`，决策日志写入 `results/m3_8_full_restart_matrix/online_decisions.jsonl`。
  - 运行完整保存阶段：`prefix_tokens=16/64/128/256`、`suffix_tokens=16`、`output_tokens=1`，4 行均 OK。
  - 只关闭并重启 vLLM，保留 sidecar 控制面和 tensor store，然后运行完整复用阶段，4 行均 OK。
  - 新增 `benchmarks/m3/summarize_restart_matrix.py` 和测试 `tests/m3/test_restart_matrix_summary.py`，汇总保存/复用阶段结果。
  - 新增 `benchmarks/m3/run_reuse_consistency_smoke.py` 和测试 `tests/m3/test_reuse_consistency_smoke.py`，支持 baseline/reuse 分阶段输出一致性冒烟。
  - 修复一致性脚本直接执行时找不到 `benchmarks` 包的问题，按仓库根路径注入 `sys.path`，并增加 CLI `--help` 回归测试。
  - 修正一致性冒烟的事件统计：只统计复用请求之后新增的 connector 事件，避免历史 `load_request` 造成误报。
- 完整矩阵结果：
  - 保存阶段 CSV：`results/m3_8_full_restart_matrix/store_phase/reuse_smoke_matrix.csv`。
  - 复用阶段 CSV：`results/m3_8_full_restart_matrix/reuse_phase/reuse_smoke_matrix.csv`。
  - 汇总报告：`results/m3_8_full_restart_matrix/restart_matrix_report.md`。
  - `restart_matrix_summary.json`：`store_rows=4`、`reuse_rows=4`、`ok_store_rows=4`、`ok_reuse_rows=4`、`external_load_rows=4`、`all_external_load_observed=true`。
  - 复用阶段加载耗时：16 tokens 约 35.397ms，64 tokens 约 45.081ms，128 tokens 约 50.360ms，256 tokens 约 63.692ms。
  - 复用阶段端到端耗时：16 tokens 约 429.867ms，64 tokens 约 273.306ms，128 tokens 约 283.603ms，256 tokens 约 339.172ms。
- 输出一致性 smoke：
  - 先跑 baseline 阶段，普通路径返回文本 `" cache"`，HTTP 200，耗时约 272.865ms。
  - 再重启 vLLM，跑 reuse 阶段，复用路径返回文本 `" cache"`，HTTP 200，耗时约 342.210ms。
  - 本次复用新增观测到 `load_request ok` 1 次，加载耗时约 32.618ms。
  - `consistency_smoke.json` 记录 `status=OK`、`texts_match=true`、`external_load_observed=true`。
- 遇到的问题：
  - vLLM 退出时仍出现 `resource_tracker` leaked semaphore warning；与 M3.5/M3.6/M3.7/M3.8 之前相同，记录为关闭阶段资源追踪警告。
  - 一致性脚本初版连续跑 baseline/reuse 时可能被 vLLM 内置前缀缓存污染；改为 `--phase baseline` 和 `--phase reuse` 两阶段，并在两阶段之间重启 vLLM。
- 当前验证：
  - 新增测试局部验证：`pytest tests/m3/test_restart_matrix_summary.py tests/m3/test_reuse_consistency_smoke.py -q` 输出 `8 passed`。
  - 服务清理检查：未发现残留 `vllm serve`、`EngineCore` 或 `http_sidecar_cli` 进程。
  - GPU 显存清理检查：`nvidia-smi` 输出 `11, 45606`。
- 当前判断：
  - M3.8 已完成真实短前缀外部 KV 保存、自动提交、重启后外部加载、小矩阵覆盖和输出一致性冒烟。
  - 下一步应进入 DRAM/NVMe 分层迁移原型前的设计确认：把本地 tensor store 从“只保存文件”扩展成可表达 HBM/DRAM/NVMe 状态、异步迁移和准入前预取的执行模型。

### 阶段 M3.9：DRAM/NVMe 分层迁移原型
- **状态：** in_progress
- 执行的操作：
  - 恢复 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`，确认 M3.8 已完成，下一步进入 DRAM/NVMe 分层迁移原型。
  - 阅读 `benchmarks/m3/tensor_store.py`、`benchmarks/m3/noop_connector.py`、`benchmarks/m3/control_plane.py` 及对应测试，确认第一版不改 vLLM 内核，只扩展本地张量存储与 connector 事件语义。
  - 使用测试先行扩展 `tests/m3/test_tensor_store.py`：要求 manifest 记录 `tier/ready`，降级到 NVMe 后直接 `load_layer()` 抛出 `TierNotReady`，显式 `prefetch_to_dram()` 后才能加载。
  - 扩展 `benchmarks/m3/tensor_store.py`：新增 `TierNotReady`、`KVBlockManifest.tier`、`KVBlockManifest.ready`、`demote_to_nvme()`、`prefetch_to_dram()`、`read_migration_events()`。
  - 使用测试先行扩展 `tests/m3/test_noop_connector.py`：要求 connector 在 prefix 仍位于 NVMe 时记录 `load_request error` 并暴露 load error block ids；预取到 DRAM 后可正常加载。
  - 新增 `benchmarks/m3/run_tier_migration_smoke.py` 和 `tests/m3/test_tier_migration_smoke.py`，用于离线执行 `DRAM -> NVME -> DRAM` 状态循环并生成 CSV/JSON/报告。
  - 新增 `docs/specs/m3_9_tier_migration_prototype.md`，明确当前是元数据级迁移原型，不是真实异步 I/O 或 3FS 接入。
- 冒烟结果：
  - 命令：`python benchmarks/m3/run_tier_migration_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_tier_migration_smoke --prefix-ids m3-8-16 m3-8-64 m3-8-128 m3-8-256`
  - 输出：`results/m3_9_tier_migration_smoke/tier_migration_smoke.csv`、`tier_migration_summary.json`、`tier_migration_report.md`。
  - 结果：4 个前缀均完成 `DRAM -> NVME -> DRAM`，`status=OK`。
  - 当前记录 token 数：`m3-8-16=32`、`m3-8-64=80`、`m3-8-128=144`、`m3-8-256=272`。
- 遇到的问题：
  - 新增测试初始无法导入 `TierNotReady`，按 TDD 预期补实现。
  - 迁移冒烟脚本初始不存在，按 TDD 预期补实现。
  - CSV 字段顺序初始和测试预期不一致，调整为 `prefix_id,initial_tier,demoted_tier,prefetched_tier,...`，方便人工扫读。
- 当前验证：
  - `pytest tests/m3/test_tensor_store.py -q` 输出 `9 passed`。
  - `pytest tests/m3/test_tier_migration_smoke.py tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py -q` 输出 `24 passed, 2 warnings`。
- 当前限制：
  - M3.9 当前只改变 manifest 状态和事件，不做真实文件搬运到另一个设备。
  - sidecar 的 prefetch plan 还没有自动调用 `prefetch_to_dram()`。
  - 尚未引入带宽队列、deadline miss 统计和异步后台迁移。

### 阶段 M3.9 续：sidecar 预取绑定
- **状态：** in_progress
- 执行的操作：
  - 恢复 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md` 与 M3.9 交接上下文，确认当前缺口是把 sidecar 的延迟/预取决策接到 `KVTensorStore.prefetch_to_dram()`。
  - 先运行相关回归：`pytest tests/m3/test_control_plane.py tests/m3/test_http_sidecar.py tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py -q`，结果 `38 passed, 2 warnings`。
  - 测试先行扩展 `tests/m3/test_http_sidecar_cli.py`：要求 CLI 支持 `--tensor-store-path` 与 `--tensor-store-block-size`。初次运行失败，原因是参数未定义。
  - 更新 `benchmarks/m3/http_sidecar_cli.py`，将 tensor store 路径和块大小传入 `M3SidecarConfig`，并在 dry-run JSON 中输出路径。
  - 新增 `benchmarks/m3/run_sidecar_prefetch_smoke.py` 与 `tests/m3/test_sidecar_prefetch_smoke.py`，验证第一次准入延迟、sidecar 触发 `prefetch_to_dram`、第二次同一前缀准入成功。
  - 用真实 M3.8 张量仓库运行 smoke：
    `python benchmarks/m3/run_sidecar_prefetch_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_sidecar_prefetch_smoke --prefix-id m3-8-64`
- 冒烟结果：
  - 输出目录：`results/m3_9_sidecar_prefetch_smoke/`。
  - `sidecar_prefetch_summary.json` 记录 `status=OK`。
  - 第一次准入：`DELAY/required_kv_not_ready_before_decode`。
  - 第二次准入：`ADMIT/required_kv_ready_before_decode`。
  - 最终 manifest 状态：`DRAM` 且 `ready=true`。
  - `decisions.jsonl` 中包含 `source=prefetch_to_dram` 记录。
- 遇到的问题：
  - `http_sidecar_cli.py` 初始无法识别 `--tensor-store-path`，补充参数后 `pytest tests/m3/test_http_sidecar_cli.py -q` 输出 `2 passed`。
  - `run_sidecar_prefetch_smoke.py` 初始不存在，按测试补实现后 `pytest tests/m3/test_sidecar_prefetch_smoke.py -q` 输出 `2 passed`。
- 当前限制：
  - sidecar 预取仍是元数据级状态切换，不是真实 NVMe/3FS 异步 I/O。
  - 当前请求不会被同步等待到可执行；它仍然返回 `DELAY`，预取结果只让后续请求变为 `ADMIT`。
  - 尚未实现带宽队列、deadline miss 统计、预取完成时间预测和设备级 cold tier。
- 收尾验证：
  - `pytest tests/m3/test_control_plane.py tests/m3/test_http_sidecar.py tests/m3/test_http_sidecar_cli.py tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py tests/m3/test_tier_migration_smoke.py tests/m3/test_sidecar_prefetch_smoke.py -q` 输出 `43 passed, 2 warnings`。
  - `pytest tests -q` 输出 `90 passed, 2 warnings`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有本次检查命令自身。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。

### 阶段 M3.9 续：带宽与 deadline 感知迁移队列
- **状态：** in_progress
- 执行的操作：
  - 恢复 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`，确认 M3.9 剩余任务是“增加带宽和 deadline 感知的迁移队列”。
  - 测试先行新增 `tests/m3/test_prefetch_queue.py`，要求小前缀能在 deadline 内完成并标记 DRAM，低带宽大前缀必须返回 `DEADLINE_MISS` 且保持 `NVME/ready=false`。
  - 新增 `benchmarks/m3/prefetch_queue.py`，实现确定性 `PrefetchQueue` 和 `PrefetchResult`，按 `tokens * kv_bytes_per_token / storage_gbps` 估算迁移时间，写出 `prefetch_queue_events.jsonl`。
  - 测试先行扩展 `tests/m3/test_http_sidecar.py`，证明 sidecar 在 deadline miss 时不能把前缀误标 ready，第二次请求仍然 `DELAY`。
  - 更新 `benchmarks/m3/http_sidecar.py`，让 `_maybe_prefetch_to_dram()` 通过 `PrefetchQueue.submit()` 调度；只有 `COMPLETED` 才调用 `control_plane.mark_prefetched()`。
  - 扩展 `benchmarks/m3/run_sidecar_prefetch_smoke.py` 和 `tests/m3/test_sidecar_prefetch_smoke.py`，输出 `prefetch_status`、`prefetch_deadline_miss`、`estimated_ready_ms`、`deadline_ms`；CLI 新增 `--storage-gbps`。
- 真实烟测：
  - 正常完成命令：`python benchmarks/m3/run_sidecar_prefetch_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_sidecar_prefetch_queue_smoke/ok --prefix-id m3-8-64`
  - 正常完成结果：`status=OK`，第一次 `DELAY`，`prefetch_status=COMPLETED`，预计完成约 `1.787ms`，第二次 `ADMIT`。
  - deadline miss 命令：`python benchmarks/m3/run_sidecar_prefetch_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_sidecar_prefetch_queue_smoke/deadline_miss --prefix-id m3-8-256 --storage-gbps 0.000001 --request-token-count 288`
  - deadline miss 结果：`status=DEADLINE_MISS`，第一次 `DELAY`，预计完成约 `53477376ms`，第二次仍为 `DELAY`，最终 manifest 为 `NVME/ready=false`。
- 遇到的问题：
  - sidecar deadline miss 测试初始失败：第二次请求变成 `ADMIT`，说明原实现仍直接调用 `prefetch_to_dram()`。修复为通过 `PrefetchQueue` 判定后再迁移。
  - 预取烟测 CLI 初始无法传 `--storage-gbps`，补参数后仍因 `DEADLINE_MISS` 返回码为 1 导致测试失败。修复为 `OK` 与 `DEADLINE_MISS` 都返回 0，因为 deadline miss 是有效边界结果。
- 当前验证：
  - `pytest tests/m3/test_prefetch_queue.py -q` 输出 `2 passed`。
  - `pytest tests/m3/test_prefetch_queue.py tests/m3/test_http_sidecar.py tests/m3/test_sidecar_prefetch_smoke.py -q` 输出 `14 passed`。
  - `pytest tests/m3/test_prefetch_queue.py tests/m3/test_http_sidecar.py tests/m3/test_sidecar_prefetch_smoke.py tests/m3/test_http_sidecar_cli.py tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py tests/m3/test_tier_migration_smoke.py -q` 输出 `41 passed, 2 warnings`。
- 当前限制：
  - `PrefetchQueue` 仍是确定性元数据队列，不是真实后台异步 I/O。
  - 当前估算只覆盖 NVMe 到 DRAM 的 storage 带宽，不建模 H2D、3FS、真实 tail latency、多流并发或抢占。
  - 下一步应做后台队列、多请求排队压力测试，并把 deadline miss、队列深度和预计完成时间接入 `/metrics`。
- 收尾验证：
  - `pytest tests -q` 输出 `95 passed, 2 warnings`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有本次检查命令自身。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
  - `results/m3_9_sidecar_prefetch_queue_smoke/ok/sidecar_prefetch_smoke.csv` 记录 `COMPLETED`、`estimated_ready_ms=1.787`、第二次 `ADMIT`。
  - `results/m3_9_sidecar_prefetch_queue_smoke/deadline_miss/sidecar_prefetch_smoke.csv` 记录 `DEADLINE_MISS`、`estimated_ready_ms=53477376.0`、第二次仍为 `DELAY`。

### 阶段 M3.9 续：队列指标与多请求排队烟测
- **状态：** in_progress
- 执行的操作：
  - 恢复 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`，确认下一步是多请求排队压力和 `/metrics` 指标接入。
  - 测试先行扩展 `tests/m3/test_prefetch_queue.py`，要求 `PrefetchQueue.stats()` 能统计请求数、完成数、deadline miss、累计字节、最大队列深度和最近预计完成时间。
  - 更新 `benchmarks/m3/prefetch_queue.py`，新增统计累计字段和 `stats()` 方法。
  - 测试先行扩展 `tests/m3/test_http_sidecar.py`，要求 sidecar `/metrics` 暴露 `prefetch_queue_*` 指标。
  - 更新 `benchmarks/m3/http_sidecar.py`，把 `PrefetchQueue.stats()` 追加到 Prometheus 文本输出。
  - 新增 `benchmarks/m3/run_prefetch_queue_smoke.py` 和 `tests/m3/test_prefetch_queue_smoke.py`，用于直接对多个 prefix 提交队列并输出 CSV、JSON、报告和队列事件。
- 真实烟测：
  - 命令：`python benchmarks/m3/run_prefetch_queue_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_prefetch_queue_smoke --prefix-ids m3-8-16 m3-8-64 m3-8-128 m3-8-256 --storage-gbps 0.002 --deadline-ms 12000`
  - 输出：`results/m3_9_prefetch_queue_smoke/prefetch_queue_smoke.csv`、`prefetch_queue_summary.json`、`prefetch_queue_report.md`、`prefetch_queue_events.jsonl`。
  - 结果：4 个请求，2 个 `COMPLETED`，2 个 `DEADLINE_MISS`，累计字节 `103809024`，最大队列深度 `1`，最近预计完成时间 `37748.736ms`。
- 遇到的问题：
  - 初次实现 `stats()` 时 `read_events()` 缩进错误导致 `IndentationError`，修正缩进后队列测试通过。
  - `/metrics` 测试初始带宽设为 `0.001GB/s`，导致 64 token 小前缀也超过 12 秒，无法形成“一条完成、一条 miss”的测试场景；调整为 `0.002GB/s`。
  - 多请求队列烟测测试初始带宽过低，128 token 第二条请求因排队后超过 deadline；调整为 `0.004GB/s` 以构造两个小请求完成、1M 请求 miss 的单元测试。
- 当前验证：
  - `pytest tests/m3/test_prefetch_queue.py -q` 输出 `3 passed`。
  - `pytest tests/m3/test_http_sidecar.py::test_sidecar_metrics_include_prefetch_queue_stats -q` 输出 `1 passed`。
  - `pytest tests/m3/test_prefetch_queue.py tests/m3/test_http_sidecar.py::test_sidecar_metrics_include_prefetch_queue_stats tests/m3/test_prefetch_queue_smoke.py -q` 输出 `6 passed`。
  - `pytest tests/m3/test_prefetch_queue.py tests/m3/test_prefetch_queue_smoke.py tests/m3/test_http_sidecar.py tests/m3/test_sidecar_prefetch_smoke.py tests/m3/test_http_sidecar_cli.py tests/m3/test_tensor_store.py tests/m3/test_noop_connector.py tests/m3/test_tier_migration_smoke.py -q` 输出 `45 passed, 2 warnings`。
- 当前限制：
  - 多请求烟测直接提交队列，不会先判断 prefix 是否已经是 `DRAM/ready`；它验证的是排队估算，不是完整准入 residency hit。
  - 队列仍是确定性虚拟时间队列，不是后台异步线程。
  - 下一步应区分“已驻留 DRAM 命中”和“需要 NVMe 新预取”的队列入口，并再推进真实后台迁移。
- 收尾验证：
  - `pytest tests -q` 输出 `99 passed, 2 warnings`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有本次检查命令自身。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
  - `results/m3_9_prefetch_queue_smoke/prefetch_queue_summary.json` 记录 `requests=4`、`completed=2`、`deadline_miss=2`、`bytes_total=103809024`。

### 阶段 M3.9 续：驻留感知队列入口
- **状态：** in_progress
- 执行的操作：
  - 恢复 `task_plan.md`、`findings.md`、`progress.md`、`AGENTS.md`，确认下一步是区分“已驻留 DRAM 命中”和“需要从 NVMe 新预取”的队列入口。
  - 测试先行扩展 `tests/m3/test_http_sidecar.py`，要求本地 tensor store 已经 `DRAM/ready` 且 correctness key 匹配时，sidecar 同步控制面并让当前请求直接 `ADMIT`，不进入预取队列。
  - 增加 correctness key 保护测试：tensor store manifest 若 correctness key 不匹配，不能被 `PrefetchQueue` 预取成 ready，只能记录错误并保持请求 `DELAY`。
  - 更新 `benchmarks/m3/http_sidecar.py`，新增 residency hit 检查、`residency_hit` 事件、`residency_hit_total`、`prefetch_queued_total`、`prefetch_deadline_miss_total` 指标。
  - 新增 `benchmarks/m3/run_residency_aware_prefetch_smoke.py` 和 `tests/m3/test_residency_aware_prefetch_smoke.py`，输出 `RESIDENCY_HIT`、`PREFETCH_COMPLETED`、`DEADLINE_MISS` 三类状态。
- 真实烟测：
  - 命令：`python benchmarks/m3/run_residency_aware_prefetch_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_residency_aware_prefetch_smoke --prefix-ids m3-8-16 m3-8-64 m3-8-128 m3-8-256 --storage-gbps 0.002 --deadline-ms 12000`
  - 输出：`results/m3_9_residency_aware_prefetch_smoke/residency_aware_prefetch_smoke.csv`、`residency_aware_prefetch_summary.json`、`residency_aware_prefetch_report.md`、`prefetch_queue_events.jsonl`。
  - 结果：4 个前缀中 3 个 `RESIDENCY_HIT`，不进入队列；`m3-8-256` 为 `DEADLINE_MISS`，预计完成 `26738.688ms`，队列请求数为 1。
- 遇到的问题：
  - 驻留命中初版只是在第一次 `DELAY` 后同步控制面，导致第二次才 `ADMIT`；修正为在记录准入结果前检查 tensor store，并重新准入当前请求。
  - correctness key 不匹配的 tensor manifest 初版会被队列预取成 ready；修正为队列提交前校验 correctness key 和覆盖范围，不匹配只记录错误。
  - `run_residency_aware_prefetch_smoke.py` 初版对 `RESIDENCY_HIT` 行写死 Qwen2.5 的 `196608 bytes/token`；补测试后改为使用 CLI/函数传入的 `kv_bytes_per_token`。
- 当前验证：
  - `pytest tests/m3/test_http_sidecar.py::test_sidecar_treats_dram_ready_tensor_store_as_residency_hit -q` 先失败，确认缺少当前请求直接准入语义。
  - `pytest tests/m3/test_http_sidecar.py::test_sidecar_does_not_prefetch_tensor_manifest_with_wrong_correctness_key -q` 先失败，确认 correctness key 不匹配仍可能被错误预取。
  - `pytest tests/m3/test_residency_aware_prefetch_smoke.py -q` 输出 `3 passed`。
  - `pytest tests/m3/test_http_sidecar.py tests/m3/test_residency_aware_prefetch_smoke.py tests/m3/test_prefetch_queue.py tests/m3/test_prefetch_queue_smoke.py tests/m3/test_sidecar_prefetch_smoke.py -q` 输出 `25 passed`。
- 收尾验证：
  - `pytest tests -q` 输出 `105 passed, 2 warnings in 16.38s`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有本次检查命令自身。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
  - `results/m3_9_residency_aware_prefetch_smoke/residency_aware_prefetch_summary.json` 记录 `rows=4`、`residency_hits=3`、`queue_requests=1`、`deadline_miss=1`。

### 阶段 M3.9 续：后台异步预取队列
- **状态：** in_progress
- 执行的操作：
  - 恢复 `task_plan.md`、`findings.md`、`progress.md` 和 `docs/specs/m3_9_tier_migration_prototype.md`，确认当前剩余任务是增加后台异步队列。
  - 测试先行扩展 `tests/m3/test_prefetch_queue.py`，要求异步队列提交后 manifest 仍保持 `NVME/ready=false`，只有调用 `advance_ready()` 后才变为 `DRAM/ready=true`。
  - 在 `benchmarks/m3/prefetch_queue.py` 中新增 `AsyncPrefetchQueue`，保留同步 `PrefetchQueue` 既有行为。异步队列返回 `QUEUED`、`ALREADY_QUEUED`、`COMPLETED`、`DEADLINE_MISS`。
  - 测试先行扩展 `tests/m3/test_http_sidecar.py`，要求 sidecar 在 `async_prefetch=True` 时第一次请求只入队、推进前第二次仍延迟、显式推进后第三次才准入。
  - 更新 `benchmarks/m3/http_sidecar.py`，新增 `M3SidecarConfig.async_prefetch`、`/prefetch/advance` 端点和 `advance_prefetch_from_payload()`，用于确定性推进后台队列。
  - 新增 `benchmarks/m3/run_async_prefetch_smoke.py` 与 `tests/m3/test_async_prefetch_smoke.py`，生成异步预取 CSV、JSON、报告和 decisions 日志。
  - 更新 `benchmarks/m3/http_sidecar_cli.py` 和 `tests/m3/test_http_sidecar_cli.py`，新增 `--async-prefetch`。
- 真实烟测：
  - 命令：`python benchmarks/m3/run_async_prefetch_smoke.py --tensor-store results/m3_8_full_restart_matrix/tensor_store --result-dir results/m3_9_async_prefetch_smoke --prefix-id m3-8-64 --storage-gbps 8.8 --advance-ms 12000`
  - 输出：`results/m3_9_async_prefetch_smoke/async_prefetch_smoke.csv`、`async_prefetch_summary.json`、`async_prefetch_report.md`、`decisions.jsonl`。
  - 结果：`status=OK`；第一次请求 `DELAY` 并记录 `QUEUED`；推进前第二次请求仍为 `DELAY` 且记录 `ALREADY_QUEUED`；推进后完成 `COMPLETED`；第三次请求 `ADMIT`；最终 manifest 为 `DRAM/ready=true`。
- 遇到的问题：
  - 异步去重逻辑初版误放进同步 `PrefetchQueue.submit()`，导致同步队列测试报 `AttributeError: _pending_for_prefix`。修正为只在 `AsyncPrefetchQueue.submit()` 中去重。
  - sidecar 初版在调用 `submit()` 前就增加 `prefetch_queued_total`，导致重复请求 `ALREADY_QUEUED` 也被计入新入队。修正为只有 `COMPLETED` 或 `QUEUED` 才增加该指标。
  - 缺少 tensor manifest 的旧测试把该情况当成“fallback 入队”。按更准确语义改为预取错误，不增加队列请求或 `prefetch_queued_total`。
- 当前验证：
  - `pytest tests/m3/test_prefetch_queue.py -q` 输出 `5 passed`。
  - `pytest tests/m3/test_http_sidecar_cli.py tests/m3/test_http_sidecar.py::test_sidecar_async_prefetch_delays_until_background_queue_advances tests/m3/test_async_prefetch_smoke.py -q` 输出 `5 passed`。
  - `pytest tests/m3/test_prefetch_queue.py tests/m3/test_http_sidecar.py tests/m3/test_http_sidecar_cli.py tests/m3/test_async_prefetch_smoke.py tests/m3/test_residency_aware_prefetch_smoke.py tests/m3/test_prefetch_queue_smoke.py tests/m3/test_sidecar_prefetch_smoke.py -q` 输出 `32 passed`。
- 收尾验证：
  - `pytest tests -q` 输出 `110 passed, 2 warnings in 18.14s`。
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有本次检查命令自身。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
  - `results/m3_9_async_prefetch_smoke/async_prefetch_summary.json` 记录 `status=OK`、`queued_status=QUEUED`、`completed_status=COMPLETED`、`third_decision_after_advance=ADMIT`。

### 阶段 M3.9 续：真实在线小矩阵接入驻留/预取指标
- **状态：** complete
- 执行的操作：
  - 恢复 `task_plan.md`、`findings.md`、`progress.md`、`AGENTS.md` 和 M3.9 文档，确认剩余任务是将真实在线小矩阵接入 residency / prefetch 指标。
  - 测试先行扩展 `tests/m3/test_reuse_smoke_matrix.py`，要求小矩阵 CSV 干跑也固定输出驻留、预取和队列指标列。
  - 在 `benchmarks/m3/run_reuse_smoke_matrix.py` 中新增 `parse_sidecar_metrics()`、`fetch_sidecar_metrics()` 和 `summarize_sidecar_metric_deltas()`，每行在线请求前后读取 sidecar `/metrics` 并计算 delta。
  - 扩展小矩阵报告，每行显示驻留命中变化、预取入队变化和预取截止错过变化。
  - 继续测试先行新增 sidecar 决策日志解析：`--sidecar-decision-log` 可从本行新增的 `proxy` 记录中提取 `decision`、`reason`、`ready_barrier_all_ready` 和 `sync_ssd_miss_total`。
  - 测试先行扩展 `tests/m3/test_http_sidecar.py`，要求 `/commit` 能记录 `ready=false` 的容量层前缀；修正 `benchmarks/m3/http_sidecar.py`，将 `ready` 字段透传到控制面。
  - 更新 `docs/specs/m3_9_tier_migration_prototype.md`，记录小矩阵指标列已经接入。
- 干跑验证：
  - 命令：`python benchmarks/m3/run_reuse_smoke_matrix.py --dry-run --result-dir /tmp/m3_9_reuse_metrics_dry --prefix-tokens 16 64 --output-tokens 1`
  - 结果：`status=DRY_RUN`，CSV 表头包含 `residency_hit_delta`、`prefetch_queued_delta`、`prefetch_queue_pending`、`decision`、`reason`、`ready_barrier_all_ready`。
- 真实在线样本：
  - 启动 Qwen2.5-14B vLLM，`--max-model-len 512 --enforce-eager`，启用 `M3NoOpConnector`，张量仓库使用 `results/m3_8_full_restart_matrix/tensor_store`，连接器事件日志写入 `results/m3_9_reuse_metrics_online/connector_events.jsonl`。
  - 启动 sidecar，启用 `--inject-kv-transfer-params` 和 `--tensor-store-path results/m3_8_full_restart_matrix/tensor_store`。
  - 为构造边界样本，先把 `m3-8-16` 保持 `DRAM/ready=true`，把 `m3-8-256` 显式降级为 `NVME/ready=false`，再通过 `/admit` + `/commit ready=false tier=SSD` 将两个前缀注册进新 sidecar 控制面。
  - 命令：`python benchmarks/m3/run_reuse_smoke_matrix.py --result-dir results/m3_9_reuse_metrics_online/fresh_sidecar/reuse_phase --phase reuse --prefix-tokens 16 256 --suffix-tokens 16 --output-tokens 1 --connector-event-log results/m3_9_reuse_metrics_online/connector_events.jsonl --sidecar-decision-log results/m3_9_reuse_metrics_online/fresh_sidecar/decisions.jsonl`
  - 输出：`results/m3_9_reuse_metrics_online/fresh_sidecar/reuse_phase/reuse_smoke_matrix.csv` 和 `report.md`。
  - 结果：`m3-8-16` 行 `ADMIT/required_kv_ready_before_decode`，`ready_barrier_all_ready=true`，`residency_hit_delta=1`，`external_load_observed=yes`；`m3-8-256` 行 `DELAY/required_kv_not_ready_before_decode`，`ready_barrier_all_ready=false`，`prefetch_queued_delta=1`，`prefetch_queue_completed_delta=1`，`prefetch_queue_bytes_delta=53477376`，`sync_ssd_miss_total=0`。
- 遇到的问题：
  - 本机没有 `jq`，查看 manifest 时改用 Python 读取 JSON；不影响代码路径。
  - 初次在线重跑后 `m3-8-256` 已被同步队列预取回 `DRAM/ready=true`，导致第二次样本不再出现入队 delta；重启 sidecar 并显式 `demote_to_nvme()` 后重新生成干净样本。
  - vLLM 关闭时仍出现 `resource_tracker` leaked semaphore warning，与前几轮一致，记录为关闭阶段警告。
- 当前验证：
  - `pytest tests/m3/test_reuse_smoke_matrix.py -q` 输出 `10 passed`。
  - `pytest tests/m3/test_reuse_smoke_matrix.py tests/m3/test_http_sidecar.py tests/m3/test_prefetch_queue.py -q` 输出 `30 passed`。
  - `pytest tests -q` 输出 `115 passed, 2 warnings in 17.95s`。
  - 进程清理检查：未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有检查命令自身。
  - GPU 显存清理检查：`nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
- 当前限制：
  - 本轮在线矩阵接入的是 sidecar 指标与元数据级预取队列，不是真实目录/设备级冷层搬运。
  - 同步 `PrefetchQueue` 仍会在本次 `DELAY` 决策后立即把 manifest 标记为 `DRAM/ready=true`，用于后续请求；当前请求仍必须按 sidecar 决策列解释为延迟。
  - 下一阶段应实现真实 cold-tier 目录/设备迁移和后台执行器，再接入 3FS。

### 阶段 M3.10-A：真实 cold-tier executor 与目录级迁移
- **状态：** in_progress
- 执行的操作：
  - 恢复 `task_plan.md`、`findings.md`、`progress.md`、`docs/research/kv_anti_caching_design.md`，确认本轮目标是 M3.10-A：把 M3.9 元数据迁移替换为真实文件/目录级 cold-tier 迁移。
  - 测试先行扩展 `tests/m3/test_tensor_store.py`，要求 `demote_to_cold_object()` 移动真实 `.safetensors` layer 文件到 cold root，manifest 写入 `cold_uri/object_id/offset_table/checksum/size_bytes`，并在 restore 前拒绝 connector 同步加载。
  - 先运行 `pytest tests/m3/test_tensor_store.py::test_tensor_store_moves_layer_files_to_cold_object_and_restores -q`，得到预期红灯：`AttributeError: 'KVTensorStore' object has no attribute 'demote_to_cold_object'`。
  - 更新 `benchmarks/m3/tensor_store.py`：新增 `ChecksumMismatch`、cold object manifest 字段、`demote_to_cold_object()`、`restore_from_cold_object()`、checksum/URI/helper；保留 `demote_to_nvme()` 与 `prefetch_to_dram()` 兼容入口。
  - 测试先行扩展 `tests/m3/test_prefetch_queue.py`，要求同步队列和异步队列在完成/推进时执行真实 cold object restore，而不是只改 manifest。
  - 更新 `benchmarks/m3/prefetch_queue.py`：`PrefetchResult` 增加 `actual_bytes`、`executor_elapsed_ms`、`object_id`、`cold_uri`、`checksum_status`，从 tensor store 的迁移事件中回填真实执行器元数据。
  - 升级 `benchmarks/m3/run_tier_migration_smoke.py`：新增 `--cold-root`，默认结果目录改为 `results/m3_10_cold_tier_migration_smoke`，CSV/报告输出 cold URI、真实字节、demote/restore 耗时和 checksum 状态。
  - 更新 `benchmarks/m3/run_prefetch_queue_smoke.py` 与 `benchmarks/m3/run_async_prefetch_smoke.py`，让 smoke 输出真实执行器字段。
  - 更新 `task_plan.md` 与 `findings.md`，记录 M3.10-A 已完成的真实目录迁移能力和仍未完成的生产级 3FS executor / 在线矩阵闭环。
- 真实 CLI smoke：
  - 命令：临时生成两个 32-token prefix 的 tensor store 后执行 `python benchmarks/m3/run_tier_migration_smoke.py --tensor-store <tmp>/tensor_store --result-dir <tmp>/result --cold-root <tmp>/cold --prefix-ids session-a session-b`。
  - 结果：`status=OK`、`prefixes=2`、`bytes_total=672`；CSV 中两行均为 `DRAM -> NVME -> DRAM`，并包含 `cold_uri=file://...`、`size_bytes=336`、`checksum_status=ok`、demote/restore elapsed ms。
- 当前验证：
  - `pytest tests/m3/test_tensor_store.py::test_tensor_store_moves_layer_files_to_cold_object_and_restores -q` 先失败，确认测试覆盖缺失 API；实现后输出 `1 passed`。
  - `pytest tests/m3/test_tensor_store.py -q` 输出 `10 passed`。
  - `pytest tests/m3/test_prefetch_queue.py -q` 输出 `7 passed`。
  - `pytest tests/m3/test_tier_migration_smoke.py -q` 输出 `1 passed`。
  - `pytest tests/m3/test_tensor_store.py tests/m3/test_prefetch_queue.py tests/m3/test_tier_migration_smoke.py -q` 输出 `18 passed`。
  - `pytest tests/m3/test_http_sidecar.py tests/m3/test_sidecar_prefetch_smoke.py tests/m3/test_prefetch_queue_smoke.py tests/m3/test_async_prefetch_smoke.py tests/m3/test_residency_aware_prefetch_smoke.py -q` 输出 `26 passed`。
  - `python -m compileall benchmarks/m3 tests/m3` 输出编译通过。
  - `pytest tests/m3 -q` 输出 `86 passed, 2 warnings in 17.31s`。
- 当前限制：
  - 第一版 cold object 仍是目录对象：每个 layer 保持独立 `.safetensors` 文件，`offset_table` 是逻辑 offset 表，不是单大文件 packed object。
  - 确定性后台 executor 由 `AsyncPrefetchQueue.advance_ready()` 驱动，尚不是生产线程池或真实时间异步 I/O。
  - 尚未接入 3FS mount/RDMA/io_uring/GDS，也没有真实 queue depth、p95/p99 tail latency 聚合。
  - 尚未把真实 cold-tier 路径重新跑进在线 vLLM 小矩阵；当前完成的是本地 tensor store、sidecar 队列和离线 smoke 闭环。
- 收尾验证：
  - `pytest tests/m3/test_prefetch_queue.py -q` 输出 `7 passed`。
  - `python -m compileall benchmarks/m3 tests/m3` 输出编译通过。
  - `pytest tests -q` 输出 `118 passed, 2 warnings in 16.64s`。
  - 进程检查 `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有检查命令自身。
  - GPU 显存检查 `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。

### 阶段 M3.10-B：真实 cold-tier 在线小矩阵与端到端样本
- **状态：** complete
- 执行的操作：
  - 继续 M3.10 规划，把真实 cold-tier executor 接入 `benchmarks/m3/run_reuse_smoke_matrix.py` 的在线路径。
  - 测试先行新增控制面回归：同一 prefix/range 先提交 `DRAM/ready=true`，再提交 `SSD/ready=false` 后，后续 admit 必须返回 `DELAY`。
  - 更新 `benchmarks/m3/control_plane.py`：`KVRange` 增加内部 `epoch`，`KVManifest.lookup()` 按最新 epoch 选择覆盖 range；`mark_prefetched()` 完成后追加更新 epoch 的 `DRAM/ready=true`。
  - 扩展 `run_reuse_smoke_matrix.py`：新增 `mark_sidecar_prefix_cold()`；cold mode 在 store 后执行真实 cold object demote，再调用 sidecar `/commit` 标记 `SSD/ready=false`；复用前用 `/admit` 做 cold probe，随后 `/prefetch/advance`，最后才向 `/v1/completions` 发送复用请求。
  - 扩展 CSV 字段：新增 `cold_commit_status_code`、`cold_commit_status`，并保留 cold object、probe、advance、restore 和 sidecar metrics 字段。
  - 补充 FastAPI 级测试，验证真实 layer 文件 demote 后 `/admit -> /prefetch/advance -> /admit` 形成 `DELAY -> COMPLETED -> ADMIT`。
  - 修复 `run_reuse_smoke_matrix.py` 直接按路径执行时找不到 `benchmarks` 包的问题，按 repo root 注入 `sys.path`，并增加 CLI `--help` 回归测试。
- 真实在线样本 1：同进程 cold-tier matrix
  - 启动 Qwen2.5-14B vLLM，`--max-model-len 512 --enforce-eager`，启用 `M3NoOpConnector`，tensor store 为 `results/m3_10_cold_tier_online/tensor_store`，事件日志为 `results/m3_10_cold_tier_online/connector_events.jsonl`。
  - 启动 sidecar，启用 `--inject-kv-transfer-params`、`--tensor-store-path results/m3_10_cold_tier_online/tensor_store`、`--async-prefetch`。
  - 命令：`python benchmarks/m3/run_reuse_smoke_matrix.py --result-dir results/m3_10_cold_tier_online/reuse_phase --phase both --prefix-tokens 64 --suffix-tokens 16 --output-tokens 1 --connector-event-log results/m3_10_cold_tier_online/connector_events.jsonl --sidecar-decision-log results/m3_10_cold_tier_online/decisions.jsonl --tensor-store results/m3_10_cold_tier_online/tensor_store --cold-root results/m3_10_cold_tier_online/cold_objects --cold-tier-restore --advance-ms 12000`
  - 结果：`status=OK`；`cold_probe_decision=DELAY`；`cold_restore_status=COMPLETED`；`cold_restore_actual_bytes=12587136`；`cold_restore_checksum_status=ok`；`sync_ssd_miss_total=0`；`prefetch_queue_completed_delta=1`。
  - 重要限制：同进程最终 `external_load_observed=no`、`load_events=0`，说明 vLLM 内置 prefix cache 仍可能遮蔽外部 connector load。
- 真实在线样本 2：重启 vLLM 后端到端 connector load
  - 输出目录：`results/m3_10_cold_tier_online/restart_probe/`。
  - 先运行 `phase=store` 保存 `m3-8-64`：`store_events=48`，HTTP 200。
  - 手动执行真实 `demote_prefix_to_cold_object()` 并调用 sidecar `/commit` 标记 `m3-8-64` 为 `SSD/ready=false`；`cold_demote_summary.json` 记录 cold object 大小 `12587136` bytes、checksum `sha256:e1ddd51c...`、热目录 layer 文件已移除。
  - 只重启 vLLM，保留 sidecar 控制面和 tensor store，再运行 `phase=reuse --cold-tier-restore`。
  - 复用阶段 CSV：`results/m3_10_cold_tier_online/restart_probe/reuse_phase/reuse_smoke_matrix.csv`。
  - 结果：`status=OK`；`cold_probe_decision=DELAY`；`cold_restore_status=COMPLETED`；`cold_restore_actual_bytes=12587136`；`cold_restore_checksum_status=ok`；`decision=ADMIT`；`ready_barrier_all_ready=true`；`sync_ssd_miss_total=0`；`external_load_observed=yes`。
  - connector 事件：`results/m3_10_cold_tier_online/restart_probe/connector_events.jsonl` 中出现 `load_request ok` 1 次，`layers=48`，`tokens=64`，`elapsed_ms=37.866`。
- 遇到的问题：
  - cold demote 后控制面初版仍可能因旧 `DRAM/ready=true` range 返回 `ADMIT`；通过 `KVRange.epoch` 修正为最新状态覆盖旧状态。
  - 同进程样本会被 vLLM 内置 prefix cache 遮蔽外部 load；采用保存后重启 vLLM、保留 sidecar 控制面的对照样本，最终观测到 connector `load_request ok`。
  - vLLM 关闭时仍出现 `resource_tracker` leaked semaphore warning，与前几轮一致，记录为关闭阶段警告；进程和显存均已清理。
- 当前验证：
  - `pytest tests/m3/test_control_plane.py tests/m3/test_reuse_smoke_matrix.py tests/m3/test_http_sidecar.py tests/m3/test_prefetch_queue.py -q` 输出 `46 passed in 5.82s`。
  - `python -m compileall benchmarks/m3 tests/m3` 输出编译通过。
  - `pytest tests/m3 -q` 输出 `93 passed, 2 warnings in 21.35s`。
  - `pytest tests -q` 输出 `125 passed, 2 warnings in 20.15s`。
  - 进程清理检查：未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有检查命令自身。
  - GPU 显存清理检查：`nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
- 当前限制：
  - 仍是本地目录 cold object，不是 3FS packed object 或 RDMA/GDS/io_uring executor。
  - 在线样本只覆盖 64-token 小前缀和单请求；下一步需要 16/64/128/256 完整 cold-tier 小矩阵，再扩大到 512-32K。
  - `summarize_connector_events()` 当前按全日志聚合事件，重启样本中复用阶段 `store_events=96` 包含保存阶段和最终复用后的再次保存；后续可按日志 offset 做 per-row delta。

### 阶段 M3.10-C：完整 cold-tier restart 小矩阵与 3FS adapter 起步
- **状态：** complete
- 宏观说明：
  - 当前系统已经形成 sidecar 控制面、tensor store、cold-tier executor/async queue 和 vLLM connector 的端到端闭环。
  - cold tier 仍被严格定义为容量层；decode critical path 不允许同步读 SSD/3FS。请求必须先经 sidecar `/admit` 看到 `DELAY`，再由 `/prefetch/advance` 异步恢复到 DRAM，最终 ready barrier 通过后才允许复用。
  - 3FS Adapter 的含义是抽象 cold-tier 存储后端，让本地 SSD 目录和 3FS mount/client 共享 demote、restore、checksum、URI、backend 记录语义。SSD 是设备/介质；3FS 是分布式共享文件系统层，通常可由 SSD/RDMA 等支撑。
- 执行的操作：
  - 沿用 `results/m3_10_cold_tier_online/restart_matrix/`，启动 Qwen2.5-14B vLLM store-phase 服务和 async sidecar。
  - 运行 `phase=store`，保存 16/64/128/256 四个 prefix。connector 事件共 `192` 个 `store_layer`，每个 prefix 48 层。
  - 对四个 prefix 执行真实 `demote_prefix_to_cold_object()` 并通过 sidecar `/commit` 标记 `SSD/ready=false`。`cold_demote_summary.json` 记录 hot files 均已移除，cold object 大小分别为 `3149952`、`12587136`、`25170048`、`50335872` bytes。
  - 只重启 vLLM，保留 sidecar 控制面和 tensor store，运行 `phase=reuse --cold-tier-restore` 完整矩阵。
  - 测试先行补 3FS adapter 入口：新增 `build_cold_tier_adapter()`，并把 `--cold-backend {local_posix,3fs_posix}` 接入 `run_reuse_smoke_matrix.py`、`run_tier_migration_smoke.py` 和 `http_sidecar_cli.py`。
  - `PrefetchQueue` / `AsyncPrefetchQueue` 新增 `cold_root`、`cold_backend`、`cold_adapter`，真实 cold object restore 使用 adapter；旧的纯元数据 fixture 保留 fallback。
- 真实矩阵结果：
  - CSV：`results/m3_10_cold_tier_online/restart_matrix/reuse_phase/reuse_smoke_matrix.csv`。
  - 16/64/128/256 四行均 `status=OK`，`cold_probe_decision=DELAY`，`cold_restore_status=COMPLETED`，`cold_restore_checksum_status=ok`，`decision=ADMIT`，`ready_barrier_all_ready=true`，`sync_ssd_miss_total=0`，`external_load_observed=yes`，`load_events=1`。
  - connector 事件：`results/m3_10_cold_tier_online/restart_matrix/connector_events.jsonl` 中每个 prefix 均有 1 次 `load_request ok`，layers=48；load elapsed ms 分别约 35.859、37.809、36.516、104.736。
  - sidecar metrics 结束状态：`prefetch_queued_total=4`，`prefetch_queue_completed_total=4`，`prefetch_queue_deadline_miss_total=0`，`sync_ssd_miss_total=0`。
- 3FS adapter smoke：
  - 命令：临时生成两个 prefix 后运行 `python benchmarks/m3/run_tier_migration_smoke.py --tensor-store <tmp>/tensor_store --result-dir results/m3_10_3fs_adapter_smoke --cold-root <tmp>/threefs_mount --cold-backend 3fs_posix --prefix-ids m3-8-16 m3-8-64`。
  - 结果：`results/m3_10_3fs_adapter_smoke/tier_migration_summary.json` 记录 `status=OK`、`cold_backend=3fs_posix`、`prefixes=2`、`bytes_total=672`。
- 遇到的问题：
  - 3FS adapter 初始只存在类，缺少统一 backend factory 和 CLI 配置入口。通过 `build_cold_tier_adapter()` 与 `--cold-backend` 修正。
  - `PrefetchQueue` 初始改成总是 `restore_from_cold_object()` 后破坏了旧的纯元数据测试；修正为 manifest 有 `cold_uri` 和 layers 时走真实 adapter restore，否则保持 `prefetch_to_dram()` metadata fallback。
  - vLLM 关闭时仍出现已有 `resource_tracker` leaked semaphore warning；服务与 GPU 清理正常。
- 当前验证：
  - `pytest tests/m3/test_tensor_store.py tests/m3/test_prefetch_queue.py tests/m3/test_reuse_smoke_matrix.py tests/m3/test_tier_migration_smoke.py tests/m3/test_http_sidecar_cli.py -q` 输出 `44 passed`。
  - `python -m compileall benchmarks/m3 tests/m3` 编译通过。
  - `pytest tests/m3/test_tensor_store.py tests/m3/test_prefetch_queue.py tests/m3/test_reuse_smoke_matrix.py tests/m3/test_control_plane.py tests/m3/test_http_sidecar.py tests/m3/test_http_sidecar_cli.py tests/m3/test_tier_migration_smoke.py -q` 输出 `67 passed`。
  - `pytest tests/m3 -q` 输出 `101 passed, 2 warnings in 20.87s`。
  - `pytest tests -q` 输出 `133 passed, 2 warnings in 21.74s`。
- 当前限制：
  - `3fs_posix` 仍是 3FS 挂载目录的 POSIX 语义 adapter，不是 3FS 原生 client，也没有 RDMA/GDS/io_uring、队列深度控制、tail latency 聚合或 packed cold object。
  - 完整在线矩阵仍是 16/64/128/256 小前缀，下一步需要扩到 512-32K，再做真实 3FS mount/生产 SSD 的带宽和 p95/p99 校准。

### 阶段 M3.10-D：Cold-Tier Adapter Benchmark 与对比报告
- **状态：** complete
- 执行的操作：
  - 测试先行新增 `tests/m3/test_cold_tier_adapter_bench.py`，固定 adapter benchmark 输出契约：CSV、summary JSON、Markdown report，字段包含 backend、bytes、demote/restore ms、MiB/s、checksum、hot files restored 和 p50/p95/p99。
  - 新增 `benchmarks/m3/run_cold_tier_adapter_bench.py`。该脚本生成合成 KV prefix/layer，通过现有 `KVTensorStore` 和 `build_cold_tier_adapter()` 执行真实 demote/restore，不引入新的存储语义。
  - 测试先行新增 `tests/m3/test_cold_tier_adapter_summary.py`，固定多个 adapter summary 的比较 CSV/Markdown 输出。
  - 新增 `benchmarks/m3/summarize_cold_tier_adapter_bench.py`，用于合并 local/3fs_posix benchmark summary。
- 真实 smoke：
  - local_posix 命令：`python benchmarks/m3/run_cold_tier_adapter_bench.py --result-dir results/m3_10_cold_tier_adapter_bench/local_posix --cold-root results/m3_10_cold_tier_adapter_bench/local_posix/cold --cold-backend local_posix --prefix-tokens 16 64 128 256 --repeats 3 --layers 4 --block-size 16 --kv-heads 2 --head-dim 8`。
  - 3fs_posix 命令：`python benchmarks/m3/run_cold_tier_adapter_bench.py --result-dir results/m3_10_cold_tier_adapter_bench/3fs_posix --cold-root results/m3_10_cold_tier_adapter_bench/3fs_posix/threefs_mount --cold-backend 3fs_posix --prefix-tokens 16 64 128 256 --repeats 3 --layers 4 --block-size 16 --kv-heads 2 --head-dim 8`。
  - 汇总命令：`python benchmarks/m3/summarize_cold_tier_adapter_bench.py --summary results/m3_10_cold_tier_adapter_bench/local_posix/adapter_bench_summary.json results/m3_10_cold_tier_adapter_bench/3fs_posix/adapter_bench_summary.json --result-dir results/m3_10_cold_tier_adapter_bench/summary`。
  - 输出：`results/m3_10_cold_tier_adapter_bench/local_posix/`、`results/m3_10_cold_tier_adapter_bench/3fs_posix/`、`results/m3_10_cold_tier_adapter_bench/summary/adapter_bench_comparison.md`。
  - local_posix summary：rows=12、bytes_total=716736、demote p50/p95/p99=1.149/5.341/8.837ms、restore p50/p95/p99=1.518/17.018/31.566ms。
  - 3fs_posix summary：rows=12、bytes_total=716736、demote p50/p95/p99=0.988/1.376/1.396ms、restore p50/p95/p99=1.25/1.752/1.778ms。
- 重要说明：
  - 当前 3fs_posix smoke 仍使用本地目录模拟 3FS mount，仅验证 adapter 接口和报告格式。真实 3FS 性能必须在生产/实验 3FS 挂载点上重新运行同一脚本。
  - 当前合成 KV 对象很小，适合 CI/smoke；后续需增加 512-32K token、真实 Qwen2.5 layer 文件、queue depth 和并发 restore。
- 当前验证：
  - `pytest tests/m3/test_cold_tier_adapter_bench.py -q` 输出 `3 passed`。
  - `pytest tests/m3/test_cold_tier_adapter_summary.py -q` 输出 `2 passed`。
  - `pytest tests/m3/test_cold_tier_adapter_bench.py tests/m3/test_cold_tier_adapter_summary.py tests/m3/test_tensor_store.py tests/m3/test_prefetch_queue.py tests/m3/test_reuse_smoke_matrix.py tests/m3/test_tier_migration_smoke.py -q` 输出 `45 passed`。
  - `python -m compileall benchmarks/m3 tests/m3` 编译通过。
  - `pytest tests/m3 -q` 输出 `106 passed, 2 warnings in 22.56s`。
  - `pytest tests -q` 输出 `138 passed, 2 warnings in 24.04s`。

### 阶段 M3.10-E：Queue-Depth Adapter Benchmark 与 Qwen2.5 Tiny Profile
- **状态：** complete
- 时间：2026-05-26T05:37:18Z
- 执行的操作：
  - 扩展 `benchmarks/m3/run_cold_tier_adapter_bench.py`，新增 `--queue-depth` 和 `--profile {synthetic,qwen25_14b_tiny}`。
  - `qwen25_14b_tiny` profile 固定为 4 layers、8 kv heads、head_dim=128、bfloat16，用于保持 Qwen2.5-14B 的 KV 形状但降低本机 smoke 成本。
  - restore 阶段改为按 batch 使用 `ThreadPoolExecutor(max_workers=queue_depth)` 并发恢复，CSV 新增 `batch_id`、`queue_depth`、`restore_batch_ms`、`restore_effective_mib_per_s`。
  - summary / comparison report 新增 profile、queue depth、batch p50/p95/p99 和 effective restore bandwidth。
- 真实 smoke：
  - local_posix 命令：`python benchmarks/m3/run_cold_tier_adapter_bench.py --result-dir results/m3_10_cold_tier_adapter_bench_qd/local_posix_qd4 --cold-root results/m3_10_cold_tier_adapter_bench_qd/local_posix_qd4/cold --cold-backend local_posix --profile qwen25_14b_tiny --prefix-tokens 16 64 128 256 --repeats 2 --queue-depth 4`。
  - 3fs_posix 命令：`python benchmarks/m3/run_cold_tier_adapter_bench.py --result-dir results/m3_10_cold_tier_adapter_bench_qd/3fs_posix_qd4 --cold-root results/m3_10_cold_tier_adapter_bench_qd/3fs_posix_qd4/threefs_mount --cold-backend 3fs_posix --profile qwen25_14b_tiny --prefix-tokens 16 64 128 256 --repeats 2 --queue-depth 4`。
  - 汇总输出：`results/m3_10_cold_tier_adapter_bench_qd/summary/adapter_bench_comparison.md`。
  - local_posix summary：rows=8、bytes_total=15207168、restore p95=11.334ms、restore batch p95=13.584ms、effective restore p50=545.648MiB/s。
  - 3fs_posix summary：rows=8、bytes_total=15207168、restore p95=45.969ms、restore batch p95=68.175ms、effective restore p50=202.162MiB/s。
- 当前验证：
  - `pytest tests/m3/test_cold_tier_adapter_bench.py::test_cold_tier_adapter_bench_records_queue_depth_batches tests/m3/test_cold_tier_adapter_bench.py::test_cold_tier_adapter_bench_qwen25_profile_sets_shape_defaults tests/m3/test_cold_tier_adapter_bench.py::test_cold_tier_adapter_bench_cli_help_runs_from_repo_root -q` 输出 `3 passed`。
  - `pytest tests/m3/test_cold_tier_adapter_summary.py -q` 输出 `2 passed`。
  - `pytest tests/m3/test_cold_tier_adapter_bench.py tests/m3/test_cold_tier_adapter_summary.py -q` 输出 `7 passed`。
  - `python -m compileall benchmarks/m3 tests/m3` 编译通过。
  - `pytest tests/m3 -q` 输出 `108 passed, 2 warnings in 22.83s`。
  - `pytest tests -q` 输出 `140 passed, 2 warnings in 23.28s`。
- 收尾检查：
  - `ps -eo pid,cmd | rg 'vllm serve|EngineCore|http_sidecar_cli' || true` 未发现实际残留 vLLM、EngineCore 或 sidecar 服务进程；输出中只有检查命令自身。
  - `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits` 输出 `11, 45606`。
- 当前限制：
  - `3fs_posix_qd4` 仍使用本地目录模拟 3FS mount，只能验证 adapter 与报告契约，不能代表真实 3FS 集群性能。
  - 当前前缀规模仍为 16/64/128/256，小于论文/生产所需的 512-32K、128K、1M 分层压力区间。

### 研究阶段 R2 续：KV Anti-Caching 项目逻辑 PPT
- **状态：** complete
- 时间：2026-05-26T05:47:14Z
- 执行的操作：
  - 参考 `/root/MoE1/slides/moe_serving_report.tex` 的整体结构、Beamer 主题、配色和叙事顺序。
  - 在 `slides/kv_anti_caching_report.tex` 中梳理 KV Anti-Caching 项目逻辑：真正的问题、观察与动机、为什么重要、已有工作为什么没有解决、当前原型为什么还不够、目标、挑战、设计、系统闭环、请求生命周期和实验计划。
  - 明确问题表达：现有推理服务无法在精确 attention 语义下，把百万级历史 KV 同时变成容量上可保存、执行前可恢复、decode 时必定 ready、且 TTFT 满足 SLA 的服务状态；KV Anti-Caching 是方法，不是问题本身。
  - 将当前原型定位为“路径已走通但尚未证明 1M SLA”：已有真实文件迁移、restart 小矩阵、sidecar/connector 闭环和 qd4 adapter benchmark；缺口是 32K/128K/1M、真实 3FS、packed cold object、partial promote、多资源预算和强 baseline。
- 输出文件：
  - `slides/kv_anti_caching_report.tex`
  - `slides/kv_anti_caching_report.pdf`
- 验证：
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error kv_anti_caching_report.tex` 在 `slides/` 下编译成功，生成 15 页 PDF。
  - `pdfinfo slides/kv_anti_caching_report.pdf` 显示标题为 `面向超长上下文推理的 KV Anti-Caching`，页数为 15。
  - `pdftotext slides/kv_anti_caching_report.pdf -` 抽取文本确认核心页面内容可读。
- 备注：
  - 编译日志仍有当前 ctex/Fandol 字体环境下的 fontspec / missing character 警告，但未阻止 PDF 生成，文本抽取正常。

### 研究阶段 R2 续：PPT 主线修正为 Prefill + Decode 双阶段 readiness
- **状态：** complete
- 执行的操作：
  - 根据反馈修正 `slides/kv_anti_caching_report.tex`，不再把问题过度表述为 decode-only。
  - 将核心问题改为：现有推理服务无法在精确 attention 语义下，把百万级历史 KV 同时变成容量上可保存、Prefill 复用前可恢复、Decode 执行前必定 ready、且 TTFT 满足 SLA 的服务状态。
  - 将 motivation 改为两类失败：`Prefill failure` 表示 cold miss 导致等待 restore 或历史 KV 重算，放大 TTFT；`Decode failure` 表示执行期 cold miss 导致同步冷层 I/O，放大 ITL/TPOT 和尾延迟。
  - 将设计页和系统图中的 `Decode Ready Barrier` 改为 `Prefill / Decode Ready Barrier`，并在请求生命周期中增加 `Restore/Fallback` 与 `Two-Phase Ready Barrier`。
  - 将 takeaway 改为：Anti-Caching 的价值是把 cold miss 从 Prefill 重算和 Decode 同步等待中移出，而不是只保护 decode path。
- 验证：
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error kv_anti_caching_report.tex` 在 `slides/` 下编译成功，重新生成 15 页 PDF。
  - `pdfinfo slides/kv_anti_caching_report.pdf` 显示页数为 15，生成时间为 2026-05-26T06:26:49Z。
  - `pdftotext slides/kv_anti_caching_report.pdf - | rg -n "Prefill|Decode|重算|真正问题|核心价值|Two-Phase|ready"` 确认关键表述已进入 PDF。
- 备注：
  - 编译日志仍有当前 ctex/Fandol 字体环境下的 fontspec / missing character 警告；未发现 Overfull 盒子警告，PDF 正常生成。

### 阶段 M3.11-A：当前方法有效性复盘与 Baseline 判断
- **状态：** complete
- 时间：2026-05-26
- 执行的操作：
  - 按 `planning-with-files-zh` 恢复 `task_plan.md`、`findings.md`、`progress.md`，按 `context-engineering` 重新读取 `AGENTS.md` 和当前研究/实验上下文。
  - 复查 M2 Qwen2.5 sweep、M3.10 cold-tier restart 小矩阵、queue-depth adapter benchmark 和当前设计文档。
  - 重新梳理“方法有没有用”的证据边界：语义正确性、容量路径、SLA 性能收益、强 baseline 胜出四层分开判断。
  - 补查近期强相关 baseline：vLLM APC、LMCache、Mooncake、Tutti、DualPath、CacheFlow、KVDrive，并将 KVDrive 纳入后续 baseline 风险。
  - 新增有效性评估文档：`docs/research/kv_anti_caching_effectiveness_assessment.md`。
  - 更新 `findings.md`，记录当前最诚实结论：M3.10 已证明 correctness/control-plane invariant，但尚未证明 1M SLA、TTFT 增幅 <=20%、命中率 >=90% 或相对强 baseline 的性能优势。
  - 更新 `task_plan.md`，新增阶段 M3.11：Baseline Readiness Matrix 与有效性验证。
- 核心判断：
  - 当前方法有用：它把 cold-tier miss 从 Prefill 重算/Decode 同步 I/O 中移出，变成 admission 前可检测、可延迟、可恢复、可校验的状态。
  - 当前还不能声称性能胜出：真实在线矩阵只到 256 tokens，`3fs_posix` 仍是本地目录模拟，尚无 vLLM APC / LMCache / Mooncake / Tutti / DualPath / CacheFlow / KVDrive 同机对比。
  - 下一步应优先做 B0-B5 统一 benchmark：vLLM full prefill、APC hot/restart、DRAM-ready reuse、naive cold restore、current Anti-Caching；随后接强系统 baseline。
- 输出文件：
  - `docs/research/kv_anti_caching_effectiveness_assessment.md`
  - `findings.md`
  - `task_plan.md`

### 阶段 M3.11-B：Persistent KV Anti-Caching 主线收敛与优先级把关
- **状态：** complete
- 时间：2026-05-26
- 执行的操作：
  - 按 `context-engineering` 重新固化项目上下文规则，更新 `AGENTS.md` 的项目目标、不可妥协语义、KVDrive 竞争边界和当前优先级。
  - 按 `planning-with-files-zh` 更新 `task_plan.md`，把当前阶段从“继续 3FS adapter”调整为 M3.11 的 P0/P1/P2 优先级规划。
  - 将项目主线明确命名为 `Persistent KV Anti-Caching for Multi-Turn Long-Context Serving`。
  - 将 KVDrive 提升为主竞争 baseline：不能再把“多级 KV 管理”本身作为创新，必须聚焦多轮 persistent historical KV 的 exact、可验证、可准入复用。
  - 将真实 3FS executor 从当前唯一主线降为 P1：它仍重要，但必须服务于 P0 证据链，而不是替代 P0。
  - 更新 `findings.md`，记录 P0/P1/P2 任务划分与真实 3FS 降级原因。
- P0 必须优先任务：
  - B0-B5 baseline readiness matrix。
  - 512/2K/8K/16K/32K 在线矩阵。
  - Persistent session lineage + committed KV range 生命周期。
  - KV PrePass：枚举 required historical KV ranges。
  - Packed cold object / extent layout 原型。
- P1 后续补强任务：
  - 生产级 restore executor、真实 3FS/生产 SSD、partial promote、attention-informed hotness sketch、LMCache / DualPath / Tutti / CacheFlow / KVDrive-style baseline。
- 输出文件：
  - `AGENTS.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 明日工作计划：M3.11-C Baseline Matrix 起步与 Gradient 候选模型评审
- **状态：** planned
- 建议目标：
  - 不继续优先堆 3FS adapter 功能；明天的核心目标是启动 P0 证据链。
  - 先把 M3.11 benchmark schema 固定下来，再决定是否把 `gradientai/Llama-3-8B-Instruct-Gradient-1048k` 加入长上下文候选基座。
- 上午优先事项：
  1. 写 `M3.11 Baseline Readiness Matrix` 规格文档，固定 B0-B5 定义、CSV 字段、目录结构、判定标准。
  2. 把必须指标落成 schema：TTFT p50/p95/p99、ITL/TPOT、`TTFT_tiered / TTFT_full_prefill`、`TTFT_tiered / TTFT_dram_ready_reuse`、historical byte hit rate、deadline miss、sync cold miss、restore bytes/useful bytes。
  3. 明确第一轮只跑 Qwen2.5-14B 的 512/2K/8K/16K/32K，高复用 workload 先覆盖 `session_append` 和 `shared_prefix`，`low_locality` 作为负对照。
- 下午优先事项：
  1. 评审 Gradient 1048k 是否作为 M1.8/M3.11 候选模型引入：下载成本、license、vLLM 启动参数、32K/64K/128K capacity probe、KV bytes/token=131072、connector smoke 风险。
  2. 如果时间允许，先实现一个轻量模型候选评估文档，不立即下载大模型；除非确认磁盘、网络和时间都允许。
  3. 开始设计 B0-B5 runner 的最小接口，但先不要启动长时间 benchmark。
- 明天不要优先做：
  - 不先做真实 3FS executor。
  - 不先做 128K/1M 大实验。
  - 不先接 LMCache / DualPath / KVDrive-style 外部 baseline。
  - 不先改 attention kernel 或引入 sparse/approximate path。
- 明天结束时的交付物：
  - `docs/specs/m3_11_baseline_readiness_matrix.md`
  - 如完成模型评审：`docs/research/long_context_model_base_selection.md`
  - 更新 `task_plan.md` / `findings.md` / `progress.md`

### 阶段 M3.11-C：Baseline Readiness Matrix 规格落地
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 读取 `AGENTS.md`、`task_plan.md`、既有 benchmark spec，确认当前 P0 目标和不做事项。
  - 新增 `docs/specs/m3_11_baseline_readiness_matrix.md`，固定上午计划中的 M3.11 baseline readiness matrix。
  - 明确 B0-B5：full prefill、APC hot-cache、APC restart/cold-cache、external DRAM-ready reuse、naive cold restore、current Anti-Caching。
  - 固定第一轮范围：Qwen2.5-14B，512/2K/8K/16K/32K prefix，128/512/2K suffix，1/16 output tokens，先 concurrency=1。
  - 固定 workload：`session_append`、`shared_prefix`、`low_locality`、`mixed_short_long`；第一轮时间不足时优先前两者，`low_locality` 至少保留 smoke。
  - 固定 `baseline_matrix.csv`、`baseline_summary.csv` 字段，覆盖 TTFT/ITL/TPOT、reuse、historical byte hit rate、restore bytes/useful bytes、ready barrier、sync cold miss、external load events 和 TTFT ratios。
  - 固定推进/暂停门槛：B5 不要求赢 APC hot-cache，但必须在高复用 8K-32K 上相对 B0/B4 有收益，并保持 `sync_cold_miss_total=0` 与 external load 可观测。
  - 更新 `task_plan.md`，将 P0 benchmark schema 标记完成。
  - 更新 `findings.md`，记录 M3.11 spec 的关键决策和禁止发散事项。
- 输出文件：
  - `docs/specs/m3_11_baseline_readiness_matrix.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 阶段 M3.11-D：Baseline Readiness Matrix dry-run runner 与 summary 骨架
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 先新增 `tests/m3/test_baseline_readiness_matrix.py`，确认缺少 runner 时测试失败。
  - 新增 `benchmarks/m3/run_baseline_readiness_matrix.py`，实现 M3.11 dry-run runner：B0-B5、workload、prefix/suffix/output/concurrency/repeat 展开，固定 `baseline_matrix.csv` schema，生成 `env.json`、`run_config.yaml`、`raw/`、三类 jsonl 日志占位和 summary 目录。
  - 新增 `benchmarks/m3/summarize_baseline_readiness_matrix.py`，实现 summary 聚合、B5 TTFT ratio 回填、`baseline_summary.csv`、`workload_summary.csv` 和 `readiness_report.md`。
  - 运行默认 dry-run：`python benchmarks/m3/run_baseline_readiness_matrix.py --dry-run --result-dir results/m3_11_baseline_readiness`。
  - 生成默认矩阵 540 行：B0-B5 各 90 行；`session_append/shared_prefix/low_locality` 各 180 行；432 行 `DRY_RUN`，108 行 `ERROR_BOUNDARY`。
  - 注意：本阶段只完成 schema/runner/summary 骨架，不是实际在线 B0-B5 性能结论。
- 输出文件：
  - `benchmarks/m3/run_baseline_readiness_matrix.py`
  - `benchmarks/m3/summarize_baseline_readiness_matrix.py`
  - `tests/m3/test_baseline_readiness_matrix.py`
  - `results/m3_11_baseline_readiness/env.json`
  - `results/m3_11_baseline_readiness/run_config.yaml`
  - `results/m3_11_baseline_readiness/baseline_matrix.csv`
  - `results/m3_11_baseline_readiness/request_metrics.csv`
  - `results/m3_11_baseline_readiness/summary/baseline_summary.csv`
  - `results/m3_11_baseline_readiness/summary/workload_summary.csv`
  - `results/m3_11_baseline_readiness/summary/readiness_report.md`
- 验证：
  - `pytest tests/m3/test_baseline_readiness_matrix.py -q`：`6 passed in 0.18s`
  - `pytest tests/m3 -q`：`114 passed, 2 warnings in 24.21s`
  - `python benchmarks/m3/summarize_baseline_readiness_matrix.py --result-dir results/m3_11_baseline_readiness`：rows=540，dry_run_count=432，error_count=108，status=PARTIAL

### 阶段 M3.11-E：B0/B1/B2 原生 vLLM 在线路径接入
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 新增 B0/B1/B2 在线路径测试：验证 B0 full prefill、B1 warm+measure APC hot、B2 cold-cache 近似行都写入统一 schema；验证 boundary 行不触发 HTTP 请求。
  - 扩展 `benchmarks/m3/run_baseline_readiness_matrix.py`：新增 `CompletionTiming`、`OpenAICompletionClient`、`--base-url`、非 dry-run B0/B1/B2 执行路径、raw streaming JSON 记录和 `OK/PARTIAL` 产物状态码。
  - 启动 Qwen2.5-14B vLLM：`env VLLM_PLUGINS= NO_PROXY=127.0.0.1,localhost vllm serve /root/models/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.90 --disable-log-stats --disable-uvicorn-access-log --enforce-eager --enable-prefix-caching`。
  - 首次 512/128/1 smoke 出现 B0/B2 400；系统化调试定位为 prompt 单元带连字符，Qwen tokenizer 将 640 个空格词拆成 3584 tokens，超过 2048 smoke server。
  - 将在线 runner 的 prompt 单元改为 `full/cache/cold/tail` 等短 token，并重新跑真实 smoke。
  - 成功生成 `results/m3_11_native_vllm_online_smoke/`：B0/B1/B2 × `session_append/shared_prefix` × 512/128/1 共 6 行全部 `OK`。
  - 关闭 vLLM 后确认无残留 `vllm serve`/`EngineCore`/sidecar 进程，GPU 显存回落到 11MiB。
- 当前边界：
  - B2 仍是 cold/unique prefix 近似 APC cold-cache，不是严格 restart 后同 prefix 再测。
  - B3/B5 还未接入当前 M3.11 runner，下一步需要复用 M3.8/M3.10 sidecar + connector + cold-tier restart 路径。
  - 在线 smoke 只证明 runner 可以采集真实 TTFT/E2E，不代表 8K/16K/32K 或最终 SLA 结论。
- 输出文件：
  - `benchmarks/m3/run_baseline_readiness_matrix.py`
  - `tests/m3/test_baseline_readiness_matrix.py`
  - `results/m3_11_b0_b2_online_smoke/`
  - `results/m3_11_native_vllm_online_smoke/`
- 验证：
  - `pytest tests/m3/test_baseline_readiness_matrix.py -q`：`9 passed in 0.43s`
  - `pytest tests/m3 -q`：`117 passed, 2 warnings in 22.55s`

### 阶段 M3.11-F：严格 B2 restart/cold-cache orchestration
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 新增严格 B2 测试：验证 `strict_b2_restart=True` 时服务生命周期必须是 start/warm/stop/start/measure/stop，且 warm 与 measure 使用相同 prefix。
  - 首次定向测试按预期失败：`BaselineReadinessConfig.__init__()` 不支持 `strict_b2_restart`。
  - 扩展 `benchmarks/m3/run_baseline_readiness_matrix.py`：新增 `VLLMServiceManager` 协议、`LocalVLLMServiceManager`、`--strict-b2-restart`、`--serve-host`、`--serve-port`、`--serve-max-model-len`、`--startup-timeout-sec`、`--enable-prefix-caching` 等 CLI 参数。
  - 严格 B2 行为改为：启动 vLLM warm 同一 prefix，停止 vLLM，重启 vLLM 后测量同一 prefix+suffix；CSV 标记 `admission_reason=apc_restart_cold_cache`。
  - 真实 smoke 运行：Qwen2.5-14B、`--max-model-len 2048`、512 prefix + 128 suffix + 1 output、`session_append`，由 runner 托管两次 vLLM 生命周期。
  - smoke 完成后检查无残留 `vllm serve` / `EngineCore` / sidecar 进程，端口 8000 关闭，GPU 显存回落到 11MiB。
- 输出文件：
  - `benchmarks/m3/run_baseline_readiness_matrix.py`
  - `tests/m3/test_baseline_readiness_matrix.py`
  - `results/m3_11_b2_restart_online_smoke/`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 验证：
  - `pytest tests/m3/test_baseline_readiness_matrix.py -q`：`11 passed in 0.49s`
  - `pytest tests/m3 -q`：`119 passed, 2 warnings in 24.17s`
  - `python benchmarks/m3/run_baseline_readiness_matrix.py --phase b2_apc_restart --workload session_append --prefix-tokens 512 --suffix-tokens 128 --output-tokens 1 --result-dir results/m3_11_b2_restart_online_smoke --model /root/models/Qwen2.5-14B-Instruct --base-url http://127.0.0.1:8000 --model-context-limit 2048 --strict-b2-restart --serve-host 127.0.0.1 --serve-port 8000 --serve-max-model-len 2048 --gpu-memory-utilization 0.90 --enforce-eager --enable-prefix-caching`：`status=OK`, `rows=1`

### 阶段 M3.11-G：B3/B5 reuse bridge 接入统一 baseline schema
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 新增 B3/B5 bridge 测试：验证 reuse executor 返回的 `reuse_smoke_matrix.csv` 行能映射到 M3.11 `baseline_matrix.csv` 字段，包括 TTFT、historical byte hit、restore、ready、external load 和 sync cold miss。
  - 首次定向测试按预期失败：`run_baseline_readiness_matrix()` 不支持 `reuse_executor` 参数。
  - 扩展 `benchmarks/m3/run_baseline_readiness_matrix.py`：新增 `ReuseExecutor` 协议、`ReuseSmokeMatrixExecutor`、`ReuseCsvImportExecutor`。
  - B3/B5 在线模式现在不再返回 `unsupported_online_baseline`，而是通过 sidecar/connector reuse bridge 生成统一 schema 行。
  - 新增 CLI 参数：`--sidecar-url`、`--connector-event-log`、`--sidecar-decision-log`、`--tensor-store`、`--cold-root`、`--cold-backend`、`--cold-tier-advance-ms`、`--block-size`、`--import-reuse-csv`。
  - 生成 dry-run 样本 `results/m3_11_b3_b5_bridge_dry_run/`。
  - 使用历史 restart CSV 生成导入样本 `results/m3_11_b3_b5_imported_reuse_sample/`：B3 与 B5 均进入统一 M3.11 summary，B5 restore 字段可聚合。
  - 检查无残留 `vllm serve` / `EngineCore` / sidecar 进程，8000/8010 端口关闭，GPU 显存 11MiB。
- 输出文件：
  - `benchmarks/m3/run_baseline_readiness_matrix.py`
  - `tests/m3/test_baseline_readiness_matrix.py`
  - `results/m3_11_b3_b5_bridge_dry_run/`
  - `results/m3_11_b3_b5_imported_reuse_sample/`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 验证：
  - `pytest tests/m3/test_baseline_readiness_matrix.py -q`：`14 passed in 2.95s`
  - `pytest tests/m3 -q`：`122 passed, 2 warnings in 25.51s`
  - `python benchmarks/m3/run_baseline_readiness_matrix.py --baselines B3 B5 --workload session_append --prefix-tokens 64 --suffix-tokens 16 --output-tokens 1 --result-dir results/m3_11_b3_b5_imported_reuse_sample --model-context-limit 2048 --import-reuse-csv B3:64:16:1:results/m3_8_full_restart_matrix/reuse_phase/reuse_smoke_matrix.csv --import-reuse-csv B5:64:16:1:results/m3_10_cold_tier_online/restart_matrix/reuse_phase/reuse_smoke_matrix.csv`：`status=OK`, `rows=2`

### 阶段 M3.11-H：2K B0/B1/B2/B3/B5 真机同机矩阵
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 使用 Qwen2.5-14B、`--max-model-len 8192`、2048 prefix + 128 suffix + 1 output、concurrency=1 跑第一轮真实同机矩阵。
  - B0/B1 使用原生 vLLM + APC：`results/m3_11_2k_true_matrix/native_b0_b1/baseline_matrix.csv`。
  - B2 使用严格 restart/cold-cache orchestration：`results/m3_11_2k_true_matrix/native_b2_restart/baseline_matrix.csv`。
  - B3 使用 sidecar + connector + restart 后外部 DRAM-ready load：`results/m3_11_2k_true_matrix/b3_external/reuse_phase/reuse_smoke_matrix.csv`。
  - 发现首次 B5 store 没带 `--cold-tier-restore`，没有生成真实 cold object；保留原目录，另起 `results/m3_11_2k_true_matrix/b5_anti_caching_true/` 干净重跑。
  - B5 true 路径完成 store -> demote cold object -> sidecar mark cold/not-ready -> restart vLLM -> cold probe DELAY -> advance restore -> reuse ADMIT。
  - 将 B3/B5 reuse CSV 导入统一 schema，并合并 B0/B1/B2，输出 `results/m3_11_2k_true_matrix/final/baseline_matrix.csv` 与 `summary/readiness_report.md`。
- 关键结果：
  - B0 full prefill：TTFT `717.061ms`，reuse tokens `0`。
  - B1 APC hot-cache：TTFT `91.016ms`，reuse tokens `2048`。
  - B2 APC restart/cold-cache：TTFT `686.379ms`，reuse tokens `0`。
  - B3 external DRAM-ready reuse：TTFT `6055.039ms`，external load `yes`，load events `1`，sync cold miss `0`。
  - B5 Anti-Caching true cold-tier：TTFT `6590.921ms`，cold probe `DELAY`，restore `COMPLETED`，restore bytes `402657408`，checksum `ok`，external load `yes`，sync cold miss `0`。
  - B5/B0 TTFT ratio 为 `9.191576`；B5/B3 为 `1.088502`；B5/B1 为 `72.414971`。
- 结论：
  - 本轮证明了 B5 的 exact cold-tier Anti-Caching 语义：冷历史不能直接准入，必须先恢复并校验，ready barrier 成立后才进入 vLLM external load，且 SLA path 没有同步 cold miss。
  - 本轮没有证明性能收益。2K 下 B3/B5 明显慢于 B0，当前瓶颈更像 Python/safetensors per-layer tensor store、vLLM connector external load/store 生命周期和小文件布局开销。
  - 下一步不宜直接扩 16K/32K 或继续深挖真实 3FS executor；应优先拆分 TTFT、实现 packed cold object/extent layout、减少 per-layer 文件与复用阶段重写，再扩尺寸矩阵。
- 输出文件：
  - `results/m3_11_2k_true_matrix/final/baseline_matrix.csv`
  - `results/m3_11_2k_true_matrix/final/summary/baseline_summary.csv`
  - `results/m3_11_2k_true_matrix/final/summary/workload_summary.csv`
  - `results/m3_11_2k_true_matrix/final/summary/readiness_report.md`
  - `results/m3_11_2k_true_matrix/b5_anti_caching_true/store_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_2k_true_matrix/b5_anti_caching_true/reuse_phase/reuse_smoke_matrix.csv`
- 验证：
  - `pytest tests/m3/test_baseline_readiness_matrix.py -q`：`14 passed in 3.07s`
  - `pytest tests/m3 -q`：`122 passed, 2 warnings in 24.82s`
  - 收尾检查：无残留 `vllm serve` / `EngineCore` / `http_sidecar_cli` 进程，端口 8000/8010 关闭，GPU 显存回落。

### 阶段 M3.11-I：TTFT breakdown 与性能归因字段
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 为 M3.11 baseline schema 增加 TTFT breakdown 字段：`online_service_ttft_ms`、`restore_inclusive_ttft_ms`、`restore_wait_ms`、`connector_store_elapsed_ms`、`connector_load_elapsed_ms`、`connector_total_elapsed_ms`、`unattributed_ttft_ms`。
  - 修正 `run_reuse_smoke_matrix.py` 的 connector event 汇总口径，支持 `start_line`，避免 reuse 阶段把历史 store 事件全部混入当前行。
  - 扩展 `summarize_baseline_readiness_matrix.py`，在 summary 中聚合 restore-inclusive TTFT、connector store/load/total 和未解释 TTFT，并在 readiness report 中显示 B5 关键分解。
  - 重新导入已有 2K B3/B5 reuse CSV，生成 breakdown 版统一矩阵。
- 输出文件：
  - `results/m3_11_2k_true_matrix/imported_b3_b5_breakdown/baseline_matrix.csv`
  - `results/m3_11_2k_true_matrix/final_breakdown/baseline_matrix.csv`
  - `results/m3_11_2k_true_matrix/final_breakdown/summary/baseline_summary.csv`
  - `results/m3_11_2k_true_matrix/final_breakdown/summary/readiness_report.md`
- 关键结果：
  - B3：online TTFT `6055.039ms`，connector load `92.02ms`，connector store_layer 求和 `7553.592ms`，未解释在线 TTFT `5963.019ms`。
  - B5：online TTFT `6590.921ms`，restore wait `874.308ms`，connector load `183.665ms`，restore-inclusive TTFT `7465.229ms`，connector store_layer 求和 `6484.31ms`，未解释在线 TTFT `5532.948ms`。
  - 解释：`store_layer` 是逐层事件耗时求和，可能与 vLLM forward/writeback 生命周期重叠，不能和墙钟 TTFT 直接相减；但它明确暴露复用请求仍在做大量写回，这是下一步 P0 性能优化目标。
- 结论：
  - 当前 B5 慢不只是 cold restore 慢。restore 约 874ms，load 约 184ms，而在线服务 TTFT 仍有约 5.5s 未被解释。
  - 下一步应先修复复用阶段全量 store/writeback，再推进 packed cold object / extent layout；直接扩大到 16K/32K 会放大当前原型开销。
- 验证：
  - `pytest tests/m3/test_reuse_smoke_matrix.py tests/m3/test_baseline_readiness_matrix.py -q`：`33 passed in 4.77s`

### 阶段 M3.11-J：load-only reuse 与复用阶段写回修复
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 先补红测：load-only reuse 即使残留 `store_token_end`，也不能在 `build_connector_meta()` 中生成 `store`；load-only 请求结束时不能返回 auto-commit manifest；sidecar 普通复用请求必须生成 `store_policy=load_only` 且不携带 `store_prefix_id/store_token_end`。
  - 更新 `benchmarks/m3/http_sidecar.py`：`build_kv_transfer_params()` 新增显式 `store_policy`。只有 `m3_control.store_prefix_id` 存在时才设置 `store_policy=store_prefix`、`store_prefix_id` 和 `store_token_end`；普通复用请求默认 `load_only`。
  - 更新 `benchmarks/m3/noop_connector.py`：新增 store policy 判断；required range 的 `prefix_id` 只作为 load 来源，不再自动变成本轮写回目标；`request_finished()` 对 load-only 请求不再返回 `m3_noop_connector.manifest`。
  - 运行真实在线 smoke：Qwen2.5-14B、`--max-model-len 512`，store 阶段保存 64 token 前缀，停止并重启 vLLM，保留 sidecar 与 tensor store 后执行 reuse 阶段。
- 输出文件：
  - `benchmarks/m3/http_sidecar.py`
  - `benchmarks/m3/noop_connector.py`
  - `tests/m3/test_http_sidecar.py`
  - `tests/m3/test_noop_connector.py`
  - `results/m3_11_load_only_reuse_smoke/store_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_load_only_reuse_smoke/reuse_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_load_only_reuse_smoke/connector_events.jsonl`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - store 阶段：`status=OK`，`store_events=48`，`load_events=0`，`connector_store_elapsed_ms=2458.67`。
  - reuse 阶段：`status=OK`，`decision=ADMIT`，`reason=required_kv_ready_before_decode`，`external_load_observed=yes`，`store_events=0`，`load_events=1`，`connector_load_elapsed_ms=35.314`。
  - 全量 connector event 统计为 `store_layer=48`、`load_request=1`；这证明复用阶段不再重写 48 层 KV。
- 验证：
  - `pytest tests/m3/test_noop_connector.py::test_build_connector_meta_does_not_store_load_only_reuse_by_default tests/m3/test_noop_connector.py::test_request_finished_suppresses_manifest_for_load_only_reuse tests/m3/test_http_sidecar.py::test_build_kv_transfer_params_carries_sidecar_plan tests/m3/test_http_sidecar.py::test_build_kv_transfer_params_marks_explicit_store_request -q`：`4 passed, 2 warnings`
  - `pytest tests/m3/test_noop_connector.py tests/m3/test_http_sidecar.py tests/m3/test_reuse_smoke_matrix.py -q`：`50 passed, 2 warnings`
  - `pytest tests/m3 -q`：`126 passed, 2 warnings`
  - 收尾检查：8000/8010 端口关闭；无残留 `vllm serve` / `EngineCore` / `http_sidecar_cli` 进程；GPU 显存回落为 `11, 45606` MiB。

### 阶段 M3.11-K：load-only 后 2K B3/B5 真机重跑与统一报告修正
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 用已修复的 load-only connector 重新跑 2K B3 external DRAM-ready reuse。流程为 store 2048-token prefix、停止并重启 vLLM、保留 sidecar 与 tensor store、reuse 阶段只触发 external load。
  - 用已修复的 load-only connector 重新跑 2K B5 true cold-tier Anti-Caching。流程为 store -> demote true cold object -> mark cold/not-ready -> restart vLLM -> cold probe DELAY -> async advance restore -> reuse ADMIT。
  - 将新 B3/B5 reuse CSV 通过 `run_baseline_readiness_matrix.py --import-reuse-csv` 导入 M3.11 baseline schema。
  - 合并旧的 B0/B1/B2 原生结果与新的 B3/B5，重新生成 `results/m3_11_2k_load_only_rerun/final/baseline_matrix.csv` 和 summary report。
  - 按 TDD 修复 baseline import/report 的 `unattributed_ttft_ms` 口径：该字段只拆 online 服务阶段，不能从 online TTFT 里扣除请求前 restore wait。
- 输出文件：
  - `results/m3_11_2k_load_only_rerun/b3_external/store_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_2k_load_only_rerun/b3_external/reuse_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_2k_load_only_rerun/b5_anti_caching_true/store_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_2k_load_only_rerun/b5_anti_caching_true/reuse_phase/reuse_smoke_matrix.csv`
  - `results/m3_11_2k_load_only_rerun/imported_b3_b5/baseline_matrix.csv`
  - `results/m3_11_2k_load_only_rerun/final/baseline_matrix.csv`
  - `results/m3_11_2k_load_only_rerun/final/summary/baseline_summary.csv`
  - `results/m3_11_2k_load_only_rerun/final/summary/readiness_report.md`
  - `benchmarks/m3/run_baseline_readiness_matrix.py`
  - `benchmarks/m3/summarize_baseline_readiness_matrix.py`
  - `tests/m3/test_baseline_readiness_matrix.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - B3 store 阶段：`status=OK`，`first_ttft_ms=1005.291`，`store_events=48`，`load_events=0`，tensor store 约 `385MiB`。
  - B3 reuse 阶段：`status=OK`，online TTFT `283.027ms`，`decision=ADMIT`，`store_events=0`，`load_events=1`，`external_load_observed=yes`，connector load `94.531ms`。
  - B5 store/demote 阶段：`status=OK`，`first_ttft_ms=1088.864`，`store_events=48`，cold object bytes `402657408`，`cold_commit_status=committed`。
  - B5 reuse/restore 阶段：online TTFT `327.512ms`，restore executor `1035.707ms`，restore-inclusive TTFT `1363.219ms`，connector load `139.882ms`，`store_events=0`，`load_events=1`，`sync_ssd_miss_total=0`，checksum `ok`。
  - 合并矩阵：B0 `717.061ms`，B1 `91.016ms`，B2 `686.379ms`，B3 `283.027ms`，B5 online `327.512ms`。B5 online/B0 为 `0.456742`，B5 online/B3 为 `1.157176`，B5 online/B1 为 `3.5984`。
  - B5 restore-inclusive/B0 约为 `1.90x`。这说明 PrePass/idle async restore 是否能隐藏 cold restore，是判断 SLA 能否成立的关键。
- 结论：
  - load-only 修复后，旧 2K B3/B5 的 6 秒级 TTFT 主要被 reuse 阶段误写回污染；当前 2K 在线路径已经变成有希望的性能样本。
  - 但 B5 还不能直接声称端到端满足 SLA。若 restore wait 计入请求 TTFT，当前 2K 仍超过 B0+20% 阈值；下一步必须实现/验证 KV PrePass 和后台 restore hiding。
  - 接下来应扩 512/8K/16K/32K 小矩阵，并优先做 packed cold object / extent layout，避免 8K+ 时每层文件布局和 restore tail 成为新瓶颈。
- 验证：
  - `pytest tests/m3/test_baseline_readiness_matrix.py::test_online_b3_b5_bridge_maps_reuse_executor_rows_to_baseline_schema tests/m3/test_baseline_readiness_matrix.py::test_summarizer_computes_ratios_and_readiness_report -q`：`2 passed in 1.44s`

### 阶段 M3.12-A：最小 KV PrePass 与 restore hiding 控制面验证
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 新增 `/prepass` 行为测试：cold prefix 在 online request 前通过 PrePass 恢复为 `DRAM/ready=true`，后续 `/admit` 直接 `ADMIT`；async 模式下 PrePass 只 `QUEUED`，advance 前不会伪装 ready。
  - 新增 `benchmarks/m3/prepass_planner.py`，提供最小 PrePass 状态归类：`READY`、`QUEUED`、`MISSED`、`ERROR`、`PENDING`。
  - 扩展 `benchmarks/m3/http_sidecar.py`：新增 `/prepass` endpoint、`SidecarRuntime.prepass_from_payload()`、`prepass_requests_total`、`prepass_ready_total` 和 `prepass_result` 结构化日志。
  - 修正 PrePass 指标口径：PrePass 不再增加 `prefill_tokens_saved_total` 或 online admission metrics，避免把规划阶段误计为线上执行。
  - 新增 `benchmarks/m3/run_prepass_restore_smoke.py` 与测试，生成 deterministic PrePass restore smoke CSV、summary 和 Markdown 报告。
  - 扩展 `benchmarks/m3/run_reuse_smoke_matrix.py`：新增 `--prepass-before-reuse`，并写出 `prepass_status`、`ready_before_request`、`prepass_elapsed_ms` 等 CSV 字段，为真实 vLLM B5 PrePass 样本做准备。
- 输出文件：
  - `benchmarks/m3/prepass_planner.py`
  - `benchmarks/m3/http_sidecar.py`
  - `benchmarks/m3/run_prepass_restore_smoke.py`
  - `benchmarks/m3/run_reuse_smoke_matrix.py`
  - `tests/m3/test_prepass_planner.py`
  - `tests/m3/test_prepass_planner_unit.py`
  - `tests/m3/test_prepass_restore_smoke.py`
  - `tests/m3/test_reuse_smoke_matrix.py`
  - `results/m3_12_prepass_restore_smoke/prepass_restore_smoke.csv`
  - `results/m3_12_prepass_restore_smoke/prepass_restore_report.md`
  - `results/m3_12_prepass_restore_smoke/prepass_restore_summary.json`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - deterministic smoke：2048 prefix + 128 suffix，`prepass_status=READY`，`ready_before_request=true`，`restore_status=COMPLETED`。
  - online `/admit` 在 PrePass 后直接返回 `ADMIT/required_kv_ready_before_decode`，`online_reuse_tokens=2048`，`online_delta_prefill_tokens=128`，`online_ready_barrier=true`，`sync_cold_miss_total=0`。
  - PrePass 日志中 `required_count=1`、`ready_count=1`，已修复初版重复计数问题。
- 当前边界：
  - 本阶段是控制面 proof，不是最终 vLLM 性能结论。deterministic smoke 使用 manifest-only restore，不能替代 402MB true cold object restore。
  - 下一步需要跑真实 vLLM B5 PrePass-before-reuse 样本，对比 reactive B5 的 `restore-inclusive=1363.219ms` 与 PrePass 后的 online TTFT。
- 验证：
  - `pytest tests/m3/test_prepass_planner_unit.py tests/m3/test_prepass_planner.py tests/m3/test_prepass_restore_smoke.py -q`：`5 passed in 3.60s`
  - `pytest tests/m3/test_prepass_planner.py tests/m3/test_http_sidecar.py tests/m3/test_prepass_restore_smoke.py tests/m3/test_reuse_smoke_matrix.py::test_run_matrix_dry_run_writes_csv_and_report tests/m3/test_reuse_smoke_matrix.py::test_reuse_smoke_matrix_cli_help_runs_from_repo_root tests/m3/test_reuse_smoke_matrix.py::test_summarize_prepass_response_extracts_ready_and_restore_status -q`：`23 passed in 6.16s`

### 阶段 M3.12-B：2K 真实 B5 PrePass-before-reuse 样本
- **状态：** complete
- 时间：2026-05-27
- 执行的操作：
  - 按 TDD 为 `run_reuse_smoke_matrix.py` 补齐 PrePass restore 真实指标字段：`prepass_restore_actual_bytes`、`prepass_restore_checksum_status`、`prepass_restore_executor_elapsed_ms`。
  - 启动 Qwen2.5-14B vLLM，配置 `M3NoOpConnector`、`m3_tensor_store_path`、`m3_event_log_path` 和 `m3_kv_layout=NHD`。
  - 启动 sidecar，保留 `--inject-kv-transfer-params`、`--tensor-store-path`、`--cold-root`、`--async-prefetch`。
  - 执行 2K store 阶段：保存 2048-token prefix 的真实 KV，demote 为 true cold object，并将 sidecar manifest 标记为 cold/not-ready。
  - 停止并重启 vLLM，保留 sidecar、tensor store、cold root 和决策日志，避免同进程 APC 热缓存污染。
  - 执行 reuse 阶段：先 `/prepass` 枚举 required historical KV 并入队 cold restore，再 `/prefetch/advance` 恢复，最后发送真正 reuse 请求。
  - 将 PrePass reuse CSV 导入 M3.11 baseline schema，模型上下文上限按本次 vLLM `--max-model-len 8192` 记录。
  - 写入独立实验报告 `docs/research/m3_12_prepass_true_2k_results.md`。
- 输出文件：
  - `benchmarks/m3/run_reuse_smoke_matrix.py`
  - `tests/m3/test_reuse_smoke_matrix.py`
  - `docs/research/m3_12_prepass_true_2k_results.md`
  - `results/m3_12_2k_prepass_true/b5_prepass_true/store_phase/reuse_smoke_matrix.csv`
  - `results/m3_12_2k_prepass_true/b5_prepass_true/reuse_phase/reuse_smoke_matrix.csv`
  - `results/m3_12_2k_prepass_true/b5_prepass_true/connector_events.jsonl`
  - `results/m3_12_2k_prepass_true/b5_prepass_true/decisions.jsonl`
  - `results/m3_12_2k_prepass_true/b5_prepass_true/prefetch_queue_events.jsonl`
  - `results/m3_12_2k_prepass_true/imported_b5_prepass/baseline_matrix.csv`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - store 阶段：`status=OK`，`first_ttft_ms=953.17`，`store_events=48`，connector store 求和 `323.865ms`。
  - cold object：`402657408` bytes，checksum `sha256:4d3577c3a74f0eef3feb048b3974d07116ecc357d69661d1c6e4b41d2973cf28`，backend `local_posix`。
  - PrePass 阶段：`prepass_status=QUEUED`，`ready_before_request=false`，`prepass_elapsed_ms=11.07`，restore 入队字节 `402657408`。
  - advance restore：`cold_restore_status=COMPLETED`，checksum `ok`，executor elapsed `839.601ms`，`prefetch_queue_completed_delta=1`。
  - reuse 阶段：online TTFT `271.782ms`，connector load `97.694ms`，`store_events=0`，`load_events=1`，`external_load_observed=yes`，`sync_ssd_miss_total=0`。
  - 对比 M3.11 load-only reactive B5：reactive online TTFT `327.512ms`，restore-inclusive `1363.219ms`；PrePass online TTFT `271.782ms`，restore-inclusive `1111.383ms`。
  - 与 B0 full prefill `717.061ms` 对比：PrePass online/B0 为 `0.379x`，但 restore-inclusive/B0 为 `1.550x`。这证明 PrePass 可以把 cold restore 移出 online request path，但如果没有足够提前量，严格 SLA 仍不成立。
- 结论：
  - 当前样本首次在真实 vLLM + true cold object 路径上证明了 PrePass 的核心价值：执行前枚举 required KV、提前恢复、在线请求只做 external load + delta prefill。
  - 该结果不能解释为 cold restore 已经被优化到足够快。restore 成本仍是 `839.601ms`，只是被前移；下一步必须研究 lead time、packed cold object 和更长上下文下的 restore tail。
- 验证待收尾：
  - 需要在服务清理后运行针对性 pytest 与端口/GPU 检查。

### 阶段 M3.13-A：PrePass lead-time planner 与下一步研究方向优化
- **状态：** complete
- 时间：2026-05-27T15:08:36Z
- 执行的操作：
  - 按用户要求结合 `level2-research`、`context-engineering`、`planning-with-files-zh` 重新恢复项目上下文，读取 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`、M3.12 报告和 PrePass/Baseline 相关代码。
  - 将下一步研究方向从“继续盲跑更大矩阵”优化为“PrePass lead-time、restore tail、packed cold object gate”。核心判断：PrePass online TTFT 好看只有在 restore 被提前隐藏时成立，因此必须先量化需要多少提前量。
  - 按 TDD 新增 `tests/m3/test_prepass_lead_time_planner.py`，先观察到缺失模块红灯：`ModuleNotFoundError: No module named 'benchmarks.m3.prepass_lead_time_planner'`。
  - 实现 `benchmarks/m3/prepass_lead_time_planner.py`：读取真实 M3.12 PrePass CSV 和 M3.11 baseline CSV，推导 restore profile，生成 `hide_restore`、`residual_online_wait_ms`、`packed_object_required`、`priority_class`、`risk_class`。
  - 实现 CLI `benchmarks/m3/plan_prepass_lead_time_matrix.py`，输出 CSV、Markdown 报告和 JSON profile。
  - 用真实数据生成 `results/m3_13_prepass_lead_time_plan/`，输入为 `results/m3_12_2k_prepass_true/b5_prepass_true/reuse_phase/reuse_smoke_matrix.csv` 与 `results/m3_11_2k_load_only_rerun/final/baseline_matrix.csv`。
  - 写入研究计划文档 `docs/research/m3_13_next_research_plan.md`，明确 512/2K/8K true PrePass 小矩阵、packed cold object、16K/32K gate matrix 的执行顺序。
  - 更新 `task_plan.md`、`findings.md` 和 `progress.md`。
- 输出文件：
  - `benchmarks/m3/prepass_lead_time_planner.py`
  - `benchmarks/m3/plan_prepass_lead_time_matrix.py`
  - `tests/m3/test_prepass_lead_time_planner.py`
  - `results/m3_13_prepass_lead_time_plan/prepass_lead_time_plan.csv`
  - `results/m3_13_prepass_lead_time_plan/prepass_lead_time_report.md`
  - `results/m3_13_prepass_lead_time_plan/restore_profile.json`
  - `docs/research/m3_13_next_research_plan.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - 真实 restore profile：2K prefix、`402657408` bytes、restore executor `839.601ms`、KV bytes/token `196608`、restore throughput `457.365MiB/s`、PrePass overhead `11.070ms`。
  - 在 `tail_multiplier=1.2` 与 `safety_margin=250ms` 下，required lead 估算为：512 `512.948ms`，2K `1268.581ms`，8K `4291.113ms`，16K `8321.155ms`，32K `16381.240ms`。
  - 计划表共 20 行：restore hidden rows `8`，lead-time risk rows `12`，packed-object attention rows `12`。
  - 推荐顺序：512 downscale calibration -> 2K source repeat -> 8K first scaling probe -> 16K packed-layout gate -> 32K context boundary。
  - 8K 是下一步最关键 gate：5s lead 理论上可隐藏 restore，但已经触发 packed cold object 风险；16K/32K 不应在 8K 和 packed layout 证据前盲跑。
- 验证：
  - `pytest tests/m3/test_prepass_lead_time_planner.py -q`：先红灯缺失模块，补实现后 `4 passed in 0.10s`。
  - `python benchmarks/m3/plan_prepass_lead_time_matrix.py --prepass-csv results/m3_12_2k_prepass_true/b5_prepass_true/reuse_phase/reuse_smoke_matrix.csv --baseline-csv results/m3_11_2k_load_only_rerun/final/baseline_matrix.csv --result-dir results/m3_13_prepass_lead_time_plan --prefix-tokens 512 2048 8192 16384 32768 --suffix-tokens 128 --output-tokens 1 --available-leads-ms 0 1000 5000 12000`：`status=OK`，`rows=20`。
  - 发现本机缺少 `column` 命令，已改用 `sed` 查看 CSV。

### 阶段 M3.14-A：Anti-Cache 启发整理与两个核心模块设计
- **状态：** complete
- 时间：2026-05-28T02:25:25Z
- 执行的操作：
  - 按用户要求使用 `context-engineering` 和 `planning-with-files-zh` 恢复项目上下文，读取 `AGENTS.md`、`task_plan.md`、`findings.md`、`progress.md`。
  - 重新聚焦 Anti-Cache 论文给 KV 系统的机制启发：不是 100% 提前预测请求，而是通过内存常驻索引发现 cold data，通过 pre-pass 收集恢复位置，异步恢复后重新准入。
  - 读取当前相关实现：`benchmarks/m3/control_plane.py`、`benchmarks/m3/tensor_store.py`、`benchmarks/m3/http_sidecar.py`，确认现有系统已有 KV manifest、tier/ready、cold object、PrePass 和 restore queue，但还缺内存常驻的统一 KV Evicted Index，以及真正的 packed cold object 布局。
  - 设计下一步最应该落地的两个模块：`KV Evicted Index` 和 `Packed Cold Object`。
  - 写入设计规格 `docs/specs/m3_14_kv_evicted_index_and_packed_object.md`。
  - 更新 `task_plan.md`：新增 M3.14 阶段，并把 `KV Evicted Index` 提升为 P0。
  - 更新 `findings.md`：记录 Anti-Cache 到 KV 的映射、三层存储语义和模块边界。
- 输出文件：
  - `docs/specs/m3_14_kv_evicted_index_and_packed_object.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键设计：
  - `KV Evicted Index`：常驻 CPU 内存，记录 prefix/range/correctness key/tier/ready/object id/cold uri/extents/checksum/热度，支持 READY/COLD/FETCHING/MISSING/MISMATCH 分类。
  - `Packed Cold Object`：SSD/3FS 上的连续大对象，对应 Anti-Cache Block Table，记录 object/extent offset/length/checksum，支撑顺序读和后续部分恢复。
  - 请求进入 GPU 前必须经过控制面 PrePass：如果 required KV 在 SSD，则请求 `DELAY`，后台 SSD->CPU restore，校验后重新准入；SSD KV 永远不能被视为 ready。
  - SSD->CPU 恢复队列与 CPU->HBM 加载队列应分离；前者是冷区恢复，后者是执行前加载。
- 下一步建议：
  - M3.14-B：先按 TDD 实现 `benchmarks/m3/kv_evicted_index.py` 和 `tests/m3/test_kv_evicted_index.py`。
  - M3.14-C：把 `/prepass` 输出升级为 ready/cold/fetching/missing/mismatch sets。
  - M3.14-D：实现 packed cold object v1 并跑 2K/8K packed vs per-layer restore 对比。

### 阶段 M3.14-A：KV Evicted Index 独立模块实现
- **状态：** complete
- 时间：2026-05-28T03:18:00Z
- 执行的操作：
  - 按 TDD 新增 `tests/m3/test_kv_evicted_index.py`，先观察到红灯：`ModuleNotFoundError: No module named 'benchmarks.m3.kv_evicted_index'`。
  - 新增 `benchmarks/m3/kv_evicted_index.py`，实现 CPU 常驻元数据索引，不保存 KV 张量本体。
  - 支持从现有 `KVBlockManifest` 导入 prefix/range/correctness_key/tier/ready/cold_uri/object_id/offset_table/checksum/size_bytes。
  - 支持 `lookup_required()`、`classify_required()`、`mark_fetching()`、`mark_ready()`、`mark_evicted()` 和 `touch()`。
  - 将 `AGENTS.md` 补充 M3.14 当前阶段规则：先做 `KV Evicted Index` 与 `Packed Cold Object`，真实 3FS executor 暂列 P1。
  - 更新 `task_plan.md`、`findings.md` 和 `progress.md`。
- 输出文件：
  - `benchmarks/m3/kv_evicted_index.py`
  - `tests/m3/test_kv_evicted_index.py`
  - `AGENTS.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - READY：只有 `HBM/DRAM/CPU` 且 `ready=true` 才可直接进入可复用集合。
  - COLD：`SSD/NVME/LOCAL_NVME/THREE_FS/3FS` 或存在 `cold_uri/object_id/extents` 时归为冷区，并返回 object/offset/length 信息。
  - FETCHING：恢复任务入队后标记为 `FETCHING`，后续 PrePass 可据此避免重复恢复同一 range。
  - MISMATCH：correctness key 不匹配时与 missing 分开，后续应禁止复用而不是尝试恢复。
  - MISSING：prefix 未知或 range 未被现有 entry 覆盖时返回 missing。
- 验证：
  - `pytest tests/m3/test_kv_evicted_index.py -q`：先红灯缺失模块，补实现后 `7 passed in 1.27s`。
  - `pytest tests/m3/test_kv_evicted_index.py tests/m3/test_prepass_planner.py tests/m3/test_prepass_planner_unit.py tests/m3/test_prefetch_queue.py -q`：`18 passed in 2.34s`。

### 阶段 M3.14-B：PrePass 分类集合接入
- **状态：** complete
- 时间：2026-05-28T03:37:00Z
- 执行的操作：
  - 按 TDD 扩展 `tests/m3/test_prepass_planner.py`，要求 `/prepass` 响应新增 `classification` 与 `classification_counts` 字段。
  - 红灯确认旧接口缺少分类集合：三个测试均因 `KeyError: 'classification'` 失败。
  - 在 `SidecarRuntime` 中增加 runtime 级 `KVEvictedIndex`，保存 `requests` 以便 commit 时同步 correctness key。
  - 在 `/commit`、proxy auto commit、tensor store manifest 读取、同步 restore 完成和异步 `/prefetch/advance` 完成时，同步索引元数据。
  - 在 `/prepass` 中调用 `_classify_prepass_request()`，输出 `ready/cold/fetching/missing/mismatch` 五类集合，同时保留原有 `status`、`ready_before_request`、`restore_results` 字段。
  - 对异步 restore 中的旧 cold manifest 做防覆盖处理：未 ready 的控制面或 tensor-store 状态不会覆盖索引中的 `FETCHING` 状态，只有 ready 后才能更新为 ready。
- 输出文件：
  - `benchmarks/m3/http_sidecar.py`
  - `benchmarks/m3/kv_evicted_index.py`
  - `tests/m3/test_prepass_planner.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 关键结果：
  - 同步 PrePass 冷前缀响应中可看到 `classification.cold[0].prefix_id=session-a`，即使 restore 完成后整体 `status=READY`。
  - 异步 PrePass 第一次请求为 `COLD -> QUEUED`；第二次请求在 restore 未完成前归入 `classification.fetching`，避免重复恢复同一 range。
  - correctness mismatch 与 missing range 被分开输出：前者进入 `classification.mismatch` 并保持 `REJECT`，后者进入 `classification.missing` 并可走 full/delta prefill fallback 语义。
- 验证：
  - `pytest tests/m3/test_prepass_planner.py -q`：先红灯缺少分类字段，补实现后 `3 passed in 1.55s`。
  - `python -m py_compile benchmarks/m3/kv_evicted_index.py benchmarks/m3/http_sidecar.py`：通过。
  - `pytest tests/m3/test_kv_evicted_index.py tests/m3/test_prepass_planner.py tests/m3/test_prepass_planner_unit.py tests/m3/test_http_sidecar.py tests/m3/test_prefetch_queue.py -q`：`35 passed in 3.60s`。

### 阶段 M3.14-B2：GPU/CPU/SSD 三级状态语义修正
- **状态：** complete
- 时间：2026-05-28T04:12:00Z
- 执行的操作：
  - 根据用户追问重新审视上一版 `READY/COLD` 分类，确认它会把 `HBM ready` 与 `CPU/DRAM ready` 混在一起，无法完整表达三级存储调度。
  - 按 TDD 扩展 `tests/m3/test_kv_evicted_index.py`，要求 HBM -> `GPU_READY`、DRAM/CPU -> `CPU_READY`、SSD/NVME/3FS -> `SSD_COLD`，并新增 `mark_loading()` 表示 CPU->GPU load 进行中。
  - 扩展 `tests/m3/test_prepass_planner.py`，要求 `/prepass` 输出 `gpu_ready/cpu_ready/ssd_cold/fetching/loading/missing/mismatch` 细分集合，同时保留 `ready/cold` 兼容汇总。
  - 修改 `benchmarks/m3/kv_evicted_index.py`：新增 `GPU_READY`、`CPU_READY`、`SSD_COLD`、`LOADING` 状态；`READY` 和 `COLD` 不再作为新逻辑主状态，而由分类集合做兼容汇总。
  - 修改 `benchmarks/m3/http_sidecar.py`：`classification_counts` 增加 `gpu_ready`、`cpu_ready`、`ssd_cold`、`loading`。
  - 更新 `AGENTS.md`、`task_plan.md`、`findings.md` 和 `progress.md`，把下一步方案修正为三级状态机，而不是二分 ready/cold。
- 关键结果：
  - `GPU_READY`：KV 已在 HBM，可视为执行 ready。
  - `CPU_READY`：KV 已在 CPU/DRAM 且校验通过，但还需要 CPU->GPU load；不能再和 `GPU_READY` 混为一谈。
  - `SSD_COLD`：KV 在 SSD/3FS 冷区或持久化层，必须先 SSD->CPU restore。
  - `FETCHING`：SSD/3FS->CPU restore 进行中。
  - `LOADING`：CPU/DRAM->GPU/HBM load 进行中。
  - `ready` 与 `cold` 字段继续保留为兼容汇总：`ready = gpu_ready + cpu_ready`，`cold = ssd_cold`。
- 验证：
  - `pytest tests/m3/test_kv_evicted_index.py tests/m3/test_prepass_planner.py -q`：先红灯 8 个失败，补实现后 `13 passed in 1.63s`。

### 阶段 M3.14-B3：CPU_READY -> GPU_READY load readiness 观测
- **状态：** complete
- 时间：2026-05-28T05:02:00Z
- 执行的操作：
  - 在 B2 三级状态修正基础上继续推进最小 B3，不进入 attention kernel 或 packed object。
  - 按 TDD 新增 `tests/m3/test_prepass_planner.py::test_prepass_reports_cpu_ready_h2d_load_budget`，先观察红灯：`KeyError: 'load_readiness'`。
  - 修改 `benchmarks/m3/http_sidecar.py`，为 `/prepass` 响应新增 `load_readiness` 字段。
  - 根据 `classification.cpu_ready` 计算 `cpu_ready_not_gpu_ready_count`、`cpu_ready_tokens`、`expected_h2d_load_bytes`、`estimated_h2d_load_ms`。
  - 在 `/metrics` 中新增 `cpu_ready_not_gpu_ready_total`，用于观测 CPU 已 ready 但 GPU 尚未 ready 的历史 KV。
  - 验证并保留 `mark_ready()` 的状态收口：CPU->GPU load 完成并标记 HBM 后会清理 `loading_request_id`，因此 `LOADING -> GPU_READY` 可以正确闭合。
- 关键结果：
  - 对 64 tokens、Qwen2.5-14B 每 token KV `196608` bytes、H2D `25GB/s` 的 CPU_READY 前缀，PrePass 估算 `expected_h2d_load_bytes=12582912`，`estimated_h2d_load_ms=0.503`。
  - 当前 B3 仍主要是观测口径，不是完整在线 GPU residency 调度；后续还要把真实 connector load 完成事件回写到索引。
- 验证：
  - `pytest tests/m3/test_prepass_planner.py::test_prepass_reports_cpu_ready_h2d_load_budget -q`：先红灯缺少 `load_readiness`，补实现后 `1 passed in 1.69s`。
  - `pytest tests/m3/test_kv_evicted_index.py::test_mark_ready_after_loading_promotes_range_to_gpu_ready -q`：先红灯 `LOADING` 未提升为 `GPU_READY`，补实现后 `1 passed in 1.26s`。

### 阶段 M3.14-C：Packed Cold Object v1
- **状态：** complete
- 时间：2026-05-28T05:47:00Z
- 执行的操作：
  - 按规划推进 `Packed Cold Object v1`，第一版选择 layer-major packed layout，保留 extent/offset 元数据，暂不做 extent-major partial restore。
  - 按 TDD 新增 `tests/m3/test_packed_cold_object.py`，先观察红灯：`ImportError: cannot import name 'PackedColdTierAdapter'`。
  - 在 `benchmarks/m3/cold_tier.py` 新增 `PackedColdTierAdapter`，backend 名为 `packed_v1`。
  - `packed_v1` demote 会把每层 safetensors 文件按层顺序拼入 `packed_object.bin`，写出 `packed_manifest.json`，并删除热目录中的 per-layer 文件。
  - `packed_v1` restore 会根据 `offset_table` 从 `packed_object.bin` 切片恢复每层 safetensors 文件，并做每层 checksum 校验。
  - `build_cold_tier_adapter()` 支持 `packed_v1` / `local_packed_v1`。
  - 将 `packed_v1` 加入 `run_cold_tier_adapter_bench.py`、`http_sidecar_cli.py`、`run_tier_migration_smoke.py`、`run_reuse_smoke_matrix.py`、`run_baseline_readiness_matrix.py` 的 `--cold-backend` choices。
  - 新增 cold-tier adapter benchmark 测试，验证 `--cold-backend packed_v1` 能跑通并生成 packed object。
- 关键结果：
  - packed cold object 目录现在只需要 `packed_object.bin` + `packed_manifest.json`，不再在冷区保留每层 safetensors 文件。
  - `KVBlockManifest.offset_table` 中每层都记录 `file_name=packed_object.bin`、`original_file_name`、`offset`、`size_bytes`、`checksum`、`packed_layout=packed_v1`。
  - packed object 被篡改导致 size/checksum mismatch 时，restore 会失败，manifest 保持 cold/not-ready。
  - 该实现仍是 local POSIX packed layout，不是 3FS 原生 packed object；M3.14-D 需要用它和 per-layer `local_posix` 做 2K/8K 对比。
- 验证：
  - `pytest tests/m3/test_packed_cold_object.py -q`：先红灯缺少 adapter，补实现后 `3 passed in 1.28s`。
  - `pytest tests/m3/test_packed_cold_object.py tests/m3/test_cold_tier_adapter_bench.py -q`：`9 passed in 2.53s`。

### 阶段 M3.14-D：2K/8K Packed vs Per-Layer Restore 对比
- **状态：** complete
- 时间：2026-05-28T06:28:00Z
- 执行的操作：
  - 按 TDD 新增 `tests/m3/test_packed_vs_per_layer_bench.py`，先观察红灯：`ModuleNotFoundError: No module named 'benchmarks.m3.run_packed_vs_per_layer_bench'`。
  - 新增 `benchmarks/m3/run_packed_vs_per_layer_bench.py`，复用已有 `run_cold_tier_adapter_bench.py`，同一命令生成 `local_posix` 与 `packed_v1` 对比。
  - 输出 aggregate CSV/JSON/Markdown：`packed_vs_per_layer_comparison.csv`、`packed_vs_per_layer_summary.json`、`packed_vs_per_layer_report.md`。
  - 补充 by-token 输出：`packed_vs_per_layer_by_token.csv`，避免 aggregate 掩盖 2K/8K 差异。
  - 运行正式 2K/8K 小矩阵：`python benchmarks/m3/run_packed_vs_per_layer_bench.py --result-dir results/m3_14_packed_vs_per_layer --prefix-tokens 2048 8192 --repeats 3 --queue-depth 1 --profile qwen25_14b_tiny`。
- 关键结果：
  - 结果目录：`results/m3_14_packed_vs_per_layer/`。
  - aggregate：`local_posix` restore p50/p95/p99 为 `153.920/257.851/260.170ms`，effective restore p50 `517.004MiB/s`。
  - aggregate：`packed_v1` restore p50/p95/p99 为 `315.211/531.195/531.347ms`，effective restore p50 `274.663MiB/s`。
  - `packed_v1` cold data files 从 24 降到 6，但 restore p95 比 per-layer 增加 `273.344ms`。
  - by-token：2K local/packed restore p95 为 `60.637ms` vs `102.856ms`；8K local/packed restore p95 为 `259.590ms` vs `531.309ms`。
  - 结论：当前 packed_v1 证明了文件数减少和接口可用，但没有证明 restore 更快；不能据此进入 16K/32K gate matrix。
- 下一步：
  - 进入 M3.14-E：优化 packed restore 数据面，避免 Python `read_bytes()` / bytes 拼接和二次切片开销，或实现更接近生产路径的顺序读/批量 pread/extent-major restore。
- 验证：
  - `pytest tests/m3/test_packed_vs_per_layer_bench.py -q`：先红灯缺少 runner，补实现后 `2 passed in 2.65s`。
  - 正式矩阵命令退出码 0，并生成 `packed_vs_per_layer_summary.json` / `packed_vs_per_layer_by_token.csv` / `packed_vs_per_layer_report.md`。

### 阶段 M3.14-E：Packed Restore 数据面优化
- **状态：** complete
- 时间：2026-05-28T07:18:00Z
- 执行的操作：
  - 继续 M3.14-D 发现的问题：初始 packed_v1 虽减少 cold data files，但因为 Python `read_bytes()`、整段 bytes 切片和重复 checksum，restore p95 明显慢于 per-layer。
  - 将 packed demote 改为 chunk streaming append，每层写入时同步计算 extent checksum，`offset_table` 与 `packed_manifest.json` 记录 `copy_mode=streaming`。
  - 将 packed restore 改为按 extent streaming copy，不再把整个 extent 一次性读入内存后再写出。
  - 将 `PackedColdTierAdapter.summarize()` 改为 fast manifest path：当 `packed_object.bin` 实际大小与 manifest 记录一致时，直接复用 manifest checksum；大小不一致时返回 `sha256:size-mismatch:<actual_size>` 触发上层 checksum mismatch。
  - 按 TDD 新增回归测试，阻止 demote 收尾阶段在 streaming append 后重新以读模式打开 `packed_object.bin`；红灯确认 `_summarize_from_offsets()` 会二次读 packed object 后，改为用 append 阶段已记录的 extent checksum 组合 manifest checksum。
  - 运行 2K/8K 小矩阵 r3：`python benchmarks/m3/run_packed_vs_per_layer_bench.py --result-dir results/m3_14_packed_vs_per_layer_fastmanifest --prefix-tokens 2048 8192 --repeats 3 --queue-depth 1 --profile qwen25_14b_tiny`。
  - 去掉 demote summary 二次读后，再运行 r5：`python benchmarks/m3/run_packed_vs_per_layer_bench.py --result-dir results/m3_14_packed_vs_per_layer_streaming_summary_r5 --prefix-tokens 2048 8192 --repeats 5 --queue-depth 1 --profile qwen25_14b_tiny`。
- 关键结果：
  - r3 fastmanifest 结果：local_posix restore p50/p95/p99 为 `158.667/254.746/255.478ms`，packed_v1 为 `143.334/233.673/235.484ms`；packed restore p95 delta `-21.073ms`，cold data file count delta `-18`。
  - r5 streaming-summary 结果：local_posix restore p50/p95/p99 为 `216.930/283.665/287.823ms`，packed_v1 为 `145.103/225.144/225.181ms`；packed restore p95 delta `-58.521ms`，cold data files 从 `40` 降到 `10`。
  - r5 by-token：2K local/packed restore p95 为 `159.497ms` vs `66.766ms`；8K local/packed restore p95 为 `286.552ms` vs `225.169ms`。
  - 谨慎结论：packed_v1 已从“功能可用但性能明显落后”变成“local POSIX 小矩阵中可与 per-layer 打平或胜出”。这支持继续进入 8K/16K gate，但仍不能声称真实 3FS 或生产 executor 性能优势。
- 修改文件：
  - `benchmarks/m3/cold_tier.py`
  - `tests/m3/test_packed_cold_object.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/specs/m3_14_kv_evicted_index_and_packed_object.md`
- 验证：
  - `pytest tests/m3/test_packed_cold_object.py::test_packed_adapter_builds_manifest_checksum_from_extent_metadata -q`：先红灯命中 packed object 二次读，修复后 `1 passed in 1.20s`。
  - `pytest tests/m3/test_packed_cold_object.py tests/m3/test_packed_vs_per_layer_bench.py tests/m3/test_cold_tier_adapter_bench.py -q`：`12 passed in 3.95s`。
  - `python -m py_compile benchmarks/m3/cold_tier.py benchmarks/m3/run_packed_vs_per_layer_bench.py`：通过。

### 阶段 M3.15-A：8K packed PrePass gate
- **状态：** complete
- 时间：2026-05-28T04:49:08Z
- 执行的操作：
  - 继续规划中的下一步：将优化后的 `packed_v1` cold object 接入真实 8K PrePass/reuse gate。
  - 恢复并确认 vLLM store 服务配置：`--max-model-len 16384 --max-num-batched-tokens 16384`，避免默认 chunked prefill 只暴露 2048 tokens。
  - 启动 sidecar，配置 `--cold-backend packed_v1 --async-prefetch`，保留 tensor store 与 cold root。
  - 运行 8K store 阶段：`prefix_tokens=8192`、`suffix_tokens=128`、`output_tokens=1`、`--cold-tier-restore`。
  - 校验 store 有效性：manifest token range 为 `0..8192`，connector 事件中每层 `tokens=8192`、`block_count=512`，packed object 大小 `1610616960` bytes，`cold_saved_token_mismatch=no`。
  - 重启 vLLM 清空内置 prefix cache，保留 sidecar 和 cold object 状态。
  - 运行 8K reuse 阶段：`--prepass-before-reuse --cold-backend packed_v1 --cold-tier-restore`。
  - 写入研究报告 `docs/research/m3_15_8k_packed_prepass_gate.md`，并更新 `task_plan.md`、`findings.md`。
- 关键结果：
  - store 阶段：TTFT `5513.874ms`，connector store 求和 `3203.352ms`，`store_events=48`。
  - cold object：`packed_v1`，`1610616960` bytes，冷区仅 `packed_object.bin` + `packed_manifest.json`，hot per-layer files demote 后不存在。
  - PrePass：状态 `QUEUED`，耗时 `25.280ms`，枚举出 `SSD_COLD` required range 并入队 `1610616960` bytes restore。
  - restore：`cold_restore_status=COMPLETED`，checksum `ok`，executor `3046.794ms`，advance `3061.115ms`。
  - online reuse：`ADMIT/required_kv_ready_before_decode`，`ready_barrier_all_ready=true`，`sync_ssd_miss_total=0`，`external_load_observed=yes`，`load_events=1`，`store_events=0`，connector load `469.372ms`，online TTFT `710.835ms`。
  - restore-inclusive 口径：`prepass_elapsed + cold_advance + online_ttft = 3797.230ms`。该结果证明 8K packed PrePass 语义链路成立，但不能声称端到端 SLA 已满足。
- 修改文件：
  - `benchmarks/m3/run_reuse_smoke_matrix.py`
  - `tests/m3/test_reuse_smoke_matrix.py`
  - `docs/research/m3_15_8k_packed_prepass_gate.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 下一步：
  - 补同配置 8K B0/B2 baseline。
  - 跑 512/2K/8K compact packed PrePass matrix。
  - 继续优化 packed restore/load 数据面，尤其是 8K `3046.794ms` restore executor 与 `469.372ms` connector load。
- 验证：
  - `pytest tests/m3/test_reuse_smoke_matrix.py tests/m3/test_packed_cold_object.py -q`：`24 passed in 3.11s`。
  - `python -m py_compile benchmarks/m3/run_reuse_smoke_matrix.py benchmarks/m3/cold_tier.py`：通过。
  - `git diff --check`：通过。
