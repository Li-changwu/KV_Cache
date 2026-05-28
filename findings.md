# 发现与决策：1M Tokens KV Cache 分层管理

## 需求
- 用户希望依赖 `using-superpowers`、`context-engineering`、`planning-with-files-zh` 三个技能，理解并构思新项目。
- 当前核心输入是 `1M_Tokens_KV_Cache_分层管理技术方案.md`。
- 本轮输出需要包含：研究问题理解、整体架构理解、实施规划，并以 Markdown 文件形式记录。

## 研究发现
- 方案版本为 v0.5，日期 2026-05-24，范围是面向 Agent 长上下文、多轮交互和科学发现推理的精确 attention 多级 KV Cache 管理。
- 核心研究问题不是单纯扩容，而是让百万级上下文请求变成可预测、可准入、可观测、可降级的服务系统问题。
- 第一版坚持精确 attention 语义，不引入有损压缩、稀疏近似或语义检索替代。压缩、量化、稀疏注意力等只作为后续可插拔扩展。
- 系统采用 HBM / DRAM / 3FS(SSD) 三层：HBM 和 DRAM 是执行层，3FS/SSD 是冷容量层。
- 硬边界：SLA decode critical path 不允许同步读 SSD。所有本轮 decode 必需 KV 必须在 decode 前通过 admission / prefetch 补齐到 HBM 或 DRAM。
- 两条同等重要闭环：
  - Prefill 侧：通过 committed prefix/session KV 复用历史上下文，只对新增 token 做 delta prefill。
  - Decode 侧：通过 ready barrier 保证 required KV 已在 HBM/DRAM 可达集合中。
- 硬性验收指标包括：最大上下文 1M tokens、`TTFT_offload <= 1.2 * TTFT_DRAM_baseline`、适用场景多级 KV 命中率 `>= 90%`、committed ranges 默认 0 全量重算、storage/network sustained utilization `<= 70%`、SLA 队列同步 SSD miss 为 0。
- 整体执行路径：Gateway -> SLA Admission Guard -> correctness/prefix/manifest lookup -> admission window 内共享规划 -> PrefillReusePlan -> residency lookup -> PrefetchPlan / 3FS restore -> delta prefill -> decode ready barrier -> vLLM decode scheduler -> async writeback / eviction / metrics。
- 系统元数据核心包括 `KVCorrectnessKey`、`KVManifest`、`PrefixIndex`、`KVIndex`、`PrefillReusePlan`、`KVSharingPlan`、`PrefetchPlan`、`SchedulerDecision`。
- 差异化定位应集中在三点：面向 1M 精确上下文的准入控制、3FS 感知控制平面、在线服务隔离，而不只是“KV 可以跨 GPU/CPU/磁盘移动”。
- 请求生命周期关键路径：
  - Admission：查 manifest / correctness key / prefix index，生成 reuse plan，估算 TTFT、hit rate、带宽。
  - Prefill：复用 committed ranges 和 shared prefix，只对新增 user turn / tool result / suffix 做 delta prefill。
  - Pre-pass：在执行前解析 required KV blocks，避免按 miss 粒度触发 I/O。
  - Prefetch：把 HBM/DRAM/3FS/FETCHING 状态统一纳入 deadline 约束，critical blocks 必须按时 ready。
  - Decode：启动前 pin required blocks；若出现意外 SSD miss，暂停该 sequence 并重排，不能阻塞整个 batch。
  - Eviction/writeback：按 HBM/DRAM/3FS 水位和 predicted reuse 进行分层迁移。
- 命中率 `>=90%` 的实现方式不是单一替换算法，而是：限定适用 workload、admission window 内共享规划、准入前预测、prefetch 转换冷 miss、驻留策略保护可复用 KV、在线反馈收紧准入。
- vLLM 集成建议低侵入：外部 sidecar / gateway 做 `SLA Admission Guard` 和 `Tiered KV Control Plane`，通过 vLLM KV connector 接入 HBM/DRAM/3FS，不改 attention kernel 和核心 GPU block allocator。
- 第一版最小闭环应优先证明：同一 1M session 第二轮不 full prefill，只 delta prefill 新增 token；decode 前 ready barrier 生效；`sync_3fs_miss_total == 0`；能输出 TTFT、prefill reuse、effective hit rate、3FS bandwidth、admission decision。
- 本地方案引用的 `p1942-debrabant.pdf` 与 `Multi-Query_Optimization_for_Complex_Event_Processing_in_SAP_ESP.pdf` 当前不在 `/root/KV` 目录；如后续要写论文/相关工作，需要补齐原文或做外部文献校验。
- 外部主资料快速校验：
  - 3FS 官方 README 将 3FS 定位为面向 AI 训练和推理负载的高性能分布式文件系统，提到基于现代 SSD/RDMA 的共享存储层和 KVCache for inference。
  - vLLM 最新 disaggregated prefill 文档显示其已支持多类 KV connector，并将实现集中在 `vllm/distributed/kv_transfer`；这支持本方案“connector + sidecar 控制面”的低侵入落地路线。
  - vLLM prefix caching 设计采用 block hash，并把 parent hash、block tokens、LoRA / multimodal / salt 等 extra hashes 纳入唯一性；这与本方案的 strict correctness key 思路一致。
  - vLLM NIXL connector 文档支持异步 KV transfer、多实例和多轮 conversation proxy，但也提示需要 stateful proxy、CUDA device-buffer 等限制；本方案需要把这些限制纳入 M3/M4 风险。
  - DualPath arXiv 论文提出 storage-to-prefill 与 storage-to-decode 双路径加载，动机是 agentic LLM inference 的 KV storage I/O 瓶颈；这与方案第 6.5/9.11 的 DualPath 定位一致。
- 2026-05-24 新讨论结论：原先“仿真先行”的表述需要细化为“Benchmark 校准后的仿真”。用户判断先构建 benchmark、明确指标、测当前 vLLM 性能，再用真实硬件数据驱动仿真；该判断成立。
- 当前本机环境适合做 M1.5 校准：vLLM `0.19.0`、NVIDIA RTX A6000 约 48GB HBM、503GiB DRAM、Samsung 990 PRO 2TB NVMe、本地已有 `gpt-oss-20b` 等模型。
- vLLM 当前 `bench serve` 支持 `ttft/tpot/itl/e2el` percentile metrics、random dataset、prefix repetition dataset、结果 JSON 保存与 detailed per-request 信息，适合作为第一轮 benchmark harness 的基础。
- vLLM 官方 metrics 文档显示 `/metrics` 暴露 TTFT、TPOT/ITL、KV cache usage 等 Prometheus 指标；这些可以与 `vllm bench serve` 结果互相校验。
- 本机 vLLM CLI 启动时出现若干已有插件加载失败日志（`cachemoe`、`moe_infinity`、`pregated`），但 `vllm bench` 子命令仍可列出。正式 benchmark 前需要隔离或记录这些插件环境，避免污染性能结果。
- vLLM 插件系统读取 `VLLM_PLUGINS` 白名单；将环境变量设为空字符串 `VLLM_PLUGINS=` 可避免加载本机残留的 `cachemoe`、`moe_infinity`、`pregated` entry points。真实 benchmark 应使用该设置或在报告中记录插件污染。
- 本机设置了 `HTTP_PROXY/HTTPS_PROXY=socks5://11.11.11.7:1080`，会影响 localhost OpenAI API 请求并导致 `vllm bench serve` 出现 503。正式 benchmark 命令应清理代理变量或设置 `NO_PROXY=127.0.0.1,localhost`。
- `/root/models/gpt-oss-20b` 的模型配置已由 `collect_env.py` 采集：`num_hidden_layers=24`、`num_attention_heads=64`、`num_key_value_heads=8`、`head_dim=64`、`max_position_embeddings=131072`、`sliding_window=128`、`quant_method=mxfp4`。该模型适合校准系统路径，不适合代表 dense 1M attention。
- `vllm bench serve` 真实输出中，成功请求的 `errors` 列可能是空字符串列表（如 `["", ""]`），不能直接按列表长度计错；parser 已改为只统计非空错误。
- 用临时小矩阵验证了 runner-based integration smoke：random `input_len=128/output_len=8/num_prompts=2` 与 prefix repetition `prefix_len=256/suffix_len=64/output_len=8/num_prompts=2` 均 completed 2、failed 0，并可解析到 `vllm_latency.csv` 与 `apc_prefix_reuse.csv`。
- 轻量硬件 smoke 结果仅用于验证脚本路径，不是正式校准数据：4K Python fallback 顺序读约 1.75 GB/s，64MB H2D pinned copy 约 24.85 GB/s。正式 M1.5 仍需跑完整 block/tensor size 矩阵。
- 阶段 6 正式 benchmark 已完成：`results/benchmark_calibration/raw` 中有 92 个 vLLM raw JSON，`vllm_latency.csv` 有 44 行（41 OK + 3 ERROR），`apc_prefix_reuse.csv` 有 48 行（48 OK）。
- 3 个正式 `ERROR` 均来自 `input_len=32768` 且 `output_len` 为 1/16/128 时超出 `--max-model-len 32768` 的 Bad Request；这是 max-context 边界校验，不是 OOM，也不是 server crash。
- 正式 random/serving 关键观测：2048 长度 serving 在 rps4/c4 下约 2.419 req/s、309.7 tok/s、p50 TTFT 约 72.2 ms；8192 rps4/c4 约 2.16 req/s、276.5 tok/s、p50 TTFT 约 124.7 ms；16384 场景吞吐明显塌缩，rps4/c4 约 0.36 req/s、46.1 tok/s、p50 TTFT 约 4638.7 ms。
- 正式 APC prefix repetition 48 行全部 OK。`build_simulator_params.py` 当前按 `prefix_len` 折叠后的代表值显示：prefix 1024/4096/8192/16384 的 p50 TTFT 约为 61.96/96.23/120.79/167.47 ms。
- 正式 NVMe 校准未使用 fio，因为本机缺少 `fio`；`hardware_io.csv` 记录 `python_fallback_fio_missing`。4K/64K/1M/16M/64M 顺序读带宽约 2.735/8.766/8.472/7.274/2.413 GB/s，64M p50 latency 约 27.58 ms。
- 正式 H2D pinned-memory copy 带宽稳定在约 25 GB/s：64MB/256MB/1GB/4GB 分别约 24.797/25.111/25.076/24.974 GB/s。
- `results/benchmark_calibration/simulator_params.yaml` 已由 measured CSV 生成，`verification_status=MEASURED`，可以作为 M2 仿真器第一版输入。该 YAML 当前会按 input_len/prefix_len 对多种 output/concurrency 组合做最后值折叠，后续若要更精细仿真应改为保留维度或选择明确代表行。
- 用户已决定将后续真实基线模型改为 `/root/models/Qwen3-32B`。该模型本地配置为 `max_position_embeddings=40960`、`use_sliding_window=false`、`sliding_window=null`、`rope_scaling=null`、`num_hidden_layers=64`、`num_attention_heads=64`、`num_key_value_heads=8`、`head_dim=128`、`torch_dtype=bfloat16`。
- Qwen3-32B README 说明原生上下文最长约 32768 tokens；超过该长度建议使用 YaRN/RoPE scaling，并给出 vLLM 命令示例：`--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072`。因此 65536/131072 必须标记为扩展上下文实验，不能和默认原生上下文混为一谈。
- Qwen3-32B 的 tokenizer 配置 `model_max_length=131072`，但这不等于默认模型权重原生支持 131072；实际服务配置需要同时考虑模型配置、位置编码扩展参数和 `--max-model-len`。
- 粗略 KV cache 估算：Qwen3-32B 每 token KV 约 `64 layers * 8 kv heads * 128 head_dim * 2(K,V) * 2 bytes = 262144 bytes`，即约 256 KiB/token。单请求 32768 tokens 的 KV 约 8 GiB，131072 tokens 约 32 GiB，1M tokens 约 244 GiB。该估算不含模型权重和临时显存，说明 1M 必须依赖 HBM/DRAM/SSD 分层管理。
- 新实验顺序应改为 M1.6：先做 Qwen3-32B 加载可行性和小规模基线，再上探 32768/40960，并在需要时单独做 YaRN 扩展到 65536/131072；随后再进入 M2 1M 分层仿真。
- Qwen3-32B 加载实验给出关键边界：单张 RTX A6000 48GB 不使用主存卸载时，即便 `--max-model-len 2048` 也在权重构建阶段 OOM。日志显示显卡 44.55GiB 中仅剩约 50MiB，还需要再分配 100MiB。
- Qwen3-32B 使用 `--cpu-offload-gb 24` 后可以加载，日志显示 `Total CPU offloaded parameters: 24.28`，模型加载占用显存约 36.76GiB；默认图捕获阶段极慢，实际用 `--enforce-eager` 才完成服务启动。
- Qwen3-32B `--cpu-offload-gb 24 --enforce-eager --max-model-len 2048` 成功启动后，GPU KV cache size 为 10720 tokens，2K 请求最大并发约 5.23x。这说明短上下文可服务，但距离 32768/40960 上探仍缺少显存余量。
- Qwen3-32B 主存卸载极小推理结果：手动 `Hello` 输入、1 token 输出耗时约 6.682 秒；benchmark random 128 输入、1 输出 TTFT p50 约 4372.88ms；prefix repetition 128 前缀 + 16 后缀、1 输出 TTFT p50 约 24090.90ms。该结果说明主存卸载路径非常慢，不能作为理想 GPU-only 性能基线。
- Qwen3-32B 512 输入、16 输出、4 请求的 random smoke 在主存卸载路径下运行过慢，已中断并改为极小样本。这不是功能失败，而是当前硬件/卸载配置下的性能不可接受信号。
- 2026-05-25 规划更新：Qwen3-32B 后续实验采用“容量探测优先”。原因是当前最先暴露的问题不是 1M 策略细节，而是 32B bf16 权重和 KV cache 在单张 RTX A6000 上的可承载边界；真实实验显示这个边界强依赖 `--cpu-offload-gb`。
- 新增 `benchmarks/m1_5/probe_vllm_capacity.py` 与 `results/qwen3_32b_calibration/qwen3_capacity_probe.csv`，用于把 vLLM serve 启动日志解析为结构化容量结果。该表保留了无卸载 OOM、24GB 主存卸载容量失败、32GB 主存卸载成功等边界行。
- Qwen3-32B 容量探测结果：无主存卸载 `--max-model-len 2048` 权重加载 OOM；24GB 主存卸载下 2048 和 8192 可启动，GPU KV cache size 为 10720 tokens，8192 最大并发约 1.31x；24GB 主存卸载下 16384 失败，vLLM 报告需要 4.0GiB KV cache，但可用只有 2.62GiB，并估计最大模型长度 10720。
- Qwen3-32B 32GB 主存卸载结果：`--max-model-len 32768` 和 `40960` 均可启动，CPU offloaded parameters 约 32.45GB，模型加载显存约 28.58GiB，GPU KV cache size 为 44128 tokens；32768 最大并发约 1.35x，40960 最大并发约 1.08x。
- 上述结果说明：32768/40960 不是模型配置层面完全不能启动，而是在单卡 48GB 上必须牺牲大量权重驻留到主存；因此它可以作为容量边界事实，但不能作为理想 GPU-only 性能基线。继续跑 CPU-offload 长上下文吞吐矩阵的信息增益低。
- M2 仿真器第一版已实现：`benchmarks/m2/workloads.py` 生成 `session_append`、`shared_prefix`、`low_locality` 三类工作负载；`benchmarks/m2/simulator.py` 加载 Qwen3 measured calibration/capacity CSV，建立 HBM/DRAM/NVMe 三级命中与预取模型，输出 summary/request CSV 和 Markdown report。
- M2 第一版采用控制平面级仿真：SSD/NVMe 只能作为 decode 前预取来源，`sync_ssd_miss_rate` 固定约束为 0；低局部性 1M 请求默认不进入精确 SLA 快路径，而是延迟/降级。
- M2 严格 250ms deadline 结果：`session_append` 有效命中率约 0.688，`shared_prefix` 约 0.656，二者都节省大量 prefill tokens，但 deadline miss rate 约 0.875；`low_locality` 命中率 0，全部延迟。这说明本机 25GB/s H2D 和 44128 HBM KV tokens 下，1M 精确请求不能直接进入 250ms 级队列。
- M2 宽松 12000ms deadline 对照：`session_append` 和 `shared_prefix` 各 8 个请求中 7 个可准入，首个无复用请求延迟；`low_locality` 仍 8 个全部延迟。该结果支持当前研究主张：有局部性/复用的长上下文可以受益，随机低局部性 1M 应该排队、降级或拒绝。
- M2 第一版结果路径：`results/m2_simulation/strict_250ms/`、`results/m2_simulation/relaxed_12000ms/`、`results/m2_simulation/experiment_summary.md`。
- 阶段 9 已将 M2 仿真器扩展为敏感性分析工具：`SimulationConfig` 新增 `prefetch_lead_ms`、`h2d_parallelism`、`nvme_parallelism`、`reuse_prediction_error_rate`，用于表达提前预取、并发搬运和复用预测误差。
- 新增 `benchmarks/m2/sweep.py`，用 Qwen3 measured calibration 运行了 311040 行参数扫描，覆盖 deadline、预取提前量、H2D 带宽、NVMe/3FS 带宽、DRAM tokens、HBM tokens、并行度和预测误差。
- 阶段 9 结果路径：`results/m2_sweep/m2_sweep_summary.csv`、`results/m2_sweep/m2_sweep_boundaries.csv`、`results/m2_sweep/m2_sweep_deadline_frontier.csv`、`results/m2_sweep/sweep_report.md`。
- 阶段 9 边界结果：`low_locality` 在 250ms、1000ms、5000ms、12000ms、30000ms 下均没有进入 SLA 的区域，最佳 admitted rate 仍为 0。这进一步确认随机低局部性 1M 精确请求应被排队、降级或拒绝，不应承诺分层 KV 快路径收益。
- `session_append` 的 250ms 边界需要约 5000ms 提前预取、H2D 25GB/s x4、NVMe/3FS 20GB/s x4、DRAM 500000 tokens、HBM 44128 tokens，admitted rate 0.875，effective hit rate 0.6875。
- `shared_prefix` 的 250ms 边界更依赖极高搬运带宽：H2D 200GB/s x4、NVMe/3FS 50GB/s x2、DRAM 500000 tokens、HBM 262144 tokens，admitted rate 0.875，effective hit rate 0.65625。
- 按 deadline 前沿看，放宽到 5000ms 后，`session_append` 和 `shared_prefix` 都可在 H2D 25GB/s x4、NVMe/3FS 20GB/s 或 8.8GB/s x4 的假设下进入目标准入率；放宽到 12000ms 后，本机 H2D 25GB/s 搭配更低 NVMe/3FS 带宽即可进入复用型 SLA 区间。
- 阶段 9 边界选择规则已修正：优先低复用预测误差和高有效命中率，再比较 deadline、提前量、容量和带宽。原因是“预测少复用”会减少搬运量并让 deadline 更容易满足，但这不代表分层 KV 复用收益更好。
- M3 最小接口已定义在 `docs/specs/m3_connector_sidecar_minimal_interface.md`：sidecar 负责 correctness key、manifest/prefix lookup、admission、prefetch plan 和 ready barrier；vLLM connector 只执行 lookup/prefetch/pin/release；decode 前 `sync_ssd_miss_total` 必须为 0。
- 2026-05-25 新模型评估：`/root/models/Qwen2.5-14B-Instruct` 本地模型文件约 28GB，而 `/root/models/Qwen3-32B` 约 62GB。Qwen2.5-14B 更可能在 RTX A6000 48GB 上不依赖 CPU offload 启动，从而得到更干净的 vLLM/APC baseline。
- Qwen2.5-14B 本地配置：`num_hidden_layers=48`、`num_attention_heads=40`、`num_key_value_heads=8`、`head_dim=128`、`torch_dtype=bfloat16`、`max_position_embeddings=32768`、`use_sliding_window=false`、`rope_scaling=null`。虽然配置里有 `sliding_window=131072` 字段，但 `use_sliding_window=false`，因此默认不是滑动窗口路径。
- Qwen2.5-14B README 说明支持 128K 长上下文，但当前 `config.json` 默认只设置到 32768；超过 32768 需要添加 YaRN/RoPE scaling，且短上下文性能可能受静态 YaRN 影响。因此 131072 应作为扩展上下文实验标注，不应和默认原生 32768 混淆。
- Qwen2.5-14B 每 token KV 约 `48 layers * 8 kv heads * 128 head_dim * 2(K,V) * 2 bytes = 196608 bytes`，约 192KiB/token。相比 Qwen3-32B 的 256KiB/token，KV 容量压力降低约 25%。按 1048576 tokens 计算，Qwen2.5-14B KV 约 192GiB，Qwen3-32B 约 256GiB；按十进制 1000000 tokens 计算分别约 183GiB 和 244GiB。
- 初步判断：有必要将后续在线 benchmark 和 M3 原型默认基座从 Qwen3-32B 切换到 Qwen2.5-14B-Instruct。Qwen3-32B 的价值应保留为“大模型压力上界/反例”，而不是当前单卡开发主线。
- 已执行切换：Qwen2.5-14B-Instruct 现为默认在线实验模型。容量探测显示该模型在 RTX A6000 48GB 上无 CPU offload、`--enforce-eager`、`--gpu-memory-utilization 0.90` 下，`--max-model-len` 为 2048、8192、16384、32768 均可启动。
- Qwen2.5-14B 容量探测结果：GPU KV cache size 为 64752 tokens；2048/8192/16384/32768 的最大并发约为 31.62x / 7.90x / 3.95x / 1.98x。相比 Qwen3-32B 32768 需要 32GB CPU offload 且最大并发约 1.35x，Qwen2.5 更适合当前单卡在线实验。
- Qwen2.5-14B 极小基线结果：random 128 输入、1 输出、1 请求 TTFT p50 约 216.28ms；prefix repetition 128 前缀 + 16 后缀、1 输出、1 请求 TTFT p50 约 85.22ms。该结果可作为 smoke，不代表完整性能矩阵。
- `results/qwen25_14b_calibration/simulator_params.yaml` 已生成，`vllm.kv_bytes_per_token=196608`，`dense_attention_representativeness=full_attention_path_for_configured_context`。
- 用 Qwen2.5-14B 参数重跑 M2 sweep 后，`low_locality` 仍无 SLA 区域；`session_append` 在 250ms 边界下为 H2D 200GB/s x4、NVMe/3FS 2.5GB/s x1、DRAM 1000000 tokens、HBM 64752 tokens；`shared_prefix` 在 250ms 边界下为 H2D 200GB/s x4、NVMe/3FS 20GB/s x2、DRAM 500000 tokens、HBM 262144 tokens。相较 Qwen3，Qwen2.5 减轻了 KV/token 和无卸载运行负担，但 250ms 级 1M 精确服务仍主要受搬运带宽与复用形态约束。
- M3 第一版已实现最小 sidecar/control-plane 原型：`benchmarks/m3/control_plane.py` 包含 `CorrectnessKey`、`KVManifest`、`MockKVConnector`、`TieredKVControlPlane`、`ADMIT/DELAY/REJECT/FULL_PREFILL_FALLBACK` 决策和 ready barrier；`benchmarks/m3/replay.py` / `replay_cli.py` 支持离线请求 replay。
- M3 replay 使用 Qwen2.5 校准参数得到：`session_a_turn_1` 因无可复用前缀走 `FULL_PREFILL_FALLBACK`，`delta_prefill_tokens=1048576`；`session_a_turn_2` 复用 `786432` tokens，`delta_prefill_tokens=262144`，decision 为 `ADMIT`，`sync_ssd_miss_total=0`。
- ready barrier mock 已覆盖“缺失或未就绪 KV 不允许进入同步 SSD miss”的语义：当 manifest 中 required range 位于 SSD 且未 ready，sidecar 返回 `DELAY`，barrier `all_required_blocks_ready=false`，但 `sync_ssd_miss_total=0`。
- correctness key 不匹配时，sidecar 返回 `REJECT` 和结构化原因 `correctness_key_mismatch`，不会复用旧模型/旧 rope/tokenizer 下的 KV。
- M3.5 已新增 HTTP sidecar / OpenAI completions 前置代理：`POST /admit`、`POST /commit`、`GET /ready_barrier/{request_id}`、`GET /metrics` 和 `POST /v1/completions`。前置代理会先记录控制面 decision，再在配置上游时剥离 `m3_control` 字段转发到真实 vLLM。
- M3.5 sidecar 配置默认从 `results/qwen25_14b_calibration/simulator_params.yaml` 与 `qwen25_capacity_probe.csv` 推导 Qwen2.5-14B 参数，避免重新手填 `kv_bytes_per_token`、HBM KV tokens、H2D 和 NVMe 带宽。
- HTTP sidecar 转发时必须使用 `httpx.AsyncClient(trust_env=False)`，否则本机 `HTTP_PROXY/HTTPS_PROXY=socks5://11.11.11.7:1080` 会污染 localhost 转发。这和 vLLM benchmark 中的代理污染是同一类问题。
- M3.5 在线 smoke 已完成：Qwen2.5-14B vLLM 以 `--max-model-len 2048 --enforce-eager` 启动在 `127.0.0.1:8000`，sidecar 启动在 `127.0.0.1:8010` 并转发 `/v1/completions`；真实 vLLM 返回正常 completion，sidecar 日志记录 `online-smoke-1` 为 `ADMIT/no_reusable_prefix`，`sync_ssd_miss_total=0`。
- M3.6 接入点评估确认：HTTP sidecar 只能作为准入和日志前置层；真正替换 mock connector 必须实现 vLLM `KVConnectorBase_V1`。OpenAI completions/chat 协议已经支持 `kv_transfer_params`，该字段会进入 `Request.kv_transfer_params`，可作为 sidecar plan 进入 scheduler/worker connector 的最小通道。
- M3.6 第一版已新增 `benchmarks/m3/noop_connector.py`。`M3NoOpConnector` 会读取 sidecar 注入的 `kv_transfer_params`，只接受 `decision=ADMIT` 且 `sync_ssd_miss_allowed=false` 的计划，将 `reuse_tokens` 按 block size 向下对齐，再通过 `update_state_after_alloc()` 和 `build_connector_meta()` 记录 local block ids 与 required ranges。
- sidecar 的 `/v1/completions` 现在可通过 `--inject-kv-transfer-params` 将 `m3_control` 翻译为 vLLM OpenAI 协议原生 `kv_transfer_params`，并继续剥离 `m3_control`，让上游 vLLM 只看到标准字段。
- vLLM `KVConnectorFactory` 能通过 `kv_connector_module_path="benchmarks.m3.noop_connector"` 动态加载 `M3NoOpConnector`，说明 M3.6 具备真实 vLLM 启动 smoke 的配置前提。
- M3.6 真实在线 smoke 已完成：Qwen2.5-14B vLLM 以 `--kv-transfer-config '{"kv_connector":"M3NoOpConnector","kv_role":"kv_both","kv_connector_module_path":"benchmarks.m3.noop_connector","kv_load_failure_policy":"fail"}'` 启动成功，日志显示 `Creating v1 connector with name: M3NoOpConnector`。sidecar 以 `--inject-kv-transfer-params` 转发短请求后，vLLM 返回正常 completion，响应中包含 `kv_transfer_params.m3_noop_connector.saved_block_ids=[1]`，证明请求进入了自定义 connector 的 `request_finished()` 路径。

## 技术决策
| 决策 | 理由 |
|------|------|
| 使用持久化规划文件组织本次研究 | 该任务涉及多步骤阅读、归纳和规划，符合 `planning-with-files-zh` 使用场景 |
| 采用分层上下文组织方式 | `context-engineering` 建议将规则、方案、相关文件、验证输出分层管理 |
| 第一轮规划以独立仿真/MVP 和 vLLM 最小接入双轨推进 | 方案同时包含控制平面、元数据、调度策略与 vLLM 接入点；先用仿真固化指标与策略，再做最小在线接入可降低风险 |
| 将 vLLM 作为第一集成目标，但先完成离线仿真与最小 connector 验证 | vLLM 连接器和 prefix caching 基础已存在，但 API 与限制需要随版本验证；仿真可先固化策略收益 |
| 第一版只承诺适用 workload 的 1M 精确 SLA | 方案明确随机低局部性负载不适用 90% hit rate，准入层必须诚实拒绝或迁移 |
| 在 M1 与 M2 之间插入 M1.5：Benchmark 与硬件校准 | 仿真必须以真实 vLLM TTFT/TPOT、APC 收益、KV usage、NVMe/3FS latency、H2D bandwidth 为输入，否则无法支撑工程或论文结论 |
| M1.5 harness 不修改 vLLM internals | 第一阶段目标是 baseline characterization 与仿真输入校准，不提前进入 connector/kernel 改造 |
| runner 在 vLLM 子命令失败且没有写出 JSON 时主动生成失败 JSON | 确保 OOM/ERROR 行不会在后续 CSV 中静默丢失 |
| `build_simulator_params.py` 同时生成 `report.md` | simulator YAML 给机器消费，report 给研究记录/异常审计消费 |
| 阶段 6 后进入 M2 仿真器第一版 | benchmark/calibration layer 已给出 measured 输入，下一步应验证 admission/prefetch/residency 策略而不是继续扩 benchmark |
| M2 前插入 M1.6 Qwen3-32B 校准 | Qwen3-32B 更接近完整 attention 路径；必须先获得模型相关真实曲线，再做 1M 分层仿真 |
| Qwen3-32B 当前只能以主存卸载路径做极小冒烟 | 单张 A6000 48GB 无法承载该 32B bf16 模型的全 GPU 权重加载；后续若要可靠上探，需要多卡、量化或更大显存 |
| Qwen3-32B 长上下文实验先测服务容量，不先测吞吐 | 当前主存卸载路径极慢，完整矩阵的信息增益低；服务能否启动到 8192/16384 是更紧要的分界线 |
| M2 仿真应消费 Qwen3 容量边界，而不是等待完整 Qwen3 CPU-offload 性能曲线 | 32GB 主存卸载虽可启动 32K/40K，但执行路径已经和目标分层系统不同；仿真更需要 KV 尺寸、HBM/DRAM/NVMe 带宽和容量约束 |
| M2 第一版先做准入/预取/复用边界，不做内核级性能复现 | 这能最快回答研究问题中的正负样本边界；真实 vLLM connector 可在策略成立后接入 |
| 阶段 9 sweep 使用低预测误差优先的边界排序 | 避免把“少复用导致少搬运”的伪优势误判为更好的分层 KV 收益 |
| M3 以 sidecar/connector 契约为主，不直接改 vLLM 内核 | 当前最需要验证的是准入、预取、ready barrier 和低局部性降级这些控制面语义 |
| 建议将 M3/M1.7 默认实验模型切换到 Qwen2.5-14B-Instruct | 它保留完整注意力路径，KV/token 更低，模型文件更小，更可能避免 CPU offload，能让 benchmark 和 connector 原型更干净 |
| Qwen2.5-14B 已正式成为默认在线实验模型 | 实测 32768 无 CPU offload 可启动，GPU KV cache 64752 tokens，适合后续 M3 sidecar / connector 原型 |
| M2/M3 必须使用模型专属 KV/token | Qwen2.5 是 196608 bytes/token，Qwen3 是 262144 bytes/token；继续硬编码会污染容量和搬运时间结论 |
| M3 第一版采用离线 replay + mock connector | 该方式能先验证控制面硬语义，再逐步替换为真实 vLLM 前置代理或 KV connector |
| M3.5 先实现旁路记录型 HTTP sidecar，不拦截真实 KV | 这样能把 correctness key、prefix candidates、decision、ready barrier 和 metrics 接入真实请求路径，同时保持 vLLM 内部语义不变 |
| M3.6 应先做 no-op connector smoke，再做真实 tensor copy | vLLM connector API 是实验接口，且 KV tensor layout 与 attention backend 有关；先验证 scheduler 外部 token 路径能降低风险 |
| `kv_transfer_params` 是 sidecar 到 vLLM connector 的最小控制通道 | vLLM OpenAI 协议和 `Request` 已支持该字段；无需在第一版修改 vLLM 请求协议 |
| M3.7 开始处理真实 KV tensor copy | M3.6 已证明控制参数通路和 connector 生命周期可用，下一步瓶颈转为 KV tensor layout、block 对齐和 load failure 语义 |
| M3.7 本地块级张量复制原型已形成 | 新增 `KVTensorStore` 与连接器可选存储路径，已能保存/加载小型分页 KV 张量，并用正确性键和布局校验保护复用 |
| M3.7 真实在线两轮短前缀冒烟通过 | Qwen2.5-14B + 自定义 connector 生成 `session-a` manifest，第二轮 sidecar 决策复用 64 tokens，`sync_ssd_miss_total=0` |
| M3.8 先补可观测性和自动提交，再跑在线小矩阵 | M3.7 的主要人工步骤是手动 commit，且保存/加载耗时不可结构化分析；先稳定这些基础，矩阵结果才可解释 |

## M3.8 新发现
- 连接器新增 `m3_event_log_path`，会把 `store_layer` 与 `load_request` 写成 JSONL，并维护保存/加载成功、失败、令牌数计数。这样后续小矩阵可以分析真实保存/加载耗时，而不是只看端到端请求耗时。
- 前置代理已支持从上游 vLLM 响应的 `kv_transfer_params.m3_noop_connector.manifest` 自动提交控制面 manifest。提交时使用 sidecar 请求编号，而不是 vLLM 内部请求编号，避免 `cmpl-*` 或内部编号和控制面不一致。
- 新增 `benchmarks/m3/run_reuse_smoke_matrix.py`，默认展开 `16/64/128/256` tokens 前缀复用小矩阵，支持 dry-run 生成 `reuse_smoke_matrix.csv` 和 `report.md`，在线模式可读取连接器事件日志汇总保存/加载耗时。
- 如果第二轮请求的全部提示词都声明为外部 KV 命中，vLLM 0.19.0 scheduler 会触发 `assert num_new_tokens > 0`。因此真实复用请求必须至少包含一个新增后缀，符合多轮会话追加场景。
- 已完成 `16+16` 与 `64+16` 两个在线请求成功样本：sidecar 自动提交有效，第二轮均为 `ADMIT/required_kv_ready_before_decode`，HTTP 请求均返回 200。但 connector 事件日志只有 `store_layer`，没有 `load_request`，说明本次请求成功很可能被 vLLM 内置前缀缓存覆盖，尚不能声称真实外部 KV 加载成功。
- 重启 vLLM 对照已完成：第一阶段保存 `m3-8-64` 前缀后关闭并重启 vLLM，保留同一个 sidecar 控制面和 tensor store；第二阶段复用 `64+16` 请求返回 200，CSV 中 `external_load_observed=yes`，连接器事件日志出现 1 次 `load_request ok`，加载耗时约 24.635ms。
- 完整重启对照矩阵已完成，结果路径为 `results/m3_8_full_restart_matrix/`。保存阶段与复用阶段各 4 行均为 OK，覆盖 `16/64/128/256` tokens 前缀、`16` tokens 新增后缀、`1` token 输出。复用阶段 4/4 行均为 `external_load_observed=yes`，说明四个长度都绕过了 vLLM 内置前缀缓存并进入外部 KV 加载路径。
- 完整矩阵复用阶段关键数据：`16` tokens 加载约 35.397ms、端到端约 429.867ms；`64` tokens 加载约 45.081ms、端到端约 273.306ms；`128` tokens 加载约 50.360ms、端到端约 283.603ms；`256` tokens 加载约 63.692ms、端到端约 339.172ms。本轮小矩阵主要证明路径正确与可观测，不应解读为稳定性能曲线。
- 新增 `benchmarks/m3/summarize_restart_matrix.py` 产出 `restart_matrix_summary.json` 与 `restart_matrix_report.md`。本轮汇总为 `store_rows=4`、`reuse_rows=4`、`external_load_rows=4`、`total_store_elapsed_ms=383.038`、`total_load_elapsed_ms=194.53`。
- 输出一致性 smoke 已完成，路径为 `results/m3_8_full_restart_matrix/consistency_smoke/`。普通全量路径输出 `" cache"`，重启 vLLM 后复用路径输出 `" cache"`，`texts_match=true`，且本次复用新增观测到 1 次 `load_request ok`，加载耗时约 32.618ms。
- 一致性 smoke 必须分阶段运行：先保存普通路径输出，再重启 vLLM，然后复用路径比较。否则普通路径会先把同一提示词放入 vLLM 内置前缀缓存，后续复用请求可能不经过外部 connector load。

## M3.9 新发现
- M3.9 已进入 DRAM/NVMe 分层迁移原型。当前实现仍是本地 tensor store 元数据级迁移，不是真实异步 I/O 或 3FS 接入。
- `KVBlockManifest` 新增 `tier` 与 `ready` 字段，默认保存后的前缀为 `DRAM` 且 `ready=true`，保持对旧 manifest 的向后兼容。
- `KVTensorStore.demote_to_nvme(prefix_id)` 会把前缀标记为 `NVME` 且 `ready=false`；`prefetch_to_dram(prefix_id)` 会把它恢复到 `DRAM` 且 `ready=true`；每次迁移都会写入该前缀目录下的 `migration_events.jsonl`。
- `KVTensorStore.load_layer()` 现在会在 correctness key、layout、token ids 检查前先检查层级可达性。当前只允许从 `HBM` 或 `DRAM` 且 ready 的前缀加载；如果前缀仍在 `NVME`，会抛出 `TierNotReady`，要求先预取到 DRAM。
- connector 已覆盖“未预取 NVMe 不能加载”的单元测试：当 prefix 被 demote 到 NVMe 后，`start_load_kv()` 会记录 `load_request error` 并设置 load error block ids；显式 `prefetch_to_dram()` 后同一前缀可正常加载。
- 新增离线冒烟脚本 `benchmarks/m3/run_tier_migration_smoke.py`，对 `results/m3_8_full_restart_matrix/tensor_store` 中的 `m3-8-16/64/128/256` 执行 `DRAM -> NVME -> DRAM` 状态循环。
- M3.9 真实数据冒烟结果路径为 `results/m3_9_tier_migration_smoke/`，4 个前缀均为 OK。CSV 中 token 数分别为 32、80、144、272；这些是当前 manifest 在 M3.8 复用/一致性 smoke 后记录的最新 token_end，不再等同于最初前缀长度。
- 当前 M3.9 的关键意义：把“KV 文件存在”与“decode 前是否执行可达”分开，开始落实“容量层不能在关键路径同步读取”的非协商约束。
- HTTP sidecar 现在可以通过 `tensor_store_path` 绑定本地张量仓库。若准入结果为 `DELAY` 且原因为 `required_kv_not_ready_before_decode` 或 `prefetch_misses_admission_window`，sidecar 会在记录本次延迟决策后调用 `KVTensorStore.prefetch_to_dram()`，并把控制面 manifest 标记为 `DRAM/ready`。
- sidecar 预取烟测路径为 `results/m3_9_sidecar_prefetch_smoke/`。真实 M3.8 张量仓库上的结果为：第一次 `DELAY/required_kv_not_ready_before_decode`，日志中出现 `source=prefetch_to_dram`，第二次同一前缀变为 `ADMIT/required_kv_ready_before_decode`。
- 该 sidecar hook 的边界很重要：它没有让当前请求边等待边同步读 NVMe，而是让当前请求继续延迟，预取完成后由后续请求准入。这符合“decode 关键路径不同步读 SSD/3FS”的约束。
- 新增 `benchmarks/m3/prefetch_queue.py`。`PrefetchQueue` 用 `tokens * kv_bytes_per_token / storage_gbps` 估算 NVMe 到 DRAM 的迁移耗时，只有预计完成时间不超过 `deadline_ms` 时才调用 `prefetch_to_dram()`；否则返回 `DEADLINE_MISS`，保留 manifest 为 `NVME/ready=false`。
- sidecar 现在通过 `PrefetchQueue` 调度预取，而不是直接改状态。`decisions.jsonl` 的 `prefetch_to_dram` 记录中新增 `prefetch_results`，包含 `status`、`deadline_miss`、`duration_ms`、`estimated_ready_ms`、`queue_depth_before` 和 `bytes`。
- 新增队列版真实烟测结果：`results/m3_9_sidecar_prefetch_queue_smoke/ok/` 中 `m3-8-64` 预计约 `1.787ms` 完成，第二次准入 `ADMIT`；`results/m3_9_sidecar_prefetch_queue_smoke/deadline_miss/` 中人为设置极低带宽，`m3-8-256` 预计约 `53477376ms`，第二次仍为 `DELAY`。
- 当前队列仍是确定性元数据队列，不是真实后台线程，不做真实文件搬运，也没有 3FS/H2D/并发 I/O/tail latency 建模。
- `PrefetchQueue.stats()` 已新增队列统计：请求数、完成数、deadline miss 数、累计字节、最大队列深度、最近预计完成时间和队列虚拟可用时间。
- sidecar `/metrics` 已接入 `prefetch_queue_*` 指标。当前可观测 `prefetch_queue_requests_total`、`prefetch_queue_completed_total`、`prefetch_queue_deadline_miss_total`、`prefetch_queue_bytes_total`、`prefetch_queue_max_depth`、`prefetch_queue_last_estimated_ready_ms`、`prefetch_queue_available_at_ms`。
- 新增多请求队列烟测脚本 `benchmarks/m3/run_prefetch_queue_smoke.py`。真实 M3.8 张量仓库结果路径为 `results/m3_9_prefetch_queue_smoke/`，4 个前缀中 2 个完成、2 个 deadline miss，累计字节 `103809024`，最近预计完成时间 `37748.736ms`。
- 多请求队列烟测直接提交队列，验证排队估算，不会先检查 manifest 是否已经驻留 DRAM。因此它表达的是“如果本轮需要调度预取，是否赶得上”，不是完整准入层的 residency hit 判断。
- sidecar 已实现驻留感知入口：若控制面旧状态仍为 `SSD/ready=false`，但本地 tensor store 中对应前缀已经是 `DRAM/ready` 或 `HBM/ready`，且 correctness key 完全匹配、覆盖 required range，则当前请求会先同步控制面并重新准入为 `ADMIT`，不会进入 `PrefetchQueue`。
- 新增 sidecar 指标 `residency_hit_total`、`prefetch_queued_total`、`prefetch_deadline_miss_total`，用于把已驻留命中、冷层预取入队和截止时间错过分开观察。
- correctness key 不匹配的 tensor manifest 不能被预取成 ready。sidecar 在提交 `PrefetchQueue` 前会检查本地 manifest 的 correctness key 和覆盖范围；不匹配时只记录错误，保持请求 `DELAY`。
- 新增 `benchmarks/m3/run_residency_aware_prefetch_smoke.py`，输出 `RESIDENCY_HIT`、`PREFETCH_COMPLETED`、`DEADLINE_MISS` 三类状态。真实 M3.8 张量仓库结果路径为 `results/m3_9_residency_aware_prefetch_smoke/`：4 个前缀中 3 个 `RESIDENCY_HIT`，`m3-8-256` 为 `DEADLINE_MISS`，队列请求数为 1。
- 新增 `AsyncPrefetchQueue`。它与同步 `PrefetchQueue` 的关键差异是：入队成功返回 `QUEUED`，不会立刻调用 `prefetch_to_dram()`；只有显式推进后台队列后才返回 `COMPLETED` 并把 manifest 改为 `DRAM/ready=true`。
- sidecar 新增可选 `async_prefetch` 配置，CLI 对应 `--async-prefetch`。异步模式下，重复请求同一前缀时会返回 `ALREADY_QUEUED`，不会重复增加 `prefetch_queued_total`。
- 新增 `benchmarks/m3/run_async_prefetch_smoke.py`。真实 M3.8 张量仓库结果路径为 `results/m3_9_async_prefetch_smoke/`：第一次请求 `DELAY/QUEUED`，推进前第二次仍 `DELAY`，推进后台队列后第三次 `ADMIT`，最终 manifest 为 `DRAM/ready=true`。
- 缺少 tensor store manifest 的情况现在被视为预取错误，而不是“入队”。这可以防止控制面把没有源数据的 prefix 误统计为预取请求。
- `run_reuse_smoke_matrix.py` 已接入 sidecar `/metrics` 和 sidecar 决策日志。每行在线矩阵现在记录 `decision`、`reason`、`ready_barrier_all_ready`、`sync_ssd_miss_total`、`residency_hit_delta`、`prefetch_queued_delta`、`prefetch_deadline_miss_delta`、`prefetch_queue_requests_delta`、`prefetch_queue_completed_delta`、`prefetch_queue_deadline_miss_delta`、`prefetch_queue_bytes_delta` 和 `prefetch_queue_pending`。
- HTTP sidecar `/commit` 现在支持 `ready` 字段，能显式把已提交前缀写成 `SSD/ready=false`，用于构造真实在线矩阵中的“控制面知道前缀存在，但当前执行层不可达”的样本。
- 真实在线小矩阵指标样本已生成到 `results/m3_9_reuse_metrics_online/fresh_sidecar/reuse_phase/`。在同一个 Qwen2.5-14B + M3NoOpConnector 在线服务中，`m3-8-16` 行为 `ADMIT/required_kv_ready_before_decode`，`residency_hit_delta=1`，`external_load_observed=yes`；`m3-8-256` 行为 `DELAY/required_kv_not_ready_before_decode`，`prefetch_queued_delta=1`，`prefetch_queue_completed_delta=1`，`prefetch_queue_bytes_delta=53477376`，`sync_ssd_miss_total=0`。
- 上述在线样本说明小矩阵已经能区分“请求返回成功”“控制面准入或延迟”“本地张量仓库驻留命中”“冷层预取入队/完成”四类事实。尤其是 `m3-8-256` 虽然 HTTP 转发仍返回 200，但 sidecar 决策列明确记录本轮应为 `DELAY`，不能把它解读为当前请求已经安全复用容量层 KV。
- 为构造该样本，测试过程中先将 `m3-8-256` manifest 降级为 `NVME/ready=false`；在线请求触发同步 `PrefetchQueue` 后最终又被标记回 `DRAM/ready=true`。这仍是元数据级迁移，不是真实设备级搬运。

## KV Anti-Caching 设计深化
- KV Anti-Caching 方向可行性评分为 4/5。原因是当前 M3.9 已有 sidecar、manifest、tier state、ready barrier、deadline-aware prefetch queue 和在线小矩阵指标，可直接承接 Anti-Caching 的 pre-pass / non-blocking restore / requeue 语义。
- KV Anti-Caching 方向创新性评分为 4/5。单纯 SSD-backed KV 或 offload 已经很拥挤，Tutti、LMCache、Mooncake、DualPath 都是强相关工作；真正可写成论文贡献的是 KV 专用的 memory-primary anti-cache semantics、KVPrePass admission、cold-object construction、partial promotion 和 budgeted no-sync-miss scheduler。
- Tutti 主要解决本地 SSD-backed KV 的 fast data path，重点是 GPU-centric object store、GPU io_uring、SGL 和 slack-aware I/O；它没有完整解决哪些 KV 应该冷却、何时 restore、如何准入、如何保证 admitted SLA hit rate >= 90% 和如何处理 3FS/shared cold tier 的全局调度问题。
- DeBrabant 等人的 Anti-Caching 可以映射为：Evicted Table -> `KVResidencyTable`，Block Table -> `ColdBlockTable`，tuple-level eviction -> KV range 选择，pre-pass -> admission 前枚举 required KV，abort/requeue -> `DELAY_RESTORE` / `REQUEUE_AFTER_READY`，tuple-merge -> partial KV promotion，lazy compaction -> cold object valid bitmap + 后台整理。
- 第一版 KV Anti-Caching 不应承诺 low-locality 1M 请求低 TTFT。适用场景应限定为多轮 session append、共享长文档、agent trace、idle restore 等有历史 KV 复用的 workload，并把 low-locality 作为负对照报告。
- 设计文档已写入 `/root/KV/docs/research/kv_anti_caching_design.md`。该文档建议下一步按 M3.10-A 真实 cold-tier executor、M3.10-B KVPrePass/partial promote、M3.11 Budgeted Anti-Cache Scheduler、M4 策略族与强 baseline 推进。

## M3.10-A 新发现
- `KVTensorStore` 已从 M3.9 的纯元数据迁移推进到真实 cold-tier 目录迁移：`demote_to_cold_object()` 会把 prefix 热目录中的 layer `.safetensors` 文件移动到 cold root 下的 object 目录，并把 manifest 标成 `LOCAL_NVME`/`NVME`、`ready=false`。
- `KVBlockManifest` 已增加 `cold_uri`、`object_id`、`offset_table`、`checksum`、`size_bytes`。其中 `offset_table` 第一版按 layer 文件记录 file name、逻辑 offset、size 和每文件 checksum；整体 checksum 是 cold object manifest 级摘要。
- `restore_from_cold_object()` 会在恢复前校验 cold object checksum，成功后把 layer 文件复制回热目录并标记 `DRAM/ready=true`；校验或文件缺失失败会写入 `migration_events.jsonl`，`checksum_status=failed`，不会把 manifest 误标 ready。
- `demote_to_nvme()` 和 `prefetch_to_dram()` 保持兼容：有真实 layer 文件时走 cold object 迁移/恢复；旧的无 layer fixture 仍走元数据 fallback，避免现有控制面测试被测试夹具破坏。
- `PrefetchQueue` / `AsyncPrefetchQueue` 现在是确定性 cold-tier restore executor：同步队列在 deadline 允许时立即执行真实 restore，异步队列在 `advance_ready()` 时才执行真实 restore。二者都记录 `actual_bytes`、`executor_elapsed_ms`、`object_id`、`cold_uri`、`checksum_status`。
- `run_tier_migration_smoke.py` 已升级为 M3.10-A 真实 cold-tier 迁移烟测，支持 `--cold-root`，报告 cold URI、真实迁移字节数、demote/restore 耗时和 checksum 状态。
- 当前仍不是生产级 3FS executor：没有 io_uring/GDS/RDMA、线程池、3FS queue depth、多资源调度、真实 p95/p99 tail 聚合，也尚未把真实 cold-tier 路径重新跑进在线 vLLM 小矩阵。
- 在线小矩阵已接入真实 cold-tier 路径。`run_reuse_smoke_matrix.py` 的 cold mode 现在会在 store 后执行真实 `demote_to_cold_object()`，再通过 sidecar `/commit` 将同一 prefix 标记为 `SSD/ready=false`，随后只用 `/admit` 做 cold probe，避免把应延迟的请求转发给 vLLM 并污染内置 prefix cache。
- `KVManifest` 现在对同一 prefix/range 使用最新 epoch 语义：后提交的 `SSD/ready=false` 会覆盖旧的 `DRAM/ready=true`，而 `/prefetch/advance` 完成后追加更新的 `DRAM/ready=true`。这修复了 cold demote 后控制面仍可能错误 `ADMIT` 的问题。
- 同进程 online 样本路径为 `results/m3_10_cold_tier_online/reuse_phase/`。该样本证明 `DELAY -> QUEUED -> checksum ok restore -> ADMIT` 成立：`cold_probe_decision=DELAY`，`cold_restore_status=COMPLETED`，`cold_restore_actual_bytes=12587136`，`cold_restore_checksum_status=ok`，`sync_ssd_miss_total=0`。但同进程最终 `load_events=0`，说明 vLLM 内置 prefix cache 仍可能遮蔽外部 connector load。
- 重启 vLLM 后的端到端样本路径为 `results/m3_10_cold_tier_online/restart_probe/`。保存阶段写出 48 层 KV；demote 后 cold object 大小 `12587136` bytes；复用阶段先 `DELAY`，`/prefetch/advance` 真实恢复并 checksum ok，最终 vLLM 复用请求 `ADMIT/required_kv_ready_before_decode` 且 connector 事件出现 `load_request ok` 1 次、48 层、64 tokens、加载耗时约 `37.866ms`。
- M3.10 在线样本的核心意义：首次把 sidecar 控制面、真实 cold object 文件迁移、异步 restore executor、vLLM connector 外部 KV load 串成端到端闭环，并保持 decode critical path 不同步读 SSD/3FS。
- 完整 16/64/128/256 cold-tier restart 小矩阵已生成到 `results/m3_10_cold_tier_online/restart_matrix/`。四个前缀均为 `status=OK`，cold probe 均为 `DELAY/required_kv_not_ready_before_decode`，`/prefetch/advance` 均为 `COMPLETED` 且 `checksum_status=ok`，最终复用请求均为 `ADMIT/required_kv_ready_before_decode`，`ready_barrier_all_ready=true`，`sync_ssd_miss_total=0`。
- 该矩阵的真实 restore 字节数分别为：16 tokens `3149952` bytes，64 tokens `12587136` bytes，128 tokens `25170048` bytes，256 tokens `50335872` bytes。connector 事件中每个 prefix 均出现 1 次 `load_request ok`，说明重启 vLLM 后外部 KV load 路径被真实触发。
- 3FS adapter 第一版已启动：`benchmarks/m3/cold_tier.py` 提供 `ColdTierAdapter` 协议、`LocalColdTierAdapter`、`ThreeFSColdTierAdapter` 和 `build_cold_tier_adapter()`。当前 `ThreeFSColdTierAdapter` 是 3FS mount 上的 POSIX 文件语义适配器，已接入 `run_reuse_smoke_matrix.py`、`run_tier_migration_smoke.py`、`PrefetchQueue` / `AsyncPrefetchQueue` 和 sidecar CLI 的 `--cold-backend 3fs_posix` 参数。
- 3FS adapter smoke 已生成到 `results/m3_10_3fs_adapter_smoke/`：在模拟 3FS mount 目录上完成两个 prefix 的 `DRAM -> NVME -> DRAM` 迁移，summary 为 `status=OK`、`cold_backend=3fs_posix`、`bytes_total=672`。这证明业务路径已经能通过 adapter 选择后端，但仍不是真实 3FS 集群/RDMA/GDS/io_uring executor。
- 新增 cold-tier adapter benchmark：`benchmarks/m3/run_cold_tier_adapter_bench.py` 可以对任意 `cold_root` 和 `cold_backend` 生成合成 KV prefix，执行真实 demote/restore，并输出 `adapter_bench.csv`、`adapter_bench_summary.json`、`adapter_bench_report.md`。字段包括 demote/restore ms、MiB/s、checksum、hot file restore 状态和 p50/p95/p99 summary。
- 新增 adapter bench 汇总：`benchmarks/m3/summarize_cold_tier_adapter_bench.py` 能把 local/3fs_posix 多个 summary 合成 `adapter_bench_comparison.csv` 与 Markdown 报告。当前 smoke 结果在 `results/m3_10_cold_tier_adapter_bench/summary/`，local_posix 和 3fs_posix 都是本机目录级验证，不代表真实 3FS 集群性能。
- 当前 adapter benchmark smoke 使用 16/64/128/256 tokens、3 repeats、4 synthetic layers。`local_posix` summary：rows=12、bytes_total=716736、demote p95=5.341ms、restore p95=17.018ms；`3fs_posix` summary：rows=12、bytes_total=716736、demote p95=1.376ms、restore p95=1.752ms。两者差异来自本机目录/运行抖动，不能用于声称 3FS 性能优于本地 SSD。
- adapter benchmark 已扩展 queue-depth / batch restore 统计和 `qwen25_14b_tiny` profile。该 profile 使用 Qwen2.5-14B 形状的缩小层数配置：4 layers、8 kv heads、head_dim=128、bfloat16，用于在本机快速暴露 restore batch 延迟、有效吞吐和并发读写抖动。
- queue-depth=4 的 qwen25 tiny-shaped 结果已生成到 `results/m3_10_cold_tier_adapter_bench_qd/summary/adapter_bench_comparison.md`。`local_posix`：rows=8、bytes_total=15207168、restore batch p95=13.584ms、effective restore p50=545.648MiB/s；`3fs_posix`：rows=8、bytes_total=15207168、restore batch p95=68.175ms、effective restore p50=202.162MiB/s。
- 上述 `3fs_posix` 仍是本地目录模拟 3FS mount 的 POSIX 路径，不是 DeepSeek 3FS 集群/RDMA/GDS/io_uring 原生结果。它的价值是固定 backend adapter、batch CSV/schema 和比较报告格式；真实性能结论必须在实际 3FS 挂载点上复跑。
- KV Anti-Caching PPT 叙事已明确区分“问题”和“方法”：真正问题是现有推理服务无法在精确 attention 语义下，把百万级历史 KV 同时变成容量上可保存、执行前可恢复、decode 时必定 ready、且 TTFT 满足 SLA 的服务状态；Anti-Caching 是为解决该问题提出的 memory-primary 控制协议，而不是问题本身。
- KV Anti-Caching PPT 叙事已进一步修正为 Prefill + Decode 双阶段问题：cold miss 在 Prefill reuse 前会导致等待 restore 或退化成历史重算，直接放大 TTFT；cold miss 在 Decode execution 中会导致同步冷层 I/O，放大 ITL/TPOT 和尾延迟。因此系统不变量应是 Prefill 复用前 ready + Decode 执行前 ready，而不只是 decode path no-sync-miss。
- 2026-05-26 有效性复盘结论：当前 KV Anti-Caching **有用但尚未证明比强 baseline 更有效**。M3.10 已证明 correctness/control-plane invariant：cold-tier probe 返回 DELAY，真实 restore checksum ok，ready 后 ADMIT，restart 后 connector `load_request ok`，并保持 `sync_ssd_miss_total=0`。但这只证明路径与语义，不证明 1M SLA、TTFT 增幅 <=20%、命中率 >=90% 或相对 vLLM APC / LMCache / Mooncake / Tutti / DualPath / CacheFlow / KVDrive 的性能优势。
- 当前有效性证据分四层：语义正确性强，容量层路径中等，小规模性能证据弱，强 baseline 胜出证据缺失。最诚实表述应是“已经验证 exact KV Anti-Caching 的必要系统不变量，但 end-to-end performance superiority 仍需 512-32K、真实 3FS/生产 SSD 和强 baseline 矩阵验证”。
- M3.10 restart 小矩阵的关键证据：16/64/128/256 四个 prefix 均 `status=OK`、`cold_probe_decision=DELAY`、`cold_restore_status=COMPLETED`、`cold_restore_checksum_status=ok`、最终 `decision=ADMIT`、`ready_barrier_all_ready=true`、`sync_ssd_miss_total=0`、`external_load_observed=yes`、`load_events=1`。真实 restore 字节数分别约 3.15MB、12.59MB、25.17MB、50.34MB。
- M3.10 queue-depth adapter benchmark 只固定接口和报告格式，不能当作真实 3FS 结论。`qwen25_14b_tiny` qd4 下 `local_posix` restore batch p95 为 13.584ms、effective restore p50 为 545.648MiB/s；`3fs_posix` restore batch p95 为 68.175ms、effective restore p50 为 202.162MiB/s。但 `3fs_posix` 仍是本地目录模拟 3FS mount，不代表 DeepSeek 3FS 集群/RDMA/GDS/io_uring 性能。
- 当前方法最可能有效的区域：高复用历史上下文、同 session 多轮 append、共享长文档/系统 prompt、idle session restore、vLLM APC 因 restart/capacity/cross-engine 失效的场景、以及需要保护 SLA 队列不被 cold miss 污染的场景。最不可能有效的区域：low-locality random 1M、小 prefix、APC hot-cache、低带宽 cold tier、每层小文件布局和强 data-plane baseline。
- 下一步应定义 M3.11 Baseline Readiness Matrix：B0 vLLM full prefill/APC off，B1 vLLM APC hot-cache，B2 APC restart/capacity-pressure，B3 external DRAM-ready KV reuse，B4 naive cold restore/synchronous offload，B5 current KV Anti-Caching，B6 LMCache，B7 DualPath/Tutti/CacheFlow-style executor or simulation，B8 KVDrive-style multi-tier orchestration。
- 近期相关工作更新：Tutti（2026-05-05）声称 SSD-backed KV 通过 GPU-centric object store / GPU io_uring / slack-aware scheduling 显著降低 TTFT；DualPath（2026-02-25）提出 storage-to-prefill 与 storage-to-decode 双路径并利用 RDMA 平衡 prefill/decode engine I/O；CacheFlow（2026-04-30）提出 token/layer/GPU 三维 KV restoration；KVDrive（2026-05-18）提出 GPU/DRAM/SSD holistic multi-tier KV 管理。这些工作说明“多级 KV offload/restore”已非常拥挤，我们的论文差异必须集中在 Anti-Caching 式 pre-pass admission、Prefill/Decode readiness、3FS/shared cold tier 和 workload boundary。
- 2026-05-26 规划把关更新：用户确认 KVDrive 已被 SIGMOD 2026 收录，且其关注点与“长上下文 LLM 推理的多层级 KV 缓存管理系统”高度一致。因此本项目不能再以“多级 KV 管理”作为核心创新表述，必须收敛为 **Persistent KV Anti-Caching for Multi-Turn Long-Context Serving**：把多轮 agent session 产生的 committed historical KV 当作持久服务状态，解决其冷却、恢复、校验、准入和 exact 复用问题。
- 与 KVDrive 的边界应固定为：KVDrive 管的是 active long-context decoding 中 critical KV working set 如何在 GPU/DRAM/SSD 间高效缓存/搬运；本项目管的是多轮请求之间已经冷却的历史 KV 如何 exact、可验证、可准入地恢复为 Prefill/Decode 可复用状态。KVDrive 的 packed/extent layout、attention-informed placement、pipeline overlap 可吸收为数据面启发，但不能成为论文唯一创新。
- 后续任务已按 P0/P1/P2 重新排序。P0 是不做就无法证明方向的任务：B0-B5 baseline readiness matrix、512-32K 在线矩阵、persistent session lineage + committed KV ranges、KV PrePass、packed cold object / extent layout。P1 是在 P0 证明收益后必须补强的任务：生产级 restore executor、真实 3FS/生产 SSD 校准、partial promote、attention-informed hotness sketch、强外部 baseline。P2 是 128K/1M、多节点、租户隔离、配额、公平性和生产监控。
- 真实 3FS executor 重要但已降为 P1：原因不是 3FS 不重要，而是继续优化 3FS adapter 不能回答“相对 KVDrive/vLLM APC/LMCache 是否有必要做”的核心问题。没有 session lineage、PrePass、packed object 和 baseline matrix，真实 3FS 性能只能说明数据面速度，不能证明 Persistent KV Anti-Caching 的科研价值。
- 2026-05-27 上午完成 M3.11 Baseline Readiness Matrix 规格文档：`docs/specs/m3_11_baseline_readiness_matrix.md`。该文档固定了 B0-B5 baseline、512/2K/8K/16K/32K 第一轮在线矩阵、`session_append/shared_prefix/low_locality/mixed_short_long` workload、`baseline_matrix.csv` 和 `baseline_summary.csv` schema、ratio 计算规则、推进/暂停门槛、实验污染控制和 runner 设计建议。
- M3.11 第一轮判定标准已固定：B5 不要求赢 B1 APC hot-cache；B5 必须在 8K/16K/32K 高复用 workload 上相对 B0 或 B4 显示收益，并保持 `sync_cold_miss_total=0`、`external_load_observed_rate>0`。如果 B5 没有 connector external load event，实验应判为被 APC hot-cache 污染。
- M3.11 第一轮明确不做：真实 3FS executor、外部 LMCache/Mooncake/DualPath/KVDrive-style baseline、128K/1M、大规模多节点、attention kernel 修改、sparse/approximate path。这样可以防止 P0 证据链尚未成立时任务发散。
- 2026-05-27 上午进一步完成 M3.11 dry-run 执行骨架：`benchmarks/m3/run_baseline_readiness_matrix.py`、`benchmarks/m3/summarize_baseline_readiness_matrix.py`、`tests/m3/test_baseline_readiness_matrix.py`。当前 runner 只固定 schema、run_id、目录结构、ratio 回填和 summary/report，不执行真实 vLLM；这不是性能证据。
- `results/m3_11_baseline_readiness/` 已生成 dry-run 样本：`baseline_matrix.csv` 共 540 行，B0-B5 各 90 行，`session_append/shared_prefix/low_locality` 各 180 行；其中 432 行 `DRY_RUN`，108 行 `ERROR_BOUNDARY`。边界行主要来自 Qwen2.5-14B native 32768 context limit 下 `32768 + suffix + output` 超界，符合“保留失败/边界行”的实验原则。
- M3.11 summary 聚合器当前能回填 B5 对 B0/B3/B1 的 TTFT ratio（当真实 `OK` 行存在时），并输出 `summary/baseline_summary.csv`、`summary/workload_summary.csv` 和短版 `summary/readiness_report.md`。后续真实执行路径必须填充 `ttft_ms`、`external_load_observed`、`connector_load_events`、`restore_*`、`historical_byte_hit_rate`，否则报告仍只能显示 pending。
- M3.11 原生 vLLM 在线路径已接入 B0/B1/B2：`OpenAICompletionClient` 使用 streaming `/v1/completions` 粗测 TTFT/E2E，并统一写入 `baseline_matrix.csv`。B0 为完整 prefill/unique prefix，B1 为同进程 warm prefix 后 APC hot-cache，B2 已支持两种模式：默认 cold/unique prefix 近似，以及 `--strict-b2-restart` 的同 prefix warm、停止 vLLM、重启后同 prefix+suffix 测量。
- 真实 smoke 结果写入 `results/m3_11_native_vllm_online_smoke/`：Qwen2.5-14B 在 `--max-model-len 2048 --enable-prefix-caching --enforce-eager` 下，512 prefix + 128 suffix + 1 output，`session_append/shared_prefix` 两个 workload 的 B0/B1/B2 共 6 行全部 `OK`。本结果只验证 runner 在线路径，不代表 8K/16K/32K 性能结论。
- 2026-05-27 严格 B2 restart smoke 写入 `results/m3_11_b2_restart_online_smoke/`：runner 托管 vLLM 两次生命周期，生成 warm/measure 两份 raw 响应和两份 vLLM 日志；`baseline_matrix.csv` 中 B2 行为 `status=OK`、`admission_reason=apc_restart_cold_cache`、`reuse_tokens=0`、`historical_kv_hit_rate=0.0`。这只证明 APC restart/cold-cache 边界测量路径成立，不代表最终性能收益。
- 2026-05-27 继续完成 M3.11 B3/B5 bridge：`run_baseline_readiness_matrix.py` 现在可通过 `ReuseSmokeMatrixExecutor` 调用已有 `run_reuse_smoke_matrix.py`，也可通过 `ReuseCsvImportExecutor`/`--import-reuse-csv` 导入历史 `reuse_smoke_matrix.csv`。字段映射已覆盖 `second_ttft_ms`、`reuse_tokens`、`load_events`、`external_load_observed`、`cold_probe_decision`、`cold_restore_status`、`cold_restore_actual_bytes`、`cold_restore_executor_elapsed_ms`、`sync_ssd_miss_total` 等统一 schema 字段。
- 导入样本 `results/m3_11_b3_b5_imported_reuse_sample/` 使用历史 restart CSV 生成 B3/B5 两行统一矩阵：B3/B5 均 `status=OK`、`historical_byte_hit_rate=1.0`、`external_load_observed=true`、`connector_load_events=1`、`sync_cold_miss_total=0`，B5 `restore_status=COMPLETED`。该样本用于验证 schema bridge，不是完整同机公平对比，因为 B0/B1/B2 不是同一轮采样。
- 调试发现：以带连字符的词作为“目标 token”会被 Qwen tokenizer 拆成多 token，例如 512+128 个 `b0-full-prefill/session_append-suffix` 词实际变成约 3584 tokens，导致 2048 smoke server 返回 context-length 400。runner prompt 单元已改成 `full/cache/cold/tail` 等 1:1 短 token；后续真正矩阵仍需用 `/tokenize` 或 tokenizer 记录 `actual_prefix_tokens/actual_suffix_tokens`，不能假设空格词数等于 token 数。

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| vLLM CLI 默认加载本机残留插件并打印 `cachemoe`/`moe_infinity`/`pregated` 错误 | 正式 benchmark 使用 `VLLM_PLUGINS=`，并在 `env.json`/report 中记录环境异常 |
| localhost benchmark 被 socks5 proxy 污染 | 使用 `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy NO_PROXY=127.0.0.1,localhost ...` |
| vLLM 0.19.0 不支持 `--disable-log-requests` | 使用当前 CLI 支持的 `--disable-log-stats --disable-uvicorn-access-log` |
| `build_simulator_params.py` 当前对相同 input_len/prefix_len 的多行结果做折叠 | M2 第一版可接受；若要分析 request_rate/max_concurrency/output_len 敏感性，应扩展 YAML schema |
| NVMe 正式校准缺少 fio 随机读/队列深度结果 | 第一版用 Python fallback 作为 lower-bound/shape calibration；后续 3FS 或生产 SSD 校准应安装 fio 并补随机读与 QD sweep |
| Qwen3-32B 32B bf16 权重可能无法在单张 RTX A6000 上以全 GPU 方式加载 | M1.6 第一项必须是加载可行性；若 OOM，则记录为结果，并改测量策略为更小 `max_model_len`、CPU offload、量化或多卡方案 |
| Qwen3-32B 主存卸载路径可用但过慢 | 可用于证明加载边界和建立保守下界；不适合继续跑完整 benchmark 矩阵，除非用户明确接受很长耗时 |
| Qwen3-32B 24GB 主存卸载无法启动 16384 | 调高到 32GB 主存卸载后可启动 32768/40960；该结果表明单卡 48GB 的关键矛盾是权重驻留与 KV 空间竞争 |
| M2 默认 250ms SLA 过严导致复用型负载仍被延迟 | 增加 12000ms 对照，证明放宽 deadline 后复用型可准入而低局部性仍被延迟；下一步做 deadline/H2D/DRAM 敏感性 sweep |
| M2 sweep 初版边界结果偏向 `reuse_prediction_error_rate=0.25` | 增加 `test_boundary_prefers_lower_prediction_error_before_resource_cost`，将边界排序改为先选低预测误差和高命中率 |
| Qwen3-32B 对单卡 A6000 造成过大实验负担 | 建议切换 Qwen2.5-14B-Instruct 做默认在线实验，同时保留 Qwen3-32B 结果作为容量压力参照 |
| Qwen2.5 benchmark 首次运行使用了错误参数 `--suites` | 根因是 runner CLI 参数为 `--suite` 且可重复；改用 `--suite random_smoke --suite apc_sweep` 后成功 |
| M3.5 FastAPI 路由返回类型 `Response | dict` 触发响应模型错误 | 显式设置 `response_model=None`，并将返回类型标注为 `Any` |
| M3.5 HTTP 转发默认读取系统代理导致 socksio 依赖错误 | 在 httpx 客户端设置 `trust_env=False`，保持本机 sidecar 到 vLLM 的转发不经过代理 |
| M3.5 在线 smoke 关闭 vLLM 后出现 leaked semaphore warning | 记录为进程关闭阶段资源追踪警告；服务请求成功，后续检查未发现残留 vLLM/EngineCore 进程且 GPU 显存回落 |
| M3.6 no-op connector 初版调用不存在的实例方法 `_is_usable_plan` | 修正为调用模块函数 `_is_usable_plan(params)`；`pytest tests/m3/test_noop_connector.py -q` 随后通过 |
| M3.6 关闭真实 vLLM smoke 时再次出现 leaked semaphore warning | 与 M3.5 相同，记录为关闭阶段资源追踪警告；本次请求成功，进程和显存随后清理干净 |
| M3.7 真实保存时序不能依赖 `request_finished()` 后再安排 | vLLM 的 `save_kv_layer()` 在前向计算中被调用，必须通过 `build_connector_meta()` 提前把 store 请求带给 worker |
| M3.7 token_count 与 vLLM 实际分词长度可能不一致 | connector 保存时按实际 prompt token 数和实际分配 block 数向下裁剪，manifest 记录真实 token ids；加载允许已保存前缀的开头子集匹配 |
| M3.8 自动提交不能使用上游内部请求编号 | vLLM 响应中的 connector request id 可能和 sidecar request id 不一致；自动提交改为使用本轮 sidecar admission response 的 request id |
| M3.8 全提示词外部命中会触发 vLLM 调度断言 | 矩阵脚本改为前缀复用加新增后缀，并把 `external_load_observed` 单独记录，避免误判 |
| M3.8 完整矩阵如果不重启 vLLM，会被内置前缀缓存遮蔽外部加载 | 保存阶段后只重启 vLLM，保持 sidecar 与 tensor store，复用阶段 16/64/128/256 四行均观测到 `load_request ok` |
| 一致性 smoke 不能直接按全日志统计外部加载 | 脚本记录复用请求前的事件日志行号，只统计本次请求之后新增的 `load_request`，避免历史矩阵事件造成误报 |
| `run_reuse_consistency_smoke.py` 直接执行时初始找不到 `benchmarks` 包 | 按 `replay_cli.py` 模式注入仓库根路径，并新增 CLI `--help` 回归测试 |
| M3.9 离线迁移不是实际 I/O 搬运 | 当前只改变 manifest 层级状态并记录事件；下一步需要将 sidecar prefetch plan 接到真实目录/设备级迁移或后台队列 |
| M3.9 sidecar hook 已有第一版带宽和 deadline 队列，但不是异步真实 I/O | 当前可以记录预计完成时间、deadline miss、多请求排队烟测和 `/metrics` 指标；下一步要引入后台执行、真实设备迁移和完整 residency hit 判断 |
| M3.9 驻留命中初版仍让当前请求先 `DELAY` | 将 residency hit 检查提前到准入记录前：同步控制面后重新准入，使当前请求直接 `ADMIT` |
| M3.9 correctness key 不匹配的 tensor manifest 不能进入预取队列 | 在队列提交前增加源 manifest 校验；不匹配和覆盖不足只写错误记录，不调用 `PrefetchQueue.submit()` |
| M3.9 异步队列去重初版误伤同步队列 | 将 pending 去重逻辑只放在 `AsyncPrefetchQueue.submit()`，恢复同步队列既有行为 |
| M3.9 缺少 tensor manifest 初版被测试描述为 fallback 入队 | 修正语义：没有源 manifest 只能记录预取错误，不能算队列请求 |
| 真实在线小矩阵初始缺少 sidecar 指标列 | 在每行请求前后读取 `/metrics` 并计算 delta，CSV 与报告固定输出驻留/预取/队列指标 |
| 真实在线小矩阵初始 `decision` 列为空 | 增加 `--sidecar-decision-log`，从当前行新增的 sidecar `proxy` 记录解析 `decision/reason/ready_barrier/sync_ssd_miss_total` |
| sidecar `/commit` 初始忽略 `ready=false` | 将 `ready` 字段透传到 `control_plane.commit_request()`，允许构造 `SSD/ready=false` 的在线边界样本 |
| 重复在线样本会改变 `m3-8-256` 的层级状态 | 记录该状态变化；需要冷层样本时先显式 `demote_to_nvme()` 或启用后续真实 cold-tier 恢复机制 |

## 资源
- `/root/KV/1M_Tokens_KV_Cache_分层管理技术方案.md`
- `/root/KV/task_plan.md`
- `/root/KV/progress.md`
- 3FS 官方仓库：https://github.com/deepseek-ai/3fs
- vLLM disaggregated prefill 文档：https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM prefix caching 设计：https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM NIXL connector 文档：https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
- DualPath arXiv：https://arxiv.org/abs/2602.21548
- vLLM bench serve CLI：https://docs.vllm.ai/en/stable/cli/bench/serve.html
- vLLM metrics 文档：https://docs.vllm.ai/en/stable/design/metrics/
- `/root/KV/Benchmark与仿真校准下一步规划.md`
- `/root/KV/results/benchmark_calibration/simulator_params.yaml`
- `/root/KV/results/benchmark_calibration/report.md`
- `/root/models/Qwen3-32B/config.json`
- `/root/models/Qwen3-32B/README.md`
- `/root/KV/results/qwen3_32b_calibration/report.md`
- `/root/KV/results/qwen3_32b_calibration/simulator_params.yaml`
- `/root/KV/results/qwen3_32b_calibration/qwen3_capacity_probe.csv`
- `/root/KV/benchmarks/m2/simulator.py`
- `/root/KV/benchmarks/m2/workloads.py`
- `/root/KV/results/m2_simulation/experiment_summary.md`
- `/root/KV/benchmarks/m2/sweep.py`
- `/root/KV/results/m2_sweep/sweep_report.md`
- `/root/KV/results/m2_sweep/m2_sweep_boundaries.csv`
- `/root/KV/results/m2_sweep/m2_sweep_deadline_frontier.csv`
- `/root/KV/docs/specs/m3_connector_sidecar_minimal_interface.md`
- `/root/models/Qwen2.5-14B-Instruct/config.json`
- `/root/models/Qwen2.5-14B-Instruct/README.md`
- `/root/KV/results/qwen25_14b_calibration/qwen25_capacity_probe.csv`
- `/root/KV/results/qwen25_14b_calibration/simulator_params.yaml`
- `/root/KV/results/qwen25_14b_calibration/report.md`
- `/root/KV/results/m2_sweep_qwen25_14b/sweep_report.md`
- `/root/KV/benchmarks/m3/control_plane.py`
- `/root/KV/benchmarks/m3/replay.py`

## 2026-05-25 相关工作调研与论文化规划补充
- 本轮按 academic-research-suite / level2-research / vibe-research 的组合框架，把项目重新定位为 `SLA-governed exact long-context KV service`，而不是单纯 KV offload。
- 相关工作需要分层讨论：vLLM/PagedAttention 和 APC 是 GPU KV 管理与精确 prefix reuse 基线；SGLang/RadixAttention 是 agent/structured workload 的 prefix-tree reuse 基线；DistServe/vLLM NIXL 是 prefill/decode disaggregation 和 KV transfer 基线；LMCache/Mooncake/Tutti/CacheFlow/DualPath 是最接近的 persistent / multi-tier / SSD-backed KV 竞争方向；H2O/Scissorhands/CacheGen 等压缩/驱逐方法应作为近似或可插拔扩展，不作为第一版 exact path 主线。
- 论文贡献必须和 Mooncake、LMCache、Tutti、DualPath 区分：突出精确 1M tokens、SLA admission、ready barrier、decode path no synchronous SSD/3FS miss、3FS-aware cold tier、负对照 workload 边界和 vLLM connector 原型。
- 当前 M2 有效命中率约 65%-69%，未达到目标 90%。下一步应重新定义并报告至少四类 hit rate：`all_tokens`、`historical_kv`、`byte_hit_rate`、`admitted_sla_class`；90% 指标应约束高复用适用场景，而 low-locality 1M random 必须作为 negative control delay/reject。
- 推荐系统设计命名为 `SLA-KV`：Gateway / Workload Classifier / CorrectnessKey + PrefixIndex / Reuse Planner / Residency Planner / Admission Guard / Prefetch-Restore Scheduler / vLLM KV Connector / Decode Ready Barrier / Eviction-Writeback-Metrics。
- Level2 视角下，KV 多级管理的内环是 `observe -> propose reuse/placement/prefetch plan -> evaluate TTFT/hit/bandwidth -> admit/delay/fallback -> observe actual latency`。当前瓶颈是离线网格 sweep 和确定性队列，缺少在线自适应。
- 推荐下一步策略是 `Budgeted Slack Scheduler`：primal-dual budget controller 限制 3FS/H2D/network utilization，EDF/slack queue 控制 deadline，contextual bandit 学习 prefetch lead、tier placement、chunk size 和 DualPath/offload path 选择；但学习策略不得绕过 ready barrier。
- 实验路线应先做 M3.10 真实 cold-tier 目录/设备级迁移，再扩 512-32K 在线矩阵，随后做 128K/1M scaling；论文级对比至少包括 vLLM APC、LMCache、disaggregated prefill、offloading、DualPath-style 策略。
- CCF-A 潜力判断：若能证明真实 1M exact context、TTFT ratio <=1.2、适用 workload hit rate >=90%、真实 3FS/NVMe cold tier、强 baseline 对比和可复现实验，则可冲 OSDI/SOSP/USENIX ATC/ASPLOS 等系统方向；若停在 M3.9 元数据迁移和短前缀 smoke，则论文强度不足。
- 详细文档已写入 `/root/KV/docs/research/kv_multitier_related_work_and_design_plan.md`。
- `/root/KV/benchmarks/m3/replay_cli.py`
- `/root/KV/results/m3_sidecar_replay/m3_replay_decisions.csv`
- `/root/KV/results/m3_sidecar_replay/report.md`
- `/root/KV/benchmarks/m3/http_sidecar.py`
- `/root/KV/benchmarks/m3/http_sidecar_cli.py`
- `/root/KV/docs/specs/m3_5_http_sidecar_proxy.md`
- `/root/KV/docs/specs/m3_6_vllm_connector_integration_notes.md`
- `/root/KV/results/m3_5_http_sidecar/online_smoke_decisions.jsonl`
- `/root/KV/benchmarks/m3/noop_connector.py`
- `/root/KV/benchmarks/m3/tensor_store.py`
- `/root/KV/tests/m3/test_noop_connector.py`
- `/root/KV/tests/m3/test_tensor_store.py`
- `/root/KV/docs/specs/m3_7_tensor_copy_prototype.md`
- `/root/KV/results/m3_7_tensor_copy/online_smoke_decisions.jsonl`
- `/root/KV/results/m3_7_tensor_copy/tensor_store/session-a/manifest.json`
- `/root/KV/docs/specs/m3_8_reuse_stabilization.md`
- `/root/KV/benchmarks/m3/run_reuse_smoke_matrix.py`
- `/root/KV/benchmarks/m3/summarize_restart_matrix.py`
- `/root/KV/benchmarks/m3/run_reuse_consistency_smoke.py`
- `/root/KV/results/m3_8_full_restart_matrix/restart_matrix_report.md`
- `/root/KV/results/m3_8_full_restart_matrix/consistency_smoke/consistency_smoke_report.md`
- `/root/KV/docs/specs/m3_9_tier_migration_prototype.md`
- `/root/KV/benchmarks/m3/run_tier_migration_smoke.py`
- `/root/KV/results/m3_9_tier_migration_smoke/tier_migration_report.md`
- `/root/KV/benchmarks/m3/run_sidecar_prefetch_smoke.py`
- `/root/KV/results/m3_9_sidecar_prefetch_smoke/sidecar_prefetch_report.md`
- `/root/KV/benchmarks/m3/prefetch_queue.py`
- `/root/KV/results/m3_9_sidecar_prefetch_queue_smoke/ok/sidecar_prefetch_report.md`
- `/root/KV/results/m3_9_sidecar_prefetch_queue_smoke/deadline_miss/sidecar_prefetch_report.md`
- `/root/KV/benchmarks/m3/run_prefetch_queue_smoke.py`
- `/root/KV/results/m3_9_prefetch_queue_smoke/prefetch_queue_report.md`
- `/root/KV/benchmarks/m3/run_residency_aware_prefetch_smoke.py`
- `/root/KV/results/m3_9_residency_aware_prefetch_smoke/residency_aware_prefetch_report.md`
- `/root/KV/benchmarks/m3/run_async_prefetch_smoke.py`
- `/root/KV/results/m3_9_async_prefetch_smoke/async_prefetch_report.md`
- `/root/KV/results/m3_9_reuse_metrics_online/fresh_sidecar/reuse_phase/reuse_smoke_matrix.csv`
- `/root/KV/results/m3_9_reuse_metrics_online/fresh_sidecar/reuse_phase/report.md`

## 2026-05-27 M3.11 2K 真机同机矩阵发现
- 已完成 Qwen2.5-14B、`session_append`、2048 prefix + 128 suffix + 1 output、concurrency=1 的 B0/B1/B2/B3/B5 同机小矩阵，统一结果在 `results/m3_11_2k_true_matrix/final/baseline_matrix.csv`，summary 在 `results/m3_11_2k_true_matrix/final/summary/readiness_report.md`。
- B0 full prefill TTFT 为 `717.061ms`；B1 APC hot-cache TTFT 为 `91.016ms`；严格 B2 restart/cold-cache TTFT 为 `686.379ms`。B2 接近 B0，说明 restart 后内置 APC 不保留历史 KV，baseline 边界可用。
- B3 external DRAM-ready reuse 语义成立：`reuse_tokens=2048`、historical byte hit rate `1.0`、`external_load_observed=true`、`connector_load_events=1`、`sync_cold_miss_total=0`。但 B3 TTFT 为 `6055.039ms`，远慢于 B0，说明当前 Python/safetensors per-layer connector 路径不是性能可用路径。
- B5 Anti-Caching true cold-tier 语义成立：cold probe 先 `DELAY/required_kv_not_ready_before_decode`，`/prefetch/advance` 后 `restore_status=COMPLETED`、`restore_actual_bytes=402657408`、`restore_checksum_status=ok`、`ready_barrier_all_ready=true`，最终 reuse `ADMIT/required_kv_ready_before_decode`，`external_load_observed=true`，`sync_cold_miss_total=0`。
- B5 TTFT 为 `6590.921ms`，是 B0 的 `9.191576x`、B3 的 `1.088502x`、B1 的 `72.414971x`。因此 2K 下只能声称“exact persistent KV Anti-Caching 控制语义可跑通”，不能声称“性能有效”或“满足 SLA 增幅 <=20%”。
- B5 restore 本身耗时 `874.308ms`，恢复字节/有用字节比 `1.00001`；主要 TTFT 过大还包含 connector load/store 与 vLLM external KV transfer path 的系统开销。下一步性能优先级应高于继续扩 3FS adapter：packed cold object / extent layout、减少每层 safetensors 文件、避免每次 reuse 再写全量 store、优化 load path 与 sidecar/vLLM 阶段计时拆解。
- 本轮修正了一个实验语义问题：首次 B5 store 阶段没有带 `--cold-tier-restore`，没有真实 demote 到 cold tier；因此另起 `results/m3_11_2k_true_matrix/b5_anti_caching_true/` 干净重跑，最终表使用该 true cold-tier 样本。
- 当前矩阵还缺 B4 naive cold restore/offload baseline，缺 512/8K/16K/32K 尺寸扩展，缺 shared_prefix/low_locality 工作负载，也缺 LMCache/DualPath/KVDrive-style 强 baseline。现阶段下一步必须先解释和压低 2K 性能开销，再扩长上下文矩阵。
- 验证结果：`pytest tests/m3/test_baseline_readiness_matrix.py -q` 为 `14 passed in 3.07s`；`pytest tests/m3 -q` 为 `122 passed, 2 warnings in 24.82s`。服务清理后 8000/8010 端口关闭，GPU 显存回落。

## 2026-05-27 M3.11 TTFT breakdown 归因发现
- 已扩展 M3.11 baseline schema 与 summary，新增 `online_service_ttft_ms`、`restore_inclusive_ttft_ms`、`restore_wait_ms`、`connector_store_elapsed_ms`、`connector_load_elapsed_ms`、`connector_total_elapsed_ms`、`unattributed_ttft_ms`。新产物在 `results/m3_11_2k_true_matrix/final_breakdown/`。
- 该 breakdown 的语义是：`restore_wait_ms` 和 `connector_load_elapsed_ms` 可近似看作关键路径可解释项；`connector_store_elapsed_ms` / `connector_total_elapsed_ms` 是逐层事件求和诊断值，可能与 vLLM forward/writeback 重叠，不能直接从 TTFT 中相减。
- 2K 结果显示 B3：在线 TTFT `6055.039ms`，load `92.02ms`，store_layer 求和 `7553.592ms`，未解释在线 TTFT 约 `5963.019ms`。这说明 external DRAM-ready 慢主要不是 load 本身，而是 connector 生命周期/复用后 store/writeback/调度开销。
- 2K 结果显示 B5：在线 TTFT `6590.921ms`，restore wait `874.308ms`，load `183.665ms`，restore-inclusive TTFT `7465.229ms`，store_layer 求和 `6484.31ms`，未解释在线 TTFT 约 `5532.948ms`。这进一步说明 cold restore 不是唯一瓶颈；复用请求仍写回大量 KV 是 P0 嫌疑点。
- 因此下一步优先任务应调整为：先修复复用阶段不必要的全量 store/writeback，明确 store/load 是否与 TTFT 重叠，再做 packed cold object / extent layout。直接扩 16K/32K 会把当前 2K 原型开销放大，信息增益低。

## 2026-05-27 M3.11 load-only reuse 修复发现
- 已修复 sidecar/connector 协议里“复用请求误触发写回”的问题：`build_kv_transfer_params()` 现在显式输出 `store_policy`；只有请求里明确携带 `store_prefix_id` 时才设置 `store_policy=store_prefix` 与 `store_token_end`，普通复用请求默认 `store_policy=load_only`。
- `M3NoOpConnector` 现在只有在 store policy 允许且存在显式 `store_prefix_id` 时才生成 `store` metadata 或 request-finished manifest；required range 的 `prefix_id` 只用于 load，不再被当作本轮写回目标。
- 新增回归覆盖：复用请求即使带有历史 `store_token_end`，只要没有显式 store prefix，就只生成 `load`；load-only 请求结束时不再返回 auto-commit manifest；sidecar 生成复用参数时不再附带 `store_prefix_id/store_token_end`。
- 真实在线 smoke 路径为 `results/m3_11_load_only_reuse_smoke/`。store 阶段 64 tokens 前缀写出 `48` 个 `store_layer` 事件；重启 vLLM、保留 sidecar 与 tensor store 后，reuse 阶段为 `ADMIT/required_kv_ready_before_decode`，`external_load_observed=yes`，`load_events=1`，`store_events=0`，connector load 耗时 `35.314ms`。
- 该修复消除了“复用阶段再次写 48 层 KV”的确定性开销，但还不能自动说明 B3/B5 已满足性能目标；下一步需要重跑 2K B3/B5 和完整 512-32K 矩阵，重新衡量 TTFT、未解释开销和 packed object 需求。

## 2026-05-27 M3.11 load-only 后 2K B3/B5 重跑发现
- 已用 Qwen2.5-14B、`--max-model-len 8192`、2048 prefix + 128 suffix + 1 output 重新跑 B3 external DRAM-ready reuse 与 B5 true cold-tier Anti-Caching，并导入统一 M3.11 schema。结果目录为 `results/m3_11_2k_load_only_rerun/`。
- B3 重跑结果：store 阶段 `store_events=48`，reuse 阶段 `store_events=0`、`load_events=1`、`external_load_observed=yes`、`connector_load_elapsed_ms=94.531`，online TTFT 为 `283.027ms`。旧 B3 TTFT 为 `6055.039ms`，说明旧结果主要被 reuse 阶段写回污染。
- B5 重跑结果：store/demote 阶段写出真实 cold object `402657408` bytes，hot 目录 demote 后清空；reuse 阶段 cold probe 为 `DELAY/required_kv_not_ready_before_decode`，advance restore 为 `COMPLETED`，checksum `ok`，`sync_ssd_miss_total=0`，最终 `ADMIT/required_kv_ready_before_decode`，`store_events=0`、`load_events=1`。
- B5 online 服务 TTFT 为 `327.512ms`，connector load 为 `139.882ms`，cold restore executor 为 `1035.707ms`，restore-inclusive TTFT 为 `1363.219ms`。旧 B5 online TTFT 为 `6590.921ms`，load-only 修复同样消除了主导性的写回伪开销。
- 合并后的 2K 统一矩阵路径为 `results/m3_11_2k_load_only_rerun/final/baseline_matrix.csv`。关键对比：B0 `717.061ms`，B1 APC hot `91.016ms`，B2 restart `686.379ms`，B3 `283.027ms`，B5 online `327.512ms`。B5 online/B0 为 `0.456742`，B5 online/B3 为 `1.157176`，B5 online/B1 为 `3.5984`。
- 需要非常谨慎解释：若 cold restore 能由 KV PrePass、idle session 或后台 async restore 提前隐藏，则 B5 online TTFT 已经显示出相对 full prefill 的潜在收益；若请求到达后才等待 cold restore，则 restore-inclusive/B0 约为 `1.90x`，仍不满足“较无卸载增幅 <=20%”这类 SLA 目标。
- 本轮还修复了一个报告口径 bug：`unattributed_ttft_ms` 表示 online 服务阶段剩余开销，应按 `online_service_ttft_ms - connector_load_elapsed_ms` 计算，不能再扣除请求前的 `restore_wait_ms`。修复后 B5 `unattributed_ttft_ms=187.63ms`，不再出现负值。
- 下一步 P0 不应只继续扩 3FS adapter，而应并行推进：512/8K/16K/32K 小矩阵、KV PrePass/async restore hiding、packed cold object/extent layout，以及 B4 naive cold restore baseline。否则目前的 B5 结果只能说明“在线 reuse path 有希望”，还不能证明多轮 persistent cold KV 在端到端 SLA 下成立。

## 2026-05-27 M3.12-A KV PrePass 初版发现
- 已新增最小 PrePass 控制面入口：`/prepass` 会根据本轮 future request 的 prefix candidates 先调用 admission 生成 required historical KV ranges，再检查 tensor store residency，最后通过现有 `PrefetchQueue` / `AsyncPrefetchQueue` 执行或排队 cold restore。
- 新增 `benchmarks/m3/prepass_planner.py`，当前只做最小状态归类：`READY`、`QUEUED`、`MISSED`、`ERROR`、`PENDING`。这不是 bandit 或复杂策略，只是为证明 restore hiding 建立稳定接口。
- PrePass 与在线 admission 指标已分离：`/prepass` 不再增加 `prefill_tokens_saved_total` 或 online admission decision counters，只单独暴露 `prepass_requests_total` 和 `prepass_ready_total`，避免把规划阶段误算成一次线上请求执行。
- 新增 deterministic smoke：`results/m3_12_prepass_restore_smoke/`。在 2048 prefix + 128 suffix 下，PrePass 输出 `prepass_status=READY`、`ready_before_request=true`、`restore_status=COMPLETED`，随后 online `/admit` 直接 `ADMIT/required_kv_ready_before_decode`，`online_reuse_tokens=2048`、`sync_cold_miss_total=0`。
- 已为 `run_reuse_smoke_matrix.py` 增加 `--prepass-before-reuse` 和 CSV 字段：`prepass_status_code`、`prepass_status`、`ready_before_request`、`prepass_elapsed_ms`、`prepass_restore_status`、`prepass_restore_estimated_ready_ms`。这为下一轮真实 vLLM B5 PrePass-before-reuse 在线样本做了脚手架。
- 当前 M3.12-A 仍是控制面证明，不是最终性能结论。deterministic smoke 使用 manifest-only restore，因此 `restore_executor_elapsed_ms=0.0`；它证明 PrePass 语义与指标口径，不证明真实 402MB cold object restore 可被实际业务窗口隐藏。
- 下一步必须用真实 vLLM 路径重跑 2K B5：store -> demote true cold object -> `/prepass` restore -> restart/online reuse。若 reuse 阶段仍 `store_events=0`、`load_events=1`、online TTFT 接近 `327ms`，同时 restore wait 不计入 online request，则才能说“PrePass 有效隐藏了 B5 的 cold restore”。

## 2026-05-27 M3.12-B 2K 真实 PrePass 样本发现
- 已完成真实 vLLM B5 PrePass-before-reuse 样本，路径为 `store -> demote true cold object -> restart vLLM -> /prepass -> advance restore -> reuse`，结果目录为 `results/m3_12_2k_prepass_true/`，独立报告为 `docs/research/m3_12_prepass_true_2k_results.md`。
- store 阶段写出 `48` 个 store layer，connector store 求和 `323.865ms`；真实 cold object 大小为 `402657408` bytes，demote 后 hot 目录清空，sidecar commit 为 cold/not-ready。
- PrePass 阶段在 async 模式下返回 `QUEUED`，耗时 `11.070ms`，枚举出 1 个 required historical range，并入队 `402657408` bytes restore；随后 `/prefetch/advance` 完成真实 restore，`cold_restore_status=COMPLETED`，checksum `ok`，executor elapsed `839.601ms`。
- reuse 阶段最终为 `ADMIT/required_kv_ready_before_decode`，`ready_barrier_all_ready=true`，`sync_ssd_miss_total=0`，`store_events=0`，`load_events=1`，`external_load_observed=yes`，connector load `97.694ms`。
- B5 PrePass online TTFT 为 `271.782ms`，低于 B0 full prefill `717.061ms`，约为 B0 的 `0.379x`；也接近 B3 DRAM-ready reuse `283.027ms`，约为 B3 的 `0.960x`。这说明只要 restore 被提前隐藏，当前 2K online 路径已经能体现 persistent KV reuse 的价值。
- 必须谨慎区分指标口径：如果从 PrePass/advance 开始计算，restore-inclusive 为 `1111.383ms`，约为 B0 的 `1.550x`，仍不满足严格 `B0 + 20%` SLA。PrePass 的意义不是让 402MB restore 消失，而是把 restore 从用户在线请求路径前移。
- 与 reactive B5 相比，PrePass 样本的 online TTFT 从 `327.512ms` 降到 `271.782ms`，restore-inclusive 从 `1363.219ms` 降到 `1111.383ms`。但单样本差异可能受运行抖动影响，不能过度解释为数据面优化已经完成。
- 下一步 P0 应转向两件事：一是扩展 512/2K/8K/16K/32K PrePass 小矩阵，测需要多少 lead time 才能把 restore 完全隐藏；二是做 packed cold object / extent layout，减少每层 safetensors 小文件和 Python 拷贝带来的 restore tail。

## 2026-05-27 M3.13 PrePass lead-time planner 发现
- 本轮按 level2-research 重新定义下一步内环：`选择 prefix/layout/lead-time -> 运行 vLLM -> 观察 TTFT/restore hiding -> 决定扩展或回退`。当前瓶颈不是缺少又一个接口，而是实验选择太依赖直觉，容易把“提前量不足、restore executor 慢、per-layer layout tail”三件事混成一个长上下文矩阵结果。
- 已新增 `benchmarks/m3/prepass_lead_time_planner.py` 与 CLI `benchmarks/m3/plan_prepass_lead_time_matrix.py`，从 M3.12 真实 PrePass CSV 和 M3.11 baseline CSV 学习 restore profile，并输出 512/2K/8K/16K/32K × available lead 的风险表。
- 真实 M3.12 profile：source prefix `2048`，restore bytes `402657408`，restore executor `839.601ms`，KV bytes/token `196608`，有效 restore throughput `457.365MiB/s`，PrePass 控制开销 `11.070ms`，online TTFT `271.782ms`。
- 在 `tail_multiplier=1.2`、`safety_margin=250ms` 下，planner 估算 required lead：512 约 `512.948ms`，2K 约 `1268.581ms`，8K 约 `4291.113ms`，16K 约 `8321.155ms`，32K 约 `16381.240ms`。
- 结果目录为 `results/m3_13_prepass_lead_time_plan/`，其中 20 行计划里只有 8 行能隐藏 restore，12 行是 lead-time risk，12 行触发 packed cold object attention。8K 是最关键的下一步 gate：5s lead 理论上可隐藏 restore，但已触发布局风险；16K/32K 不应在 packed layout 和 8K 结果前盲跑。
- 新研究文档为 `docs/research/m3_13_next_research_plan.md`。它明确下一步顺序应是 512/2K/8K true PrePass 小矩阵 -> packed cold object / extent layout -> 16K/32K gate matrix。
- 本机缺少 `column` 命令，查看 CSV 时改用 `sed`；后续脚本和验证不能依赖 `column`。

## 2026-05-28 M3.14 Anti-Cache 启发整理与模块设计发现
- 本轮结合 context-engineering 与 planning-with-files-zh，把数据库 Anti-Cache 的核心机制整理进项目规划。关键启发是：Anti-Cache 不需要在请求到达前 100% 预知是否访问冷数据，而是让所有索引留在内存中，请求执行时通过 evicted 标记发现冷数据，再进入 pre-pass 收集 block id / offset，异步读取并重新执行。
- 映射到 KV 系统后，我们不能让 GPU 做真正试跑；缺失历史 KV 时注意力语义不完整。因此应做“控制面试跑”：请求进入 vLLM 前由 sidecar/control plane 查内存 KV 状态索引，判断 required KV 是否处于 HBM/CPU ready、SSD cold、FETCHING、MISSING 或 correctness mismatch。
- 设计上明确 GPU/CPU/SSD 三层职责：HBM 是在线执行层，CPU 是可准入准备层，SSD 是冷区和持久化层。SSD 上的 KV 不是 ready，必须恢复到 CPU/HBM 并校验后才能被 connector 加载。
- 下一步最应该落地的两个模块是：`KV Evicted Index` 与 `Packed Cold Object`。前者对应 Anti-Cache 的 Evicted Table，常驻 CPU 内存，只保存 prefix/range/tier/ready/object_id/offset/checksum 等元数据；后者对应 Block Table，把冷 KV 组织成一个或少量连续对象，避免长上下文下每层 safetensors 小文件造成 restore tail。
- 已新增设计规格 `docs/specs/m3_14_kv_evicted_index_and_packed_object.md`。规格定义了 M3.14-A/B/C/D：独立 KV Evicted Index、PrePass 分类集合、packed object v1、2K/8K packed vs per-layer restore 对比。
- 设计边界：第一版不引入压缩、不改变 exact attention 语义、不把 SSD->CPU 恢复队列和 CPU->HBM 加载队列混在一起、不因 prefix_id 相同绕过 correctness key。
- M3.14-A 已完成独立实现：`benchmarks/m3/kv_evicted_index.py` 提供 `KVEvictedIndex`、`KVIndexEntry`、`KVExtentRef`、`KVLookupResult` 和 `KVClassification`，可以从现有 `KVBlockManifest` 导入状态，并按 `READY/COLD/FETCHING/MISSING/MISMATCH` 分类 required ranges。
- `KVEvictedIndex` 的第一版保持纯内存、纯元数据，不保存 KV 张量本体；`SSD/NVME/LOCAL_NVME/THREE_FS/3FS` 等 tier 被归为 cold，`HBM/DRAM/CPU` 且 `ready=true` 才归为 ready，`FETCHING` 或存在 `fetching_request_id` 时会避免重复入队。
- 新增测试 `tests/m3/test_kv_evicted_index.py` 覆盖：ready 命中、cold extent 定位、fetching 去重、correctness mismatch、missing range、mark_ready 状态切换和批量分类集合。
- M3.14-B 已把 `/prepass` 响应升级为分类集合：新增 `classification` 与 `classification_counts` 字段，包含 `ready/cold/fetching/missing/mismatch` 五类。旧字段如 `status`、`ready_before_request`、`restore_results` 保持不变，避免破坏已有实验脚本。
- sidecar 现在维护一个 runtime 级 `KVEvictedIndex`，从控制面 commit、tensor store manifest、同步/异步 restore 完成事件同步元数据。该索引仍是增强层，不强行替代现有 `KVManifest`，这是为了降低 M3.14-B 对在线路径的侵入。
- 异步 PrePass 的第二次请求现在能看到 `FETCHING` 分类，而不是再次把同一个 cold prefix 当作普通 cold 任务入队。这对应数据库 Anti-Cache 中 evicted tuple 进入 pre-pass 后避免重复 fetch 同一 block 的思想。
- 2026-05-28 追问修正：上一版 `READY/COLD` 对 Anti-Caching 的 SSD 冷区判断够用，但没有充分表达 GPU/CPU/SSD 三级调度。必须把 `READY` 拆成 `GPU_READY` 与 `CPU_READY`：前者表示 KV 已在 HBM、可直接进入执行；后者表示 KV 已经从 SSD 恢复到 CPU/DRAM 并校验，但仍需要 CPU->GPU load。
- 同理，`COLD` 应精确为 `SSD_COLD`：它表示 KV 在 SSD/3FS 冷区或持久化层，不能直接进入 vLLM 执行。`FETCHING` 表示 SSD/3FS->CPU restore 进行中；新增 `LOADING` 表示 CPU/DRAM->GPU/HBM load 进行中。这样 SSD->CPU 与 CPU->GPU 两条路径不会被混成一个“ready”。
- 修正后的 PrePass/Index 输出同时保留兼容汇总字段 `ready` 与 `cold`，但新研究和调度逻辑应使用 `gpu_ready/cpu_ready/ssd_cold/fetching/loading/missing/mismatch`。下一步整体方案必须增加 CPU_READY->GPU_READY 的 load readiness 观测口径，否则会低估 online TTFT 与 decode ready 风险。
- M3.14-B3 已补上第一版 CPU_READY->GPU_READY load readiness 观测：`/prepass` 输出 `load_readiness`，包括 `cpu_ready_not_gpu_ready_count`、`cpu_ready_tokens`、`expected_h2d_load_bytes` 和 `estimated_h2d_load_ms`；`/metrics` 新增 `cpu_ready_not_gpu_ready_total`。这一步让 CPU_READY 的 H2D 代价显式进入控制面，但还没有完成真正的 GPU load 回写闭环。
- 重新反思整体方案：当前系统已经从二分 ready/cold 变成三级状态可见，但还没有完成三级调度闭环。闭环还缺两件 P0：一是 packed cold object 降低 SSD/3FS->CPU restore 尾延迟；二是把 connector 的 CPU->GPU load 完成事件回写为 `GPU_READY`，并在 admission 中区分“可直接执行”和“需要执行前加载”。
- M3.14-C 已实现 `packed_v1` cold-tier backend。第一版采用 layer-major packed object：每个 prefix 冷区目录中只有 `packed_object.bin` 和 `packed_manifest.json`，每层 safetensors 的字节范围以 `offset/size_bytes/checksum` 写入 manifest 和 `KVBlockManifest.offset_table`。这降低了冷区小文件数量，并为后续 extent-major/partial restore 保留 offset 接口。
- `packed_v1` 已接入 `build_cold_tier_adapter()`、`KVTensorStore.demote_to_cold_object()` / `restore_from_cold_object()`、prefetch queue 和已有 cold-tier adapter benchmark 入口。CLI 的 `--cold-backend` 现在支持 `local_posix`、`3fs_posix`、`packed_v1`。当前 packed_v1 是 local POSIX packed layout，还不是 3FS 原生 packed object。
- packed_v1 功能测试覆盖：demote 后热层文件被移除；冷区不再生成每层 safetensors 文件；restore 后原有 `load_layer()` 能读回两个 layer 的原始 KV；packed object 尺寸被篡改时 checksum mismatch 会失败且 manifest 保持 cold/not-ready。
- M3.14-D 已完成 2K/8K `local_posix` vs `packed_v1` 对比，结果目录为 `results/m3_14_packed_vs_per_layer/`。本轮使用 `qwen25_14b_tiny` profile、4 layers、bf16、repeats=3、queue_depth=1。结论很明确：packed_v1 降低了冷区数据文件数，但当前实现没有降低 restore latency。
- Aggregate 结果：`local_posix` restore p50/p95/p99 为 `153.920/257.851/260.170ms`，effective restore p50 `517.004MiB/s`；`packed_v1` restore p50/p95/p99 为 `315.211/531.195/531.347ms`，effective restore p50 `274.663MiB/s`。packed_v1 的 cold data files 从 24 降到 6，但 restore p95 增加 `273.344ms`。
- 按 token 拆分：2K 下 local/packed restore p95 为 `60.637ms` vs `102.856ms`；8K 下 local/packed restore p95 为 `259.590ms` vs `531.309ms`。这说明当前 layer-major packed Python 实现的 read/write/copy 开销抵消了文件数收益，不能作为进入 16K/32K gate 的性能依据。
- 下一步不应盲目进入 16K/32K，而应先做 M3.14-E：优化 packed restore 数据面，例如避免 `read_bytes()`/整段 bytes 拼接与二次切片、用更接近顺序读的 copy path、减少重复 checksum、或实现 extent-major/批量 pread。packed 方向仍有系统价值，但当前版本只是功能原型，不是性能可用实现。
- M3.14-E 已修正 packed_v1 的主要 Python 数据面问题：demote 从 `read_bytes()/write` 改为 chunk streaming append；restore 从整段 bytes 切片改为按 extent streaming copy；restore 前的 `summarize()` 在 packed object 大小匹配 manifest 时直接复用 manifest checksum，不再重读整个 packed object；demote 收尾的 manifest checksum 也改为复用 append 阶段已计算的 extent checksum，不再二次读取 `packed_object.bin`。
- 修正后第一轮 2K/8K r3 小矩阵在 `results/m3_14_packed_vs_per_layer_fastmanifest/` 显示 packed_v1 从“明显慢”变为“略优”：local restore p50/p95/p99 为 `158.667/254.746/255.478ms`，packed 为 `143.334/233.673/235.484ms`，packed restore p95 delta 为 `-21.073ms`，冷区数据文件数 delta 为 `-18`。
- 进一步去掉 demote summary 二次读后，r5 小矩阵结果在 `results/m3_14_packed_vs_per_layer_streaming_summary_r5/`：local_posix restore p50/p95/p99 为 `216.930/283.665/287.823ms`，packed_v1 为 `145.103/225.144/225.181ms`，packed restore p95 delta 为 `-58.521ms`，cold data files 从 `40` 降到 `10`，effective restore p50 从 `442.556MiB/s` 提升到 `526.998MiB/s`。
- 按 token 拆分的 r5 结果：2K local/packed restore p95 为 `159.497ms` vs `66.766ms`；8K local/packed restore p95 为 `286.552ms` vs `225.169ms`。这说明 packed_v1 数据面优化后已具备继续做 8K/16K gate 的价值，但结论仍受本地 POSIX、synthetic qwen25 tiny profile、Python safetensors restore 和小样本抖动限制，不能外推为真实 3FS/生产 executor 性能。

## 2026-05-28 M3.15 8K packed PrePass gate 发现
- 本轮完成真实 Qwen2.5-14B、8192 prefix + 128 suffix + 1 output、`packed_v1` cold backend 的 PrePass gate。结果目录为 `results/m3_15_8k_packed_prepass_gate/`，研究报告为 `docs/research/m3_15_8k_packed_prepass_gate.md`。
- 先发现并修复了一个重要实验有效性风险：首次 8K store 虽请求 `8192` tokens，但 connector 只保存 `2048` tokens、`128` blocks，manifest token range 是 `0..2048`。根因很可能是 vLLM 默认 `max_num_batched_tokens=2048` 导致 prefill chunk 暴露不完整。后续所有长上下文 store 必须校验 saved tokens。
- 已在 `run_reuse_smoke_matrix.py` 中增加 guard：demote 后记录 `cold_saved_tokens`、`cold_expected_prefix_tokens`、`cold_saved_token_mismatch`；如果保存 token 数和请求 prefix 不一致，矩阵行直接标为错误。新增测试覆盖该场景。
- 使用 `--max-model-len 16384 --max-num-batched-tokens 16384` 重跑后，8K store 有效：manifest token range 为 `0..8192`，connector 每层 `tokens=8192`、`block_count=512`，`cold_saved_token_mismatch=no`。
- store 阶段写出 `48` 个 layer，connector store 求和 `3203.352ms`，请求 TTFT `5513.874ms`。demote 后冷区只有 `packed_object.bin` 和 `packed_manifest.json`，packed object 大小 `1610616960` bytes，hot per-layer files 被移除。
- reuse 阶段采用 `store -> packed demote -> restart vLLM -> /prepass -> advance restore -> reuse`。PrePass 将 required historical range 分类为 `SSD_COLD`，状态 `QUEUED`，耗时 `25.280ms`，入队 `1610616960` bytes restore。
- `/prefetch/advance` 完成真实 packed restore：`cold_restore_status=COMPLETED`，checksum `ok`，executor elapsed `3046.794ms`，advance elapsed `3061.115ms`。随后在线请求为 `ADMIT/required_kv_ready_before_decode`，`ready_barrier_all_ready=true`，`sync_ssd_miss_total=0`。
- 在线 reuse 侧观测到 `external_load_observed=yes`、`load_events=1`、`store_events=0`、connector load `469.372ms`，online TTFT `710.835ms`。这说明 8K 的 external KV load path 真实触发，且没有复用阶段写回污染。
- 需要谨慎解释：该样本证明 8K packed Anti-Caching 语义链路成立，但性能尚未达终局。若 restore 已被提前窗口隐藏，online TTFT 为 `710.835ms`；若从 PrePass 开始计入，restore-inclusive 为 `25.280 + 3061.115 + 710.835 = 3797.230ms`，主要由 packed restore executor 支配。
- 与 M3.13 planner 相比，实际 8K restore executor `3046.794ms` 低于先前保守 required lead 估计 `4291.113ms`，说明 5s lead 在本机样本上有机会隐藏 restore。但进入 16K/32K 前必须补同配置 8K B0/B2 baseline，并继续优化 restore/load 数据面。
- 下一步优先级应调整为：同配置 8K B0/B2 baseline、512/2K/8K compact packed PrePass matrix、packed restore/load 归因优化。直接冲 32K 信息增益不高，容易把 baseline 差异、restore executor、connector load 和 chunked prefill 问题混在一起。
- 已补齐同配置 8K B0/B2 baseline，结果目录为 `results/m3_15_8k_baseline_same_config/`。配置为 `--max-model-len 16384 --max-num-batched-tokens 16384 --enforce-eager --enable-prefix-caching`，shape 同为 8192 prefix + 128 suffix + 1 output。
- 8K B0 full prefill TTFT 为 `2409.141ms`；严格 B2 restart TTFT 为 `2418.796ms`，B2/B0 为 `1.004008`。这说明同配置下 vLLM 内置 APC 在进程重启后没有保留 8K 历史 KV，B2 可作为 cold-cache 边界。
- 与同配置 B0 相比，B5 packed PrePass online TTFT `710.835ms` 是 B0 的 `0.295057x`，说明如果 PrePass/restore 能被工作流提前隐藏，8K online 请求已有明显收益。
- 但 B5 packed PrePass restore-inclusive 为 `3797.230ms`，是 B0 的 `1.576176x`；瓶颈仍是 `3046.794ms` packed restore executor 和 `469.372ms` connector load。因此当前贡献应表述为“PrePass lead-time 下的 online TTFT 改善”，而不是“请求到达后同步恢复也满足 SLA”。
- 产物包括 `results/m3_15_8k_baseline_same_config/b0_b2_b5_8k_comparison.md`、`b0_b2_b5_8k_comparison.csv` 和 `b0_b2_b5_8k_summary.json`。M3.15 研究报告已更新同配置 baseline 口径。

## 2026-05-28 M3.15 2K/8K/16K packed PrePass 趋势发现
- 按用户要求本轮不跑 512，直接补齐 2K 与 16K packed PrePass gate，并复用既有 8K 样本形成趋势表。新报告为 `docs/research/m3_15_2k16k_packed_prepass_trend.md`，汇总产物为 `results/m3_15_2k16k_trend/trend_summary.csv` 和 `trend_summary.json`。
- 2K/16K 均使用真实路径：`store -> packed demote -> restart vLLM -> /prepass -> /prefetch/advance -> reuse`。2K 使用 `--max-model-len 16384 --max-num-batched-tokens 16384`；16K 使用 `--max-model-len 32768 --max-num-batched-tokens 32768`，避免 chunked prefill 再次截断保存 KV。
- 2K 有效性：保存 `2048` tokens、`128` blocks、`48` layers，packed object `402657408` bytes，`cold_saved_token_mismatch=no`；reuse 阶段 `external_load_observed=yes`、`load_events=1`、`store_events=0`、`sync_ssd_miss_total=0`。
- 16K 有效性：保存 `16384` tokens、`1024` blocks、`48` layers，packed object `3221229696` bytes，`cold_saved_token_mismatch=no`；reuse 阶段同样观测到外部 load、无写回污染、无同步 SSD miss。
- 2K 结果：B0 TTFT `673.969ms`，B2 strict restart `689.483ms`，B5 online `301.511ms`，B5 online/B0 `0.447366x`；restore executor `823.925ms`，request-arrival restore-inclusive `1151.451ms`，restore-inclusive/B0 `1.708463x`。
- 8K 结果：B0 TTFT `2409.141ms`，B2 `2418.796ms`，B5 online `710.835ms`，B5 online/B0 `0.295057x`；restore executor `3046.794ms`，request-arrival restore-inclusive `3797.230ms`，restore-inclusive/B0 `1.576176x`。
- 16K 结果：B0 TTFT `5493.655ms`，B2 `5533.513ms`，B5 online `882.298ms`，B5 online/B0 `0.160603x`；restore executor `4581.719ms`，request-arrival restore-inclusive `5519.967ms`，restore-inclusive/B0 `1.004790x`。
- 趋势判断：方向稳定。随着上下文从 2K 增至 16K，B5 online 相对 B0 的优势持续增强，说明如果 PrePass restore 能被工作流提前量隐藏，persistent KV Anti-Caching 的收益会随历史上下文增大而增强。
- 谨慎判断：2K/8K 如果请求来了才开始 restore，仍明显慢于 B0；16K 的 request-arrival restore-inclusive 才接近打平 B0。这说明当前系统已经看到 break-even 拐点，但还不能声称任意长上下文下“同步恢复也满足 SLA”。
- B2 在 2K/8K/16K 均接近 B0，说明 vLLM APC 跨进程重启不保留历史 KV，这继续支持 persistent KV 复用问题的必要性。
- 新瓶颈：语义链路已经不是主问题，P0 应转向 packed restore executor 与 CPU-ready->GPU connector load。connector load 从 2K 的 `117.915ms` 增到 16K 的 `610.991ms`，已经占 online TTFT 的大部分；restore executor 仍是 request-arrival 口径的主导项。
- 本轮启动 vLLM connector 时首次漏设 `PYTHONPATH=/root/KV`，导致 `ModuleNotFoundError: No module named 'benchmarks'`。该失败未纳入实验结果；后续 connector 版 vLLM 启动命令必须显式带 `PYTHONPATH=/root/KV`。

## 2026-05-28 M3.16 LongMemEval workload 接入发现
- LongMemEval cleaned 官方 Hugging Face dataset metadata 显示 license 为 MIT，包含 `longmemeval_s_cleaned.json`、`longmemeval_m_cleaned.json` 和 `longmemeval_oracle.json`。官方 README 说明 LongMemEval-S 拼接历史约 115K tokens，字段包括 `question_id`、`question_type`、`question`、`answer`、`question_date`、`haystack_session_ids`、`haystack_dates`、`haystack_sessions` 和 `answer_session_ids`。
- LongMemEval 与本项目研究问题自然对齐：`haystack_sessions` 可作为可持久化、可复用的 historical KV prefix；当前 `question` 是在线 suffix；`answer` 和 evidence session 可作为未来语义正确性 sanity check，但 TTFT/restore 实验不依赖答案评测。
- 已实现 `benchmarks/m3/longmemeval_workload.py`，输出 `longmemeval_workload.jsonl/csv/summary/report`。核心策略是保留最近历史 token 作为 prefix，并保留当前问题开头作为 suffix；默认用 Qwen2.5 tokenizer 计数，测试用 whitespace tokenizer 保持确定性。
- `run_reuse_smoke_matrix.py` 已支持 `--workload-manifest`。这意味着现有 B5 packed PrePass/cold-tier 路径可以消费 LongMemEval 真实多轮文本 prompt，而不再只能用 synthetic `cache` token。
- 当前已完成 fixture 级 dry-run：`results/m3_16_longmemeval_fixture/manifest/` 证明 adapter 能生成 manifest；`results/m3_16_longmemeval_fixture/reuse_dryrun/` 证明 reuse runner 能消费 manifest。这是接口验证，不是官方 LongMemEval-S 性能结果。
- 官方数据下载在本轮网络条件下不稳定：S 文件约 277MB，oracle 约 15MB；`curl` 留下半截 JSON，`aria2c` 出现 TLS handshake failure，`datasets` streaming 长时间不返回。因此本轮未声称已完成官方 LongMemEval-S 真实 vLLM 结果。
- 已补完整性检查，防止 partial JSON 被误用。后续拿到官方文件后，第一步应先运行 adapter manifest 命令，再用 1 个 2K 样本接入真实 store -> packed demote -> restart -> PrePass -> reuse path。

## 视觉/浏览器发现
- 本轮尚未使用视觉或浏览器资料。

## 2026-05-28 M3.17 论文初稿发现
- 本轮将项目当前叙事收敛为英文系统论文草稿 `docs/paper/persistent_kv_anti_caching.md`，标题为 **Persistent KV Anti-Caching for Multi-Turn Long-Context LLM Serving**。草稿重点完成 Abstract、Introduction、Related Work 和 Design，按 CCF-A 系统论文风格写作，但保持 prototype 证据边界。
- 论文核心问题被表述为：多轮长上下文服务中，历史 KV 是可复用的状态对象；HBM/DRAM 无法容纳所有历史 KV；SSD/3FS 只能作为 cold persistence tier；系统必须在请求进入 GPU 前 exact、可验证、可准入地恢复所需历史 KV，避免 cold miss 污染 Prefill/Decode critical path。
- 核心方法被表述为数据库 Anti-Caching 到 KV serving 的迁移：KV Evicted Index 对应 Evicted Table，Packed Cold Object 对应 Block Table，KV PrePass 对应 pre-pass execution，DELAY/requeue/ready barrier 对应 abort/restart 前的非阻塞恢复语义。
- Related Work 的关键边界：KVDrive 已经覆盖 GPU/DRAM/SSD holistic multi-tier KV 管理，因此本文不能泛泛声称“多级 KV 管理”是创新；我们的创新必须聚焦 persistent historical KV objects、correctness-keyed exact reuse、PrePass admission、ready barrier 和 no synchronous cold-tier miss。
- 论文证据表 `docs/paper/evidence_table.md` 记录了当前可以支撑的结论：B5 online TTFT 在 synthetic 2K/8K/16K 中为 B0 的 `0.447x/0.295x/0.161x`；LongMemEval-S 单样本 2K/8K 为 `0.537x/0.297x`；但 restore-inclusive 在 2K/8K 仍高于 B0，不能声称请求到达后同步恢复满足 SLA。
- 新增 `docs/paper/references.bib`，包含 Anti-Caching、PagedAttention、vLLM APC、LMCache、Mooncake、Tutti、KVDrive、CacheBlend 和 LongMemEval。当前引用以 arXiv/官方文档/PVLDB 信息为准，未声称 KVDrive 已被 SIGMOD 2026 收录。
- 论文中明确保留当前缺口：尚无真实 3FS 集群性能、尚无强 baseline 全面对比、尚无 p95/p99 并发结果、尚未证明 admitted workload historical byte hit rate >=90%、尚未实现 1M 真实端到端服务。

---
*每执行2次查看/浏览器/搜索操作后更新此文件。*
