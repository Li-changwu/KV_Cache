# M3.9 DRAM/NVMe 分层迁移原型

## 目标

M3.9 的目标不是实现完整异步 I/O 引擎，而是在 M3.8 已证明真实外部 KV 加载后，先给本地 KV tensor store 增加可观测、可测试的层级状态：

- `DRAM`：执行可达层，可被 connector 直接加载。
- `NVME`：容量层，不允许 decode 关键路径直接同步读取。
- `ready`：是否已经完成准入前预取。

这一步用于把“文件存在”与“当前可以安全加载”分开，为后续异步预取、淘汰和 3FS 接入做准备。

## 当前实现

- `KVBlockManifest` 新增 `tier` 与 `ready` 字段。
- `KVTensorStore.demote_to_nvme(prefix_id)` 将前缀标记为 `NVME` 且 `ready=false`。
- `KVTensorStore.prefetch_to_dram(prefix_id)` 将前缀恢复为 `DRAM` 且 `ready=true`。
- `KVTensorStore.load_layer()` 在加载前检查层级状态；如果前缀仍在 `NVME` 或未 ready，会抛出 `TierNotReady`。
- 迁移事件写入每个前缀目录下的 `migration_events.jsonl`。
- 新增 `PrefetchQueue`，根据 `tokens * kv_bytes_per_token / storage_gbps` 估算 NVMe 到 DRAM 的迁移耗时，并写出 `prefetch_queue_events.jsonl`。
- HTTP sidecar 可以配置 `tensor_store_path`。当准入结果为 `DELAY`，且原因是 `required_kv_not_ready_before_decode` 或 `prefetch_misses_admission_window` 时，sidecar 会在记录本次延迟决策之后把请求提交给 `PrefetchQueue`。
- 若队列估算能赶上 `deadline_ms`，才调用 `prefetch_to_dram()` 并把控制面 manifest 标记为 `DRAM/ready`；若估算错过截止时间，则记录 `DEADLINE_MISS`，保留 `NVME/ready=false`。
- 这条链路刻意保持“当前请求仍然延迟，后续请求才可准入”，避免把 NVMe/SSD 读取塞进当前 decode 关键路径。
- `/metrics` 现在暴露 `prefetch_queue_*` 指标，包括请求数、完成数、deadline miss 数、累计字节数、最大队列深度和最近预计完成时间。
- sidecar 已增加驻留感知入口：如果控制面 manifest 还是 `SSD/ready=false`，但本地 tensor store 已经是 `DRAM` 或 `HBM` 且 `ready=true`，并且 correctness key 与请求完全一致，sidecar 会先把控制面同步为 ready，再重新准入当前请求。这个路径记录为 `RESIDENCY_HIT`，不会提交 `PrefetchQueue`。
- 如果 tensor store manifest 存在但 correctness key 不匹配，sidecar 不会预取，也不会把它标记为 ready；该情况会写入 `prefetch_to_dram` 错误记录。
- `/metrics` 新增 `residency_hit_total`、`prefetch_queued_total`、`prefetch_deadline_miss_total`，用于区分“已有执行层命中”“冷层预取入队”和“预取赶不上截止时间”。
- 新增 `AsyncPrefetchQueue`。异步模式下 `submit()` 只返回 `QUEUED`，manifest 仍保持 `NVME/ready=false`；只有显式推进后台队列后，才写出 `COMPLETED` 并调用 `prefetch_to_dram()`。
- sidecar CLI 可通过 `--async-prefetch` 开启异步预取模式。测试和烟测中使用 `/prefetch/advance` 或 `SidecarRuntime.advance_prefetch_from_payload()` 做确定性推进，避免真实时间等待污染结果。
- 异步队列会对同一前缀去重：若前缀已在后台队列中，后续请求记录 `ALREADY_QUEUED`，不会重复计入 `prefetch_queued_total`。

## 冒烟验证

脚本：

```bash
python benchmarks/m3/run_tier_migration_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_tier_migration_smoke \
  --prefix-ids m3-8-16 m3-8-64 m3-8-128 m3-8-256
```

输出：

- `results/m3_9_tier_migration_smoke/tier_migration_smoke.csv`
- `results/m3_9_tier_migration_smoke/tier_migration_summary.json`
- `results/m3_9_tier_migration_smoke/tier_migration_report.md`

本轮结果显示 4 个前缀都完成了 `DRAM -> NVME -> DRAM` 状态循环。

侧车预取烟测：

```bash
python benchmarks/m3/run_sidecar_prefetch_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_sidecar_prefetch_smoke \
  --prefix-id m3-8-64
```

输出：

- `results/m3_9_sidecar_prefetch_smoke/sidecar_prefetch_smoke.csv`
- `results/m3_9_sidecar_prefetch_smoke/sidecar_prefetch_summary.json`
- `results/m3_9_sidecar_prefetch_smoke/sidecar_prefetch_report.md`
- `results/m3_9_sidecar_prefetch_smoke/decisions.jsonl`

本轮结果显示：第一次准入为 `DELAY/required_kv_not_ready_before_decode`，sidecar 随后触发 `prefetch_to_dram` 事件；第二次同一前缀准入为 `ADMIT/required_kv_ready_before_decode`。

带宽与截止时间队列烟测：

```bash
python benchmarks/m3/run_sidecar_prefetch_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_sidecar_prefetch_queue_smoke/ok \
  --prefix-id m3-8-64

python benchmarks/m3/run_sidecar_prefetch_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_sidecar_prefetch_queue_smoke/deadline_miss \
  --prefix-id m3-8-256 \
  --storage-gbps 0.000001 \
  --request-token-count 288
```

结果：

- `ok`：第一次 `DELAY`，预取状态 `COMPLETED`，预计完成约 `1.787ms`，第二次 `ADMIT`。
- `deadline_miss`：第一次 `DELAY`，预取状态 `DEADLINE_MISS`，预计完成约 `53477376ms`，第二次仍为 `DELAY`，manifest 保持 `NVME/ready=false`。

多请求队列烟测：

```bash
python benchmarks/m3/run_prefetch_queue_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_prefetch_queue_smoke \
  --prefix-ids m3-8-16 m3-8-64 m3-8-128 m3-8-256 \
  --storage-gbps 0.002 \
  --deadline-ms 12000
```

输出：

- `results/m3_9_prefetch_queue_smoke/prefetch_queue_smoke.csv`
- `results/m3_9_prefetch_queue_smoke/prefetch_queue_summary.json`
- `results/m3_9_prefetch_queue_smoke/prefetch_queue_report.md`
- `results/m3_9_prefetch_queue_smoke/prefetch_queue_events.jsonl`

本轮结果：4 个前缀中 2 个 `COMPLETED`，2 个 `DEADLINE_MISS`，累计迁移字节数 `103809024`，最近预计完成时间 `37748.736ms`。

驻留感知预取烟测：

```bash
python benchmarks/m3/run_residency_aware_prefetch_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_residency_aware_prefetch_smoke \
  --prefix-ids m3-8-16 m3-8-64 m3-8-128 m3-8-256 \
  --storage-gbps 0.002 \
  --deadline-ms 12000
```

输出：

- `results/m3_9_residency_aware_prefetch_smoke/residency_aware_prefetch_smoke.csv`
- `results/m3_9_residency_aware_prefetch_smoke/residency_aware_prefetch_summary.json`
- `results/m3_9_residency_aware_prefetch_smoke/residency_aware_prefetch_report.md`
- `results/m3_9_residency_aware_prefetch_smoke/prefetch_queue_events.jsonl`

本轮真实张量仓库结果：4 个前缀中 3 个 `RESIDENCY_HIT`，没有进入队列；`m3-8-256` 为 `NVME/ready=false`，在 `0.002GB/s` 和 `12000ms` deadline 下预计 `26738.688ms`，因此为 `DEADLINE_MISS`。

异步预取烟测：

```bash
python benchmarks/m3/run_async_prefetch_smoke.py \
  --tensor-store results/m3_8_full_restart_matrix/tensor_store \
  --result-dir results/m3_9_async_prefetch_smoke \
  --prefix-id m3-8-64 \
  --storage-gbps 8.8 \
  --advance-ms 12000
```

输出：

- `results/m3_9_async_prefetch_smoke/async_prefetch_smoke.csv`
- `results/m3_9_async_prefetch_smoke/async_prefetch_summary.json`
- `results/m3_9_async_prefetch_smoke/async_prefetch_report.md`
- `results/m3_9_async_prefetch_smoke/decisions.jsonl`

本轮真实张量仓库结果：第一次请求 `DELAY` 并返回 `QUEUED`；推进后台队列前第二次请求仍为 `DELAY`；推进后队列返回 `COMPLETED`，第三次请求变为 `ADMIT`。这证明异步模式不会在当前请求里伪装 ready。

## 限制

- 当前迁移是元数据级模拟，没有真实搬运到另一个目录、另一个设备或 3FS。
- 当前异步队列是确定性后台队列，需要显式推进；它不是实际线程池、真实睡眠时间或真实 I/O 并发。
- 当前带宽估算只覆盖单流 NVMe 到 DRAM 的容量层迁移时间；还没有建模 H2D、3FS、队列深度并发、真实 tail latency 或抢占。
- `run_prefetch_queue_smoke.py` 直接提交队列，主要验证排队估算；完整准入层 residency hit 判断由 `run_residency_aware_prefetch_smoke.py` 和 sidecar 测试覆盖。

## 下一步

- `run_reuse_smoke_matrix.py` 现在会在每个在线矩阵行前后读取 sidecar `/metrics`，并把 `residency_hit_delta`、`prefetch_queued_delta`、`prefetch_deadline_miss_delta`、`prefetch_queue_*_delta` 和 `prefetch_queue_pending` 写入 CSV 与报告。
- 干跑模式也保留这些列，便于后续真实在线矩阵和离线报告使用同一张表结构。
- 将本地 `NVME` 状态替换为真实目录/设备级 cold tier，后续再接 3FS。
- 将确定性后台队列替换或扩展为真实后台线程、真实设备级 cold tier 和 3FS 接入。
