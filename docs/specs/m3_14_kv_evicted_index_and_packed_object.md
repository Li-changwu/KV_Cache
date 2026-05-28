# M3.14 设计规格：KV Evicted Index 与 Packed Cold Object

## 背景

数据库 Anti-Cache 的关键启发不是“提前百分百预测事务会访问什么”，而是：

1. 数据本体可以被驱逐到磁盘；
2. 数据状态索引必须留在内存；
3. 请求执行前或执行中一旦发现访问冷数据，就进入 pre-pass；
4. pre-pass 收集冷数据所在的 block id / offset；
5. 后台异步读取；
6. 数据合并回内存后，请求重新执行。

迁移到 KV 系统后，我们不能让 GPU 做真正的错误试跑，因为缺失历史 KV 时注意力语义不完整。我们应该做的是 **控制面试跑**：请求进入 vLLM 前，先在 sidecar/control plane 查内存 KV 索引，判断所需 KV 是 `GPU_READY`、`CPU_READY` 还是 `SSD_COLD`。如果在 SSD/3FS，就延迟请求、异步恢复到 CPU；如果只在 CPU/DRAM，还需要进入 CPU->GPU load 准备阶段，不能把它和 HBM ready 混为一谈。

因此 M3.14 最应该落地两个模块：

1. **KV Evicted Index**：常驻 CPU 内存的 KV 状态索引表。
2. **Packed Cold Object**：SSD/3FS 上的打包式冷 KV 对象。

这两个模块分别对应 Anti-Cache 的 Evicted Table 和 Block Table。

## 模块一：KV Evicted Index

### 目标

让系统在请求进入 GPU 前快速回答：

> 这次请求需要的历史 KV 当前在 HBM、CPU 还是 SSD？是否 ready？如果在 SSD，它在哪个冷对象、哪个 offset、需要读多长？

它不保存 KV 张量本体，只保存元数据。

### 三级状态语义

M3.14-B2 后，索引和 PrePass 使用以下细分状态：

- `GPU_READY`：KV 已在 HBM，满足执行 ready。
- `CPU_READY`：KV 已在 CPU/DRAM 且校验通过，但仍需要 CPU->GPU load。
- `SSD_COLD`：KV 在 SSD/3FS 冷区或持久化层，必须先 SSD/3FS->CPU restore。
- `FETCHING`：SSD/3FS->CPU restore 进行中。
- `LOADING`：CPU/DRAM->GPU/HBM load 进行中。
- `MISSING`：没有可复用 KV。
- `MISMATCH`：correctness key 不匹配，禁止复用。

`ready` 与 `cold` 可作为兼容汇总字段保留：`ready = gpu_ready + cpu_ready`，`cold = ssd_cold`。新的调度逻辑必须使用细分状态。

### 第一版数据模型

建议新增 `benchmarks/m3/kv_evicted_index.py`。

核心数据结构：

```python
KVIndexEntry:
  prefix_id: str
  token_start: int
  token_end: int
  correctness_key: dict
  tier: "HBM" | "CPU" | "DRAM" | "SSD" | "FETCHING" | "LOADING"
  ready: bool
  object_id: str | None
  cold_uri: str | None
  extents: list[KVExtentRef]
  checksum: str
  size_bytes: int
  last_access_epoch: int
  access_count: int
  fetching_request_id: str | None
  loading_request_id: str | None
```

```python
KVExtentRef:
  layer_name: str
  token_start: int
  token_end: int
  offset: int
  length: int
  checksum: str
```

第一版可以先按 prefix/range 粒度做索引，不按每个 token 建索引。后续再实验 512/1024 token extent 粒度。

### 必须支持的操作

- `upsert_from_manifest(manifest)`：从现有 `KVBlockManifest` 导入或更新索引。
- `lookup_required(prefix_id, token_start, token_end, correctness_key)`：返回该请求需要的 KV 状态。
- `classify_required(...)`：输出 `GPU_READY`、`CPU_READY`、`SSD_COLD`、`FETCHING`、`LOADING`、`MISSING`、`MISMATCH`，并保留 `ready/cold` 兼容汇总。
- `mark_fetching(prefix_id, range, request_id)`：恢复任务入队后标记为恢复中。
- `mark_ready(prefix_id, range, target_tier="CPU")`：恢复完成并校验后标记 ready。
- `mark_loading(prefix_id, range, request_id)`：CPU->GPU load 入队或执行中时标记为 loading。
- `mark_evicted(prefix_id, range, object_id, extents)`：落入 SSD 后更新 object/extent 信息。
- `touch(prefix_id, range)`：访问后更新热度。

### 和现有系统的接入点

第一阶段不要替换现有 `KVManifest`，而是作为 sidecar 的增强索引：

- `commit_from_payload()` 或 connector auto commit 后，把 manifest 同步进 `KV Evicted Index`。
- `prepass_from_payload()` 先查 `KV Evicted Index`，再决定是否入队恢复。
- `_apply_residency_hits()` 从直接读 tensor store manifest，逐步迁移为查索引。
- `advance_prefetch_from_payload()` 恢复完成后调用 `mark_ready()`。

### 成功标准

单元测试先覆盖：

- HBM ready 的 range 返回 `GPU_READY`；
- CPU/DRAM ready 的 range 返回 `CPU_READY`；
- SSD/not-ready 的 range 返回 `SSD_COLD`，并带 object id / offset / length；
- FETCHING 状态不会重复入队；
- LOADING 状态不会被误判为 SSD cold；
- correctness key 不匹配返回 MISMATCH；
- missing range 返回 MISSING；
- 恢复完成后从 `SSD_COLD/FETCHING` 变为 `CPU_READY`；CPU->GPU load 完成后才是 `GPU_READY`。

工程指标：

- PrePass 不再只依赖零散 manifest 文件判断状态；
- 每个请求可以输出 gpu_ready/cpu_ready/ssd_cold/fetching/loading/missing/mismatch sets；
- 所有 SSD range 必须先进入恢复队列，不能直接进入 vLLM load。
- 所有 CPU_READY range 仍需经过 CPU->GPU load 预算判断，不能直接等同于 GPU_READY。

## 模块二：Packed Cold Object

### 目标

把当前每层一个 safetensors 文件的冷层布局，升级成一个或少量连续大对象，减少长上下文下的小文件开销和恢复尾延迟。

它对应 Anti-Cache 的 Block Table。

### 第一版布局

建议新增或扩展 `benchmarks/m3/cold_tier.py`：

- 保留现有 `local_posix` 和 `3fs_posix` adapter。
- 新增 `PackedColdTierAdapter` 或在 adapter 中增加 `layout="packed_v1"`。

第一版推荐 **extent-major + layer-inner**：

```text
packed_object.bin
  header
  extent 0: tokens [0, 512)
    layer 0 KV bytes
    layer 1 KV bytes
    ...
  extent 1: tokens [512, 1024)
    layer 0 KV bytes
    layer 1 KV bytes
    ...
  ...

packed_manifest.json
  object_id
  prefix_id
  token_start
  token_end
  block_size
  extent_tokens
  layout
  extents[]
  checksum
```

如果第一版切 token extent 的成本太高，也可以先做 **layer-major packed**：

```text
packed_object.bin
  layer 0 bytes
  layer 1 bytes
  ...
```

但从研究价值看，extent-major 更能支撑后续部分恢复。

M3.14-C 当前实现状态：第一版已落地 `packed_v1` backend，采用 layer-major packed object。每个 cold object 目录包含 `packed_object.bin` 和 `packed_manifest.json`，每层原始 safetensors 字节在 manifest / offset table 中记录 `offset`、`size_bytes`、`checksum` 和 `packed_layout=packed_v1`。这一步先减少冷区文件数量并统一顺序读入口；extent-major 和 partial restore 仍是后续增强。

### 必须支持的操作

- `demote_packed(prefix_id, extent_tokens=512)`：把 hot tensor files 打包为 packed object。
- `summarize_packed(uri)`：读取 packed manifest，计算 checksum 和 bytes。
- `restore_packed(prefix_id, required_range=None)`：恢复整个对象或指定 token range。
- `verify_packed()`：恢复前后校验 object checksum 或 extent checksum。

### 和 KV Evicted Index 的关系

Packed object 生成后，必须把 object id / extent offset / length 写入 `KV Evicted Index`。这样 PrePass 才能不读磁盘目录，只靠内存索引快速知道需要恢复哪些 bytes。

### 成功标准

功能测试：

- packed demote 后 hot per-layer 文件被移除或标记不可用；
- index 中能查到 object id / extents；
- packed restore 后 manifest 变为 CPU/ready；
- checksum mismatch 会失败，不会标记 ready。

性能实验：

- 对同一个 2K/8K prefix，对比 per-layer cold object 和 packed cold object：
  - demote elapsed
  - restore elapsed
  - restore p50/p95/p99
  - effective MiB/s
  - file count
  - checksum time

M3.13 planner 已估算 8K 需要约 4.29s lead。M3.14 的目标是验证 packed layout 是否能降低 8K restore tail，为后续 16K/32K gate matrix 提供依据。

## 请求流程：M3.14 后的目标形态

1. 请求到达 sidecar。
2. sidecar 根据 prefix candidates 枚举 required KV ranges。
3. `KV Evicted Index` 分类：
   - GPU_READY：HBM 可执行；
   - CPU_READY：CPU/DRAM 已恢复并校验，但需要 CPU->GPU load；
   - SSD_COLD：SSD/3FS 上有对象，需要恢复；
   - FETCHING：已经在恢复，等待；
   - LOADING：正在 CPU->GPU load；
   - MISSING：没有可复用 KV；
   - MISMATCH：正确性不匹配，禁止复用。
4. 如果全部 GPU_READY，请求可以直接进入 vLLM。
5. 如果存在 CPU_READY，PrePass 或 admission 必须检查 CPU->GPU load 预算，并进入 load 准备阶段。
6. 如果存在 SSD_COLD，PrePass 记录 object/offset/length，入队 SSD/3FS->CPU restore。
7. 请求返回 DELAY，不进入 GPU。
8. restore 完成并校验后，Index 更新为 CPU_READY。
9. CPU->GPU load 完成后更新为 GPU_READY，或由 connector 在执行前完成加载。
10. vLLM connector 加载 historical KV 后，只计算新增 token。

## 实施顺序

### M3.14-A：KV Evicted Index 独立模块

优先级：P0。

先不接真实 vLLM，只做纯内存和 manifest 的单元测试。

产物：

- `benchmarks/m3/kv_evicted_index.py`
- `tests/m3/test_kv_evicted_index.py`
- `docs/specs/m3_14_kv_evicted_index_and_packed_object.md`

### M3.14-B：PrePass 输出三级分类集合

优先级：P0。

把 `/prepass` 的结果从现在的计数状态，升级为明确的三级分类集合。第一版可以仍然用现有 `KVTensorStore`，但接口要对齐 `KV Evicted Index`。

产物：

- 更新 `benchmarks/m3/prepass_planner.py`
- 更新 `benchmarks/m3/http_sidecar.py`
- 增加 prepass classification tests

### M3.14-B3：CPU_READY -> GPU_READY load readiness 观测

优先级：P0。

`CPU_READY` 只能说明 SSD/3FS->CPU restore 已经完成。要满足 Prefill/Decode ready，还需要显式记录 CPU->GPU load 时间、预算和状态。该阶段先不改 attention kernel，只建立观测口径和控制面字段。

产物：

- PrePass 输出 `load_readiness`，包含 `cpu_ready_not_gpu_ready_count`、`cpu_ready_tokens`、`expected_h2d_load_bytes`、`estimated_h2d_load_ms`。
- connector load event 与 PrePass classification 关联。
- 指标区分 `restore_wait_ms`、`connector_load_elapsed_ms`、`cpu_ready_not_gpu_ready_total`。

M3.14-B3 的边界：当前阶段只建立 CPU->GPU load 预算与观测字段，不把 CPU_READY 自动提升为 GPU_READY，也不修改 vLLM attention kernel。真正的执行闭环需要后续把 connector load 完成事件回写到 `KV Evicted Index`，再由 admission/prepass 确认 `GPU_READY`。

### M3.14-C：Packed Cold Object v1

优先级：P0。

先做 local_posix packed object，再考虑 3fs_posix。第一版可以从 layer-major packed 起步，但接口必须保留 extent 字段。

产物：

- `PackedColdTierAdapter` 或 `packed_v1` backend
- `tests/m3/test_packed_cold_object.py`
- packed vs per-layer microbench

当前已完成：

- `benchmarks/m3/cold_tier.py` 中的 `PackedColdTierAdapter`
- `tests/m3/test_packed_cold_object.py`
- `run_cold_tier_adapter_bench.py --cold-backend packed_v1`

### M3.14-D：2K/8K packed vs per-layer restore 对比

优先级：P0。

只有当 8K packed restore tail 可控时，才进入 16K/32K gate matrix。

当前 M3.14-D 结果已经生成到 `results/m3_14_packed_vs_per_layer/`。在 `qwen25_14b_tiny` profile、2K/8K、repeats=3、queue_depth=1 下，`packed_v1` 把 cold data files 从 24 降到 6，但 restore p95 从 `257.851ms` 增加到 `531.195ms`。按 token 拆分，2K 为 `60.637ms` vs `102.856ms`，8K 为 `259.590ms` vs `531.309ms`。因此当前 packed_v1 是可用的功能原型，但还不是性能胜出的 packed restore 路径。

新的推进条件：先做 M3.14-E packed restore 数据面优化，或至少证明优化后 8K restore tail 不劣于 per-layer baseline，再进入 16K/32K gate matrix。

### M3.14-E：Packed Restore 数据面优化

优先级：P0。

M3.14-D 的反例说明：只把多个 layer 文件“打包成一个文件”还不够。如果实现路径仍然把整个文件读成 bytes、再切片、再写出，packed object 会因为 Python 内存拷贝和重复 checksum 变慢。

本阶段已完成的修正：

- demote 使用 chunk streaming append，不再 `read_bytes()` 后整体写入；
- restore 使用 extent streaming copy，不再整段 bytes 切片；
- restore 前的 `summarize()` 在 packed object 大小匹配 manifest 时复用 manifest checksum，不再重读整个 packed object；
- demote 收尾的 object checksum 复用 append 阶段已计算的 extent checksum，不再第二次读取 `packed_object.bin`；
- manifest 与 offset table 标记 `copy_mode=streaming`，便于实验报告区分实现版本。

优化后的结果：

- `results/m3_14_packed_vs_per_layer_fastmanifest/`：2K/8K、repeats=3 下，local_posix restore p50/p95/p99 为 `158.667/254.746/255.478ms`，packed_v1 为 `143.334/233.673/235.484ms`，packed restore p95 delta 为 `-21.073ms`。
- `results/m3_14_packed_vs_per_layer_streaming_summary_r5/`：2K/8K、repeats=5 下，local_posix restore p50/p95/p99 为 `216.930/283.665/287.823ms`，packed_v1 为 `145.103/225.144/225.181ms`，packed restore p95 delta 为 `-58.521ms`，cold data files 从 `40` 降到 `10`。
- r5 by-token：2K local/packed restore p95 为 `159.497ms` vs `66.766ms`；8K local/packed restore p95 为 `286.552ms` vs `225.169ms`。

谨慎结论：packed_v1 已经从“减少文件数但 restore 明显更慢”修正为“local POSIX 小矩阵中可与 per-layer 打平或胜出”。这足以支持继续做 8K/16K gate matrix；但当前结果仍是 synthetic `qwen25_14b_tiny` profile、本地 POSIX、Python safetensors 恢复路径，不代表真实 3FS/RDMA/GDS/io_uring 生产 executor。

## 风险与边界

- `KV Evicted Index` 不能成为新的大内存负担，因此索引粒度不能过细。
- index 中的状态必须以 correctness key 为硬约束，不能因为 prefix_id 相同就复用。
- SSD 上的 KV 永远不是 ready；只有恢复到 CPU 并校验后才是 CPU_READY，加载到 HBM 后才是 GPU_READY。
- packed object 第一版不要引入压缩，避免改变 exact semantics。
- 不要把 CPU->HBM 与 SSD->CPU 混成一个队列；前者是执行前加载，后者是冷区恢复。

## 论文叙事中的位置

M3.14 是把数据库 Anti-Cache 迁移到 KV 系统的关键阶段：

- `KV Evicted Index` 说明我们如何判断请求访问的是 HBM 执行 KV、CPU ready KV 还是 SSD cold KV；
- `Packed Cold Object` 说明我们如何让冷 KV 恢复足够高效；
- `PrePass + DELAY + async restore + re-admit` 说明我们如何避免在线推理同步踩 SSD；
- GPU/CPU/SSD 三层状态让我们的方案区别于单纯 SSD KV cache 或 decode-only working-set 管理。
