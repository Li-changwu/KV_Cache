# M3 vLLM Connector / Sidecar 最小接口

## 目标

M3 原型只验证分层 KV 控制面的最小闭环：网关或 sidecar 在请求进入 vLLM 前完成复用判断、准入、预取计划和 ready barrier 判断；vLLM connector 只负责按明确的 block range 搬运 KV，不负责策略决策。

第一版不修改 attention kernel，不改变精确 attention 语义，不允许 decode 路径同步读 SSD/3FS。

## 组件边界

| 组件 | 职责 | 不负责 |
|------|------|--------|
| Gateway / Sidecar | 解析请求、计算正确性键、查 prefix/session manifest、生成准入与预取计划 | GPU block 分配细节、attention 计算 |
| Tiered KV Control Plane | 维护 HBM/DRAM/SSD residency、预测复用、做 eviction/admission 决策 | 修改模型输出语义 |
| vLLM KV Connector | 执行 KV block 的 load/store/pin/unpin，返回 ready 状态 | 决定哪些 block 值得复用 |
| vLLM Scheduler | 在 KV ready 后执行 prefill/decode | 等待同步 SSD miss |

## 请求入口

Sidecar 接收原始推理请求后，先生成控制面请求：

```json
{
  "request_id": "req-0001",
  "model_id": "/root/models/Qwen3-32B",
  "token_count": 1048576,
  "decode_sla_ms": 250,
  "admission_window_ms": 12000,
  "correctness_key": {
    "model_fingerprint": "string",
    "tokenizer_fingerprint": "string",
    "rope_config": "string",
    "dtype": "bf16",
    "kv_layout": "vllm-paged"
  },
  "prefix_candidates": [
    {
      "prefix_id": "session-a",
      "token_start": 0,
      "token_end": 786432,
      "committed": true
    }
  ]
}
```

## 准入输出

控制面必须在请求交给 vLLM 前返回：

```json
{
  "request_id": "req-0001",
  "decision": "ADMIT",
  "reason": "required_kv_ready_before_decode",
  "reuse_tokens": 786432,
  "delta_prefill_tokens": 262144,
  "estimated_prefetch_ms": 8200,
  "estimated_ttft_ms": 9800,
  "sync_ssd_miss_allowed": false,
  "required_ranges": [
    {
      "prefix_id": "session-a",
      "token_start": 0,
      "token_end": 786432,
      "required_tier": "DRAM_OR_HBM",
      "deadline_ms": 12000
    }
  ]
}
```

`decision` 取值：

| 值 | 含义 |
|----|------|
| `ADMIT` | 所需 KV 能在 decode 前就绪 |
| `DELAY` | 容量足够，但预取或 prefill 赶不上当前 SLA |
| `REJECT` | 容量不足，或正确性键不匹配，不能进入精确 SLA 队列 |
| `FULL_PREFILL_FALLBACK` | 可执行全量 prefill，但不能声明分层 KV SLA 收益 |

## Connector 操作

M3 只需要四类操作：

| 操作 | 输入 | 输出 |
|------|------|------|
| `lookup(range)` | correctness key、prefix id、token range | residency tier、block ids、committed 状态 |
| `prefetch(plan)` | block ids、source tier、target tier、deadline | accepted、estimated ready time、failure reason |
| `pin(blocks)` | block ids、sequence id | pinned 或失败原因 |
| `release(blocks)` | block ids、sequence id、writeback hint | released、writeback scheduled |

## Ready Barrier

vLLM decode 前必须检查：

```json
{
  "request_id": "req-0001",
  "all_required_blocks_ready": true,
  "missing_blocks": [],
  "sync_ssd_miss_total": 0
}
```

若 `all_required_blocks_ready=false`，请求只能延迟、重排或降级，不能在 decode critical path 同步读 SSD/3FS。

## 最小观测指标

- `admission_decision_total{decision,reason}`
- `prefill_tokens_saved_total`
- `effective_kv_hit_rate`
- `prefetch_deadline_miss_total`
- `sync_ssd_miss_total`
- `hbm_resident_tokens`
- `dram_resident_tokens`
- `ssd_resident_tokens`
- `h2d_bytes_total`
- `storage_read_bytes_total`

## M3 验收条件

- 同一会话第二轮请求能复用已 committed prefix，只对新增 token 做 delta prefill。
- decode 前 ready barrier 能阻止未就绪 KV 进入同步 SSD 读取路径。
- sidecar 能把低局部性 1M 请求标为 `DELAY`、`REJECT` 或 `FULL_PREFILL_FALLBACK`。
- 所有失败、容量不足、正确性键不匹配和 deadline miss 都有结构化原因。
