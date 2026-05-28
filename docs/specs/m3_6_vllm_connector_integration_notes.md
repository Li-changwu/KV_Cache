# M3.6 vLLM KV Connector 接入点评估

## 结论

M3.5 的 HTTP sidecar 已能旁路记录真实请求，但它不能直接改变 vLLM 的真实 KV block 行为。要从 mock manifest 进入真实 KV 复用，需要实现一个自定义 vLLM `KVConnectorBase_V1` connector，并通过 OpenAI 请求里的 `kv_transfer_params` 把 sidecar 决策传入 vLLM。

当前 vLLM 版本为 `0.19.0`，相关源码位于：

- `/opt/miniconda3/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `/opt/miniconda3/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/factory.py`
- `/opt/miniconda3/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`
- `/opt/miniconda3/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`

## 可用入口

vLLM OpenAI completions/chat 请求协议已经支持 `kv_transfer_params`。请求进入 vLLM 后，该字段会进入 `sampling_params.extra_args["kv_transfer_params"]`，再进入 `Request.kv_transfer_params`。

因此 M3.6 的最小路径是：

1. HTTP sidecar 生成 `m3_control` 决策。
2. 前置代理把必要字段翻译成 vLLM 原生 `kv_transfer_params`。
3. 自定义 connector 在 scheduler 侧读取 `request.kv_transfer_params`。
4. connector 决定 `get_num_new_matched_tokens()` 返回多少外部 KV token。
5. vLLM 分配 block 后，connector 在 `build_connector_meta()` 中把 block id 与 sidecar plan 传给 worker 侧。
6. worker 侧在 `start_load_kv()` / `wait_for_layer_load()` 中完成真正 KV 注入。

## 最小替换面

第一版自定义 connector 只需要替换 mock connector 的执行部分，sidecar 继续负责策略：

- Scheduler side:
  - `get_num_new_matched_tokens(request, num_computed_tokens)`
  - `update_state_after_alloc(request, blocks, num_external_tokens)`
  - `build_connector_meta(scheduler_output)`
  - `request_finished(request, block_ids)`
- Worker side:
  - `register_kv_caches(kv_caches)`
  - `start_load_kv(forward_context)`
  - `wait_for_layer_load(layer_name)`
  - `save_kv_layer(layer_name, kv_layer, attn_metadata)`
  - `wait_for_save()`

## 建议的 M3.6 最小实验

先不要接 SSD/3FS。先实现一个“显式元数据 + 本地文件/内存 KV”的 connector smoke：

- 对一个短 prompt 做 full prefill，并在 `request_finished()` 记录 request id、block ids、token ids 和 correctness key。
- 第二个请求通过 `kv_transfer_params` 指定同一 prefix。
- `get_num_new_matched_tokens()` 返回 block 对齐后的可复用 token 数。
- worker 侧先只做 no-op load，并记录将要注入哪些 block；通过日志验证 vLLM scheduler 的外部 token 路径被走通。
- 之后再把 no-op load 替换为真实 KV tensor copy。

## 当前 M3.6 第一版实现

已新增：

- `benchmarks/m3/noop_connector.py`
- `tests/m3/test_noop_connector.py`

第一版 connector 名称为 `M3NoOpConnector`，通过 vLLM 的动态模块加载路径使用：

```json
{
  "kv_connector": "M3NoOpConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "benchmarks.m3.noop_connector",
  "kv_load_failure_policy": "fail"
}
```

它现在只做三件事：

1. 读取 `Request.kv_transfer_params`。
2. 若 `decision=ADMIT` 且 `sync_ssd_miss_allowed=false`，将 `reuse_tokens` 向下对齐到 vLLM block size。
3. 在 `update_state_after_alloc()` / `build_connector_meta()` 中记录 request id、local block ids 和 required ranges。

worker 侧 `start_load_kv()` 是 no-op，只记录已经看到的 request id；它还没有注入真实 KV tensor。

HTTP sidecar 新增 `--inject-kv-transfer-params`。开启后，`/v1/completions` 转发到 vLLM 前会把 `m3_control` 翻译成 OpenAI 协议原生的 `kv_transfer_params` 字段，并剥离 `m3_control`。

## 启动示例

vLLM 侧：

```bash
env VLLM_PLUGINS= \
  NO_PROXY=127.0.0.1,localhost \
  vllm serve /root/models/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"M3NoOpConnector","kv_role":"kv_both","kv_connector_module_path":"benchmarks.m3.noop_connector","kv_load_failure_policy":"fail"}'
```

sidecar 侧：

```bash
python benchmarks/m3/http_sidecar_cli.py \
  --simulator-params results/qwen25_14b_calibration/simulator_params.yaml \
  --capacity-csv results/qwen25_14b_calibration/qwen25_capacity_probe.csv \
  --decision-log results/m3_6_noop_connector/decisions.jsonl \
  --upstream-base-url http://127.0.0.1:8000 \
  --inject-kv-transfer-params \
  --host 127.0.0.1 \
  --port 8010
```

## 不能跳过的约束

- connector 不能绕过 sidecar 的 correctness key 判断。
- `get_num_new_matched_tokens()` 只能返回当前已经确定可用的最大连续前缀。
- 如果 KV 未 ready，connector 应返回 `None` 或 `0`，让 scheduler 延迟或本地 prefill，不能让 decode 同步读 SSD。
- 第一版不修改 attention kernel，不修改核心 GPU block allocator。
- 任何 load 失败必须通过 connector 的 load-error 路径或结构化日志暴露，不能静默退化。

## 风险

- `KVConnectorBase_V1` 在 vLLM 源码中明确标注为实验 API，后续版本可能改变。
- worker 侧真实 KV 注入需要正确处理 attention backend 的 KV tensor layout，`ExampleConnector` 已分别处理 MLA、Triton attention 和普通 paged KV，不能简化为单一形状。
- 当前 M3.5 sidecar 的 token_count 由调用方提供；M3.6 若要让 connector 真实复用，必须使用 vLLM 实际 tokenizer 后的 token ids 与 block 对齐结果。
