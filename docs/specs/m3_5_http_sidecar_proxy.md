# M3.5 HTTP Sidecar / Front Proxy 实验说明

## 目标

M3.5 把阶段 10 的离线控制面原型包装成一个可在线调用的 HTTP sidecar。它可以放在真实 vLLM 前面，先只做 admission、prefix reuse 规划、ready barrier 判断和日志记录，不改 vLLM 内核，也不拦截真实 KV。

当前默认在线实验模型是 `/root/models/Qwen2.5-14B-Instruct`。

## 当前能力

- `POST /admit`：直接提交控制面请求，返回 admission decision 和 ready barrier。
- `POST /commit`：把某个请求执行后的 prefix range 写入 mock manifest。
- `GET /ready_barrier/{request_id}`：查询某个请求当前 required KV 是否就绪。
- `GET /metrics`：导出最小 Prometheus 风格指标。
- `POST /v1/completions`：OpenAI completions 兼容前置代理入口。
  - 请求中带 `m3_control` 时，sidecar 按该字段做控制面判断。
  - 未配置上游时，返回 `202` 和 sidecar 决策，不调用 vLLM。
  - 配置上游时，剥离 `m3_control` 字段后转发到真实 vLLM。
  - 开启 `--inject-kv-transfer-params` 时，会把 `m3_control` 翻译为 vLLM 原生 `kv_transfer_params` 字段，供 M3.6 自定义 connector 读取。

## 启动命令

先启动真实 vLLM，注意清理插件和代理环境：

```bash
env VLLM_PLUGINS= \
  NO_PROXY=127.0.0.1,localhost \
  vllm serve /root/models/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --enforce-eager \
  --gpu-memory-utilization 0.90
```

再启动 M3.5 sidecar：

```bash
python benchmarks/m3/http_sidecar_cli.py \
  --simulator-params results/qwen25_14b_calibration/simulator_params.yaml \
  --capacity-csv results/qwen25_14b_calibration/qwen25_capacity_probe.csv \
  --decision-log results/m3_5_http_sidecar/decisions.jsonl \
  --upstream-base-url http://127.0.0.1:8000 \
  --host 127.0.0.1 \
  --port 8010
```

配置检查可以先 dry-run：

```bash
python benchmarks/m3/http_sidecar_cli.py \
  --simulator-params results/qwen25_14b_calibration/simulator_params.yaml \
  --capacity-csv results/qwen25_14b_calibration/qwen25_capacity_probe.csv \
  --decision-log results/m3_5_http_sidecar/decisions.jsonl \
  --dry-run
```

## 请求示例

```bash
curl -s http://127.0.0.1:8010/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/root/models/Qwen2.5-14B-Instruct",
    "prompt": "hello",
    "max_tokens": 1,
    "m3_control": {
      "request_id": "req-0001",
      "model_id": "/root/models/Qwen2.5-14B-Instruct",
      "token_count": 128,
      "decode_sla_ms": 12000,
      "admission_window_ms": 12000,
      "correctness_key": {
        "model_fingerprint": "/root/models/Qwen2.5-14B-Instruct",
        "tokenizer_fingerprint": "qwen2.5-tokenizer",
        "rope_config": "native-32768",
        "dtype": "bf16",
        "kv_layout": "vllm-paged"
      },
      "prefix_candidates": []
    }
  }'
```

## 输出与观测

- 决策日志：`results/m3_5_http_sidecar/decisions.jsonl`
- 指标端点：`http://127.0.0.1:8010/metrics`
- 当前核心指标：
  - `admission_decision_total{decision,reason}`
  - `prefill_tokens_saved_total`
  - `sync_ssd_miss_total`

## 限制

- 当前 connector 仍是 mock manifest，不读取真实 vLLM KV block。
- `/v1/completions` 只对请求做旁路记录和转发，不改变 vLLM 的真实 prefix cache 行为。
- `token_count` 需要由调用方通过 `m3_control` 提供；未提供时只用非常粗略的空格分词估算，不能作为 benchmark 数据。
- M3.5 的目标是稳定控制面契约和观测字段，下一阶段才把 mock connector 替换为真实 vLLM KV connector 接入点。
