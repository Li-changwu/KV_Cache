# M3.8 真实复用路径稳定化说明

## 目标

M3.8 的目标是把 M3.7 的“真实 KV 张量能保存和加载”推进到“可以稳定观测、自动提交、可做小矩阵验证”。

这个阶段仍然不是 1M tokens 终局实验。它验证的是短前缀真实复用链路是否足够可靠，能不能支撑后续主存、固态硬盘、三文件系统分层迁移实验。

## 已实现内容

### 连接器结构化日志

`benchmarks/m3/noop_connector.py` 新增可选配置：

- `m3_event_log_path`

启用后，连接器会写 JSONL 事件：

- `store_layer`：每层 KV 保存完成或失败。
- `load_request`：一次请求的 KV 加载完成或失败。

事件字段包含：

- 请求编号
- 前缀编号
- 层名
- 令牌数
- 块数量
- 状态
- 错误信息
- 耗时毫秒

连接器同时维护内存计数：

- `store_layer_ok_total`
- `store_layer_error_total`
- `load_request_ok_total`
- `load_request_error_total`
- `store_tokens_total`
- `load_tokens_total`

### 前置代理自动提交

`benchmarks/m3/http_sidecar.py` 现在会在转发上游 vLLM 响应后检查：

```json
{
  "kv_transfer_params": {
    "m3_noop_connector": {
      "manifest": {
        "prefix_id": "...",
        "token_end": 64
      }
    }
  }
}
```

如果存在该清单，前置代理会用 sidecar 自己的请求编号自动调用控制面的 commit 逻辑。这样第二轮请求可以直接通过 `prefix_candidates` 复用第一轮保存的前缀，不再需要人工 `/commit`。

注意：vLLM 响应里的内部请求编号可能是 `cmpl-*` 或连接器内部编号，控制面提交仍以 sidecar 请求编号为准。

### 小矩阵脚本

新增 `benchmarks/m3/run_reuse_smoke_matrix.py`。

默认矩阵：

- 前缀长度：`16`、`64`、`128`、`256`
- 输出长度：`1`
- 默认模型：`/root/models/Qwen2.5-14B-Instruct`
- 默认前置代理：`http://127.0.0.1:8010`

干跑示例：

```bash
python benchmarks/m3/run_reuse_smoke_matrix.py \
  --dry-run \
  --result-dir results/m3_8_reuse_matrix
```

在线运行时，可额外传入连接器事件日志：

```bash
python benchmarks/m3/run_reuse_smoke_matrix.py \
  --result-dir results/m3_8_reuse_matrix \
  --connector-event-log results/m3_8_reuse_matrix/connector_events.jsonl
```

输出文件：

- `reuse_smoke_matrix.csv`
- `report.md`

CSV 会记录请求状态、首轮/二轮请求耗时、清单大小，以及从连接器事件日志汇总出来的保存/加载耗时。

## 下一步实验

下一步应启动 Qwen2.5-14B 在线服务，并在连接器配置里同时启用：

- `m3_tensor_store_path`
- `m3_event_log_path`

然后启动前置代理并运行小矩阵。每一行都应该保留成功、失败或边界状态，不要静默丢弃。

小矩阵完成后再判断是否进入：

- 主存驻留层原型
- 本地固态硬盘迁移原型
- 三文件系统迁移接口

进入这些阶段的前提是：短前缀真实复用路径没有正确性键、布局、自动提交或加载失败方面的明显不稳定。
