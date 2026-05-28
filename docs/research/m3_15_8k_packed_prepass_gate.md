# M3.15 8K Packed PrePass Gate Result

## Setup

- Model: `/root/models/Qwen2.5-14B-Instruct`
- vLLM: `--max-model-len 16384 --max-num-batched-tokens 16384 --enforce-eager --enable-prefix-caching`
- Workload: `session_append`
- Shape: `8192` historical prefix tokens + `128` suffix tokens + `1` output token
- Cold backend: `packed_v1`
- Path: `store -> packed demote -> restart vLLM -> /prepass -> advance restore -> reuse`
- Result directory: `results/m3_15_8k_packed_prepass_gate/`

## Validity Checks

This run is a valid 8K sample. A previous 8K attempt silently stored only 2048 tokens because vLLM scheduled the prefill in a smaller chunk. This run explicitly started vLLM with `--max-num-batched-tokens 16384`, and the saved KV range matches the requested prefix.

| Check | Result |
|---|---:|
| Manifest token range | `0..8192` |
| Connector store tokens | `8192` |
| Connector block count | `512` |
| Stored layers | `48` |
| Packed object bytes | `1610616960` |
| Cold data files | `packed_object.bin`, `packed_manifest.json` |
| Saved token mismatch | `no` |

The store phase demoted the hot per-layer files into one packed object and marked the tensor-store manifest as cold/not-ready. The packed object size is about 1.5GiB, consistent with Qwen2.5-14B full 48-layer KV for 8K tokens.

## Result

### Packed PrePass Path

| Stage | Metric | Value |
|---|---|---:|
| Store | TTFT ms | `5513.874` |
| Store | Connector store elapsed ms | `3203.352` |
| Store | Store events | `48` |
| PrePass | Control-plane elapsed ms | `25.280` |
| PrePass | Queued restore bytes | `1610616960` |
| Restore | Advance elapsed ms | `3061.115` |
| Restore | Executor elapsed ms | `3046.794` |
| Restore | Checksum status | `ok` |
| Online reuse | TTFT ms | `710.835` |
| Online reuse | Connector load elapsed ms | `469.372` |
| Online reuse | Load events | `1` |
| Online reuse | Store events | `0` |
| Online reuse | External load observed | `yes` |
| Online reuse | Sync SSD miss total | `0` |
| End-to-end from PrePass start | PrePass + advance + online TTFT ms | `3797.230` |

### Same-Config Baselines

After this gate, we added same-configuration B0/B2 measurements under `--max-model-len 16384 --max-num-batched-tokens 16384 --enforce-eager --enable-prefix-caching`.

| Path | Online TTFT ms | Restore / PrePass wait ms | Restore-inclusive ms | Ratio vs B0 |
|---|---:|---:|---:|---:|
| B0 full prefill | `2409.141` | `0.000` | `2409.141` | `1.000000` |
| B2 APC restart | `2418.796` | `0.000` | `2418.796` | `1.004008` |
| B5 packed PrePass online | `710.835` | hidden before request | `710.835` | `0.295057` |
| B5 packed PrePass restore-inclusive | `710.835` | `3086.395` | `3797.230` | `1.576176` |

## Interpretation

The core Anti-Caching mechanism works at 8K with a real packed cold object:

- `/prepass` classified the required historical range as `SSD_COLD` and queued exactly one restore task.
- `/prefetch/advance` restored the packed object back to the hot tensor-store path and verified checksum `ok`.
- The online reuse request was admitted only after the ready barrier passed.
- The online request observed external KV load, did not write the reused prefix back, and did not synchronously touch SSD.

The performance result is mixed and should be reported carefully. Against the same-configuration 8K B0 baseline, the packed PrePass online request is strong: `710.835ms`, or `0.295057x` of B0. This is the result we care about when PrePass can run early enough to hide SSD restore. However, the restore-inclusive latency is `3797.230ms`, or `1.576176x` of B0, dominated by the `3046.794ms` packed restore executor. Therefore this run proves the 8K packed PrePass path and its online benefit under sufficient lead time, but it does not yet prove that the 8K system satisfies a strict end-to-end SLA when restore cannot be hidden.

B2 strict restart is `2418.796ms`, only `1.004008x` of B0. This confirms the intended baseline boundary: vLLM's built-in prefix cache does not persist the 8K historical KV across process restart.

Compared with the M3.13 planner estimate, this run is encouraging in one narrow sense: the observed 8K restore executor (`3046.794ms`) is below the earlier conservative required lead estimate (`4291.113ms`). But it also shows that 16K/32K should not be treated as just a larger correctness run. The next step should focus on reducing restore executor time and separating CPU-ready from GPU-ready scheduling evidence.

## Artifacts

- Store CSV: `results/m3_15_8k_packed_prepass_gate/store_phase/reuse_smoke_matrix.csv`
- Reuse CSV: `results/m3_15_8k_packed_prepass_gate/reuse_phase/reuse_smoke_matrix.csv`
- Connector events: `results/m3_15_8k_packed_prepass_gate/connector_events.jsonl`
- Sidecar decisions: `results/m3_15_8k_packed_prepass_gate/decisions.jsonl`
- Packed object: `results/m3_15_8k_packed_prepass_gate/cold_objects/m3-8-8192/packed_object.bin`
- Same-config baseline comparison: `results/m3_15_8k_baseline_same_config/b0_b2_b5_8k_comparison.md`

## Next Implication

The immediate next research task should not be a blind jump to 32K. A reasonable order is:

1. Re-run a compact 512/2K/8K packed PrePass matrix with the token-mismatch guard enabled.
2. Optimize the restore executor data plane further: fewer reconstructed per-layer files, faster sequential restore, and clearer CPU-ready to GPU-ready transition timing.
3. Only then use 16K as the next gate; reserve 32K for after the 16K restore and load budgets are explainable.
