# M3.15 2K/8K/16K Packed PrePass Trend

## Setup

- Model: `/root/models/Qwen2.5-14B-Instruct`
- Workload: `session_append`
- Shape: historical prefix + `128` suffix tokens + `1` output token
- Cold backend: `packed_v1`
- B5 path: `store -> packed demote -> restart vLLM -> /prepass -> advance restore -> reuse`
- 2K/8K vLLM config: `--max-model-len 16384 --max-num-batched-tokens 16384 --enforce-eager --enable-prefix-caching`
- 16K vLLM config: `--max-model-len 32768 --max-num-batched-tokens 32768 --enforce-eager --enable-prefix-caching`
- Result root: `results/m3_15_2k16k_trend/`

## Validity

All three B5 samples passed the long-context save guard: the saved KV tokens match the requested prefix length. The reuse phase also observed exactly one external connector load, no reuse-side store events, and zero synchronous SSD misses.

| Prefix | Saved tokens | Connector store blocks | Packed object | Mismatch | External load | Sync SSD miss |
|---:|---:|---:|---:|---|---|---:|
| 2K | `2048` | `128` | `402657408` bytes | `no` | `yes` | `0` |
| 8K | `8192` | `512` | `1610616960` bytes | `no` | `yes` | `0` |
| 16K | `16384` | `1024` | `3221229696` bytes | `no` | `yes` | `0` |

## Trend

`B5 online` is the latency after PrePass has already restored the cold KV to CPU-ready state and the online request only performs external load plus suffix prefill. `B5 request-arrival restore-inclusive` is the pessimistic case where the request arrives before restore starts, approximated as `prepass + advance + online`.

| Prefix | B0 TTFT | B2 restart TTFT | B5 online TTFT | Online / B0 | Restore executor | Connector load | Request-arrival restore-inclusive | Inclusive / B0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | `673.969ms` | `689.483ms` | `301.511ms` | `0.447366x` | `823.925ms` | `117.915ms` | `1151.451ms` | `1.708463x` |
| 8K | `2409.141ms` | `2418.796ms` | `710.835ms` | `0.295057x` | `3046.794ms` | `469.372ms` | `3797.230ms` | `1.576176x` |
| 16K | `5493.655ms` | `5533.513ms` | `882.298ms` | `0.160603x` | `4581.719ms` | `610.991ms` | `5519.967ms` | `1.004790x` |

## Interpretation

The trend is stable in the direction we hoped to see: as context length grows, the online benefit of exact persistent KV reuse becomes stronger. B5 online improves from `0.447x` of B0 at 2K to `0.161x` at 16K. This is the research signal for PrePass: if restore is hidden by workflow lead time, persistent KV Anti-Caching can avoid a large amount of repeated prefill.

The pessimistic request-arrival path also improves with length, but for a different reason. At 2K and 8K, charging restore to the user request is still worse than full prefill. At 16K, request-arrival restore-inclusive latency is roughly equal to B0 (`1.004790x`). This does not mean the system is done; it means the break-even point is starting to appear around 16K on this local packed path.

B2 remains close to B0 across 2K/8K/16K. That supports the baseline boundary: vLLM APC does not persist the historical KV across process restart, so B2 is a cold-cache baseline, not a persistent reuse solution.

## Bottlenecks

The next bottleneck is no longer semantic correctness. The semantic path is working: `SSD_COLD -> PrePass -> restore -> ready barrier -> external load -> no sync SSD miss`.

The remaining performance bottlenecks are:

- SSD/packed object restore: `823.925ms` at 2K, `3046.794ms` at 8K, `4581.719ms` at 16K.
- CPU-ready to GPU/load path: connector load grows from `117.915ms` to `610.991ms`.
- Restore scheduling realism: current `/prefetch/advance` is still a controlled executor step, not a fully overlapped production scheduler.

## Artifacts

- Trend CSV: `results/m3_15_2k16k_trend/trend_summary.csv`
- Trend JSON: `results/m3_15_2k16k_trend/trend_summary.json`
- 2K packed PrePass: `results/m3_15_2k16k_trend/packed_prepass_2k/`
- 16K packed PrePass: `results/m3_15_2k16k_trend/packed_prepass_16k/`
- 8K packed PrePass: `results/m3_15_8k_packed_prepass_gate/`

## Next Step

Do not jump straight to 32K as a pure scale run. The best next step is to optimize and instrument the two data paths that now dominate the result: packed restore and connector load. In parallel, add a small lead-time matrix for 2K/8K/16K so the report can say how much PrePass advance notice is required to turn the online gains into SLA-safe behavior.
