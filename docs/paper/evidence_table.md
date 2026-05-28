# Evidence Table for Persistent KV Anti-Caching

This file records the local evidence used by `persistent_kv_anti_caching.md`.
It is intentionally conservative: each item states what the current prototype
does and what it does not yet prove.

## Project Claim Boundaries

| Claim | Current Evidence | Boundary |
|---|---|---|
| Cold-tier KV is not accessed synchronously during online serving. | B5 runs report `sync_ssd_miss_total=0`, `ready_barrier_all_ready=true`, and `external_load_observed=yes`. | Verified in controlled single-node prototype, not yet in production cluster. |
| PrePass can move cold restore before the online request. | M3.12/M3.15/M3.16 B5 path: `store -> packed demote -> restart vLLM -> /prepass -> restore -> reuse`. | If restore starts at request arrival, restore-inclusive latency can still exceed full prefill at 2K/8K. |
| Online TTFT benefit grows with reusable historical context length. | Synthetic packed PrePass trend: B5 online/B0 is `0.447x` at 2K, `0.295x` at 8K, and `0.161x` at 16K. | This assumes restore is hidden by lead time. |
| vLLM APC is not a persistent historical KV mechanism across restart. | B2 restart baseline is close to B0 at 2K/8K/16K. | APC hot-cache remains a strong baseline in a live process. |
| Packed cold objects reduce cold-tier file fragmentation. | `packed_v1` uses `packed_object.bin` and `packed_manifest.json` instead of per-layer cold files. | Current layout is local POSIX layer-major, not native 3FS or extent-major production layout. |
| LongMemEval-S can drive a public multi-turn workload. | Manifest generated from `longmemeval_s_cleaned`; real 2K/8K B0/B5 samples completed. | Only one sample has been used so far; no accuracy evaluation yet. |

## Key Prototype Results

### Synthetic Packed PrePass Trend

Source: `results/m3_15_2k16k_trend/trend_summary.json` and
`docs/research/m3_15_2k16k_packed_prepass_trend.md`.

| Prefix | B0 TTFT | B2 Restart TTFT | B5 Online TTFT | B5 Online / B0 | Restore Executor | Restore-Inclusive | Inclusive / B0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 673.969 ms | 689.483 ms | 301.511 ms | 0.447366 | 823.925 ms | 1151.451 ms | 1.708463 |
| 8K | 2409.141 ms | 2418.796 ms | 710.835 ms | 0.295057 | 3046.794 ms | 3797.230 ms | 1.576176 |
| 16K | 5493.655 ms | 5533.513 ms | 882.298 ms | 0.160603 | 4581.719 ms | 5519.967 ms | 1.004790 |

Validity checks:
- saved KV tokens match requested prefix length;
- reuse phase has `external_load_observed=yes`;
- reuse phase has `store_events=0`;
- reuse phase has `sync_ssd_miss_total=0`.

### LongMemEval-S Real Workload Samples

Source: `results/m3_16_longmemeval_s_b0_b5_real/summary.md`.

Same sample: `question_id=e47becba`, suffix 35 tokens, output 1 token,
model `/root/models/Qwen2.5-14B-Instruct`, RTX A6000 48GB, vLLM 0.19.0.

| Prefix | Prompt Tokens | B0 TTFT | B5 Online TTFT | Online / B0 | Restore Advance | Restore-Inclusive | Inclusive / B0 | Connector Load |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2083 | 719.556 ms | 386.062 ms | 0.536528 | 775.454 ms | 1177.106 ms | 1.635878 | 177.805 ms |
| 8192 | 8227 | 2461.202 ms | 730.217 ms | 0.296691 | 3043.337 ms | 3799.756 ms | 1.543862 | 470.813 ms |

Interpretation:
- If restore is hidden before online service, B5 lowers TTFT and the benefit
  improves from 2K to 8K.
- If restore starts only after request arrival, B5 is slower than B0 at these
  lengths because SSD-to-DRAM restore dominates.

## Current Prototype Components

| Component | Implemented Artifact |
|---|---|
| HTTP sidecar/control plane | `benchmarks/m3/http_sidecar.py`, `benchmarks/m3/control_plane.py` |
| vLLM KV connector | `benchmarks/m3/noop_connector.py` |
| Tensor store and manifests | `benchmarks/m3/tensor_store.py` |
| Cold-tier adapters | `benchmarks/m3/cold_tier.py` |
| PrePass planner | `benchmarks/m3/prepass_planner.py` |
| KV Evicted Index | `benchmarks/m3/kv_evicted_index.py` |
| Prefetch queue | `benchmarks/m3/prefetch_queue.py` |
| LongMemEval adapter | `benchmarks/m3/longmemeval_workload.py` |

## Remaining Gaps Before a Full CCF-A Submission

- Production-scale 3FS or high-performance SSD executor measurements.
- Strong baseline comparison against LMCache, Tutti, Mooncake-style
  disaggregation, KVDrive-style multi-tier management, and naive synchronous
  cold restore.
- p95/p99 latency and concurrency results, not only single-request TTFT.
- Full admitted-workload historical byte hit rate evidence.
- 32K/128K/1M scaling path with validated packed/extent layout and H2D overlap.
- Semantic correctness checks beyond token-level KV reuse, including output
  consistency and LongMemEval answer quality sanity tests.
