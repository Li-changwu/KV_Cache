# M3.12 2K True PrePass Result

## Setup

- Model: `/root/models/Qwen2.5-14B-Instruct`
- vLLM: `--max-model-len 8192 --enforce-eager --enable-prefix-caching`
- Workload: `session_append`
- Shape: `2048` historical prefix tokens + `128` suffix tokens + `1` output token
- Path: `store -> demote true cold object -> restart vLLM -> /prepass -> advance restore -> reuse`
- Result directory: `results/m3_12_2k_prepass_true/`

## Result

| Path | Online TTFT ms | Restore / PrePass wait ms | Restore-inclusive ms | External load | Sync cold miss |
|---|---:|---:|---:|---|---:|
| B0 full prefill | 717.061 | 0.000 | 717.061 | no | 0 |
| B3 DRAM-ready external reuse | 283.027 | 0.000 | 283.027 | yes | 0 |
| B5 reactive cold restore | 327.512 | 1035.707 | 1363.219 | yes | 0 |
| B5 PrePass-before-reuse | 271.782 | 839.601 | 1111.383 | yes | 0 |

PrePass produced `QUEUED` in async mode, not immediate `READY`. That is expected for the current sidecar configuration: `/prepass` enumerated the required historical KV range and queued the 402.7MB cold object restore in `11.070ms`; `/prefetch/advance` then completed the real restore with checksum `ok` in `839.601ms`. The final online request saw `ADMIT/required_kv_ready_before_decode`, `ready_barrier_all_ready=true`, and `sync_ssd_miss_total=0`.

## Interpretation

The experiment validates the central mechanism: cold restore can be moved before the online request. The restore cost did not disappear; it was shifted into the PrePass/advance window. Once the KV was ready, the online request TTFT was `271.782ms`, which is `0.379x` of B0 full prefill and `0.960x` of B3 DRAM-ready reuse.

If a request arrives before PrePass/advance has completed, the relevant TTFT is still restore-inclusive. In this sample, PrePass restore-inclusive latency is `1111.383ms`, or `1.550x` of B0. That is better than the reactive B5 restore-inclusive ratio of `1.901x`, but it still does not satisfy a strict `B0 + 20%` SLA if restore is charged to the request.

## Artifacts

- True B5 PrePass CSV: `results/m3_12_2k_prepass_true/b5_prepass_true/reuse_phase/reuse_smoke_matrix.csv`
- Unified import: `results/m3_12_2k_prepass_true/imported_b5_prepass/baseline_matrix.csv`
- Connector events: `results/m3_12_2k_prepass_true/b5_prepass_true/connector_events.jsonl`
- Sidecar decisions: `results/m3_12_2k_prepass_true/b5_prepass_true/decisions.jsonl`
- Prefetch queue events: `results/m3_12_2k_prepass_true/b5_prepass_true/prefetch_queue_events.jsonl`

## Next Implication

The next evidence target is no longer just "does restore work"; it is "how much lead time and object-layout optimization are needed for restore to finish before online admission across 512/2K/8K/16K/32K". This makes PrePass lead-time scheduling and packed cold object layout the immediate P0 work.
