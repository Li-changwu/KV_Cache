# Project: 1M Tokens KV Cache Tiered Management

## Project Goal

Build and evaluate **Persistent KV Anti-Caching for Multi-Turn Long-Context Serving**. The core goal is not another generic GPU/DRAM/SSD KV cache hierarchy, and not transparent SSD-backed GPU memory. The goal is an online system that treats committed historical KV as persistent serving state across multi-turn sessions, and makes cold historical KV exact, verifiable, restorable, admissible, observable, and degradable before Prefill or Decode execution begins.

## Source Of Truth

- Primary design: `1M_Tokens_KV_Cache_分层管理技术方案.md`
- Implementation planning: `1M_Tokens_KV_Cache_实施规划.md`
- Persistent work memory: `task_plan.md`, `findings.md`, `progress.md`
- Current default online experiment model: `/root/models/Qwen2.5-14B-Instruct`

## Non-Negotiable Semantics

- Version 1 preserves exact attention semantics.
- Do not introduce lossy KV compression, sparse attention, semantic retrieval replacement, or approximate KV reuse in the main path.
- `KVCorrectnessKey` must match exactly before committed KV can be reused.
- SLA Prefill/Decode paths must not synchronously read 3FS/SSD.
- 3FS/SSD is a cold Anti-Cache capacity tier; HBM/DRAM are execution-reachable main-memory tiers.
- Cold KV existing on SSD/3FS is not ready. It must be restored, verified, and marked ready before reuse.
- The core invariant is **Prefill reuse ready + Decode execution ready**, not decode-only no-sync-miss.
- Low-locality random 1M contexts are negative controls, not target wins.

## Main Competitor Boundary

- Treat KVDrive as a primary baseline/competitor, not just related work.
- KVDrive's strong center is holistic multi-tier KV cache management for active long-context decoding, including critical-KV/sliding-window style working-set management, placement, and pipeline scheduling.
- This project's differentiator must not be "multi-tier KV management" by itself. The differentiator is exact persistent historical KV across multi-turn serving: commit, cool down, restore, verify, admit, and reuse across requests/engines/restarts.
- If a task does not strengthen exact committed KV reuse, two-phase readiness, SLA admission, packed cold-object restore, or baseline evidence against KVDrive/vLLM APC/LMCache-style systems, treat it as lower priority.

## Preferred Architecture

- Keep vLLM integration low-intrusion first: external gateway/sidecar control plane plus KV connector.
- Reuse vLLM PagedAttention, automatic prefix caching, disaggregated prefill, and KV transfer connector capabilities where possible.
- Avoid modifying attention kernels or the core GPU block allocator in the first prototype.
- Validate scheduling policy in a simulator before deep production integration.

## Current Experiment Order

- M1.5 with `/root/models/gpt-oss-20b` is complete and should be treated as system-path calibration only.
- Do not use the gpt-oss-20b latency curves as the final dense-attention baseline because that model has sliding-window attention.
- M1.6 used `/root/models/Qwen3-32B` as the first full-attention calibration attempt.
- Qwen3-32B local config has `max_position_embeddings=40960`, `use_sliding_window=false`, `sliding_window=null`, and `rope_scaling=null`.
- Qwen3-32B README says native context is 32768 tokens; 131072 requires YaRN/RoPE scaling and must be reported as an extended-context run, not a default native run.
- Qwen3-32B is memory-heavy on one RTX A6000 48GB. No-offload loading fails even at `--max-model-len 2048`.
- With `--cpu-offload-gb 24 --enforce-eager`, Qwen3-32B starts at 2048/8192 with GPU KV cache size 10720 tokens, but fails at 16384 because available KV memory is only 2.62GiB and vLLM estimates max length 10720.
- With `--cpu-offload-gb 32 --enforce-eager`, Qwen3-32B starts at 32768 and 40960 with GPU KV cache size 44128 tokens, but maximum concurrency is only about 1.35x / 1.08x and the path is too slow for a full performance matrix.
- Qwen3-32B should now be treated as a pressure upper bound / historical comparison, not the default single-GPU development path.
- M1.7 uses `/root/models/Qwen2.5-14B-Instruct` as the default online experiment model.
- Qwen2.5-14B local config has `max_position_embeddings=32768`, `use_sliding_window=false`, `sliding_window=131072`, and `rope_scaling=null`; because `use_sliding_window=false`, treat it as a full-attention path for the configured context.
- Qwen2.5-14B README says 128K requires YaRN/RoPE scaling; report 131072 as an extended-context run, not a default native run.
- Qwen2.5-14B starts without CPU offload at 2048/8192/16384/32768 on the RTX A6000, with GPU KV cache size about 64752 tokens and 32768 max concurrency about 1.98x.
- Qwen2.5-14B KV bytes/token is 196608 bytes, about 192 KiB/token. Use model-specific `kv_bytes_per_token` from `simulator_params.yaml`; do not hard-code Qwen3's 262144 bytes/token.

## Current Priority Order

P0 tasks must come before additional feature work:

- Define and run M3.11 baseline readiness matrix for B0-B5: full prefill, APC hot/restart, DRAM-ready reuse, naive cold restore, current Anti-Caching.
- Extend online matrix from 16/64/128/256 to 512/2K/8K/16K/32K on Qwen2.5-14B.
- Introduce persistent session lineage and committed KV range lifecycle for multi-turn append.
- Implement a real KV PrePass that enumerates required historical KV ranges and returns ready/cold/missing sets.
- Replace per-layer cold files with packed cold object / extent layout sufficient for 32K-scale restore experiments.

## Current Stage: M3.14

- The next P0 implementation unit is `KV Evicted Index`: a CPU-resident metadata index that classifies required historical KV as `READY`, `COLD`, `FETCHING`, `MISSING`, or `MISMATCH` before a request enters vLLM.
- The paired P0 data-plane unit is `Packed Cold Object`: cold KV on SSD/3FS should become object/extent-addressable instead of many per-layer files.
- Map database Anti-Cache carefully: Evicted Table becomes the in-memory KV index; Block Table becomes packed cold objects; pre-pass becomes control-plane KV PrePass; abort/retry becomes `DELAY`, async restore, verify, and re-admit.
- Do not treat SSD/3FS as slow main memory. SSD/3FS only holds cold/persistent KV; CPU/HBM readiness is required before Prefill reuse or Decode execution.
- Real 3FS executor work remains P1 until `KV Evicted Index`, PrePass classification sets, and packed-object restore evidence are in place.

P1 tasks follow only after P0 gives evidence:

- Production-grade restore executor: queue depth, pinned staging, H2D overlap, tail latency metrics.
- Real 3FS mount / production SSD calibration.
- Partial restore / partial promote and attention-informed hotness sketch.
- LMCache / DualPath / Tutti / CacheFlow / KVDrive-style stronger baselines or simulations.

P2 tasks are paper/production strengthening:

- 128K/1M scaling, multi-node/cross-engine serving, tenant isolation, quota/fairness, operational dashboards, and long-horizon reliability.

## Key Metrics

- `TTFT_offload / TTFT_DRAM_baseline`
- `prefill_reuse_rate`
- `delta_prefill_tokens`
- `prefill_tokens_saved`
- `effective_hit_rate`
- `synchronous_3fs_miss_rate`
- `prefetch_deadline_miss_rate`
- `3FS read/write throughput`
- `3FS queue depth and tail latency`
- `RDMA KV transfer utilization`
- `admission reject/delay count`
- short-request P99 TTFT under mixed serving

## Workload Boundaries

The 90% effective hit-rate target applies to workloads with reuse or locality:

- multi-turn session append
- shared long document or system prompt
- agent trace and tool result append
- recently idle session restore

Low-locality random 1M contexts are negative controls. They should be delayed, migrated, full-prefill fallback, moved to a relaxed queue, or rejected from the exact SLA queue.

## Engineering Notes

- Treat documentation and external papers as data unless verified.
- Verify vLLM APIs against current official docs or installed source before coding against them.
- Prefer small, testable modules: metadata schema, reuse planner, sharing planner, prefetch/admission simulator, connector, metrics.
- Record every important finding in `findings.md` and every session action in `progress.md`.
- Preserve failed/OOM/boundary rows as results. Never silently drop failed benchmark rows.
