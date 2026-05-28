# M3.16 LongMemEval Workload Integration

## Goal

Use a public multi-turn long-memory dataset to replace the synthetic `cache cache ...` prompt in the current B5 packed PrePass path. The system question is not answer accuracy yet; it is whether a real multi-session history can be materialized as a reusable historical KV prefix and replayed through the same Anti-Caching control path.

## Dataset Choice

Primary target: LongMemEval cleaned, especially `longmemeval_s_cleaned.json`.

Reasons:
- It is a long-term chat assistant memory benchmark with timestamped multi-session histories.
- The official README describes LongMemEval-S as roughly 115K tokens of concatenated chat history per instance.
- The Hugging Face dataset metadata reports MIT license.
- Its shape maps naturally to our system:
  - `haystack_sessions` -> historical prefix / committed KV object.
  - `question` -> online suffix.
  - `answer` and evidence fields -> future semantic sanity checks, not required for TTFT.

## Implemented

- Added `benchmarks/m3/longmemeval_workload.py`.
- The adapter reads LongMemEval JSON records and writes:
  - `longmemeval_workload.jsonl`
  - `longmemeval_workload.csv`
  - `longmemeval_workload_summary.json`
  - `longmemeval_workload_report.md`
- Added tokenizer-aware token budgets. The default tokenizer is `/root/models/Qwen2.5-14B-Instruct`; tests can use deterministic whitespace tokenization.
- Added truncation policy: keep the most recent history tokens for the prefix, keep the start of the current question suffix.
- Added file integrity check so a partial Hugging Face download is not silently used as an experiment input.
- Added `--workload-manifest` to `benchmarks/m3/run_reuse_smoke_matrix.py`; generated LongMemEval prompts can now override the synthetic store/reuse prompts while preserving the current sidecar + connector + packed cold-tier path.

## Verified Artifacts

Fixture manifest:
- `results/m3_16_longmemeval_fixture/manifest/longmemeval_workload.jsonl`
- `results/m3_16_longmemeval_fixture/manifest/longmemeval_workload.csv`

Fixture reuse dry-run:
- `results/m3_16_longmemeval_fixture/reuse_dryrun/reuse_smoke_matrix.csv`

This proves the prompt adapter and runner interface are wired. It is not a public-data performance result.

## Current Blocker

The official Hugging Face file download was unstable in this session.

Observed:
- `longmemeval_s_cleaned.json` is about 277MB.
- `longmemeval_oracle.json` is about 15MB.
- `curl` repeatedly timed out and left a partial JSON file.
- `aria2c` failed once with TLS handshake failure.
- The adapter now rejects these partial files through JSON integrity validation.

## Repro Commands

After the official file is available:

```bash
python benchmarks/m3/longmemeval_workload.py \
  --data-path data/longmemeval/longmemeval_s_cleaned.json \
  --output-dir results/m3_16_longmemeval_s_manifest \
  --max-samples 1 \
  --target-prefix-tokens 2048 8192 16384 \
  --suffix-token-budget 256 \
  --tokenizer /root/models/Qwen2.5-14B-Instruct \
  --skip-abstention
```

Then the current reuse runner can consume the generated manifest:

```bash
python benchmarks/m3/run_reuse_smoke_matrix.py \
  --result-dir results/m3_16_longmemeval_s_reuse_dryrun \
  --prefix-tokens 2048 \
  --suffix-tokens 256 \
  --output-tokens 1 \
  --workload-manifest results/m3_16_longmemeval_s_manifest/longmemeval_workload.jsonl \
  --dry-run
```

For a real B5 run, use the same vLLM + sidecar lifecycle as M3.15, adding `--workload-manifest` to store and reuse phases.

## Interpretation Boundary

The synthetic 2K/8K/16K trend remains useful for data-plane scaling because it controls token length exactly. LongMemEval adds workload credibility: repeated multi-session histories, realistic turn formatting, and evidence-bearing sessions. We should report both:
- synthetic microbenchmark: controlled token/restore scaling;
- LongMemEval workload: public multi-turn history realism.
