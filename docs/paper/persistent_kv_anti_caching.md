# Persistent KV Anti-Caching for Multi-Turn Long-Context LLM Serving

## Abstract

Long-context LLM serving has been increasingly used in enterprise RAG, code
assistant, and multi-agent workflows, where long historical contexts are reused
across many turns. However, existing prefix caching and KV offloading mechanisms
cannot effectively support such workloads when reusable KV states exceed GPU
memory: hot in-engine caches are not persistent, while transparent SSD misses
can stall prefill or decode.

To address these issues, we propose **Persistent KV Anti-Caching**, a
memory-primary KV management architecture for multi-turn long-context serving.
Unlike existing methods that mainly optimize hot-cache reuse or SSD transfer
speed, our approach actively manages historical KV as exact, persistent, and
admission-controlled state objects. It is developed with three key designs:
1) a KV Evicted Index that records each committed KV range across GPU, CPU, and
SSD states; 2) a KV PrePass that enumerates cold historical KV before a request
enters the LLM engine; and 3) a packed cold-object layout that stores cold KV as
large sequential objects instead of many per-layer files.

We implement a vLLM-based prototype with a sidecar control plane and a custom
KV connector. Preliminary experiments on Qwen2.5-14B-Instruct show that, when
PrePass restore is hidden before online serving, our method reduces online TTFT
to 0.447x, 0.295x, and 0.161x of full prefill at 2K, 8K, and 16K reusable
prefixes, respectively. On a LongMemEval-S sample, it further achieves 0.537x
and 0.297x online TTFT at 2K and 8K. These results demonstrate the feasibility
of exact persistent KV reuse without synchronous SSD misses, while also showing
that production-scale 1M-token serving and restore-inclusive SLA guarantees
remain future work.

## 1. Introduction

The serving bottleneck of long-context LLMs is not only the quadratic or
near-quadratic cost of attention kernels. In production-like workflows, a large
part of the cost is repeated prefill over text that has already been processed
before. A RAG application may attach the same organizational policy or retrieved
document to many requests. A code assistant may repeatedly reason over the same
repository files while the user asks small follow-up questions. A multi-agent
workflow may carry shared system prompts, tool descriptions, intermediate
plans, and long-lived state across many turns. In all of these cases, the
historical context is expensive to recompute, but it is also too large and too
numerous to keep entirely in GPU memory.

Modern serving systems already provide important pieces of the answer.
PagedAttention and vLLM reduce fragmentation and enable efficient in-memory KV
sharing within the serving engine [@kwon2023pagedattention]. Automatic prefix
caching can skip prefill for hot prefixes in the same cache lifetime
[@vllmPrefixCaching]. External KV systems such as LMCache extract KV from the
engine and enable reuse across requests and engines [@liu2025lmcache].
Disaggregated systems such as Mooncake make KV a first-class scheduling object
[@qin2024mooncake]. Recent SSD-backed and multi-tier systems, including Tutti
and KVDrive, directly target the data movement bottleneck between GPU memory,
host memory, and SSD [@qiu2026tutti; @lin2026kvdrive]. These systems establish
that KV is now a storage and scheduling problem, not only an attention-kernel
optimization problem.

However, multi-turn long-context serving exposes a distinct problem:
**how should a serving system manage cold historical KV as persistent state
objects while preserving exactness and protecting online latency?** A cached
prefix block in HBM is immediately useful; a KV object on SSD or 3FS is not.
If the engine discovers this fact during prefill or decode, the request either
stalls on storage I/O or falls back to recomputation. Both outcomes defeat the
purpose of using the cold tier to support long histories. Conversely, blindly
prefetching all possible histories wastes bandwidth and pollutes memory.

This paper argues that the right abstraction is not conventional caching but
**anti-caching**. In database anti-caching, main memory is the primary execution
storage, cold tuples are evicted to disk, and an in-memory evicted table records
where each tuple lives [@debrabant2013anticaching]. When a transaction touches
evicted data, the system enters a pre-pass phase to discover the needed disk
blocks, fetches them asynchronously, and restarts the transaction after the data
is available. The same principle fits persistent KV serving with one crucial
change: the LLM engine cannot speculatively execute with missing historical KV
because attention semantics would be incomplete. Therefore, the pre-pass must
be a control-plane execution before the request enters the GPU.

We propose Persistent KV Anti-Caching, a system design that adapts anti-caching
to exact KV reuse. The design uses four ideas.

First, it separates execution tiers from persistence tiers. HBM and host DRAM
are the tiers from which KV can become execution-ready; SSD and 3FS are cold
capacity tiers. A KV range in SSD is not a slow ready object. It is a
non-ready object that must be restored, verified, and admitted.

Second, it keeps an in-memory KV Evicted Index. The index records all committed
historical KV ranges with correctness keys, token ranges, layer ranges, tier
states, object identifiers, offsets, lengths, checksums, and in-flight restore
or load identifiers. This lets the sidecar classify required historical KV
without scanning storage or asking the GPU to discover misses.

Third, it introduces KV PrePass. Before an online request is admitted, PrePass
tokenizes the request, computes a correctness key, resolves session lineage or
prefix candidates, enumerates required historical ranges, and classifies them as
GPU-ready, CPU-ready, SSD-cold, fetching, loading, missing, or mismatched. The
admission rule is explicit: requests with cold required KV are delayed or
fallbacked; they are not allowed to synchronously read SSD in the execution
path. This transforms a cold miss from an opaque TTFT/ITL tail event into a
schedulable restore operation.

Fourth, it stores cold KV as packed cold objects. A long prefix for a
multi-layer transformer may otherwise become dozens or hundreds of small files.
That layout is simple but fragile at 8K, 16K, 32K, and beyond, because file
count, metadata overhead, Python copy overhead, and tail latency dominate.
Packed cold objects provide a block-table-like representation: the cold tier
stores large sequential objects, while the in-memory index stores the offset
table needed to restore only the relevant ranges.

Our current prototype is deliberately narrow. It is a vLLM-integrated research
prototype, not a production 1M-token serving system. It implements the sidecar
control plane, correctness-keyed manifests, a vLLM KV connector, true cold-tier
file migration, a packed cold-object backend, a PrePass endpoint, a ready
barrier, and a LongMemEval-S workload adapter. The most important evidence is
semantic and directional. The prototype can persist KV across vLLM restart,
demote it to a cold object, reject online execution while the object is cold,
restore it through PrePass, verify it, and then load exact external KV during a
reuse request with zero synchronous SSD misses. In controlled 2K/8K/16K
experiments, the online portion of TTFT improves as reusable context grows.
But when restore is charged to the arriving request, the system only approaches
break-even at 16K in our local setup. This limitation clarifies the central
research agenda: Persistent KV Anti-Caching is valuable when workflows provide
lead time, idle windows, or predictable next-state transitions that allow
restore to be hidden, and when the restore/load data path is engineered to
avoid cold-tier tail latency.

This paper makes the following contributions:

1. It formulates **persistent historical KV serving** as an anti-caching problem
   rather than a conventional cache-hit problem.
2. It defines a multi-tier KV state model and an in-memory KV Evicted Index that
   separates GPU-ready, CPU-ready, SSD-cold, fetching, loading, missing, and
   mismatched states.
3. It designs KV PrePass and a ready-barrier admission protocol that prevents
   synchronous cold-tier misses from entering prefill or decode.
4. It introduces packed cold objects as the cold-tier data model for exact,
   verifiable, offset-addressable KV restoration.
5. It reports preliminary vLLM prototype evidence on synthetic scaling prompts
   and LongMemEval-S, while clearly identifying the remaining gap to a
   production-scale CCF-A systems evaluation.

## 2. Related Work

### 2.1 Database Anti-Caching

Persistent KV Anti-Caching is directly inspired by DeBrabant et al.'s
anti-caching architecture for main-memory OLTP databases
[@debrabant2013anticaching]. Anti-caching reverses the traditional buffer-pool
hierarchy: main memory is the primary storage, and cold tuples are moved to disk
when memory is exhausted. The system maintains an in-memory Evicted Table that
maps evicted tuples to disk blocks, a Block Table for disk-resident blocks, and
a pre-pass phase that discovers evicted blocks needed by a transaction before
fetching them together. The core lesson for KV serving is not the particular
tuple layout, but the control invariant: cold data should be explicitly
indexed, fetched outside the critical execution path, and merged back into the
execution tier before the transaction proceeds.

KV serving differs from database transactions in two ways. First, the object is
not a tuple but a model-, tokenizer-, position-, layout-, and prompt-specific KV
state. A correctness key is therefore mandatory. Second, an LLM request cannot
execute a meaningful "trial" decode with missing historical KV. Our PrePass is
therefore a control-plane pre-pass that resolves required KV from session
lineage and prefix manifests before vLLM execution.

### 2.2 In-Engine KV Memory Management and Prefix Caching

PagedAttention and vLLM provide the foundation for efficient KV memory
management in modern LLM serving [@kwon2023pagedattention]. PagedAttention uses
paging-inspired block management to reduce KV fragmentation and support
flexible sharing of KV cache within and across requests. vLLM's automatic
prefix caching further hashes computed blocks and reuses matching prefixes in
the engine cache [@vllmPrefixCaching]. These mechanisms are highly effective
for hot prefixes within the engine's cache lifetime.

Our work targets a different point in the design space. We focus on persistent
historical KV that may survive process restart, capacity eviction, and cold-tier
demotion. In our prototype, strict restart baselines remain close to full
prefill at 2K, 8K, and 16K, while B5 external reuse can still load historical
KV after restart. This does not make APC unimportant; rather, APC is the right
hot-cache baseline, and Persistent KV Anti-Caching addresses the cold
persistent state below it.

### 2.3 External KV Reuse and Enterprise KV Cache Layers

LMCache is the closest practical external-cache system: it extracts and stores
KV caches outside GPU memory, supports vLLM and SGLang, and exposes a control
API for cache orchestration across GPU, CPU, storage, and network layers
[@liu2025lmcache]. It is especially relevant to enterprise scenarios such as
multi-round QA and document analysis, where KV reuse can reduce TTFT and GPU
cycles.

Persistent KV Anti-Caching is complementary but more specific. We do not claim
that merely externalizing KV is new. Our focus is the anti-caching admission
semantics for cold historical KV: an in-memory residency index, explicit
GPU/CPU/SSD states, a PrePass that enumerates required historical ranges before
execution, a ready barrier, and a policy that forbids synchronous cold-tier
misses on the online path. These mechanisms are intended to make external KV
reuse predictable under SLA constraints, not just possible.

CacheBlend and related RAG-oriented KV reuse systems address a different
correctness challenge: precomputed chunks are often not exact prefixes, so they
must be fused or partially recomputed to preserve quality. Our current design
chooses the exact-prefix and committed-history regime. It does not attempt to
reuse arbitrary non-prefix chunks or approximate attention; instead it asks how
to persist, schedule, and restore exact historical KV efficiently
[@yao2024cacheblend].

### 2.4 Disaggregated and KV-Centric Serving

Mooncake proposes a KVCache-centric disaggregated architecture that separates
prefill and decode clusters and uses underutilized CPU, DRAM, and SSD resources
to manage KV across the serving system. It also treats SLO-aware scheduling and
early rejection as first-class mechanisms [@qin2024mooncake]. This is a strong signal that KV
management has become a system-wide scheduling problem.

Persistent KV Anti-Caching can be viewed as a cold-state subsystem that could
live inside such a KV-centric architecture. Its distinctive mechanism is the
anti-cache transition: committed historical KV moves from execution tiers to a
persistent cold tier, remains discoverable through an in-memory index, and
returns to execution tiers through PrePass and admission. We do not replace
disaggregation; we specify how cold historical state should be represented and
admitted when it is no longer hot in the serving engine.

### 2.5 SSD-Backed and Multi-Tier KV Systems

Tutti and KVDrive are especially close to our target. Tutti focuses on making
SSD-backed KV practical by reworking the SSD-HBM data path around GPU-centric
object abstractions, GPU direct object I/O, and slack-aware scheduling
[@qiu2026tutti]. KVDrive
builds a holistic multi-tier KV cache management system spanning GPU memory,
host DRAM, and SSD, with attention-aware placement, pipeline scheduling, and
cross-tier coordination for long-context inference [@lin2026kvdrive]. These systems directly
address the fact that KV movement can dominate long-context serving latency.

Our contribution should not be stated as "we add SSD to KV cache" or "we manage
GPU/DRAM/SSD tiers"; KVDrive already makes that framing too broad. The narrower
claim is that multi-turn serving needs a **persistent anti-cache semantics** for
historical KV. KVDrive primarily motivates how to move critical KV efficiently
across tiers during long-context inference. Persistent KV Anti-Caching asks when
a cold historical KV object is allowed to become execution state again: it must
be exact, correctness-key matched, indexed, restored before admission, and
observed by a ready barrier. In this sense, Tutti and KVDrive are strong
data-plane and multi-tier baselines, while our design emphasizes the
control-plane semantics for persistent historical reuse.

### 2.6 Long-Term Memory Workloads

LongMemEval evaluates long-term interactive memory in chat assistants, with
questions spanning information extraction, multi-session reasoning, temporal
reasoning, knowledge updates, and abstention [@wu2024longmemeval]. Its multi-session histories map
well to our workload abstraction: `haystack_sessions` become committed
historical KV prefixes, and the current question becomes the online suffix.
We use LongMemEval-S as an early public workload because it captures repeated,
long-lived conversational state more realistically than synthetic repeated
tokens. Our current use is systems-level rather than accuracy-level: we measure
whether the historical prefix can be stored, demoted, restored, and reused with
real multi-turn text. Answer-quality evaluation remains future work.

## 3. Design

### 3.1 Design Goals

Persistent KV Anti-Caching is built around six goals.

**Exact reuse.** A restored KV range must be equivalent to the KV that the same
model would have produced by full prefill over the same token sequence under
the same model revision, tokenizer, position encoding, dtype, KV layout,
attention backend, and adapter configuration. The system does not use lossy KV
compression or approximate sparse attention in its first design.

**Execution-tier clarity.** GPU HBM and host DRAM are execution tiers. SSD and
3FS are cold persistence tiers. A cold-tier object may be durable and reusable,
but it is not ready for execution until restored and verified.

**No synchronous cold miss.** Online prefill and decode should not block on
unexpected SSD/3FS reads for required historical KV. Cold misses must appear as
admission decisions, restore tasks, or fallbacks.

**Multi-turn persistence.** Historical KV should survive the lifetime of a
single request and, in the target design, the lifetime of a single engine
process. This supports restarted engines, capacity pressure, cross-engine
serving, and idle session restoration.

**Schedulability.** Required cold bytes, expected restore time, H2D load time,
and queue state should be visible before admission. The system should be able
to delay, requeue, fallback, or reject low-locality requests rather than hiding
them inside tail latency.

**Workload boundary honesty.** The system is designed for high-reuse
historical contexts, not arbitrary low-locality 1M-token prompts. Low-reuse
requests should be measured as negative controls and handled through separate
queues or fallback paths.

### 3.2 Non-Goals

The first design does not modify attention kernels, change model semantics, or
claim that random 1M-token requests can be served within tight TTFT by storage
alone. It also does not claim that the current prototype already reaches 1M
tokens or production 3FS performance. The intended contribution is a systems
architecture and prototype evidence for exact, persistent, schedulable KV reuse.

### 3.3 System Overview

The system sits between the application gateway and the LLM engine. A sidecar
control plane owns the durable metadata: session lineage, prefix manifests,
correctness keys, the KV Evicted Index, cold-object manifests, restore queues,
and metrics. The vLLM connector owns controlled data movement into and out of
the engine's KV representation.

The request path has five stages:

1. **Commit.** After a request computes a reusable prefix, the connector stores
   the KV tensors and the sidecar commits a manifest with token ranges, layer
   ranges, block alignment, correctness key, checksums, and version metadata.
2. **Demotion.** When hot memory must be reclaimed, the system packs selected
   historical ranges into cold objects on SSD or 3FS, verifies checksums, and
   updates the index from ready execution state to `SSD_COLD`.
3. **PrePass.** Before a future request enters vLLM, the sidecar enumerates its
   required historical ranges and classifies each range with the in-memory
   index.
4. **Restore and admission.** If all required ranges are execution-ready, the
   request is admitted. If some ranges are cold, restore tasks are issued and
   the request is delayed or requeued. Admission occurs only after the ready
   barrier observes that required ranges are restored and correctness-checked.
5. **External load and suffix prefill.** The connector loads historical KV and
   vLLM computes only the new suffix and decode tokens.

This architecture intentionally separates policy from engine internals. The
sidecar determines whether a request is allowed to use external KV; the
connector performs the actual load/store under vLLM's scheduling lifecycle.

### 3.4 KV Object Model and Correctness Key

A KV object is a committed, versioned range of transformer KV states. The
logical unit is:

```text
KVRange = {
  tenant_id,
  session_id,
  prefix_id,
  token_start,
  token_end,
  layer_start,
  layer_end,
  block_ids,
  correctness_key,
  committed_epoch
}
```

The correctness key is part of the object identity. It includes model identity
and revision, tokenizer hash, prompt token hash, position encoding and scaling
configuration, dtype, KV layout, attention backend, block size, adapter state,
and tenant or security domain. A prefix identifier alone is not sufficient:
two objects with the same text but different tokenizer, RoPE scaling, LoRA
adapter, or KV layout are different objects and must not be reused.

The first prototype stores whole-prefix, all-layer ranges because that is
sufficient to validate the control path. The design leaves room for smaller
token extents and layer groups, which become important for partial promotion
and parallel restore.

### 3.5 KV Evicted Index

The KV Evicted Index is the in-memory metadata structure that makes
anti-caching possible. It plays the same role as an evicted table in a database:
the data itself may be cold, but the system must always know where it is and
whether it can be used.

Each entry records:

```text
KVIndexEntry = {
  prefix_id,
  token_start,
  token_end,
  correctness_key_hash,
  tier,
  ready,
  object_id,
  cold_uri,
  extents[],
  checksum,
  size_bytes,
  last_access_epoch,
  access_count,
  fetching_request_id,
  loading_request_id
}
```

The index classifies ranges into seven states:

| State | Meaning | Admission Implication |
|---|---|---|
| `GPU_READY` | KV is already in HBM. | Can enter execution directly. |
| `CPU_READY` | KV is in host DRAM and verified. | Needs CPU-to-GPU load budget or connector load. |
| `SSD_COLD` | KV is only in SSD/3FS cold tier. | Must restore before admission. |
| `FETCHING` | SSD/3FS-to-CPU restore is in progress. | Do not enqueue duplicate restore. |
| `LOADING` | CPU-to-GPU load is in progress. | Wait for load completion or load barrier. |
| `MISSING` | No reusable KV range exists. | Full prefill, fallback, or reject. |
| `MISMATCH` | Correctness key does not match. | Reuse is forbidden. |

The split between `CPU_READY` and `GPU_READY` is essential. Treating DRAM-ready
KV as execution-ready would hide H2D cost inside the online path and understate
TTFT. In the current prototype, PrePass already reports load-readiness fields
such as CPU-ready token count, expected H2D bytes, and estimated H2D load time;
a production system should close the loop by writing connector load-completion
events back into the index.

### 3.6 KV PrePass

KV PrePass is the anti-cache probe for LLM serving. It runs before the request
is submitted to the engine. Its inputs are the request tokens, session or
prefix candidates, and the desired correctness key. Its output is a reuse plan,
a restore plan, and an admission decision.

PrePass performs:

1. Tokenize the prompt and compute the correctness key.
2. Resolve session lineage or shared-prefix candidates.
3. Enumerate required historical ranges and the new suffix range.
4. Look up required ranges in the KV Evicted Index.
5. Return classification sets and byte estimates.
6. Enqueue restore for `SSD_COLD` ranges, avoid duplicate enqueue for
   `FETCHING`, and block reuse for `MISSING` or `MISMATCH`.
7. Emit `ADMIT`, `DELAY`, `FULL_PREFILL_FALLBACK`, or `REJECT`.

The key policy is simple: a request with required cold KV cannot enter the GPU
fast path. It may be delayed until restore completes; it may execute full
prefill if recomputation is cheaper or the SLA cannot tolerate waiting; or it
may be rejected from the strict queue. This is how the system prevents cold
misses from becoming hidden TTFT or decode-tail spikes.

### 3.7 Admission and Ready Barrier

The ready barrier is the enforcement point. It checks that every required range
in the reuse plan is present, verified, and loadable before decode depends on
it. In the prototype, the barrier is implemented through sidecar admission
state and connector metadata. A cold probe first returns `DELAY` with
`required_kv_not_ready_before_decode`; after `/prefetch/advance` completes
restore and checksum verification, a repeated admission returns `ADMIT`.

This barrier also provides a clean metric surface:

- `sync_ssd_miss_total` should remain zero for admitted strict-SLA requests;
- `ready_barrier_all_ready` should be true before external load;
- `external_load_observed` confirms the engine used the connector path;
- restore-inclusive TTFT and online TTFT must be reported separately.

The distinction between online TTFT and restore-inclusive TTFT is central. If a
workflow can predict that a historical object will be needed, restore can happen
before the user-visible request. If restore starts only when the request
arrives, the restore cost must be charged to the request. The system should not
hide this distinction.

### 3.8 Packed Cold Objects

Early prototypes often store one KV tensor file per layer. That is manageable
for tiny prompts, but it becomes a poor cold-tier format as context grows. A
32-layer or 48-layer model with many prefixes quickly creates many small files,
fragmented reads, repeated metadata operations, and long restore tails. The
anti-cache analogue is the difference between tuple identifiers and disk
blocks: the system needs fine-grained logical metadata but coarse-grained,
sequential physical I/O.

A packed cold object stores many layer files or extents in one object:

```text
packed_object.bin
packed_manifest.json
```

The manifest records the object id, correctness key hash, token range, layer
range, layout version, checksum, and an offset table:

```text
KVExtentRef = {
  layer_name,
  token_start,
  token_end,
  offset,
  length,
  checksum
}
```

The current `packed_v1` prototype uses a layer-major packed object because it
was the lowest-risk path to validate true cold restore. Even this first version
surfaced an important systems lesson: packing is not automatically faster.
The initial implementation reduced file count but was slower because it used
whole-object reads, byte slicing, and repeated checksum passes. After changing
demotion and restore to streaming copy paths and avoiding redundant whole-object
summaries, packed restore became slightly faster than the local per-layer path
in a 2K/8K tiny-profile benchmark. This reinforces the design direction while
also showing that a CCF-A-quality evaluation must include layout and data-path
engineering, not just metadata design.

The next design step is extent-major packing, where token extents become the
outer layout and layers are stored within each extent. That would let PrePass
restore only the ranges required by a request and would align better with
partial promotion.

### 3.9 Restore Scheduler and Multi-Tier Transitions

Persistent KV Anti-Caching uses explicit state transitions:

```text
GPU_READY -> CPU_READY -> SSD_COLD
SSD_COLD -> FETCHING -> CPU_READY
CPU_READY -> LOADING -> GPU_READY
```

Demotion moves ranges out of execution tiers when they become cold. Restore
moves them back from SSD/3FS to CPU. Load moves them from CPU to GPU or makes
them available to the connector during execution. The scheduler must account
for at least four resources: cold-tier bandwidth, host memory capacity, H2D
bandwidth, and GPU scheduling slack.

The current prototype provides a deterministic queue and an asynchronous queue
interface. It estimates bytes and restore time, prevents duplicate restore for
already fetching ranges, exposes queue metrics, and records actual restore
bytes and checksums. It does not yet implement a production-grade executor with
queue depth tuning, pinned staging buffers, H2D overlap, or 3FS-specific
semantics. These are engineering gaps, not conceptual changes.

### 3.10 vLLM Integration

The prototype integrates with vLLM through a sidecar and a custom connector.
The sidecar translates admission results into `kv_transfer_params`; the
connector receives metadata during scheduling and worker execution, stores KV
for committed prefixes, and loads external KV for reuse requests. The system
uses restart experiments to distinguish true external persistence from vLLM's
in-process prefix cache.

The connector path revealed two practical issues. First, reuse requests must be
load-only unless explicitly asked to commit a new prefix. An earlier version
accidentally wrote back full KV during reuse, inflating TTFT by seconds. Adding
a `store_policy` field fixed this and reduced 2K B3/B5 TTFT from multi-second
artifacts to hundreds of milliseconds. Second, long-context store experiments
must verify saved token counts. An 8K run initially saved only 2K tokens because
the engine's batched-token configuration exposed an incomplete prefill to the
connector. The prototype now fails rows where saved tokens do not match the
requested prefix.

These lessons are part of the system design: persistent KV reuse needs
correctness guards around connector lifecycle, block alignment, token ranges,
and store/load policies.

### 3.11 Prototype Evaluation Scope

The prototype currently supports two evaluation modes. The synthetic mode uses
controlled prefix lengths to expose scaling. The LongMemEval-S mode uses a
public multi-session history to replace repeated synthetic tokens with real
conversation text.

On synthetic 2K/8K/16K prefixes with Qwen2.5-14B-Instruct, B5 online TTFT is
0.447x, 0.295x, and 0.161x of full prefill when restore is completed before
the online request. The same runs show that restore-inclusive latency is still
1.708x and 1.576x of full prefill at 2K and 8K, and approximately breaks even
at 16K. On LongMemEval-S, one real sample shows online/B0 ratios of 0.537x at
2K and 0.297x at 8K, with restore-inclusive ratios of 1.636x and 1.544x.

These results support two claims and reject one premature claim. They support
the claim that exact persistent KV can be restored before admission and loaded
without synchronous SSD misses. They also support the directional claim that
online TTFT savings grow with reusable history length. They do not support a
claim that the current prototype already satisfies a 1M-token production SLA or
outperforms strong baselines under all accounting methods.

## 4. Discussion and Open Problems

The strongest version of this paper will require a broader evaluation. The
system must compare against vLLM APC hot and restart cases, LMCache, naive
synchronous cold restore, Tutti-style SSD data paths, Mooncake-style
disaggregated scheduling, and KVDrive-style multi-tier management. It must also
report p95/p99 latency, concurrency, bandwidth utilization, admitted-workload
historical byte hit rate, and negative-control workloads. The current prototype
is best understood as establishing the design invariant and exposing the next
performance bottlenecks: cold restore executor time and CPU-ready-to-GPU load
time.

The core research bet remains promising. KVDrive and Tutti show that KV data
movement is a first-order bottleneck; LMCache and Mooncake show that KV must be
managed outside a single GPU cache; LongMemEval-style workloads show that
long-lived histories are real. Persistent KV Anti-Caching adds the missing
state-management semantics: cold historical KV should be persistent, exact,
indexed, restored before admission, and protected by a ready barrier. That is
the system problem this project should continue to sharpen.
