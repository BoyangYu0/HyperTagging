# Joint terminal evidence guard authority proposal v3

Date: 2026-08-26

Status: **revised proposal pending third independent review; not authority**.

V3 supersedes v2 commit `c1d5de434fd1a494d329055757855e55e92d8d71` after an independent `REVISE`. The JSON binds the verbatim rereview and its provenance. All authority, execution, submission, recovery, scheduler, payload, and promotion flags are false.

## Bound authority

- Sealed plan: `b10c65e02e36dcb5878d3aa932adc559d5676a70`, JSON SHA `45747d32d5af2682f712c399d23f0a8f04f5f137757b93cc0664d4f5e556dfe2`.
- Discovery: `1a5c26eab5f7649dac76b95b7c698e1658ce0170`, JSON SHA `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`.
- Stage-A authority: `d7068ef529cd3c07764e545f0c83c700d783bdeb`, tag `ht-reconstruction-paired-stage-a-v2r3r2r8-final-authorization-20260826-v1`.
- q32 contract/config: `7298ddde4d7a0458dfda89f11e72e038ef4ed0146929cacfbdfa860ff5a653fe` / `5f56d1645472f6d46cb7704ae326c94ef9b9de79da9348a099c0db0fa9f9a931`.
- relbias contract/config: `73142dc1dbbc4d96ef9989fb25d3333dd566df18979793161e5f82576cd333b7` / `5b5cafacc5ab540b565ef2ca088d47c58e120a5dadcc1b2af34b15e00b5ef29b`.
- Jobs: q32 primary/controller `16030125/16030127`; relbias `16030126/16030128`.

## Frozen rollout aggregation

V3 restores the frozen per-UID semantics. For each rollout UID, compute edge precision, recall and F1 with the exact empty-edge conventions; subtree exactness uses `common/max(truth_nodes,1)`; mother type and leaf assignment return 1 when their conditional denominator is zero; source overlap returns 0 when no alignments exist; tree edit uses its per-UID normalized denominator. Full-tree, root success and representable-target rate are also per UID.

Every rollout guard except complete-target efficiency is the arithmetic macro mean of its 1,000 per-UID values, accumulated with `math.fsum` in exact UID order. Do not pool edge, subtree, type, leaf, source, or edit counts. Complete-target efficiency alone is pooled from correct/eligible counts.

Teacher and predicted P4 store closed/composite counts on the exact rollout 1,000. Structural denominators are positive at cohort level. Every UID must be tree-valid, reconstruction-valid, leaf-retaining, and scheduled-rollout-valid; recursive source conflicts sum to zero. Validation-source fallback must be absent, while training `fallback_teacher_count` remains outside this domain.

## Exact evidence closure

The receipt envelope now requires terminal and controller receipt hashes, result and metrics hashes, contract/config hashes, selection manifest, dataset-index file and embedded hashes, split and normalizer manifests, terminal checkpoint file and model-state hashes, checkpoint validation-UID digest, treatment flag, and `data_access=validation_only`.

Before any model loads, the future replay must materialize and hash exactly 2,000 validation events, their normalized tensors, 500 ordered teacher batches of size four, and 1,000 singleton rollout batches. The rollout list is the first 1,000 teacher UIDs. Checkpoint, selection manifest, both arms, and all four replay blocks must have exact UID equality. An ordered batch-boundary digest binds each batch ordinal and its four UIDs.

Teacher inference preserves batch size four. Each batch has one model forward. Per-event sufficient statistics and total loss are obtained only by slicing the already-produced output, matching, confidence-target and normalized tensors. A second forward, single-event normalization, or changed boundary is forbidden.

Level diagnostics were removed from authority receipts. A separately schema-bound descriptive artifact enumerates exact per-level accuracy, precision, recall, F1 and loss-component keys; it is never read by promotion inference.

## Model-state identity

`ht-model-state-digest-v1` sorts names by UTF-8 bytes and hashes a domain prefix plus, for every tensor, big-endian length-prefixed name, dtype string, dimension count, shape, payload length, and contiguous CPU logical bytes viewed as `uint8`. Sparse, meta and quantized tensors are rejected. The checkpoint model mapping and loaded state before/after every block must share this digest, and the checkpoint file hash must remain unchanged.

## Exact bootstrap reproducibility

Both cohorts use NumPy `Generator(PCG64(seed_integer))` with 10,000 matrices:

- Teacher domain `ht-guard-v3/teacher/2000/pcg64/20260826`, seed `0x4b004223af6f36bd1cb553952753a421`, matrix shape `(10000,2000)`.
- Rollout domain `ht-guard-v3/rollout/1000/pcg64/20260826`, seed `0xaf89e3bbe40d7dc29e4b3d6aaf8cc690`, matrix shape `(10000,1000)`.

Matrices are C-order little-endian `uint32`, row-major, hashed as raw bytes. Repeated indices repeat the complete UID record. Integers sum exactly; binary64 values use `math.fsum` in matrix-column order. Brier, ten-bin ECE and sliced loss are recomputed from resampled sufficient statistics. JSON numeric serialization, tail equality, Holm tie-breaking, Bonferroni order statistics 124/9874, and NI intersection-union rules remain exact in the JSON.

## Balanced four-block recovery

The required future replay order is `q32_A, relbias_A, relbias_B, q32_B`, giving each arm one outer and one inner position. Before every block: unload prior state, collect, empty CUDA cache, synchronize, load and verify only the selected checkpoint, run identical warmup, synchronize, and reset CUDA peak stats.

Warmup is two unmeasured production-shaped teacher batches of four plus eight singleton rollouts. The measured block runs all 500 teacher batches and 1,000 rollouts. Afterward it writes/fsyncs receipts, verifies state identity, unloads, clears and synchronizes.

Repeated blocks for an arm must have identical teacher-statistics, rollout-statistics, model-state, UID, event, normalization, batch-plan, and causal-input hashes. Timing, peak memory and block identity are intentionally separate. Arm throughput is the geometric mean of its two block rates; relbias/q32 must be at least 0.9. Arm process-memory peaks are the maxima across its two blocks. Fifteen-second device-wide telemetry remains diagnostic only.

The exact ceiling is one sequential H100 NVL allocation, 8 CPUs, 64 GiB, four hours, no requeue/restarts, and at most 4.0 H100 GPU-hours. Original job host MaxRSS and walltime guards remain; recovery MaxRSS is at most 64 GiB and ElapsedRaw at most 14,400 seconds.

## Remaining feasibility blocker

No tracked, independently reviewed, non-outcome evidence currently demonstrates that four complete measured blocks plus four warmups, checkpoint load/unload, clear/synchronize cycles, durable receipts and overhead fit within one H100 NVL, 8 CPUs, 64 GiB, four hours and 4.0 GPU-hours. The future evaluator, wrapper, recovery contract, schema hashes, terminal checkpoint bindings and manifests also do not yet exist.

No old aggregate, train log, inferred timestamp, scheduler-time division, 15-second telemetry, two-block replay, changed batch size, second forward or reserialized receipt can substitute. Recovery remains unauthorized until the exact feasibility evidence, implementation and contract are independently reviewed.

Audit counters: zero scheduler calls, payload reads, tests, and scientific runs.
