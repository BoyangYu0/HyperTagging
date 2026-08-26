# Joint terminal evidence guard authority proposal v4

Date: 2026-08-26

Status: **revised proposal pending fourth independent review; not authority**.

V4 supersedes v3 commit `06744aa60e5da9748a55b5ed2385322a046fdfe9` after a third independent `REVISE`. The companion JSON binds the verbatim rereview and provenance. All authority, execution, submission, recovery, feasibility, scheduler, payload and promotion flags are false.

## Frozen teacher execution

Raw validation data is decoded once into exactly 2,000 ordered validation objects. Before any evaluator starts, the materializer creates and hashes:

- 500 teacher tensors by the frozen `collate(4) → normalize` path;
- 1,000 rollout tensors by separately collating and normalizing singletons;
- every tensor's phase, ordinal, key, dtype, shape, byte length and SHA;
- the ordered UID, target-level, tensor and batch-boundary manifests.

Each teacher batch binds its sorted unique positive target-level list. Measured evaluation performs exactly one forward for each ordered `(batch ordinal, target level)` pair, in batch then ascending-level order. Per-event counts and losses are sliced from those already-produced tensors and matches. A duplicate forward, second single-event forward, recollation or renormalization blocks.

Teacher warmup chooses the lowest batch ordinal for every distinct target-level-list signature and executes every selected signature once. Rollout warmup analogously covers every distinct `(truth depth, frozen leaf-multiplicity bucket)` signature. Selected ordinals, signatures, total forward count and digests are fixed before block one.

## Data and causal closure

The receipt now binds terminal/controller/result/metrics hashes; contracts/configs; selection, dataset-index file and embedded, split and normalizer manifests; terminal checkpoint and model-state; checkpoint validation-UID digest; treatment flag; and `data_access=validation_only`.

The access receipt requires exactly 2,000 validation logical objects decoded once, 500 teacher batches, 1,000 rollout singletons, zero raw reopens after materialization, and zero train/test/stress/other-role objects. It binds canonical logical-access and physical-read-only-open logs. Every block consumes immutable materialized tensors only.

Inputs are confined to three exact nonsymlink read-only roots. The future output root is separate, initially absent, exclusively claimed with parent fsync, and cannot alias any input by path or inode. All inputs are hashed/statted before parent page-touch and after the final block.

## Numeric receipt contract

Sigmoid, ECE comparisons, confidence subtraction/square, and sliced loss execute first on the frozen device/dtypes. Only resulting scalar contributions are cast to Python binary64. Brier and ECE iterate event slices in logical order and use `math.fsum`; loss contributions use ascending target-level order. Bootstrap repetitions repeat stored contributions and use v3's exact PCG64 matrices, ordering, bins, tails and serialization.

The JSON inlines strict rollout field types/ranges, edge-count inequalities, positive cohort denominators and frozen empty conventions. Edge, subtree, type, leaf, source, edit and other rollout guards remain per-UID macro means; only complete-target efficiency is pooled. Every UID must also have `scheduled_rollout_valid=true`.

## Four fresh-process ABBA replay

After materialization, the non-GPU parent reads and verifies every immutable input byte once before block one. It then launches four newly exec'd evaluator processes in order `q32_A, relbias_A, relbias_B, q32_B`; every process creates a fresh CUDA context.

Each process performs the identical lifecycle: verified checkpoint load, signature-covering warmup, synchronization and peak reset, complete measurement, synchronization, receipt full-write/fsync, state/input recheck, and clean exit. No interpreter, CUDA allocation or model survives between blocks. Repeated-arm statistic and causal hashes must match exactly.

Arm throughput is the geometric mean of its two block rates; arm memory is the worse block. The ceiling remains one H100 NVL, 8 CPUs, 64 GiB, four hours and 4.0 GPU-hours.

## Separate synthetic feasibility gate

Before recovery authorization, a separate future H100 gate must run four fresh processes in control ABBA order. It may use no terminal checkpoint, UID, event, label, target, endpoint, metric or outcome. Instead it exercises reviewed upper bounds for checkpoint-sized reads, materialization byte movement, maximum teacher/rollout control schedules, CUDA allocation/arithmetic, serialization, fsync, unload and synchronization.

It must complete end-to-end within 10,800 seconds, keep `.batch` MaxRSS at or below 48 GiB, and keep process CUDA reserved memory at or below 80% of device total—leaving headroom below the 14,400-second/64-GiB recovery ceiling. Failure requires a resource amendment and new review; it may never truncate scientific work or evidence.

That feasibility gate is not implemented or authorized, and its result can never count as scientific evidence.

## Disposition

V4 is pending fourth review. Terminal checkpoint bindings and the future evaluator, wrapper, schemas, recovery contract, synthetic-feasibility implementation and accepted feasibility receipt remain absent. No posthoc aggregate, raw reread, changed batching, second forward, reused process, shorter replay or feasibility output can substitute.

Audit counters: zero scheduler calls, payload reads, tests, and scientific runs.
