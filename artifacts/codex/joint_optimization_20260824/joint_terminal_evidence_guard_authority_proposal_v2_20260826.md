# Joint terminal evidence guard authority proposal v2

Date: 2026-08-26

Status: **revised proposal pending second independent review; not authority**.

This supersedes v1 (`3e12aeceba54cc8f7a2b3db520b9a8ccdfb64571`) after an independent `REVISE`. No separately tracked review artifact was present, so the companion JSON binds the exact relayed review text and provenance. All authority, execution, submission, recovery, and promotion flags remain false.

## Bound sources

- Sealed plan: `b10c65e02e36dcb5878d3aa932adc559d5676a70`, tag `ht-reconstruction-post-terminal-decision-plan-v1-20260826`, SHA `45747d32d5af2682f712c399d23f0a8f04f5f137757b93cc0664d4f5e556dfe2`.
- Authority discovery: `1a5c26eab5f7649dac76b95b7c698e1658ce0170`, tag `ht-joint-terminal-evidence-v5-authority-discovery-20260826-v1`, JSON SHA `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`.
- Current Stage-A authority source: `d7068ef529cd3c07764e545f0c83c700d783bdeb`, tag `ht-reconstruction-paired-stage-a-v2r3r2r8-final-authorization-20260826-v1`.
- q32 contract/config SHA: `7298ddde4d7a0458dfda89f11e72e038ef4ed0146929cacfbdfa860ff5a653fe` / `5f56d1645472f6d46cb7704ae326c94ef9b9de79da9348a099c0db0fa9f9a931`.
- relbias contract/config SHA: `73142dc1dbbc4d96ef9989fb25d3333dd566df18979793161e5f82576cd333b7` / `5b5cafacc5ab540b565ef2ca088d47c58e120a5dadcc1b2af34b15e00b5ef29b`.
- Expected immutable job binding: q32 primary/controller `16030125/16030127`; relbias primary/controller `16030126/16030128`.

## Corrected primary inference

Use 10,000 paired UID bootstrap resamples, seed 20260826. For endpoint `j`, observed effect `d_j`, boundary `h=0`, and bootstrap effect `d_j,b`, define centered-null `z_j,b=d_j,b-d_j+h` and raw `p_j(h)=(1+count(z_j,b>=d_j))/(B+1)`; equality is included.

Order the two raw primary p-values ascending, breaking ties lexicographically by endpoint ID. Holm adjusted values are `min(1,2*p_(1))` and `max(adjusted_p_(1),p_(2))`. Both must be at most 0.05.

Confidence intervals are explicitly **Bonferroni simultaneous 95% familywise paired-percentile intervals**, not Holm intervals. Sort 10,000 bootstrap effects, retain ties, use zero-based elements 124 and 9874, and do no interpolation. Both lower bounds must be strictly positive. Both observed deltas—rollout mean event edge F1 and teacher micro pointer F1—must also be at least 0.01.

## Corrected noninferiority inference

Teacher and secondary guards use intersection-union testing, with no Holm adjustment. Promotion requires every component alternative, so each one-sided paired-bootstrap raw p-value must be at most 0.05 and its estimate strictly inside its margin.

- Higher is better: `h=-margin`, `H0:d<=h`, `z_b=d_b-d+h`, upper-tail p-value.
- Lower is better: `h=+margin`, `H0:d>=h`, same centering, lower-tail p-value.
- Loss: compute `L` from sliced per-event sums/counts, then `d=L_relbias/L_q32-1`; require `L_q32>0`, `d<0.02`, and p at most 0.05.

Complete-target efficiency has one secondary NI test. The efficiency gate reuses it; no duplicate test or p-value is allowed.

## Mandatory event receipts

Each arm must produce a canonical, digest-bound envelope with exact contract/config/checkpoint/code/runtime hashes, optimizer step 512, presentations 32768, seed 20260826, role `validation`, ordered UID digests, model/checkpoint before/after hashes, and zero training-operation counters.

The teacher receipt has exactly 2,000 UID-keyed records. Each stores per-event object and pointer TP/FP/FN, type/cardinality correct and total, Brier sum/count, ten ECE bins with count/probability sum/target sum, and explicitly single-event-sliced total-loss sum/count. Bootstrap Brier, ECE, and loss are recomputed by summing resampled sufficient statistics; published aggregate averages are forbidden substitutes.

The rollout receipt has exactly 1,000 records, equal in order to the teacher prefix. It stores canonical truth/predicted/common edge counts, full-tree indicator, subtree counts, type/leaf/source/edit/root statistics, representable and complete-target counts, predicted and teacher P4 closed/composite counts, validity, recursive conflicts, and leaf-retention counts. Teacher P4 is defined only on these 1,000 rollout UIDs.

`truth_edge_count` is the canonical truth-edge multiset cardinality, regardless of the historical aggregate name `predicted_edge_denominator`. Its cohort sum and every listed structural denominator must be positive. The JSON gives exact empty-set conventions and all structural formulas.

Fallback is scoped to evidence selection: every UID must come from validation and must not use train fallback. The resulting validation/rollout fallback count is zero. Training `fallback_teacher_count` is outside this domain and is not required to be zero; it is not a promotion metric.

## Corrected A05 protocol

The timing receipt is contemporaneous with final step-512 validation. Synchronize CUDA, capture `time.monotonic_ns` immediately before evaluation, synchronize and capture it immediately after return, before serialization. Exactly 3,000 work units are divided by positive elapsed seconds; relbias/q32 must be at least 0.9. Reconstructed timestamps are forbidden.

Process GPU memory is measured after identical warmup by synchronizing, calling `torch.cuda.reset_peak_memory_stats`, running the complete measured validation, synchronizing, then recording `max_memory_allocated` and `max_memory_reserved`. Require `0 < allocated <= reserved <= device total` on exactly one H100 NVL.

Fifteen-second device-wide `nvidia-smi memory.used` is demoted to a required finite diagnostic because it can include other processes and miss short peaks. Unique `.batch` MaxRSS must match an integer plus optional binary K/M/G/T suffix, with no decimal or whitespace, and remain in `(0,64 GiB]`; original scientific elapsed and scheduler ElapsedRaw must remain in `(0,7200]`.

## Required evaluation-only recovery contract

The immutable Stage-A jobs did not emit the required UID events, monotonic boundaries, or reset process-memory peaks. No posthoc aggregate, train-log, scheduler-time division, inferred timestamp, device-wide telemetry, or reserialized receipt may replace them.

A separate future contract must load both accepted terminal step-512 checkpoints sequentially in one `gpu:h100nvl:1` allocation, fixed lexicographic order q32 then relbias, 8 CPUs, 64G, two hours, no requeue/restarts. The order is preregistered before receipts exist; identical warmup and exclusion of loading/warmup from timing reduce its residual bias, which second review must still accept. It uses the exact frozen validation role, ordered UIDs, formulas, runtime, seed, and each arm's treatment config. Each arm receives identical unmeasured 32-teacher/16-rollout warmup; then CUDA is synchronized and peak stats reset before the full measured cohort.

The replay is evaluation only: `model.eval`, frozen no-grad path, no added autocast, no backward, optimizer, scheduler, parameter update, checkpoint write, selection, or test/stress access. Model-state and checkpoint hashes must be identical before and after. It creates new exclusive receipts and cannot repair a failed original job.

That recovery contract, evaluator, schemas, wrappers, tests, terminal checkpoint bindings, and exact output paths do not yet exist. They require implementation and independent review. This v2 proposal does not authorize them.

## Decision boundary

Promotion is impossible until second independent review accepts or exactly amends v2, a later recovery contract is separately reviewed and authorized, both original checkpoints are terminal-valid, every receipt/schema/inference/resource guard passes, and a later promotion authority explicitly approves the result. Otherwise q32 remains the baseline.

Audit counters: zero scheduler calls, payload reads, tests, and scientific runs.
