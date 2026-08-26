# Joint terminal evidence guard authority proposal v1

Date: 2026-08-26

Status: **proposal pending independent scientific and statistical review; not authority**.

This proposal closes the specification gaps identified as A03, A04, and residual A05 in discovery commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170`. It uses only tracked q32/relbias contracts and frozen metric code. It authorizes no execution, submission, scheduler call, payload access, or scientific run.

## Frozen sources

- Sealed plan: commit `b10c65e02e36dcb5878d3aa932adc559d5676a70`, tag `ht-reconstruction-post-terminal-decision-plan-v1-20260826`, JSON SHA `45747d32d5af2682f712c399d23f0a8f04f5f137757b93cc0664d4f5e556dfe2`.
- Authority discovery: commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170`, tag `ht-joint-terminal-evidence-v5-authority-discovery-20260826-v1`, JSON SHA `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`.
- Stage-A source: commit `d7068ef529cd3c07764e545f0c83c700d783bdeb`, tag `ht-reconstruction-paired-stage-a-v2r3r2r8-final-authorization-20260826-v1`.
- q32 contract/config: file SHA `7298ddde4d7a0458dfda89f11e72e038ef4ed0146929cacfbdfa860ff5a653fe`; config SHA `5f56d1645472f6d46cb7704ae326c94ef9b9de79da9348a099c0db0fa9f9a931`.
- relbias contract/config: file SHA `73142dc1dbbc4d96ef9989fb25d3333dd566df18979793161e5f82576cd333b7`; config SHA `5b5cafacc5ab540b565ef2ca088d47c58e120a5dadcc1b2af34b15e00b5ef29b`.
- Metric implementations: reconstruction trainer `5b480a95ac87fef4427ca5ee6c212f0d48c3dc2013cfd03805a4da158177c1e0`; hierarchical metrics `6a8c3cc51907b68039816a9988284917b5e10263f7209f3e9693ee5d90080dff`; level loss/calibration `156ea8d9cfaf8cd99265fe3550a596ea3fd22387c4310a2fbfcd08d73755d892`; Stage-A runner `d0cffc0997a8faf7b1f87789b79693f08da674a869562e7022d33baf5333684a`; GPU telemetry `5755f58813ca07ada8def8921d0694842338483000c99fbf1f0ae4287ae0ea3d`.

## Evidence and statistical unit

All comparisons use the exact ordered paired validation UIDs at optimizer step 512: 2,000 fixed-next-level teacher events and 1,000 rollout events, seed 20260826. Missing, duplicate, extra, reordered, or mismatched UIDs or endpoint keys block. Ratios are recomputed from event-level sufficient statistics; aggregate-only and train-loss-only evidence cannot promote.

Use 10,000 paired event-UID bootstrap resamples with seed 20260826. The two promotion endpoints remain one Holm family: rollout `predicted_edge_f1` and teacher micro pointer F1. Each requires relbias-minus-q32 delta at least 0.01, Holm-adjusted lower confidence bound above zero, and adjusted p at most 0.05.

For noninferiority, higher-is-better endpoint `j` tests `H0: delta <= h`, where `h=-margin`. For each bootstrap delta `delta_b` and observed `delta_hat`, form `null_b=delta_b-delta_hat+h` and use `(1 + count(null_b >= delta_hat))/(B+1)`. Lower-is-better uses `h=+margin`, `H0: delta >= h`, and `(1 + count(null_b <= delta_hat))/(B+1)`. Apply Holm step-down within the declared family. The adjusted p-value must be at most 0.05 and the observed delta must lie strictly inside the noninferiority boundary.

## A03 — teacher direction and margin table

| Endpoint | Exact construction | Direction | Margin/family |
|---|---|---:|---|
| Teacher micro pointer F1 | F1 from summed pointer TP/FP/FN sufficient statistics | Higher | Improvement `>=0.01`; primary family |
| Micro pointer precision | summed TP / summed predicted positives | Higher | `0.005`; discrimination |
| Micro pointer recall | summed TP / summed truth positives | Higher | `0.005`; discrimination |
| Micro object precision | summed query TP / summed predicted-active queries | Higher | `0.005`; discrimination |
| Micro object recall | summed query TP / summed truth mothers | Higher | `0.005`; discrimination |
| Micro mother-type accuracy | summed correct types / summed matched mothers | Higher | `0.01`; discrimination |
| Micro cardinality accuracy | summed correct cardinalities / summed matched mothers | Higher | `0.01`; discrimination |
| Brier score | mean squared confidence error, event UID as resampling unit | Lower | `0.005`; calibration/loss |
| Ten-bin ECE | frozen bins `[0,.1),...,[.9,1]` | Lower | `0.005`; calibration/loss |
| Total validation loss | per-UID mean across target-level terms | Lower | relbias `<=1.02*q32`; calibration/loss |
| Teacher-forced P4 closure | mean per-UID closure rate | Higher | exact `1.0`; hard invariant |

The six discrimination endpoints form one Holm family; Brier, ECE, and total loss form a second. A one-percentage-point tolerance is used only for conditional type/cardinality denominators; direct edge/query and calibration endpoints use half a point. Total loss gets a 2% relative tolerance because its fixed weighted components can trade off while topology improves.

The frozen code emits no teacher-forced complete-target-efficiency endpoint. The proposal therefore resolves the sealed plan's teacher “edge/complete-target” family member to micro pointer F1, the available direct teacher-forced edge-selection endpoint. Predicted complete-target efficiency remains a separate mandatory secondary/efficiency guard; it is not relabeled as teacher-forced.

Available level-specific accuracy/precision/recall/F1 fields, individual loss components, and `macro_*` aliases remain mandatory finite diagnostics with identical keys and denominators, but do not add veto tests. Level strata can be sparse; macro aliases duplicate underlying observations; and requiring every weighted loss component to decrease would incorrectly prohibit scientifically useful tradeoffs.

## A04 — exact secondary guard table

The following ten endpoints form one one-sided Holm noninferiority family:

| Endpoint | Direction | Absolute margin |
|---|---:|---:|
| Predicted full-tree exact match | Higher | 0.005 |
| Predicted canonical-subtree exact match | Higher | 0.005 |
| Predicted edge precision | Higher | 0.005 |
| Predicted edge recall | Higher | 0.005 |
| Predicted mother-type accuracy | Higher | 0.010 |
| Predicted leaf-assignment accuracy | Higher | 0.010 |
| Predicted recursive-source Jaccard overlap | Higher | 0.005 |
| Predicted normalized tree-edit-like distance | Lower | 0.005 |
| Predicted root-reconstruction success | Higher | 0.010 |
| Complete-target efficiency | Higher | 0.005 |

All formulas and empty-set conventions are exactly those in hierarchical metrics SHA `6a8c3c…`; complete-target efficiency is recomputed from summed numerator/denominator, not averaged per-event ratios.

Hard exact guards remain: predicted tree validity, teacher and predicted P4 closure, reconstruction validity, leaf retention, and scheduled-rollout validity equal 1; recursive-source conflicts and fallback count equal 0; predicted edge denominator positive; all fields finite; no training fallback.

First-divergence level, node counts, depth/multiplicity strata, bounded-resolution diagnostics, and proposal-ambiguity fields remain finite schema diagnostics rather than extra tests. This avoids a brittle veto from sparse or highly correlated views.

The positive-weight preregistration introduced at commit `ee0648f972f7da469f61d5b4f547d2b4633b010a`, SHA `1f99b05afd193ca7d20361b596ab5ff831da7ae5ca35058913bf07b5e3179801`, is explicitly excluded as outcome authority: it studies a different intervention.

## A05 — exact efficiency protocol

Throughput measures only the optimizer-step-512 `validate_reconstruction` call. Start `time.monotonic_ns` immediately before entry; stop immediately after return and before logging/checkpointing. Exclude step-128/256/384 validations and all initialization; discard no events within step 512. Numerator is `validation_events + rollout_validation_events = 3000` work units. Duration must be positive, clocks and UID orders identical, and relbias/q32 throughput must be at least `0.9`.

The frozen trainer does not record those boundary timestamps. Training `events_per_second` and whole-run elapsed time are not substitutes. Until a reviewed, digest-bound timing receipt supplies them, throughput fails closed.

GPU peak memory is the maximum device-wide `nvidia-smi memory.used` over all raw 15-second samples while the trainer lifecycle runs. Require at least two valid samples and `0 < peak <= minimum reported memory.total`; missing, error, or zero blocks. It is not PyTorch allocated/reserved memory.

Host peak memory is the primary job `.batch` `MaxRSS`, parsed with binary K/M/G/T suffixes into bytes. Require `0 < MaxRSS <= 68,719,476,736` bytes (64 GiB). Empty, duplicate, malformed, zero, or non-batch records block.

Scientific walltime is `result.json elapsed_seconds`, measured by frozen `run_stage_a.py` from immediately before to immediately after `train_level_reconstruction`; require `0 < elapsed <= 7200`. Independently require primary-job `sacct ElapsedRaw` in `(0,7200]`. Thus scientific training/validation and allocation duration both respect the two-hour contract.

## Decision and review boundary

Promote relbias only if terminal evidence is complete, both primary improvements pass, every teacher and secondary noninferiority test passes, every hard invariant passes, and the entire efficiency protocol passes. Otherwise retain q32. A missing measurement is a block, not equality.

The proposal deliberately keeps optimization pressure at the sealed one-point gain, bounds direct topology/calibration regressions at half a point, allows one point on smaller conditional denominators, and permits 2% total-loss movement. Holm applies only to compact prespecified families; duplicate aliases and sparse strata do not multiply vetoes.

This proposal becomes authority only through a later, separately reviewed and sealed acceptance/amendment artifact binding its exact SHA. All authorization fields are false. Audit counters are zero scheduler calls, payload reads, tests, and scientific runs.
