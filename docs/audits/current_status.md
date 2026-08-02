# Current repository audit status

This is the sole authoritative current audit report. Historical reports under
`archive/` are immutable evidence snapshots and are not current truth.

<!-- GENERATED_STATUS_SUMMARY_START -->
## Generated authoritative summary

- Audited source SHA: `ede387e195caabf41b7da0350de15eb4b90b4417`
- Audit metadata HEAD: resolved dynamically with `git rev-parse HEAD`
- Canonical complete CPU pytest result: 321 passed, 8 skipped, 20 warnings
- Human visual review: `NOT_REVIEWED`

| Ledger status | Count |
|---|---:|
| `FIXED_AND_TESTED` | 79 |
| `IMPLEMENTED_NOT_REAL_VERIFIED` | 5 |
| `INTENTIONALLY_DEFERRED_SCIENCE` | 4 |
| `OBSOLETE_OR_DUPLICATE` | 1 |
| `PARTIAL` | 8 |
<!-- GENERATED_STATUS_SUMMARY_END -->
## Evidence boundary and CI

This focused pass began from clean commit
`9399425394eeaaf8a09c5398ad95bfd5519ed229` and used the isolated branch
`focused-post-audit-consolidation-20260802`. Corrections, focused tests, the
guarded real-pilot notebook source, and implementation documentation were
committed across source commits `91dd17143356c17c9272c6e232b9f6adc3946f3b`
and `ede387e195caabf41b7da0350de15eb4b90b4417`.

The metadata commit is intentionally identified with `git rev-parse HEAD`; a
tracked file cannot contain its own commit SHA. The validator requires the
audited source to be an ancestor and rejects every later path not explicitly
listed in the ledger. The later commit is audit/index metadata only.

The audited source is local-only and has no exact-SHA remote run.
`workflow_dispatch` accepts an explicit SHA and both workflows fetch full
history; this is runnable infrastructure, not successful current-SHA CI
evidence. `NB-003` and `AUDIT-002` therefore remain `PARTIAL`.

## Independently executable CPU evidence

- Baseline: `314 passed, 8 skipped, 20 warnings in 342.92s`.
- Focused changed-module verification: `89 passed, 1 deselected, 5 warnings`.
- Final complete audited-source result: `321 passed, 8 skipped, 20 warnings in
  320.96s`.
- Generated source consistency passed for all 18 tracked notebooks.
- All 15 default fixture notebooks passed under
  `/tmp/hypertagging-post-audit-final`.
- The first-level ambiguity diagnostic passed separately under
  `/tmp/hypertagging-first-level-final`.
- Validation overview artifacts keep both real-only notebooks `NOT_RUN` for
  this pass and visual review `NOT_REVIEWED`.

These are CPU software and fixture-mechanics results. They are not trained
physics evidence or representative throughput measurements.

## Corrected contracts

- Reconstruction checkpoint selection distinguishes teacher-forced loss,
  rollout edge F1, and rollout tree validity. Rollout tracks require executed
  rollout, and checkpoints record deterministic validation and rollout UID
  selections, objective-specific denominators, actual policy/threshold
  contracts, and selection reasons. Resume rejects semantic mismatch.
- Pretraining preserves the full-objective principal checkpoint and may save
  diagnostic topology/parent/distance/non-collapse/channel tracks only when
  their own denominators are active. The bounded pilot preflight checks shared,
  tree, and hyperbolic projection norms and staged configs remain optional
  HTCondor ablations.
- Query repulsion is mask-correct, query-order invariant, differentiable, and
  disabled by default. No fixture result enables it scientifically.
- Track-fit selection has two implemented MC-independent policies. The
  max-p-value policy remains default; canonical-pion closest-mass is an
  ablation. Policy names propagate from sidecar or dataset index into
  checkpoints.
- Node-kind adapter and PID dispatch use named vocabulary constants. KLM has a
  dedicated masked adapter and the trainer's default constraint policy now
  admits it as a reconstruction daughter.
- Channel multiplicities are compared through canonical count records rather
  than traversal or list positions; the legacy sorted list remains serialized.
- Dataset indexes report full-truth-to-reconstructable channel collision
  groups without mislabeling mechanism co-occurrence as causality.
- Batched rollout exposes optional unsynchronized host-phase instrumentation.
  Reference equivalence remains CPU-tested, but CUDA production readiness is
  not claimed.
- Notebook runs generate one consolidated JSON/Markdown/HTML overview and a
  separate visual-review checklist. Real `NOT_RUN` is never promoted to PASS.
- The active backlog is unresolved-only; the all-issue mapping is generated in
  `evidence_matrix.md`. Archive inventory, worktree state, and immutable hashes
  come from explicit `archive/metadata.yaml`, not prose regexes.

All previous truth-topology/reconstructed-kinematics invariants, future-level
invariance tests, exact daughter-summed mother p4, separate detector adapters,
and source-conflict protections continue to pass.

## Real and scientific boundaries

No real mDST preprocessing was run in this pass. The previous 50-event
charged-B pilot at ancestor `88270d0` remains historical bounded evidence; it
was not silently refreshed. The real-only pilot source now supports bounded
category maps and reports charged/mixed/continuum, multiplicity/depth, KLM,
copied/shared-source, incomplete-branch, fit-policy momentum/PID-energy, and
daughter-summed mother-mass diagnostics. Trained Level-1 pointer-logit
comparison remains explicitly `NOT_RUN` without matched policy datasets,
indexes, and a checkpoint.

No trained checkpoint, held-out physics validation, representative KLM/K_L
completeness study, multi-category capacity scan, CUDA profile, ten-million
event run, or human visual review was produced. Level encoding, radius target,
channel pooling, PID mode, track-fit policy, loss staging/weights, query
repulsion, and decoding thresholds remain ablations. Whole-set scoring and the
iterative pointer decoder remain deferred designs with no runnable decorative
config.

No HTCondor job was submitted. No production-readiness or physics-improvement
claim is made.
