# Current repository audit status

This is the sole authoritative current audit report. Historical reports under
`archive/` are immutable evidence snapshots and are not current truth.

<!-- GENERATED_STATUS_SUMMARY_START -->
## Generated authoritative summary

- Audited source SHA: `56e0323a22195457fb69aad35925538219a95c0b`
- Audit metadata HEAD: resolved dynamically with `git rev-parse HEAD`
- Canonical complete CPU pytest result: 314 passed, 8 skipped, 20 warnings
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

The pass began from clean `master` commit
`ab9feb1da0942e7b9cd06205aa73dd65690821c3`. Implementation, tests, configs,
notebook sources, workflows, and implementation documentation were committed
separately as `56e0323a22195457fb69aad35925538219a95c0b`.

The metadata commit is intentionally identified with `git rev-parse HEAD`; a
tracked file cannot contain its own commit SHA. The validator requires the
audited source to be an ancestor and rejects every later path not explicitly
listed in the ledger. The later commit is audit/index metadata only.

The public Actions page shows successful CPU correctness run
`30703389232` for starting SHA `ab9feb1`. The `gh` client is unavailable here,
and the new local source commit has not been pushed, so no exact-`56e0323`
remote run exists. `workflow_dispatch` accepts an explicit SHA and both
workflows fetch full history; this is runnable infrastructure, not successful
current-SHA CI evidence.

## Independently executable CPU evidence

- Baseline: `301 passed, 8 skipped, 19 warnings in 445.23s`.
- Source-boundary run before metadata advance: `313 passed, 8 skipped, 20
  warnings`; the only failure was the expected stale audit boundary rejecting
  the new source commit.
- Final complete committed-tree result: `314 passed, 8 skipped, 20 warnings`.
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
  rollout, and checkpoints record deterministic validation UIDs, denominators,
  policy/threshold contracts, and selection reasons. Resume rejects semantic
  mismatch.
- Pretraining preserves the full-objective principal checkpoint and may save
  diagnostic topology/parent/distance/non-collapse/channel tracks. The bounded
  pilot config requires objective-gradient preflight and staged configs remain
  optional HTCondor ablations.
- Query repulsion is mask-correct, query-order invariant, differentiable, and
  disabled by default. No fixture result enables it scientifically.
- Track-fit selection now has two implemented MC-independent policies. The
  max-p-value policy remains default; canonical-pion closest-mass is an
  ablation. Policy names are validated and serialized.
- Node-kind adapter and PID dispatch use named vocabulary constants. KLM has a
  dedicated masked adapter and is an allowed reconstruction daughter.
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
was not silently refreshed. The real pilot notebook now supports several
explicit bounded category inputs, but remains real-only.

No trained checkpoint, held-out physics validation, representative KLM/K_L
completeness study, multi-category capacity scan, CUDA profile, ten-million
event run, or human visual review was produced. Level encoding, radius target,
channel pooling, PID mode, track-fit policy, loss staging/weights, query
repulsion, and decoding thresholds remain ablations. Whole-set scoring and the
iterative pointer decoder remain deferred designs with no runnable decorative
config.

No HTCondor job was submitted. No production-readiness or physics-improvement
claim is made.
