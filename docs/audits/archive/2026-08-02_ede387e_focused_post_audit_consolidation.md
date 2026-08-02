# Focused post-audit correction and consolidation

Date: 2026-08-02

Starting source: `9399425394eeaaf8a09c5398ad95bfd5519ed229`

Audited source: `ede387e195caabf41b7da0350de15eb4b90b4417`

Starting worktree: clean

Report worktree: audit-metadata-only changes after the committed source
boundary

This is the single historical report for this pass. It is an immutable
evidence snapshot after metadata finalization; `../current_status.md`
supersedes it as current truth.

## Baseline evidence

- The documented CPU interpreter completed `314 passed, 8 skipped, 20
  warnings in 342.92s`.
- Audit integrity passed for 15 archived reports and 97 ledger items.
- Generated-source consistency passed for all 18 registered notebooks.
- All 15 default deterministic fixture notebooks passed, and the first-level
  ambiguity diagnostic passed separately.
- Real mDST preprocessing, a trained checkpoint, CUDA, HTCondor submission,
  production throughput, remote exact-SHA CI, and human visual review were not
  run.

## Corrections

- The default reconstruction constraint policy now admits KLM clusters by the
  named node-kind vocabulary, matching the rollout contract.
- Pretraining diagnostic checkpoint tracks now require their own active
  denominators. The topology track is the LCA/tree-relation objective rather
  than the full principal loss, and empty parent/channel evidence cannot write
  misleading best checkpoints.
- Reconstruction checkpoints record the exact rollout validation UID subset
  and the actual object, pointer, confidence, and type thresholds. Track-fit
  policy metadata is retained even when a trainer scans sidecars without a
  prebuilt dataset index.
- Pilot objective preflight reports and validates shared-encoder, tree-
  projection, and hyperbolic-projection norms, including non-finite gradients,
  while avoiding false zero-gradient failures for head-local objectives.
- Channel daughter multiplicity similarity now uses canonical multiplicity
  counts rather than sorted-list positions; the historical list remains for
  schema compatibility.
- The guarded real-mDST pilot aggregates category-aware detector/topology
  slices and MC-independent fit-policy momentum, PID-energy, unavailable-fit,
  and recursively daughter-summed composite-mass diagnostics. Trained Level-1
  pointer-logit comparison remains explicitly `NOT_RUN` until matching policy
  datasets, indexes, and a trained checkpoint exist.

## Verification evidence

Final verification results are recorded in
`../verification_runs.yaml` and the generated current status. No fixture result
is treated as physics-performance evidence.

## Unresolved scientific boundaries

No trained held-out performance, rare/unseen-channel result, calibration,
representative multi-category KLM/capacity study, CUDA profile, ten-million-
event throughput run, exact-current-SHA remote CI, or human visual review was
produced. Scientific weights, pooling, PID rollout, track-fit policy, level
encoding, radius target, query repulsion, and decoding thresholds remain
ablations. No production-readiness or physics-improvement claim is made.
