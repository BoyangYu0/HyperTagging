# Current repository audit status

This is the sole authoritative current audit report. Historical numerical
results and old worktree descriptions remain immutable under `archive/`.

## Evidence boundary

The audit began from clean `master` commit
`77707eb181b0ec011663f9fff5f0e0a454dc1758`. Source, tests, deterministic
notebook sources, and implementation documentation were corrected and
committed separately as audited code SHA
`554a553f465d7fe3056e1e1ba95ad71b37530816`.

The audit-metadata commit is intentionally identified dynamically by
`git rev-parse HEAD`; embedding a commit's own future SHA in a tracked file
would recreate the self-referential contract that this audit removed. The
validator proves that the audited code SHA is an ancestor and that every later
changed path is one of the exact audit/index paths in the ledger. Any later
scientific or runtime source change invalidates this report.

No exact-SHA GitHub Actions run exists for the local source or audit commits
because they were not pushed. The last observed remote run belongs to an older
commit and is not current evidence.

## Independently executable CPU evidence

- The source tree passed 279 tests with 8 skipped when the deliberately stale
  pre-metadata audit-integration file was excluded; the complete post-metadata
  result is recorded in the ledger verification summary.
- All 15 default deterministic fixture notebooks passed at the audited source
  SHA under `/tmp/hypertagging-notebook-audit-554a553`.
- The first-level ambiguity diagnostic passed separately under
  `/tmp/hypertagging-first-level-audit-554a553` and exercised the actual model
  baseline and soft type-conditioned query-node bias.
- Generated-notebook consistency passed for all 18 tracked notebooks.
- Fixture output is software-mechanics evidence only. The generated visual
  review index remains `NOT_REVIEWED`; execution is not human visual review.

## Corrected software contracts

- Release-08-03-00 track fits are selected without MC information by maximum
  reconstructed p-value from `getTrackFitResults`, with explicit fallback and
  availability metadata. PIDLikelihood uses the actual `PIDLikelihoods`
  relation and distinguishes missing relations, unavailable detectors, absent
  methods, and valid likelihood values.
- KLM clusters have an explicit reconstructed node kind and optional feature
  block. Existing v3 output remains unchanged; v4 carries the new contract.
- Strict B-root diagnostics no longer label an event with no B candidate as a
  fallback success.
- `batched_free_rollout` supports multiple events and levels, padded variable
  appends, source-conflict propagation, deterministic IDs, exact daughter-sum
  p4, and per-event stopping. Deterministic CPU fixtures match independent
  batch-size-one reference rollouts.
- Attention matrices are opt-in diagnostics (`return_attention=False` by
  default). Physical Stage-A and hyperbolic Stage-B diagnostics remain
  separate.
- Conditional collapse diagnostics include level, node kind, and B-side ranks;
  objective-gradient reporting includes the corruption, candidate-correctness,
  and hard-negative objectives when active.

## Bounded real-mDST evidence

A fresh charged-B pilot processed 50 events with basf2 release 08-03-00 and the
audited source SHA. The real-only notebook report is
`/tmp/hypertagging-real-pilot-554a553.json`.

- all 578 tracks had an available selected fit and all five charged-hypothesis
  PID log likelihoods;
- 50/50 events used strict resonance-daughter B roots, with zero fallback
  roots, 100 valid B-side labels, and 100 active channel branches;
- 92 reconstructed KLM clusters were collected; only 16 of 48 K_L-like leaves
  carried KLM provenance, so KLM/K_L completeness remains partial;
- truth-derived detector inputs, p4-closure failures, cycles, missing links,
  and level failures were all zero.

This is a bounded contract pilot, not a production-scale or physics-performance
measurement.

## External and scientific boundaries

The generated [current backlog](current_backlog.md) is the authoritative list
of partial, deferred, open, and externally bounded items and includes the full
issue-evidence matrix. In particular, there is no trained held-out physics
result, no local normal CUDA training, no guarded CUDA rollout profile, no
ten-million-event throughput result, no human figure review, and no exact-SHA
remote CI run. Radius target, channel pooling, PID rollout mode, objective
weights, whole-set scoring, and iterative pointer decoding remain ablations or
deferred designs rather than fixture-selected scientific conclusions.

No HTCondor job was submitted and no physics improvement is claimed.
