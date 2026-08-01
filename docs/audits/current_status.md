# Current repository audit status

This is the sole authoritative current audit report. Historical numerical
results and old worktree descriptions remain immutable under `archive/`.
Status counts and the issue-evidence matrix are generated from
`issue_ledger.yaml` into `current_backlog.md`.

## Evidence boundary

The focused pass began from clean `master` commit
`e9b828c45c9c26bf4dcd2b76e96fb788ee6d3f1b`. Source, tests, deterministic
notebook sources, workflows, configs, and implementation documentation were
corrected in the separate audited-code commit
`88270d00fb5c9fc6311daab2f9443832ebe7c3bf`.

The audit-metadata commit is intentionally identified dynamically by
`git rev-parse HEAD`; a tracked file cannot truthfully contain the SHA of the
commit that contains that same file. The validator instead proves that the
audited code SHA is an ancestor and that every later changed path is an exact
audit/index path allowed by the ledger. Any later source, model, loss,
preprocessing, training, reconstruction, runtime, config, test, workflow, or
notebook-source change invalidates this report.

The exact starting SHA has a GitHub Actions run, but it is a failure rather
than positive CI evidence: run 30698801983 passed its unit suite and notebook
smoke job, then failed audit integrity because the workflow checkout did not
contain the audited ancestor. Both workflows now fetch full history and CPU
correctness supports `workflow_dispatch` with an explicit source SHA. The
audited source commit is local-only, so no post-fix exact-SHA remote run exists.

## Independently executable CPU evidence

- At the exact audited source commit, the source suite excluding the
  deliberately stale audit-metadata integration module passed `292 passed,
  8 skipped, 19 warnings` in 397.00 seconds.
- After applying only audit/index metadata, the complete suite passed `301
  passed, 8 skipped, 19 warnings`; this is the canonical final pytest count.
- All 15 default deterministic fixture notebooks passed at the audited source
  SHA under `/tmp/hypertagging-notebook-post-audit-88270d0`.
- The first-level ambiguity diagnostic passed separately under
  `/tmp/hypertagging-first-level-post-audit-88270d0` and exercised the actual
  model ablation path.
- Generated-notebook consistency passed for all 18 tracked notebooks.
- Compile-all and `git diff --check` passed. Fixture output proves software
  mechanics only. The visual-review index remains `NOT_REVIEWED`; successful
  execution is not human visual review.

## Corrected software contracts

- For target level `t`, both contextual stages, runtime PID reconstruction,
  and the type-conditioned relation summary now consume only `S_<t`.
  Perturbing or removing every future node leaves every target-level decoder
  output invariant at levels 1 and 2; the corrected relation path retains
  finite nonzero gradients.
- Physical-relation features have explicit provenance. Exact parent and
  ancestor targets are excluded from contextual pretraining by default and
  remain loss targets. A named, checkpointed compatibility ablation can restore
  those inputs for matched HTCondor studies; FSP-only and truth-guided
  multilevel validation denominators are reported separately.
- KLM values and availability masks are collated and normalized by a dedicated
  `KlmNodeEncoder`; old shards remain readable with unavailable masks.
  Associated ECL reconstructed IDs become shared recursive source identity, so
  retained ECL/KLM representations of one neutral object conflict rather than
  being selected together.
- Track-fit selection is the versioned, MC-independent
  `max_p_value_then_pion_fallback-v1` policy. Unknown policies fail, and the
  value is serialized through parquet metadata and production manifests.
- Dataset indexes now report per-level mother/cardinality quantiles and bounded
  source, event-multiplicity, neutral-multiplicity, and channel-frequency
  slices. Training preflight rejects architectures that cannot represent an
  indexed target.
- Batched multi-event, multi-level rollout remains implemented and CPU
  reference-equivalent. It is not GPU-production-ready without the guarded
  CUDA smoke and representative profile already retained in the backlog.

## Bounded real-mDST evidence

A fresh charged-B pilot processed events 0--49 with basf2 release 08-03-00 and
the exact audited source. It wrote a new, non-production parquet output and the
real-only notebook report `/tmp/hypertagging-real-pilot-88270d0.json`.

- all 578 tracks had a selected fit and all five PIDLikelihood values;
- fit hypotheses were 158 kaon, 163 pion, and 257 proton; comparison with the
  pion closest-mass fit covered all 578 tracks, and 3,351 two-body comparisons
  had a 0.00714 GeV median and 0.0789 GeV 95th-percentile absolute mass shift;
- all 50 events had strict B roots, zero fallback roots, 100 valid B-side
  labels, and 100 active channel branches;
- 92 KLM nodes exposed all nine configured accessors; 37 ECL associations were
  observed, and all 10 associations retained on both sides had source-conflict
  protection;
- only 16 of 48 K_L-like leaves had KLM provenance, so neutral completeness
  remains partial;
- truth-derived detector inputs, p4-closure failures, cycles, missing links,
  and level failures were all zero;
- the bounded index had no query/cardinality overflow under the production
  baseline, but one charged-B file is not representative production evidence.

This is a bounded contract pilot, not production-scale or physics-performance
evidence.

## External and scientific boundaries

The generated [current backlog](current_backlog.md) is the authoritative list
of partial, deferred, open, and externally bounded items. There is no trained
held-out physics result, no normal local CUDA training, no guarded CUDA rollout
profile, no ten-million-event throughput result, no human figure review, and
no successful exact-SHA post-fix remote CI run. Radius target, channel pooling,
PID rollout mode, objective weights, structural-input compatibility, whole-set
scoring, and iterative pointer decoding remain ablations or deferred designs;
fixture results did not select a scientific default.

No HTCondor job was submitted and no production readiness or physics
improvement is claimed.
