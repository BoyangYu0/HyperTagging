# Focused post-audit correction report

Date: 2026-08-01

Starting source: `ab9feb1da0942e7b9cd06205aa73dd65690821c3`

Audited source: `56e0323a22195457fb69aad35925538219a95c0b`

Starting worktree: clean

Report worktree: audit-metadata-only changes after the committed source boundary

This is the single historical report for the pass. It is an evidence snapshot,
not current truth; `../current_status.md` supersedes it.

## Baseline evidence

- Python 3.11.15 from `/data/dust/user/boyangyu/uv_env/bin/python`.
- `301 passed, 8 skipped, 19 warnings in 445.23s`.
- audit integrity passed for 14 archived reports and 88 ledger issues.
- generated source consistency passed for 18 notebooks.
- all 15 default CPU fixture notebooks passed, and the first-level ambiguity
  diagnostic passed separately.
- starting SHA `ab9feb1` has successful public CPU correctness run
  `30703389232`; the new local audited source has no exact-SHA remote run.

## Corrections

- Reconstruction checkpoint selection now has independent teacher-forced,
  rollout-edge-F1, and rollout-tree-validity tracks. Rollout tracks require a
  nonzero rollout denominator, checkpoint metadata records deterministic event
  UIDs and selection semantics, and resume rejects semantic drift.
- Pretraining retains its principal checkpoint and optionally records topology,
  parent, distance, non-collapse, and channel diagnostic tracks. A bounded pilot
  configuration requires objective-gradient preflight, and four staged-loss
  configs are available without changing production defaults.
- Query repulsion ignores no-object slots and overlapping genuine hypotheses,
  is query-permutation invariant, receives gradients, and remains disabled by
  default with explicit off/weak/stronger configs.
- Track-fit selection gained the MC-independent
  `canonical_pion_closest_mass-v1` comparison policy while retaining
  `max_p_value_then_pion_fallback-v1` as default. Policies propagate through
  metadata, indexes, and checkpoint feature contracts.
- Node-kind routing no longer uses literal numeric IDs. KLM clusters retain a
  dedicated adapter and are valid reconstruction daughters.
- Dataset indexes expose full-truth to reconstructable-channel collision groups
  and explicitly label mechanism counts as co-occurrence rather than causality.
- Batched rollout can emit optional unsynchronized host-phase instrumentation;
  this is not CUDA throughput evidence.
- Notebook runs now emit consolidated JSON, Markdown, and HTML over every
  registry entry, keeping real-only unexecuted notebooks `NOT_RUN` and visual
  review `NOT_REVIEWED`. The real pilot notebook accepts bounded category maps.
- The unresolved-only backlog and all-issue evidence matrix are separate
  generated views. Archive state is explicit structured metadata with immutable
  report digests; no prose regex determines clean/dirty history.

## Source-boundary verification

- Complete source-boundary suite before advancing audit metadata: `313 passed,
  8 skipped, 20 warnings`; the sole expected failure was audit integrity
  rejecting source changes after the old audited SHA.
- After fixing the only implementation regression discovered during that run,
  the focused reference/batched rollout tests passed `2 passed`.
- All 15 default deterministic fixture notebooks passed at the audited source.
- The first-level ambiguity diagnostic passed at the audited source.
- Generated-notebook consistency passed for all 18 tracked notebooks.
- Real mDST preprocessing and the trained-physics notebook were not run in this
  pass. The earlier charged-B pilot remains ancestor evidence only.

## Unresolved scientific boundaries

No trained checkpoint, held-out physics result, representative KLM/K_L study,
multi-category capacity scan, CUDA profile, ten-million-event run, or human
visual review was produced. Track-fit choice, objective weights/staging, query
repulsion, level encoding, radius target, channel pooling, PID construction,
and decoding thresholds remain explicit ablations. No production-readiness or
physics-improvement claim is made.
