# Current-head verification

Verification date: 2026-07-31 (Europe/Berlin)

## Evidence boundary

This verification started from `master` at
`f064f49985da98b69c99fb02764d854f76c12e3c` (`small fix`), with a clean
worktree. The SHA remains the checked-out commit; the corrections described
here are an uncommitted working-tree patch and therefore are not contained in
that commit. Historical audit documents are immutable snapshots and are
indexed, not rewritten, in `docs/audit_index.md`.

The starting log was:

```text
f064f49 (HEAD -> master, origin/master, origin/HEAD) small fix
68a7ebe small fixs
d8be636 corrections
274d2df code audit
d236284 fix data production
9d37fde fix preprocessing
da8f357 update gpt like functions
cf63b90 update data production
7471c97 add new components for GPT training on generic
bbcd120 cleanup and add new data production
82ebd98 init
f7ff44b init
```

The first live-tree pytest command overlapped with edits and is not used as a
baseline. An immutable `git archive HEAD` at
`/tmp/hypertagging-f064f49-baseline.KXVJRd` was then tested from inside that
archive:

```bash
cd /tmp/hypertagging-f064f49-baseline.KXVJRd
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

Exact result: `1 failed, 240 passed, 8 skipped, 18 warnings in 138.47s`. The
only failure was
`tests/test_revised_notebooks_cpu.py::test_revised_notebooks_generate_and_execute_on_cpu_fixtures`.

## Current implementation status

- Exact retained-tree geometry is collated once per event. LCA labels, exact
  path distances, root depth, ancestor/descendant relations, and the
  topology-safe parent-negative mask are consumed explicitly by normal
  pretraining and relation-attention paths. CPU-only fallbacks reject CUDA
  inputs before Python traversal.
- Source-conflict loss is normalized over weighted query/conflict
  opportunities. Object probabilities or an explicit active-query mask remove
  inactive/padded-query scale dependence.
- Reconstruction has bounded periodic next-level validation, lower-cadence
  rollout validation, latest/best/numbered checkpoints, early stopping, and
  exact-resume restoration of best metric, patience, validation step, and
  deterministic event UID selection. Production validation never silently
  falls back to training; the fallback has an explicitly diagnostic flag.
- Pretraining restores and serializes its selected best metric and exposes or
  fails a configurable window of zero-positive channel validation batches.
- `hypertagging.evaluation.load_trained_evaluation_context` restores all four
  checkpoint normalizers, rebuilds the runtime normalizer, validates schema,
  feature, PID, architecture, policy, legacy/data-compatible, dataset-index,
  and split contracts, and proves each selected UID is assigned to validation
  or test rather than training.
- Radius target, channel pooling, level encoding, first-level decoder, and
  channel-memory ablations are explicit and disabled/default-preserving where
  required. Upsilon(4S) constraints reject direct B_s branches while the
  unknown-initial-state ontology stays broad.
- Overlapping recursive sources no longer expose momentum dot as an ordinary
  independent-particle relation; it is masked with an availability bit under
  `physical-relations-overlap-aware-v3`.
- The model remains a shared baseline, not an MoE model. A future bounded
  geometric-MoE design is documented without implementation.

## Scientific and external boundary

No HTCondor job, full preprocessing, long training, or local GPU run was made.
`basf2` is absent and no concrete documented real input is available, so the
sub-100-event real mDST pilot is `NOT RUN`; no fixture substitutes for it. No
real data-compatible trained checkpoint plus matching index was supplied, so
trained held-out physics validation is also `NOT RUN`. Fixture timings support
runtime-mechanics checks only and are not throughput claims. Mbc, DeltaE, and
missing mass remain unavailable without verified frame, beam-energy, and
channel-specific missing-particle contracts.

Final commands, exact results, worktree status, and the fixed/partial/open
ledger are recorded in `docs/current_head_completion_report.md` after final
verification.
