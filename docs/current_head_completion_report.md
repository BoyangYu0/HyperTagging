# Current-head completion report

Completion date: 2026-07-31 (Europe/Berlin)

## Repository identity and worktree

- Branch: `master`, the current default branch.
- Final `git rev-parse HEAD`:
  `f064f49985da98b69c99fb02764d854f76c12e3c`.
- Initial worktree: clean.
- Final worktree: dirty by design. The completed implementation is an
  uncommitted patch on the SHA above; that SHA alone does not contain it.
- Final short-status summary: 55 tracked files modified and 21 untracked files
  across `.github/workflows`, `README.md`, configs, current docs, generated
  notebook sources/generators, training/data/model/evaluation source, and
  focused tests. Historical audit snapshots were not modified; the only
  tracked document changed is `docs/deferred_model_ablations.md`, and the three
  requested current reports are new files.
- Final diff summary before adding this report was 55 files changed, 1535
  insertions, and 425 deletions; this report adds one untracked file.
- No commit, push, pull request, HTCondor submission, full-data preprocessing,
  long training, or GPU run was performed.

The authoritative exact status is the final `git status --short` output. Its
modified groups are the two CI workflows, README/notebook documentation, 15
stable generated notebooks plus the two real-only notebooks, relevant configs
and generators, and the source/test files named in the issue ledger below. Its
untracked files are 12 ablation configs, three current reports, the first-level
ambiguity notebook/generator, the trained-context loader, the bounded
first-level ablation module, and two focused tests.

## Exact commands and results

### Immutable starting-SHA baseline

```bash
cd /tmp/hypertagging-f064f49-baseline.KXVJRd
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

Result: `1 failed, 240 passed, 8 skipped, 18 warnings in 138.47s`. The only
failure was the all-notebook smoke test. An earlier live-tree pytest command
overlapped with edits and is intentionally excluded from baseline evidence.

### Final complete CPU suite

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

Result: `248 passed, 8 skipped, 19 warnings in 386.76s (0:06:26)`.

### Independent all-notebook execution

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py --timeout 180 \
  --keep-output /tmp/hypertagging-current-head-final-notebooks
```

Result: `Executed 15 notebooks on CPU fixtures`. The runner-derived summary at
`/tmp/hypertagging-current-head-final-notebooks/notebook_execution_summary.json`
records configured count 15, executed count 15, and `fixture_only: true`.

### Focused acceptance tests

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q \
  tests/test_current_head_acceptance_cpu.py \
  tests/test_trained_evaluation_context_cpu.py \
  tests/test_real_training_pipeline_cpu.py
```

Result: `8 passed, 4 warnings in 14.60s`. This covers geometry reuse,
precomputed/fallback and gradient equivalence, scalar-extraction guards,
source-conflict normalization, radius/FSP modes, Upsilon constraints,
evaluation normalization/splits, and exact best-state resume.

### First-level ambiguity diagnostic

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-current-head-final-ambiguity \
  --diagnostic-only --diagnostic-first-level-ambiguity
```

Result: `Executed 1 notebooks on CPU fixtures`. A preceding combined-shell
attempt passed the focused tests, then hit the environment's local-socket
sandbox restriction before kernel startup. The standalone command above
passed.

### Compile and diff

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m compileall -q src scripts tests
git diff --check
```

Result: both passed with no output.

## Fixed, partial, and open ledger

| Area | Status | Evidence |
|---|---|---|
| Exact retained-tree geometry reuse | fixed | One build per event in collators; explicit geometry/ancestor tensors and topology-safe masks in training; instrumentation and equivalence tests pass. |
| GPU-path Python traversal/synchronization | fixed | Normal relation/parent paths are tensorized and use precomputed matrices. CUDA fallbacks fail clearly; focused scalar-extraction guard passes. |
| Source-conflict normalization | fixed | Object/active-query weighting and opportunity normalization are invariant to inactive duplication and per-level query capacity. |
| Reconstruction validation/checkpointing | fixed | Periodic bounded next-level and lower-cadence rollout validation, latest/best/numbered checkpoints, early stopping, UID selection, and best/patience resume are implemented and tested. |
| Empty production validation | fixed | Fails by default. Only the named pilot override or existing explicitly diagnostic legacy/pilot modes permit train fallback, which is labeled in the checkpoint. |
| Pretraining best/channel state | fixed | Best metric/value restore; zero-positive channel windows are visible and configurable as warn/fail/ignore. |
| Trained evaluation preparation | fixed in software | All four normalizers and runtime normalization restore; schema, feature, PID, architecture, policy, compatibility, index, split, and UID assignments validate. |
| Evaluation slicing/search | fixed in notebook implementation | Local child-PID edge and mother-PID/level/B-side denominators, both B IDs/Y pair, complete/partial denominators, teacher/free/set-packing/beam/PID modes, and validation-to-test thresholds are explicit. |
| Real-only pilot notebook | fixed in implementation | No fixture fallback; expanded topology, provenance, availability, invariant, denominator, and p4 diagnostics. Real execution remains external. |
| Radius target | partial scientifically | Required ablations and gradient-cosine diagnostics exist; default remains generation height pending held-out evidence. |
| FSP channel shortcut | partial scientifically | FSP-only mask, comparison views, pair counts, and bounded-memory configs exist; benefit remains unmeasured. |
| First-level ambiguity | partial scientifically | Required diagnostics and bounded disabled ablations exist; decoder choice remains empirical. |
| Upsilon(4S) ontology | fixed | B0/B+ conjugates allowed; direct B_s conjugates rejected; unknown-state ontology stays broad; policy is serialized and resume-checked. |
| Overlap relation semantics | fixed | Momentum dot, mass, and energy are masked to disjoint recursive sources with availability bits. |
| MoE | explicitly deferred | Current model is not MoE; the requested geometric-MoE design note exists without changing the shared baseline. |
| README/audit/CI drift | fixed | v3 marked historical, v4 current; runner owns count; push/PR is bounded; weekly/manual runs all groups; summaries upload as artifacts. |

## Notebooks executed

The 15 stable fixture groups passed:

1. `inspect_leaf_pid_and_composite_inputs.ipynb`
2. `inspect_streaming_dataset.ipynb`
3. `inspect_leaf_input_pid_contract.ipynb`
4. `inspect_preprocessed_dataset.ipynb`
5. `inspect_hyperbolic_pretraining.ipynb`
6. `inspect_exact_tree_geometry_and_loss_scales.ipynb`
7. `inspect_rollout_search_and_calibration.ipynb`
8. `inspect_runtime_scaling.ipynb`
9. `inspect_query_capacity_and_losses.ipynb`
10. `inspect_training_pipeline.ipynb`
11. `inspect_level_autoregressive_reconstruction.ipynb`
12. `preprocessing_qa_report.ipynb`
13. `inspect_production_manifest.ipynb`
14. `preprocessing_four_momentum_validation.ipynb`
15. `inspect_preprocessed_parquet_and_gpt_like.ipynb`

The separate `inspect_first_level_ambiguity.ipynb` also passed. Executed
notebooks, figures, and reports are under `/tmp`. The runtime notebook covers N
= 32, 64, 100, and 160 with separate collation/geometry, first context, PID
rebuild, second context, relation bias, pointer, and one-level rollout timings.
Its report explicitly sets `throughput_claim: false`.

## Real pilot, trained checkpoint, and CI

Real mDST pilot: **NOT RUN**. `basf2` is absent and no concrete documented real
file is available. The exact generated command is:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/generic_mdst.root \
  --output /data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
  --schema-version direct-mdst-tree-v4 --entry-sequence 0:49 \
  --max-events 50 --event-buffer-size 32 --row-group-size 16
HYPERTAGGING_REAL_PILOT=/data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
  /data/dust/user/boyangyu/uv_env/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/inspect_real_mdst_pilot.ipynb
```

Trained-checkpoint physics validation: **NOT RUN**. No real data-compatible
trained checkpoint, matching dataset index, and held-out validation/test sample
were supplied. The loader's CPU contract test is not a physics result.

Current CI run ID: **none**. The final state is uncommitted and unpushed, so no
CI run exists for this patch. Historical runs are not claimed as current.

## Remaining scientific uncertainties

- Radius target choice, FSP-only channel pooling, channel memory, and the
  first-level decoder require matched real held-out experiments.
- Object, pointer, confidence, and type calibration must be selected on
  validation and applied once to an untouched test sample.
- Raw KLM/K_L provenance, contracted-intermediate frequencies, and B-root
  fallback behavior require the bounded real basf2 pilot.
- Mbc, DeltaE, and missing mass require verified frame, beam-energy, and
  channel-specific missing-particle contracts.
- Fixture timings do not establish production throughput or ten-million-event
  readiness.
- No physics efficiency, purity, calibration, rare-channel, or mass claim is
  made without real held-out data and a trained checkpoint.
