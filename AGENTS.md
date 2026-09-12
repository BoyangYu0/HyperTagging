# HyperTagging agent guidance

## Scope and working discipline

- Work in this Git root. Historical sibling repositories are read-only
  references. Check `git status --short` before editing; tracked modifications,
  untracked files, deployment code, checkpoints, reports, and snapshots already
  present belong to the user. Build on relevant work without replacing it.
- Do not commit, push, reset, clean, delete, or overwrite unrelated files unless
  the user explicitly requests that action. Keep generated test artifacts in
  a new, task-specific directory and inspect the final diff.
- Read `README.md` and the relevant current docs/code/tests
  before changing scientific behavior. `docs/audits/current_status.md` owns
  mutable audit claims; historical audits and run receipts are evidence, not
  current implementation specifications. Do not infer physics quality from
  synthetic CPU fixtures.

## Architecture and authoritative paths

Package paths below are relative to `src/hypertagging/`; CLI paths are relative
to the repository root.

- `preprocessing/schema_v4.py` produces streamable `direct-mdst-tree-v4` event
  rows with detector inputs, retained truth targets, masks, and provenance.
  V1–v3 are compatibility inputs; v5 is experimental nested Arrow storage.
- `data/heterogeneous.py`, `data/streaming.py`, `data/dataset_index.py`, and
  `training/data_module.py` implement tensor contracts, lazy loading, masked
  normalization, indexed capacity checks, and source-safe selection/splits.
- `models/heterogeneous.py` supplies detector/composite adapters and shared
  contextual features. `models/level_autoregressive.py` performs detector PID
  inference, reconstructs PID-dependent quantities, then runs the refined
  context and mother/pointer decoder. `models/relations.py` owns physical and
  hyperbolic relation features; `models/stair_masks.py` owns level causality.
- `losses/hyperbolic_pretraining.py` implements exact-tree relation/parent,
  distance, radius, channel, variance, and covariance objectives.
  `losses/level_reconstruction.py` and `losses/set_matching.py` implement
  unordered mother targets and Hungarian matching; production requires SciPy.
- `training/pretrain_trainer.py` and `training/reconstruction_trainer.py` own
  real training. `reconstruction/constraints.py`, `pid_state.py`, and
  `level_rollout.py` share decoder policy and persistent composite construction.
- Strict offline inference starts at `reconstruction/hierarchical_inference.py`;
  `evaluation/full_decay_metrics.py` and `full_decay_runner.py` score and
  serialize its results. `scripts/evaluate_full_decay.py` is the current CLI.
  Historical `full_reconstruction.py` / `evaluate_reconstruction.py` consume
  GraFEI pairs and preserve different legacy metrics; keep their regressions.
- `deployment/` and `basf2_integration/` export and execute manifest-bound ONNX
  level graphs and construct basf2 candidates. Keep deployment imports usable
  without training-time PyTorch or basf2 where their contracts require it.

## Scientific invariants

- Never use truth PID/adjacency/parents, truth geometry, B-side/channel labels,
  completeness flags, or stored truth composites to generate, score, prune, or
  stop inference. Strict offline inference physically projects to detector FSPs
  and scrubs target-only fields. Its evaluation-only source-key map must stay
  outside the model. Current links created by reconstruction are valid state.
- Model-input daughter PID histograms and truth-supervision histograms are
  separate contracts. Missing detector values have availability masks; zero is
  a placeholder, not a measurement. Do not fabricate track/ECL/KLM features.
- Every new mother points only to eligible, pre-existing, parentless forest
  roots. Enforce one parent, unique daughter/query use, allowed mother ontology,
  charge/cardinality/physics policy, and recursive detector-source disjointness
  within each coherent hypothesis. Distinct node IDs can share an ECL/KLM or
  copied source. Competing hypotheses may reuse a source independently.
- Persistent mother p4 and charge are exact recursive daughter sums. Rebuild
  source masks, current daughter PID summaries, and composite features from
  those daughters. Never copy truth mother kinematics or add a free p4 head.
- Raw-track PID choices must respect charge. Preserve the checkpoint's separate
  decision and construction modes: `soft_decision_hard_construction` uses soft
  kinematics for neural decisions and hard PID when persisting daughters.
- Track and ECL static blocks use train-fitted normalization; KLM uses the
  fixed physical scales in its encoder. Common and
  composite blocks remain in physical units until the model-owned runtime
  normalizer; apply the same transform after PID/composite rebuilding. Runtime
  slots without training observations use identity scaling. Categorical slots
  are embeddings, and target-only composite slots remain masked.
- Reconstruction level is generation height, not edge depth. Exact retained
  parent/LCA/path geometry drives geometry targets; leaves lie farther out in
  the radius convention. Keep hyperbolic and VIC numerical kernels in FP32.

## Training, configuration, and checkpoints

- YAML precedence is defaults < YAML < explicitly supplied CLI flags. Reject
  unknown settings and invalid capacities rather than silently ignoring them.
  Architecture presets/overrides, PID vocabulary, feature specifications,
  normalizers, target/constraint policies, and rollout PID mode are contracts.
- Scientific runs use immutable source-role selection manifests plus their
  authenticated dataset index. Fit statistics on train only; preserve fixed
  validation UID cohorts. Raw event prefixes are diagnostic, and the sealed
  test role is not a tuning/normalization source. Validate policy-specific
  query/cardinality capacity before training; overflow must be accounted for.
- `complete_only`, `reconstructable_partial`, and diagnostic targets describe
  different populations. Keep their denominators and reports separate.
  `complete_only` does not mean complete reconstruction of every physical event.
- Checkpoints include model/encoder, optimizer/scheduler/scaler, RNG, curriculum
  and sampling state, normalization, architecture/features/PID, split/data-order
  identity, and validation selection. Preserve mismatch checks and atomic saves.
  Only load trusted checkpoints. Exact mid-epoch replay is supported with zero
  loader workers; do not claim exact prefetched multiworker resume.
- Encoder transfer needs measured key/shape coverage and checkpoint lineage.
  Learned confidence is usable only when its trained-state flag permits it.
  Teacher-forced and rollout best-checkpoint tracks have distinct meanings;
  primary rollout selection retains its denominator, validity, p4-closure, and
  zero-source-conflict gates. See `docs/training.md` for the complete contract.

## Reconstruction and evaluation

- Validate checkpoint lineage with
  `scripts/validate_reconstruction_checkpoint_pair.py`, then evaluate immutable
  validation/test selections with `scripts/evaluate_full_decay.py`. The strict
  offline path is CPU-only and uses `eval()` plus inference mode. Output must
  not alias checkpoints, manifests, indexes, or input shards.
- Greedy remains the default. Opt-in full-depth beam search retains multiple
  mother/type/daughter alternatives and coherent event states across levels.
  Preserve its explicit proposal/state bounds, deterministic canonical
  deduplication, source exclusivity, node-axis cap, terminal-root lane, and
  comparable cumulative scoring. Live states with different model-visible
  confidence/features are not equivalent even when topology matches.
  Width one follows greedy for valid source-exclusive semantics. Do not replace
  global state search with local top-k followed by immediate greedy collapse.
- Search is truth-free. Oracle@K metrics may inspect truth only after all
  candidate generation/ranking is complete; label them as oracle metrics and
  keep deployable top-1 and existing greedy metric keys separate. Report search
  limits, pruning, candidate ranks/scores, and numerator/denominator counts.
- Full scope contributes one eligible root unit/event; B-half scope contributes
  two units plus one event-level both-halves result. Continuum uses explicit
  top-level retained components, never invented truth hemispheres. Direct-target
  incompatibilities remain failed primary trials; contraction is diagnostic.
- Match by reconstructed source/topology before PID or p4 scoring. Aggregate
  counts before division, preserve unavailable/failed distinctions, and do not
  average per-event rates into micro metrics. Daughter-sum p4 closure is an
  implementation invariant, not physical momentum resolution.
- See `docs/full_decay_reconstruction_evaluation.md` for API/CLI beam settings,
  score semantics, report structure, and interpretation, and
  `docs/basf2_onnx_full_decay.md` for deployment-specific contracts.

## Validation and environment

Use the installed project Python environment; inspect availability before
assuming a documented host-specific path exists. `.venv` is the normal project
environment; the separately locked GPU environment is described in
`environment/gpu/README.md`. Do not mutate an existing frozen environment or
relock dependencies as an incidental fix.

Run focused CPU tests first, then the broadest practical CPU suite:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_bounded_search_cpu.py tests/test_full_depth_beam_cpu.py \
  tests/test_full_depth_beam_regressions_cpu.py \
  tests/test_hierarchical_inference_cpu.py tests/test_full_decay_metrics_cpu.py \
  tests/test_full_decay_runner_cpu.py tests/test_full_decay_cli_cpu.py \
  tests/test_recursive_source_exclusivity_cpu.py tests/test_no_truth_leakage_cpu.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python scripts/train_hyperbolic_pretrain.py --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
python scripts/train_level_reconstruction.py --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
```

Also run new feature tests and relevant ONNX contract tests when changing shared
search/deployment code. Basf2 smoke scripts require the documented CVMFS release;
ordinary CPU fixtures do not establish basf2 or physics performance. Notebook
checks use `scripts/execute_notebook_smoke_tests.py` when notebook behavior changes.

Real-size CUDA/production jobs use the tracked guarded Slurm or HTCondor
workflows. Renderers do not submit jobs. Local CUDA diagnostics have explicit
admission/watchdog or tiny-test guards; never bypass them or borrow an active
training GPU. Keep CPU workers/BLAS threads bounded and large data/checkpoints
on the configured data/project volume. Report tests actually run, failures,
external environment limitations, and any unverified scientific claims.
