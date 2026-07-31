# Current-head verification report

Audit date: 2026-07-31 (Europe/Berlin)

## Scope and evidence boundary

This report verifies the working tree based on default-branch commit
`d8be6363fce7faaf7b86a1e6edd8c1175a8fac60` (`corrections`) plus the
uncommitted corrections listed below. The starting worktree was clean. The
historical audit documents remain unchanged and should be read as snapshots;
the pre-edit classification is recorded in `docs/current_head_gap_audit.md`.

The final Git SHA is still `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60`
because this audit environment did not create or push a commit. Consequently,
there cannot yet be a GitHub Actions run for the exact corrected worktree.
GitHub Actions run
[30629273834](https://github.com/BoyangYu0/HyperTagging/actions/runs/30629273834)
is successful for the exact starting SHA. After these changes are committed
and pushed, the resulting SHA must receive a successful `CPU correctness` run
and the all-notebook workflow should be invoked or observed before production
use.

No HTCondor job was submitted. No full dataset was processed, no production
parquet was overwritten, and no GPU was used. Fixture results establish
software contracts only; they do not establish physics performance,
calibration, rare-channel quality, optimal decoding, or ten-million-event
readiness.

## Starting baseline

- Initial `git status --short`: empty.
- Initial and current HEAD: `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60`.
- Baseline CPU suite: `223 passed, 8 skipped, 18 warnings in 418.20s`.
- Baseline notebooks: all 12 groups passed; artifacts retained under
  `/tmp/hypertagging-current-head-notebooks`.
- Eight skips: one absent legacy preprocessing-script check, five absent
  `graFEI_reduced/models.py` parity checks, and two absent
  `graFEI_gpt/models.py` parity checks.
- Eighteen warnings: 15 PyTorch odd-attention-head/nested-tensor warnings and
  three deliberate legacy-conflated-data diagnostic warnings.

## Corrections implemented

### Reconstruction representation and decoding

- The first-pass leaf PID head and every mother reconstruction head now consume
  `reconstruction_projection`. It is no longer an apparently meaningful dead
  projection.
- A reconstruction-loss backward test checks every parameter in the
  reconstruction projection, leaf PID head, and mother decoder for a nonzero,
  finite gradient.
- Mother queries now interact through self-attention before cross-attention.
- Duplicate daughter-set, duplicate type/daughter-set, unused-query, and
  pre/post-exclusive overlap metrics are reported.
- Confidence-ordered greedy resolution remains the production default. A
  bounded, exact weighted set-packing comparator is evaluation-only and
  default-off; a tiny test proves that it can beat greedy. No global-optimality
  claim is made for greedy resolution.

### Directed parent objective and relation contract

- Directed parent ranking now constructs an explicit topology-safe negative
  mask. It excludes the child, true parent, descendants, ancestors, immediate
  siblings, and LCA positive/near-positive relations, then prefers the other B
  branch and explicit negative LCA classes.
- A multilevel-tree test proves that a child, sibling, grandparent, and
  descendant cannot become a hard negative.
- The directed parent objective is documented separately from symmetric
  tree-distance geometry.
- `tangent_dot` is now the dot product of origin-tangent vectors obtained with
  `logmap0`, not a direct Poincare-ball coordinate dot product.
- Physical pair features use the versioned
  `physical-relations-logscale-v1` contract: bounded level/charge scaling and
  signed `log1p` scaling for mass, energy, and momentum dot products.
- The lossy summed node-kind scalar was replaced with a collision-free
  symmetric pair-kind embedding. Directed level and ordered-radius features
  intentionally make the complete relation bias asymmetric.
- Tests cover finiteness under large inputs, documented asymmetry, masking, and
  finite gradients through every enabled relation component.

### Static mother ontology and PID kinematics

- The versioned `reduced-mother-ontology-v1` mask is always applied. Unknown
  token 0 and clearly stable leaf-only reduced types cannot be emitted as
  mothers.
- Empirical level priors are secondary and configurable as `hard`, `soft`, or
  `off`. A legitimate unseen J/psi token remains reachable in soft/off modes.
- The policy and ontology version round-trip through checkpoint feature
  contracts.
- Training keeps differentiable soft expected track energy; rollout keeps a
  discrete physical mass hypothesis. Configurable modes now include soft
  expectation, temperature-annealed softmax, straight-through hard PID, hard
  PID, and canonical-pion first level.
- Diagnostics report soft-hard energy and mother-mass differences, PID entropy,
  relation-bias change, and pointer-logit change. Defaults were not changed on
  fixture evidence.

### Validation, channel semantics, capacity, and partial reconstruction

- `PretrainConfig.validate_every` and `log_every` are honored.
- Periodic validation is bounded by `validation_batches` and aggregates total
  and component losses, relation accuracy, topology-safe parent ranking,
  tree-distance, radius monotonicity, variance/covariance, effective rank,
  boundary fraction, channel retrieval, and leaf PID accuracy/entropy.
- Pretraining writes separate `latest.pt` and best-validation `best.pt`
  checkpoints, while retaining the final compatibility checkpoint.
- Channel pooling ablations are available for all-node mean, B-root, learned
  attention, and level-weighted pooling.
- All reachable commits, remote-tracking branches, tags, migration/archive
  references, string history, and unreachable Git objects were searched once
  more. No exact historical dictionary-array channel implementation was found.
  The canonical signature and structured-count representation therefore remain
  explicitly a fallback; exact legacy equivalence is not claimed.
- The production capacity report covers every indexed retained level, including
  global fallback capacity, with mother/cardinality distributions, configured
  limits, overflow counts, and margins. Any target-policy mismatch or overflow
  refuses production training.
- Explicit `complete_only` and `reconstructable_partial` policy configs keep
  metrics and denominators separate. `complete_only` is not described as
  complete reconstruction of incomplete, neutrino, K_L, or otherwise partial
  topologies.

### Notebooks, CI, and deferred designs

- `notebooks/README.md` now names all 12 groups and schema-v4 as the production
  fixture contract.
- A manual and weekly `full-notebook-smoke` workflow runs all 12 groups; the
  regular push workflow remains bounded.
- Notebook assertions now inspect semantic JSON fields for reconstruction,
  capacity, and PID contracts instead of only checking figure existence.
- `inspect_real_mdst_pilot.ipynb` is deliberately guarded against fixture
  substitution. It checks provenance, PIDLikelihood availability, fit mode,
  charge, p4 closure, B roots, PID/level distributions, and failures. `basf2`
  is unavailable here, so it was not executed. The documented command is a
  50-event pilot, below the 100-event limit.
- The trained-rollout notebook reports first divergence, missing/extra edges,
  duplicate proposals, greedy versus bounded resolution, channel/multiplicity/
  depth slices, teacher-forced versus free rollout, and soft/hard PID effects.
- The production model is explicitly not mixture-of-experts. A deferred design
  retains the shared heterogeneous encoder and non-MoE baseline while proposing
  level/topology FFN experts, a hyperbolic-prototype or learned router, and a
  load-balancing loss.
- Level information is explicitly a learned Euclidean embedding plus
  hyperbolic-radius supervision, not literal predefined hyperbolic positional
  encoding.
- Schema-v4 remains the production default. The fixture storage benchmark now
  reports size, write/full/projected throughput, JSON decode CPU, RSS, events/s,
  and nodes/s, while retaining the explicit review gate before any 10M claim.

## Final verification evidence

### CPU tests

Exact command:

```text
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

Result: `231 passed, 8 skipped, 18 warnings in 326.77s`.

The skipped tests and warning classes are unchanged from the starting baseline.
The eight newly added acceptance tests cover reconstruction gradients,
topology-safe negatives, ontology/checkpoint behavior, relation contracts,
ambiguity metrics/set packing, query self-attention, PID/pooling ablations, and
all-level capacity reporting.

### All notebook groups

Exact command:

```text
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-final-current-head-notebooks
```

Result: `Executed 12 notebooks on CPU fixtures` with all groups passing:

1. leaf/composite input contract;
2. streaming dataset;
3. leaf PID input contract;
4. preprocessed dataset;
5. hyperbolic pretraining;
6. query capacity and losses;
7. training pipeline;
8. level-autoregressive reconstruction;
9. preprocessing QA;
10. production manifest;
11. four-vector validation;
12. direct GPT-like parquet path.

Machine-readable outputs include `leaf_composite_contract.json`,
`leaf_input_leakage_check.json`, `capacity_report.json`,
`curriculum_runtime_report.json`, `scheduled_context_report.json`,
`preprocessing_qa.json`, `production_manifest_report.json`,
`streaming_report.json`, `storage_benchmark.json`, dataset indexes/manifests,
CSV diagnostic tables, JSONL training metrics, split manifests, and tiny
checkpoints. They are retained beneath
`/tmp/hypertagging-final-current-head-notebooks`.

Selected executable observations are contract evidence, not performance claims:

- target-metadata invariance: true;
- recursive input/teacher p4 contract: true;
- reconstruction projection gradient sum: `0.06886892765760422`;
- soft-hard track-energy difference: `0.3193936347961426`;
- soft-hard mother-mass difference: `0.5176765322685242`;
- schema default: `direct-mdst-tree-v4`;
- fixture capacity overflow: zero at both retained levels;
- pretraining `best.pt`, `latest.pt`, and compatibility checkpoint produced;
- greedy and bounded rollout metrics both emitted;
- fixture p4 closure rate: 1.0.

### Dry-runs and operational checks

- Hyperbolic pretraining: two CPU steps, finite loss `2.81716251373291`.
- Level reconstruction: two CPU steps, finite loss `5.582403659820557`.
- Pretraining and reconstruction submit descriptions/executables rendered to
  `/tmp/hypertagging-*-validation.{sub,sh}`; nothing was submitted.
- `bash -n` passed for every repository submit wrapper and both rendered
  executables.
- `python -m compileall -q src scripts` passed.
- `git diff --check` passed.
- `basf2` was not present; real-pilot status is therefore **not run**.

## Acceptance matrix

| Acceptance requirement | Final status and evidence |
|---|---|
| No MC-derived detector input | **Passed.** Existing schema-v4 leakage tests and leaf PID contract JSON pass. |
| Exact daughter-sum mother p4 | **Passed.** Construction/teacher/rollout tests plus four-vector and QA notebooks pass. |
| Target metadata cannot affect output | **Passed.** Direct invariance test and `target_metadata_invariance: true`. |
| Reconstruction projection gets gradients or is removed | **Passed.** It is consumed by all reconstruction heads; exhaustive intended-parameter backward test passes. |
| Topology-safe parent negatives | **Passed.** Explicit mask and multilevel family-exclusion test pass. |
| Unknown mother token rejected | **Passed.** Static ontology masks token 0 in policy and decoder; round-trip test passes. |
| Relation features defined geometrically/numerically | **Passed.** `logmap0` dot, versioned scaling, pair-kind embedding, stability/mask/gradient tests pass. |
| Scheduled fallback cannot become false no-object supervision | **Passed.** Existing parity tests and reconstruction notebook report fallback/representability separately. |
| Leaf PID loss applied once per selected event | **Passed.** Existing dedicated trainer test passes. |
| All capacity overflows rejected | **Passed.** Every indexed level is checked and the overflow fixture refuses production. |
| All 12 notebooks execute in fixture mode | **Passed.** One complete 12-group command passed. |
| Documentation matches schema-v4/notebook set | **Passed.** README and design/training documentation were updated and checked by the suite. |

## Historical issue-by-issue disposition

The following is the post-correction disposition of every issue classified in
the pre-edit gap audit.

### Preprocessing, schema, and input contracts

| Issue | Final status |
|---|---|
| MC-derived track fit/charge/input PID/energy | **Fixed and directly verified** |
| Target-derived composite input histogram | **Fixed and directly verified** |
| Non-daughter-sum composite p4 | **Fixed and directly verified** |
| Ambiguous feature missingness/node kinds | **Fixed and directly verified** |
| Multiple reco matches/provenance deduplication | **Fixed and directly verified** |
| Missing source/category metadata | **Fixed and directly verified** |
| Missing partial-topology propagation | **Fixed and directly verified** |
| Real PIDLikelihood and fit selection on basf2 data | **Implemented but not independently verified**; basf2/pilot unavailable |
| V3 shard-wide buffering | **Fixed and directly verified** |
| V4 marker/UID/worker publication contracts | **Fixed and directly verified** |
| Unmeasured v5 promotion | **Fixed and directly verified**; v4 remains default |
| Ten-million-event readiness | **Intentionally deferred scientific question** |

Direct-mDST preprocessing was not redesigned because no invariant test failed.

### Model, geometry, and reconstruction

| Issue | Final status |
|---|---|
| Separate detector/composite latent spaces | **Fixed and directly verified** |
| Geometry-before-context/circular relation dependency | **Fixed and directly verified** |
| Relation bias discarded | **Fixed and directly verified** |
| Euclidean parent/radius/tree-distance/anti-collapse gaps | **Fixed and directly verified** |
| Unsafe principal parent negatives | **Fixed and directly verified** |
| Dead reconstruction projection | **Fixed and directly verified** |
| Missing target-level/type conditioning | **Fixed and directly verified** |
| Query duplication/collapse observability | **Fixed and directly verified** with self-attention and metrics; scientific collapse rate remains deferred |
| Greedy presented as global solution | **Fixed and directly verified**; bounded comparator added and claim removed |
| Query/cardinality truncation | **Fixed and directly verified** |
| Capacity reporting limited to levels 1-3 | **Fixed and directly verified** |
| Unknown/leaf-only mother emission | **Fixed and directly verified** |
| Self-referential/untrained confidence | **Fixed and directly verified** |
| Recursive source reuse | **Fixed and directly verified** within proposals and selected sets; globally optimal decoding remains deferred |
| Raw IDs in exact-tree evaluation | **Fixed and directly verified** |
| Non-autoregressive rollout | **Fixed and directly verified** |
| False no-object scheduled fallback | **Fixed and directly verified** |
| Repeated leaf PID loss | **Fixed and directly verified** |
| Invisible soft/hard PID mismatch | **Fixed and directly verified** as diagnostics/ablations; best scientific mode deferred |
| Incorrect `tangent_dot` | **Fixed and directly verified** |
| Unscaled physical relation features | **Fixed and directly verified** |
| Lossy node-kind pair code | **Fixed and directly verified** |

### Training, channel, capacity, and production

| Issue | Final status |
|---|---|
| Fixture-only training CLIs | **Fixed and directly verified** |
| Eager loading/startup scans | **Fixed and directly verified** |
| Incomplete exact resume | **Fixed and directly verified** |
| Scheduled double-counting | **Fixed and directly verified** |
| Reconstruction validation bounds/denominators | **Fixed and directly verified** |
| Ignored pretraining `validate_every` | **Fixed and directly verified** |
| Ignored pretraining `log_every` | **Fixed and directly verified** |
| Missing aggregate pretraining metrics/best/latest | **Fixed and directly verified** |
| Corruption targets not optimized | **Fixed and directly verified** |
| Exact historical channel equivalence claimed | **Fixed and directly verified**; no exact implementation found, no claim retained |
| Ambiguous channel semantics | **Fixed and directly verified** |
| Single channel-pooling choice | **Fixed and directly verified** as configurable modes; held-out comparison deferred |
| `complete_only` called full-event reconstruction | **Fixed and directly verified** in configs/docs/denominators |
| Model implicitly described as MoE | **Fixed and directly verified** in deferred design note; MoE not added |
| Level embedding called literal hyperbolic positional encoding | **Fixed and directly verified** in deferred design note |
| Condor safety | **Fixed and directly verified**; render only, no submission |
| Physics performance/calibration/rare-channel/optimal decoding | **Intentionally deferred scientific question** |

### Notebooks and CI

| Issue | Final status |
|---|---|
| Notebook executability/machine-readable outputs | **Fixed and directly verified** |
| README lists obsolete notebook/schema set | **Fixed and directly verified** |
| No all-12 workflow | **Fixed and directly verified** statically; remote execution awaits push |
| Assertions only check figure existence | **Partially fixed**; core reconstruction/capacity/PID groups now assert semantic JSON, while visual-quality interpretation remains human review |
| Missing real sub-100-event pilot artifact | **Implemented but not independently verified**; guarded notebook added, basf2 unavailable |
| Rollout notebook missing divergence/ambiguity/slices/PID effects | **Fixed and directly verified** |

## Files changed

The correction touches these source groups:

- model/loss/reconstruction code under `src/hypertagging/{models,losses,reconstruction}`;
- pretraining and reconstruction trainers;
- ontology, capacity, and benchmark support;
- training, capacity, real-pilot, and notebook-runner scripts;
- ablation and target-policy YAML configs;
- all 12 regenerated fixture notebooks plus the new real-pilot notebook;
- notebook, preprocessing, channel, reconstruction, training, and deferred-design documentation;
- the new full-notebook GitHub Actions workflow;
- focused CPU acceptance tests and updated integration fixtures;
- this report and the immutable pre-edit gap audit.

The exact path-level state is represented by the final `git status --short` in
the audit handoff; no historical audit document was edited.

## Recommended first HTCondor ablation sequence

Use one immutable schema-v4 dataset index, split manifest, ontology version,
capacity report, and random seed set for all comparisons. Keep
`complete_only` and `reconstructable_partial` results in separate tables.

1. Establish the corrected non-MoE production baseline: static ontology,
   topology-safe parent loss, soft expected PID training, all-node mean channel
   pooling, greedy production decoding, and bounded set packing only as an
   evaluation comparator.
2. Compare PID kinematics one factor at a time: soft expectation,
   temperature-annealed softmax, straight-through hard, then canonical-pion
   first level. Report all mismatch diagnostics and physics/validity guards.
3. Compare channel pooling: all-node mean, B-root, learned attention, and
   level-weighted pooling, using channel retrieval and downstream reconstruction
   slices rather than fixture loss alone.
4. Measure query ambiguity on held-out events, then compare greedy with bounded
   set packing on the subset within the configured proposal bound. Do not make
   the bounded exponential solver a production default from this study.
5. Repeat the selected configuration under `reconstructable_partial`, with
   explicit incomplete/neutrino/K_L slices and denominators separate from
   `complete_only`.
6. Only after the sub-100-event basf2 pilot passes provenance, PIDLikelihood,
   fit-selection, p4 closure, B-root, and failure-example review, increase the
   pilot progressively and run the storage/throughput benchmark. Do not infer
   ten-million-event readiness from the two-event fixtures.

