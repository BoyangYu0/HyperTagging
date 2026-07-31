# Post-audit geometry and rollout review

Date: 2026-07-31  
Inspected HEAD: `68a7ebe81b14b6c329d78de91be18b5458c49089` (`master`, the current default branch)  
Review scope: the uncommitted post-audit patch on that HEAD. Historical audit
documents were read as snapshots and were not edited.

## Evidence established before editing

The initial worktree was clean. The commands were:

```bash
git status --short
git rev-parse HEAD
git log --oneline -15
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
/data/dust/user/boyangyu/uv_env/bin/python scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-post-audit-baseline-notebooks
```

The initial suite result was `231 passed, 8 skipped, 18 warnings in 351.78s`.
All 12 historical CPU fixture notebook groups executed. The required audit,
geometry, representation, and notebook documents were then read in full.

## Newly confirmed bugs at the inspected HEAD

1. Explicit relation class 4 (different B, same Upsilon event) was constructed,
   but an unconditional `lca_depth >= 0` exclusion removed it from directed
   parent negatives because the two B branches share the Upsilon LCA.
2. Reconstruction level (retained-tree height) was used to approximate edge
   distance. This fails on unbalanced trees.
3. Predicted hyperbolic distances were normalized by their own per-event
   maximum, making the regression scale depend on an outlier.
4. The variance floor was `gamma=1` per dimension even for `hyper_dim=64`, a
   scale inconsistent with the intended radius range.
5. Boundary occupancy compared `||z||` directly with a unit-ball threshold and
   omitted the required `sqrt(curvature)` factor.
6. The rollout forward produced relation, pointer, and type decisions under
   soft PID kinematics before switching to hard p4 for construction. Calling
   that behavior simply “hard rollout” was inaccurate.
7. Pair mass and energy were presented for overlapping composite/descendant or
   copied-source pairs even though their p4 sum double-counted recursive leaves.

## Fixed and directly verified

- `retained-tree-exact-edges-v2` derives `lca_node_id`, both directed
  edges-to-LCA matrices, exact path distance, root depth, and nearest-root
  distance solely from `parent_ids`. On the explicit unbalanced fixture, the
  direct level-0 daughter is one edge from the level-4 root; the rejected
  height difference is four.
- `exact-edge-log-fixed-scale-v2` uses exact paths and an event-independent
  `log1p(distance) / log1p(8 edges)` target. A test proves that adding a distant
  outlier does not rescale existing targets. The old height objective remains
  named only as `reconstruction-height-distance-v1` ablation code.
- Explicit parent-negative classes are authoritative: 0--3 are excluded, 4 is
  preferred, and 5 is used only when class 4 is unavailable. Fallback mode uses
  exact ancestors, descendants, siblings, and short graph-edge relationships.
  On the connected Upsilon fixture, B1 child 0 can select B2 positions 3 and 7;
  all 8 directed children have an eligible different-B negative, none lack a
  negative, active fraction is 1.0, and the ranking denominator is 8.
- `dimension-aware-tangent-radius-v2` adds configurable per-dimension variance
  targets, projection initialization scale, and fixed or learned-bounded
  tangent scaling. Presets initialize as follows on the deterministic fixture:

  | preset | hyper dim | variance target | init scale | radius p95 | boundary fraction |
  |---|---:|---:|---:|---:|---:|
  | tiny_cpu | 8 | 0.12 | 0.08 | 0.3822 | 0.0 |
  | gpu_debug | 16 | 0.08 | 0.065 | 0.3643 | 0.0 |
  | production_baseline | 64 | 0.05 | 0.05 | 0.6975 | 0.0 |

  All three produced finite ball coordinates, component losses, projection
  gradients, and tangent standard deviations. These initialization checks do
  not tune scientific loss weights.
- Boundary diagnostics now use `sqrt(curvature) * ||z||`. Hyperbolic training
  logs component losses, parent-negative coverage, per-loss gradient norms to
  the hyperbolic projection, and per-projection gradient norms.
- `rollout_pid_kinematics_mode` supports
  `soft_decision_hard_construction`, `hard`, `temperature_softmax`, and
  `straight_through_hard` through forward-call overrides. The default is now
  described accurately: its neural decisions are soft-conditioned and only
  construction is hard. Every mode was reproducible on the fixture and had
  exact daughter-sum p4 closure. Perturbing truth PID target tensors did not
  change hard-mode relation or pointer outputs.
- `physical-relations-overlap-aware-v2` adds recursive overlap,
  ancestor/descendant, disjoint-source, same-reco-source, copied-source
  conflict, and availability fields. Pair mass and energy are zero/unavailable
  for overlapping sources and remain available for disjoint composites.
- Greedy exclusive resolution remains the default. Bounded weighted set
  packing and one/two-level beam hypotheses are evaluation comparators. The
  fixture report includes duplicate daughter and typed-set rates, recursive
  overlap, query utilization, candidate survival, oracle coverage, first
  divergence, and greedy-versus-set-packing/beam differences. Query repulsion
  is optional and defaults to zero.
- All runtime geometry, distance, scale, and relation versions are serialized
  in model architecture/checkpoint contracts and named in the schema-v4
  runtime model specification without changing persisted parquet columns.

## Implemented but not real-data verified

- `inspect_trained_physics_validation.ipynb` requires an explicit real parquet
  and trained checkpoint. It contains no fixture fallback and reports
  edge purity/efficiency, tree metrics, PID/level/multiplicity/partial-topology
  slices, masses, and only contract-valid Mbc/DeltaE or rare-channel fields.
- `inspect_real_mdst_pilot.ipynb` checks track/ECL provenance,
  PIDLikelihood, selected energy hypothesis, charge, p4 closure, K_L/KLM
  provenance status, B roots, distributions, and truth-input exclusion.
- The three numerical/search notebooks validate mechanisms on fixtures, not
  calibration, reconstruction efficiency, physics distributions, or scale
  throughput.

## Scientific design choices

- Reconstruction level remains `L(mother)=1+max L(daughter)` because the
  autoregressive schedule requires retained-tree height.
- The fixed exact-distance scales (8 target edges and prediction scale 3),
  radius targets, preset tangent targets, and initialization scales are safe
  numerical defaults, not physics-optimized values. Two ablation configs expose
  lower-radius and learned-bounded alternatives.
- `soft_decision_hard_construction` remains the compatibility default; fully
  hard conditioning is explicit rather than silently substituted.
- Greedy remains the matched production baseline. Beam/set packing and query
  repulsion are not promoted without held-out evidence.
- The 41-token retained vocabulary is unchanged.

## Deferred physics or scale questions

- No real basf2 pilot ran because `basf2` is unavailable on this host.
- No trained real-data checkpoint was supplied, so calibration, rare/unseen
  channels, mass resolution, Mbc, DeltaE, missing mass, and teacher/free physics
  differences remain unmeasured.
- No representative full-data preprocessing, long training, GPU benchmark, or
  HTCondor submission was run.
- Free rollout and bounded beam are explicitly evaluation-only batch-size-one
  implementations. The documented batched design uses padded/ragged event and
  beam states, masked node/query axes, segmented append, and compaction between
  levels; representative throughput remains open.
- Real frequencies of contracted generator intermediates, truth-topology-only
  nodes, unmatched objects, K_L/KLM inputs, and real denominator/depth
  distributions remain open. See
  `docs/post_audit_retained_tree_definition.md` for the exact retained-tree
  scope and fixture-only inventory.

