# Correctness Revision Audit

Audit date: 2026-07-30  
Audited branch: `master`  
Audited revision: `da8f35796ae81835645ac12721a5287829c02c31`
(`update gpt like functions`, also `origin/HEAD`)

## Baseline verification

The worktree was clean at audit time.

The literal requested command, `python -m pytest -q`, selected
`/usr/bin/python` and failed before collection because that interpreter does
not contain pytest. The repository's documented CPU environment produced:

```text
114 passed, 8 skipped, 15 warnings in 349.03s
```

The skips are existing historical-equivalence tests whose external legacy
sources are not present. The warnings are existing PyTorch nested-tensor
warnings for historical models with an odd number of attention heads.

## Verified scientific invariants to preserve

1. MC truth may define retained topology, training labels, channels, and
   diagnostics, but not data-path detector inputs.
2. Every reconstructed composite four-vector is recursively the exact sum of
   its selected reconstructed daughters.
3. MC four-vectors remain confined to `mc_px`, `mc_py`, `mc_pz`, and
   `mc_energy`.
4. Schema-v1 export and schema-v2 loading remain supported without changing
   their stored semantics.
5. Model-internal PID/type values are reduced tokens in
   `[0, len(PDG_TOKENS))`; raw PDGs are separate data.
6. Node identity, reconstructed-object identity, source identity, copy
   identity, recursive leaf sources, and MC identity are distinct.
7. Full training and full preprocessing remain HTCondor workflows; local work
   is limited to CPU pilots and explicitly guarded tiny CUDA smoke tests.

## Belle II API audit

The release-08-03-00 `Track` API documents
`getTrackFitResultWithClosestMass(requestedType)` and
`getTrackFitResultWithBestPValue()`. The former chooses the stored fit closest
to a requested particle mass, so a requested type derived from MC truth leaks
truth into reconstructed momentum selection. V3 will use a documented
data-independent policy: the best-p-value fit when available, with a pion-fit
request as the deterministic compatibility fallback.

Official Belle II data-object documentation confirms that a `Track` can be
related to `PIDLikelihood`, and that `PIDLikelihood` provides `isAvailable`,
`getLogL`, and charged-stable hypotheses. Release-8 documentation uses
`isAvailable` for the availability test. V3 will read only these supported
relations/accessors and will mark all unavailable likelihood values explicitly.

References:

- <https://software.belle2.org/release-08-03-00/doxygen/classBelle2_1_1Track.html>
- <https://software.belle2.org/release-09-00-14/doxygen/classBelle2_1_1PIDLikelihood.html>
- <https://software.belle2.org/development/doxygen/ECLCluster_8h_source.html>

## Issues found

### Preprocessing and schema

- `_collect_tracks` takes related MC PDG, requests the corresponding charged
  stable fit, chooses the PDG mass, and computes input energy from it.
- A matched retained `TreeNode` stores MC charge rather than the reconstructed
  fitted charge even though `RecoRecord.charge` is available.
- Track `RecoRecord.pdg` is truth PDG for matched tracks and an implicit pion
  otherwise; input and target PID concepts are conflated.
- Schema-v2 exports `pdg`, `token`, `charge`, and `energy` without separating
  truth targets from reconstructed inputs. Its semantics must not be silently
  changed.
- PID likelihoods, per-mass energy hypotheses, fit-selection provenance,
  leaf-kinematics mode, relation multiplicity, reconstruction completeness,
  and recursive leaf-source sets are absent.
- The builder silently chooses the first reco record related to an
  `MCParticle`; further records are later treated generically as unmatched.
- Particle/Track/ECL deduplication uses only generated string IDs, not shared
  underlying mDST provenance.
- Multi-input events store an empty source-file/category field rather than the
  actual source file.
- Partial truth decays are not explicitly counted or policy-controlled.
- `min_daughters` defaults to one.

### PID contract

- Tiny level fixtures use raw PDGs, including negative values, in
  `pid_labels`.
- Model constructors and dry-runs use PID/type vocabularies of 4096.
- Both flat and heterogeneous encoders apply `abs()` and clamping before PID
  embeddings.
- Target construction applies `abs()` and clamps to 4095.
- Histogram construction clamps invalid tokens instead of rejecting them.
- Flat-batch node-kind inference tests `pid_labels == 22`, confusing raw photon
  PDG with the reduced photon token.
- Mother decoding does not apply an allowed reduced-token mask.

### Kinematics

- There is no authoritative leaf/composite p4 module.
- Raw track input energy is coupled to MC-derived PID.
- No differentiable soft PID-energy path or hard predicted-PID path exists.
- Teacher forcing can therefore receive better truth-dependent leaf
  kinematics than inference.
- The physics loss applies one unscaled MSE over raw `(px, py, pz, E)`.

### Context and hyperbolic geometry

- `HeterogeneousNodeEncoder` projects independent per-node adapter outputs to
  the Poincare ball before event contextualization.
- `RelationBias` requires hyperbolic distances in the first contextual layer,
  creating circular dependence between context and geometry.
- The contextual output is used only by the reconstruction decoder; tree and
  channel projections remain pre-contextual.
- Curvature is passed through basic maps, but several losses/diagnostics call
  geometry helpers without the configured curvature.
- The main hierarchy is learned through a Euclidean relation classifier and
  parent ranking; there is no direct multiscale hyperbolic tree-distance loss.
- There is no FSP-only, truth-composite, corrupted-composite curriculum.

### Channel supervision

- One retained-tree signature is used for both generator channel identity and
  reconstructable topology.
- `find_b_branches` admits `B_s` under Upsilon(4S) and falls back silently to
  unrelated top-level B nodes.
- It does not validate exactly two direct B daughters of the same resonance.
- Channel loss compares only B1 and B2 inside each event. The two branches are
  not automatically same-channel positives, and cross-event positives are
  absent.
- No long-tail memory-bank interface exists.

### Query decoding, matching, and losses

- Query slots have no target-level conditioning.
- Mother and daughter-cardinality capacity is not measured per level.
- Truth mothers beyond `n_queries` and cardinalities beyond the head are
  silently truncated/clamped.
- Pointer scores are independent of expected mother type.
- Invalid node kinds, charge, duplicate source use, and level combinations are
  not masked.
- Confidence controls exclusivity ranking but has no target or loss.
- Object/no-object and sparse pointer targets use unweighted BCE.
- Matching cost contains only type and soft Jaccard.
- `hungarian_or_greedy` silently changes to greedy assignment for larger
  matrices if SciPy is unavailable; SciPy is not a declared dependency.

### Rollout and evaluation

- Composite `source_node_ids` are newly generated IDs, so recursive reuse of a
  leaf through different intermediate composites is not detected.
- No explicit recursive `leaf_source_ids` representation is propagated.
- Evaluation compares raw generated node IDs and edges, so structurally
  identical predicted trees with different IDs can fail exact match.
- Tree validity checks level ordering but does not perform general cycle
  detection.
- Teacher-forced, scheduled, and free rollouts exist only for fixture-scale
  scaffolding.

### Training and production

- Both new training CLIs call fixture-only dry-run functions regardless of
  data/output/checkpoint arguments.
- There is no real shard/manifest dataset, split manifest, dataloader, all-level
  sampling, validation loop, AMP/scaler path, encoder-only transfer, or
  production checkpoint metadata.
- Checkpoint writes are non-atomic and omit encoder-only state, scaler, git
  commit, PID vocabulary, feature hash, split hash, and random states.
- Production workers omit `--schema-version` and validators accept only v1/v2.
- Manifest records omit scientific schema/configuration provenance.
- Shards are buffered completely in memory.
- There is no global cross-shard UID/range/schema/feature/PID validator.
- No GitHub Actions workflow exists.

### Notebooks

- Existing generated notebooks cover v1/v2, preliminary geometry, rollout, and
  QA but not the corrected leaf PID contract, real trainer, capacity/loss
  balance, or production manifest.
- Smoke tests mainly assert section strings, no execution error, and at least
  one PNG. They do not validate the scientific JSON/table/checkpoint artifacts
  requested for this revision.

## Compatibility strategy

1. Preserve `export_trees()` as the v1 exporter and retain v2 loading/export
   semantics.
2. Add an independent `direct-mdst-tree-v3` exporter and normalize v1/v2/v3
   into a v3 model contract in memory.
3. Mark legacy v1/v2 track PID/energy provenance as unknown/legacy rather than
   pretending it satisfies the corrected raw-track contract.
4. Make v3 the explicit production-worker schema; do not depend on CLI
   defaults.
5. Keep raw PDG and truth supervision fields outside all model input token
   tensors.
6. Centralize data-compatible track, cluster, and composite p4 construction and
   reuse it in preprocessing, training, rollout, and validation.
7. Add new tests before changing shared tree-building behavior and retain the
   existing v1/v2 regression suite.

## Implementation stages and files

### Stage 1: scientific data contract

- Add `preprocessing/schema_v3.py` and
  `reconstruction/kinematics.py`.
- Extend `mdst_tree_builder.py`, `basf2_mdst.py`, `channels.py`,
  `pid_filter.py`, `preprocess_mdst.py`, and package exports.
- Update `data/heterogeneous.py`, level fixtures/collation, notebook fixtures,
  and splitting/capacity statistics.

### Stage 2: model and losses

- Update heterogeneous, hyperbolic, relation, attention,
  level-autoregressive, pointer, ablation, hyperbolic-pretraining,
  reconstruction-loss, physics, and set-matching modules.
- Add curriculum, cross-event channels, confidence calibration, recursive
  source propagation, and canonical structural metrics.

### Stage 3: real training and production

- Add `training/data_module.py`, `pretrain_trainer.py`,
  `reconstruction_trainer.py`, `pretrained_transfer.py`, and `validation.py`.
- Upgrade checkpointing, logging, both training CLIs, production planner,
  global validator, Condor scripts/configs, and dependencies.

### Stage 4: physicist-facing verification

- Add the leaf-PID, capacity/loss, training-pipeline, and production-manifest
  notebooks and generators.
- Update dataset, hyperbolic, rollout, QA, notebook execution, documentation,
  ablations, and CI.
- Add the full requested CPU regression and integration test surface.

## Verification boundaries

CPU tests and tiny parquet pilots can verify schema semantics, absence of truth
leakage, token ranges, p4 construction, context ordering, finite geometry,
matching, checkpoint transfer/resume, canonical evaluation, production command
construction, and notebook execution. They cannot establish physics
performance, PID calibration quality, optimal thresholds, or production
memory/runtime at ten-million-event scale.

No full preprocessing, long training, local unguarded CUDA use, or HTCondor
submission is authorized. If basf2 is unavailable, only the exact sub-100-event
pilot command will be reported.
