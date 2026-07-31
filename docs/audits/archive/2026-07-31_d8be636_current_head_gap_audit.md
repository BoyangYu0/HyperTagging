# Current-head gap audit

Audit date: 2026-07-31 (Europe/Berlin)

## Starting evidence

- Branch: `master`, the repository default branch.
- Starting SHA: `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60`
  (`corrections`).
- Starting worktree: clean. The initial `git status --short` produced no
  entries.
- The requested initial history was:

  ```text
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

- Baseline CPU suite:
  `223 passed, 8 skipped, 18 warnings in 418.20s` from
  `/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q`.
- The eight skips are one legacy-preprocessing-script presence test, five
  `graFEI_reduced/models.py` parity tests, and two
  `graFEI_gpt/models.py` parity tests. Those external historical sources are
  absent from this checkout.
- The 18 warnings are 15 PyTorch odd-attention-head/nested-tensor warnings and
  three explicit legacy-conflated-data diagnostic warnings.
- All 12 notebook groups executed successfully on CPU fixtures. Executed
  notebooks, JSON/JSONL/CSV reports, figures, tiny parquet files, and tiny
  checkpoints are retained under
  `/tmp/hypertagging-current-head-notebooks`.
- GitHub Actions run
  [30629273834](https://github.com/BoyangYu0/HyperTagging/actions/runs/30629273834)
  is a successful `CPU correctness` push run for the exact starting SHA.
- Running the documented notebook command regenerated the 12 checked-in
  notebook sources. Those mechanical changes occurred after the clean
  starting-state capture and before any deliberate source edit.

## Classification method

Every issue below was checked against current source, current tests, or a
fresh executable artifact. Historical prose was not accepted as evidence by
itself. Status terms are exactly:

- **fixed and directly verified**: current implementation plus a current-head
  test or executable artifact directly exercises the claim;
- **implemented but not independently verified**: an implementation exists,
  but this host cannot directly validate the real-data or production claim;
- **partially fixed**: a material part exists but the stated contract is not
  complete;
- **still open**: the required behavior is absent or a current code path
  contradicts it;
- **intentionally deferred scientific question**: fixtures cannot establish
  the requested scientific or scale conclusion.

## Historical preprocessing, schema, and input-contract issues

| Historical issue | Current-head status | Direct evidence |
|---|---|---|
| MC-derived track-fit selection, charge, input PID, or energy | **fixed and directly verified** | Schema-v4 raw tracks use reconstructed p3/charge, unknown input token, canonical pion energy, separate hypotheses, and target-only truth. Current no-truth-leakage, leaf-kinematics, v4, and notebook contract checks pass. |
| Target-derived daughter PID histogram consumed by composites | **fixed and directly verified** | V4 separates input and truth histograms; `adapt_model_composite_features` excludes target metadata; target-metadata invariance and gradient tests pass. |
| Composite mother p4 sourced from MC or regressed independently | **fixed and directly verified** | Construction, corruption rebuild, teacher forcing, scheduled contexts, and rollout use recursive daughter sums; closure tests and QA notebook pass. MC p4 remains diagnostic. |
| Ambiguous leaf/composite feature missingness and node kinds | **fixed and directly verified** | Explicit value masks, node kinds, leaf-mode IDs, model-only composite feature names, and categorical embeddings are present and tested. |
| Multiple reco matches and provenance deduplication | **fixed and directly verified** | Basf2 adapter deduplicates by underlying provenance and reports relation multiplicity; deterministic adapter tests pass. |
| Source-file/category metadata missing for multi-input production | **fixed and directly verified** | V4 sidecars/manifests and dataset indexes retain source/category/range contracts; manifest/index tests pass. |
| Partial topology/completeness not propagated | **fixed and directly verified** | Local and recursive completeness, `complete_only`, `reconstructable_partial`, and diagnostic policies propagate through collation, capacity, targets, validation, and metrics. |
| Real PIDLikelihood availability and fit-selection behavior on basf2 data | **implemented but not independently verified** | Supported accessors and provenance fields exist, but basf2 is unavailable and no real sub-100-event pilot has run on this host. |
| V3 shard-wide JSON row buffering | **fixed and directly verified** | Production default is event-row schema-v4 with bounded writer, sidecar, and completion marker. Streaming fixture artifacts and tests pass. |
| V4 publication validity, marker ordering, UID scalability, worker I/O partitioning | **fixed and directly verified** | Marker-last overwrite behavior, index validation, SQLite UID uniqueness, and disjoint row-group/file tests pass. |
| Native nested v5 as an unmeasured production promotion | **fixed and directly verified** | V5 remains explicitly experimental/default-off and schema-v4 remains production. The bounded fixture benchmark reports size/write/full/projected/JSON/RSS measures. |
| Ten-million-event throughput and memory readiness | **intentionally deferred scientific question** | No representative production-scale measurement exists; the two-event fixture benchmark cannot establish readiness. |

## Historical model, geometry, and reconstruction issues

| Historical issue | Current-head status | Direct evidence |
|---|---|---|
| Flat detector/composite projection and separate latent spaces | **fixed and directly verified** | Heterogeneous adapters enter one shared contextual state and one shared Poincare projection; contextual encoder tests pass. |
| Geometry projected before physical context / circular hyperbolic relation dependency | **fixed and directly verified** | Stage-A physical attention precedes task/hyperbolic projections; optional Stage-B hyperbolic refinement is downstream and testable. |
| Relation bias computed but discarded | **fixed and directly verified** | Bias enters attention logits; output-change and finite-gradient tests pass. |
| Euclidean parent ranking, reversed radius, missing tree-distance and anti-collapse objectives | **fixed and directly verified** | Poincare distance, leaves-outside radius targets, normalized tree-distance, VICReg variance/covariance, and collapse diagnostics are present and covered. |
| Generic parent-margin negative can be a descendant, ancestor, sibling, or near-positive | **still open** | `parent_child_margin_loss` currently selects every valid node except child and true parent. The separate curriculum hard-negative helper is safer, but it does not constrain this principal directed-parent loss. |
| Reconstruction-specific projection appears meaningful but is unused | **still open** | `HeterogeneousNodeEncoder.reconstruction_head` produces `reconstruction_projection`; both the leaf PID head and mother decoder consume generic `node_embeddings`. No reconstruction loss reaches the reconstruction head. |
| Decoder queries have no target-level/type conditioning | **fixed and directly verified** | Target-level embeddings and expected-type pointer conditioning are implemented and tested. |
| Query duplication/collapse and within-level ambiguity | **partially fixed** | Learned slots, cross-attention, Hungarian matching, confidence, source-conflict pointer decoding, and greedy exclusive resolution exist. There is no query-query self-attention or proposal-repulsion loss, no duplicate/unused-query metrics, and no bounded global resolver. |
| Greedy exclusive resolution presented as global ambiguity solution | **still open** | Only deterministic confidence-ordered greedy selection exists; no beam or weighted set-packing comparator exists. |
| Query/cardinality target overflow silently truncated | **fixed and directly verified** | Dataset/index capacity is policy-specific and `require_capacity` refuses overflow; target construction also raises. |
| Production-baseline capacity only considered levels 1-3 | **partially fixed** | Levels above three fall back to global capacities and are checked, but the preset/config and notebook report do not explicitly list every retained level, per-level configured capacity, overflow, and margin. |
| Unknown or physically impossible mother tokens can be emitted | **still open** | `ReconstructionConstraintPolicy.type_constraints` starts with all 41 tokens allowed. Token 0 and leaf-only species are not excluded by a versioned static ontology; observed-level frequencies are the only mask/prior. |
| Confidence target self-referential and untrained | **fixed and directly verified** | Confidence target uses hard type correctness, pointer overlap, and structural validity, with Brier/ECE reporting and checkpoint flagging. |
| Recursive source reuse only checked after construction | **partially fixed** | Pointer validity and source-conflict penalties/decoding exist within proposals and greedy final exclusivity uses recursive leaf sources. Cross-query duplicate/overlap repulsion and global set resolution are absent. |
| Raw node IDs used for exact-tree evaluation | **fixed and directly verified** | Canonical source/topology-first alignment is type-independent; non-tautological metric tests pass. |
| Rollout not truly level-autoregressive | **fixed and directly verified** | Each level re-encodes the accumulated state, appends daughter-summed symbolic mothers, and stops on explicit conditions; teacher/scheduled/free fixture tests pass. |
| Scheduled fallback can create false all-no-object supervision | **fixed and directly verified** | Default `fallback_teacher`, masked/recovery alternatives, representability counts, and acceptance tests directly cover the case. |
| Leaf PID loss repeated at every selected level | **fixed and directly verified** | It is applied only at the first selected event level; dedicated tests pass. |
| Soft training PID versus hard rollout PID mismatch invisible | **partially fixed** | Training uses differentiable charge-compatible soft expected energy and rollout uses a hard charge-compatible discrete mass. Canonical-pion mode exists, but soft-hard energy/mass/entropy/relation/pointer diagnostics and temperature/straight-through modes are absent. |
| Relation feature named `tangent_dot` is not tangent geometry | **still open** | It is a direct dot product of Poincare-ball coordinates rather than `logmap0(z_i) dot logmap0(z_j)`. |
| Physical relation features mix incompatible raw scales | **still open** | Mass, energy, p3 dot, charge, levels, and binary flags enter one MLP without an explicit scaling/normalization contract. |
| Lossy summed node-kind pair code | **still open** | `kind_pair=(kind_i+kind_j)/8` aliases distinct ordered pairs; there is no proper pair-kind embedding. |

## Historical training, validation, channel, and production issues

| Historical issue | Current-head status | Direct evidence |
|---|---|---|
| Fixture-only training CLIs | **fixed and directly verified** | Real parquet/manifest streaming trainers, CPU dry runs, transfer, validation, resume, and YAML precedence are tested. |
| Eager full-dataset loading and repeated startup scans | **fixed and directly verified** | Event-row streaming, bounded shuffle, online Welford state, dataset indexes, and disjoint workers are implemented and tested. |
| Resume omits optimizer/scaler/RNG/data-order state | **fixed and directly verified** | Exact single-worker resume stores and checks full training/data-order contracts; mismatch tests pass. |
| Scheduled training double-counts teacher and predicted events | **fixed and directly verified** | One primary sampled context is optimized; auxiliary teacher loss defaults to zero. |
| Validation only checks one batch/event and mixes macro/micro denominators | **fixed and directly verified** | Reconstruction validation honors batch/event bounds and reports macro plus sufficient-statistic micro metrics. |
| Pretraining `validate_every` is ignored and validation is one final batch | **still open** | The field is parsed but never used in the loop. Only one final batch is passed to collapse diagnostics. |
| Pretraining `log_every` is ignored | **still open** | Every step is logged unconditionally. |
| Pretraining validation lacks requested aggregate component/quality metrics and best/latest checkpoints | **still open** | Current validation only returns collapse diagnostics; no periodic aggregate loss/relation/parent/tree/radius/channel/PID metrics or separate best/latest files exist. |
| Corruption labels/derived fields/hard negatives do not affect training | **fixed and directly verified** | Only applied corruptions are labelled, derived fields rebuild, invalid nodes leave positive masks, and corruption/correctness/hard-negative losses are optimized. |
| Exact historical channel dictionary-array equivalence claimed | **fixed and directly verified** | No such claim is made. All reachable commits, branches/tags, migration/legacy/archive files, string history, and unreachable Git objects were searched again; no exact implementation was found. |
| Current channel representation lacks explicit semantics | **fixed and directly verified** | Canonical signatures, structured counts, exact/full/reconstructable IDs, depth/multiplicity/intermediate arrays, and distinction among same-event/B-side/exact/similarity labels are documented and round-trip tested. |
| Channel pooling has only one untested choice | **still open** | Pretraining uses mean pooling over all B-branch nodes. B-root, learned-attention, and level-weighted pooling ablations/configs are absent. |
| `complete_only` described as complete full-event reconstruction | **partially fixed** | Documentation identifies a target policy and separate denominators, but no explicit paired production configs/design note prominently covers incomplete, neutrino, K_L, and other partial topologies. |
| Production model is implicitly mixture-of-experts | **still open** | The model is not MoE, but there is no explicit current limitation/future-ablation design note. |
| Level information claimed as literal hyperbolic positional encoding | **still open** | Implementation is a learned Euclidean level embedding plus radius supervision; the distinction is not stated prominently in a dedicated note. |
| HTCondor safety and accidental submission | **fixed and directly verified** | Renderers are separate from explicit submission and safety tests pass. No Condor job was submitted in this audit. |
| Scientific performance, PID calibration, rare-channel quality, and optimal decoding | **intentionally deferred scientific question** | Deterministic fixtures establish software contracts only. Matched held-out HTCondor ablations are still required. |

## Historical notebook and CI issues

| Historical issue | Current-head status | Direct evidence |
|---|---|---|
| Notebook suite not executable or not machine-readable | **fixed and directly verified** | All 12 groups execute and emit retained figures plus JSON/CSV/checkpoints where specified. |
| `notebooks/README.md` describes six schema-v1/v2 notebooks | **still open** | It does not list the 12 current groups and does not describe schema-v4 fixture defaults. |
| Normal push CI runs all notebooks | **partially fixed** | Push CI runs the full CPU unit suite and eight notebook groups. There is no manual/scheduled all-12 workflow. |
| Notebook assertions only prove a figure exists | **partially fixed** | Several groups validate JSON semantics, but four-vector/direct-GPT and some dataset visual checks remain mostly figure/text assertions. |
| Real sub-100-event mDST pilot inspection artifact | **still open** | No `inspect_real_mdst_pilot.ipynb` exists. basf2 is unavailable, so only an honest command/template can be produced here. |
| Trained-checkpoint rollout notebook lacks divergence/error slices and PID mismatch diagnostics | **partially fixed** | The reconstruction notebook reports first divergence, edge/type errors, teacher/free rollout, and checkpoint loading. It lacks duplicate metrics, greedy/global comparison, bounded channel/multiplicity/depth slices, and soft-hard PID effects. |

## Focused correction scope selected from this audit

The correction will preserve direct-mDST preprocessing and schema-v4 while
addressing the open/partial source-level items above: reconstruction projection
flow, topology-safe directed parent negatives, query interaction and ambiguity
metrics plus a bounded alternative resolver, static mother ontology, measurable
PID modes, relation geometry/scaling/pair-kind contract, periodic pretraining
validation, channel pooling ablations, per-level capacity reporting, partial
topology configs/docs, notebook/CI coverage, real-pilot and trained-rollout
artifacts, and explicit deferred-feature notes.

No current-head evidence supports physics-performance or ten-million-event
readiness claims.
