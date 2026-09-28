# Scientific design and study review — 2026-09-28

Reviewed scientific source: `db1ff41467aba5080b04bd89ae532fbdbd8788c2`, the
remote `master` revision observed at review start. The detailed evidence is in
[preprocessing/pretraining](review_20260928/pretraining.md),
[reconstruction/inference](review_20260928/reconstruction.md), and
[the study ledger](review_20260928/studies.md). Source citations in those
appendices refer to that revision. This review did not retrain models, rerun
all historical physics evaluations, inspect sealed test, or change scientific
implementation. One bounded CPU reproducer verifies a specific training defect.

## Decision

The scientific goal and main architectural constraints are coherent. The
recent decision to hold 70,000 training events, expand independent validation,
preserve failed comparisons, and avoid promotion is justified. The evidence
does **not** yet justify reliable recursive reconstruction, a hyperbolic
advantage, improved pretraining over no refinement, or a causal data-size
benefit. More data and longer training have not been ruled out either.

The next direction needs revision: finish and classify the already submitted
Phase62 under its immutable contract, then prioritize a verified training
eligibility defect and misleading pretraining diagnostics before another
loss-weight sweep. Use failure decomposition to decide whether the following
experiment should change preprocessing, geometry, proposal generation or
ranking. A final read-only scheduler check on 2026-09-28 observed both Phase62 jobs
running; this is execution state, not completion or validated physics evidence.
The timestamped [observation](review_20260928/phase62_scheduler_observation.json)
is preserved. Neither job was altered or resubmitted.

## Goal and what counts as success

Infer a physically admissible retained decay forest from detector observations:
which reconstructed sources form each mother, their particle identities, and
the recursive topology. Hyperbolic geometry is a hypothesis for encoding
hierarchical relationships, not a required conclusion. The model's estimand
is a retained, detector-supported tree; it is not every unobserved physical
decay in the event.

Keep truth out of generation, pruning, ranking and stopping. Preserve source
exclusivity, one parent per node, charge and ontology constraints, and exact
recursive daughter-sum p4. Missing measurements need availability masks;
train-only statistics and source-separated roles remain essential.

Promote against nontrivial recursive topology, topology plus PID, depth and
relevant B-half/B-pair or category results with explicit denominators. A
configured root, leaf-only forest, source-set match, finite checkpoint or
daughter-sum closure is not evidence of a correctly reconstructed decay tree.
Use physical reconstructed-minus-truth momentum/mass resolution only after
topology/source matching, separately from algebraic p4 closure.

## What the studies establish

The ledger inventories 193 tracked evidence/configuration files and 29 external
historical study families. It covers migration Phases1–13, early pretraining,
transfer/resource calibration, Stage A and reconstruction Phases34–62.
No standalone reconstruction closeouts numbered14–33 were found; migration,
curriculum and reconstruction phase numbers must not be conflated.

| Study family | Supported conclusion |
| --- | --- |
| Historical migration and early pilots | Software/data contracts and bounded runtime evidence; not modern end-to-end physics quality. |
| Stage A and Phases34–39 | Small local improvements, replay/transfer caveats, no promotion-grade deep topology. |
| Phase40/40r1 | Training size, updates and cohort changed together; no isolated learning-curve result. |
| Phases41–47 | Pointer/pretraining-length/encoder changes give mixed sparse results; scientific fixes change comparability. |
| Phases48–51 | Phase48 did not execute PID treatment; later repaired treatments execute but do not establish joint topology benefit. |
| Phases52–54 | Recovery-weight changes improve some local counts while exact components/forests stagnate or regress. Stop dose tuning. |
| Phases55–58 | Parent/PID studies include evaluator repair, missing controls and selection contamination; no reproducible quality winner. |
| Phases59–60 | Phase59 is a 25-event feasibility pilot; Phase60 fails before late-PID treatment and has no physics endpoints. |
| Phase61 | Both schedules complete with mixed shallow quality; pre-treatment divergence prevents an identical-prefix causal comparison. |
| Phase62 | Fresh seed and untouched cohort replication running at final scheduler check; no completed outcome verified here. |

Latest completed Phase61, control late PID0.2 versus candidate0.1:

| Endpoint | Control | Candidate |
| --- | ---: | ---: |
| Selection source-set plus mother-PID recovery | 330/3,576 | 326/3,576 |
| Configured full-root construction | 4/100 | 0/100 |
| Retained full LCAG pairs | 27/3,412 | 24/3,412 |
| Retained B-half/component LCAG pairs | 31/1,961 | 28/1,961 |
| Retained exact nontrivial full components | 14/142 | 15/142 |
| Coherent primary retained forests | 0/100 | 0/100 |

Every exact component is depth one. Paired event intervals include zero.
The apparent ~9% selection recovery is not recursive forest efficiency.
Only96/142 nontrivial full components are marked representable; leaf-heavy
overall representability obscures this ceiling. Both unconditional results
and attainable-subset diagnostics are needed. On the 20-event beam subset,
even candidate oracle achieves only4/673 retained full LCAG pairs and no
coherent forest, so ranking alone is not an established solution.

Phase61's two arms differ in123/141 tensors at common-prefix step1094, with
loss divergence starting at944. Phase62 acknowledges and preserves that
variation. It is a useful exploratory replication, but future small-effect
late-treatment experiments should fork one authenticated full prefix state
and measure the training noise floor across independent seeds.

## Findings that should determine the next update

| Priority | Finding and evidence | Required next action |
| --- | --- | --- |
| P1 | Scheduled-sampling alignment accepts already-parented daughters while inference excludes them. Six-node CPU reproduction confirms1/1 incorrectly representable target. | Share truth-free forest eligibility between predicted-context target alignment and decoder masks. Handle teacher prefixes separately so future truth parents do not exclude legitimate daughters. Measure incidence before a matched corrected run. |
| P1 | Channel retrieval concatenates repeated curriculum views of the same physical branch, masking only self-diagonal. | Exclude same-event/branch identity, report per-view and cross-source metrics, preserve old values as versioned historical diagnostics. |
| P1 | Parent metric includes unsupported FSP zeros; its objective ranks against other-B/unrelated nodes rather than plausible same-branch wrong parents. | Publish supported numerator/denominator; distinguish coarse separation from exact-parent classification. Test an ordered immediate-parent objective separately. |
| P1 diagnostic | Native radius loss and near-zero radius gradients strongly suggest tangent-cap saturation; the existing boundary metric misses the model's own cap. | Measure pre-cap norms/derivatives and per-level radius on a frozen development cohort. Change parameterization only if confirmed; do not blindly increase radius weight. |
| P2 | Object-only recovery gives no missing-subtree pointer/type correction and can reward an already matched query. | Rename its interpretation, diagnose first-error reachability, then compare masking-only, current recovery and a legal corrective objective at matched exposure. |
| P2 | Confidence targets use raw0.5 pointer threshold/argmax type; deployment uses0.35, cardinality and constrained alternative candidates. | Calibrate actual decoded candidates on development data, then freeze one rule for independent top-1 evaluation. |
| P2 | Current per-level decoder capacities fail full ONNX bundle export; basf2 proposes/ranks differently from offline beam. | Extend per-level manifest support and establish model/search parity before a current real-mDST deployment pilot. |

The scheduled-sampling defect is demonstrated; its prevalence and effect on
historical models are unmeasured. Radius saturation remains a mechanistic
inference. Neither finding alone explains all weak reconstruction results.
The detailed reviews include exact source lines and distinguish these evidence
levels. The [reproducer](review_20260928/reproduce_parented_alignment.py) and
[observed output](review_20260928/reproduce_parented_alignment.json) are retained.

## Recommended updates throughout the pipeline

**Preprocessing:** retain schema-v4 and the truth/input separation. Produce a
development representability census from physical roots through detector,
ontology, arity/query capacity and legal targets. Slice by category, depth,
multiplicity, missing charged/neutral sources and KLM policy. Check feature
availability, duplicated sources, charge/PID support, train-only normalization
and source-domain shift. Historical10M production and an old pilot do not
establish current-revision KLM completeness or current basf2 behavior.

**Pretraining:** retain FP32 geometry and variance/covariance kernels. Correct
validation reductions and metric identity; inspect supported gradient magnitudes
and conflicts, not only a maximum-to-LCA ratio. Diagnose radial usability,
coarse parent negatives and synthetic-corruption realism. Test charge-compatible
PID support and actual training-only decoder negatives as separate hypotheses.
Establish current-refinement versus no-refinement, and Euclidean versus
hyperbolic/unpretrained baselines before attributing value to hyperbolic learning.

**Losses:** the detailed pretraining table covers relation CE, parent margin,
tree distance, radius, channel contrast/structured regression, variance,
covariance, leaf PID, corruption CE, correctness BCE, hard negatives and inactive
legacy pair BCE. The reconstruction table covers object/pointer focal BCE,
mother type, cardinality, confidence, soft p4/charge consistency, source conflict,
mother charge, query repulsion, object recovery, leaf PID and optional teacher
auxiliary loss. Keep their supports and gradient contributions separate.
Resolved arm overrides, not common config fragments, determine effective weights.
Do not optimize a weighted total as if it were directly comparable physics quality.

**Reconstruction:** fix legal predicted-context supervision first. Keep
Hungarian unordered-set matching and exact persistent daughter sums. Log
assignment stability and matched/unmatched query behavior. Diagnose exposure
error by generation, teacher-state performance and the first irreversible wrong
merge. A frozen PID head can still send PID-loss gradients through a trainable
encoder, so audit actual parameter gradients rather than only head flags.

**Inference and beam:** preserve truth-free detector projection, coherent
cross-level states, source exclusivity, width-one compatibility, canonical
deduplication and explicit bounds. Measure whether the correct legal candidate
exists, enters proposal support, survives each cap and ranks first. Wider beams
cannot recover candidates excluded before proposal generation. The current
normalized-joint score is a heuristic using mean selected pointer probability,
not a normalized tree likelihood. Report deployable top-1, coherent event oracle
and component oracle separately; component-wise oracle choices may conflict.

## Ordered experiment procedure

1. Close Phase62 using its original frozen source/gates; retain failed or
   missing outcomes. No retrospective relabeling or automatic successor.
2. Repair eligibility and diagnostic definitions with contract regressions;
   quantify affected targets and saturation on existing train/development data.
   This stage needs no fresh strict validation consumption.
3. Build the representability → legal proposal → survival → top-1 diagnostic
   ladder, depth-stratified, on a frozen checkpoint. Use it to choose one
   hypothesis rather than changing preprocessing, losses and search together.
4. Run a matched corrected reconstruction baseline and one justified
   representation or assembly contrast. Use a common full pre-treatment state
   for late interventions, at least three independent seeds when estimating
   small training effects, fixed presentations/compute and preregistered
   nontrivial topology endpoints. Treat three seeds as a practical starting
   point, not a guaranteed power calculation.
5. Lock candidate/ranker on development data, then use a fresh confirmation
   cohort once. Maintain the UID-use registry; reserve source-domain validation
   separately. Estimate uncertainty at event/source level and report training
   variation. Do not use the sealed test to decide the next loss weight.
6. Once a meaningful recursive endpoint responds, run a controlled data/compute
   learning curve and pursue per-level ONNX plus host-search parity, followed
   by the bounded current-revision basf2 pilot.

## Repository and online-operation deliverables

The [storage migration](repository_storage.md) archives74 originals with
checksums and retains58 portable compressed registries. It preserves all raw
study receipts and immutable selections, changes no scientific model behavior,
and does not rewrite history. This is an active-tree reduction, not a promise
of shrinking every existing clone.

The [online deployment procedure](online_study_operations.md) defines a private
Site and ChatGPT Work client backed by one authenticated operations service on
the institute infrastructure. It specifies read/plan/validate/submit tools,
freshness, budget authority, idempotency and receipt handling. The current Pages
dashboard remains a reviewed evidence publication, while the proposed service
supplies live monitoring and bounded execution. No online service was deployed
or new training submitted in this review.

Validation results and remote integration status are recorded in the change's
review/PR and the accompanying verification note; passing software checks
does not remove the scientific limitations above.
