# Assembly diagnosis and next-study policy — 2026-10-07

Five frozen-checkpoint CPU search comparisons and two direct-membership
feasibility fits completed. Their evidence shifts the next-study priority from
local coefficient tuning or reranking to learning global membership and the
representations needed to assemble it. No new model is selected or promoted.

## Population and provenance

The search experiments use the Phase71 `aux_teacher_050` selected checkpoint and
the same already-evaluated 60-collision beam cohort: ten each of charged, mixed,
ccbar, uubar, ddbar and ssbar. This gives 40 nominal B trials and 40 continuum
collisions per condition. All events processed in every condition, without
failures or replacements. Conditions do not multiply the 60 distinct events.
This is exploratory reuse, not independent primary coverage or a rare-tag power
study. Full scope only was evaluated in this diagnostic.

The reference reproduces the entire historical full-scope greedy and beam tag
reports exactly, including channel/type coverage. Tracing is optional and records
only reconstructed states, model outputs and proposals; truth joins happen after
all generation, pruning and ranking. Behavioral tests establish unchanged model
input keys, candidates, scores and counters with tracing enabled.

Compact evidence: [authenticated aggregate](../docs/assembly_diagnosis_20261007/evidence.json).
Private frozen sources, input bindings, traces, event rows, fitted heads and
scheduler receipts reside under the configured data volume's
`assembly_diagnosis_20261007/`. The study uses no sealed test, new validation
reservation, pretraining or GPU. Historical Phase71/72 contracts remain intact.

## Search results

All five conditions have **0/40 exact and 0/40 inclusive B successes**, both at
model-ranked top-1 and across the retained pool. No complete correct B membership
was generated even before local candidate caps in the visited search states.

| Condition | Forest/set width | Candidates/query; proposals/level; subset expansions/query | Generated proposals | Correct level-1 daughter sets / 132 | Correct level-2 daughter sets / 62 | B memberships losing clean-root cover / 40 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Reference | 2 | 2; 8; 32 | 6,018 | 47 | 0 | 21 |
| Width only | 8 | 2; 8; 32 | 16,317 | 47 | 1 | 5 |
| Proposal budget only | 2 | 8; 32; 256 | 17,478 | 57 | 0 | 27 |
| Width and proposals | 8 | 8; 32; 256 | 45,705 | 57 | 1 | 8 |
| Both, lower thresholds | 8 | 8; 32; 256 | 127,206 | 71 | 4 | 6 |

The broader proposal bundle also increases daughter options from four to eight
and cardinality options from two to four. Mother-type options remain three.
The threshold probe changes object/pointer gates jointly from 0.6/0.35 to
0.2/0.1. Width controls both proposal-set and forest retention, so the width arm
is not an isolated final-forest pruning intervention. These are predefined
mechanism probes, not optimized production settings.

Correct daughter sets here require the exact immediate daughter source sets,
but may have the wrong mother type. With the correct type required, the
reference retains 46/132 at generation before local caps, 36/132 after the query
cap and 32/132 after the level cap. Local pruning therefore does lose useful
shallow proposals. Nevertheless, later assembly is already severely deficient:
only 21/62 level-two targets have all required daughters eligible in any
reference state, zero exact level-two proposals are generated, and no level-three
target has its required daughters available. The lower-threshold probe reaches
only one of 43 level-three targets with all required daughters available.
Here `eligible_daughters_present` means presence in parentless context after
the level policy mask, before the separate committed-source alias filter and
joint proposal constraints. It is an optimistic availability count, not a
count of legal daughter combinations.

The clean-root bound asks whether a true membership already exists as a
composite, or can still be covered by current roots containing no foreign FSPs.
It is a **necessary, optimistic no-split condition**, not proof of legal
reachability: ontology, cardinality, charge and detector-source conflicts can
still forbid construction. In the reference, 18 memberships first lose this
cover at level one and three more at level three. Wider search preserves more
potential memberships, yet does not learn how to construct them.

Across the entire pool, unique explicit groups increase from 204 to 778, with
zero correct B groups. All conditions have zero top-1 continuum fake-B events
on this small cohort; that does not establish background equivalence. Search
time ranges from about 47 to 220 seconds on one allocated CPU per condition;
the reference additionally computes the greedy comparison. Timings are not a
controlled performance benchmark.

**Interpretation:** candidate diversity and early commitment both matter, but a
reranker cannot rescue the complete B groups absent from these generated pools.
The data support investigating global/within-branch assembly supervision. They
do not prove that unlimited search, another checkpoint or another architecture
would also fail.

## Direct-membership feasibility

An experimental head consumes frozen, truth-scrubbed detector embeddings and
predicts two unordered optional groups competing with an unassigned-source
class. Its loss matches the two possible B-slot permutations and combines source
classification, set overlap and group presence. Continuum targets contain no B
groups. It produces flat memberships; internal tree reconstruction is not claimed.

The encoder and PID are frozen. The pilot uses 384 training identities selected
by UID hash, 64/category, disjoint from the 60 development collisions. A separate
24-event memorization subset contains eight charged, eight mixed and eight ccbar
training events. Each fit has 1,000 updates at a fixed final checkpoint, with
batch sizes 24 and 16 respectively. The shared seed is 20261007. Neither
validation fitting nor checkpoint/threshold selection is performed.

| Fit and population | Raw exact memberships | Accepted exact memberships | Mean best-group IoU | Continuum accepted events |
| --- | ---: | ---: | ---: | ---: |
| Tiny fit, training | 32/32 | 28/32 | 1.000 | 0/8 |
| Tiny fit, development | 0/40 | 0/40 | 0.274 | 1/40 |
| 384-event fit, training | 0/256 | 0/256 | 0.378 | 0/256 |
| 384-event fit, development | 0/40 | 0/40 | 0.229 | 3/40 |

Acceptance applies presence >=0.5, at least two sources, detector-source
exclusivity and charge in {-1,0,1}. Raw memberships are an explicitly pre-cut
diagnostic, not accepted tags. A post-hoc truth-guard audit finds four of the
32 tiny-training retained source groups have incompatible summed charge and no
detector-source conflicts. Thus raw set learnability and physical-candidate
compatibility must remain separate. Retained incomplete source groups need not
have the full physical B charge; do not silently remove those nominal trials.

Both saved heads were additionally evaluated through the shared tag-efficiency
and channel/type accounting on all 60 development events. Predictions are
explicit flat composite candidates with exact daughter-sum p4 and hard predicted
leaf PID; charge determines the B token, with a fixed B0 convention for neutral
groups because flavour is not predicted. Shared inclusive counts agree with the
pilot, and exact tags remain 0/40. These separately versioned flat candidates do
not follow the production intermediate-level ontology. Complete shared reports
and head hashes are retained privately and bound in the compact evidence.

The tiny result validates basic supervision/optimization feasibility. The larger
pilot does **not** establish a useful representation or generalization benefit;
it does not even fit all its training memberships. This cannot distinguish
head capacity, optimization, frozen representation and sample size. It is not a
controlled comparison to the full Phase71 model, nor evidence against all direct
membership architectures. It blocks promoting this fitted head or proceeding
straight to a hierarchy trained on its predicted groups.

## Policy for subsequent studies

The machine-readable policy is
[`next_study_policy.json`](../configs/reconstruction/next_study_policy.json), with
a concrete [development plan](../configs/reconstruction/structural_membership_development_plan.json).
Before new preregistration, run:

```bash
python scripts/validate_next_reconstruction_study.py \
  --plan configs/reconstruction/structural_membership_development_plan.json
```

1. **Require a measured mechanism.** Preserve the by-generation eligibility,
   threshold support, generated proposals, local caps, surviving states and
   final-pool comparison. Report exact daughters separately from correct type,
   and optimistic membership cover separately from legal reachability.
2. **Prioritize a structural development contrast.** Test direct unordered
   membership with a frozen versus adapted encoder, using the same head,
   objective, initial state, identities and presentation budget. Frozen PID,
   train-fitted normalization and source-exclusive inference remain mandatory.
   This is joint task learning, not another teacher/pointer coefficient sweep.
   Record differing encoder compute rather than claiming equal FLOPs.
3. **Gate larger studies on generalization.** Require >=95% raw exact membership
   on the tiny training check, then a count-backed held-out membership gain over
   the matched control with continuum acceptance and validity accounting.
   Memorization, a lower loss, additional proposals or leaf-dominated micro
   metrics alone cannot pass. This pilot passes only the tiny raw-membership
   check. Reserve no additional primary cohort solely to repeat a zero-signal
   development result.
4. **Make later interventions conditional.** Final-B-pool reranking requires a
   demonstrable correct-pool/top-1 gap. Local proposal scoring is a different
   intervention: the measured loss of correct shallow proposals can motivate it,
   but success must include improved later-level recovery. More search requires measured deep proposal/group
   survival gains. If bottom-up reconstruction remains, test legal first-error
   correction or delayed commitment; an object-presence recovery loss is not
   subtree repair. Add conditional hierarchy only after group quality improves,
   retaining competing memberships rather than forcing one early partition.
5. **Preserve the scientific comparison.** New primary studies still require
   2,000 distinct collisions/category/arm, separate checkpoint selection,
   authenticated fresh reservations, original nominal denominators, exact and
   inclusive tagging, channel/type coverage, continuum fake acceptance, source
   validity and daughter-sum closure. The sealed test stays closed. A flat group
   pilot cannot be relabelled exact topology or physical FEI performance.

Already accepted Phase72 jobs are not cancelled or altered by this policy.
Review their bounded soft-Jaccard result when available, then use these gates
for successors. No automatic training chain or model promotion follows this
diagnosis. The plan validator is a scientific planning check, not a replacement
for authenticated data, frozen runtime or scheduler admission.

## Reproduction and limitations

### Capacity and pretraining policy extension

The subsequent user-requested extension treats capacity and pretraining alignment
as testable hypotheses. The checkpoint has 128-dimensional contextual features
and a separate 32-dimensional hyperbolic projection; reconstruction retains the
wider features. The frozen membership pilot additionally projects to a 64-wide
head. Its memorization result does not prove adequate generalization capacity.

The [capacity/pretraining development plan](../configs/reconstruction/capacity_pretraining_development_plan.json)
defines four arms: contextual width 128/256 crossed with existing/assembly-targeted
pretraining objectives. Initially hold encoder depth, hyperbolic dimension 32,
curvature and tangent-radius contract fixed. Use comparable pretraining histories
at both widths and shared initial states between objective arms within each
width; a historical pretrained 128-wide model versus fresh 256-wide model is not
a clean width comparison. Match data exposure and downstream schedules, control
head bottlenecks, and report parameters, actual compute, memory and convergence.
Equal update counts do not imply equal compute.

Assembly-targeted objectives should supervise unordered global memberships with
an unassigned class, relation-specific within-B parent/sibling discrimination,
and detector-only or model-generated partial states. Current pretraining already
supervises hierarchy and branch relations. Its directed parent-negative mask
excludes same-B relatives; whether more targeted discrimination helps remains
unproven. Do not indiscriminately repel legitimate relatives. Truth is a loss
target only, never an inference input or candidate-selection aid. All four arms
must share the downstream interface that consumes predicted group information
and the encoder-adaptation schedule.

Evaluate held-out raw/accepted membership, within-B discrimination, correct deep
proposal survival, inclusive/exact B top-1 and pool recovery, and continuum/source
validity. Membership gains alone do not establish hierarchy success. Report width
effects, objective effects and their interaction separately. A reviewed downstream
gain is required before proposing 512-wide context or a separate hyperbolic-width
study; the latter must preserve dimension-aware geometric scale controls. Simply
extending unchanged pretraining is not the default next intervention.

This plan complements the existing frozen-versus-adapted membership diagnostic;
neither plan launches the other. The validator checks declared factorial arms,
controls and endpoints, not their actual execution. New runtime admission and
scientific evidence remain necessary. No capacity/pretraining experiment has run.

```bash
python scripts/validate_next_reconstruction_study.py \
  --plan configs/reconstruction/capacity_pretraining_development_plan.json
```

### Frozen experiment reproduction

`scripts/diagnose_search_survival.py prepare` authenticates the existing input
bindings, copies the instrumented source to a new data-volume directory and
records five immutable conditions. The CPU Slurm wrapper executes only that
frozen source. `scripts/probe_direct_membership.py` uses its separately frozen
source and stores the selected training identities and cached detached features
privately. `scripts/export_assembly_diagnosis.py` checks completed receipts and
file hashes and requires historical greedy/beam tag-report equality before
exporting the compact evidence. Initial raw diagnostic `max_*` counter fields
were sums of event maxima; the exporter omits those misleading fields, and the
runner now uses maximum aggregation for future runs. Scientific counts and
predictions are unaffected.

No coefficient sweep, additional GPU study, primary cohort reservation, sealed
test access or production decoder change was made. The deployed search is
unchanged when tracing is disabled. These results inform study allocation;
they do not establish zero population efficiency or a winning replacement.

## Verification

All five search jobs, both membership fits and the shared tag-evaluation job
completed successfully. The final focused trace, membership and policy checks
passed (13 tests), as did the relevant documentation, evaluation and audit
checks (156 tests). Ruff and `git diff --check` passed, and the next-study plan
validator accepts the evidence-bound development plan.

The broader CPU regression run recorded 1,986 passed, 34 skipped, 348 deselected
and one failure: the existing repromotion preflight test encountered
`EINVAL` from `renameat2(RENAME_NOREPLACE)` on the project filesystem. Rerunning
its entire module using local temporary storage passed (15 passed, two skipped),
without changing that implementation. The original failed run remains recorded;
the HTML publication build was not run.

## Terminal follow-up and Phase74 feasibility — 2026-10-08

Phase73 completed: frozen and adapted arms each recovered32/32 raw tiny
memberships, but0/40 raw and accepted held-out memberships. Their main-train
raw counts were2/256 and30/256. The held-out gate failed; the sparse main-training
fit does not prove irrecoverably missing encoder information or data scarcity.
The capacity/pretraining factorial remains a falsifiable development proposal.

The new request excludes all historical primary and checkpoint-selection cohorts.
Authenticated inspection found that all60 designated membership-development
events are Phase71 primary identities. Recent beam alternatives are also primary
subsets, while the32 geometry-development identities are Phase63 selection
events. No eligible replacement was found among these designated candidates.
Phase74 training is blocked before runtime implementation and submission; passing
the scientific-plan validator does not override cohort admission. Historical
receipts and Phase73's explicit exploratory-reuse interpretation remain unchanged.
See the [feasibility review](phase74_feasibility_20261008.md).

## Follow-up through Phase74 — 2026-10-09

Phase73 terminal tiny raw32/32 did not generalize (both0/40 heldout). Phase74
matched fresh capacity/pretraining histories and the common downstream interface:
all tiny raw32/32, main train0,0,0,1/1024 and heldout0,0,0,1/400, with continuum
acceptance31,24,52,72/400. The lone positive is a three-FSP retained membership,
not a B tree. No primary gate or width/objective winner is established. A measured
training-only gradient conflict motivates one controlled development projection
contrast, not an arbitrary loss-weight grid. See the Phase74 wiki review and
phase75_gradient_development_plan.json. The authenticated historical evidence
JSON and policy hash remain unchanged.
