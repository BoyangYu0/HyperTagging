Phase44 recovery and next training decision
===========================================

Hold the dataset at 70,000 events and keep the 81,096-step pretrained
checkpoint. Phase44 does not establish a reliable benefit from earlier
encoder adaptation. Its recovered measurements are diagnostic: both original
jobs failed after training because a nested preregistration seed was stale.
The submitted Phase45 comparison tests stronger late encoder adaptation at a
fixed update count. No model is promoted.

Complete retained-tree and beam checks
--------------------------------------

The expanded evaluator checks all explicit retained roots, including roots
outside the training policy. Full scope checks the retained forest, including
isolated particles, without inventing a missing initial-state root. Half scope
uses explicit B halves when available and labels the explicit-component
fallback when the retained data do not supply a B partition.

Each greedy result and every returned beam candidate receives the same source,
LCAG, mother/PID coverage, representability, and kinematic checks. Coherent
retained-forest agreement requires all components in one candidate and rejects
extra predicted roots in full scope. Singleton particles have source/PID
measurements but do not receive trivial LCAG successes. Training-incompatible
roots remain failed perfect-topology trials rather than disappearing.

The new population is published separately from the original policy-eligible
metrics below, so their historical denominators and gates remain comparable.
All raw retained topology is scored directly, including in the supplemental
checks attached to contracted-diagnostic runs. No truth information influences
candidate generation, pruning, stopping, or model-only rankings.

The expanded primary evaluation checks all 100 events without dropping any
retained unit. Full scope contains 3,057 units: 2,838 isolated leaves, 77
single-source composites, and 142 components with nontrivial LCAG trials.
Half/component scope contains 2,573 units: 2,336 isolated leaves, 77
single-source composites, and 160 nontrivial components. Explicit B halves
are present in 18 events; the other 82 use the labelled component fallback.
There are no source-empty units in this cohort.

.. list-table:: Expanded retained-tree metrics, late versus early adaptation
   :header-rows: 1

   * - Metric
     - Late adaptation
     - Early adaptation
   * - Full LCAG pairs
     - 22/3,348 (0.657%)
     - 21/3,348 (0.627%)
   * - Full topology-aligned mother coverage
     - 22/498 (4.418%)
     - 21/498 (4.217%)
   * - Full perfect nontrivial components
     - 6/142 (4.225%)
     - 7/142 (4.930%)
   * - Half/component LCAG pairs
     - 20/1,869 (1.070%)
     - 22/1,869 (1.177%)
   * - Half/component topology-aligned mother coverage
     - 20/480 (4.167%)
     - 22/480 (4.583%)
   * - Half/component perfect nontrivial components
     - 6/160 (3.750%)
     - 7/160 (4.375%)
   * - Full coherent retained-forest agreement
     - 0/100
     - 0/100
   * - Half/component coherent retained-forest agreement
     - 0/100
     - 0/100

Source recall is 2,882/3,567 versus 2,876/3,567 in full scope and
2,413/3,065 versus 2,403/3,065 in half/component scope. These larger rates
include preservation of isolated inputs and are not hierarchy efficiency.
Conditional mother PID accuracy is 21/22 versus 21/21 in full scope and
19/20 versus 22/22 in half/component scope; the small alignment denominators
must accompany those rates. Of the nontrivial targets, 105/142 full and
115/160 half/component units are representable under the checkpoint policy.
Only 26/100 events have no flagged target incompatibility in either scope;
coherent agreement is also 0/26 within that subset. The other events remain
failed trials in the all-retained comparison rather than being dropped.

Every perfect nontrivial component is a two-leaf, one-mother tree of retained
depth one: six for late adaptation and seven for early adaptation in each
scope. No deeper retained component is perfectly reconstructed.

The strongest observed gap is assembling the correct hierarchy. Mother PID
is usually correct among the few aligned mothers, but their low coverage
prevents extrapolating that conditional accuracy to all mothers. More
pretraining or more examples cannot by itself repair a hard target-policy
incompatibility.

The paired event-cluster bootstrap uses 10,000 resamples. Early-minus-late
LCAG differences have 95% intervals of -0.182 to +0.185 percentage points
(full) and -0.216 to +0.483 points (half/component). Both span zero. This is
conditional on the trained weights and does not measure training-seed
uncertainty. The all-zero coherent-forest result has a degenerate empirical
bootstrap; it is not evidence of zero uncertainty about the population rate.

All 41 returned late-adaptation beam candidates and all 43 early-adaptation
candidates were checked in both scopes on the fixed twenty-event subset.
Every model-only ranker and the coherent diagnostic oracle yields zero
coherent retained-forest successes out of twenty events in both arms/scopes.

For late adaptation, greedy, confidence-sum, average-link, normalized-joint,
and the coherent oracle recover 4/517 full LCAG pairs and 4/89 mothers;
half/component results are 4/300 pairs and 4/87 mothers. Confidence-mean
recovers only three pairs and three mothers in each scope. Perfect nontrivial
components remain 1/30 full and 1/32 half/component for every ranker.
For early adaptation, all rankers and greedy recover the same four pairs and
four mothers, with 2/30 full and 2/32 half/component perfect components.
Thus this bounded search does not improve LCAG or perfect-component recovery
over greedy. Ranked Oracle@4 also finds no coherent event or complete B pair;
its per-unit perfect counts are the same 1-versus-2 components, already found
at rank one. These are truth-after-inference diagnostics, not deployable scores.

What failed and what was recovered
----------------------------------

Both arms reached 4,376 optimizer steps. Late adaptation froze the encoder
for 2,188 steps; early adaptation used zero frozen steps. Their top-level
configuration and
saved replay records used seed 20260911, while the nested balanced-replay
contract still specified 20260910. The post-training exact contract check
therefore failed before full-decay evaluation. Original immutable sources,
checkpoints, and failed receipts are preserved.

The recovery permits exactly this documented seed deviation. It checks
checkpoint finiteness and lineage, fixed validation selection, all 4,376
optimizer records, 280,064 replay slots, level counts, eligible-pool identity,
input hashes, and unchanged source pretraining checkpoint. The planned replay
schedule is independently derived from the fixed budget. Separate recovered
outputs use the original strict evaluator and unchanged acceptance gates.

These are post-hoc diagnostics with a preregistration deviation, not a clean
confirmatory replication or successful original jobs. The dashboard retains
that distinction. A passing replay audit after the documented reconciliation
does not retroactively change the preregistration.

Two supplemental step-1,000 evaluations initially stopped at the encoder
lineage check because they incorrectly asserted a frozen early encoder.
Separate retries used the original audited fine-tuned-encoder setting. The
rejected attempts are preserved; they produced no scores. Successful expanded
reports must reproduce all original scientific metrics exactly before export.

Original selection and policy-eligible reconstruction
-----------------------------------------------------

Both arms selected step 4,376. Late adaptation recovered 185/3,496 micro
complete targets (5.2918%); early adaptation recovered 173/3,496 (4.9485%).
Early minus late is -0.3432 percentage points. This reverses Phase43's small
positive direction, although the seed and validation cohort also changed.
It is not a controlled improvement over prior phases.

Selection uses 1,000 rollout events within a fixed 2,000-event selection
cohort. Strict reconstruction uses a separate 100-event cohort; beam search
uses its fixed 20-event subset. Target efficiency is not full-event success.

.. list-table:: Strict reconstruction on the shared Phase44 cohort
   :header-rows: 1

   * - Metric
     - Late adaptation
     - Early adaptation
   * - Configured full-root constructions
     - 0/100 (0.00%)
     - 0/100 (0.00%)
   * - Full source recall
     - 59/318 (18.55%)
     - 48/318 (15.09%)
   * - Full source precision
     - 59/74 (79.73%)
     - 48/54 (88.89%)
   * - Full LCAG pairs
     - 3/2,910 (0.10%)
     - 4/2,910 (0.14%)
   * - Topology-aligned mothers
     - 3/178 (1.69%)
     - 4/178 (2.25%)
   * - Half source recall
     - 127/619 (20.52%)
     - 131/619 (21.16%)
   * - Half source precision
     - 127/270 (47.04%)
     - 131/262 (50.00%)
   * - Half LCAG pairs
     - 18/1,845 (0.98%)
     - 17/1,845 (0.92%)
   * - Perfect half LCAG units
     - 6/145 (4.14%)
     - 7/145 (4.83%)
   * - Half root PID accuracy
     - 10/145 (6.90%)
     - 13/145 (8.97%)

Neither arm constructed a full root in 100 processed events. Full topology
metrics have only 18 available truth-root units; 82 are unavailable because
the retained truth root is absent, not zero-valued topology measurements.
Only 4/18 available full targets are directly representable under the current
contract; 14 contain an ineligible direct intermediate. Half scope contains
207 units (B halves and explicit continuum components), with 145 available,
62 unavailable, and 102/145 directly representable. Incompatible direct targets
remain in primary evaluation; contraction is diagnostic only. ``complete_only``
describes eligible mother targets, not complete physical events. Local
improvements remain insufficient for promotion.

A paired event-cluster bootstrap (10,000 resamples, seed 20260911) gives a
95% percentile interval of -7.01 to 0.00 percentage points for early-minus-late
full-source recall and -1.97 to +3.17 points for half-source recall. Each draw
keeps both arms and both half units grouped by event, then recomputes ratios
from summed counts. These exploratory intervals condition on the trained
checkpoints, exclude training-seed uncertainty, and do not measure uncertainty
in the separate checkpoint-selection metric. The recovery deviation still applies.

Forest inspection shows 320 accepted mothers, 3,046 leftover input FSPs,
332 empty levels out of 600 opportunities, and 23 predicted B roots for late
adaptation. Early adaptation has 327 mothers, 3,032 leftover FSPs, 321 empty
levels, and 33 predicted B roots. These predicted roots are not correctly
completed full roots. Both reports have 100/100 recursively source-disjoint
accepted-mother events but only 77/100 globally detector-resource-disjoint
forests; these are distinct diagnostics. Closure is exact for 320/320 and
327/327 accepted mothers, respectively.

Checkpoint-selection diagnostics
--------------------------------

.. list-table:: Distinct selected checkpoint states
   :header-rows: 1

   * - Arm / selection
     - Step
     - Configured roots
     - Full LCAG
     - Half LCAG
   * - Late / complete target
     - 4,376
     - 0/100
     - 3/2,910
     - 18/1,845
   * - Late / tree validity
     - 1,000
     - 1/100
     - 1/2,910
     - 8/1,845
   * - Early / complete target
     - 4,376
     - 0/100
     - 4/2,910
     - 17/1,845
   * - Early / depth and tree validity
     - 1,000
     - 0/100
     - 6/2,910
     - 16/1,845

The independent complete-target tracks exactly match the primary reports in
both arms; the late depth track also selects the same final state. The early
step-1,000 alternative improves full LCAG but has half-source recall of
108/619 (17.45%), versus 131/619 for its primary. The late tree-validity
checkpoint also has half-source recall of 108/619. Its one constructed root
occurs in an event with no available retained truth root and cannot establish
an exact full reconstruction. Root construction is a model-state diagnostic,
not a truth-validated success count. Neither alternative is substituted for
the preregistered primary after examining strict validation results.

The original recovered gate reports reject both arms. Late adaptation fails
the nonzero-full-root-construction gate; early adaptation additionally fails
the minimum full-source-recall gate. The expanded retained-tree metrics do not
replace those gates or retroactively change either result.

The verified original export contains 15,434 aggregate and 139,610 detailed
metric entries. The additive retained-tree export contains 19,604
aggregate and 283,866 detailed entries, plus
9,371,165 per-event, per-tree, and per-candidate scalar entries.
Every one of the fourteen expanded reports reproduces all original scientific
results exactly; both repeated primary evaluations are also exact for the
new metrics. The two dashboard downloads preserve every aggregate entry.

Complete metric coverage and interpretation
-------------------------------------------

The :doc:`dashboard <_generated/status/index>` provides the aggregate metric
download, including the five checkpoint records required by the evaluation
protocol, micro/macro training metrics,
seven evaluation reports per arm, exact strict repeats, contracted-topology
diagnostics, calibration, PID confusion, availability, structural metrics,
and both beam scopes with all preregistered rankers. The complete local
CSV/JSON export also retains source-category and target-shape breakdowns.
A separate local history supplement contains 731,626 scalar entries from all
8,840 log records and all 13 saved checkpoint files per arm, including
teacher-forced, edge-F1, periodic, and alias checkpoints. Aliases remain
separate records and are not independent trials. These additional checkpoint
metrics are recorded training diagnostics, not new strict evaluations.

Conditional PID accuracy requires its topology-alignment denominator.
Global forest source disjointness is separate from recursive accepted-mother
validity. Physical mother momentum resolution is unavailable because retained
truth mother four-vectors are absent. Daughter-sum closure is an implementation
invariant, not physical momentum resolution. The original beam table and
retained ``oracle_diagnostic`` select one coherent candidate per scope using
lexicographic topology ranking. The additional ranked Oracle@K download
contains per-unit upper bounds over the model-ranked candidate prefix; its
separate coherent-event field requires all units in one candidate. Those unit
bounds must not be interpreted as one reconstructed event. Every oracle uses
truth only after inference and is not a deployable ranking.

Decision across the studies
---------------------------

The :doc:`Phase41 synthesis <phase41>` covers the historical evidence.
Legacy Stage A metrics cannot be substituted for strict full-decay metrics.
Phase34 did not establish a reliable downstream advantage from longer
pretraining; repaired Phase35 tied frozen and late adaptation with an older
decoder. Phases36-39 improved local metrics without robust full hierarchies.
Phase40 increased dataset size and compute while changing the evaluation
cohort, so it did not isolate a data-volume effect. Phase41's pointer-weight
change traded recall for precision without solving topology. The
:doc:`Phase42 comparison <phase42>` did not support extending pretraining
from 81,096 to 108,128 steps. :doc:`Phase43 <phase43>` found only a four-target
selection advantage from early adaptation; Phase44's recovered selection
counts move in the opposite direction, with the recovery limitation above.

.. list-table:: Primary complete-target counts within each study
   :header-rows: 1

   * - Study
     - Changed factor
     - Control
     - Candidate
   * - Phase41
     - Lower level-one pointer weight
     - 143/3,620
     - 134/3,620
   * - Phase42
     - Later pretrained checkpoint
     - 141/3,938
     - 136/3,938
   * - Phase43
     - Earlier encoder adaptation
     - 136/3,592
     - 140/3,592
   * - Phase44 (diagnostic)
     - Earlier adaptation, new seed/cohort
     - 185/3,496
     - 173/3,496

Compare arms within a row. Different cohorts, denominators, and the Phase44
recovery deviation prevent treating these rows as a controlled learning curve
or pooling them into a single pretraining ranking. No arm was promoted.

Increasing the dataset is not necessary on this evidence. No controlled
learning curve establishes a data-limited regime. Keep 70,000 events now;
revisit size with both equal-compute and equal-exposure comparisons and an
explicit rare-category coverage analysis. This does not prove that more data
cannot help. The replay contract contains 55,323 eligible level-one events
and 1,744 at level six. Its 46,677 level-six slots imply about 26.8
presentations per eligible event on average. This motivates measuring rare-level
coverage and generalization; it does not prove that extra data would help.

Target compatibility and truth-root coverage also limit interpretation. More
pretraining alone cannot repair a hard target-definition incompatibility.
Improving representation learning and hierarchical composition is a more
useful next hypothesis than adding data or simply extending pretraining.
However, these studies do not prove that a new pretraining objective will
help. Task-specific encoder adaptation is distinct from pretraining-objective
improvement. The next bounded test changes its strength while keeping its
start time and update count fixed; downstream topology gates remain the
criterion for success. A positive single-seed result would need replication.
The recovered comparison informs this exploratory allocation because the actual
paired training inputs and replay are auditable; it does not substitute for a
clean confirmatory replication. The original failed jobs remain failed.

Phase45 training contract
-------------------------

Compare encoder learning-rate multipliers 0.05 and 0.10, with both encoders
frozen for the first 2,188 of 4,376 total steps. Both arms retain the same
pretrained checkpoint, 70,000 training events, batch size 64, PID-head freeze,
model architecture, decoder learning rate, thresholds, replay slot budget,
and every strict acceptance gate. This isolates the encoder learning-rate
multiplier within the paired study; it is not a pretraining-duration study.

Seed 20260912 must agree in the training configuration, nested replay contract,
and fresh validation cohort. New preflight checks reject disagreement before
rendering or submitting jobs. The cohort excludes all prior tuning/evaluation
cohorts and training events: 2,000 selection events, 100 separate strict events,
and twenty of those for beam diagnostics. Two guarded one-GPU jobs have
36-hour limits, no requeue, and no automatic promotion or sealed-test access.
Both jobs were submitted and released from one immutable source after 1,589
CPU tests passed (34 skipped) and both two-step training smoke checks passed.
The dashboard records the submission snapshot. These software checks do not
establish scientific benefit; the training results and unchanged gates remain
pending.
