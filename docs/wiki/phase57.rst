Phase57 reconstruction and pretraining review
=============================================

**Phase57 cannot establish an independent validation effect.** Both arms
completed training, but every one of the 100 reserved strict events was used
for checkpoint selection. The original jobs failed the ordered-cohort audit.
Their receipts remain failed; recovered reconstruction metrics are explicitly
**selection-contaminated diagnostics**. No model is promoted.

Review date: 23 September 2026. The :doc:`dashboard <_generated/status/index>`
provides original-policy and retained full/half metrics, all returned beam
candidates and rankings, micro/macro and exact-combination counts, and
pretraining execution metrics. Complete exports retain event/tree scalars and
training histories locally. Diagnostic completeness does not repair independence.

Training execution and cohort failure
-------------------------------------

Both arms completed 2,188 pretraining refinement steps (70,000 presentations)
and 4,376 reconstruction steps. Parent-ranking weight is 2 in both; late
leaf-PID weights are 0.4 versus 0.2. The models are tensor-identical at step
1,094, before that contrast begins. Both start from the original step-81096
parameters with fresh train-only normalization and fresh optimizer, schedule,
RNG and memory. Transferred PID stays frozen during reconstruction; encoder
adaptation starts after step 2,188 at multiplier 0.05. Objective-dominance
threshold 20 and its fail action remain unchanged. Across 553 recorded
objective-preflight rows per arm, the largest weighted dominance ratio is
18.2054 for the control and 10.3093 for lower late PID. These execution diagnostics
show more observed headroom in this run; they do not establish downstream quality.

Phase57 intended to reuse the exact Phase56 selection sequence of 1,000 events.
Pretraining used the correct explicit validation sequence. During reconstruction,
passing historical exclusions did not enforce that sequence: the
trainer selected again with the new seed. Historical exclusions left all 391
unused events eligible, including the newly reserved strict cohort. Both arms
selected the same actual
1,000 events, comprising 734 intended events and 266 unintended events.
All 100 reserved strict events are among those 266. Another 166 previously
untouched events were also consumed. Only 125 validation events remain untouched.
There is no overlap with older forbidden historical cohorts or the sealed test.

The ordered-cohort audit correctly rejected the result after training and before
evaluation. Diagnostic recovery authenticates the original contracts, receipts,
refined checkpoint hashes, actual selection identities, all checkpoint tracks,
finite tensors, replay execution and encoder lineage. It retains the failure
rather than rewriting an original result or labeling the audit as passed.
The evaluation code, cohorts, models, scoring policies and search budgets are
unchanged. No remaining untouched events were spent on a post-hoc Phase57 test.

Both primary and independent complete-target checkpoints select step 4,376;
depth selects step 2,000 and validity step 1,000. Selection source-set plus
mother-PID recovery is 289/3,533 for the control and 283/3,533 for lower late PID.
This is an in-sample checkpoint-selection statistic, not recursive-tree efficiency
or an independent treatment effect.

Pretraining selection diagnostics
----------------------------------

Final step-2,188 metrics use the reused selection cohort. They describe the
refinement and are not independent reconstruction evidence. Loss values have
their recorded objective semantics; a lower weighted total with a changed PID
weight would not by itself mean better representation quality.

.. list-table::
   :header-rows: 1

   * - Recorded metric
     - Late PID 0.4
     - Late PID 0.2
   * - validation_leaf_pid_accuracy
     - 0.88791743
     - 0.88647266
   * - validation_parent_ranking_accuracy
     - 0.72594094
     - 0.7257576
   * - validation_relation_accuracy
     - 0.84373763
     - 0.84338562
   * - validation_fsp_only_relation_accuracy_separate
     - 0.87155926
     - 0.87092811
   * - validation_effective_rank
     - 24.279286
     - 24.246325
   * - validation_loss_parent
     - 0.0048531287
     - 0.0048570805
   * - validation_loss_lca
     - 1.3460693
     - 1.3873291

Primary full and half diagnostics
----------------------------------

Control versus lower late PID has 9 versus 11 exact nontrivial retained
components in each scope; all these primary successes have depth one and two
leaves. Full retained LCAG increases 28 to 29 correct pairs, while half LCAG
falls 33 to 32. Neither arm has a coherent retained-forest success in 100 events.
These mixed in-sample differences cannot establish a pretraining benefit.
Original-policy constructed full roots are 3/100 versus 0/100, and full-source
recall is 102/408 versus 93/408. A constructed root is not an exact decay tree.

Original-policy full scope:

.. list-table::
   :header-rows: 1

   * - Diagnostic
     - Late PID 0.4
     - Late PID 0.2
   * - source_precision
     - 102/117
     - 93/101
   * - source_recall
     - 102/408
     - 93/408
   * - lcag_pair_accuracy
     - 7/3746
     - 4/3746
   * - perfect_lcag
     - 0/22
     - 0/22
   * - mother_pid_coverage
     - 7/231
     - 4/231
   * - root_pid_accuracy
     - 3/22
     - 0/22

Retained full scope (all retained roots):

.. list-table::
   :header-rows: 1

   * - Diagnostic
     - Late PID 0.4
     - Late PID 0.2
   * - source_precision
     - 3003/3309
     - 2989/3304
   * - source_recall
     - 3003/3683
     - 2989/3683
   * - lcag_pair_accuracy
     - 28/4102
     - 29/4102
   * - perfect_lcag
     - 9/136
     - 11/136
   * - coherent_retained_forest
     - 0/100
     - 0/100
   * - mother_pid_coverage
     - 28/556
     - 29/556
   * - mother_pid_accuracy
     - 28/28
     - 29/29
   * - root_pid_accuracy
     - 388/1668
     - 387/1668
   * - target_representable
     - 2979/3104
     - 2979/3104

Original-policy half scope:

.. list-table::
   :header-rows: 1

   * - Diagnostic
     - Late PID 0.4
     - Late PID 0.2
   * - source_precision
     - 195/330
     - 188/342
   * - source_recall
     - 195/689
     - 188/689
   * - lcag_pair_accuracy
     - 32/2258
     - 29/2258
   * - perfect_lcag
     - 9/146
     - 11/146
   * - mother_pid_coverage
     - 31/355
     - 28/355
   * - root_pid_accuracy
     - 25/146
     - 23/146

Retained half scope (all retained roots):

.. list-table::
   :header-rows: 1

   * - Diagnostic
     - Late PID 0.4
     - Late PID 0.2
   * - source_precision
     - 2443/2726
     - 2422/2730
   * - source_recall
     - 2443/3097
     - 2422/3097
   * - lcag_pair_accuracy
     - 33/2275
     - 32/2275
   * - perfect_lcag
     - 9/158
     - 11/158
   * - coherent_retained_forest
     - 0/100
     - 0/100
   * - mother_pid_coverage
     - 32/534
     - 31/534
   * - mother_pid_accuracy
     - 32/32
     - 31/31
   * - root_pid_accuracy
     - 332/1408
     - 331/1408
   * - target_representable
     - 2407/2540
     - 2407/2540

Interpretation of reconstruction diagnostics
--------------------------------------------

All full and half retained roots must be checked, including incompatible targets
and every returned coherent beam candidate. Direct incompatibilities remain
failed trials. Isolated leaves do not earn trivial exact nontrivial LCAG
successes. Half scope uses B partitions where available and explicit retained
components otherwise. Counts are aggregated before division; micro, macro and
exact source/topology/PID combinations remain separate.

Beam model-only top-1 rankings remain distinct from oracle diagnostics, which
consult truth only after generation. Repeated scopes and checkpoint views are
not independent successes. Coherent retained-forest topology does not imply
correct PID labels. Daughter-sum p4 closure is an implementation invariant,
not physical momentum resolution. High retained source coverage can reflect
preserved inputs rather than correct mother assembly. The full reference
contains 2,886 isolated leaves, 82 single-source composites and 136 nontrivial
topology units; only 93 of those nontrivial units are policy-representable.
Half scope contains 2,300 isolated leaves, 82 single-source composites and 158
nontrivial units, of which 107 are representable. Incompatible units remain
failed primary trials, not removed denominators. Only 22/100 events have no
flagged target incompatibility. More examples or better pretraining cannot
by themselves remove incompatibilities imposed by an unchanged hard target
policy; assembly quality and target representability remain separate concerns.

Numerical gate checks describe the recovered diagnostics only. Phase57 cannot
pass independent validation gates even if every numerical threshold is met.
Paired event-bootstrap intervals use 10,000 resamples and ratios of summed
counts; they are conditional in-sample descriptions and do not remove selection
bias or quantify training-seed uncertainty.

Remaining checkpoint and beam diagnostics
-----------------------------------------

All seven views per arm and every returned beam candidate have been checked
in both scopes. Counts below are selection-contaminated diagnostics. Beam uses
the first 20 events; other views use 100. Contracted topology changes the
original-policy reference and remains a separate diagnostic. The retained-direct
reference stays unchanged. Repeats and views are not additional independent trials.

.. list-table::
   :header-rows: 1

   * - View
     - Scope
     - 0.4 exact
     - 0.2 exact
     - 0.4 forest
     - 0.2 forest
   * - independent_complete_target_direct
     - full
     - 9/136
     - 11/136
     - 0/100
     - 0/100
   * - independent_complete_target_direct
     - half
     - 9/158
     - 11/158
     - 0/100
     - 0/100
   * - independent_depth_direct
     - full
     - 7/136
     - 4/136
     - 0/100
     - 0/100
   * - independent_depth_direct
     - half
     - 7/158
     - 4/158
     - 0/100
     - 0/100
   * - independent_tree_validity_direct
     - full
     - 5/136
     - 6/136
     - 0/100
     - 0/100
   * - independent_tree_validity_direct
     - half
     - 5/158
     - 6/158
     - 0/100
     - 0/100
   * - primary_complete_target_beam_direct
     - full
     - 2/20
     - 3/20
     - 0/20
     - 0/20
   * - primary_complete_target_beam_direct
     - half
     - 2/22
     - 3/22
     - 0/20
     - 0/20
   * - primary_complete_target_contracted_diagnostic
     - full
     - 9/136
     - 11/136
     - 0/100
     - 0/100
   * - primary_complete_target_contracted_diagnostic
     - half
     - 9/158
     - 11/158
     - 0/100
     - 0/100
   * - primary_complete_target_direct
     - full
     - 9/136
     - 11/136
     - 0/100
     - 0/100
   * - primary_complete_target_direct
     - half
     - 9/158
     - 11/158
     - 0/100
     - 0/100
   * - primary_complete_target_repeat2_direct
     - full
     - 9/136
     - 11/136
     - 0/100
     - 0/100
   * - primary_complete_target_repeat2_direct
     - half
     - 9/158
     - 11/158
     - 0/100
     - 0/100

The control returns 52 coherent beam candidates and lower late PID returns
53; all receive full/half checks. Model-only rankings and post-generation
oracle diagnostics are separate. Oracle values do not describe a deployable
selection rule.

Under normalized-joint top-1 ranking, both arms recover two exact nontrivial
components: 2/20 full and 2/22 half. Oracle@4 reaches four components for the
control and three for lower late PID in each scope, while coherent event
forests remain 0/20 even for this oracle. Extra candidates provide some local
alternatives, without establishing complete-event reconstruction.

Retained full beam ranking diagnostics:

.. list-table::
   :header-rows: 1

   * - Late PID
     - Ranking
     - LCAG
     - Mother coverage
     - Exact component
     - Coherent forest
   * - 0.4
     - average_link_probability
     - 4/305
     - 4/71
     - 3/20
     - 0/20
   * - 0.4
     - learned_confidence_mean
     - 4/305
     - 4/71
     - 2/20
     - 0/20
   * - 0.4
     - learned_confidence_sum
     - 3/305
     - 3/71
     - 2/20
     - 0/20
   * - 0.4
     - normalized_joint_log_probability
     - 4/305
     - 4/71
     - 2/20
     - 0/20
   * - 0.4
     - oracle_diagnostic
     - 5/305
     - 5/71
     - 4/20
     - 0/20
   * - 0.2
     - average_link_probability
     - 4/305
     - 4/71
     - 2/20
     - 0/20
   * - 0.2
     - learned_confidence_mean
     - 4/305
     - 4/71
     - 2/20
     - 0/20
   * - 0.2
     - learned_confidence_sum
     - 5/305
     - 5/71
     - 2/20
     - 0/20
   * - 0.2
     - normalized_joint_log_probability
     - 5/305
     - 5/71
     - 2/20
     - 0/20
   * - 0.2
     - oracle_diagnostic
     - 7/305
     - 7/71
     - 3/20
     - 0/20

Retained half beam ranking diagnostics:

.. list-table::
   :header-rows: 1

   * - Late PID
     - Ranking
     - LCAG
     - Mother coverage
     - Exact component
     - Coherent forest
   * - 0.4
     - average_link_probability
     - 6/185
     - 6/69
     - 3/22
     - 0/20
   * - 0.4
     - learned_confidence_mean
     - 5/185
     - 5/69
     - 2/22
     - 0/20
   * - 0.4
     - learned_confidence_sum
     - 5/185
     - 5/69
     - 2/22
     - 0/20
   * - 0.4
     - normalized_joint_log_probability
     - 5/185
     - 5/69
     - 2/22
     - 0/20
   * - 0.4
     - oracle_diagnostic
     - 6/185
     - 6/69
     - 4/22
     - 0/20
   * - 0.2
     - average_link_probability
     - 5/185
     - 5/69
     - 2/22
     - 0/20
   * - 0.2
     - learned_confidence_mean
     - 6/185
     - 6/69
     - 2/22
     - 0/20
   * - 0.2
     - learned_confidence_sum
     - 6/185
     - 6/69
     - 2/22
     - 0/20
   * - 0.2
     - normalized_joint_log_probability
     - 5/185
     - 5/69
     - 2/22
     - 0/20
   * - 0.2
     - oracle_diagnostic
     - 7/185
     - 7/69
     - 3/22
     - 0/20

The complete depth audit finds 0 exact-component records
deeper than one generation across views and candidates. Counts include repeated
views and scopes. The all-view forest audit identifies
0 distinct events with a coherent retained-forest result,
of which 0 contain nontrivial truth topology.
Forest topology coherence does not require all PID labels to be correct.

Numerical checks:

.. list-table::
   :header-rows: 1

   * - Check
     - Late PID 0.4
     - Late PID 0.2
   * - minimum_complete_target_efficiency
     - True
     - True
   * - minimum_depth_fraction
     - True
     - True
   * - minimum_full_source_precision
     - True
     - True
   * - minimum_full_source_recall
     - True
     - True
   * - minimum_half_lcag
     - True
     - True
   * - minimum_half_perfect_lcag
     - True
     - True
   * - minimum_half_root_pid_accuracy
     - True
     - True
   * - minimum_half_source_precision
     - True
     - True
   * - minimum_half_source_recall
     - True
     - True
   * - nonzero_exact_mother_coverage
     - True
     - True
   * - nonzero_full_lcag
     - True
     - True
   * - nonzero_full_root_completion
     - True
     - False
   * - primary_repeat_identical
     - True
     - True
   * - structural_guardrails
     - True
     - True

Independent gate certification remains **false for both arms**, regardless of
these numerical thresholds, because the reserved events entered selection.

Conditional paired descriptions (percentage points, candidate minus control):

.. list-table::
   :header-rows: 1

   * - Scope
     - Retained metric
     - Difference
     - 95% bootstrap interval
   * - full
     - lcag_pair_accuracy
     - +0.024
     - [-0.147, +0.215]
   * - full
     - mother_pid_coverage
     - +0.180
     - [-1.078, +1.468]
   * - full
     - source_precision
     - -0.286
     - [-1.337, +0.794]
   * - full
     - source_recall
     - -0.380
     - [-1.259, +0.507]
   * - half
     - lcag_pair_accuracy
     - -0.044
     - [-0.314, +0.222]
   * - half
     - mother_pid_coverage
     - -0.187
     - [-1.318, +0.888]
   * - half
     - source_precision
     - -0.901
     - [-2.051, +0.256]
   * - half
     - source_recall
     - -0.678
     - [-1.622, +0.283]

These intervals describe resampling the contaminated selection cohort. They
do not repair independence, establish a causal generalization effect, or
quantify training-seed uncertainty.

Dataset size versus pretraining
-------------------------------

**Keep 70,000 training events for the next bounded comparison.** Phase40 changed
size, compute and cohort together; no controlled learning curve establishes an
immediate benefit from training-data growth. Phases34/42 did not establish that
longer pretraining alone helps reconstruction. The evidence does not establish
that improving pretraining is more beneficial than adding training data.

Phase41's reduced pointer weight worsened reconstruction. Phase43 adaptation
gains were not independently replicated by Phase44's seed-deviating diagnostic.
Phase45's larger encoder update tied; corrected Phases46/47 were mixed.
Phase48's intended PID update did not execute because of the autocast cache
defect; corrected Phases49–51 did not establish a stable benefit. Recovery
weights in Phases52–54 failed to yield reproducible joint topology/forest
improvement, ending recovery-dose tuning. Phase55's parent-weight effects were
small, shallow and mixed, with runtime sensitivity. Phase56's failed control
prevented replication. Phase57's cohort contamination prevents an independent
late-PID comparison. Cross-phase rates use different seeds/cohorts and are not
causal contrasts.

The next useful pretraining study repeats the intended objective-balance test
with enforced selection eligibility. It does not choose a new dose from the
contaminated scores. Joint source recovery, nontrivial exact topology, coherent
forests and the original gates matter; lower pretraining loss alone is insufficient.

**Expanding independent validation data is now a priority.** This is separate
from training-set growth. Phase58 reserves 100 of the 125 untouched validation
events, leaving only 25. Another full 100-event independent study would require
additional validation capacity from events not used for pretraining, training
or model selection. Merely moving previously seen training events into a new
validation manifest would not create independent evidence. The sealed test
remains closed.

Phase58: corrected selection and one bounded pair
-------------------------------------------------

Repeat late leaf-PID phase weights [1, 1, 0.4, 0.4] versus [1, 1, 0.2, 0.2],
with parent weight 2, seed 20260925, 2,188 refinement steps and 4,376 reconstruction
steps. Keep original parameter initialization, 70,000 training events, recovery
weight 2, frozen PID, encoder schedule and all objective guards matched.

Reuse the exact Phase56 selection **set** of 1,000 events, reordered by the
trainer's UID ranking under the new seed. Exclude **all other 49,000 validation
UIDs**, including strict events, from reconstruction selection. Before any
training, run the actual trainer selector over an authenticated full validation
UID census and require the exact preregistered sequence and zero strict overlap.
Retain the post-training ordered audit as a second check. Pretraining uses the
same explicit selection sequence. Strict scoring uses 100 fresh events and beam
its first 20. Selection reuse is intentional and is not independent cross-phase
selection replication.

Each guarded job requests one H100 NVL, eight CPUs, 64 GiB and 36 hours without
requeue. Authorization covers this one pair; no automatic campaign chain,
model promotion or sealed-test access. The dashboard records a submission
snapshot rather than a live scheduler feed.

Provenance
----------

Original training source is ``55abc338060088fb9f58be15f38cace54654a652``.
The diagnostic evaluator is frozen at
``f52ee4b79e2ade40b9149d63789c90799311f739``. Its model and data inputs
are hash-bound to the original contracts and retained checkpoints. Original
failed receipts are preserved. Private event identities and filesystem paths
remain local; Pages publishes allowlisted aggregates and source hashes.

Metric coverage
---------------

The exports contain 15,522 original aggregate scalars,
19,671 retained aggregate scalars,
297,204 detailed retained scalars and
9,554,795 tree/candidate scalars. There are
5,040 micro/macro and exact-combination supplement scalars,
749,269 reconstruction-history scalars and
1,318,918 pretraining-history scalars. Four dashboard downloads
provide the public aggregates; the local complete-metrics archive retains
native reports, failed receipts, full exports, source bindings and verification
hashes. Physical mother momentum resolution remains unavailable.
