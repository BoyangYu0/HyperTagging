Phase55 reconstruction and pretraining review
=============================================

The parent-ranking contrast is mixed. Stronger pretraining increases primary
source-set plus mother-PID recovery from **265/3,548 to 271/3,548**, and retained
exact nontrivial components from **5 to 6**. Coherent retained forests fall from
**1/100 to 0/100**. Control misses full-source recall; the candidate misses
full-root construction. Neither arm passes every original gate. No promotion
or sealed-test access is authorized.

Review date: 21 September 2026. The :doc:`dashboard <_generated/status/index>`
contains all original, retained, micro/macro, exact-combination and beam-ranking
metrics, plus pretraining execution and validation downloads. Counts below
compare weight 1 control with weight 2 candidate on the same populations.

Evaluation recovery and provenance
----------------------------------

Both original Slurm jobs failed **after training completed**.
Each completed 2,188 pretraining steps and 4,376 reconstruction steps. The
post-training evaluator supplied the original step-81096 checkpoint rather
than the arm-specific refined step-2188 checkpoint. The frozen early
reconstruction track correctly failed encoder equality against the wrong source.

The repair authenticates the refined checkpoint checksum and original-source
binding before evaluation. All seven views per arm were regenerated on CPU
from an immutable repaired evaluator checkout. Early frozen tracks retain exact
encoder equality; late adapted tracks retain the preregistered compatibility
checks. All reconstruction checkpoint configurations name their actual refined
source. Models, checkpoint selection, cohorts, thresholds, search and gates
were unchanged. Original failed receipts and partial reports are preserved;
they are not relabelled successful jobs. Recovered reports control this review.

Strict primary repeats are identical. Original-primary scientific results are
compared with the recovered reports, separately from corrected lineage metadata.
Two control and four candidate event records differ between original-node and
recovery-host execution. The candidate retained full LCAG count changes from
28 to 31, and half from 37 to 40; exact components remain 6. Checkpoint bytes,
inference/scoring source, event order and scoring settings match. Floating-point
runtime differences are plausible but not established as the sole cause. Do
not attribute these differences to the checkpoint metadata repair. Recovered
same-host results control the tables, and the complete variation audit is
preserved locally. This sensitivity further limits small-count claims.

Pretraining execution
---------------------

Both arms started from the same 129 parameters of encoder81096, with fresh
train-only normalization, optimizer, schedule, RNG and channel memory. Only
parent-ranking weight differs, 1 versus 2. Each processed exactly **70,000**
training events: 2,187 batches of 32 and a final batch of 16. The original
70,016 field described nominal capacity, not actual presentations.

The objective had nonzero losses and gradients in its active phases. Configurations,
normalizer tensors, feature contracts, architecture, validation selections and
data order verify the intended single-factor contrast. Both
fixed final refined checkpoints are distinct; reconstruction transfers each
arm's own checkpoint and keeps its PID head frozen. This is an objective-weight
comparison, **not refinement versus no refinement**. Differently weighted total
losses alone cannot establish better representations or reconstruction.

.. list-table::
   :header-rows: 1

   * - Recorded final pretraining metric
     - Weight 1
     - Weight 2
   * - validation_parent_ranking_accuracy
     - 0.724321
     - 0.726125
   * - validation_loss_parent
     - 0.00543924
     - 0.00485159
   * - validation_relation_accuracy
     - 0.838698
     - 0.839105
   * - validation_leaf_pid_accuracy
     - 0.886219
     - 0.886297
   * - validation_effective_rank
     - 23.7262
     - 23.9535

These retain the trainer’s validation aggregation semantics, including batch means; they are not newly computed global micro rates. Validation uses the same 1,000 selection events and four views.

Original strict reconstruction
------------------------------

Both arms selected step 4,376. Selection used 2,000 validation events, including 1,000 rollout events. Strict evaluation used 100 disjoint events; beam used a 20-event subset. Primary recovery is not recursive topology exactness. A constructed B root is not necessarily a correct tree.

.. list-table::
   :header-rows: 1

   * - Metric
     - Weight 1
     - Weight 2
   * - exact_mother_coverage
     - 5/242
     - 9/242
   * - full_lcag
     - 5/4197
     - 10/4197
   * - full_root_completion
     - 5/100
     - 0/100
   * - full_source_precision
     - 71/84
     - 86/103
   * - full_source_recall
     - 71/430
     - 86/430
   * - half_lcag
     - 30/2567
     - 36/2567
   * - half_perfect_lcag
     - 5/183
     - 6/183
   * - half_root_pid_accuracy
     - 30/183
     - 28/183
   * - half_source_precision
     - 186/354
     - 199/359
   * - half_source_recall
     - 186/809
     - 199/809

All retained full and half trees
--------------------------------

Every retained root is checked. Incompatible targets remain failed primary trials. Continuum events use explicit components, not invented B hemispheres. Isolated leaves do not earn trivial LCAG successes. High source coverage is dominated by retained detector inputs and is not hierarchy efficiency.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Weight 1
     - Weight 2
   * - full
     - lcag_pair_accuracy
     - 23/4700
     - 31/4700
   * - full
     - mother_pid_coverage
     - 22/607
     - 28/607
   * - full
     - mother_pid_accuracy
     - 21/22
     - 26/28
   * - full
     - source_precision
     - 2829/3149
     - 2862/3166
   * - full
     - source_recall
     - 2829/3631
     - 2862/3631
   * - full
     - perfect_lcag
     - 5/171
     - 6/171
   * - full
     - coherent_retained_forest
     - 1/100
     - 0/100
   * - full
     - root_pid_accuracy
     - 335/1615
     - 348/1615
   * - full
     - target_representable
     - 2842/2960
     - 2842/2960
   * - half
     - lcag_pair_accuracy
     - 32/2619
     - 40/2619
   * - half
     - mother_pid_coverage
     - 30/584
     - 37/584
   * - half
     - mother_pid_accuracy
     - 29/30
     - 35/37
   * - half
     - source_precision
     - 2280/2566
     - 2316/2590
   * - half
     - source_recall
     - 2280/3046
     - 2316/3046
   * - half
     - perfect_lcag
     - 5/194
     - 6/194
   * - half
     - coherent_retained_forest
     - 1/100
     - 0/100
   * - half
     - root_pid_accuracy
     - 271/1352
     - 289/1352
   * - half
     - target_representable
     - 2272/2398
     - 2272/2398

.. list-table::
   :header-rows: 1

   * - Reference population
     - Full
     - Half/component
   * - events_without_flagged_target_incompatibility
     - 18
     - 18
   * - isolated_leaf_units
     - 2729
     - 2144
   * - nontrivial_topology_units
     - 171
     - 194
   * - representable_nontrivial_units
     - 113
     - 128
   * - single_source_composite_units
     - 60
     - 60
   * - source_empty_units
     - 0
     - 0

All checkpoint and diagnostic views
-----------------------------------

The following retained greedy metrics check every checkpoint view. Repeat and checkpoint-alias rows are reproducibility checks, not independent observations. The contracted view is diagnostic; the beam view here shows its 20-event greedy baseline, with every returned candidate reported separately below.

.. list-table::
   :header-rows: 1

   * - Arm
     - View
     - Full LCAG
     - Full exact component
     - Half LCAG
     - Half exact component
   * - late_adaptation_control
     - independent_complete_target_direct
     - 23/4700
     - 5/171
     - 32/2619
     - 5/194
   * - late_adaptation_control
     - independent_depth_direct
     - 29/4700
     - 8/171
     - 31/2619
     - 8/194
   * - late_adaptation_control
     - independent_tree_validity_direct
     - 29/4700
     - 8/171
     - 31/2619
     - 8/194
   * - late_adaptation_control
     - primary_complete_target_beam_direct
     - 3/836
     - 0/36
     - 5/494
     - 0/39
   * - late_adaptation_control
     - primary_complete_target_contracted_diagnostic
     - 23/4700
     - 5/171
     - 32/2619
     - 5/194
   * - late_adaptation_control
     - primary_complete_target_direct
     - 23/4700
     - 5/171
     - 32/2619
     - 5/194
   * - late_adaptation_control
     - primary_complete_target_repeat2_direct
     - 23/4700
     - 5/171
     - 32/2619
     - 5/194
   * - stronger_parent_pretraining
     - independent_complete_target_direct
     - 31/4700
     - 6/171
     - 40/2619
     - 6/194
   * - stronger_parent_pretraining
     - independent_depth_direct
     - 26/4700
     - 8/171
     - 30/2619
     - 8/194
   * - stronger_parent_pretraining
     - independent_tree_validity_direct
     - 26/4700
     - 8/171
     - 30/2619
     - 8/194
   * - stronger_parent_pretraining
     - primary_complete_target_beam_direct
     - 7/836
     - 2/36
     - 9/494
     - 2/39
   * - stronger_parent_pretraining
     - primary_complete_target_contracted_diagnostic
     - 31/4700
     - 6/171
     - 40/2619
     - 6/194
   * - stronger_parent_pretraining
     - primary_complete_target_direct
     - 31/4700
     - 6/171
     - 40/2619
     - 6/194
   * - stronger_parent_pretraining
     - primary_complete_target_repeat2_direct
     - 31/4700
     - 6/171
     - 40/2619
     - 6/194

Exactness, depth and distinct events
------------------------------------

All fourteen views and every returned candidate were scanned for exact-component depth. There are 0 exact records with depth greater than one across repeated views/scopes; these records are not independent successes. In the primary reports there are 0 distinct events with a deeper exact component.

.. list-table::
   :header-rows: 1

   * - Arm
     - Category
     - Scope
     - Units
     - Nontrivial units
     - Mothers
   * - late_adaptation_control
     - taupair
     - full
     - 17
     - 0
     - 0
   * - late_adaptation_control
     - taupair
     - half
     - 17
     - 0
     - 0

Coherent forest exactness does not require all PID labels to be correct. Repeated full/half records are the same event, not additional successes. No B-pair success should be inferred from a continuum component.

Beam search
-----------

All **128 returned candidates** are checked in both full and half scope. Width-four model-only rankings, the coherent post-inference oracle, and per-unit Oracle@K bounds remain separate. An oracle bound does not describe one deployable hypothesis. No tree is skipped because it is a beam candidate.

.. list-table::
   :header-rows: 1

   * - Arm
     - Scope
     - Ranking
     - LCAG
     - Exact component
     - Coherent forest
   * - late_adaptation_control
     - full
     - average_link_probability
     - 3/836
     - 1/36
     - 0/20
   * - late_adaptation_control
     - full
     - learned_confidence_mean
     - 3/836
     - 0/36
     - 0/20
   * - late_adaptation_control
     - full
     - learned_confidence_sum
     - 3/836
     - 0/36
     - 0/20
   * - late_adaptation_control
     - full
     - normalized_joint_log_probability
     - 3/836
     - 1/36
     - 0/20
   * - late_adaptation_control
     - full
     - oracle_diagnostic
     - 5/836
     - 2/36
     - 0/20
   * - late_adaptation_control
     - half
     - average_link_probability
     - 5/494
     - 1/39
     - 0/20
   * - late_adaptation_control
     - half
     - learned_confidence_mean
     - 5/494
     - 0/39
     - 0/20
   * - late_adaptation_control
     - half
     - learned_confidence_sum
     - 5/494
     - 0/39
     - 0/20
   * - late_adaptation_control
     - half
     - normalized_joint_log_probability
     - 5/494
     - 1/39
     - 0/20
   * - late_adaptation_control
     - half
     - oracle_diagnostic
     - 6/494
     - 2/39
     - 0/20
   * - stronger_parent_pretraining
     - full
     - average_link_probability
     - 6/836
     - 1/36
     - 0/20
   * - stronger_parent_pretraining
     - full
     - learned_confidence_mean
     - 7/836
     - 2/36
     - 0/20
   * - stronger_parent_pretraining
     - full
     - learned_confidence_sum
     - 6/836
     - 2/36
     - 0/20
   * - stronger_parent_pretraining
     - full
     - normalized_joint_log_probability
     - 4/836
     - 2/36
     - 0/20
   * - stronger_parent_pretraining
     - full
     - oracle_diagnostic
     - 8/836
     - 3/36
     - 0/20
   * - stronger_parent_pretraining
     - half
     - average_link_probability
     - 8/494
     - 1/39
     - 0/20
   * - stronger_parent_pretraining
     - half
     - learned_confidence_mean
     - 9/494
     - 2/39
     - 0/20
   * - stronger_parent_pretraining
     - half
     - learned_confidence_sum
     - 8/494
     - 2/39
     - 0/20
   * - stronger_parent_pretraining
     - half
     - normalized_joint_log_probability
     - 6/494
     - 2/39
     - 0/20
   * - stronger_parent_pretraining
     - half
     - oracle_diagnostic
     - 9/494
     - 3/39
     - 0/20

Paired uncertainty
------------------

10,000 paired event-cluster bootstrap draws preserve numerator/denominator aggregation. Intervals are conditional on these selected models, omit training-seed uncertainty, and are exploratory without multiplicity correction.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Candidate minus control
     - 95% interval
   * - full
     - lcag_pair_accuracy
     - 0.00170213
     - [0, 0.00414411]
   * - full
     - mother_pid_coverage
     - 0.00988468
     - [0, 0.0206186]
   * - full
     - source_precision
     - 0.00559935
     - [-0.00602106, 0.0177431]
   * - full
     - source_recall
     - 0.00908841
     - [0.000260874, 0.0183241]
   * - half
     - lcag_pair_accuracy
     - 0.0030546
     - [0.000454334, 0.00691569]
   * - half
     - mother_pid_coverage
     - 0.0119863
     - [0.00428243, 0.0214521]
   * - half
     - source_precision
     - 0.00566602
     - [-0.00931132, 0.0206109]
   * - half
     - source_recall
     - 0.0118188
     - [0.00130889, 0.0226668]

Runtime sensitivity of the intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The positive recovered lower bounds for half LCAG and full source recall include zero in the original-node outputs. Half mother coverage and half source recall have positive intervals in both executions. These are exploratory sensitivity comparisons, not additional independent trials or multiplicity-adjusted evidence.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Original-node 95% interval
     - Recovery-host 95% interval
   * - full
     - lcag_pair_accuracy
     - [-0.000255952, 0.00278015]
     - [0, 0.00414411]
   * - full
     - mother_pid_coverage
     - [-0.00148396, 0.0185193]
     - [0, 0.0206186]
   * - full
     - source_precision
     - [-0.00652756, 0.017125]
     - [-0.00602106, 0.0177431]
   * - full
     - source_recall
     - [0, 0.0180632]
     - [0.000260874, 0.0183241]
   * - half
     - lcag_pair_accuracy
     - [0, 0.00442485]
     - [0.000454334, 0.00691569]
   * - half
     - mother_pid_coverage
     - [0.00329489, 0.0191972]
     - [0.00428243, 0.0214521]
   * - half
     - source_precision
     - [-0.00986102, 0.019808]
     - [-0.00931132, 0.0206109]
   * - half
     - source_recall
     - [0.00102739, 0.0223311]
     - [0.00130889, 0.0226668]

Decision from the accumulated studies
-------------------------------------

**Hold the training dataset at 70,000 events now.** No controlled learning
curve establishes an immediate benefit from growing it. Phase40 mixed data
size, steps and cohort changes; it does not isolate a data-size effect.
The lack of established benefit is not evidence that additional data can
never help.

**Prioritize one replication of pretraining quality, not simply more duration.**
Phase55 shows small component-level gains alongside a coherent-forest regression.
It does not establish that improving pretraining is more beneficial than adding
data. It provides a concrete controlled hypothesis to test before either
scaling expense. Phases34/42 did not establish a benefit from longer historical
pretraining alone. Parent supervision also does not repair incompatible target
representations.

The earlier reconstruction sequence remains mixed: Phase41's reduced pointer
weight worsened results; Phase44's seed-deviating diagnostic recovery did not establish a replication
of Phase43's adaptation gain;
Phase45's larger encoder update tied, and corrected Phases46/47 were mixed.
Phase48's intended PID update did not execute because of the autocast cache
bug; corrected Phases49–51 did not establish a stable PID benefit. Higher
recovery weight in Phase52 and lower recovery weight in Phases53/54 failed to
produce a reproducible joint topology/forest gain. Recovery-dose tuning stops.
Different phase cohorts and seeds prohibit causal comparisons of raw rates.

Phase56: one bounded replication
--------------------------------

Repeat pretraining parent-ranking weights **1 versus 2**, with seed 20260923,
matched 2,188-step refinement and identical 4,376-step reconstruction. Keep
70,000 training events, fixed final refined checkpoint transfer, recovery weight
2, frozen transferred PID, and encoder adaptation after step 2,188. The corrected
checkpoint handoff is included before submission.

Only 1,491 untouched validation events remain before this study. Both arms use
**1,000 untouched selection events**, including 1,000 rollout events, and
**100 separate untouched strict events** with a 20-event beam subset. No old
selection or strict UID is reused. This leaves **391** untouched validation
events. Teacher-forced validation work and selection precision are lower than Phase55's 2,000-event selection;
compare matched arms within phase. Validation-pool planning is a separate need
from increasing the training dataset.

Require joint reconstruction benefit and the original gates; do not select on
pretraining loss alone. If this replication again fails to establish joint
benefit, stop parent-ranking dose tuning and diagnose representation/assembly
errors before choosing another factor. No automatic campaign chain, promotion
or sealed-test access.

Metric completeness and limitations
-----------------------------------

The export contains 15,655 original aggregate scalars, 20,339 retained aggregate scalars, 354,042 detailed retained scalars and 9,296,365 tree/candidate scalars. It also includes 749,104 reconstruction-history scalars and 1,318,890 pretraining scalars, every recorded checkpoint diagnostic, micro/unit-macro/event-macro summaries, and exact source/PID/topology combinations.

Physical mother momentum resolution remains unavailable: daughter-sum p4 closure is an implementation invariant. Private event identities and paths stay in the local complete-metrics archive; Pages publishes allowlisted aggregates and source hashes.

Publication and submission verification are recorded in the dashboard and the local final manifest.

Submission verification
-----------------------

Both Phase56 jobs were submitted through held, verified contracts and released
as one bounded pair. Each requests one H100 NVL, eight CPUs, 64 GiB and a
36-hour limit without requeue. The first verified scheduler snapshot was
pending; submission does not establish successful training. The immutable source
revision is ``8b2a994b0ef5a7dfa4011a1eccad5c966a4fcd93`` and the tag is
``reconstruction-phase56-pretraining-parent-replication-20260921``.
The complete input bindings and submission receipt are retained locally.

The frozen preflight passed 192 CPU tests. The broad run passed 1,786 tests
and skipped 34; its one stale dashboard fixture assertion was corrected and
passed both its focused recheck and the frozen preflight. The dashboard's
submission field is a recorded snapshot, not a live scheduler feed.
