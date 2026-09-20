Phase54 reconstruction review
=============================

The lower recovery dose does not reproduce a joint reconstruction benefit.
Primary source-set plus mother-PID recovery rises from 243/3,783 (6.4235%) to
261/3,783 (6.8993%), but retained exact nontrivial components decline from
12 to 11. Both arms fail full-source-recall and full-root-construction gates.
Neither model is promoted. Stop recovery-dose tuning, hold 70,000 training
events, and test a controlled pretraining-quality change next.

Both runs completed 4,376 steps and select step 4,000. Recovery weights 2 and 1
are verified in all 8,752 logged steps and final checkpoints. Each arm has
4,296 positive recovery-loss records; missing-target counts sum to 106,246
and 106,139. PID remained frozen with unchanged pretrained weights, no optimizer
state and zero PID gradients. Final models differ. Authenticated receipts,
input and artifact hashes, finite checkpoints, strict repeats and lineage
are verified. All fourteen views and every full/half beam candidate are checked.

Selection uses 2,000 validation events (1,000 rollout). Strict scoring uses
100 disjoint events and beam its fixed 20-event subset. No sealed test was read.
Primary recovery does not require correct recursive topology. Constructing a
full root is also different from reconstructing a correct tree.

Original strict metrics
-----------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Recovery weight 2
     - Recovery weight 1
   * - exact_mother_coverage
     - 3 / 190
     - 3 / 190
   * - full_lcag
     - 3 / 3618
     - 3 / 3618
   * - full_root_completion
     - 0 / 100
     - 1 / 100
   * - full_source_precision
     - 55 / 63
     - 59 / 71
   * - full_source_recall
     - 55 / 374
     - 59 / 374
   * - half_lcag
     - 19 / 2274
     - 20 / 2274
   * - half_perfect_lcag
     - 11 / 165
     - 10 / 165
   * - half_root_pid_accuracy
     - 36 / 165
     - 35 / 165
   * - half_source_precision
     - 167 / 299
     - 163 / 306
   * - half_source_recall
     - 167 / 706
     - 163 / 706


All retained full and half trees
--------------------------------

Incompatible targets remain failed primary trials. Isolated leaves cannot earn
trivial LCAG successes. Half scope uses 20 explicit B-partition events and
80 component fallbacks. Full scope has 3,060 units; half/component has 2,571.
Only 115 of 155 full and 124 of 175 half nontrivial units are representable.
Twenty-nine events have no flagged target incompatibility. Source coverage
includes preserved inputs and is not hierarchy efficiency.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Recovery weight 2
     - Recovery weight 1
   * - full
     - source_precision
     - 2936 / 3230
     - 2954 / 3242
   * - full
     - source_recall
     - 2936 / 3644
     - 2954 / 3644
   * - full
     - lcag_pair_accuracy
     - 20 / 4058
     - 23 / 4058
   * - full
     - mother_pid_coverage
     - 20 / 553
     - 23 / 553
   * - full
     - mother_pid_accuracy
     - 19 / 20
     - 22 / 23
   * - full
     - perfect_lcag
     - 12 / 155
     - 11 / 155
   * - full
     - coherent_retained_forest
     - 1 / 100
     - 1 / 100
   * - full
     - root_pid_accuracy
     - 386 / 1694
     - 400 / 1694
   * - half
     - source_precision
     - 2458 / 2720
     - 2476 / 2722
   * - half
     - source_recall
     - 2458 / 3135
     - 2476 / 3135
   * - half
     - lcag_pair_accuracy
     - 22 / 2328
     - 23 / 2328
   * - half
     - mother_pid_coverage
     - 22 / 533
     - 23 / 533
   * - half
     - mother_pid_accuracy
     - 21 / 22
     - 22 / 23
   * - half
     - perfect_lcag
     - 12 / 175
     - 11 / 175
   * - half
     - coherent_retained_forest
     - 1 / 100
     - 1 / 100
   * - half
     - root_pid_accuracy
     - 352 / 1460
     - 367 / 1460

Retained exact nontrivial components fall from 12 to 11 in both scopes.
Original policy half exactness separately falls from 11/165 to 10/165; these
are different populations. Every exact component across all views and returned
candidates has depth one. Phase53's shared depth-two success does not recur on
this new cohort; this is not a controlled cross-cohort regression claim.

Each arm has one coherent retained forest, on different continuum tau-pair
events. The control success contains 13 isolated leaves and no mother. The
candidate success contains 22 units including one nontrivial one-mother
component. Thus the tied forest count hides different structural content;
neither is a reconstructed B-pair or a deep hierarchy. Full and half records
of the same event are not independent successes. Forest coherence does not
require all PIDs to be correct; the exact source/topology/all-PID conjunction
is reported separately.

Beam search and uncertainty
---------------------------

All 124 returned candidates (65 control, 59 weaker recovery) receive full and
half checks. Model-only rankings are separate from post-inference coherent
oracles. Candidate ranks retain actual availability counts. Per-unit Oracle@K
maxima are diagnostic bounds, not a single coherent hypothesis.

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
     - 5 / 706
     - 3 / 36
     - 0 / 20
   * - late_adaptation_control
     - full
     - learned_confidence_mean
     - 5 / 706
     - 4 / 36
     - 0 / 20
   * - late_adaptation_control
     - full
     - learned_confidence_sum
     - 5 / 706
     - 4 / 36
     - 0 / 20
   * - late_adaptation_control
     - full
     - normalized_joint_log_probability
     - 5 / 706
     - 4 / 36
     - 0 / 20
   * - late_adaptation_control
     - full
     - oracle_diagnostic
     - 7 / 706
     - 6 / 36
     - 0 / 20
   * - late_adaptation_control
     - half
     - average_link_probability
     - 6 / 407
     - 3 / 38
     - 0 / 20
   * - late_adaptation_control
     - half
     - learned_confidence_mean
     - 6 / 407
     - 4 / 38
     - 0 / 20
   * - late_adaptation_control
     - half
     - learned_confidence_sum
     - 6 / 407
     - 4 / 38
     - 0 / 20
   * - late_adaptation_control
     - half
     - normalized_joint_log_probability
     - 6 / 407
     - 4 / 38
     - 0 / 20
   * - late_adaptation_control
     - half
     - oracle_diagnostic
     - 7 / 407
     - 6 / 38
     - 0 / 20
   * - weaker_recovery
     - full
     - average_link_probability
     - 6 / 706
     - 3 / 36
     - 0 / 20
   * - weaker_recovery
     - full
     - learned_confidence_mean
     - 6 / 706
     - 4 / 36
     - 0 / 20
   * - weaker_recovery
     - full
     - learned_confidence_sum
     - 7 / 706
     - 2 / 36
     - 0 / 20
   * - weaker_recovery
     - full
     - normalized_joint_log_probability
     - 7 / 706
     - 4 / 36
     - 0 / 20
   * - weaker_recovery
     - full
     - oracle_diagnostic
     - 7 / 706
     - 5 / 36
     - 0 / 20
   * - weaker_recovery
     - half
     - average_link_probability
     - 6 / 407
     - 3 / 38
     - 0 / 20
   * - weaker_recovery
     - half
     - learned_confidence_mean
     - 6 / 407
     - 4 / 38
     - 0 / 20
   * - weaker_recovery
     - half
     - learned_confidence_sum
     - 7 / 407
     - 2 / 38
     - 0 / 20
   * - weaker_recovery
     - half
     - normalized_joint_log_probability
     - 7 / 407
     - 4 / 38
     - 0 / 20
   * - weaker_recovery
     - half
     - oracle_diagnostic
     - 7 / 407
     - 5 / 38
     - 0 / 20

Paired event-cluster bootstrap uses 10,000 resamples and seed 20260920.
All reported LCAG, mother-coverage and source precision/recall intervals include
zero. Intervals condition on selected models, do not measure training-seed
uncertainty, and have no multiplicity correction.

.. list-table::
   :header-rows: 1

   * - Scope
     - Candidate minus control
     - Difference (percentage points)
     - 95% interval
   * - full
     - lcag_pair_accuracy
     - +0.074
     - -0.033 to +0.172
   * - full
     - mother_pid_coverage
     - +0.542
     - -0.201 to +1.309
   * - full
     - source_precision
     - +0.219
     - -0.625 to +1.039
   * - full
     - source_recall
     - +0.494
     - -0.223 to +1.206
   * - half
     - lcag_pair_accuracy
     - +0.043
     - -0.109 to +0.202
   * - half
     - mother_pid_coverage
     - +0.188
     - -0.424 to +0.847
   * - half
     - source_precision
     - +0.595
     - -0.291 to +1.480
   * - half
     - source_recall
     - +0.574
     - -0.236 to +1.374

Physical mother momentum resolution is unavailable because truth-mother
four-vectors are not retained. Daughter-sum closure is an implementation
invariant, not physical resolution.

Complete metric coverage
------------------------

Exports contain 15,729 original aggregate and
142,644 original detailed rows; 19,974 retained aggregate
and 306,145 retained detailed rows; 9,870,833 tree/candidate
scalar rows; 749,208 training/checkpoint scalar rows; and 5,040
micro/macro and exact-conjunction rows. Repeated views are not independent trials.

The :doc:`dashboard <_generated/status/index>` contains complete aggregate
metric downloads, hashes, gates, population, micro/macro and beam tables.
The local complete archive additionally preserves native reports, per-tree
and per-candidate scalars, training history, uncertainty and verification.
All earlier metric downloads remain byte-identical. Phase41 bulk rows now
have a separate hash-bound download to keep the status manifest bounded;
its displayed diagnostics and complete numeric values are preserved.

Study history and next training
-------------------------------

Do not increase the dataset now. Phase40 changed data, steps and cohort
together, so it does not provide a controlled learning curve. Recursive
assembly and direct-target incompatibility remain measured limitations.
Additional data may eventually help, but no controlled result establishes
that it is the best immediate use of compute.

Longer pretraining did not establish a downstream benefit in Phases34/42.
Pretraining-quality improvement is the more useful next hypothesis to test,
not an already demonstrated superior intervention. Phase41's lower pointer
weight was worse; small Phase43 adaptation gains reversed in Phase44;
Phase45's larger encoder update tied. Corrected Phase46/47 encoder contrasts
were mixed. Phase48's intended PID update did not execute due to a downstream
autocast defect. Effective PID updates in Phases49–51 did not establish a
replicated topology benefit. This defect is not evidence that pretraining
itself had the same failure.

Phase52's higher recovery weight reduced retained exactness. Phase53's lower
weight improved component exactness but lost full-root passage and a coherent
forest. Phase54 improves primary recovery but reduces component exactness and
fails two gates in both arms. This meets the preregistered stopping condition
for recovery-dose tuning. The loss is an object-presence surrogate, not direct
daughter or recursive-topology supervision. Its weight cannot repair
structurally incompatible targets. Seeds and cohorts differ across campaigns;
within-phase contrasts, not raw cross-phase rates, support these conclusions.

Phase55 is one matched parent-ranking pretraining comparison. Both arms start
from the same step-81,096 parameters and run 2,188 refinement steps, batch 32,
on the same 70,000 training events. Four curriculum phases receive 547 steps
each. Only parent-ranking loss weight differs: 1 versus 2. Learning rate is
0.00005 with the same schedule. Fresh corrected train-only normalization,
optimizer, schedule, RNG and channel memory are used; this is parameter
initialization, not exact resume. Strengthening direct-parent supervision
is a testable response to shallow exactness, not a claim that it repairs it.

Each arm then transfers its fixed final refined checkpoint into an identical
4,376-step reconstruction run. Recovery weight stays 2, PID is frozen after
transfer, and late encoder adaptation and all strict gates remain unchanged.
Selection, strict scoring and full/half beam checks use one fresh shared cohort.
Pretraining validation uses only the first 1,000 selection events, never strict
scoring events. The cohort excludes all prior selection, strict and audit UIDs.

The two jobs have matched data and compute. They compare objective weights,
not refinement against no refinement, and do not estimate a data-scaling curve.
Transfer must improve the joint downstream metrics and meet the original
gates; lower pretraining loss alone is insufficient. No promotion, sealed-test
access, or automatic subsequent campaign is authorized. The dashboard records
the two-job submission snapshot.

After reserving Phase55, only 1,491 untouched validation UIDs remain in the
50,000-event validation role. Another fresh 2,100-event selection/scoring cohort
will not fit. Plan a validation refresh or a preregistered reuse design before
a later campaign; this is distinct from increasing training data and does not
authorize opening the sealed test.
