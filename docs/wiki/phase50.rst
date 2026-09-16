Phase50 reconstruction review
=============================

Both jobs completed successfully. All fourteen native views, strict repeats,
receipts, contracts, input hashes and finite checkpoints were checked. Effective
PID adaptation executed: the candidate PID weights changed and have optimizer
state; control PID weights remain frozen at the pretrained initialization.

Measured reconstruction
-----------------------

Primary exact source-set plus mother-PID recovery is 285/3,588 (7.9431%) for
frozen PID and 280/3,588 (7.8038%) for late PID adaptation. It does not require
recursive topology. Both select step 4,376 and train for 4,376 steps on 70,000
events. Selection uses 2,000 validation events, including 1,000 rollout events;
strict evaluation uses 100 disjoint events and beam its fixed 20-event subset.

.. list-table::
   :header-rows: 1

   * - Original strict policy metric
     - Frozen PID control
     - Late PID adaptation
   * - exact_mother_coverage
     - 4 / 209
     - 5 / 209
   * - full_lcag
     - 4 / 3902
     - 5 / 3902
   * - full_root_completion
     - 2 / 100
     - 1 / 100
   * - full_source_precision
     - 76 / 86
     - 73 / 84
   * - full_source_recall
     - 76 / 415
     - 73 / 415
   * - half_lcag
     - 20 / 2469
     - 21 / 2469
   * - half_perfect_lcag
     - 6 / 157
     - 6 / 157
   * - half_root_pid_accuracy
     - 28 / 157
     - 26 / 157
   * - half_source_precision
     - 176 / 317
     - 170 / 289
   * - half_source_recall
     - 176 / 747
     - 170 / 747

The control passes all original gates; the candidate fails the full-root
construction gate (1/100 against the minimum 2/100). Root construction is not
correct tree reconstruction. Original full LCAG improves by one pair, but full
and half source recall and half-root PID decline. Neither model is promoted.

Every retained tree and beam candidate
--------------------------------------

All 129 returned beam candidates (65 control, 64 candidate) receive full and half
checks. Incompatible targets remain failed trials; isolated leaves earn no
trivial LCAG successes. No missing roots or hemispheres are invented.

.. list-table::
   :header-rows: 1

   * - Retained scope
     - Metric
     - Frozen PID control
     - Late PID adaptation
   * - full
     - lcag_pair_accuracy
     - 17 / 4366
     - 20 / 4366
   * - full
     - mother_pid_coverage
     - 17 / 554
     - 20 / 554
   * - full
     - perfect_lcag
     - 6 / 143
     - 6 / 143
   * - full
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - full
     - source_precision
     - 2925 / 3254
     - 2924 / 3246
   * - full
     - source_recall
     - 2925 / 3688
     - 2924 / 3688
   * - full
     - target_representable
     - 2950 / 3068
     - 2950 / 3068
   * - half
     - lcag_pair_accuracy
     - 20 / 2477
     - 22 / 2477
   * - half
     - mother_pid_coverage
     - 20 / 532
     - 22 / 532
   * - half
     - perfect_lcag
     - 6 / 165
     - 6 / 165
   * - half
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - half
     - source_precision
     - 2350 / 2641
     - 2353 / 2631
   * - half
     - source_recall
     - 2350 / 3074
     - 2353 / 3074
   * - half
     - target_representable
     - 2347 / 2476
     - 2347 / 2476

.. list-table::
   :header-rows: 1

   * - Reference population
     - Full
     - Half/component
   * - events_without_flagged_target_incompatibility
     - 21
     - 21
   * - isolated_leaf_units
     - 2852
     - 2238
   * - nontrivial_topology_units
     - 143
     - 165
   * - representable_nontrivial_units
     - 98
     - 109
   * - single_source_composite_units
     - 73
     - 73
   * - source_empty_units
     - 0
     - 0

Both models recover six exact nontrivial components per scope and zero coherent
forests. Every exact component across all views and candidates remains depth one.
Retained source coverage includes preserved detector inputs; it is not hierarchy
reconstruction efficiency. Model-only beam rankings, coherent diagnostic oracle
and per-unit Oracle@K bounds are separate in the dashboard. Oracle@K bounds need
not correspond to one coherent event.

On the 20-event beam cohort, normalized-joint top-1 recovers one versus two
exact components in each scope. Learned-confidence mean recovers two in both
arms, while full-scope LCAG declines from three to two pairs. No model ranking
or coherent diagnostic oracle recovers a coherent forest. The beam evidence
is mixed and does not justify changing the preregistered deployment ranker.

Complete metrics and uncertainty
--------------------------------

The :doc:`dashboard <_generated/status/index>` provides all original-policy and
retained full/half metrics, beam rankings, micro/macro populations and exact
source, leaf-PID, topology and all-PID conjunctions with aggregate downloads.
The external complete archive includes native reports, provenance, training logs,
checkpoint scalar histories and every tree/candidate scalar.

Exports contain 15,960 original and 19,409 retained aggregate metrics; 148,613
original and 335,569 retained detailed metrics; 9,589,798 tree/candidate scalars;
749,284 training-history scalars; and 5,040 aggregation supplement scalars.
Micro rates sum counts before division; macro means use only defined ratios and
count unavailable values separately. Physical mother momentum resolution is
unavailable because truth mother four-momenta are not retained. Daughter-sum
closure measures an implementation invariant, not physical resolution.

Paired event-cluster bootstrap uses 10,000 resamples. The retained full LCAG gain
is 0.069 percentage points (95% interval 0.000 to 0.165); half gains 0.081 points
(0.000 to 0.218). Both intervals include zero. Source precision and recall
intervals also include zero. These exploratory intervals condition on selected
models, do not estimate training-seed uncertainty and have no multiple-comparison
correction. Repeated views and checkpoint aliases are not independent trials.

Decision across the studies
--------------------------

Keep 70,000 training events for now. Phase40 changed dataset size, compute and
cohort together; it does not establish a controlled data scaling benefit.
Phases34/42 do not establish that longer pretraining helps. Phase41's lower
pointer weight was worse; Phase43's small adaptation gain reversed in Phase44,
and Phase45's larger late-encoder learning rate tied. Corrected Phase46 effects
did not consistently repeat in Phase47. Phase48 did not execute its PID factor
because of the autocast gradient defect, so it is not a valid negative PID study.

With repaired execution, Phase49 and Phase50 both show small retained LCAG gains,
but neither improves exact component counts or coherent forests. Phase49 primary
recovery tied and its candidate passed gates; Phase50 primary recovery declined
and its candidate failed a gate. This is not a robust overall adaptation benefit.
Cross-phase seed and validation cohort both change; compare paired effects
within studies rather than treating raw rates as a learning curve.

Improving pretraining quality and transfer alignment is a more targeted research
hypothesis than blindly adding data or epochs, but its benefit is still unmeasured.
These results do not identify pretraining as the cause of shallow trees or target
incompatibility. The next inexpensive controlled test reduces the downstream PID
update size before committing to a new pretraining campaign.

Next bounded training
---------------------

Phase51 compares the frozen-PID control with late PID adaptation at a tenfold
smaller learning-rate multiplier, 0.1. Both use late encoder unfreeze at step
2,188 with multiplier 0.05, the same pretrained checkpoint, corrected train-only
statistics, 70,000 events and 4,376 steps. Candidate PID unfreezes at step 2,188;
control stays frozen. This tests the low-rate adaptation regimen as a whole,
not independent timing and learning-rate effects. It is exploratory, not a proven
remedy or a corrected-pretraining experiment.

A fresh seed and untouched validation cohort exclude all earlier inspected
cohorts and the independent audit. Gates, search budgets and selection procedure
remain fixed. Exactly two bounded jobs are authorized, with no automatic campaign
chain, promotion or sealed-test access. The dashboard records submission state.
