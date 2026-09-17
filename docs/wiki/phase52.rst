Phase52 reconstruction review
=============================

Both recovery-objective runs completed 4,376 steps on 70,000 events. Recovery
weights 2 and 4 are verified in the final checkpoints; the term records nonzero
missing-target counts in 4,292 steps in each arm (106,439 and 106,442 total).
Both PID heads remained frozen with unchanged pretrained weights and zero
optimizer-state entries. The final models differ. All fourteen native views,
strict repeats, authenticated receipts, input hashes, checkpoint finiteness and
encoder lineage passed verification.

Primary source-set plus mother-PID recovery increases from 279/3,619 (7.7093%)
to 282/3,619 (7.7922%). This primary does not require exact recursive topology.
Control selects step 4,376; stronger recovery selects step 4,000. Both pass all
original gates, but retained topology and original half-tree exactness decline.
Full-root construction falls from 7/100 to 2/100; construction does not establish
a correct recursive tree. Neither model is promoted.

Selection uses 2,000 validation events (1,000 rollout). Strict scoring uses
100 disjoint events and beam its fixed 20-event subset. No sealed test was read.

Original strict metrics
-----------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Recovery weight 2
     - Recovery weight 4
   * - exact_mother_coverage
     - 3 / 150
     - 4 / 150
   * - full_lcag
     - 3 / 2484
     - 4 / 2484
   * - full_root_completion
     - 7 / 100
     - 2 / 100
   * - full_source_precision
     - 74 / 91
     - 54 / 64
   * - full_source_recall
     - 74 / 282
     - 54 / 282
   * - half_lcag
     - 23 / 1812
     - 19 / 1812
   * - half_perfect_lcag
     - 10 / 152
     - 8 / 152
   * - half_root_pid_accuracy
     - 24 / 152
     - 20 / 152
   * - half_source_precision
     - 181 / 353
     - 186 / 375
   * - half_source_recall
     - 181 / 645
     - 186 / 645


All retained full and half trees
--------------------------------

Every retained root is scored, including incompatible targets as failed trials.
Isolated leaves cannot earn trivial LCAG successes. Half scope contains 16 events
with explicit B partitions and 84 with explicit component fallbacks. Full scope
has 3,097 units (2,971 representable); half/component has 2,696 (2,564
representable). Of 146 full and 162 half nontrivial units, only 100 and 110 are
representable under the direct policy. Only 22 events have no flagged target
incompatibility. Source coverage includes preserved inputs and is not hierarchy
efficiency.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Recovery weight 2
     - Recovery weight 4
   * - full
     - source_precision
     - 2945 / 3302
     - 2932 / 3304
   * - full
     - source_recall
     - 2945 / 3624
     - 2932 / 3624
   * - full
     - lcag_pair_accuracy
     - 22 / 3063
     - 18 / 3063
   * - full
     - mother_pid_coverage
     - 22 / 504
     - 18 / 504
   * - full
     - mother_pid_accuracy
     - 22 / 22
     - 18 / 18
   * - full
     - perfect_lcag
     - 10 / 146
     - 9 / 146
   * - full
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - full
     - root_pid_accuracy
     - 338 / 1662
     - 347 / 1662
   * - half
     - source_precision
     - 2549 / 2895
     - 2545 / 2882
   * - half
     - source_recall
     - 2549 / 3207
     - 2545 / 3207
   * - half
     - lcag_pair_accuracy
     - 26 / 1843
     - 21 / 1843
   * - half
     - mother_pid_coverage
     - 26 / 488
     - 21 / 488
   * - half
     - mother_pid_accuracy
     - 26 / 26
     - 21 / 21
   * - half
     - perfect_lcag
     - 10 / 162
     - 9 / 162
   * - half
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - half
     - root_pid_accuracy
     - 307 / 1478
     - 321 / 1478

Retained exact nontrivial components decline from 10 to 9 in both scopes.
The original policy half metric separately declines from 10/152 to 8/152;
these populations must not be conflated. All exact components have depth one
across every view and beam candidate. Exact LCAG does not imply correct PID;
the dashboard also exports source, source plus leaf PID, source plus topology,
and source plus topology plus all PID conjunctions at their own denominators.

Beam search and uncertainty
---------------------------

All 118 returned candidates (58 control, 60 stronger recovery) receive full
and half checks. Every model-only ranking and the coherent post-inference oracle
is included. Normalized-joint full LCAG declines from 2/616 to 0/616 and exact
components from 1/21 to 0/21; half LCAG declines from 3/347 to 1/347 and exact
components from 1/25 to 0/25. Even the coherent oracle has zero exact candidate
components in the stronger-recovery arm. Coherent retained forests are zero for
all rankings and both arms. Per-unit Oracle@K maxima are diagnostic bounds,
not one coherent hypothesis.

Paired event-cluster bootstrap uses 10,000 resamples. Candidate minus control
retained full LCAG is -0.131 percentage points (95% interval -0.356 to +0.069);
half LCAG is -0.271 points (-0.617 to +0.001). Both include zero, as do mother
coverage and source precision/recall intervals. These exploratory intervals
condition on the selected models, do not measure training-seed uncertainty,
and have no multiplicity correction. Checkpoint selection is part of the
preregistered procedure; selected steps differ between arms.

Physical mother momentum resolution is unavailable because truth-mother
four-vectors are not retained. Daughter-sum closure is an implementation
invariant, not physical resolution.

Complete metric coverage
------------------------

The exports include 15,593 original aggregate and 136,898 original detailed
rows; 19,433 retained aggregate and 319,864 retained detailed rows; 9,742,642
tree/candidate scalar rows; 749,102 training/checkpoint scalar rows; and 5,040
micro/macro and exact-conjunction rows. Repeated views are not independent trials.

The :doc:`dashboard <_generated/status/index>` provides all aggregate values,
source hashes, gates, population counts, micro/macro and beam tables. The
complete local archive additionally preserves all native reports, candidate
scalars, training history, uncertainty, plots and verification evidence.

Study history and next training
-------------------------------

Hold the dataset at 70,000 events. Phase40 changed data, steps and cohort
together and therefore does not provide a controlled learning curve. The
accumulated studies do not show that increasing data is the immediate remedy.

Longer pretraining did not establish a benefit in Phases34/42. Improving
representation quality remains worth a controlled experiment, but its advantage
over downstream changes is unmeasured. Reusing the historical pretrained
initialization limits the comparison; it does not prove a pretraining-loop bug.
Phase48's ineffective PID adaptation was a reconstruction autocast defect;
real gradients and updates were subsequently verified in Phases49–51.

Phase41's lower pointer weight was worse. Small adaptation gains in Phase43
reversed in Phase44; Phase45's larger encoder update tied. Corrected
Phase46/47 encoder contrasts were mixed. Phase49's small effective PID gains
were not clearly replicated by Phase50; Phase51's lower PID learning rate
also failed to establish a topology benefit. Phase52's stronger recovery term
raises the selection score slightly but worsens retained exact reconstruction.
Seeds and cohorts differ across phases; raw cross-phase rates are not causal
effects. Compare within-phase contrasts.

Phase53 tests recovery weight 2 versus 1 with PID frozen, identical late encoder
adaptation, 70,000 events, 4,376 steps, the same initialization and unchanged
gates. This lower-dose test probes whether the object-presence surrogate
competes with exact reconstruction. Phase52 does not prove that mechanism or
that reducing the weight will help. The recovery term does not supervise
correct daughters or recursive topology and cannot repair target incompatibility.
A lower weight could sacrifice recall, so the same recall and precision gates
remain necessary. This is one bounded exploratory pair, not a promotion.

The new validation cohort excludes every previous selection and evaluation UID,
including Phase52 and the independent audit. There is no automatic campaign
chain or sealed-test access. The dashboard records the submission snapshot.
