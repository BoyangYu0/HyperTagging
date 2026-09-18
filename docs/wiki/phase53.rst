Phase53 reconstruction review
=============================

Both recovery-objective runs completed 4,376 steps on 70,000 events. Recovery
weights 2 and 1 are verified at every logged step and in final checkpoints.
Each arm has 4,304 positive recovery-loss records; total missing-target counts
are 105,888 and 105,874. Both PID heads stayed frozen with unchanged pretrained
weights, zero optimizer-state entries and zero PID gradients. Final model tensors
differ. All fourteen native views, strict repeats, authenticated receipts,
input hashes, finite checkpoints and encoder lineage passed verification.

Primary source-set plus mother-PID recovery is 278/3,807 (7.3023%) versus
279/3,807 (7.3286%). This score does not require exact recursive topology.
Control selects step 4,376; weaker recovery selects step 4,000. Control passes
all original gates. Weaker recovery misses full-root construction (1/100 versus
2/100); construction is not a correct-tree metric. Neither model is promoted.

Selection uses 2,000 validation events (1,000 rollout). Strict scoring uses
100 disjoint events and beam its fixed 20-event subset. No sealed test was read.

Original strict metrics
-----------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Recovery weight 2
     - Recovery weight 1
   * - exact_mother_coverage
     - 5 / 156
     - 6 / 156
   * - full_lcag
     - 5 / 2896
     - 6 / 2896
   * - full_root_completion
     - 2 / 100
     - 1 / 100
   * - full_source_precision
     - 78 / 89
     - 62 / 69
   * - full_source_recall
     - 78 / 324
     - 62 / 324
   * - half_lcag
     - 24 / 2173
     - 27 / 2173
   * - half_perfect_lcag
     - 5 / 147
     - 8 / 147
   * - half_root_pid_accuracy
     - 18 / 147
     - 24 / 147
   * - half_source_precision
     - 188 / 348
     - 186 / 338
   * - half_source_recall
     - 188 / 699
     - 186 / 699


All retained full and half trees
--------------------------------

Every retained root is scored, including incompatible targets as failed trials.
Isolated leaves cannot earn trivial LCAG successes. Half scope uses 18 explicit
B-partition events and 82 explicit component fallbacks. Full scope has 3,017
units; half/component has 2,571. Of 141 full and 159 half nontrivial units, only
95 and 103 are representable under the direct policy. Only 24 events have no
flagged target incompatibility. Source coverage includes preserved inputs and
is not hierarchy efficiency.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Recovery weight 2
     - Recovery weight 1
   * - full
     - source_precision
     - 2863 / 3228
     - 2876 / 3223
   * - full
     - source_recall
     - 2863 / 3605
     - 2876 / 3605
   * - full
     - lcag_pair_accuracy
     - 27 / 3634
     - 26 / 3634
   * - full
     - mother_pid_coverage
     - 24 / 551
     - 23 / 551
   * - full
     - mother_pid_accuracy
     - 23 / 24
     - 22 / 23
   * - full
     - perfect_lcag
     - 7 / 141
     - 9 / 141
   * - full
     - coherent_retained_forest
     - 1 / 100
     - 0 / 100
   * - full
     - root_pid_accuracy
     - 359 / 1639
     - 388 / 1639
   * - half
     - source_precision
     - 2418 / 2768
     - 2435 / 2745
   * - half
     - source_recall
     - 2418 / 3141
     - 2435 / 3141
   * - half
     - lcag_pair_accuracy
     - 28 / 2204
     - 29 / 2204
   * - half
     - mother_pid_coverage
     - 25 / 533
     - 26 / 533
   * - half
     - mother_pid_accuracy
     - 24 / 25
     - 25 / 26
   * - half
     - perfect_lcag
     - 7 / 159
     - 9 / 159
   * - half
     - coherent_retained_forest
     - 1 / 100
     - 0 / 100
   * - half
     - root_pid_accuracy
     - 318 / 1429
     - 343 / 1429

Retained exact nontrivial components increase from 7 to 9 in both scopes.
The original policy half metric separately rises from 5/147 to 8/147; these
populations must not be conflated. Coherent retained forests decline from
1/100 to 0/100 in both scopes. The control success is one continuum tau-pair
event with 19 retained units and one nontrivial, one-mother component. It is
not a reconstructed B-pair event or evidence of broad deep-tree performance.
Coherent-forest success requires structure and source agreement, not all-PID
correctness; the separate exact-all-PID conjunction retains that distinction.

Both arms exactly recover the same four-leaf, three-mother, depth-two component
in one continuum event, including all leaf and mother PIDs. This event is not
a coherent-forest success. Its 16 records across arms, scopes and repeated or
alternate views represent one distinct event, not 16 independent successes.
All other exact components have depth one; no returned beam candidate adds a
depth-two exact component. This corrects the earlier phases' depth-one-only
observation for this cohort, but does not identify a lower-weight benefit.

Exact LCAG need not imply correct PID. The dashboard separately exports source,
source plus leaf PID, source plus topology, and source plus topology plus all
PID conjunctions with their own denominators.

Beam search and uncertainty
---------------------------

All 141 returned candidates (72 control, 69 weaker recovery) receive full and
half checks. Every model-only ranking and the coherent post-inference oracle
is included. Candidate-rank populations keep their actual availability counts;
missing ranks are not silently assigned the full cohort denominator. Per-unit
Oracle@K maxima are diagnostic bounds, not one coherent hypothesis.

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
     - 4 / 656
     - 1 / 34
     - 0 / 20
   * - late_adaptation_control
     - full
     - learned_confidence_mean
     - 4 / 656
     - 1 / 34
     - 0 / 20
   * - late_adaptation_control
     - full
     - learned_confidence_sum
     - 4 / 656
     - 1 / 34
     - 0 / 20
   * - late_adaptation_control
     - full
     - normalized_joint_log_probability
     - 4 / 656
     - 1 / 34
     - 0 / 20
   * - late_adaptation_control
     - full
     - oracle_diagnostic
     - 4 / 656
     - 2 / 34
     - 0 / 20
   * - late_adaptation_control
     - half
     - average_link_probability
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - late_adaptation_control
     - half
     - learned_confidence_mean
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - late_adaptation_control
     - half
     - learned_confidence_sum
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - late_adaptation_control
     - half
     - normalized_joint_log_probability
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - late_adaptation_control
     - half
     - oracle_diagnostic
     - 4 / 481
     - 2 / 36
     - 0 / 20
   * - weaker_recovery
     - full
     - average_link_probability
     - 4 / 656
     - 1 / 34
     - 0 / 20
   * - weaker_recovery
     - full
     - learned_confidence_mean
     - 3 / 656
     - 1 / 34
     - 0 / 20
   * - weaker_recovery
     - full
     - learned_confidence_sum
     - 4 / 656
     - 1 / 34
     - 0 / 20
   * - weaker_recovery
     - full
     - normalized_joint_log_probability
     - 5 / 656
     - 1 / 34
     - 0 / 20
   * - weaker_recovery
     - full
     - oracle_diagnostic
     - 5 / 656
     - 1 / 34
     - 0 / 20
   * - weaker_recovery
     - half
     - average_link_probability
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - weaker_recovery
     - half
     - learned_confidence_mean
     - 3 / 481
     - 1 / 36
     - 0 / 20
   * - weaker_recovery
     - half
     - learned_confidence_sum
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - weaker_recovery
     - half
     - normalized_joint_log_probability
     - 4 / 481
     - 1 / 36
     - 0 / 20
   * - weaker_recovery
     - half
     - oracle_diagnostic
     - 5 / 481
     - 1 / 36
     - 0 / 20

Paired event-cluster bootstrap uses 10,000 resamples. Candidate minus control
retained full LCAG is -0.028 percentage points (95% interval -0.206 to +0.138);
half LCAG is +0.045 points (-0.231 to +0.328). Both include zero. Retained
half-source precision rises 1.351 points (0.160 to 2.619); other source and
mother-coverage intervals include zero. This exploratory positive interval has
no multiplicity correction and does not override the failed full-root gate or
lost coherent forest. Intervals condition on the selected models and do not
measure training-seed uncertainty. Selected checkpoint steps differ as allowed
by the preregistered selection procedure.

Physical mother momentum resolution is unavailable because truth-mother
four-vectors are not retained. Daughter-sum closure is an implementation
invariant, not physical resolution.

Complete metric coverage
------------------------

Exports contain 15,764 original aggregate and
149,050 original detailed rows; 19,424 retained aggregate
and 361,489 retained detailed rows; 9,811,328 tree/candidate
scalar rows; 749,126 training/checkpoint scalar rows; and 5,040
micro/macro and exact-conjunction rows. Repeated views are not independent trials.

The :doc:`dashboard <_generated/status/index>` provides aggregate values,
source hashes, gates, populations, micro/macro and beam tables. The complete
local archive preserves every native report, all tree and candidate scalars,
training history, uncertainty, plots and verification evidence.

Study history and next training
-------------------------------

Hold the dataset at 70,000 events. Phase40 changed data, steps and cohort
together and does not provide a controlled learning curve. The accumulated
studies do not establish that more data is the immediate remedy. Target
incompatibility and weak recursive assembly remain measured limitations.

Longer pretraining did not establish a benefit in Phases34/42. Improving
representation quality remains a worthwhile controlled direction, but its
advantage over downstream changes is unmeasured. These studies reuse historical
pretraining and cannot answer whether a changed pretraining objective is better.
A future quality comparison should hold data and compute fixed and evaluate
transfer with these same full/half gates, rather than simply train longer.
Phase48's ineffective PID adaptation was a reconstruction autocast defect;
real PID gradients and updates were subsequently verified in Phases49–51.

Phase41's lower pointer weight was worse. Small adaptation gains in Phase43
reversed in Phase44; Phase45's larger encoder update tied. Corrected Phase46/47
encoder contrasts were mixed. Phase49's small effective PID gains were not
clearly replicated by Phase50; Phase51's lower PID learning rate also failed
to establish a topology benefit. Phase52's stronger recovery worsened retained
exact reconstruction. Phase53's weaker recovery improves component exactness
and half-source precision, but loses a coherent forest and full-root passage.
Seeds and cohorts differ across phases; compare within-phase effects rather
than treating cross-phase raw rates as causal evidence.

Phase54 is one bounded replication of recovery weight 2 versus 1, with a new
seed and untouched cohort, frozen PID, identical late encoder adaptation,
70,000 events, 4,376 steps, the same pretrained initialization and unchanged
gates. Two additional exact components in a small strict cohort are insufficient
to adopt the dose. Replication tests whether the mixed signal survives another
seed/cohort before further tuning. If a joint benefit does not reproduce,
stop recovery-dose tuning and prioritize a controlled representation-quality
experiment. No automatic follow-on campaign is authorized or submitted.

The recovery loss is an object-presence surrogate, not supervision of correct
daughters or recursive topology; its weight cannot repair incompatible targets.
The new cohort excludes all earlier selection and scoring events, including
Phase53 and the independent audit. The dashboard records the submission snapshot.
