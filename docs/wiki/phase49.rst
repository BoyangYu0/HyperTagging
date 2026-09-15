Phase49 reconstruction review
=============================

Both native jobs completed successfully. PID adaptation executed after the
Phase48 gradient repair: the candidate PID head has two optimizer-state entries,
each with 2,188 updates, and differs from its pretrained initialization. The
control has no PID optimizer state and its PID weights remain unchanged.
Training logs show two PID gradient tensors on every candidate step from 2,189
to 4,376, and none while frozen. This verifies real GPU execution of the repair;
it does not establish a pretraining-loop defect or promotion-grade physics.

Measured reconstruction
-----------------------

Primary exact source-set plus mother-PID recovery ties at 270/3,670 (7.3569%).
This metric does not require recursive topology. The control selects step 4,376;
the candidate selects step 4,000. Both train for 4,376 steps on 70,000 events.
Checkpoint selection uses 2,000 validation events, including 1,000 rollout events;
strict evaluation uses 100 disjoint events and beam its fixed 20-event subset.
Different selected steps are part of the preregistered selection procedure.


.. list-table::
   :header-rows: 1

   * - Original strict policy metric
     - Frozen PID control
     - Late PID adaptation
   * - exact_mother_coverage
     - 4 / 131
     - 3 / 131
   * - full_lcag
     - 4 / 2034
     - 3 / 2034
   * - full_root_completion
     - 3 / 100
     - 8 / 100
   * - full_source_precision
     - 41 / 50
     - 48 / 59
   * - full_source_recall
     - 41 / 245
     - 48 / 245
   * - half_lcag
     - 23 / 1487
     - 24 / 1487
   * - half_perfect_lcag
     - 11 / 147
     - 9 / 147
   * - half_root_pid_accuracy
     - 27 / 147
     - 20 / 147
   * - half_source_precision
     - 148 / 312
     - 155 / 320
   * - half_source_recall
     - 148 / 574
     - 155 / 574

The candidate passes all preregistered gates; the control misses the minimum
full-source recall threshold (16.73% against 17%). Root construction rises from
3/100 to 8/100, but construction is not correct full-tree reconstruction. Original
full LCAG falls from 4 to 3 correct pairs, and half-root PID falls from 27 to 20.
Gate passage is a screening result, not automatic promotion.

Every retained tree and beam candidate
--------------------------------------

All fourteen native checkpoint and search views, identical strict repeats and all
121 returned beam candidates (61 control, 60 candidate) were checked in full and
half scope. Incompatible targets remain failed trials; isolated leaves earn no
trivial LCAG successes. Missing roots or B hemispheres are never invented.


.. list-table::
   :header-rows: 1

   * - Retained scope
     - Metric
     - Frozen PID control
     - Late PID adaptation
   * - full
     - lcag_pair_accuracy
     - 24 / 2529
     - 28 / 2529
   * - full
     - mother_pid_coverage
     - 24 / 501
     - 28 / 501
   * - full
     - perfect_lcag
     - 11 / 155
     - 11 / 155
   * - full
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - full
     - source_precision
     - 2831 / 3185
     - 2831 / 3198
   * - full
     - source_recall
     - 2831 / 3484
     - 2831 / 3484
   * - half
     - lcag_pair_accuracy
     - 25 / 1548
     - 29 / 1548
   * - half
     - mother_pid_coverage
     - 25 / 486
     - 29 / 486
   * - half
     - perfect_lcag
     - 11 / 170
     - 11 / 170
   * - half
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - half
     - source_precision
     - 2445 / 2770
     - 2456 / 2786
   * - half
     - source_recall
     - 2445 / 3082
     - 2456 / 3082


.. list-table::
   :header-rows: 1

   * - Population
     - Full
     - Half/component
   * - events_without_flagged_target_incompatibility
     - 26
     - 26
   * - isolated_leaf_units
     - 2771
     - 2369
   * - nontrivial_topology_units
     - 155
     - 170
   * - representable_nontrivial_units
     - 113
     - 120
   * - single_source_composite_units
     - 79
     - 79
   * - source_empty_units
     - 0
     - 0

All exact components remain depth one across every view and candidate. The
candidate gains four correct retained LCAG pairs in each scope, but exact
nontrivial component counts tie and coherent retained forests remain 0/100.
Source coverage includes preserved inputs and must not be read as hierarchy
reconstruction efficiency. The dashboard separates original-policy and retained
populations, all model-only beam rankings, coherent diagnostic oracle and per-unit
Oracle@K bounds. Oracle bounds do not describe one coherent event.

Complete metrics and uncertainty
--------------------------------

The :doc:`dashboard <_generated/status/index>` supplies original gates, full/half
retained populations, beam rankings, micro/macro summaries and exact source,
leaf-PID, topology and all-PID conjunctions, with downloadable aggregate metrics.
The external complete archive preserves all native reports, provenance, training
logs, checkpoint scalar histories and detailed tree/candidate exports.

Exports contain 14,801 original and 19,138 retained aggregate metrics; 132,398
original and 303,018 retained detailed metrics; 9,776,138 tree/candidate scalars;
749,106 training-history scalars; and 5,040 aggregation supplement scalars.
Every recorded checkpoint is finite. Undefined denominators remain unavailable;
micro rates sum counts before division and macros average only defined ratios.

Paired event-cluster bootstrap uses 10,000 resamples and ratios of summed counts.
Candidate-minus-control retained full LCAG improves 0.158 percentage points
(95% interval 0.031 to 0.376); half LCAG improves 0.258 points (0.053 to 0.575).
These exploratory intervals are conditional on the two selected models, not
training-seed uncertainty, and are not corrected for multiple comparisons.
Source precision/recall intervals include zero. Physical mother momentum
resolution is unavailable because truth mother four-momenta are not retained;
daughter-sum closure is an implementation invariant.

Decision across the studies
---------------------------

Keep the dataset at 70,000 events now. Phase40 scaling confounded data volume,
steps and cohort, so no controlled learning curve establishes a need for more
data. Phases34 and42 did not establish a benefit from longer pretraining.
Phase41 reduced pointer weight was worse; Phase43's small adaptation gain
reversed in Phase44, which also had a seed deviation. Phase45's larger encoder
rate tied. The mixed corrected-source Phase46 outcome did not repeat in Phase47.
Phase48 did not execute its PID factor and cannot serve as evidence against
adaptation. Phase49 verifies execution and gives a modest, mixed retained-topology
signal; cross-phase cohorts and seeds differ, so raw Phase48-to49 improvement
cannot isolate the gradient repair's causal effect.

Improving representation quality is a more useful direction to test than blindly
adding events or pretraining epochs, but corrected pretraining benefits remain
unmeasured. The immediate next step is to replicate effective downstream PID
adaptation. Phase50 repeats the same frozen-PID control and late-PID candidate
with a new shared seed, fresh disjoint validation cohort, historical pretraining
checkpoint, train-only statistics, gates and 4,376-step budgets. This checks
repeatability before changing another scientific factor. Two seeds still provide
limited evidence about training variability. No sealed-test access or promotion
is authorized. Current submission status appears on the dashboard.
