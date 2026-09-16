Phase51 reconstruction review
=============================

Both runs completed 4,376 steps on 70,000 events. The lower-rate PID adaptation
candidate has verified PID gradients and changed weights; the control remains
frozen. All fourteen native views, strict repeats, authenticated receipts,
contracts, input hashes and finite checkpoints were checked.

Primary source-set plus mother-PID recovery rises from 275/3,597 (7.6453%) to
280/3,597 (7.7843%), but retained topology does not improve. Both select step
3,000 and miss the unchanged full-root construction gate (1/100 versus the
required 2/100). Root construction is not correct recursive-tree reconstruction.
Neither model is promoted.

Selection uses 2,000 validation events, including 1,000 rollout events. Strict
scoring uses 100 disjoint events and beam the fixed 20-event subset. No sealed
test data was accessed.

Original strict metrics
-----------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Frozen PID
     - Lower-rate PID
   * - exact_mother_coverage
     - 6 / 206
     - 5 / 206
   * - full_lcag
     - 6 / 4018
     - 5 / 4018
   * - full_root_completion
     - 1 / 100
     - 1 / 100
   * - full_source_precision
     - 75 / 80
     - 78 / 89
   * - full_source_recall
     - 75 / 393
     - 78 / 393
   * - half_lcag
     - 28 / 2504
     - 27 / 2504
   * - half_perfect_lcag
     - 11 / 146
     - 10 / 146
   * - half_root_pid_accuracy
     - 17 / 146
     - 14 / 146
   * - half_source_precision
     - 182 / 328
     - 187 / 356
   * - half_source_recall
     - 182 / 709
     - 187 / 709

All retained full and half trees
--------------------------------

Every retained root is checked. Incompatible targets remain failed primary
trials; isolated leaves cannot earn trivial LCAG successes. Half scope contains
19 events with explicit B partitions and 81 with explicit component fallbacks.
There are 3,031 full units (2,913 representable) and 2,538 half/component units
(2,406 representable). Nontrivial units number 139 full and 158 half/component;
only 91 and 96 respectively are representable under the direct target policy.
Source coverage includes preserved inputs and is not hierarchy efficiency.

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Frozen PID
     - Lower-rate PID
   * - full
     - source_precision
     - 2885 / 3234
     - 2884 / 3230
   * - full
     - source_recall
     - 2885 / 3643
     - 2884 / 3643
   * - full
     - lcag_pair_accuracy
     - 30 / 4519
     - 29 / 4519
   * - full
     - mother_pid_coverage
     - 29 / 568
     - 29 / 568
   * - full
     - perfect_lcag
     - 11 / 139
     - 10 / 139
   * - full
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - full
     - root_pid_accuracy
     - 282 / 1598
     - 302 / 1598
   * - half
     - source_precision
     - 2384 / 2709
     - 2392 / 2727
   * - half
     - source_recall
     - 2384 / 3131
     - 2392 / 3131
   * - half
     - lcag_pair_accuracy
     - 30 / 2580
     - 29 / 2580
   * - half
     - mother_pid_coverage
     - 29 / 549
     - 29 / 549
   * - half
     - perfect_lcag
     - 11 / 158
     - 10 / 158
   * - half
     - coherent_retained_forest
     - 0 / 100
     - 0 / 100
   * - half
     - root_pid_accuracy
     - 243 / 1388
     - 262 / 1388

Exact nontrivial LCAG components decline from 11 to 10 in each scope. All exact
components have depth one, across every checkpoint view and returned beam
candidate. Exact LCAG does not by itself imply correct PID; the dashboard also
reports source, source plus leaf PID, source plus topology, and source plus
topology plus all PID conjunctions with their own denominators.

Beam coverage and uncertainty
-----------------------------

All 124 returned candidates (65 control, 59 candidate) receive full and half
checks. The dashboard includes every model-only ranking and the post-inference
coherent oracle. Per-unit Oracle@K maxima are diagnostic bounds, not one coherent
event hypothesis. No beam ranking establishes a coherent retained forest.

Paired event bootstrap uses 10,000 resamples. Candidate minus control retained
full LCAG is -0.022 percentage points (95% interval -0.106 to +0.050); half LCAG
is -0.039 points (-0.185 to +0.086). Mother coverage ties. Source precision and
recall intervals also include zero. These exploratory intervals condition on the
selected trained models, do not measure training-seed uncertainty, and have no
multiplicity correction.

Physical mother momentum resolution remains unavailable because truth-mother
four-vectors are not retained. Daughter-sum closure is an implementation
invariant and must not be presented as physical resolution.

Complete metric coverage
------------------------

Exports contain 15,800 original aggregate rows, 138,399 original detailed rows,
19,297 retained aggregate rows, 316,671 retained detailed rows, 9,539,232
tree/candidate scalar rows, 749,170 training/checkpoint scalar rows and 5,040
micro/macro and exact-conjunction rows. Repeated views are not independent trials.

The :doc:`dashboard <_generated/status/index>` provides all aggregate metric
values and source hashes, alongside micro/macro populations, gates and beam
tables. The complete local archive additionally retains native event reports,
all candidate scalars, training history, uncertainty and verification records.

Study history and next training
------------------------------

Hold the dataset at 70,000 events. Phase40 changed data, steps and cohort
together, so it does not supply a controlled learning curve. The current
failures provide no evidence that more events are the immediate remedy.

Longer pretraining did not establish a benefit in Phases34/42. Improving the
pretraining representation is a worthwhile future controlled experiment, but
its advantage over downstream changes is still unmeasured. The historical
pretrained initialization remains a limitation, not proof of a pretraining-loop
bug. Phase48's ineffective PID adaptation was a reconstruction autocast problem;
real gradients and updates were verified after its repair in Phases49–51.

Phase41's lower pointer weight was worse. Small adaptation effects in Phase43
reversed in Phase44, and Phase45's larger encoder update tied. Corrected
Phase46/47 encoder contrasts were mixed. Effective PID adaptation in Phase49
had small retained-topology gains, but Phase50 failed to replicate an overall
benefit. Phase51's lower dose also supplies no clear topology benefit. Cohorts
and seeds differ between phases; compare within-phase effects rather than raw
cross-phase rates.

Phase52 tests recovery-objective weight 2 versus 4, keeping PID frozen in both
arms, the same late encoder adaptation, 70,000 events, 4,376 steps and unchanged
evaluation gates. The existing recovery term encourages object presence when
predicted context has lost required targets. It is not direct daughter or
recursive-topology supervision and cannot repair target incompatibility;
stronger weighting may create false positives, so precision remains a guardrail.
The control records nonzero recovery counts in 4,294 of 4,376 training steps;
the proposed factor therefore acts on an exercised objective.
This is an exploratory downstream objective test, not an established remedy.

A fresh validation cohort excludes every previous selection and evaluation UID.
One bounded pair is authorized; there is no automatic promotion, campaign chain
or sealed-test access. The dashboard records the submission snapshot.

Both Phase52 jobs were submitted from the validated immutable revision. The
submission snapshot is queued; scheduler state is not a training-completion
claim. The frozen source passed 156 CPU preflight tests, and the broader suite
passed after correcting the subprocess import path (1,690 initial passes plus
12 successful rechecks; 34 environment-dependent skips). Focused reconstruction
tests, two tiny CPU training checks, audit integrity and metric exports passed.
