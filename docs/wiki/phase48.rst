Phase48: ineffective PID adaptation and gradient repair
=======================================================

Both jobs completed 4,376 steps on 70,000 training events. All native receipts,
contracts, checkpoint finiteness, seven evaluation views per arm and exact strict
repeats were verified. The comparison intended to unfreeze the PID head after
2,188 steps while the control kept it frozen. Both use late encoder adaptation.

The intended adaptation did not execute
---------------------------------------

The final models are tensor-for-tensor identical, and neither PID head has any
optimizer state. The recorded configurations contain the intended different
freeze durations. A mixed-precision reproduction identifies the execution problem:
no-gradient rollout forwards populate the autocast weight cache before the
training forward, suppressing gradients on reused weights. Disabling that cache
restores finite, nonzero PID gradients and parameter updates in a CPU model test.

Phase48 is therefore an observation of two identical trained models, not evidence
that effective PID-head adaptation has no benefit. Earlier mixed-precision
scheduled-sampling studies also used this path; their recorded scores remain
valid, but representation-adaptation interpretations require reevaluation. This
finding does not by itself establish a failure in the pretraining loop.

Measured reconstruction
-----------------------

Both selected step 4,000 with 115/3,670 (3.1335%) exact source-set plus mother-PID
recovery on checkpoint-selection rollout. This metric does not require recursive
topology. The 2,000-event selection cohort (1,000 rollout events) is disjoint
from the 100-event strict cohort; beam uses its fixed 20-event subset.

.. list-table:: Original strict policy counts, identical in both arms
   :header-rows: 1

   * - Metric
     - Numerator / denominator
   * - Full root construction
     - 0 / 100
   * - Full LCAG
     - 3 / 4,913
   * - Exact mother coverage
     - 3 / 260
   * - Full source precision / recall
     - 102 / 121; 102 / 516
   * - Half LCAG
     - 20 / 2,825
   * - Half perfect LCAG / root PID
     - 3 / 146; 13 / 146
   * - Half source precision / recall
     - 170 / 312; 170 / 772

Both fail the full-root gate. Passing the other gates does not establish complete
hierarchical reconstruction. No promotion or sealed-test access occurred.

Every retained tree and beam candidate
--------------------------------------

Full scope includes 3,042 retained units: 2,830 isolated leaves, 80 single-source
composites and 132 nontrivial components. Half/component scope includes 2,335:
2,096 isolated leaves, 80 single-source composites and 159 nontrivial components.
Only 86/132 full and 99/159 half nontrivial targets pass representability checks;
20/100 events have no flagged incompatibility. This is a necessary check, not an
achievable efficiency estimate. Incompatible targets remain failed trials.

Both arms have retained full LCAG 17/5,289, mother coverage 17/599 and perfect
components 3/132; half LCAG 19/2,881, mother coverage 19/572 and perfect components
3/159. Coherent retained forests are 0/100. No exact component above depth one
appears across any checkpoint view or returned candidate. Isolated leaves do not
receive trivial LCAG successes. Missing full/B roots are never invented.

All 46 returned width-four beam candidates per arm receive both full and half
checks. The dashboard includes each deployable model ranking and coherent
post-inference oracle. Per-unit Oracle@K is a separate bound, not a coherent event.
Original strict and all-retained populations have different denominators and are
kept separate. Micro rates pool counts; unit/event macros average defined ratios
and report unavailable denominators. Exact source, leaf-PID, topology and all-PID
conjunctions are reported separately.

Complete metrics and uncertainty
--------------------------------

The :doc:`dashboard <_generated/status/index>` provides all aggregate downloads,
original gates, retained populations, beam rankings, and micro/macro conjunctions.
The preserved external archive contains all native reports, training histories,
checkpoint records, detailed exports, tree/candidate scalars and validation hashes.
Exports contain 15,936 original and 19,316 retained aggregate metrics, 162,672 and
339,902 detailed metrics, 8,953,020 tree/candidate scalars, 731,812 training-history
scalars, and a 5,040-scalar aggregation supplement.

Paired event-cluster bootstrap differences for retained LCAG, mother coverage,
source precision and source recall are exactly zero. Degenerate intervals reflect
identical observed counts; they do not imply zero population uncertainty or measure
training-seed uncertainty. Physical mother momentum resolution is unavailable:
truth mother four-momenta are not retained. Daughter-sum closure is an invariant.

Decision across the studies
---------------------------

Hold dataset size at 70,000 until gradient execution is repaired and a controlled
learning curve can isolate data volume from steps and cohort. Phase40 scaling
confounded these factors. Phases34 and42 did not establish benefits from longer
pretraining. Phase41 reduced pointer weight was worse; Phase43's small adaptation
gain reversed in Phase44, which also had a seed deviation. Phase45's larger late
encoder rate tied. Corrected-source Phase46's mixed topology advantage for the
frozen arm did not repeat in Phase47. Phase48 cannot evaluate its intended PID
factor because it produced identical models.

Better representation learning remains a plausible research direction, but neither
more pretraining epochs nor a larger dataset is the immediate remedy. Corrected
pretraining quality remains unmeasured. First restore real downstream gradients.
Phase49 repeats late PID adaptation versus the frozen PID control, with gradient-safe
mixed precision in both arms, unchanged 4,376-step budgets, historical pretrained
encoder, train-only statistics, strict gates and fresh isolated validation cohorts.
The new cohort excludes all 33,809 previously inspected validation UIDs, including
the independent audit sample, and all required overlaps are zero.
A trainable PID head with nonzero supervised loss and absent gradients now fails
explicitly; gradient availability is logged. This is one bounded two-arm study,
not an automatic campaign chain. Current submission status is on the dashboard.
