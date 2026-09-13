Phase46 reconstruction review
=============================

Both corrected-source arms completed 4,376 steps on the same 70,000 training
events. Late adaptation selected 120/3,854 complete target source sets plus mother
PID (3.11%); the frozen encoder selected 110/3,854 (2.85%). This primary does not
require exact recursive topology. On the disjoint strict cohort, the frozen
encoder instead recovered nine exact nontrivial components versus six for late
adaptation. Neither arm passes every preregistered gate. No model is promoted.

Source and population boundaries
--------------------------------

Both runs use immutable corrected source
``776a48a74e674d31fc3ddadf705e0167b418a699`` and freshly fitted train-only
normalization. The historical encoder checkpoint at step 81,096 is unchanged;
these runs do not test newly corrected pretraining objectives. The control freezes
the encoder for 2,188 steps then adapts at multiplier 0.05; the comparison freezes
it for all 4,376 steps. Decoder settings, data, replay slots and gates match.
The corrected source and new cohort prevent causal comparisons with Phase45.

Training, replay and cohort seeds agree. Selection uses 2,000 validation events,
including 1,000 rollout events. Strict inference uses 100 disjoint events; beam
uses its fixed twenty-event subset. All seven evaluation views per arm are
complete, primary repeats are exact, all saved checkpoint tensors are finite,
and immutable input, checkpoint, report and receipt hashes are verified. No sealed
test was accessed. Selection chooses step 4,376 for control and 4,000 for frozen.

The :doc:`dashboard <_generated/status/index>` provides both complete aggregate
metric downloads, original gates, separate all-retained populations and every
beam ranking. Original full LCAG is 0/3,268 control versus 1/3,268 frozen; half
LCAG is 15/2,148 versus 19/2,148. Configured full-root construction is 0/100 in
both arms; it is an inference diagnostic, not truth-exact reconstruction.
The control fails full-root, full-LCAG, exact-mother, full-recall and half-recall
gates. Frozen fails full-root, full-LCAG, exact-mother and primary-efficiency
gates. Gate names containing “nonzero” retain their preregistered numeric
thresholds; one correct pair or mother does not necessarily pass them.

Every retained tree and beam candidate
--------------------------------------

Every arm checks all 3,106 full units and 2,667 half/component units. Full units
comprise 2,868 isolated leaves, 78 single-source composites and 160 nontrivial
components. Half units comprise 2,412 isolated leaves, 78 single-source composites
and 177 nontrivial components. Missing initial roots and B hemispheres are never
invented. Incompatible targets remain failed trials; isolated leaves do not earn
trivial LCAG successes. Only 108/160 full and 116/177 half nontrivial targets pass
representability checks. Just 22/100 events have no flagged incompatibility;
these are necessary checks, not an achievable efficiency prediction.

Control versus frozen all-retained full LCAG is 18/3,828 versus 21/3,828;
matched-mother coverage is 18/560 versus 21/560. Half LCAG is 18/2,238 versus
23/2,238; mother coverage is 18/543 versus 23/543. Perfect nontrivial components
are 6/160 versus 9/160 full and 6/177 versus 9/177 half. All perfect primary
components have two leaves, one mother and retained depth one. No deeper exact
component was recovered. Coherent retained-forest agreement is 0/100 in both
scopes. High source coverage largely reflects isolated-input preservation.

All 40 control and 39 frozen returned beam hypotheses receive full and half
checks. No model ranking or coherent oracle reconstructs an exact nontrivial
component on this twenty-event subset (0/28 full, 0/34 half), or a coherent forest
(0/20). LCAG varies between three and four correct pairs across rankings, out
of 1,610 full and 868 half pairs. The coherent post-inference oracle reaches
four in both arms and scopes. Per-unit Oracle@K maxima remain labelled bounds;
they do not describe one coherent reconstructed event. Truth never enters search.

Paired event-cluster bootstrap uses 10,000 resamples, seed 20260913, and ratios
of summed counts. Frozen-minus-control half LCAG is +0.223 percentage points
(95% percentile interval +0.052 to +0.439); full LCAG is +0.078 points
(0.000 to +0.201). These intervals condition on the two trained models and do
not measure training-seed uncertainty. The selection primary favors control,
while strict topology favors frozen. This mixed result motivates replication.

Physical mother momentum resolution is unavailable because truth mother
four-vectors were not retained. Daughter-sum closure checks implementation
consistency and cannot substitute for physical momentum resolution.

Complete evidence
-----------------

Public downloads preserve 16,035 original and 19,484 retained aggregate metrics,
with exact registries and source hashes. The complete local bundle additionally
contains 142,876 original detailed entries, 324,769 retained detailed entries,
9,444,998 per-tree/candidate scalar records and 731,740 training-history/checkpoint
scalar records. All fourteen native reports, alternate checkpoint tracks,
contracted diagnostics, repeats, source-category and target-shape breakdowns,
PID confusion, calibration, kinematic availability and beam/oracle metrics remain
inspectable. Checkpoint aliases are records, not independent experiments.

Allocation based on all studies
-------------------------------

The :doc:`Phase41 synthesis <phase41>` covers the earlier studies. Phase34 did
not establish a reliable downstream benefit from longer pretraining; repaired
Phase35 tied frozen and late adaptation under an older decoder. Phases36-39
improved local metrics without reliable full hierarchies. Historical Stage A
metrics are different measurements and cannot replace strict full-decay scores.

Phase40 scaling changed data, compute and cohort together. Phase41 pointer-weight
reduction had a lower observed primary (143/3,620 versus 134/3,620). Phase42 longer
pretraining also had a lower observed primary (141/3,938 versus 136/3,938).
Earlier adaptation had a small positive Phase43 difference (136/3,592 versus
140/3,592), followed by a reversed Phase44 diagnostic difference (185/3,496 versus
173/3,496). Phase44 also had a replay-seed deviation and is not a clean replication.
Phase45 doubling the late encoder learning rate was neutral at 148/3,805 and
on retained topology. Phase46 introduces corrected scientific supervision and
normalization; its result cannot establish that those fixes caused a change
relative to historical phases because cohort and source changed together.

Keep dataset size at 70,000 now. No controlled learning curve demonstrates that
data volume is the limiting factor, while representability and deeper exact
reconstruction remain unresolved. Better pretraining supervision and representation
quality are more defensible research targets than simply adding epochs, but their
benefit remains unmeasured. Phase42 does not support longer pretraining alone.
Phase46 does not establish that adaptation is beneficial overall: its primary
advantage comes with worse strict retained topology.

Phase47 repeats late adaptation versus frozen encoder with seed 20260914 and
another untouched validation cohort, excluding all earlier study and audit UIDs.
It keeps the exact Phase46 train-only index and normalization because training
records and scientific feature code are unchanged. Contracts authenticate those
source hashes before allocation. Both arms retain the historical encoder81096,
70,000 events, 4,376 steps, identical decoder settings and unchanged strict gates.
This checks whether the tradeoff repeats before a larger pretraining campaign.
A new cohort also changes event composition, so cross-phase differences are not
isolated seed effects. Equal steps do not imply equal FLOPs when encoder update
counts differ. All full/half roots and returned beam candidates remain mandatory.

The dashboard records the Phase47 submission snapshot. Each bounded job requests
one H100 NVL, eight CPUs, 64 GiB memory and a 36-hour limit without requeue.
No automatic promotion or sealed-test access is authorized. Historical production
NO-GO remains unchanged. CPU checks establish software contracts, not trained
physics performance.

Validation before submission
----------------------------

The complete CPU suite passed: 1,640 passed, 34 skipped and 117 warnings. Both
two-step CPU training smoke checks passed. Campaign isolation, untouched cohorts,
complete metric downloads and retained-tree publication checks are covered by
focused regressions. Final site and submission checks are recorded after freezing
the training source. These results validate software contracts, not physics quality.

Both Phase47 jobs were submitted and released from immutable source
``595a150d7d1f2318e9c550ea1a5935e47c75407d``. The frozen checkout passed 128 additional
preflight tests. The new cohort excludes 29,609 previously inspected validation
UIDs; all required overlaps are zero. The dashboard submission snapshot binds
the receipt and source revision without publishing operational records.
