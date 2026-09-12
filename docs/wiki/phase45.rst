Phase45 reconstruction review
=============================

Both learning-rate arms completed 4,376 steps on the same 70,000 training events.
Doubling the encoder learning-rate multiplier from 0.05 to 0.10 after a 2,188-step
freeze did not improve the primary: both selected 148/3,805 complete target
source sets plus mother PID (3.89%). This metric does not require exact recursive
topology. Neither arm passes all preregistered strict gates; no model is promoted.

Source and population boundaries
--------------------------------

These are immutable native Phase45 results from before the independent scientific
audit fixes. That audit repaired reconstructed charge, daughter-physics targets,
corrupted structural labels and source reuse, among other issues. Historical
weights and reports are preserved. Phase45 is a completed preregistered study at
its recorded source; it does not validate corrected training or inference.

Checkpoint selection uses 2,000 validation events, including 1,000 rollout events.
Strict evaluation uses a disjoint 100-event cohort and beam its fixed 20-event
subset. Training, replay and cohort seeds agree; all seven views per arm are
complete and primary repeats are exact. No sealed test was accessed.

The :doc:`dashboard <_generated/status/index>` provides both complete aggregate
metric downloads, checkpoint and beam comparisons, original acceptance gates,
and the separate all-retained population. Original policy denominators remain
unchanged. Original full LCAG is 1/3,986 and half LCAG 9/2,427 in both arms.
Configured full-root construction is 0/100; it is a model diagnostic, not proof
of truth-exact reconstruction.

Every retained tree and beam candidate
--------------------------------------

All 3,084 full units and 2,439 half/component units are checked in each arm,
including training-incompatible roots. Full units comprise 2,864 isolated leaves,
75 single-source composites and 145 nontrivial components. Half units comprise
2,196 isolated leaves, 75 single-source composites and 168 nontrivial components.
There are 23 explicit B partitions and 77 labelled component fallbacks; missing
initial roots or B hemispheres are never invented. Isolated leaves do not receive
trivial LCAG successes. Only 108/145 full and 116/168 half nontrivial targets pass
the checkpoint representability checks; incompatible targets remain failed trials.

Both arms have full LCAG 12/4,405 and matched-mother coverage 12/555; half/component
LCAG is 11/2,448 and mother coverage 11/532. Perfect nontrivial components are
6/145 full and 6/168 half. Every perfect component has two leaves, one mother
and retained depth one; no deeper exact component was recovered. Coherent retained-forest agreement is 0/100 in both
scopes. Source coverage largely measures isolated-input preservation, rather than
successful hierarchy construction. Conditional mother PID accuracy is 12/12 full
and 11/11 half; these small matched populations do not establish overall PID quality.

All 37 control and 35 candidate beam hypotheses receive full and half checks.
Greedy recovers two exact nontrivial components on the twenty-event subset.
Average-link ranking and the coherent post-inference oracle recover three
(3/37 full, 3/39 half), in both arms. LCAG remains five correct pairs (5/345 full,
5/226 half) and coherent forest agreement remains 0/20. Other model-only rankings
retain two perfect components. Oracle@K per-unit maxima are labelled bounds and
must not be interpreted as one coherent reconstructed event.

Physical mother momentum resolution is unavailable because truth mother
four-vectors were not retained. Daughter-sum closure checks implementation
consistency. A single training seed and small strict cohort limit inference;
paired event-level LCAG and mother-coverage counts are identical, yielding a
zero difference in every paired event bootstrap resample. This is conditional on
these trained models, not evidence of zero population or training-seed uncertainty.
For full source precision, the candidate-minus-control difference is -0.327
percentage points (paired event-cluster 95% percentile interval -0.644 to -0.056,
10,000 resamples). This secondary coverage result does not establish a hierarchy
improvement or justify a promotion.

Complete evidence
-----------------

The public downloads preserve all 16,031 original aggregate entries and 19,036
retained aggregate entries, with source hashes and exact metric registries.
The local complete evidence bundle additionally contains 154,860 original detailed
entries, 297,566 retained detailed entries, 9,179,553 per-tree/candidate scalar
records, and 731,754 training-history/checkpoint scalar records. It preserves
all native evaluation reports, including repeats, alternate checkpoint tracks,
contracted diagnostics, source-category and target-shape breakdowns, PID confusion,
calibration, kinematic availability, all beam rankings and Oracle@K.

Allocation based on the accumulated studies
-------------------------------------------

The :doc:`Phase41 historical synthesis <phase41>` records the earlier studies.
Phase34 did not establish a reliable downstream benefit from longer pretraining;
repaired Phase35 tied frozen and late adaptation with an older decoder.
Phases36-39 improved local metrics without robust full hierarchies. Legacy
Stage A metrics cannot be substituted for strict full-decay measurements.

Phase41 pointer-weight reduction had a lower observed primary (143/3,620 versus
134/3,620). Phase42 longer pretraining also had a lower observed primary
(141/3,938 versus 136/3,938).
Earlier adaptation had a small positive Phase43 difference (136/3,592 versus
140/3,592), then a reversed Phase44 diagnostic difference (185/3,496 versus
173/3,496). Phase44 also has a preregistration seed deviation and is not a clean
replication. Phase45 doubling the late encoder learning rate is neutral on the
primary and retained topology counts. These are within-study comparisons;
different cohorts and source revisions preclude a causal cross-phase trend.

Do not increase dataset size now. Earlier scaling changed data, compute and cohort
together, so it did not establish a data-limited learning curve. Correct training
supervision and target/inference compatibility take priority. Better pretraining
objectives may help, but a benefit is unmeasured; simply extending pretraining
was not supported by Phase42.

Phase46 holds 70,000 events, pretrained step 81,096 and 4,376 total steps. Both
arms use the corrected implementation and freshly fitted train-only statistics.
The control adapts the encoder after 2,188 steps at multiplier 0.05; the comparison
keeps it frozen for all 4,376 steps. This tests the value of downstream adaptation
under corrected supervision, while keeping the historical encoder weights fixed.
It does not test a newly trained corrected pretraining objective. Encoder update
counts differ intentionally; equal step budgets do not imply equal FLOPs.
A fresh cohort excludes all previous study cohorts and the independent audit
validation sample, with matched training/replay
seeds. All retained-tree and beam checks are mandatory alongside unchanged gates.
The dashboard records the submission snapshot. No automatic promotion or
sealed-test access is authorized; historical production NO-GO remains unchanged.

Submission and validation
-------------------------

Both Phase46 jobs were submitted and released from immutable source
``776a48a74e674d31fc3ddadf705e0167b418a699`` after the full CPU suite passed
(1,628 passed, 34 skipped, 117 warnings). Both two-step CPU training smoke checks
also passed from that frozen checkout. Each job requests one H100 NVL, eight CPUs,
64 GiB memory and a 36-hour limit, with no requeue or automatic promotion.
The new cohort excludes 27,509 previously inspected validation events, including
the independent audit sample; all required overlaps are zero. Fresh train-only
statistics correct the common and composite blocks. Query and cardinality
admission report zero overflows. These checks establish software and campaign
contracts, not trained physics performance.
