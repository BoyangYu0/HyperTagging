Phase47 reconstruction review
=============================

Late adaptation selected 139/3,738 complete target source sets plus mother PID
(3.72%), versus 133/3,738 (3.56%) with a frozen encoder. This primary does not
require recursive topology. Both arms recover three exact nontrivial components
in each retained scope and zero coherent event forests. Neither passes every
strict gate. The Phase46 primary advantage repeats; its frozen topology advantage
does not. No model is promoted.

Source and population boundaries
--------------------------------

Both arms completed 4,376 steps on 70,000 training events from immutable corrected
source ``595a150d7d1f2318e9c550ea1a5935e47c75407d``. They use the historical
pretrained step 81,096 and the authenticated Phase46 train-only statistics.
The control freezes the encoder for 2,188 steps then adapts at multiplier 0.05;
the frozen arm retains that freeze for all 4,376 steps. Both keep the transferred
PID head frozen. Training and replay seed is 20260914. The scientific code is
unchanged from Phase46, but both seed and validation cohort differ; cross-phase
differences are not isolated seed uncertainty.

Checkpoint selection uses 2,000 validation events, including 1,000 rollout events.
Strict inference uses 100 disjoint events; beam uses its fixed twenty-event subset.
The cohort excludes 29,609 earlier study and audit UIDs. Both primary checkpoints
are selected at step 4,376. All seven evaluation views per arm, exact native
repeats, input, contract, checkpoint and report hashes and saved tensor finiteness are
verified. No sealed test was accessed. Alternate checkpoints and repeats are
records, not independent trained-model trials.

The :doc:`dashboard <_generated/status/index>` retains every original strict gate,
original policy metric and full/half beam ranking, alongside additive all-retained
populations. Neither arm passes full-root, full-LCAG, exact-mother, full-recall,
half-recall, half-LCAG or half-perfect-LCAG gates. Gate names containing “nonzero”
retain their preregistered numeric thresholds. Daughter-sum closure and structural
validity do not establish useful physical reconstruction.

Every retained tree and candidate
---------------------------------

Each arm scores all 2,818 full units and 2,264 half/component units. Full units
contain 2,611 isolated leaves, 82 single-source composites and 125 nontrivial
components. Half units contain 2,035 isolated leaves, 82 single-source composites
and 147 nontrivial components. All unit records are available; unavailable unit counts are zero.
Some individual ratios, such as LCAG for isolated leaves, are undefined.
Incompatible targets remain failed trials. No missing roots or B hemispheres are
invented; continuum uses explicitly retained components.

Only 86/125 full and 98/147 half nontrivial components pass representability
checks. Only 22/100 events have no flagged incompatibility. These necessary
checks are not an achievable efficiency prediction. Isolated leaves do not earn
trivial LCAG successes, and high source coverage largely measures preservation
of detector inputs rather than hierarchy recovery.

Both arms have full LCAG 9/4,144 and mother coverage 9/527, with perfect nontrivial
components 3/125. Both have half LCAG 9/2,288 and mother coverage 9/505, with
perfect nontrivial components 3/147. Coherent forest agreement is 0/100 in both
scopes. Every perfect component in every view and candidate has retained depth
one; no exact deeper component was found.

Full source recall is 2,718/3,374 control versus 2,715/3,374 frozen; precision is
2,718/2,942 versus 2,715/2,937. Half recall is 2,164/2,798 versus 2,162/2,798;
precision is 2,164/2,372 versus 2,162/2,369. These are ratios of summed counts
(micro). The separate aggregation supplement reports macro event/unit means;
they are not substituted for micro results. Full/half perfect source, PID and
topology, depth and representability breakdowns, calibration and PID confusion
are preserved in the native reports and detailed scalar exports.

All 42 returned beam candidates per arm (84 total) receive both full and half
scores. The dashboard reports each deployable model-only ranking separately
from the coherent post-inference oracle and original per-unit Oracle@K bounds.
The bounds may select different hypotheses for different units and therefore
do not describe one coherent reconstructed event. No ranking or coherent oracle
recovers a coherent forest (0/20). Search generation, pruning and ranking remain
truth-free; oracle scoring occurs only afterwards. All proposal bounds, candidate
indices, scores, pruning and diagnostic views remain in the native artifacts.

Explicit macro and exact-PID populations
---------------------------------------

Full LCAG is 0.2172% micro (9/4,144), 5.8713% unit-macro over 125 defined
nontrivial units, and 6.9831% event-macro over 69 events with defined LCAG.
Half LCAG is 0.3934% micro (9/2,288), 5.0076% unit-macro over 147 defined units,
and 6.9925% event-macro over 69 events. Both arms have these same values.
The other 31 events have no defined LCAG denominator; they are reported as
unavailable for this ratio, not successful trees. Defined failures remain zeros.
These means weight units/events equally and must not replace pooled pair counts.

The supplement separately counts exact source sets, exact source sets with every
leaf PID, exact source/topology, and exact source/topology with every leaf and
mother PID. On the primary nontrivial population all four criteria yield 3/125
full and 3/147 half in each arm. This agreement is measured, not assumed from
LCAG alone. Source-only successes also require structural validity and target
representability; incompatible units remain failed trials. Isolated leaves can
pass source checks but never receive trivial topology credit. Every conjunction
and denominator, including separate all-retained counts, remains downloadable.

Paired uncertainty and interpretation
-------------------------------------

The paired event-cluster bootstrap uses 10,000 resamples and seed 20260913.
It computes ratios of summed counts on the same 100 strict events. Retained
LCAG and mother-coverage counts match exactly event by event, giving zero paired
differences and degenerate percentile intervals. This does not imply zero
population uncertainty. Full frozen-minus-control source recall is -0.089
percentage points (95% interval -0.494 to +0.298); precision is +0.055 points
(-0.479 to +0.601). Half recall is -0.071 points (-0.461 to +0.332); precision
is +0.031 points (-0.499 to +0.646). These intervals condition on these trained
models; they do not quantify training-seed variation or the selection primary.

Physical mother momentum resolution is unavailable because genuine truth mother
four-vectors were not retained. Daughter-sum closure is an implementation
invariant and cannot substitute for physical momentum resolution.

Complete evidence
-----------------

The dashboard provides lossless downloads of 16,172 original and 19,937 retained
aggregate scalar metrics, exact registries and source hashes, preserving earlier
phase downloads. An additional 5,040 registered aggregation scalars report
micro, unit-macro and event-macro populations and exact source/PID/topology
conjunctions across every view and beam ranking. The reproducible review bundle contains all fourteen native
reports and logs, original gates, receipts/contracts, 148,974 original detailed
metrics, 317,272 retained detailed metrics, 8,383,056 per-tree/candidate scalars,
and 731,566 training-history/checkpoint scalars. CSV/JSON equivalence and archive
member checksums are verified separately. Native operational evidence remains
outside the public documentation boundary.

Allocation based on all studies
-------------------------------

The :doc:`Phase41 synthesis <phase41>` covers the early studies; the
:doc:`Phase46 review <phase46>` preserves the corrected-source comparison.
Phase34 did not establish reliable downstream gains from longer pretraining;
repaired Phase35 tied frozen and late adaptation under an older decoder.
Phases36-39 improved local metrics without reliable full hierarchies. Historical
Stage A metrics are different measurements and do not replace full-decay scores.

Phase40 changed dataset size, steps and cohort together. Phase41 reduced pointer
weight lost primary recovery (143/3,620 versus 134/3,620). Phase42 longer
pretraining also had a lower observed primary (141/3,938 versus 136/3,938).
Phase43 earlier adaptation gave a small gain (136/3,592 versus 140/3,592), reversed
in Phase44 (185/3,496 versus 173/3,496), which also had a stale replay-seed
deviation. Phase45 doubled late encoder learning rate was neutral at 148/3,805
and predates corrected scientific source. Phase46 cannot causally compare with
Phase45 because source and cohort changed together.

Phase46 favored late adaptation on primary recovery (120/3,854 versus 110/3,854)
but frozen on exact components (nine versus six). Phase47 again favors late
adaptation on the primary, while strict topology ties. Two distinct seeds and
cohorts do not establish a general advantage, and neither result supports
promotion or a full hierarchy claim.

Hold dataset size at 70,000. No controlled learning curve shows data limitation.
Target representability and deeper exact reconstruction remain unresolved.
Better supervision and representation quality are stronger research hypotheses
than simply adding epochs, but corrected pretraining benefits remain unmeasured.
Do not infer that longer pretraining or a larger dataset fixes these failures.

One bounded next campaign: late PID-head adaptation
---------------------------------------------------

Phase48 compares the unchanged late-adaptation control with an arm that unfreezes
the transferred detector PID head after 2,188 steps. Both use late encoder
adaptation, the same historical checkpoint, 70,000 training events, 4,376 steps,
batch 64 and identical loss weights, statistics, policy and gates. Only PID-head
freeze duration differs (4,376 versus 2,188); its existing learning-rate multiplier
is 1.0 in both arms. Training/replay/cohort seed is 20260915.

This tests whether adapting the historical PID head to corrected downstream
context improves PID-dependent reconstruction. The trainer already supports this
schedule and supervised detector PID loss. It is neither corrected pretraining
nor a repair of structurally incompatible targets. Preserve charge-valid PID,
soft-decision/hard-construction semantics and exact recursive daughter sums.
Jointly assess the primary, detector PID diagnostics, retained full/half topology,
source metrics and coherent forests against unchanged strict gates.

The fresh selection and strict cohorts exclude all 31,709 prior study/audit UIDs,
including Phase47. Each job is bounded to one H100 NVL, eight CPUs, 64 GiB and
36 hours, with held atomic release, source-bound contracts and preflights.
The dashboard records the submission snapshot. No campaign chain, automatic
promotion or sealed-test access is authorized. Historical production NO-GO
remains unchanged; CPU fixtures establish software contracts, not physics quality.

Validation
----------

The complete CPU suite passed: 1,652 passed, 34 skipped and 117 warnings. Both
two-step CPU training smoke checks passed. The campaign and PID contract tests
passed, including fresh cohort isolation and the single-factor freeze schedule.
Publication requires a warning-free docs build, privacy and local-link checks,
correct-revision GitHub checks and live HTTP download verification. These are
recorded separately from scientific measurements.

Both Phase48 jobs were submitted and released from immutable source
``1d3424ed5a0dc12e8ec10039a951919710131a90`` after 136 frozen-source preflight tests passed.
The submission snapshot is not a live scheduler claim. The external review
manifest records actual running-state and live-publication verification.

The later aggregation supplement passed 64 focused tests, including explicit
checks that undefined ratios are not successes and exact topology does not
automatically imply correct leaf PID. It adds reporting only; the immutable
submitted training source and its bound evidence remain unchanged.
