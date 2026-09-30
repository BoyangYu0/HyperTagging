# Phase63 completed paired review — 2026-09-30

Both Phase63 arms completed 4,376 reconstruction updates and all seven native
validation views. Masking is the simpler baseline for the next experiment;
**neither arm passes every original gate or establishes recursive reconstruction
quality**. There is no demonstrated benefit from the unmatched-slot recovery
objective, and no reason to continue recovery-weight dose tuning.

## Execution and evidence

Recovery job 16742232 completed with exit 0 on September 29; replacement masking
job 16760614 completed with exit 0 on September 30. The original masking job
16742233 remains a preserved zero-update failure. Its inactive-weight admission
repair is the only scientific source difference between the two immutable
checkouts. The completed recovery run was not repeated.

The closeout validates native receipt checksums, submitted contracts, input and
checkpoint hashes, cohort separation, balanced replay, all fourteen reports,
both retained scopes and every returned beam candidate. Both primary repeated
reports are exactly identical. The two arm configurations differ only in the
registered recovery policy and coefficient. Both use the same frozen Phase62
control encoder, seed, 70,000 training events and compute budget. Their selected
checkpoints are recovery step 4,376 and masking step 4,000; both selections use
only the reserved 1,000-event selection cohort. Strict evaluation uses the same
100 events, with beam evaluation on its fixed 20-event subset.

Source commits are `aeefabdaab6f05bce817081f43109c0a8641827d` (recovery)
and `e543c1ae716946f1140a96e1fa5a705e83be55d7` (masking repair).
The compact evidence is
[the closeout](../artifacts/codex/reconstruction_phase63_closeout_20260930.json)
and [paired uncertainty](../artifacts/codex/reconstruction_phase63_paired_uncertainty_20260930.json).
The complete 164,784-row scalar export and verification logs remain on the project data
volume under `HyperTagging_artifacts/phase63_closeout_20260930`; native tree
reports and checkpoints remain in their immutable training-source directories.
No training outputs or large files are added to Git.

## Primary results and populations

| Endpoint | Corrected recovery | Masking only |
| --- | ---: | ---: |
| Selection source-set plus mother PID | 292/3,718 | 310/3,718 |
| Configured full roots | 32/100 | 44/100 |
| Original full LCAG | 7/3,474 | 8/3,474 |
| Original exact mother coverage | 7/169 | 8/169 |
| Original full source precision | 81/99 | 74/94 |
| Original full source recall | 81/351 | 74/351 |
| Original half LCAG | 22/2,158 | 25/2,158 |
| Original half perfect LCAG | 2/163 | 2/163 |
| Original half root PID | 7/163 | 9/163 |
| Original half source precision | 158/367 | 157/334 |
| Original half source recall | 158/708 | 157/708 |
| Retained full LCAG | 24/3,939 | 27/3,939 |
| Retained half/component LCAG | 23/2,168 | 27/2,168 |
| Exact nontrivial retained full components | 3/155 | 3/155 |
| Exact nontrivial retained half/components | 2/173 | 3/173 |
| Coherent retained forests | 2/100 | 2/100 |

Recovery fails half-source precision and half-perfect-LCAG gates. Masking passes
the precision floor but still fails the original half-perfect-LCAG gate (2,
minimum 3). Retained component counts use a different population and cannot
replace this failed gate. Structural validity, closure and repeat gates pass.
Configured root construction does not imply correct topology or B-pair recovery.

All exact primary nontrivial components have two leaves, one mother and depth
one. No exact component deeper than one appears in any of the fourteen views
or any of the 119 returned beam candidates (60 recovery, 59 masking). Of the
two primary coherent forests, recovery has one with no mothers and one with a
single depth-one mother; both masking forests contain no mothers. All these
forest successes have explicit components without a B partition. Thus the
apparent forest tie hides a loss of the sole nontrivial forest in masking;
there is no successful deep tree or B-pair reconstruction here. Recovery constructs 349 mothers and 18 B roots, versus 303 mothers and one
B root for masking. These output counts are not counts of correct trees.

The retained full population includes isolated leaves. Its aggregate source
recall rises from 3,059/3,763 to 3,112/3,763 and precision from 3,059/3,394 to
3,112/3,380. Those gains must not be described as better recursive reconstruction.
Restricting to components with at least two truth sources gives full recall
230/728 versus 225/728 and precision 230/302 versus 225/301. Corresponding half
counts are recall 242/728 versus 247/728 and precision 242/349 versus 247/352.
The original root-based source metrics above remain separate.

## Paired uncertainty

The analysis resamples the 100 matched events in 10,000 paired bootstrap draws
(seed 20260930), summing numerator and denominator counts before division. All
reported differences below are masking minus recovery, in percentage points.

| Retained endpoint | Difference | Paired 95% interval |
| --- | ---: | ---: |
| Full LCAG | +0.076 | −0.079 to +0.239 |
| Half/component LCAG | +0.185 | −0.102 to +0.488 |
| Exact full components | 0.000 | −3.054 to +3.145 |
| Coherent forests | 0.000 | −3.000 to +3.000 |
| Nontrivial full source recall | −0.687 | −3.719 to +2.238 |
| Nontrivial full source precision | −1.408 | −7.965 to +5.242 |
| Nontrivial half source recall | +0.687 | −1.282 to +2.815 |
| Nontrivial half source precision | +0.829 | −4.872 to +6.510 |

Aggregate retained source intervals are positive, but include isolated leaves.
The topology and nontrivial-source intervals span zero. These exploratory
intervals are conditional on these two selected models, have no multiplicity
correction, and do not measure training-seed uncertainty. Same seed does not
prove a bitwise identical common prefix. Different Phase62/63 cohorts prevent a
causal cross-phase improvement claim.

## Search review

The following are retained top-1 results on the same 20 events. Each cell gives
LCAG pairs / exact nontrivial components / coherent forests. Full denominators
are 1,393 / 34 / 20; half/component denominators are 710 / 38 / 20.

| Model-only ranking | Recovery full | Masking full | Recovery half | Masking half |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 5 / 0 / 0 | 6 / 0 / 0 | 5 / 0 / 0 | 6 / 0 / 0 |
| Average link probability | 5 / 2 / 1 | 5 / 1 / 0 | 7 / 2 / 1 | 6 / 1 / 0 |
| Learned confidence mean | 6 / 2 / 1 | 5 / 1 / 0 | 6 / 2 / 1 | 7 / 1 / 0 |
| Learned confidence sum | 5 / 0 / 0 | 5 / 0 / 0 | 6 / 0 / 0 | 5 / 0 / 0 |
| Normalized joint probability | 7 / 2 / 1 | 5 / 1 / 0 | 7 / 2 / 1 | 7 / 1 / 0 |

Search does not give a consistent masking advantage. Component-wise oracle
results are diagnostics after truth-free generation, not deployable rankers
and not evidence that one global hypothesis satisfies every component. Preserve
greedy and the registered search limits; do not tune rankings or thresholds on
this strict cohort. Candidate calibration and offline/basf2 search parity remain
separate studies.

## Next experiment specification

Use masking-only supervision as a simpler experimental baseline, retaining the
legal-daughter, confidence-mask and admission fixes. This is a complexity choice
in the absence of demonstrated recovery benefit, not checkpoint promotion.
Hold the 70,000-event training set, normalization, capacities, PID policy,
reconstruction losses, decoder, thresholds and beam budget fixed. Do not increase
radius-loss weight to compensate for a saturated projection.

The next priority is a bounded radial-projection diagnostic, followed by one
controlled reconditioning experiment if its mechanism checks pass:

1. On existing training/development events, measure pre-cap norms, radial
   derivatives, radius variance by level, and gradient support on the frozen
   initial encoder and both completed Phase63 encoders. Keep strict and sealed
   test events out of calibration. The earlier 32-event observation establishes
   saturation only for its measured initial-encoder sample.
2. Compare an unchanged projection with one explicitly specified radial
   reconditioning intervention. Fit any scale on training data only. Preserve
   direction, FP32 geometry, feature contracts and input truth separation;
   checkpoint metadata must distinguish the new parameterization. Verify finite
   gradients and usable radial spread before spending a full training budget.
3. Freeze the exact transform, initialization rule, source, diagnostic admission
   thresholds and equal refinement/reconstruction budgets before evaluation.
   Both arms must receive equal refinement compute. Use the same masking
   reconstruction baseline in both; do not combine a parent/PID loss sweep,
   geometry intervention and search change in one contrast.
4. Reserve a fresh 1,000 selection / 100 strict / 20 beam subset from the
   46,700 unreserved validation events, excluding all Phase63 and earlier
   reservations. Report all original gates plus nontrivial/depth-stratified
   topology, coherent forests, source metrics and paired uncertainty.
5. A useful outcome requires usable radial gradients **and** joint held-out
   recursive quality improvements. More radius variance, generated depth or
   isolated-leaf recovery alone is insufficient. Replicate across training seeds
   before a quality claim.

This is a proposed next study, not a rendered/submitted Phase64 contract: the
post-training geometry diagnostic and exact intervention remain to be established.
The Phase63 continuation closes the registered pair; no automatic scientific
successor, promotion or sealed-test access is performed.

## Verification

The native closeout and paired analyses ran successfully against both immutable
sources. The focused CPU checks passed 62 tests (one expected legacy-fixture
warning), covering Phase63 campaign contracts, inactive recovery admission and
retained-tree validation. The earlier complete 1,977-test suite remains the
training-code verification record; this continuation changes review/export
scripts and documentation only. Documentation-specific checks and audit lineage
are recorded in the current verification ledger. A full local HTML build and
remote publication are not implied by a source update.
