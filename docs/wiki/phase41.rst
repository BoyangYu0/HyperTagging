Phase41 closeout and next training decision
===========================================

Phase41 completed, but exact reconstruction remains the limiting outcome.
Keep 70,000 training events for the next experiment and test pretraining
transfer with the current decoder before buying a larger data or training
budget. This is an allocation recommendation, not evidence that additional
pretraining already helps.

Phase41 evidence
----------------

Both arms completed 4,376 steps. The pointer32 control selected step 4,376
with 143/3,620 complete targets (3.9503%); the level-1 pointer24 arm selected
step 4,000 with 134/3,620 (3.7017%). These are micro rollout checkpoint-selection
metrics, not event-level full-tree success rates. Teacher-forced selection
uses 2,000 validation events; the rollout track uses 1,000 of them.

On the separate 100-event strict validation cohort, both reconstructed zero
configured full roots, one exact topology-matched mother out of 153, and one
correct full-tree LCAG pair out of 2,769. That single topology-matched mother
has the wrong PID in both arms: full-tree mother PID accuracy is 0/1.
Mother coverage therefore does not mean PID-correct mother reconstruction.
Full-tree source recall was 42/288
(14.58%) for control and 38/288 (13.19%) for pointer24. Half-tree recall was
104/580 (17.93%) versus 98/580 (16.90%); half-tree precision was 104/198
(52.53%) versus 98/208 (47.12%). Both reached three perfect half LCAG targets
out of 138 available targets. Full evaluation availability was 15/100 full
units and 138/200 half units; conditional decay metrics must not be read as
rates over all input events. The event-level full-root denominator stays 100.

Reducing the level-1 positive weight raised micro pointer precision from
15.94% to 16.82%, but recall fell from 75.86% to 69.13%. Half LCAG fell from
9/1,786 to 7/1,786. Neither arm passed all preregistered gates. Repeated strict
primary metrics were identical; structural validity and daughter-sum closure
passed. Valid trees and exact daughter sums do not establish correct physics.

Inspection of the reconstructed forests supports this diagnosis: across the
100 events the control creates only 254 mothers and leaves 3,088 input FSPs
unassigned; pointer24 creates 257 mothers and leaves 3,093 FSPs unassigned.
Of the 600 event-level decoding opportunities, 394 and 377 respectively are
empty. More predicted B roots (8 versus 14) in pointer24 do not translate into
better full-tree or half-LCAG reconstruction. These are aggregate inference
counts over all events, distinct from available-target decay denominators.

The :doc:`metric dashboard <_generated/status/index>` contains the complete
aggregate metric download, including every checkpoint track, calibration,
PID confusion counts, strict and contracted diagnostics, and both scopes for
all beam rankers. Oracle-at-k remains diagnostic. Category and target-shape
breakdowns are retained in the full local review artifact. Physical mother
momentum resolution is unavailable because truth mother four-vectors were
not retained; the zero daughter-sum residual is not a resolution measurement.

On the fixed 20-event beam subset, no registered model-only ranker improves
LCAG counts over greedy. Control yields 1/567 full LCAG pairs and 3/408 half
LCAG pairs; pointer24 yields 1/567 and 2/408. The registered diagnostic oracle
has the same LCAG counts. This points to limited good hypotheses under the
tested width-four, twelve-proposal bounds; it does not rule out other search
or structured-decoding methods. The oracle table uses the registered
lexicographic topology diagnostic, so it is not an independent maximum of
every metric (for example source recall). The download also retains the
separate per-unit oracle-at-k metrics.

What the earlier studies establish
----------------------------------

* Early pretraining, capacity, transfer, and Stage A runs established runtime
  and structural contracts but did not establish promotion-grade reconstruction.
  Stage A aggregate edge-F1 comparisons lacked the paired event evidence needed
  for promotion. Their legacy tree metrics are not comparable to today's strict
  full-decay metrics.
* Phase34 compared pretrained steps 54,064, 81,096, and 108,128. The original
  downstream primary favored the earlier checkpoint, while later checkpoints
  passed pretraining gates. No checkpoint satisfied both selection conditions.
  Two fresh 1,000-event confirmations found earlier-minus-81,096 primary
  contrasts of -0.002 (95% interval -0.010 to 0.006) and -0.005 (-0.013 to 0.003).
  They did not establish an advantage for the earlier checkpoint. These are
  historical decoder results, not a modern pretraining ranking.
* Phase35 introduced depth-balanced replay and protected teacher supervision;
  the repaired frozen and late-adaptation arms tied at 1.5788% micro complete
  targets. This did not establish a transfer advantage from that adaptation.
* Phase36's weighted recovery arm cleared its then-minimal gates, but evidence
  amounted to one root, one LCAG pair, and one mother; confirmation was required.
* Phase37-39 tested teacher anchoring and pointer weighting. Phase38's within-study
  pointer24 arm reached 3.5527% versus 3.2498% for pointer16. Phase39's pointer32
  reached 3.5403% versus 3.3769% for pointer24. These modest gains motivated the
  current control; they did not solve exact hierarchy reconstruction.
* Phase40r1 doubled data from 35,000 to 70,000 and budget from 2,188 to 4,376
  steps. Its control reached 4.5015% versus 3.4521% for enlarged query capacity.
  Neither passed all strict gates. The different cohort and larger step budget
  prevent attributing the change from Phase39 to dataset size alone.
* Phase41's fresh cohort and weight ablation still show sparse exact topology,
  with no improvement from lowering the level-1 positive weight. Cross-phase
  percentage differences are descriptive, not paired treatment effects.

Dataset size and pretraining
----------------------------

Do not expand the dataset again now. Data scaling has not been isolated from
compute scaling, and no train-versus-validation learning curve establishes a
data-limited regime. The current failure pattern motivates investigating
representation transfer and structured decoding. Larger data may still help;
these studies do not prove otherwise. Revisit data size after a controlled
transfer result, using equal-compute and equal-exposure comparisons and
reporting rare-category coverage separately.

Do not simply extend pretraining on the basis of its own loss. Later
pretraining improved its objectives without a proven downstream benefit in
Phase34. Phase42 therefore reuses the existing 81,096-step control and
108,128-step candidate, retraining the current decoder under identical
conditions. The pretrained encoder and PID state jointly define the changed
factor. This measures the utility of additional pretraining for the current
reconstruction system; it does not isolate which pretraining objective helps.

Phase42 keeps the 70,000-event training selection, paired training seed,
4,376-step schedule, pointer32 loss, pointer/object thresholds 0.35/0.60,
late encoder adaptation, and all existing strict gates. It uses a new
2,000-event selection cohort and separate 100-event evaluation cohort with
zero overlap with earlier selection/evaluation cohorts; beam scoring uses
20 of those evaluation events. A candidate must improve the primary and pass
all unchanged hierarchy gates before additional pretraining is treated as a
promising allocation. A single seed and sparse outcomes still require later
replication. There is no automatic promotion or sealed-test access.
