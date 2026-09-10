Phase42 closeout and next training decision
===========================================

More pretraining did not improve the primary reconstruction result in this
controlled study. Keep 70,000 training events and the 81,096-step pretrained
checkpoint. Next, test whether earlier task-specific encoder adaptation helps.
This is an allocation decision under limited evidence, not proof that larger
datasets or improved pretraining objectives cannot help.

Phase42 evidence
----------------

Both arms completed 4,376 reconstruction steps with the same data, seed,
decoder, thresholds, and validation cohorts. Only the pretrained checkpoint
changed, including its encoder and transferred PID state. The control selected
step 4,376 with 141/3,938 micro complete targets (3.5805%); the 108,128-step
candidate selected step 4,000 with 136/3,938 (3.4535%). The difference is only
five targets, or 0.127 percentage points. This is a descriptive single-seed
comparison, not evidence of statistically established harm from pretraining.

These checkpoint-selection rates use 1,000 rollout validation events from the
2,000-event selection cohort. They are not full-event success rates. On the
separate strict 100-event cohort, both produced zero completed full roots.
Control versus candidate results were:

.. list-table::
   :header-rows: 1

   * - Strict metric
     - Pretrained 81,096
     - Pretrained 108,128
   * - Full source recall
     - 47/353 (13.31%)
     - 51/353 (14.45%)
   * - Full source precision
     - 47/53 (88.68%)
     - 51/59 (86.44%)
   * - Full LCAG pairs
     - 2/3,384
     - 1/3,384
   * - Topology-aligned mothers
     - 2/176
     - 1/176
   * - Half source recall
     - 124/656 (18.90%)
     - 119/656 (18.14%)
   * - Half source precision
     - 124/251 (49.40%)
     - 119/239 (49.79%)
   * - Half LCAG pairs
     - 13/2,109
     - 9/2,109
   * - Perfect half LCAG units
     - 2/149
     - 1/149

Full target availability was 19/100 units; half availability was 149/200.
Full mother PID accuracy was 2/2 versus 1/1, conditional on source/topology
alignment. Those tiny conditional denominators must not be confused with
overall mother reconstruction coverage. Full leaf PID accuracy was 279/353
versus 277/353. Half mother PID accuracy was 12/13 versus 9/9.

The control failed four preregistered gates: full-root completion, full source
recall, half source recall, and perfect half LCAG. The candidate failed seven,
also missing the minimum full LCAG, aligned-mother, and half LCAG counts.
The unchanged gates, not a favorable isolated metric, determine the decision.
Both strict repeats matched exactly, and both recorded structural guardrail
checks passed. No model is promoted and the sealed test remains closed.

Forest inspection shows sparse composition: control created 263 mothers and
left 3,249 input FSPs unused, versus 250 and 3,256 for the candidate. Empty
decoding levels numbered 390 versus 402 across 600 event-level opportunities.
The candidate predicted more B roots (7 versus 4), but no complete full root.
Teacher-forced level-1 pointer precision remained about 15.6% versus 15.5%,
with recall 74.8% versus 75.8%. Higher local recall did not become better
complete hierarchy reconstruction.

Beam search and metric completeness
-----------------------------------

The fixed 20-event beam subset uses width four and twelve proposals. Control
greedy and every registered model-only ranker recover 1/647 full LCAG pairs
and 3/425 half LCAG pairs. For the candidate, average-link and normalized-joint
ranking recover one full LCAG pair where greedy recovers zero; every ranker
still recovers only 2/425 half LCAG pairs. The diagnostic oracle reaches the
same LCAG counts as the best model-only ranking. This limited search contains
few exact hypotheses; it does not exclude benefits from different proposals
or structured decoder objectives. The oracle is a lexicographic topology
diagnostic, not a per-metric maximum and never an inference-time truth signal.

The :doc:`dashboard <_generated/status/index>` provides the complete aggregate
download: all checkpoint tracks, micro/macro training metrics, strict and
repeated evaluations, contracted diagnostics, calibration, PID confusion,
availability, structural metrics, and both beam scopes and all rankers. The
full local CSV/JSON additionally retains category and target-shape breakdowns.
Reports and checkpoint files were verified against immutable run receipts;
already-completed inference was not rerun merely to rebuild the dashboard.

Physical mother momentum resolution remains unavailable because retained truth
mother four-vectors are absent. Daughter-sum closure is an implementation
invariant, not a momentum-resolution result. Global forest detector-resource
disjointness is reported separately from recursive accepted-mother validity;
the full metric download preserves that distinction.

Decision across the studies
---------------------------

The :doc:`Phase41 synthesis <phase41>` reviews the earlier studies. Stage A
legacy metrics are not directly comparable to strict full-decay results.
Phase34 did not establish a reliable downstream advantage from longer
pretraining; Phase35's repaired frozen/late-adaptation comparison tied under
the older decoder. Phase36-39 improved some local and rollout metrics without
establishing robust complete hierarchies. Phase40 changed data, compute, and
validation cohort together, so it did not isolate a dataset-size effect.
Phase41's pointer-weight change traded recall for precision without solving
the hierarchy. Phase42 now provides a current-decoder pretraining-duration
comparison, and the later checkpoint fails its predefined benefit criterion.

Do not increase dataset size or simply extend pretraining duration now. There
is no controlled learning curve demonstrating a data-limited regime, and
additional pretraining has not passed downstream acceptance. Improving
representation learning remains plausible, but it should be judged by strict
downstream reconstruction rather than pretraining loss alone. Data scaling can
be revisited with equal-compute and equal-exposure comparisons and rare-category
coverage. Cross-phase percentages from different cohorts are descriptive only.

Phase43 holds the 81,096-step checkpoint, dataset, decoder, paired seed,
learning rates, 4,376 total steps, and all hierarchy gates fixed. The control
freezes the encoder for 2,188 steps; the candidate adapts it from step zero.
The PID head remains frozen, and encoder learning rate remains 0.05 times the
decoder rate. A fresh 2,000-event selection and separate 100-event strict cohort
exclude all previous selection/evaluation cohorts; 20 strict events support beam
evaluation. No sealed-test access or automatic promotion is authorized.

This tests whether task-specific adaptation should start earlier, not whether
a particular pretraining objective is optimal. The candidate receives more
encoder updates by design; equal total steps do not imply identical FLOPs or
wall time. Earlier adaptation may help or may damage useful pretrained features.
A benefit requires an improved primary and every unchanged hierarchy gate;
single-seed results still need later replication.
