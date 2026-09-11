Phase43 closeout and next training decision
===========================================

Earlier encoder adaptation produced a small, unconfirmed improvement. Keep
70,000 training events and the 81,096-step pretrained checkpoint. Phase44 will
replicate the adaptation comparison with a new seed and untouched validation
cohort before committing to more data or longer pretraining. Neither Phase43
model passes every hierarchy gate, and neither is promoted.

Phase43 evidence
----------------

Both runs completed 4,376 reconstruction steps. The only arm difference was
encoder freeze duration: 2,188 steps for the control versus zero for early
adaptation. The pretrained checkpoint, decoder, dataset, seed, learning rates,
PID-head freeze, thresholds, selection cohort, and evaluation cohort matched.
Both selected step 4,376. The control recovered 136/3,592 micro complete targets
(3.7862%); early adaptation recovered 140/3,592 (3.8976%). Four additional
targets, or 0.1114 percentage points, are a descriptive single-seed signal,
not a statistically established benefit.

Checkpoint selection uses 1,000 rollout validation events from a 2,000-event
selection cohort. These are target-level efficiencies, not full-event success
rates. Strict evaluation uses a separate 100-event cohort and beam evaluation
its fixed 20-event subset. Cross-phase rates use different cohorts and must not
be interpreted as controlled improvements over earlier phases.

.. list-table:: Strict reconstruction on the shared Phase43 cohort
   :header-rows: 1

   * - Metric
     - Late adaptation
     - Early adaptation
   * - Completed full roots
     - 0/100
     - 0/100
   * - Full source recall
     - 64/456 (14.04%)
     - 67/456 (14.69%)
   * - Full source precision
     - 64/72 (88.89%)
     - 67/77 (87.01%)
   * - Full LCAG pairs
     - 1/4,343
     - 2/4,343
   * - Topology-aligned mothers
     - 1/222
     - 2/222
   * - Half source recall
     - 129/765 (16.86%)
     - 132/765 (17.25%)
   * - Half source precision
     - 129/253 (50.99%)
     - 132/247 (53.44%)
   * - Half LCAG pairs
     - 12/2,622
     - 13/2,622
   * - Perfect half LCAG units
     - 1/154
     - 3/154
   * - Half root PID accuracy
     - 10/154
     - 8/154

The control failed six gates: full-root completion, full and half source recall,
full LCAG count, topology-aligned mother count, and perfect half LCAG count.
Early adaptation failed three: full-root completion and full and half source
recall. Both exact strict repeats and recorded structural guardrails passed.
Improvement in a few metrics does not override the unchanged acceptance gates.

A paired event bootstrap (10,000 resamples, seed 20260911) gives a 95%
percentile interval of -1.75 to +3.00 percentage points for the full-source
recall difference, and -1.67 to +2.43 points for half-source recall. Each
resample keeps both arms and both half units grouped by event, then recomputes
ratios from summed counts. These intervals include zero and condition on the
trained checkpoints; they do not capture training-seed uncertainty or provide
a confidence interval for the separate primary-selection metric.

Forest inspection remains consistent with sparse, inaccurate composition. The
control accepted 270 mothers, left 3,265 input FSPs unused, and had 386 empty
levels across 600 opportunities. Early adaptation accepted 269 mothers, left
3,274 FSPs unused, and had 390 empty levels. Predicted B roots numbered nine
versus eight; neither count represents a correctly completed full root.

Beam search and complete metrics
--------------------------------

Width-four search with twelve proposals was evaluated in both full and half
scopes using all four preregistered model-only rankings. Full LCAG was 0/1,694
for the control and 1/1,694 for early adaptation under greedy, every ranker,
and the diagnostic oracle. For half LCAG, control greedy recovered 2/947;
average-link ranking and the oracle recovered 3/947. Early-adaptation greedy,
confidence mean/sum, normalized-joint ranking, and the oracle recovered 4/947;
average-link recovered 3/947. No beam subset half unit had perfect LCAG.

This bounded search contains few exact hypotheses. It does not establish that
larger search or different proposals cannot help. Oracle results use truth only
after inference and follow lexicographic topology ranking; they are not
per-metric maxima or deployable rankings.

The :doc:`dashboard <_generated/status/index>` and its metric download retain
16,405 aggregate entries: every checkpoint track, micro/macro training metrics,
seven evaluation reports per arm, strict repeats, contracted diagnostics,
calibration, PID confusion, availability, structural metrics, and both beam
scopes and all rankers. The complete local CSV/JSON export contains 157,707
entries including source-category and target-shape breakdowns. Run receipts,
contracts, input bindings, reports, and checkpoint hashes were verified;
recorded checkpoints were finite and strict repeats matched exactly.

Conditional PID accuracy must be read with its topology-alignment denominator.
Global forest resource disjointness and recursive accepted-mother validity are
separate metrics. Physical mother momentum resolution remains unavailable
because retained truth mother four-vectors are absent. Daughter-sum closure
measures an implementation invariant, not physical momentum resolution.

Decision across all studies
---------------------------

The :doc:`Phase41 synthesis <phase41>` covers the historical studies and
:doc:`Phase42 review <phase42>` records the current-decoder pretraining test.
Stage A legacy metrics are not interchangeable with strict full-decay metrics.
Phase34 did not establish a reliable downstream advantage from longer
pretraining; the repaired Phase35 frozen/late comparison tied with an older
decoder. Phase36-39 improved some local metrics without robust full hierarchies.
Phase40 increased data and compute and changed the evaluation cohort together,
so it did not isolate dataset size. Phase41's pointer-weight change traded
recall for precision without solving topology. Phase42's 108,128-step
pretrained checkpoint did not outperform the 81,096-step checkpoint on its
primary endpoint or pass all gates. Phase43's earlier adaptation offers only
a small positive signal, with continued failure of complete reconstruction.

Increasing dataset size is not necessary on the current evidence. There is no
controlled learning curve establishing a data-limited regime. Hold 70,000 and
revisit scaling with equal-compute and equal-exposure comparisons, checking
rare-category coverage rather than assuming aggregate data volume is the issue.

Improving representation learning is a more useful next hypothesis than simply
extending pretraining duration, but no study here proves that a new pretraining
objective is superior. Earlier task-specific adaptation is also not itself a
pretraining-objective change. Replication has priority now: the observed gain
is only four targets, and changing another factor immediately would leave its
reliability unresolved. If replication confirms local improvement but complete
hierarchies still fail, investigate structured representation/decoder objectives
and proposal quality, with downstream topology gates as the success criterion.

Phase44 training contract
-------------------------

Phase44 repeats the late-versus-early adaptation comparison with seed 20260911,
the same 81,096-step checkpoint, 70,000-event training selection, and 4,376 total
steps. The encoder learning-rate multiplier remains 0.05; the PID head stays
frozen. The arm difference remains only encoder freeze steps: 2,188 versus zero.
The candidate intentionally receives more encoder updates, so equal total
steps do not imply identical FLOPs or wall time.

A fresh 2,000-event selection cohort and separate 100-event strict cohort exclude
all earlier tuning/evaluation cohorts and training events; twenty strict events
support beam diagnostics. This tests replication across seed and cohort jointly,
not an isolated estimate of seed variance. Compare paired arms within each phase;
do not pool cross-phase percentages without their numerators and denominators.
Every acceptance gate is unchanged. A favorable direction alone is insufficient:
an improvement claim requires a better primary and all strict gates. No automatic
promotion or sealed-test access is authorized. Two guarded one-GPU jobs receive
36-hour limits and no requeue; the dashboard records submission status.
