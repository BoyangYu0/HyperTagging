Phase78 training-only partition diagnosis
=========================================

This bounded follow-up resolves the question posed by :doc:`Phase77 <phase77>`:
does a small probe expose within-event B-partition information that the native
assignment head leaves unused? **It does not.** This is a limit of the measured
probe and checkpoint, not proof that the representations contain no such
information. No new scientific training, fresh validation reservation, primary
evaluation or model promotion follows this pass.

Design and source support
-------------------------

The preregistered diagnostic replays the fixed Phase76 context-on main and
independent tiny final checkpoints. All1536 authenticated training identities
are retained,256 per category, plus24 tiny views reusing those identities.
No validation or test rows are iterated or evaluated. Detached detector-only
features, queries and logits are saved before truth joins. Native logits
reproduce the preceding trace exactly; no inference, matching, charge, source,
PID, geometry or normalization behavior is changed.

Pair labels mean same-B versus cross-B **within the same collision**, never a
global B label. They require two known B constituents and disjoint detector
sources. The first attempt stops at a source-support guard before any probe
fit. Its receipt remains failed. The corrected definition excludes and counts
466 main and five tiny shared-source pairs, without dropping events or changing
native node/trial denominators. Main has88240 supported B pairs and1202387
known background-containing pairs. Unknown pair support is explicitly reported.

For each category, a fixed UID hash orders192 probe-fitting and64 assessment
identities. The384 B-bearing fit events supply pair labels;128 B-bearing
assessment events supply the pair endpoint. Both partitions were seen by the
frozen scientific model. This split tests transfer of a newly fitted probe
across training-role events, **not independent physics generalization**.

A257-parameter logistic probe consumes symmetric absolute differences and
products of the128 contextual features, with fit-only feature normalization.
It takes exactly512 Adam updates at learning rate0.01 and256 pair presentations
per update, sampling fitting events uniformly and balancing same/cross labels.
A matched null fit permutes B labels within each fitting event, retaining B
counts and background support. Seed, budget and final checkpoint are fixed;
there is no tuning or early stopping. This is two new supervised diagnostic
fits, not zero training. The scientific encoder/decoder receives zero updates.

Partition evidence
------------------

On the128 assessment B collisions, mean event pair AUC is:

.. list-table:: Fixed-checkpoint pair discrimination
   :header-rows: 1

   * - Score
     - Mean event AUC
   * - Native proposal conditional same-slot probability
     - 0.5452
   * - Native refinement conditional same-slot probability
     - 0.5394
   * - Contextual feature cosine
     - 0.5024
   * - Fitted symmetric pair probe
     - 0.5333
   * - Shuffled-label probe control
     - 0.5005

Probe minus native proposal event AUC is-0.01190, with a category-stratified
whole-collision bootstrap interval[-0.02365,-0.00097]. These2000 resamples
preserve correlated pairs within an event. This is descriptive training-role
uncertainty, not a population confidence claim or seed replication.
Pooled pair AUC, event distributions, category and source-size strata are
preserved separately; they use different weights and are not interchangeable.

The probe is weaker than native assignment in both categories: charged0.5169
versus0.5219, mixed0.5497 versus0.5685. The same direction holds in every
reported source-size stratum, including small targets. Size strata include an
event when either target lies in the range and therefore overlap. Background
controls remain essential:209390 of300378 assessment background-containing
pairs have probe score at least0.5. This is a diagnostic score frequency,
not an accepted fake-B event rate or a deployable pair threshold.

On all9437 main true-B nodes, the best conditional two-slot permutation gets
6075 correct, compared with5713 from assigning every B constituent to its
larger true group. This majority comparator is a truth-assisted size baseline,
not deployment. Refinement gets6053. These conditional-only matches differ
from Phase77's5676 proposal count under hard-error matching that also includes
unassigned decisions; changing the matching objective is not a bug or a gain.

Queries are similar but not identical. Proposal mean query cosine is0.8585 in
charged and0.8662 in mixed assessment collisions; background-only categories
are higher. The full download retains differences, norms, margins and
permutation ambiguity, including zero-support continuum ties. Similar queries
alone do not prove a symmetry-collapse defect or justify query repulsion.
Tiny fitted native pair AUC is1.0 at both stages, with310 of310 conditional
assignments correct. Tiny representation cosine need not be high for the
learned head to separate groups. Probe scores on tiny use a different encoder
checkpoint and are recorded as cross-checkpoint transfer only.

Decision and evidence limit
---------------------------

The concrete finding is **weak learned within-event partition discrimination**,
with only a modest advantage over a group-size baseline. This finite probe
finds no stronger readily extracted pair signal that assignment simply ignores.
Matching/permutation/source-support tests and exact replay demonstrate no new
implementation defect. They do not prove model convergence or sufficient
representation capacity.

Probe mean loss over its first and last50 updates is0.7096 and0.7025; the
shuffled control ends at0.7132. This fixed optimizer budget does not establish
convergence of the probe, much less the original scientific model. Nonlinear
information and alternative optimization remain unidentified. An optimizer or
loss sweep selected after these results would exceed the registered diagnostic.
No narrowly supported intervention is selected merely to launch jobs.

The actionable outcome is to stop scientific submissions for this pass, keep
the unsuccessful models as development controls, and retain the fixed corpus,
widths and guards. Do not deploy the probe, relax foreground thresholds, add
query repulsion or pair-loss coefficients, expand data/pretraining, or promote
these models on this evidence. Phase74–76 still show no reliable held-out gain;
Phase77–78 locate failures but do not establish a successful remedy.

Native endpoints and resources
------------------------------

Exact replay preserves main raw and accepted0/1024, tiny raw32/32 and accepted
27/32, including five charge-incompatible tiny targets. Main continuum acceptance
remains96/1024. Complete prior channel/type, category, relation, latent-source,
raw/accepted and validity metrics remain available with their original scope.
No physical trees, exact physical beam pools, legal deep-proposal survival or
p4 closure are defined by this flat optional-B head. These proxies are not FEI
efficiency. No new600-event development or12000-event primary evaluation occurs.

The corrected diagnostic completes in159 scheduler seconds, with154.77 model
wall seconds,150.92 process CPU seconds and1619072KiB peak process RSS. It uses
two CPUs,16GiB, a one-hour limit, no GPU and no requeue. There are1560 model
forward views,1536 unique training identities,1024 total probe updates and
262144 sampled pair presentations across two fits. Each fit visits all384
eligible B fitting events. Encoder and decoder counts are1322658 and1385991;
the event node-squared proxy is2692048, not FLOPs. Head-intermediate capture
recomputes the head and is included in measured time; it is not another encoder
fit. The failed preflight uses24 additional scheduler seconds and21.195 CPU
seconds; it is not relabelled a successful job.

The :doc:`dashboard <_generated/status/index>` provides the current summary.
Its :doc:`downloads <_generated/status/downloads>` preserve all per-event
metrics losslessly in three bounded parts, plus tiny metrics, all aggregates,
probe curves, native endpoints, source hashes and a verified decoder. Historical
receipts and downloads are unchanged. Publication requires exact-commit checks,
guarded promotion and actual live verification; private receipts record them.
