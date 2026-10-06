Phase69 supplementary category-policy evaluation
================================================

This separately versioned evaluation applies the new category policy to the
frozen Phase69 primary step4000 pair, refined encoder versus pre-refinement
encoder. Its measurements and complete downloads are in the
:doc:`dashboard <_generated/status/index>`. The original :doc:`Phase69 review
<phase69>`, 100-event reports, 20-event proposal-beam diagnostic, auxiliary
checkpoint tracks and their selection/gates remain historical evidence.

Measured results
----------------

Exact retained B tagging is 0/8,000 nominal B trials in each arm and scope;
event-any and event-both success are 0/4,000. Sixteen exact-tag truth trials
remain unavailable. Inclusive FSP grouping is 0/8,000 for refined and 1/8,000
for pre-refinement in both scopes, with event-any counts 0/4,000 and 1/4,000;
no event has both B groups correct. One membership trial is unavailable.
These are lower bounds with unknown truth, not measured physical efficiencies.

The only recovered inclusive retained channel is
``anti-B0 -> (D+ -> (K_S0 -> (pi-, pi+), e+), mu-)``:
four evaluated B trials, one correct pre-refinement group, zero refined groups,
and zero membership-unavailable trials in that channel, in each scope.
No channel has a proven exact retained-B tag. Full channel tables, including
zero successes and unknowns, are preserved in the aggregate download.

Full retained LCAG is 3,262/730,991 versus 3,266/730,991, refined versus
pre-refinement. The paired pre-minus-refined difference is 0.000547 percentage
points, with exploratory 95% interval [-0.010345, 0.011301] percentage points.
Half retained LCAG is 3,643/395,817 versus 3,627/395,817. These small differences
do not establish a joint recursive-quality benefit. Primary tree metrics,
retained components, PID, kinematics, source coverage and all 28 uncertainty
endpoints remain available separately; isolated-leaf source recovery must not
be read as nontrivial tree success.

The 60-event beam diagnostic has 0/40 exact and inclusive B successes in each
arm/scope, for both model-ranked top1 and the retained pool. All 480 returned
hypotheses across arms and scopes were evaluated. Greedy and beam show zero
committed source conflicts and zero failed daughter-sum p4 closures.

Coverage and provenance
-----------------------

Both arms process the same 12,000 distinct validation collisions in both full
and half/component scope: 2,000 each from charged, mixed, ccbar, uubar, ddbar
and ssbar. Scopes, candidates, B trials and repeated fits do not multiply the
collision count. The complete claim applies to this primary greedy evaluation.
A separate full-depth width-two beam evaluates 60 collisions per arm, ten per
category; its greedy reference, model-ranked top1 and retained-pool/oracle
results are diagnostic, not quota compliant. This search is different from the
historical proposal-set beam, so its outcomes are not a controlled comparison
with that 20-event report.

The source-role manifest and dataset index authenticate 100,000 validation
identities and 70,000 disjoint training identities. All 34 shard hashes and
source/category identity columns were checked. Excluding the 61,000 identities
reserved through Phase70 leaves 39,000; an identity-only deterministic ranking
selects 2,000 per required category, without consulting completeness,
representability, truth availability or model success. The supplementary
reservation leaves 27,000 unreserved events across all seven available
categories, including taupair. This total alone does not establish capacity for
a future six-category study. No sealed test data were accessed.

Native Phase69 completion receipts, frozen source, actual checkpoint hashes and
both checkpoint-pair validations authenticate the study choice. Phase70 was
still running when this cohort was reserved and did not supersede the latest
reviewed complete study. No training or checkpoint selection was performed.
The hybrid contrast retains historically pretrained encoders, shared refined
PID weights and train-fitted normalization; it is not pretrained versus
untrained, nor a pure hyperbolic-geometry comparison.

.. list-table:: Coverage in each arm (counts agree across arms)
   :header-rows: 1

   * - Category
     - Requested / unique attempted / processed
     - Failed
     - Primary truth unavailable (full / half)
   * - charged
     - 2,000 / 2,000 / 2,000
     - 0
     - 0 / 23
   * - mixed
     - 2,000 / 2,000 / 2,000
     - 0
     - 1 / 23
   * - ccbar
     - 2,000 / 2,000 / 2,000
     - 0
     - 2,000 / 8
   * - uubar
     - 2,000 / 2,000 / 2,000
     - 0
     - 2,000 / 562
   * - ddbar
     - 2,000 / 2,000 / 2,000
     - 0
     - 2,000 / 566
   * - ssbar
     - 2,000 / 2,000 / 2,000
     - 0
     - 2,000 / 460

Continuum has no declared single full truth root; its full primary metric is
unavailable. Explicit retained components are scored separately. These events
remain processed collisions in coverage and nominal tagging accounting.

Accounting and interpretation
-----------------------------

The aggregate download records requested, unique attempted, processed, failed
and truth-unavailable collisions by arm, category and scope. Processed events
include unsuccessful reconstruction and unavailable truth. A failed execution
means an attempted event without completed evaluator output; invalid rollout
is a separate diagnostic among processed events. Primary tree truth,
all-retained-tree truth, exact tagging truth and inclusive source membership
have separate availability counts.

The 4,000 charged/mixed B-pair collisions contribute 8,000 nominal B trials and
4,000 event-level trials per arm/view. The generic two-slot acceptance report
covers all 12,000 input collisions, including continuum, and has 24,000 nominal
slots. Missing B truth stays unavailable in nominal tagging denominators.
Primary direct-target incompatibilities remain failures. Continuum fake-B
acceptance and retained-component recovery are separate from B efficiency;
unavailable generator-parton ancestry is never assigned a zero success rate.

Exact retained tagging requires the correct FSP source set, retained tree and
reduced PID. Inclusive grouping requires exact FSP source-set equality for one
model-produced group, independently of internal topology and PID. It does not
combine disconnected predictions. Both reports retain per-B/per-event counts,
all evaluated B channels, unavailable truth and continuum source-type coverage.
These retained proxies are not physical FEI efficiencies.

Strict CPU inference uses the checkpoint's trained-confidence policy, maximum
level 6, object threshold 0.6, pointer threshold 0.35 and
``soft_decision_hard_construction`` with PID temperature 0.5. Detector projection,
normalization, masks, mother ontology and daughter-sum construction are
unchanged. Counts of shared resources among unused detector inputs remain a
separate diagnostic from source conflicts in committed reconstructed
components. The download retains both, along with p4 closure; daughter-sum
closure is not physical momentum resolution.

Uncertainty and reproducibility
-------------------------------

Tree and inference sufficient statistics are summed before division. Tagging
and channel summaries use the shared evaluator's aggregation functions.
The supplement reports 10,000 paired bootstrap resamples of collision events,
stratified by source category. Both B trials and both scopes stay with their
collision. Bounded resample batches avoid a replicate-by-event-by-metric
allocation; tested batch sizes yield identical results.

Intervals are exploratory and conditional on two frozen fitted models. They
do not include training-seed uncertainty, multiplicity correction or a new
source-domain test. A degenerate zero-success bootstrap interval does not
prove zero population efficiency. Equal-category pooled rates are not physical
mixture estimates. Original checkpoint selection and preregistered gates are
not replaced, and this supplement authorizes no model promotion or new study.

The dashboard provides a manifest, bounded lossless JSON data parts, integrity
hashes and a standalone standard-library decoder. Download the manifest and
all parts into one directory before decoding. Complete raw per-event reports,
UID manifests, scheduler receipts and source/checkpoint bindings remain in
project storage. Future cohort builders must authenticate and exclude the
tracked supplementary reservation binding as well as the historical ledger.
