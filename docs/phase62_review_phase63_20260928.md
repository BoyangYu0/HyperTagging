# Phase62 closeout and Phase63 corrected-supervision study

Phase62 completed successfully under its original frozen implementation
`2e7b2b26dd59eef087b079b5aa430f8e07532270`. Jobs 16731598 and 16731599
finished with exit status zero, each completing 2,188 pretraining updates and
4,376 reconstruction updates. The closeout verifies native receipt and contract
hashes, data/cohort bindings, all fourteen evaluation views, all returned beam
candidates in both scopes, and exact primary repeat equality. No model or
historical report was rewritten. Complete scalar and tree exports are on the
configured data volume under `HyperTagging_artifacts/phase62_review_20260928`.

## Scientific result

| Endpoint | Late PID 0.2 control | Late PID 0.1 candidate |
| --- | ---: | ---: |
| Selection source-set plus mother PID | 286/3,583 | 288/3,583 |
| Configured full roots | 4/100 | 4/100 |
| Original full LCAG | 2/2,789 | 0/2,789 |
| Retained full LCAG | 23/3,274 | 21/3,274 |
| Retained half/component LCAG | 25/1,922 | 24/1,922 |
| Exact nontrivial full components | 9/154 | 11/154 |
| Exact nontrivial half/components | 9/170 | 11/170 |
| Coherent retained forests | 1/100 | 2/100 |

The control passes the original gates. The candidate fails full LCAG and exact
mother coverage gates. Every exact nontrivial component is a two-leaf,
one-mother, depth-one tree. The coherent forests contain only depth-one mothers
and isolated leaves; these are not recovered deep decay trees. The candidate's
small exact-component gain accompanies lower LCAG and mother-PID coverage.
Paired event-bootstrap intervals include zero (the coherent-forest interval
has zero as its lower endpoint). They condition on these trained models and do
not measure variation across training seeds.

Only 107/154 nontrivial full components are representable under the recorded
policy; the 95.7% overall representability rate is dominated by isolated leaves.
The beam subset contains 28 nontrivial components. Each arm's component oracle
achieves only 3/559 full LCAG pairs and 1/28 exact components. Changing ranker or
increasing beam width alone therefore has weak support. Component oracle does
not imply mutually compatible event-level hypotheses.

Phase61 and Phase62 together do not establish a superior late-PID weight. Keep
70,000 training events and fix legal supervision before another dose or scale
study. Using the control encoder as the next baseline is a conservative choice
based on its original gates, not a claim of statistical superiority.

## Applied corrections

* Predicted-context alignment and inference decoder masks share parentless,
  lower-level, permitted-kind, committed-source eligibility. Consumed nodes
  remain available as encoder context. Teacher masks use only already formed
  generations, so future truth parents cannot hide legal teacher daughters.
  Representability diagnostics now inspect the matching rollout prefix.
* Recovery encouragement selects unmatched query slots; it cannot reward an
  already matched mother. This remains an object-only objective, not a way to
  backpropagate through an earlier irreversible merge.
* Confidence ignores slots masked for missing targets. Its primary-candidate
  target uses the configured pointer threshold, cardinality and source-conflict
  selection. The trainer serializes the rollout threshold into its policy.
  Calibration for alternative beam candidates remains unestablished.
* Retrieval excludes every view and branch of the same event, publishes explicit
  correct/query counts and per-view results, and adds cross-source support.
  Unsupported accuracy is unavailable. Parent diagnostics aggregate supported
  correct/eligible counts and are named coarse separation, not exact-parent
  classification. Corrected semantics are versioned.
* Validation adds an event-weighted, phase-specific objective using the actual
  PID schedule and active auxiliary terms. The historical fixed reference total
  keeps an explicit alias; it is not relabeled as the phase training objective.
* Radial diagnostics measure pre-cap norms, cap derivatives, saturation fraction
  and per-level radius variance. The saturation hypothesis requires these
  observations before changing geometry parameterization.
* ONNX bundle v2 declares and validates per-level query/cardinality capacities;
  runtime output checks use the selected level. The reader retains v1 support.
  Heterogeneous-capacity CPU ONNX execution is covered by a regression test.
  This does not establish basf2/offline beam search parity or real-mDST quality.
* Fixed-cohort UID filtering now runs after source-role validation and before
  expensive event tensor construction. It preserves cohort order and missing-UID
  checks. Zero predicted cardinality correctly yields an empty daughter set.
* Scientific resume rejects checkpoints from earlier supervision/metric
  contracts. Parameter-only initialization remains available for a preregistered
  new study; historical checkpoints remain intact.

The preprocessing schema, train-only normalization, source-role separation,
FP32 geometry, exact daughter-sum kinematics, ontology, source exclusivity and
sealed-test boundary are preserved. Ordered immediate-parent objectives,
radial reparameterization, charge-masked PID ablations, decoder-error negatives,
candidate calibration and unified host search remain separate scientific or
deployment studies. They are not silently combined into this comparison.

## Frozen development diagnostics

On 32 previously scored selection events (zero strict overlap), the Phase62
control encoder has a saturated radial derivative at every measured node in
all four views. Mean cap derivatives are 7.4–8.5e-8; mean batch median pre-cap
norms are 40.6–45.9 against a cap of 1.5. This confirms saturation on this sample,
not a full-population prevalence estimate. Supported coarse parent separation
is 0.9657; it is not immediate-parent classification. There are zero cross-event
retrieval peers in this small sample, so no retrieval accuracy is reported.

On 16 fixed development events, historical source alignment marks 30/48 truth
targets representable, while the legal-prefix check marks 28/48. Both excluded
targets occur at level 2 (historical 5/10, legal 3/10); no target above level 2 is
represented. This demonstrates real-data incidence on the bounded sample, not
a population prevalence or a measured benefit from retraining.

Prioritize a controlled radial projection reconditioning or parameterization
study after the legal-supervision baseline. Do not silently change historical
checkpoint geometry or simultaneously alter multiple objectives. The present
Phase63 contrast isolates assembly supervision while holding this encoder
fixed; it cannot establish that radial saturation has been repaired.

The Phase62 control checkpoint also passes the new export inspection with
queries 16/8/6/4/3/2 and cardinalities 9/16/15/15/14/2. Actual heterogeneous ONNX
execution was tested on a CPU fixture; no full scientific bundle or real basf2
pilot is claimed.

## Phase63 preregistration

Both arms start from the **same hash-bound final Phase62 control pretraining
checkpoint**, with no additional pretraining. Fresh reconstruction optimizer,
RNG and normalization follow the existing audited path. Both use every shared
correction above. The only arm differences are:

| Arm | Missing-target policy | Object recovery weight |
| --- | --- | ---: |
| corrected_recovery | Recovery on legal unmatched slots | 2 |
| masked_only | Mask unrepresentable targets | 0 |

Each arm has 4,376 updates, the same 70,000 training events, balanced replay,
per-level capacities, downstream adaptation schedule and seed 20260929. This is
one exploratory seed; no exact common-prefix or small causal-effect claim is
made. The fixed encoder avoids another confounded pretraining comparison.

Reserve 1,000 fresh selection events and 100 fresh strict events, with a
20-event beam subset. All 52,200 earlier reservations are excluded; 46,700
validation UIDs remain unreserved. The sealed test stays closed. Preserve
original gates, exact repeat evaluation and full/half retained-tree metrics.
Any scientific improvement must include nontrivial topology and depth, not
just recovery or configured roots. A stability failure remains evidence.

Submission uses the guarded immutable-source Slurm workflow: two one-H100 jobs,
8 CPU cores and 64 GiB each, a 36-hour ceiling, no restart/requeue, no automatic
successor and no promotion. Rendering and preflight are separate from actual
submission. A later receipt records actual scheduler job IDs and status.

Machine-readable evidence is in `artifacts/codex/reconstruction_phase62_*_20260928.json`;
large scalar/tree exports and job artifacts remain in the data path. The
existing [online operations proposal](online_study_operations.md) continues to
define monitoring, planning and bounded execution; no online service is deployed
by this training submission.
