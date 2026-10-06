Phase70 decoder relation-bias and efficiency review
===================================================

The category-complete evaluation does not establish an exact retained B-tag advantage. Both arms have **0/8,000 exact successes**; inclusive FSP grouping is **0/8,000 with bias enabled and 1/8,000 with bias disabled**, in both full and half scope. That single inclusive success is exploratory. These are retained-source proxies, not physical FEI efficiencies.

The same immutable cohort contains **2,000 distinct processed collisions in each of charged, mixed, ccbar, uubar, ddbar and ssbar**, 12,000 per arm. All requested collisions were processed; none was replaced or omitted for difficult or unavailable truth. Full and half scope share collisions. No sealed test was accessed and no scientific model is promoted.

The :doc:`dashboard <_generated/status/index>` gives the leading efficiency results and links every historical and current aggregate download. This review separates policy-sized primary measurements from native and beam diagnostics.

Primary tagging and coherent pairs
----------------------------------

.. list-table:: Strict greedy; nominal denominators include unavailable truth
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Bias enabled
     - Bias disabled
   * - full exact per_b_correct
     - 0/8000 (0%)
     - 0/8000 (0%)
   * - full exact event_any_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full exact event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full exact coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full inclusive per_b_correct
     - 0/8000 (0%)
     - 1/8000 (0.0125%)
   * - full inclusive event_any_correct
     - 0/4000 (0%)
     - 1/4000 (0.025%)
   * - full inclusive event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full inclusive coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half exact per_b_correct
     - 0/8000 (0%)
     - 0/8000 (0%)
   * - half exact event_any_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half exact event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half exact coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half inclusive per_b_correct
     - 0/8000 (0%)
     - 1/8000 (0.0125%)
   * - half inclusive event_any_correct
     - 0/4000 (0%)
     - 1/4000 (0.025%)
   * - half inclusive event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half inclusive coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)

.. list-table:: Full-scope B truth and channel accounting; half scope is separately exported
   :header-rows: 1
   :widths: auto

   * - Arm
     - Definition
     - Unknown nominal B trials
     - Channel rows
     - Recovered channels
     - Nominal channel trials
   * - type_bias
     - exact
     - 24
     - 7830
     - 0
     - 8000
   * - type_bias
     - inclusive
     - 0
     - 7830
     - 0
     - 8000
   * - no_type_bias
     - exact
     - 24
     - 7830
     - 0
     - 8000
   * - no_type_bias
     - inclusive
     - 0
     - 7830
     - 1
     - 8000

Each B-pair collision contributes two nominal B trials. Distinct truth B successes are deduplicated across every accepted candidate. Inclusive success requires an actual valid reconstructed composite with precisely the retained FSP membership; no disconnected partition is invented. Exact topology/PID availability and inclusive membership availability remain separate. Missing truth makes observed success rates proven-success lower bounds. Channel identifiers describe the retained signed representation; full generator decay descriptions and original quark ancestry are unavailable.

Continuum recovery and fake-B background
----------------------------------------

.. list-table:: Explicit retained components; fake slots use two nominal slots per collision
   :header-rows: 1
   :widths: auto

   * - Population and endpoint
     - Bias enabled
     - Bias disabled
   * - full ccbar exact
     - 36/6902 (0.52159%)
     - 61/6902 (0.8838%)
   * - full ccbar inclusive
     - 318/6902 (4.6074%)
     - 287/6902 (4.1582%)
   * - full ccbar fake-B event
     - 201/2000 (10.05%)
     - 155/2000 (7.75%)
   * - full ccbar fake-B slot
     - 224/4000 (5.6%)
     - 167/4000 (4.175%)
   * - full uubar exact
     - 71/5627 (1.2618%)
     - 121/5627 (2.1503%)
   * - full uubar inclusive
     - 536/5627 (9.5255%)
     - 498/5627 (8.8502%)
   * - full uubar fake-B event
     - 87/2000 (4.35%)
     - 145/2000 (7.25%)
   * - full uubar fake-B slot
     - 94/4000 (2.35%)
     - 151/4000 (3.775%)
   * - full ddbar exact
     - 73/5625 (1.2978%)
     - 127/5625 (2.2578%)
   * - full ddbar inclusive
     - 577/5625 (10.258%)
     - 511/5625 (9.0844%)
   * - full ddbar fake-B event
     - 88/2000 (4.4%)
     - 137/2000 (6.85%)
   * - full ddbar fake-B slot
     - 90/4000 (2.25%)
     - 139/4000 (3.475%)
   * - full ssbar exact
     - 104/5675 (1.8326%)
     - 161/5675 (2.837%)
   * - full ssbar inclusive
     - 763/5675 (13.445%)
     - 729/5675 (12.846%)
   * - full ssbar fake-B event
     - 123/2000 (6.15%)
     - 171/2000 (8.55%)
   * - full ssbar fake-B slot
     - 135/4000 (3.375%)
     - 180/4000 (4.5%)
   * - half ccbar exact
     - 12/6902 (0.17386%)
     - 43/6902 (0.62301%)
   * - half ccbar inclusive
     - 318/6902 (4.6074%)
     - 287/6902 (4.1582%)
   * - half ccbar fake-B event
     - 201/2000 (10.05%)
     - 156/2000 (7.8%)
   * - half ccbar fake-B slot
     - 224/4000 (5.6%)
     - 168/4000 (4.2%)
   * - half uubar exact
     - 52/5627 (0.92412%)
     - 112/5627 (1.9904%)
   * - half uubar inclusive
     - 536/5627 (9.5255%)
     - 498/5627 (8.8502%)
   * - half uubar fake-B event
     - 87/2000 (4.35%)
     - 145/2000 (7.25%)
   * - half uubar fake-B slot
     - 94/4000 (2.35%)
     - 151/4000 (3.775%)
   * - half ddbar exact
     - 52/5625 (0.92444%)
     - 118/5625 (2.0978%)
   * - half ddbar inclusive
     - 577/5625 (10.258%)
     - 511/5625 (9.0844%)
   * - half ddbar fake-B event
     - 88/2000 (4.4%)
     - 137/2000 (6.85%)
   * - half ddbar fake-B slot
     - 90/4000 (2.25%)
     - 139/4000 (3.475%)
   * - half ssbar exact
     - 48/5675 (0.84581%)
     - 135/5675 (2.3789%)
   * - half ssbar inclusive
     - 763/5675 (13.445%)
     - 729/5675 (12.846%)
   * - half ssbar fake-B event
     - 123/2000 (6.15%)
     - 171/2000 (8.55%)
   * - half ssbar fake-B slot
     - 135/4000 (3.375%)
     - 180/4000 (4.5%)

Component recovery is not q/qbar recovery: reduced data lack parton-to-FSP ancestry. Background acceptance and reconstruction correctness use different denominators. Unavailable component truth is retained in the complete aggregate tables. All accepted B candidates are inspected for tag recovery; slot-based fake-B accounting remains distinct from the total accepted-candidate count.

Retained topology and paired uncertainty
----------------------------------------

.. list-table:: All retained versus nontrivial populations
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Bias enabled
     - Bias disabled
   * - full lcag_pair_accuracy
     - 3559/729425 (0.48792%)
     - 3306/729425 (0.45323%)
   * - full source_precision
     - 379596/417845 (90.846%)
     - 381447/415082 (91.897%)
   * - full source_recall
     - 379596/480768 (78.956%)
     - 381447/480768 (79.341%)
   * - full coherent_retained_forest
     - 42/12000 (0.35%)
     - 35/12000 (0.29167%)
   * - full nontrivial perfectLCAG
     - 315/18445 (1.7078%)
     - 522/18445 (2.83%)
   * - full nontrivial source_precision
     - 34180/45731 (74.741%)
     - 33178/42976 (77.201%)
   * - full nontrivial source_recall
     - 34180/112988 (30.251%)
     - 33178/112988 (29.364%)
   * - half lcag_pair_accuracy
     - 3848/394634 (0.97508%)
     - 3608/394634 (0.91426%)
   * - half source_precision
     - 275036/311297 (88.352%)
     - 276586/308233 (89.733%)
   * - half source_recall
     - 275036/369210 (74.493%)
     - 276586/369210 (74.913%)
   * - half coherent_retained_forest
     - 42/12000 (0.35%)
     - 35/12000 (0.29167%)
   * - half nontrivial perfectLCAG
     - 175/22444 (0.77972%)
     - 448/22444 (1.9961%)
   * - half nontrivial source_precision
     - 37176/56165 (66.191%)
     - 36440/52632 (69.235%)
   * - half nontrivial source_recall
     - 37176/112987 (32.903%)
     - 36440/112987 (32.251%)

.. list-table:: Exact shapes are not interchangeable with deep B reconstruction
   :header-rows: 1
   :widths: auto

   * - Arm
     - Scope
     - Depth-one exact components
     - Depth-two-or-deeper exact components
     - Forests with mothers
     - Mother-free forests
   * - type_bias
     - full
     - 314
     - 1
     - 35
     - 7
   * - type_bias
     - half
     - 174
     - 1
     - 35
     - 7
   * - no_type_bias
     - full
     - 521
     - 1
     - 29
     - 6
   * - no_type_bias
     - half
     - 447
     - 1
     - 29
     - 6

.. list-table:: Disabled minus enabled; paired stratified collision bootstrap, 10,000 resamples
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Difference
     - 95% percentile interval
   * - full exact_tag per_b_correct
     - 0.0
     - [0.0, 0.0]
   * - full inclusive_group per_b_correct
     - 0.000125
     - [0.0, 0.000375]
   * - full retained lcag_pair_accuracy
     - -0.0003468485450868839
     - [-0.00048594995571651803, -0.0002071776292347299]
   * - full nontrivial_retained perfectLCAG
     - 0.01122255353754405
     - [0.008717298287890363, 0.013760225364320929]
   * - full nontrivial_retained source_precision
     - 0.024598059253656546
     - [0.018417559884646835, 0.03065056966320699]
   * - full nontrivial_retained source_recall
     - -0.00886819839274966
     - [-0.012484841300641824, -0.005248523190567115]
   * - full retained coherent_forest
     - -0.0005833333333333333
     - [-0.0016666666666666666, 0.0005]
   * - full pooled_continuum exact_component_recovery
     - 0.007805615006924336
     - [0.005978875435574788, 0.009674732577082516]
   * - full pooled_continuum inclusive_component_recovery
     - -0.007092198581560294
     - [-0.009919319429530837, -0.004253454290354101]
   * - full pooled_continuum top1_fake_b_event_acceptance
     - 0.013624999999999998
     - [0.0061249999999999916, 0.021374999999999998]

The downloads contain both-arm and difference intervals for every registered paired endpoint, including each continuum category, coherent pairs and fake-B slots. Both B trials stay within their collision cluster. Sparse category-specific event-any binomial bounds accompany zero/single successes. Degenerate zero-success bootstrap intervals do not prove zero population efficiency.

Coverage, independence and immutable lineage
--------------------------------------------

.. list-table:: Actual per-arm primary coverage; both arms and scopes authenticate identically
   :header-rows: 1
   :widths: auto

   * - Category
     - Old eligible
     - Old shortage
     - Expanded eligible
     - Processed
     - Failed
     - Tree unavailable
     - Exact-tag unavailable events
     - Membership unavailable events
   * - charged
     - 1921
     - 79
     - 11921
     - 2000
     - 0
     - 0
     - 10
     - 0
   * - mixed
     - 1901
     - 99
     - 11901
     - 2000
     - 0
     - 0
     - 14
     - 0
   * - ccbar
     - 5825
     - 0
     - 15825
     - 2000
     - 0
     - 2000
     - 10
     - 10
   * - uubar
     - 5778
     - 0
     - 15778
     - 2000
     - 0
     - 2000
     - 5
     - 5
   * - ddbar
     - 1913
     - 87
     - 11913
     - 2000
     - 0
     - 2000
     - 4
     - 4
   * - ssbar
     - 1833
     - 167
     - 11833
     - 2000
     - 0
     - 2000
     - 3
     - 3

The prior union contains 73,000 reservations, including the Phase69 policy supplement. The historical 39,000 figure is stale. Four categories were short before materialization. The established source-safe expansion used two fresh source files per required category, 60,000 new validation collisions, excluded the complete prior source catalogue, preserved every original role and all 70,000 training payloads, and fitted nothing on validation. The resulting index authenticates 230,000 distinct train/validation identities, unchanged train normalization and no source/category mismatches. Sealed-test payloads were never read.

The deterministic identity-only cohort reserves 12,000 events, leaving 75,000 unreserved validation identities before the next campaign. Selection, fitting, training and every historical adaptive reservation are excluded. The checkpoints remain the original selection-only step4000 artifacts. The expanded external-sample path relaxes index/split identity only; explicit cohort identities, schema, feature, PID, normalization, policy and source exclusivity checks remain.

Native training source: ``fc05d79ea4fef264cf8cd5718bde420504170590``. Supplemental evaluator source: ``167485834a25441631be4cd09be50b8c6b96e054``. Cohort SHA-256: ``aab762084fde8e2b8b22a3f3f4932224d0e8d9a5bae8ebb05580f7696a1aa35c``. Every checkpoint, input, cohort chunk and output report has an immutable digest in the exports. The exact policy handoff patch is integrated without replacing unrelated edits.

Native evidence and smaller diagnostic views
--------------------------------------------

Both native training jobs completed successfully, but scheduler completion is not scientific validation. Both arms execute 4,376 finite optimizer updates and 280,064 replay slots on 70,000 training events, seed20261006, mixed contexts and adaptation after2,188 steps, with zero additional pretraining. Shared refined encoder/PID initialization and train normalization authenticate. Module presence is independently verified on all five registered tracks:21 treatment parameter entries versus zero. Serialized model-state tensor elements are4,439,722 versus3,729,971; these are not equal-FLOP or bitwise-prefix experiments.

All fourteen native reports,26 saved checkpoints and exact native primary repeats authenticate. Native strict coverage is100 collisions: charged8, mixed13, ccbar16, uubar25, ddbar8, ssbar7 and taupair23. Its21 B-pair collisions give42 nominal B trials, with zero exact/inclusive successes in both arms and scopes. These are historical diagnostics, not policy-complete final coverage.

.. list-table:: Preserved original 100-event gates and endpoints
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Bias enabled
     - Bias disabled
   * - exact_mother_coverage
     - 7/194 (3.6082%)
     - 6/194 (3.0928%)
   * - full_lcag
     - 7/3446 (0.20313%)
     - 8/3446 (0.23215%)
   * - full_root_completion
     - 42/100 (42%)
     - 24/100 (24%)
   * - full_source_precision
     - 87/118 (73.729%)
     - 92/122 (75.41%)
   * - full_source_recall
     - 87/377 (23.077%)
     - 92/377 (24.403%)
   * - half_lcag
     - 19/2072 (0.91699%)
     - 21/2072 (1.0135%)
   * - half_perfect_lcag
     - 0/136 (0%)
     - 5/136 (3.6765%)
   * - half_root_pid_accuracy
     - 2/136 (1.4706%)
     - 11/136 (8.0882%)
   * - half_source_precision
     - 146/326 (44.785%)
     - 143/289 (49.481%)
   * - half_source_recall
     - 146/642 (22.741%)
     - 143/642 (22.274%)

Only the disabled arm passes every original gate. Enabled fails full source precision, half perfect LCAG, half root PID and half source precision. The native retained full LCAG counts are19/3,823 versus22/3,823; exact nontrivial components are2/135 versus7/135, all depth one. The native paired LCAG interval spans zero. Gate passing, leaf retention and shallow exactness do not establish tagging utility.

The original seven views per arm retain all checkpoint tracks, contracted topology, exact repeats and every proposal-beam candidate. Additive instrumentation reproduces2,600 inference fingerprints exactly; all scientific counts/categories agree. The3,676 archived continuous-value differences, maximum absolute1.4007091522216797e-6, remain documented rather than silently replacing native p4 or ranking values. The separate metadata correction changes only the inherited native classification/date, keeping original evidence and every numerical result.

The new width-two full-depth beam processes60 collisions per arm, ten per category, in both scopes. It is explicitly diagnostic. Its top1, retained pool at every K, oracle, score/rank summaries and actual candidate counts are exported separately. Incompatible candidates recovering different Bs do not count as a coherent pair; no beam result inherits12,000-event coverage.

.. list-table:: Supplemental beam diagnostic cardinality
   :header-rows: 1
   :widths: auto

   * - Arm
     - Scope
     - Distinct collisions
     - Returned candidates
   * - type_bias
     - full
     - 60
     - 119
   * - type_bias
     - half
     - 60
     - 119
   * - no_type_bias
     - full
     - 60
     - 120
   * - no_type_bias
     - half
     - 60
     - 120

Complete metric and implementation audit
----------------------------------------

All topology/LCAG/source/PID/p4/constraint aggregates, unavailable values, target representability, depth/nontrivial components, coherent forests, channel coverage, beam candidates and confidence/search diagnostics are included. The complete native/checkpoint exports contain11,711,302 native scalar records and63,089 checkpoint scalar records, including zero-dimensional tensor scalars. The native public bundle contains178,049 scalar values across nine registered views. Additive exports contain14 reports,2,480 event-scope records,240 candidate-scope records and132,208 public numeric records. The private archive also contains every supplemental report and its complete scalar census. Cardinality and SHA-256 manifests bind each family; overlapping export families are not advertised as unique independent measurements.

All48 fixed-input geometry views are unsaturated. They are not a census of generated-state geometry. Exposure audits retain105,347 predicted and174,717 teacher slots in both arms; actual forward counts396,937 versus388,985 differ. Complete checkpoint transfer and native objective histories remain downloadable. Missing strict evaluation of unregistered checkpoint tracks is unavailable, never zero. No source conflict, failed daughter-sum closure or truth-assisted inference is tolerated by the implementation audit. Daughter-sum closure is a construction invariant, not physical momentum resolution against generator truth.

Public exports omit collision identities, raw reports, private paths, scheduler identifiers and checkpoint filenames. Lossless typed-DAG parts preserve scalar types, unavailable values and all channel tables; the standalone decoder verifies the decoded digest. Download the manifest, all parts and decoder into one directory. The decoded object retains arms, native_additive_tagging, native_additional_diagnostics and extended_paired_uncertainty. A separate native scalar decoder restores every compact metric name to JSONL and verifies the178,049-record binding. The private bundle keeps raw records, native receipts and full integrity manifests. Historical reports and downloads retain their original bytes.

What the complete study history supports
---------------------------------------

The objective is retained reconstruction efficiency. Loss, leaf retention,
configured root production, syntax validity and daughter-sum closure cannot
substitute for an exact correct tag or a coherent correct pair. Inclusive
source membership is useful but does not establish intermediate topology or
PID correctness. None of these retained proxies is a physical FEI efficiency;
original quark recovery is unavailable without ancestry.

.. list-table:: Study families and negative evidence
   :header-rows: 1

   * - Family
     - Evidence and consequence
   * - Historical implementations and migration phases 1–13
     - Preserve software equivalence and distinct legacy denominators. They are not modern trained-physics comparisons. No standalone reconstruction closeouts numbered 14–33 were established by the historical inventory.
   * - Early production, pretraining and resource/transfer pilots
     - Data, runtime and transfer contracts were established with recorded failures and finite checkpoints. Throughput and objective stability are not recursive quality.
   * - Stage A and phases 34–39
     - Local relation/pointer and replay gains did not establish deep reconstruction or satisfy a promotion-grade joint endpoint.
   * - Phase40 and its recovery
     - Size, compute and evaluation population changed together. This is not a controlled training-data learning curve.
   * - Phases41–47
     - Pointer, pretraining duration and encoder variations produced mixed sparse results; scientific repairs limit comparability. Longer historical pretraining is not all pretraining versus none.
   * - Phases48–51
     - Phase48 did not execute its PID treatment. Later repaired treatments execute, but do not establish joint topology improvement.
   * - Phases52–54
     - Recovery-weight changes alter local counts while exact components and forests stagnate or regress. More dose tuning is not supported.
   * - Phases55–58
     - Evaluator repairs, missing controls and Phase57 selection contamination limit causal conclusions. Preserve failures and unavailable endpoints.
   * - Phases59–62
     - Phase59 is a 25-event feasibility pilot; Phase60 fails before treatment. Completed later schedules retain mixed shallow results and pre-treatment variation. Equal seeds are not identical realized prefixes.
   * - Phase63
     - Corrected forest-root target eligibility repairs supervision. This is necessary correctness work, not demonstrated end-to-end efficiency.
   * - Phase64
     - Radial reconditioning repairs measured saturation. A geometry repair does not demonstrate useful recursive assembly.
   * - Phase65
     - Freeze versus late adaptation has mixed sparse results and no primary exact nontrivial component. Geometry remains usable without establishing tags.
   * - Phases66–67
     - Teacher-only preserves supervision that generated contexts lose. The precision/recall tradeoff does not establish a joint recursive winner.
   * - Phases68–69
     - Hybrid refined versus pre-refinement encoder contrasts share refined PID and normalization. The confirmation does not establish a joint advantage or identify the total benefit of pretraining.
   * - Phase69 category supplement
     - Exact retained tagging is zero of 8,000 nominal B trials in each arm; inclusive grouping is zero versus one. This supersedes neither its historical 16-collision diagnostic nor the limitations of a single fitted pair.
   * - Phase70
     - The module intervention executes, but original small-cohort gate passing and shallow component recovery alone cannot establish tagging improvement. The separately reserved category-sized results above own the final efficiency claims.

The historical review and all dated studies remain in :doc:`studies` and the
complete download catalogue. Comparisons across different seeds, cohorts,
source mixtures, evaluator revisions or target policies are descriptive.
Phase69 and Phase70 cannot be treated as a paired cross-study comparison.
Neither full/half scopes, checkpoint tracks, repeated inference nor nested
beam pools multiply the collision sample size.

Training growth and pretraining recommendation
----------------------------------------------

Do not expand training automatically now. Keep 70,000 training events for the
one next bounded controlled study. This is a decision under uncertainty, not
evidence that 70,000 is optimal or that scaling cannot help. A later controlled
learning curve must distinguish more unique examples from additional updates,
exposures and compute. Historical Phase40 cannot answer that question.

The additional validation materialization improves independent evaluation
precision and source coverage; it does not grow the training set. The former
Phase69 16-collision/32-B-trial diagnostic could not resolve rare tagging.
Even the policy minimum can leave sparse channels and zero-success uncertainty.

Task-aligned representation and objective improvements are plausible priorities;
merely running more pretraining is not supported as a remedy. Geometry repair,
repeated 70,000-event holds and hybrid encoder ablations do not prove improved
pretraining is more beneficial than data growth. The total benefit of
pretraining and a controlled scaling advantage both remain unidentified.

Interpretation of uncertainty
-----------------------------

Paired uncertainty resamples whole collisions within source category, keeping
both B trials and both fitted models paired. It is conditional on the two
fitted models and available source domains, does not estimate training-seed
variation and is not corrected for exploratory endpoint multiplicity.
Category-specific collision-event binomial bounds accompany sparse successes;
they assume independent collisions within category and are not source-domain
confidence guarantees. Zero bootstrap variability is not zero population
uncertainty. Unavailable truth remains in nominal denominators, so proven
successes are lower bounds on truth-complete reconstruction.

Equal-category pooled rates describe the specified evaluation cohort, not a
physical mixture. Beam top1 and pool/oracle rates retain their smaller actual
counts and may not inherit primary coverage. A pool recovering each B in
incompatible candidates is not a coherent-pair success. No scientific model
promotion or sealed-test access follows from this review.


One next bounded objective study: Phase71
----------------------------------------

End the decoder-bias family. Disabled bias improves exact nontrivial component
recovery but lowers retained LCAG and source recall, reduces inclusive continuum
recovery, and increases pooled continuum fake-B acceptance. One inclusive B
success does not resolve that tradeoff or justify a winner.

Phase71 tests auxiliary complete-target teacher objective weight **0.5 versus1.0**.
Both arms keep the enabled-bias reference architecture, shared refined encoder
and frozen PID, train normalization, mixed-context probability0.5, late encoder
adaptation after2,188 updates,70,000 training events and seed20261007. Each arm
has4,376 updates and280,064 replay slots; there is no additional pretraining.
Increasing auxiliary supervision may help legal target learning while retaining
generated contexts, but may also worsen exposure mismatch or background. Phase37
already tested0.25 versus0.50 with weak hierarchy evidence; Phase66/67 preserved
more targets in teacher-only mode without establishing joint recursive benefit.
This is one controlled objective test under the corrected contracts, not an
assumed remedy or an automatic weight sweep.

Primary endpoints are exact and inclusive per-B correct-tag efficiency in both
scopes, with event-any, both-B and coherent pairs, channel coverage, continuum
component recovery and fake-B background. Four additional proven B successes
per8,000 nominal trials (an absolute0.05 percentage-point gain) is a preregistered
useful-effect target, not a power guarantee. Require positive paired lower
confidence bounds on full exact/inclusive gains and no point deterioration in
pooled fake-B background; report all category/half intervals and sparse bounds.
No proxy-only win, model promotion, sealed test or automatic successor follows.

The reserved next cohort has2,000 fresh events per required category, plus a
separate1,000-event checkpoint-selection cohort. All85,000 prior reservations
are excluded, leaving62,000 unreserved validation identities. Reserved is not
processed: the next study's primary evaluation has not run. Its native100-event
suite and20-event beam remain diagnostic, with explicit final-coverage rejection
until every arm/scope has processed the complete12,000-event policy cohort.

Expanded validation exposes four targets beyond old cardinalities: level-one
targets with10 and12 daughters and two level-two targets with17. The minimal
shared admission repair raises level one from9 to12 and level two/global from16
to17. No target is dropped; query limits and other level limits are unchanged,
and authenticated capacity overflow becomes zero. This shared repair, fresh
cohort and seed prohibit a causal numerical Phase70-to-Phase71 comparison.

The bounded scheduling contract permits exactly two one-H100-NVL tasks, each
with8 CPU cores,64GiB and a36-hour ceiling, no requeue. Future primary evaluation
is bounded to50 CPU tasks (48 primary chunks and two diagnostic beams), at most
16 concurrent, one CPU/16GiB/four hours each. There is no subsequent campaign.

Both tasks were accepted and released from frozen source
``1545d8ee80f26bc291498ae8718cb6e4a984f0bb``. The 2026-10-06 18:46 CEST
snapshot verifies both pending for scheduler priority, with contract hashes,
resources and no-requeue authenticated. GPU startup is not yet verified.
Scheduler acceptance is not scientific success.
