Phase56 reconstruction and pretraining review
=============================================

**Phase56 is an incomplete comparison.** The weight-1 control failed during
pretraining. The completed weight-2 arm fails full-root completion and full-source
recall, and has no exact retained component deeper than one generation in any
view or beam candidate. No parent-weight benefit can be inferred and no model
is promoted.

Review date: 22 September 2026. The :doc:`dashboard <_generated/status/index>`
provides every original and retained full/half aggregate, all model-only beam
rankings and oracle diagnostics, micro/macro and exact-combination counts,
and pretraining validation/execution downloads. Control reconstruction metrics
are **UNAVAILABLE_NOT_TRAINED**, not zero. Paired differences and paired
confidence intervals are unavailable.

Native execution and failure
----------------------------

Original control allocation completed 1,642 pretraining optimizer steps and 52,544
presentations. Before optimizer step 1,643, objective preflight rejected weighted
leaf-PID/LCA gradient dominance **20.1546 > 20**. This is a scientific objective
balance guard, not an evaluation handoff or infrastructure failure. There are
no control reconstruction checkpoints or reports. Partial pretraining checkpoints,
logs and the original failed receipt are preserved; no resume or bypass was used.

Original candidate allocation completed 2,188 refinement steps (70,000 presentations)
and 4,376 reconstruction steps, then all seven native evaluation views. Its
corresponding step-1,643 dominance diagnostic was 19.5571, close to the limit.
Both arms initialized the same original step-81096 parameters, with fresh
train-only normalization, optimizer, schedule, RNG and memory. Shared-step
checkpoint configuration and objective execution checks confirm parent weights
1 versus 2. Final validation at step 2,188 exists only for the candidate; the
control's earlier validation is not a matched final endpoint.

The candidate transfers its own final refined encoder and keeps PID frozen
throughout reconstruction. Checkpoint lineage, source checksums, finite weights
and PID-gradient state were checked. Primary strict repeats are identical.
The primary and independent complete-target selections use step 4,000; depth
and validity selections use step 1,000. Selection uses 1,000 events, strict
scoring 100 disjoint events, and beam the first 20 strict events. Native runtime
provenance is retained. Results remain conditional on this seed, cohort and
execution environment.

Original-policy reconstruction
------------------------------

Primary source-set plus mother-PID recovery is 7.868094%; this is not recursive
topology efficiency. A constructed B root is not necessarily a correct tree.

.. list-table::
   :header-rows: 1

   * - Original metric
     - Completed weight-2 arm
   * - exact_mother_coverage
     - 5.0/211.0
   * - full_lcag
     - 5.0/4507.0
   * - full_root_completion
     - 0/100
   * - full_source_precision
     - 62.0/69.0
   * - full_source_recall
     - 62.0/427.0
   * - half_lcag
     - 27.0/2758.0
   * - half_perfect_lcag
     - 9.0/161.0
   * - half_root_pid_accuracy
     - 25.0/161.0
   * - half_source_precision
     - 170.0/317.0
   * - half_source_recall
     - 170.0/760.0

Retained full, half and beam reconstruction
-------------------------------------------

All 57 returned coherent beam candidates were checked, in both scopes.
Incompatible reference trees remain failed trials. Isolated leaves do not
earn trivial exact nontrivial LCAG successes. Half scope uses B partitions
where defined and explicit component fallback otherwise.

.. list-table::
   :header-rows: 1

   * - Retained metric
     - Full
     - Half
   * - source_precision
     - 3051.0/3364.0
     - 2473.0/2747.0
   * - source_recall
     - 3051.0/3833.0
     - 2473.0/3222.0
   * - lcag_pair_accuracy
     - 26.0/5018.0
     - 27.0/2821.0
   * - perfect_lcag
     - 11.0/156.0
     - 11.0/177.0
   * - coherent_retained_forest
     - 1.0/100.0
     - 1.0/100.0
   * - mother_pid_coverage
     - 26.0/581.0
     - 27.0/560.0
   * - mother_pid_accuracy
     - 24.0/26.0
     - 25.0/27.0
   * - root_pid_accuracy
     - 389.0/1705.0
     - 334.0/1409.0
   * - target_representable
     - 3062.0/3182.0
     - 2461.0/2592.0

One distinct primary continuum event has a coherent retained forest with 27
components: 26 isolated leaves and one depth-one mother. Its appearance in
both scopes is not two independent successes, and forest topology coherence
does not require all PID labels to be correct. No deeper exact tree was found
across the complete seven-view and beam-candidate audit.

The dashboard separates greedy, four model-only beam rankings and oracle@K.
Oracle scoring uses truth only after search and is not a deployable ranking.
Preserved detector inputs dominate retained source coverage; high retained
source recall is not evidence of correct mother assembly. Daughter-sum p4
closure is an implementation invariant, not physical momentum resolution.

Event-cluster bootstrap intervals for the completed model use 10,000 resamples
and ratios of summed counts. They do not measure training-seed uncertainty or
recover a missing control contrast. Full details and all native event/tree
scalars are in the local complete-metrics archive.

Dataset size versus pretraining
-------------------------------

**Hold 70,000 training events for the next bounded pair.** Phase40 changed size,
steps and cohort together; no controlled learning curve establishes an immediate
benefit from training-data growth. Phases34/42 did not establish that longer
pretraining alone improves reconstruction. The available evidence supports
investigating objective balance and assembly errors next, but does not establish
that improving pretraining is more beneficial than adding data.

Phase41's reduced pointer weight worsened reconstruction. Phase43 adaptation
gains were not independently replicated by Phase44's seed-deviating diagnostic.
Phase45's larger encoder update tied and corrected Phases46/47 were mixed.
Phase48's PID update did not execute because of the autocast cache defect;
corrected Phases49–51 did not establish a stable benefit. Recovery weights in
Phases52–54 failed to deliver reproducible joint topology/forest improvement.
Recovery-dose tuning has stopped. Phase55 parent-weight effects were small,
shallow and mixed, with runtime sensitivity. Phase56 control attrition prevents
replication, and the completed arm still fails gates. **Stop parent-dose tuning.**
Cross-phase rates have different seeds/cohorts and are not causal contrasts.

Phase57: one bounded objective-balance test
-------------------------------------------

Fix parent weight **2 in both arms**, as a completed reference rather than a
proven winner. Compare leaf-PID phase weights **[1, 1, 0.4, 0.4]** versus
**[1, 1, 0.2, 0.2]**. Keep all other objectives, the dominance threshold 20
with fail action, 2,188 refinement steps, 4,376 reconstruction steps, the original
parameter initialization and 70,000 training events matched. Use seed 20260924,
recovery weight 2, frozen transferred PID, and encoder adaptation after step
2,188 at multiplier 0.05. A new failure-evidence file preserves objective-gradient
diagnostics before the same preflight exception; it does not relax the guard.

Only 391 untouched validation events remain after Phase56. Phase57 reuses the
**exact Phase56 sequence of 1,000 selection events**, including rollout selection,
and reserves **100 fresh strict events**, with a 20-event beam subset. Historical
strict events and the independent audit remain excluded. Intentional selection
reuse is separately recorded; all required forbidden overlaps are zero. This
leaves **291 untouched events**. This is not independent cross-phase selection
replication. Additional validation capacity needs planning separately from
increasing the training set.

Require joint recovery, nontrivial exact topology, source precision/recall and
coherent forest improvement, together with original gates. Lower pretraining
loss alone is insufficient. No automatic campaign chain, promotion or sealed-test
access is authorized. Each guarded job requests one H100 NVL, eight CPUs,
64 GiB and 36 hours without requeue.

Metric coverage
---------------

The available exports contain 8,416 original aggregate scalars, 10,197 retained aggregate scalars, 170,884 detailed retained scalars and 4,841,925 tree/candidate scalars. Reconstruction history contains 374,594 scalars and pretraining contains 816,189 scalars, including the partial control. The complete archive retains native reports, receipts, exports and verification hashes. Private event IDs and filesystem paths remain local; Pages publishes allowlisted aggregates.

Submission and publication
--------------------------

Both Phase57 jobs were submitted through held, verified contracts and released
as one bounded pair. The first scheduler snapshot was pending; submission is
not evidence of completed training. The immutable source revision is
``55abc338060088fb9f58be15f38cace54654a652`` and its tag is
``reconstruction-phase57-pretraining-objective-balance-20260922``.
Input checksums, resource requests and both scheduler receipts were verified.
The dashboard is a recorded snapshot, not a live scheduler feed.

The broad CPU suite passed **1,805 tests**, with 34 skipped. The focused
reconstruction/search suite passed 193 tests; dashboard/failure checks passed
107, and the frozen contract/failure/provenance preflight passed 14. Both
pretraining and reconstruction completed their two-step CPU smokes. These
engineering checks do not establish scientific performance of the next models.
