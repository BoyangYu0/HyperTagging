Phase59: Pretraining stability and recovered reconstruction review
================================================================================

Both arms completed all 2,188 pretraining and 4,376 reconstruction steps. The original jobs failed before strict scoring because the runner requested 100 events from the correct 25-event manifest. CPU evaluation was recovered from untouched checkpoints after binding the requested count to the authenticated cohort. No training was repeated.

Neither arm builds a configured full root (0/25). Retained LCAG ties at 4/813 full and 4/464 half. Exact nontrivial components are 2/34 versus 3/34 full and 2/39 versus 3/39 half. The primary coherent forests (1/25 versus 2/25) are tau-pair events with one depth-one, two-source mother and 27 or 29 isolated leaves. This is shallow local reconstruction, not recovery of a deep decay hierarchy. The shared exact-topology ssbar component has one incorrect daughter PID; requiring exact source, topology and all PID leaves only one control and two candidate nontrivial components.

**Classification: feasibility only, not confirmatory quality evidence.** The 25 strict events were untouched by either model; selection used the same reused 1,000 events, with every other validation event excluded. Beam scoring uses the first 20 strict events. Original failed receipts are preserved.

Hold the same 70,000 training events. The immediate data need is fresh independent validation. Phase60 adds 50,000 validation events from source files disjoint from the entire prior 1M-event inventory, preserves all historical roles and sealed test, and repeats the same 0.2-versus-0.1 contrast once with fresh selection, strict events and seed. Better pretraining quality remains a hypothesis; completion under the fail guard establishes feasibility, not superior reconstruction.

Phase59 complete metrics and provenance
--------------------------------------------------

All seven views per arm are included: primary direct, identical strict repeat, three independent checkpoint tracks, contracted diagnostic, and beam direct. Every returned candidate is checked in both full and half scopes, including targets outside the training policy. Contracted and oracle results remain diagnostic. Different checkpoint-selection tracks and the identical repeat are not independent training replications.

:download:`Original aggregate and checkpoint metrics <_generated/status/phase59-metrics.json>`; :download:`retained full/half and beam metrics <_generated/status/phase59-retained-metrics.json>`; :download:`micro/macro and exact-count metrics <_generated/status/phase59-aggregation-metrics.json>`; :download:`pretraining execution and validation metrics <_generated/status/phase59-pretraining-metrics.json>`.

The private complete-metrics bundle additionally preserves native reports, every exported scalar, candidate/tree rows, all training logs, checkpoint hashes, source contracts, uncertainty calculations and verification receipts. No model weights or event identifiers are published on this page. The :doc:`dashboard <_generated/status/index>` retains all older studies and metric downloads.

.. list-table::
   :header-rows: 1

   * - Export
     - Scalar rows
   * - Original aggregate
     - 15169
   * - Original detailed
     - 94359
   * - Retained aggregate
     - 18002
   * - Retained detailed
     - 265828
   * - Every tree/candidate
     - 3104423
   * - Reconstruction history
     - 749290
   * - Pretraining history
     - 1318894
   * - Micro/macro/exact supplement
     - 5040

Phase59 executed pretraining contrast
---------------------------------------------

Both arms use parent-ranking weight 2, the original step 81096 parameters, fresh train-only normalization/optimizer/schedule/RNG/memory, and 70,000 training presentations. PID phase weights are [1,1,0.2,0.2] and [1,1,0.1,0.1]. All logged weights were checked; step 1094 tensors match exactly before the contrast. Both complete under the unchanged dominance 20 fail guard. Each reconstruction uses its own fixed step 2188 refined checkpoint, with PID frozen and encoder adaptation after reconstruction step 2188.

.. list-table::
   :header-rows: 1

   * - Recorded final pretraining diagnostic
     - Late PID 0.2 control
     - Late PID 0.1 candidate
   * - validation_parent_ranking_accuracy
     - 0.726124
     - 0.726695
   * - validation_loss_parent
     - 0.00491123
     - 0.00493816
   * - validation_relation_accuracy
     - 0.843475
     - 0.843281
   * - validation_leaf_pid_accuracy
     - 0.887976
     - 0.88482
   * - validation_effective_rank
     - 24.2612
     - 24.2324

These retain the trainer’s batch/view aggregation semantics and are not new global micro rates. Weighted objective totals differ by design and cannot rank representation quality.

Training curves (PNG and PDF) and their source logs are preserved in the complete local metrics bundle.

Phase59 original-policy reconstruction
---------------------------------------------

.. list-table::
   :header-rows: 1

   * - Selection recovery
     - Late PID 0.2 control
     - Late PID 0.1 candidate
   * - Matched source set plus mother PID
     - 269/3457 (7.781%)
     - 267/3457 (7.723%)

Selection recovery is not recursive topology efficiency. Strict numerical thresholds are unchanged from 100-event studies but descriptive on this 25-event pilot; passing them would not certify confirmatory quality.

.. list-table::
   :header-rows: 1

   * - Original strict metric
     - Late PID 0.2 control
     - Late PID 0.1 candidate
   * - exact_mother_coverage
     - 0/51 (0.000%)
     - 0/51 (0.000%)
   * - full_lcag
     - 0/705 (0.000%)
     - 0/705 (0.000%)
   * - full_root_completion
     - 0/25 (0.000%)
     - 0/25 (0.000%)
   * - full_source_precision
     - 15/17 (88.235%)
     - 13/16 (81.250%)
   * - full_source_recall
     - 15/85 (17.647%)
     - 13/85 (15.294%)
   * - half_lcag
     - 2/424 (0.472%)
     - 2/424 (0.472%)
   * - half_perfect_lcag
     - 1/33 (3.030%)
     - 1/33 (3.030%)
   * - half_root_pid_accuracy
     - 1/33 (3.030%)
     - 4/33 (12.121%)
   * - half_source_precision
     - 36/70 (51.429%)
     - 35/66 (53.030%)
   * - half_source_recall
     - 36/148 (24.324%)
     - 35/148 (23.649%)

.. list-table::
   :header-rows: 1

   * - Strict numerical check
     - Late PID 0.2 control
     - Late PID 0.1 candidate
   * - minimum_complete_target_efficiency
     - True
     - True
   * - minimum_depth_fraction
     - True
     - True
   * - minimum_full_source_precision
     - True
     - True
   * - minimum_full_source_recall
     - True
     - False
   * - minimum_half_lcag
     - False
     - False
   * - minimum_half_perfect_lcag
     - False
     - False
   * - minimum_half_root_pid_accuracy
     - True
     - True
   * - minimum_half_source_precision
     - True
     - True
   * - minimum_half_source_recall
     - True
     - True
   * - nonzero_exact_mother_coverage
     - False
     - False
   * - nonzero_full_lcag
     - False
     - False
   * - nonzero_full_root_completion
     - False
     - False
   * - primary_repeat_identical
     - True
     - True
   * - structural_guardrails
     - True
     - True

Phase59 retained full and half trees
---------------------------------------------

Counts aggregate numerators and denominators before division. Full scope uses explicit retained components; half uses defined B partitions and explicit component fallbacks otherwise. Incompatible direct targets remain failed primary trials. Isolated one-source leaves do not earn trivial perfect-LCAG success. High source retention can reflect untouched input leaves.

.. list-table::
   :header-rows: 1

   * - Scope
     - Retained metric
     - Late PID 0.2 control
     - Late PID 0.1 candidate
   * - full
     - source_recall
     - 685/855 (80.117%)
     - 699/855 (81.754%)
   * - full
     - source_precision
     - 685/765 (89.542%)
     - 699/764 (91.492%)
   * - full
     - lcag_pair_accuracy
     - 4/813 (0.492%)
     - 4/813 (0.492%)
   * - full
     - mother_pid_coverage
     - 4/133 (3.008%)
     - 4/133 (3.008%)
   * - full
     - mother_pid_accuracy
     - 3/4 (75.000%)
     - 3/4 (75.000%)
   * - full
     - root_pid_accuracy
     - 70/328 (21.341%)
     - 69/328 (21.037%)
   * - full
     - perfect_lcag
     - 2/34 (5.882%)
     - 3/34 (8.824%)
   * - full
     - coherent_retained_forest
     - 1/25 (4.000%)
     - 2/25 (8.000%)
   * - full
     - target_representable
     - 691/720 (95.972%)
     - 691/720 (95.972%)
   * - half
     - source_recall
     - 582/745 (78.121%)
     - 595/745 (79.866%)
   * - half
     - source_precision
     - 582/654 (88.991%)
     - 595/651 (91.398%)
   * - half
     - lcag_pair_accuracy
     - 4/464 (0.862%)
     - 4/464 (0.862%)
   * - half
     - mother_pid_coverage
     - 4/128 (3.125%)
     - 4/128 (3.125%)
   * - half
     - mother_pid_accuracy
     - 3/4 (75.000%)
     - 3/4 (75.000%)
   * - half
     - root_pid_accuracy
     - 59/287 (20.557%)
     - 58/287 (20.209%)
   * - half
     - perfect_lcag
     - 2/39 (5.128%)
     - 3/39 (7.692%)
   * - half
     - coherent_retained_forest
     - 1/25 (4.000%)
     - 2/25 (8.000%)
   * - half
     - target_representable
     - 582/615 (94.634%)
     - 582/615 (94.634%)

.. list-table::
   :header-rows: 1

   * - Reference population
     - Full
     - Half/component
   * - events_without_flagged_target_incompatibility
     - 6
     - 6
   * - isolated_leaf_units
     - 668
     - 558
   * - nontrivial_topology_units
     - 34
     - 39
   * - representable_nontrivial_units
     - 23
     - 24
   * - single_source_composite_units
     - 18
     - 18
   * - source_empty_units
     - 0
     - 0

All-view depth audit found 0 exact-component records deeper than one generation (repeated views/scopes are not independent trials). Across all views and returned candidates, 2 distinct events have a coherent retained forest; 2 include nontrivial truth components.

Exact source, source+leaf PID, source+topology, and source+topology+all PID combinations are available with micro, unit macro and event macro denominators in the supplement and dashboard. Undefined ratios remain unavailable. A coherent topology forest does not alone require all PID correctness.

Phase59 full retained beam rankings
---------------------------------------------

All model-only rankings use generated candidates only. Oracle is computed after generation using truth; it is an upper-bound diagnostic, not a deployable ranking. Per-unit maxima need not be achievable by one coherent event hypothesis.

.. list-table::
   :header-rows: 1

   * - Arm
     - Ranking
     - LCAG
     - Exact component
     - Coherent forest
   * - Late PID 0.2 control
     - average_link_probability
     - 2/721 (0.277%)
     - 2/29 (6.897%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - learned_confidence_mean
     - 3/721 (0.416%)
     - 2/29 (6.897%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - learned_confidence_sum
     - 3/721 (0.416%)
     - 1/29 (3.448%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - normalized_joint_log_probability
     - 2/721 (0.277%)
     - 1/29 (3.448%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - oracle_diagnostic
     - 4/721 (0.555%)
     - 3/29 (10.345%)
     - 0/20 (0.000%)
   * - Late PID 0.1 candidate
     - average_link_probability
     - 4/721 (0.555%)
     - 3/29 (10.345%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - learned_confidence_mean
     - 4/721 (0.555%)
     - 3/29 (10.345%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - learned_confidence_sum
     - 3/721 (0.416%)
     - 2/29 (6.897%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - normalized_joint_log_probability
     - 2/721 (0.277%)
     - 1/29 (3.448%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - oracle_diagnostic
     - 4/721 (0.555%)
     - 4/29 (13.793%)
     - 1/20 (5.000%)

Phase59 half retained beam rankings
---------------------------------------------

All model-only rankings use generated candidates only. Oracle is computed after generation using truth; it is an upper-bound diagnostic, not a deployable ranking. Per-unit maxima need not be achievable by one coherent event hypothesis.

.. list-table::
   :header-rows: 1

   * - Arm
     - Ranking
     - LCAG
     - Exact component
     - Coherent forest
   * - Late PID 0.2 control
     - average_link_probability
     - 2/414 (0.483%)
     - 2/33 (6.061%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - learned_confidence_mean
     - 3/414 (0.725%)
     - 2/33 (6.061%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - learned_confidence_sum
     - 3/414 (0.725%)
     - 1/33 (3.030%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - normalized_joint_log_probability
     - 2/414 (0.483%)
     - 1/33 (3.030%)
     - 0/20 (0.000%)
   * - Late PID 0.2 control
     - oracle_diagnostic
     - 4/414 (0.966%)
     - 3/33 (9.091%)
     - 0/20 (0.000%)
   * - Late PID 0.1 candidate
     - average_link_probability
     - 4/414 (0.966%)
     - 3/33 (9.091%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - learned_confidence_mean
     - 4/414 (0.966%)
     - 3/33 (9.091%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - learned_confidence_sum
     - 3/414 (0.725%)
     - 2/33 (6.061%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - normalized_joint_log_probability
     - 2/414 (0.483%)
     - 1/33 (3.030%)
     - 1/20 (5.000%)
   * - Late PID 0.1 candidate
     - oracle_diagnostic
     - 4/414 (0.966%)
     - 4/33 (12.121%)
     - 1/20 (5.000%)

On the common 20-event beam cohort, normalized-joint ranking yields one exact component per arm; average-link and confidence-mean rankings yield two control versus three candidate components. The coherent event-oracle diagnostic reaches three versus four components, but never adds a coherent forest beyond greedy (zero control versus one candidate on this subset). Ranking affects shallow component selection; these truth-assisted bounds are not deployable gains.

.. list-table::
   :header-rows: 1

   * - Arm
     - All returned candidates checked in both scopes
   * - Late PID 0.2 control
     - 62
   * - Late PID 0.1 candidate
     - 59

Phase59 paired uncertainty
----------------------------------------

10,000 paired event-cluster bootstrap draws resample the 25 strict events, recomputing ratios from summed counts. Intervals describe event uncertainty conditional on these trained models, not training-seed uncertainty; no multiple-comparison correction. Tiny, skewed samples and degenerate zero intervals limit interpretation. All 10,000 draws have defined denominators. The exact-component and coherent-forest intervals include zero; these one-event gains do not establish superiority.

.. list-table::
   :header-rows: 1

   * - Scope
     - Retained metric
     - Candidate minus control (percentage points)
     - 95% paired interval
   * - full
     - coherent_retained_forest
     - 4.000
     - [0.000, 12.000] pp
   * - full
     - lcag_pair_accuracy
     - 0.000
     - [0.000, 0.000] pp
   * - full
     - mother_pid_coverage
     - 0.000
     - [0.000, 0.000] pp
   * - full
     - perfect_lcag
     - 2.941
     - [0.000, 10.526] pp
   * - full
     - source_precision
     - 1.950
     - [-0.640, 4.365] pp
   * - full
     - source_recall
     - 1.637
     - [0.114, 3.262] pp
   * - half
     - coherent_retained_forest
     - 4.000
     - [0.000, 12.000] pp
   * - half
     - lcag_pair_accuracy
     - 0.000
     - [0.000, 0.000] pp
   * - half
     - mother_pid_coverage
     - 0.000
     - [0.000, 0.000] pp
   * - half
     - perfect_lcag
     - 2.564
     - [0.000, 9.091] pp
   * - half
     - source_precision
     - 2.407
     - [-0.348, 5.112] pp
   * - half
     - source_recall
     - 1.745
     - [0.000, 3.468] pp

Phase59 synthesis of the accumulated studies
--------------------------------------------------

* Phase40 changed data size, compute and cohort together, so it supplies no controlled learning curve. Immediate training-set growth is not established as the best use of compute.
* Phases34/42 did not establish a benefit from longer pretraining alone. Phase41’s lower pointer weight worsened selection recovery. Phase43 adaptation gains were not independently replicated by the seed-deviating Phase44 study; corrected Phases46/47 were mixed.
* Phase48’s PID update did not execute because of the autocast cache bug. Corrected Phases49–51 did not show a stable PID-update benefit. Phases52–54 did not establish a reproducible joint topology/forest gain from recovery-dose tuning.
* Phase55 parent-weight effects were shallow, mixed and runtime-sensitive. Phase56 and Phase58 control pretraining failures prevented matched comparisons. Phase57 used strict events for selection, so its recovered quality metrics remain contaminated diagnostics.
* Phase59 now establishes complete training for both lower-PID settings at this seed. Its independent 25-event pilot does not establish that lower PID weight improves representation or deep reconstruction. Hard target-policy incompatibilities also cannot be resolved by simply adding data or pretraining steps.

These phases differ in cohort, seed and sometimes runtime; cross-phase rates are not causal comparisons. Preserve every failed and contaminated study rather than treating its metrics as independent successes.

Phase60: bounded fresh-validation replication
--------------------------------------------------

Keep identical training payloads (70,000 events), budgets (2,188 refinement plus 4,376 reconstruction), parent weight 2 and late PID 0.2 versus 0.1. Use seed 20260927 and fresh source-disjoint validation: 1,000 selection events plus 100 separate strict events, beam 20. Every other 99,000 validation UID is excluded from selection before training; 48,900 new validation events remain untouched afterward.

The added 10 shards follow the prior validation category quotas and are disjoint from all 200 prior inventory sources, including the 865k pretraining pool and sealed test. Their bytes match completion hashes. The new index scans train/validation records and fits normalization on train only. All historical roles and training shard hashes are preserved. The test payload is not opened.

The expanded full-record index reveals one level-5 target with 14 daughters, above the old limit 13. Both Phase60 arms raise that limit to 14 while retaining the global limit 16 and every acceptance threshold; no target is dropped. This shared capacity repair is an additional reason not to interpret cross-phase rate changes causally.

This is one replication of the existing objective contrast, not another dose search. Prefer repairing representation objectives, target compatibility and validation reliability to buying a larger training set now. A genuine quality advantage still needs joint topology/coherent-forest evidence and replication; numerical stability alone is insufficient.

Phase60 submission details are recorded in the dashboard snapshot and immutable contracts. No promotion, sealed-test access or automatic follow-on campaign is authorized.

Daughter-sum four-momentum closure checks implementation consistency. It is not physical mother momentum resolution, which remains unavailable.
