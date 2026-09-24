Phase58 reconstruction and pretraining review
=============================================

**Incomplete comparison; no promotion.** The late-PID 0.4 control failed
pretraining; the late-PID 0.2 arm completed but reconstructed zero full roots.
There is no paired reconstruction estimate and no demonstrated pretraining
quality advantage. Review date: 24 September 2026.

The :doc:`dashboard <_generated/status/index>` contains all original and retained
full/half metrics, beam rankings, micro/macro and exact combinations, and
pretraining downloads. The failed control's reconstruction is
**UNAVAILABLE_NOT_TRAINED**, never zero.

Execution and independence
--------------------------

The control completed 1752 pretraining steps and 56064 presentations. Attempted
step 1753 failed the unchanged objective preflight: weighted leaf-PID/LCA gradient
dominance 23.1619 exceeded 20. Its original failed receipt and partial checkpoints
are preserved. The lower-PID arm completed 2188 refinement steps, 70,000 presentations,
and 4,376 reconstruction steps. Both use parent weight 2; late PID weights differ
only after step 1094. Their step 1094 model tensors are identical. Weighted losses
under different objective weights are not representation-quality comparisons.
The completed model's PID head remains frozen during reconstruction.

The corrected selector admitted exactly the intended 1,000 checkpoint-selection
UIDs and excluded every other 49,000 validation UID before training. The completed
arm's ordered post-training cohort audit also passes with zero strict overlap.
All 100 strict evaluation events are fresh relative to observed prior studies;
beam uses the first 20. Phase57 remains selection-contaminated diagnostics and
is not an independent benchmark. Source, receipt, contract, checkpoint and report
hashes, finite weights, refined-checkpoint lineage and identical primary repeats
were checked. No sealed-test payload was accessed.

Original-policy metrics and gates
---------------------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Completed lower-PID arm
   * - exact_mother_coverage
     - 4/210 (1.905%)
   * - full_lcag
     - 4/3590 (0.111%)
   * - full_root_completion
     - 0/100 (0.000%)
   * - full_source_precision
     - 70/85 (82.353%)
   * - full_source_recall
     - 70/410 (17.073%)
   * - half_lcag
     - 22/2223 (0.990%)
   * - half_perfect_lcag
     - 3/153 (1.961%)
   * - half_root_pid_accuracy
     - 22/153 (14.379%)
   * - half_source_precision
     - 169/305 (55.410%)
   * - half_source_recall
     - 169/706 (23.938%)

.. list-table::
   :header-rows: 1

   * - Gate
     - Pass
   * - minimum_complete_target_efficiency
     - True
   * - minimum_depth_fraction
     - True
   * - minimum_full_source_precision
     - True
   * - minimum_full_source_recall
     - True
   * - minimum_half_lcag
     - True
   * - minimum_half_perfect_lcag
     - True
   * - minimum_half_root_pid_accuracy
     - True
   * - minimum_half_source_precision
     - True
   * - minimum_half_source_recall
     - True
   * - nonzero_exact_mother_coverage
     - True
   * - nonzero_full_lcag
     - True
   * - nonzero_full_root_completion
     - False
   * - primary_repeat_identical
     - True
   * - structural_guardrails
     - True

Primary selected step 4,376; checkpoint-selection recovery 277/3457. This is a selection metric, not strict tree efficiency.

Retained full and half reconstruction
-------------------------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Full
     - Half
   * - source_recall
     - 2926/3643 (80.318%)
     - 2277/2953 (77.108%)
   * - source_precision
     - 2926/3215 (91.011%)
     - 2277/2515 (90.537%)
   * - lcag_pair_accuracy
     - 26/4054 (0.641%)
     - 28/2314 (1.210%)
   * - perfect_lcag
     - 8/149 (5.369%)
     - 8/173 (4.624%)
   * - coherent_retained_forest
     - 0/100 (0.000%)
     - 0/100 (0.000%)
   * - mother_pid_coverage
     - 26/570 (4.561%)
     - 28/546 (5.128%)
   * - mother_pid_accuracy
     - 26/26 (100.000%)
     - 28/28 (100.000%)
   * - root_pid_accuracy
     - 346/1642 (21.072%)
     - 293/1323 (22.147%)
   * - target_representable
     - 2896/3028 (95.641%)
     - 2216/2362 (93.819%)

.. list-table::
   :header-rows: 1

   * - Reference population
     - Full
     - Half
   * - events_without_flagged_target_incompatibility
     - 17
     - 17
   * - isolated_leaf_units
     - 2798
     - 2108
   * - nontrivial_topology_units
     - 149
     - 173
   * - representable_nontrivial_units
     - 98
     - 108
   * - single_source_composite_units
     - 81
     - 81
   * - source_empty_units
     - 0
     - 0

All explicit retained roots are scored, including isolated leaves, single-source
composites and targets outside the training policy. Isolated leaves do not earn
exact nontrivial LCAG successes. High source retention can reflect unchanged input
leaves rather than reconstructed mothers. Policy-incompatible targets remain
failed primary trials; contracted topology is a separate diagnostic. Mother p4
closure checks daughter-sum consistency, not physical momentum resolution.

Beam search and checkpoint views
--------------------------------

All 60 returned coherent beam candidates are checked in both scopes. Search remains truth-free; oracle values use truth only after generation and are not deployable rankings.

.. list-table::
   :header-rows: 1

   * - full ranking
     - LCAG
     - Exact components
     - Coherent forest
   * - average_link_probability
     - 6/838 (0.716%)
     - 3/29 (10.345%)
     - 0/20 (0.000%)
   * - learned_confidence_mean
     - 8/838 (0.955%)
     - 5/29 (17.241%)
     - 0/20 (0.000%)
   * - learned_confidence_sum
     - 7/838 (0.835%)
     - 2/29 (6.897%)
     - 0/20 (0.000%)
   * - normalized_joint_log_probability
     - 7/838 (0.835%)
     - 3/29 (10.345%)
     - 0/20 (0.000%)
   * - oracle_diagnostic
     - 9/838 (1.074%)
     - 6/29 (20.690%)
     - 0/20 (0.000%)

.. list-table::
   :header-rows: 1

   * - half ranking
     - LCAG
     - Exact components
     - Coherent forest
   * - average_link_probability
     - 6/463 (1.296%)
     - 3/35 (8.571%)
     - 0/20 (0.000%)
   * - learned_confidence_mean
     - 8/463 (1.728%)
     - 5/35 (14.286%)
     - 0/20 (0.000%)
   * - learned_confidence_sum
     - 7/463 (1.512%)
     - 2/35 (5.714%)
     - 0/20 (0.000%)
   * - normalized_joint_log_probability
     - 7/463 (1.512%)
     - 3/35 (8.571%)
     - 0/20 (0.000%)
   * - oracle_diagnostic
     - 9/463 (1.944%)
     - 6/35 (17.143%)
     - 0/20 (0.000%)

.. list-table::
   :header-rows: 1

   * - View
     - Full exact
     - Full forest
     - Half exact
     - Half forest
   * - independent_complete_target_direct
     - 8/149 (5.369%)
     - 0/100 (0.000%)
     - 8/173 (4.624%)
     - 0/100 (0.000%)
   * - independent_depth_direct
     - 7/149 (4.698%)
     - 0/100 (0.000%)
     - 7/173 (4.046%)
     - 0/100 (0.000%)
   * - independent_tree_validity_direct
     - 7/149 (4.698%)
     - 0/100 (0.000%)
     - 7/173 (4.046%)
     - 0/100 (0.000%)
   * - primary_complete_target_beam_direct
     - 3/29 (10.345%)
     - 0/20 (0.000%)
     - 3/35 (8.571%)
     - 0/20 (0.000%)
   * - primary_complete_target_contracted_diagnostic
     - 8/149 (5.369%)
     - 0/100 (0.000%)
     - 8/173 (4.624%)
     - 0/100 (0.000%)
   * - primary_complete_target_direct
     - 8/149 (5.369%)
     - 0/100 (0.000%)
     - 8/173 (4.624%)
     - 0/100 (0.000%)
   * - primary_complete_target_repeat2_direct
     - 8/149 (5.369%)
     - 0/100 (0.000%)
     - 8/173 (4.624%)
     - 0/100 (0.000%)

No exact component deeper than one generation occurs in any of the seven views or returned beam candidates. Views, scopes, repeats and alternative candidates are not independent trials.

Uncertainty and pretraining
--------------------------

.. list-table::
   :header-rows: 1

   * - Scope
     - Metric
     - Estimate
     - 95% event-bootstrap interval
   * - full
     - lcag_pair_accuracy
     - 26/4054 (0.641%)
     - 0.003847 to 0.010687
   * - full
     - mother_pid_coverage
     - 26/570 (4.561%)
     - 0.030142 to 0.063790
   * - full
     - source_recall
     - 2926/3643 (80.318%)
     - 0.778863 to 0.827558
   * - full
     - source_precision
     - 2926/3215 (91.011%)
     - 0.900637 to 0.919084
   * - half
     - lcag_pair_accuracy
     - 28/2314 (1.210%)
     - 0.007600 to 0.018811
   * - half
     - mother_pid_coverage
     - 28/546 (5.128%)
     - 0.035026 to 0.070140
   * - half
     - source_recall
     - 2277/2953 (77.108%)
     - 0.727486 to 0.808915
   * - half
     - source_precision
     - 2277/2515 (90.537%)
     - 0.894449 to 0.915482

Intervals use 10,000 resamples of whole events and ratios of summed counts, conditional on this trained model; they do not measure training-seed uncertainty. No paired interval exists because the control never trained reconstruction.

.. list-table::
   :header-rows: 1

   * - Arm
     - Pretraining steps
     - Presentations
   * - lower_late_pid_pretraining
     - 2188
     - 70000.0
   * - pretraining_balance_control
     - 1752
     - 56064.0

The complete pretraining download retains all recorded validation checkpoints. The failed control has no step 2188 endpoint; earlier checkpoints cannot be compared as matched final performance.

Decision across the studies and Phase59
--------------------------------------

Hold the training set at 70,000 events. Phase40 changed size, compute and cohort
together and did not establish a controlled learning curve. Longer pretraining
in34/42 did not establish a benefit; pointer reduction41 worsened reconstruction.
Adaptation43-47, corrected PID studies49-51 and recovery-dose52-54 produced no
stable joint topology/forest gain. Phase48's intended PID update did not execute.
Parent refinement55 was mixed,56 lost its control,57 contaminated reconstruction
selection, and58 again lost its control. Neither training-data growth nor improved
pretraining has demonstrated superior downstream reconstruction. Cross-phase
rates on different seeds/cohorts are descriptive, not causal comparisons.

**Prioritize fresh independent validation and pretraining objective stability.**
Training-data expansion is not necessary yet on this evidence. More examples or
longer pretraining cannot alone repair hard target-policy incompatibilities.
The successful 0.2 execution supports testing stability, not claiming better
representations or successful deep reconstruction.

Phase59 is one bounded **feasibility pilot**, late PID 0.2 versus 0.1 at parent 2,
seed 20260926, fixed 2,188 refinement and 4,376 reconstruction steps. Initialization,
normalization, data, optimizer budgets, frozen reconstruction PID and the
objective-dominance 20 fail guard stay matched. It reuses the exact 1,000-event
selection set in the new seed order, excludes all other 49,000 before selection,
and reserves the final 25 untouched strict events; beam checks 20. Its unchanged
numerical gate thresholds are descriptive on this smaller sample, not a
confirmatory quality certification. No promotion or automatic campaign chain.

**Zero untouched validation events remain after Phase59 reservation.** Before
another campaign, add an independently held-out validation cohort with audited
source/event separation and immutable selection manifests. Do not repurpose
sealed test. The 25-event pilot is underpowered for generalization conclusions;
new validation capacity takes priority over another training-size or dose sweep.

Metric completeness
-------------------

.. list-table::
   :header-rows: 1

   * - Export
     - Scalar rows
   * - Original aggregate
     - 7902
   * - Original detailed
     - 76778
   * - Retained aggregate
     - 9751
   * - Retained detailed
     - 164160
   * - Every retained tree and beam candidate
     - 4581879
   * - Micro/macro and exact combinations
     - 2520
   * - Reconstruction histories
     - 374581
   * - Pretraining histories
     - 917047

The private complete-metrics archive preserves native reports, logs, checkpoints metadata, source hashes and every per-tree scalar export. Public downloads contain allowlisted aggregate fields; no event identifiers or private paths are published. Missing physical momentum-resolution estimates and failed-control reconstruction remain explicitly unavailable.
