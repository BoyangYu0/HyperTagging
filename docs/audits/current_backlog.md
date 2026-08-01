# Current audit backlog

This file is generated from `issue_ledger.yaml` by
`scripts/generate_audit_views.py`; do not edit status counts here.

## Ledger status counts

| Status | Count |
|---|---:|
| `FIXED_AND_TESTED` | 72 |
| `IMPLEMENTED_NOT_REAL_VERIFIED` | 4 |
| `INTENTIONALLY_DEFERRED_SCIENCE` | 4 |
| `OBSOLETE_OR_DUPLICATE` | 1 |
| `PARTIAL` | 7 |

## Unresolved or externally bounded items

### DATA-012: Ten-million-event throughput and memory readiness

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: representative ten-million-event production run
- Notes: Fixture timings are explicitly not throughput evidence.

### MODEL-008: Query duplication and collapse observability

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: trained held-out duplicate and unused-query rates
- Notes: Mechanics and metrics exist; scientific collapse frequency is unmeasured.

### MODEL-019: Soft training PID versus hard rollout PID mismatch invisible

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: trained held-out comparison of PID construction modes
- Notes: Modes and diagnostics work; the best physics mode is unknown.

### TRAIN-012: Channel pooling had only one untested choice

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: matched held-out channel-pooling ablations
- Notes: Modes execute; scientific ranking is deferred.

### TRAIN-016: Scientific performance, calibration, rare-channel quality, and optimal decoding

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: trained checkpoint, matched held-out real schema-v4 data
- Notes: No physics-improvement claim is permitted from fixtures.

### TRAIN-017: KLM and K_L detector completeness

- Status: `PARTIAL`
- External evidence: representative study of unmatched KLM clusters and K_L reconstructability across generic categories
- Notes: The charged-B pilot collected 92 explicit KLM nodes, but only 16 of 48 K_L-like leaves carried KLM provenance; detector completeness therefore remains partial.

### NB-003: Normal push CI did not run every notebook

- Status: `PARTIAL`
- External evidence: remote workflow execution for a committed final SHA
- Notes: Local full suite is executable; CI cadence is indexed, but the working tree has no run.

### NB-004: Notebook assertions only proved a figure existed

- Status: `PARTIAL`
- External evidence: human visual review for plot legibility
- Notes: Core JSON semantics are asserted; visual interpretation remains human review.

### NB-006: Trained rollout notebook lacked divergence, ambiguity, slices, and PID diagnostics

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: trained checkpoint, held-out real schema-v4 data
- Notes: Required diagnostics are implemented but deliberately NOT RUN here.

### FINAL-002: Whole-set compatibility scorer appeared runnable but was disconnected

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: scientifically coherent proposal-ranking and training design
- Notes: Runnable YAML was removed; CLI cannot silently accept the old field.

### FINAL-003: Iterative pointer mask appeared runnable but was disconnected

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: scientifically coherent differentiable decoding design
- Notes: Runnable YAML was removed; bounded iterative decoding remains a design only.

### FINAL-007: Free rollout lacked a batched validation step

- Status: `PARTIAL`
- External evidence: guarded CUDA smoke, representative memory and throughput profiling
- Notes: batched_free_rollout now completes multiple levels with per-event stopping and matches independent batch-size-one references on deterministic CPU fixtures; GPU readiness remains unverified.

### FINAL-013: GPU throughput and remaining scalar synchronization

- Status: `PARTIAL`
- External evidence: guarded CUDA run, representative GPU profiling
- Notes: The padded batched free path avoids tensor-scalar extraction during level traversal; the bounded reference rollout and optional diagnostics retain Python work, and no CUDA profile exists.

### DATA-018: Production query and cardinality capacities lacked representative slices

- Status: `PARTIAL`
- External evidence: representative multi-category production dataset index
- Notes: The index reports per-level quantiles and source/event/neutral/channel-frequency slices with bounded coverage; the 50-event charged-B pilot had zero overflow, but it is not representative of production.

### AUDIT-002: Exact starting-HEAD CI failed because audit ancestry was unavailable in a shallow checkout

- Status: `PARTIAL`
- External evidence: successful workflow_dispatch or push run at the audited source or audit commit
- Notes: Run 30698801983 passed unit tests and notebook smoke but failed audit integrity; checkout now fetches full history and workflow_dispatch accepts an explicit SHA, but no post-fix remote run exists.

## Issue-evidence matrix

Generated directly from the ledger; paths are repository-relative.

| Issue | Status | Source evidence | Focused tests | Notebook evidence | External evidence | Last verified code SHA |
|---|---|---|---|---|---|---|
| DATA-001 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py<br>src/hypertagging/data/heterogeneous.py | tests/test_production_schema_v4_cpu.py<br>tests/test_no_truth_leakage_cpu.py | notebooks/inspect_leaf_input_pid_contract.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-002 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py<br>src/hypertagging/models/heterogeneous.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_no_truth_leakage_cpu.py | notebooks/inspect_leaf_pid_and_composite_inputs.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-003 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_revised_rollout_cpu.py | notebooks/preprocessing_qa_report.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-004 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py | tests/test_data_contracts_cpu.py<br>tests/test_revised_model_cpu.py | notebooks/inspect_preprocessed_dataset.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-005 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py | tests/test_preprocessing_adapters_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-006 | `FIXED_AND_TESTED` | src/hypertagging/data/streaming.py<br>src/hypertagging/preprocessing/schema_v4.py | tests/test_streaming_data_pipeline_cpu.py<br>tests/test_production_manifest_training_integration_cpu.py | notebooks/inspect_production_manifest.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-007 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py<br>src/hypertagging/losses/level_reconstruction.py | tests/test_partial_target_policy_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-008 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py<br>notebooks/inspect_real_mdst_pilot.ipynb | tests/test_preprocessing_adapters_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-009 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py | tests/test_production_schema_v4_cpu.py<br>tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_streaming_dataset.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-010 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py<br>src/hypertagging/data/streaming.py | tests/test_atomic_overwrite_marker_cpu.py<br>tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_production_manifest.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-011 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v5.py<br>configs/README.md | tests/test_native_v5_schema_stability_cpu.py | notebooks/inspect_runtime_scaling.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-012 | `INTENTIONALLY_DEFERRED_SCIENCE` | notebooks/inspect_runtime_scaling.ipynb | tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_runtime_scaling.ipynb | representative ten-million-event production run | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-001 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py | tests/test_contextual_hyperbolic_encoder_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-002 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py | tests/test_contextual_hyperbolic_encoder_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-003 | `FIXED_AND_TESTED` | src/hypertagging/models/relation_attention.py | tests/test_revised_model_cpu.py | notebooks/inspect_leaf_pid_and_composite_inputs.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-004 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py<br>src/hypertagging/models/hyperbolic.py | tests/test_post_audit_geometry_rollout_cpu.py<br>tests/test_losses_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-005 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-006 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/models/level_autoregressive.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-007 | `FIXED_AND_TESTED` | src/hypertagging/models/level_autoregressive.py<br>src/hypertagging/models/mother_pointer.py | tests/test_level_autoregressive_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-008 | `IMPLEMENTED_NOT_REAL_VERIFIED` | src/hypertagging/models/level_autoregressive.py<br>src/hypertagging/losses/level_reconstruction.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | trained held-out duplicate and unused-query rates | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-009 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/level_rollout.py | tests/test_revised_rollout_cpu.py | notebooks/inspect_rollout_search_and_calibration.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-010 | `FIXED_AND_TESTED` | src/hypertagging/losses/level_reconstruction.py<br>src/hypertagging/data/dataset_index.py | tests/test_query_capacity_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-011 | `FIXED_AND_TESTED` | configs/model_presets/production_baseline.yaml<br>src/hypertagging/data/dataset_index.py | tests/test_query_capacity_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-012 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/constraints.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-013 | `FIXED_AND_TESTED` | src/hypertagging/losses/level_reconstruction.py | tests/test_confidence_calibration_cpu.py | notebooks/inspect_rollout_search_and_calibration.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-014 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/constraints.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_recursive_source_exclusivity_cpu.py | notebooks/inspect_rollout_search_and_calibration.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-015 | `FIXED_AND_TESTED` | src/hypertagging/evaluation/hierarchical_metrics.py | tests/test_non_tautological_tree_metrics_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-016 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/level_rollout.py | tests/test_revised_rollout_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-017 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_unrepresentable_scheduled_targets_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-018 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_two_pass_leaf_pid_reconstruction_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-019 | `IMPLEMENTED_NOT_REAL_VERIFIED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_two_pass_leaf_pid_reconstruction_cpu.py | notebooks/inspect_leaf_pid_and_composite_inputs.ipynb | trained held-out comparison of PID construction modes | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-020 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py | tests/test_post_audit_geometry_rollout_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-021 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py | tests/test_post_audit_geometry_rollout_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-022 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py | tests/test_post_audit_geometry_rollout_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-023 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>docs/deferred_model_ablations.md | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-001 | `FIXED_AND_TESTED` | scripts/train_hyperbolic_pretrain.py<br>scripts/train_level_reconstruction.py | tests/test_real_training_pipeline_cpu.py<br>tests/test_training_dry_run_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-002 | `FIXED_AND_TESTED` | src/hypertagging/data/streaming.py | tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_streaming_dataset.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-003 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/training/reconstruction_trainer.py | tests/test_exact_resume_cpu.py<br>tests/test_resume_data_order_contract_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-004 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_scheduled_sampling_forward_count_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-005 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_validation_batch_size_cpu.py<br>tests/test_validation_micro_aggregation_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-006 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py | tests/test_current_head_acceptance_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-007 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py | tests/test_current_head_acceptance_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-008 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py | tests/test_current_head_acceptance_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-009 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_curriculum_runtime_semantics_cpu.py<br>tests/test_corrupted_structural_loss_mask_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-010 | `OBSOLETE_OR_DUPLICATE` | docs/channel_representation.md | tests/test_channel_representation_round_trip_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-011 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/channels.py<br>docs/channel_representation.md | tests/test_channel_representation_round_trip_cpu.py | notebooks/inspect_preprocessed_dataset.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-012 | `IMPLEMENTED_NOT_REAL_VERIFIED` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/models/ablation.py | tests/test_channel_cross_event_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | matched held-out channel-pooling ablations | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-013 | `FIXED_AND_TESTED` | configs/target_policies/complete_only.yaml<br>docs/training.md | tests/test_complete_target_efficiency_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-014 | `FIXED_AND_TESTED` | docs/deferred_model_ablations.md | tests/test_model_config_round_trip_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-015 | `FIXED_AND_TESTED` | scripts/condor/render_condor_job.py<br>scripts/condor/submit_hyperbolic_pretrain.sh | tests/test_htcondor_training_safety_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-016 | `INTENTIONALLY_DEFERRED_SCIENCE` | notebooks/inspect_trained_physics_validation.ipynb | tests/test_trained_evaluation_context_cpu.py | notebooks/inspect_trained_physics_validation.ipynb | trained checkpoint<br>matched held-out real schema-v4 data | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-017 | `PARTIAL` | src/hypertagging/preprocessing/basf2_mdst.py<br>notebooks/inspect_real_mdst_pilot.ipynb | tests/test_preprocessing_adapters_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | representative study of unmatched KLM clusters and K_L reconstructability across generic categories | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-001 | `FIXED_AND_TESTED` | scripts/execute_notebook_smoke_tests.py<br>notebooks/index.yaml | tests/test_revised_notebooks_cpu.py | notebooks/inspect_preprocessed_dataset.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-002 | `FIXED_AND_TESTED` | notebooks/README.md<br>notebooks/index.yaml | tests/test_revised_notebooks_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-003 | `PARTIAL` | .github/workflows/cpu-tests.yml<br>notebooks/index.yaml | tests/test_revised_notebooks_cpu.py | none | remote workflow execution for a committed final SHA | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-004 | `PARTIAL` | scripts/execute_notebook_smoke_tests.py | tests/test_revised_notebooks_cpu.py | notebooks/preprocessing_four_momentum_validation.ipynb | human visual review for plot legibility | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-005 | `FIXED_AND_TESTED` | notebooks/inspect_real_mdst_pilot.ipynb<br>scripts/create_real_mdst_pilot_notebook.py | tests/test_revised_notebooks_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-006 | `IMPLEMENTED_NOT_REAL_VERIFIED` | notebooks/inspect_trained_physics_validation.ipynb<br>scripts/create_trained_physics_validation_notebook.py | tests/test_trained_evaluation_context_cpu.py | notebooks/inspect_trained_physics_validation.ipynb | trained checkpoint<br>held-out real schema-v4 data | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-001 | `FIXED_AND_TESTED` | src/hypertagging/models/relation_attention.py | tests/test_final_audit_corrections_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-002 | `INTENTIONALLY_DEFERRED_SCIENCE` | docs/deferred_model_ablations.md<br>src/hypertagging/models/first_level_ablations.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | scientifically coherent proposal-ranking and training design | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-003 | `INTENTIONALLY_DEFERRED_SCIENCE` | docs/deferred_model_ablations.md<br>src/hypertagging/models/first_level_ablations.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | scientifically coherent differentiable decoding design | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-004 | `FIXED_AND_TESTED` | src/hypertagging/models/mother_pointer.py<br>src/hypertagging/models/level_autoregressive.py<br>configs/ablations/first_level_type_relation_bias.yaml | tests/test_final_audit_corrections_cpu.py<br>tests/test_model_config_round_trip_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-005 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_leaf_input_pid_contract.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-006 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-007 | `PARTIAL` | src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | guarded CUDA smoke<br>representative memory and throughput profiling | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-008 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_channel_cross_event_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-009 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_channel_cross_event_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-010 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>configs/hyperbolic_pretrain.yaml | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-011 | `FIXED_AND_TESTED` | docs/audits/README.md<br>docs/audits/current_status.md<br>scripts/validate_audit_integrity.py | tests/test_audit_integrity_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-012 | `FIXED_AND_TESTED` | notebooks/index.yaml<br>scripts/execute_notebook_smoke_tests.py | tests/test_revised_notebooks_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-013 | `PARTIAL` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/training/reconstruction_trainer.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_runtime_scaling.ipynb | guarded CUDA run<br>representative GPU profiling | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| FINAL-014 | `FIXED_AND_TESTED` | scripts/execute_notebook_smoke_tests.py<br>notebooks/index.yaml | tests/test_audit_integrity_cpu.py<br>tests/test_revised_notebooks_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| AUDIT-001 | `FIXED_AND_TESTED` | scripts/validate_audit_integrity.py<br>docs/audits/issue_ledger.yaml | tests/test_audit_integrity_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-018 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>scripts/train_hyperbolic_pretrain.py<br>configs/hyperbolic_pretrain.yaml | tests/test_final_audit_corrections_cpu.py<br>tests/test_real_training_pipeline_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | trained held-out comparison before selecting scientific weights | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| ROLLOUT-006 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/level_rollout.py<br>src/hypertagging/data/heterogeneous.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DOC-001 | `FIXED_AND_TESTED` | docs/hyperbolic_level_autoregressive_reconstruction.md<br>docs/heterogeneous_node_encoding.md<br>src/hypertagging/models/relations.py | tests/test_final_audit_corrections_cpu.py | none | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-007 | `FIXED_AND_TESTED` | scripts/execute_notebook_smoke_tests.py<br>notebooks/index.yaml<br>.github/workflows/cpu-tests.yml | tests/test_audit_integrity_cpu.py | none | human figure review remains NOT_REVIEWED | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-025 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/models/level_autoregressive.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb<br>notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-013 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py<br>scripts/create_real_mdst_pilot_notebook.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-014 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py<br>scripts/preprocess_mdst.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| MODEL-026 | `FIXED_AND_TESTED` | src/hypertagging/models/level_autoregressive.py<br>src/hypertagging/models/mother_pointer.py | tests/test_post_audit_information_boundaries_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| TRAIN-019 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py<br>src/hypertagging/training/pretraining_curriculum.py<br>src/hypertagging/training/pretrain_trainer.py<br>configs/hyperbolic_pretrain.yaml | tests/test_post_audit_information_boundaries_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | matched HTCondor training before selecting the structural-input compatibility ablation | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-015 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py<br>src/hypertagging/models/heterogeneous.py<br>src/hypertagging/preprocessing/schema_v4.py | tests/test_post_audit_information_boundaries_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-016 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py<br>src/hypertagging/preprocessing/mdst_tree_builder.py<br>src/hypertagging/preprocessing/schema_v3.py | tests/test_post_audit_information_boundaries_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | broader data-category coverage for association rates | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-017 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py<br>scripts/preprocess_mdst.py<br>scripts/mdst_batch_production.py | tests/test_post_audit_information_boundaries_cpu.py<br>tests/test_mdst_batch_production_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | trained held-out comparison before changing the default fit policy | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| DATA-018 | `PARTIAL` | src/hypertagging/data/dataset_index.py<br>src/hypertagging/data/capacity.py<br>scripts/report_reconstruction_capacity.py | tests/test_post_audit_information_boundaries_cpu.py<br>tests/test_query_capacity_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb<br>notebooks/inspect_query_capacity_and_losses.ipynb | representative multi-category production dataset index | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| NB-008 | `FIXED_AND_TESTED` | scripts/create_preprocessing_visualization_notebook.py<br>notebooks/index.yaml<br>docs/dataset_visualization.md | tests/test_revised_notebooks_cpu.py | notebooks/preprocessing_four_momentum_validation.ipynb | human figure review remains NOT_REVIEWED | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
| AUDIT-002 | `PARTIAL` | .github/workflows/cpu-tests.yml<br>.github/workflows/full-notebook-smoke.yml<br>scripts/validate_audit_integrity.py | tests/test_audit_integrity_cpu.py | none | successful workflow_dispatch or push run at the audited source or audit commit | `88270d00fb5c9fc6311daab2f9443832ebe7c3bf` |
