# Current audit backlog

This file is generated from `issue_ledger.yaml` by
`scripts/generate_audit_views.py`; do not edit status counts here.

## Ledger status counts

| Status | Count |
|---|---:|
| `FIXED_AND_TESTED` | 66 |
| `IMPLEMENTED_NOT_REAL_VERIFIED` | 4 |
| `INTENTIONALLY_DEFERRED_SCIENCE` | 4 |
| `OBSOLETE_OR_DUPLICATE` | 1 |
| `PARTIAL` | 5 |

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

## Issue-evidence matrix

Generated directly from the ledger; paths are repository-relative.

| Issue | Status | Source evidence | Focused tests | Notebook evidence | External evidence | Last verified code SHA |
|---|---|---|---|---|---|---|
| DATA-001 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py<br>src/hypertagging/data/heterogeneous.py | tests/test_production_schema_v4_cpu.py<br>tests/test_no_truth_leakage_cpu.py | notebooks/inspect_leaf_input_pid_contract.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-002 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py<br>src/hypertagging/models/heterogeneous.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_no_truth_leakage_cpu.py | notebooks/inspect_leaf_pid_and_composite_inputs.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-003 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_revised_rollout_cpu.py | notebooks/preprocessing_qa_report.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-004 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py | tests/test_data_contracts_cpu.py<br>tests/test_revised_model_cpu.py | notebooks/inspect_preprocessed_dataset.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-005 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py | tests/test_preprocessing_adapters_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-006 | `FIXED_AND_TESTED` | src/hypertagging/data/streaming.py<br>src/hypertagging/preprocessing/schema_v4.py | tests/test_streaming_data_pipeline_cpu.py<br>tests/test_production_manifest_training_integration_cpu.py | notebooks/inspect_production_manifest.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-007 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py<br>src/hypertagging/losses/level_reconstruction.py | tests/test_partial_target_policy_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-008 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py<br>notebooks/inspect_real_mdst_pilot.ipynb | tests/test_preprocessing_adapters_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-009 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py | tests/test_production_schema_v4_cpu.py<br>tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_streaming_dataset.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-010 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py<br>src/hypertagging/data/streaming.py | tests/test_atomic_overwrite_marker_cpu.py<br>tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_production_manifest.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-011 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v5.py<br>configs/README.md | tests/test_native_v5_schema_stability_cpu.py | notebooks/inspect_runtime_scaling.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-012 | `INTENTIONALLY_DEFERRED_SCIENCE` | notebooks/inspect_runtime_scaling.ipynb | tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_runtime_scaling.ipynb | representative ten-million-event production run | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-001 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py | tests/test_contextual_hyperbolic_encoder_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-002 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py | tests/test_contextual_hyperbolic_encoder_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-003 | `FIXED_AND_TESTED` | src/hypertagging/models/relation_attention.py | tests/test_revised_model_cpu.py | notebooks/inspect_leaf_pid_and_composite_inputs.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-004 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py<br>src/hypertagging/models/hyperbolic.py | tests/test_post_audit_geometry_rollout_cpu.py<br>tests/test_losses_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-005 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-006 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/models/level_autoregressive.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-007 | `FIXED_AND_TESTED` | src/hypertagging/models/level_autoregressive.py<br>src/hypertagging/models/mother_pointer.py | tests/test_level_autoregressive_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-008 | `IMPLEMENTED_NOT_REAL_VERIFIED` | src/hypertagging/models/level_autoregressive.py<br>src/hypertagging/losses/level_reconstruction.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | trained held-out duplicate and unused-query rates | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-009 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/level_rollout.py | tests/test_revised_rollout_cpu.py | notebooks/inspect_rollout_search_and_calibration.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-010 | `FIXED_AND_TESTED` | src/hypertagging/losses/level_reconstruction.py<br>src/hypertagging/data/dataset_index.py | tests/test_query_capacity_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-011 | `FIXED_AND_TESTED` | configs/model_presets/production_baseline.yaml<br>src/hypertagging/data/dataset_index.py | tests/test_query_capacity_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-012 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/constraints.py | tests/test_current_head_reconstruction_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-013 | `FIXED_AND_TESTED` | src/hypertagging/losses/level_reconstruction.py | tests/test_confidence_calibration_cpu.py | notebooks/inspect_rollout_search_and_calibration.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-014 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/constraints.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_recursive_source_exclusivity_cpu.py | notebooks/inspect_rollout_search_and_calibration.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-015 | `FIXED_AND_TESTED` | src/hypertagging/evaluation/hierarchical_metrics.py | tests/test_non_tautological_tree_metrics_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-016 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/level_rollout.py | tests/test_revised_rollout_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-017 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_unrepresentable_scheduled_targets_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-018 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_two_pass_leaf_pid_reconstruction_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-019 | `IMPLEMENTED_NOT_REAL_VERIFIED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_two_pass_leaf_pid_reconstruction_cpu.py | notebooks/inspect_leaf_pid_and_composite_inputs.ipynb | trained held-out comparison of PID construction modes | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-020 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py | tests/test_post_audit_geometry_rollout_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-021 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py | tests/test_post_audit_geometry_rollout_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-022 | `FIXED_AND_TESTED` | src/hypertagging/models/relations.py | tests/test_post_audit_geometry_rollout_cpu.py | notebooks/inspect_exact_tree_geometry_and_loss_scales.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-023 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>docs/deferred_model_ablations.md | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-001 | `FIXED_AND_TESTED` | scripts/train_hyperbolic_pretrain.py<br>scripts/train_level_reconstruction.py | tests/test_real_training_pipeline_cpu.py<br>tests/test_training_dry_run_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-002 | `FIXED_AND_TESTED` | src/hypertagging/data/streaming.py | tests/test_streaming_data_pipeline_cpu.py | notebooks/inspect_streaming_dataset.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-003 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/training/reconstruction_trainer.py | tests/test_exact_resume_cpu.py<br>tests/test_resume_data_order_contract_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-004 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_scheduled_sampling_forward_count_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-005 | `FIXED_AND_TESTED` | src/hypertagging/training/reconstruction_trainer.py | tests/test_validation_batch_size_cpu.py<br>tests/test_validation_micro_aggregation_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-006 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py | tests/test_current_head_acceptance_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-007 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py | tests/test_current_head_acceptance_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-008 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py | tests/test_current_head_acceptance_cpu.py | notebooks/inspect_training_pipeline.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-009 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_curriculum_runtime_semantics_cpu.py<br>tests/test_corrupted_structural_loss_mask_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-010 | `OBSOLETE_OR_DUPLICATE` | docs/channel_representation.md | tests/test_channel_representation_round_trip_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-011 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/channels.py<br>docs/channel_representation.md | tests/test_channel_representation_round_trip_cpu.py | notebooks/inspect_preprocessed_dataset.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-012 | `IMPLEMENTED_NOT_REAL_VERIFIED` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/models/ablation.py | tests/test_channel_cross_event_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | matched held-out channel-pooling ablations | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-013 | `FIXED_AND_TESTED` | configs/target_policies/complete_only.yaml<br>docs/training.md | tests/test_complete_target_efficiency_cpu.py | notebooks/inspect_query_capacity_and_losses.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-014 | `FIXED_AND_TESTED` | docs/deferred_model_ablations.md | tests/test_model_config_round_trip_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-015 | `FIXED_AND_TESTED` | scripts/condor/render_condor_job.py<br>scripts/condor/submit_hyperbolic_pretrain.sh | tests/test_htcondor_training_safety_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-016 | `INTENTIONALLY_DEFERRED_SCIENCE` | notebooks/inspect_trained_physics_validation.ipynb | tests/test_trained_evaluation_context_cpu.py | notebooks/inspect_trained_physics_validation.ipynb | trained checkpoint<br>matched held-out real schema-v4 data | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-017 | `PARTIAL` | src/hypertagging/preprocessing/basf2_mdst.py<br>notebooks/inspect_real_mdst_pilot.ipynb | tests/test_preprocessing_adapters_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | representative study of unmatched KLM clusters and K_L reconstructability across generic categories | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-001 | `FIXED_AND_TESTED` | scripts/execute_notebook_smoke_tests.py<br>notebooks/index.yaml | tests/test_revised_notebooks_cpu.py | notebooks/inspect_preprocessed_dataset.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-002 | `FIXED_AND_TESTED` | notebooks/README.md<br>notebooks/index.yaml | tests/test_revised_notebooks_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-003 | `PARTIAL` | .github/workflows/cpu-tests.yml<br>notebooks/index.yaml | tests/test_revised_notebooks_cpu.py | none | remote workflow execution for a committed final SHA | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-004 | `PARTIAL` | scripts/execute_notebook_smoke_tests.py | tests/test_revised_notebooks_cpu.py | notebooks/preprocessing_four_momentum_validation.ipynb | human visual review for plot legibility | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-005 | `FIXED_AND_TESTED` | notebooks/inspect_real_mdst_pilot.ipynb<br>scripts/create_real_mdst_pilot_notebook.py | tests/test_revised_notebooks_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-006 | `IMPLEMENTED_NOT_REAL_VERIFIED` | notebooks/inspect_trained_physics_validation.ipynb<br>scripts/create_trained_physics_validation_notebook.py | tests/test_trained_evaluation_context_cpu.py | notebooks/inspect_trained_physics_validation.ipynb | trained checkpoint<br>held-out real schema-v4 data | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-001 | `FIXED_AND_TESTED` | src/hypertagging/models/relation_attention.py | tests/test_final_audit_corrections_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-002 | `INTENTIONALLY_DEFERRED_SCIENCE` | docs/deferred_model_ablations.md<br>src/hypertagging/models/first_level_ablations.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | scientifically coherent proposal-ranking and training design | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-003 | `INTENTIONALLY_DEFERRED_SCIENCE` | docs/deferred_model_ablations.md<br>src/hypertagging/models/first_level_ablations.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | scientifically coherent differentiable decoding design | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-004 | `FIXED_AND_TESTED` | src/hypertagging/models/mother_pointer.py<br>src/hypertagging/models/level_autoregressive.py<br>configs/ablations/first_level_type_relation_bias.yaml | tests/test_final_audit_corrections_cpu.py<br>tests/test_model_config_round_trip_cpu.py | notebooks/inspect_first_level_ambiguity.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-005 | `FIXED_AND_TESTED` | src/hypertagging/data/heterogeneous.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_leaf_input_pid_contract.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-006 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-007 | `PARTIAL` | src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | guarded CUDA smoke<br>representative memory and throughput profiling | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-008 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_channel_cross_event_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-009 | `FIXED_AND_TESTED` | src/hypertagging/losses/hyperbolic_pretraining.py | tests/test_channel_cross_event_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-010 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>configs/hyperbolic_pretrain.yaml | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-011 | `FIXED_AND_TESTED` | docs/audits/README.md<br>docs/audits/current_status.md<br>scripts/validate_audit_integrity.py | tests/test_audit_integrity_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-012 | `FIXED_AND_TESTED` | notebooks/index.yaml<br>scripts/execute_notebook_smoke_tests.py | tests/test_revised_notebooks_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-013 | `PARTIAL` | src/hypertagging/training/pretrain_trainer.py<br>src/hypertagging/training/reconstruction_trainer.py<br>src/hypertagging/reconstruction/level_rollout.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_runtime_scaling.ipynb | guarded CUDA run<br>representative GPU profiling | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| FINAL-014 | `FIXED_AND_TESTED` | scripts/execute_notebook_smoke_tests.py<br>notebooks/index.yaml | tests/test_audit_integrity_cpu.py<br>tests/test_revised_notebooks_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| AUDIT-001 | `FIXED_AND_TESTED` | scripts/validate_audit_integrity.py<br>docs/audits/issue_ledger.yaml | tests/test_audit_integrity_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| TRAIN-018 | `FIXED_AND_TESTED` | src/hypertagging/training/pretrain_trainer.py<br>scripts/train_hyperbolic_pretrain.py<br>configs/hyperbolic_pretrain.yaml | tests/test_final_audit_corrections_cpu.py<br>tests/test_real_training_pipeline_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb | trained held-out comparison before selecting scientific weights | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| ROLLOUT-006 | `FIXED_AND_TESTED` | src/hypertagging/reconstruction/level_rollout.py<br>src/hypertagging/data/heterogeneous.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DOC-001 | `FIXED_AND_TESTED` | docs/hyperbolic_level_autoregressive_reconstruction.md<br>docs/heterogeneous_node_encoding.md<br>src/hypertagging/models/relations.py | tests/test_final_audit_corrections_cpu.py | none | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| NB-007 | `FIXED_AND_TESTED` | scripts/execute_notebook_smoke_tests.py<br>notebooks/index.yaml<br>.github/workflows/cpu-tests.yml | tests/test_audit_integrity_cpu.py | none | human figure review remains NOT_REVIEWED | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| MODEL-025 | `FIXED_AND_TESTED` | src/hypertagging/models/heterogeneous.py<br>src/hypertagging/models/level_autoregressive.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_hyperbolic_pretraining.ipynb<br>notebooks/inspect_level_autoregressive_reconstruction.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-013 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/basf2_mdst.py<br>scripts/create_real_mdst_pilot_notebook.py | tests/test_final_audit_corrections_cpu.py<br>tests/test_current_source_boundary_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
| DATA-014 | `FIXED_AND_TESTED` | src/hypertagging/preprocessing/schema_v4.py<br>scripts/preprocess_mdst.py | tests/test_final_audit_corrections_cpu.py | notebooks/inspect_real_mdst_pilot.ipynb | none | `554a553f465d7fe3056e1e1ba95ad71b37530816` |
