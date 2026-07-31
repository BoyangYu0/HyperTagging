# Audit index

Index date: 2026-07-31 (Europe/Berlin)

This file is the current status map. Every listed audit/report remains an
immutable snapshot at its recorded SHA. “Fixed” below means fixed in the
current working tree based on
`f064f49985da98b69c99fb02764d854f76c12e3c`; it does not mean the fix is
contained in that commit. External physics, basf2, large-scale, GPU, and CI
claims are never inferred from CPU fixtures.

## Snapshot chronology

| Snapshot | Date | Audited SHA | Historical/current | Superseded by |
|---|---|---|---|---|
| `model_revision_audit.md` | 2026-07-30 | `cf63b90` | historical | `correctness_revision_audit.md` |
| `correctness_revision_audit.md` | 2026-07-30 | `da8f35796ae81835645ac12721a5287829c02c31` | historical | `final_correctness_and_scalability_audit.md` |
| `final_correctness_and_scalability_audit.md` | 2026-07-30 | `9d37fde7df7cd59eda1d3464ad5351f224314d6c` | historical | `final_runtime_and_scale_audit.md` |
| `final_runtime_and_scale_audit.md` | 2026-07-31 | `d2362847cf036599d955cd4c70b9d2d9a3d83a08` | historical | `final_baseline_readiness_audit.md` |
| `final_baseline_readiness_audit.md` | 2026-07-31 | `274d2df8a9df8b25142b68966dfe30c828538b90` | historical | `current_head_gap_audit.md` |
| `final_production_readiness_report.md` | 2026-07-31 | `274d2df8a9df8b25142b68966dfe30c828538b90` plus recorded dirty patch | historical | `current_head_gap_audit.md` |
| `current_head_gap_audit.md` | 2026-07-31 | `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60` | historical | `post_audit_geometry_and_rollout_review.md` |
| `current_head_verification_report.md` | 2026-07-31 | `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60` plus recorded dirty patch | historical | post-audit reports |
| `post_audit_geometry_and_rollout_review.md` | 2026-07-31 | `68a7ebe81b14b6c329d78de91be18b5458c49089` plus recorded dirty patch | historical | this verification |
| `post_audit_retained_tree_definition.md` | 2026-07-31 | `68a7ebe81b14b6c329d78de91be18b5458c49089` plus recorded dirty patch | historical definition | this verification; definition retained |
| `post_audit_completion_report.md` | 2026-07-31 | `68a7ebe81b14b6c329d78de91be18b5458c49089` plus recorded dirty patch | historical | this verification |
| `current_head_verification.md` | 2026-07-31 | `f064f49985da98b69c99fb02764d854f76c12e3c` plus current dirty patch | current | `current_head_completion_report.md` when final checks complete |

## Issue ledger by snapshot

| Snapshot | Issues raised | Current status | Current proof |
|---|---|---|---|
| Model revision | Heterogeneous detector inputs, truth/input separation, exact topology, relation bias, and reconstruction migration were incomplete. | fixed for the shared baseline; real physics externally unverified | `src/hypertagging/data/heterogeneous.py`, `models/heterogeneous.py`, `models/relations.py`, `reconstruction/level_rollout.py`; `tests/test_revised_model_cpu.py`, `tests/test_current_head_reconstruction_corrections_cpu.py` |
| Correctness revision | Direct-mDST reco/truth separation, fit/energy/PID availability, retained links, p4 closure, and real-pilot evidence. | source contracts fixed and preserved; real basf2 pilot externally unverified | `src/hypertagging/preprocessing/basf2_mdst.py`, `schema_v4.py`; `tests/test_direct_mdst_preprocessing_cpu.py`, `tests/test_no_truth_leakage_cpu.py`; real notebook guard in `inspect_real_mdst_pilot.ipynb` |
| Final correctness/scalability | v4 event-row streaming, manifests/indexes, leakage-safe splits, capacity, resume, and large-scale readiness. | software contracts fixed; ten-million-event throughput/storage externally unverified | `data/streaming.py`, `data/dataset_index.py`, `training/data_module.py`, `training/checkpointing.py`; streaming/index/manifest/capacity tests and fixture notebooks |
| Runtime and scale | Raw values reintroduced after normalization; categorical leakage; teacher/predicted double counting; cursor resume; repeated scans; worker duplication; corruption/hard negatives; validation denominator gaps. | fixed for runtime normalization, scheduled context, cursor/index, corruption/negative labels, and bounded validation; native-v5/10M conclusions remain open/external | `data/streaming.py`, `training/data_module.py`, `training/pretrain_trainer.py`, `training/reconstruction_trainer.py`; `tests/test_runtime_dynamic_normalization_cpu.py`, `test_scheduled_sampling_training_cpu.py`, `test_exact_streaming_trainer_resume_cpu.py`, `test_post_audit_geometry_rollout_cpu.py` |
| Baseline readiness | Reconstruction constraints/metrics, beam/set packing, checkpoint/split contracts, trained real validation, channel shortcut and scale questions. | partially fixed: software evaluation/search/contracts and current ablations fixed; trained held-out and real calibration externally unverified | `reconstruction/constraints.py`, `level_rollout.py`, `evaluation/trained_context.py`; `tests/test_post_audit_geometry_rollout_cpu.py`, `test_trained_evaluation_context_cpu.py`, trained notebook guard |
| Production-readiness report | Claimed CPU-complete contracts but explicitly deferred real pilot, trained physics, and production scale. | historical software claims rechecked where in scope; external items still externally unverified | current full pytest/notebook evidence in completion report; no current CI run inferred |
| Current-head gap audit | Remaining checkpoint, split, evaluation, ontology, notebook, and scientific shortcut gaps. | fixed or explicitly deferred under current tests/configs; real/trained physics remains external | `training/checkpointing.py`, `training/reconstruction_trainer.py`, `evaluation/trained_context.py`, `models/first_level_ablations.py`; current acceptance/evaluation/training tests |
| Current-head verification report | Corrected representation/rollout and documented remaining geometry/runtime/evaluation weaknesses. | superseded; its numerical results remain historical | `docs/post_audit_geometry_and_rollout_review.md`, then this verification |
| Post-audit geometry review | Exact geometry was recomputed; relation forward traversed ancestors in Python; parent negatives/diagnostics were looped; source-conflict scale was query-dependent; reconstruction validation/checkpointing and trained loader remained incomplete. | fixed | `data/heterogeneous.py`, `data/level_collate.py`, `losses/hyperbolic_pretraining.py`, `models/relations.py`, `models/mother_pointer.py`, `training/reconstruction_trainer.py`, `evaluation/trained_context.py`; `tests/test_current_head_acceptance_cpu.py`, `test_real_training_pipeline_cpu.py`, `test_trained_evaluation_context_cpu.py` |
| Retained-tree definition | Defines reduced vocabulary, contraction, truth-only/unmatched provenance, K_L/KLM limitation, denominator and root semantics. | retained; Upsilon(4S)-conditioned B_s rejection fixed; raw KLM/K_L and contracted-PDG frequencies externally unverified | `preprocessing/pid_filter.py`, `reconstruction/constraints.py`; `tests/test_current_head_acceptance_cpu.py`; real-pilot notebook reports unavailable fields honestly |
| Post-audit completion | Reported prior test/notebook state and open real/trained/scale boundaries. | superseded numerically; external boundaries remain open | current completion report only; historical numbers were not rewritten |

## Current fixed-item proof map

| Fixed item | Source | Focused test/evidence |
|---|---|---|
| One exact-geometry build per event and explicit reuse | `data/heterogeneous.py`, `data/level_collate.py`, `training/pretrain_trainer.py` | `test_current_head_acceptance_cpu.py::test_exact_geometry_is_built_once_per_event_during_collation_and_reused` |
| Precomputed/fallback target and gradient equivalence | `losses/hyperbolic_pretraining.py` | `test_precomputed_and_cpu_fallback_targets_and_gradients_match` |
| No normal attention ancestor traversal or Python scalar extraction | `models/relations.py`, `models/heterogeneous.py` | `test_normal_relation_and_parent_paths_need_no_python_scalar_conversion` |
| Query-count-invariant source-conflict normalization | `models/mother_pointer.py`, `losses/level_reconstruction.py` | source-conflict tests in `test_current_head_acceptance_cpu.py` and `test_pointer_validity_constraints_cpu.py` |
| Periodic reconstruction validation and exact best-state resume | `training/reconstruction_trainer.py`, `training/checkpointing.py` | `test_real_training_pipeline_cpu.py::test_real_parquet_train_transfer_validate_and_resume` |
| Evaluation normalization and held-out split contract | `evaluation/trained_context.py` | `test_trained_evaluation_context_cpu.py` |
| Radius/FSP/level/channel-memory ablations | `losses/hyperbolic_pretraining.py`, `models/heterogeneous.py`, `configs/ablations/` | `test_radius_fsp_pooling_and_upsilon_initial_state_ablations`; exact-geometry and hyperbolic notebooks |
| Upsilon(4S) B_s rejection | `reconstruction/constraints.py` | `test_radius_fsp_pooling_and_upsilon_initial_state_ablations` |
| Overlap-aware momentum-dot availability | `models/relations.py`, `preprocessing/schema_v4.py` | `tests/test_post_audit_geometry_rollout_cpu.py` plus current full suite |
| Stable 15-group CI contract and artifacts | `.github/workflows/`, `scripts/execute_notebook_smoke_tests.py` | runner `--list` and final all-group execution report |
