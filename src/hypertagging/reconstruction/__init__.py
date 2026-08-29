"""Reconstruction helpers exposed lazily to avoid model/loss import cycles."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULE = {
    **{
        name: "hypertagging.reconstruction.full_reconstruction"
        for name in (
            "FullReconstructionResult",
            "PredictionStep",
            "aggregate_features_by_link",
            "build_pred_lca",
            "build_pred_lca_from_pairs",
            "evaluate_event",
            "prediction_step",
            "recover",
            "remap_links_around_empty",
        )
    },
    **{
        name: "hypertagging.reconstruction.single_level"
        for name in (
            "ReconstructionVariant",
            "SingleLevelReconstruction",
            "build_goal_batch",
            "build_reconstructed_batch",
            "build_reconstructed_link_batch",
            "reconstructed_embedding_distance",
            "single_level_reconstruction_step",
            "sort_energy",
        )
    },
    **{
        name: "hypertagging.reconstruction.level_rollout"
        for name in (
            "CompositeProposal",
            "BeamRolloutHypothesis",
            "LevelRolloutResult",
            "RolloutConfig",
            "append_composite_proposals",
            "level_rollout",
            "proposal_ambiguity_metrics",
            "rollout_search_metrics",
            "resolver_difference_rate",
            "resolve_exclusive_proposals",
            "resolve_weighted_set_packing",
            "bounded_beam_proposal_sets",
            "bounded_beam_rollout",
        )
    },
    **{
        name: "hypertagging.reconstruction.hierarchical_inference"
        for name in (
            "DEFAULT_HALF_ROOT_TOKENS",
            "DETECTOR_FSP_KIND_IDS",
            "FULL_ROOT_TOKEN",
            "FSPInputAudit",
            "FSPProjection",
            "HierarchicalInferenceConfig",
            "HierarchicalInferenceResult",
            "InferenceScope",
            "OFFLINE_INFERENCE_POLICY_VERSION",
            "project_preprocessed_mdst_fsps",
            "project_schema_v4_fsps",
            "reconstruct_full_tree_from_fsps",
        )
    },
}

__all__ = sorted(_EXPORT_MODULE)


def __getattr__(name: str):
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
