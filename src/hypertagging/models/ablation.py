"""Named CPU-testable ablations for the revised architecture."""

from __future__ import annotations

from dataclasses import dataclass

from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor


@dataclass(frozen=True)
class AblationConfig:
    name: str
    heterogeneous_adapters: bool
    lca_parent: bool
    radius_depth: bool
    variance_covariance: bool
    channel_supervision: bool
    relation_attention: bool
    contextual_euclidean: bool = False
    hyperbolic_relation_attention: bool = False
    leaf_pid: bool = False
    scheduled_sampling: bool = False
    pretrained_encoder_transfer: bool = False
    canonical_pion_first_level: bool = False


ABLATIONS: dict[str, AblationConfig] = {
    "flat_baseline": AblationConfig("flat_baseline", False, False, False, False, False, False),
    "heterogeneous_only": AblationConfig("heterogeneous_only", True, False, False, False, False, False),
    "contextual_euclidean": AblationConfig(
        "contextual_euclidean", True, False, False, False, False, False, True
    ),
    "contextual_hyperbolic_parent_lca": AblationConfig(
        "contextual_hyperbolic_parent_lca", True, True, False, False, False, False, True
    ),
    "plus_radius_depth": AblationConfig(
        "plus_radius_depth", True, True, True, False, False, False, True
    ),
    "plus_variance_covariance": AblationConfig(
        "plus_variance_covariance", True, True, True, True, False, False, True
    ),
    "plus_cross_event_channel": AblationConfig(
        "plus_cross_event_channel", True, True, True, True, True, False, True
    ),
    "plus_hyperbolic_relation_attention": AblationConfig(
        "plus_hyperbolic_relation_attention",
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ),
    "plus_leaf_pid": AblationConfig(
        "plus_leaf_pid", True, True, True, True, True, True, True, True, True
    ),
    "plus_scheduled_sampling": AblationConfig(
        "plus_scheduled_sampling",
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ),
    "full_revised": AblationConfig(
        "full_revised",
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ),
}

# Orthogonal diagnostic switches are kept outside the monotonic scientific
# ablation ladder so adding one does not change the ladder's stable ordering.
EXPERIMENT_VARIANTS: dict[str, AblationConfig] = {
    "canonical_pion_first_level": AblationConfig(
        "canonical_pion_first_level",
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
    ),
}
ALL_ABLATIONS: dict[str, AblationConfig] = ABLATIONS | EXPERIMENT_VARIANTS


def build_ablation_model(
    name: str,
    *,
    n_features: int,
    n_types: int,
    hidden_dim: int = 32,
    hyper_dim: int = 16,
    n_queries: int = 8,
    max_cardinality: int = 6,
    n_heads: int = 4,
    n_context_layers: int = 2,
    curvature: float = 1.0,
    ffn_dim: int | None = None,
    dropout: float = 0.0,
    n_queries_by_level: tuple[tuple[int, int], ...] = (),
    max_cardinality_by_level: tuple[tuple[int, int], ...] = (),
) -> LevelAutoregressiveReconstructor:
    config = ALL_ABLATIONS[name]
    return LevelAutoregressiveReconstructor(
        n_features=n_features,
        n_types=n_types,
        hidden_dim=hidden_dim,
        hyper_dim=hyper_dim,
        n_queries=n_queries,
        n_heads=n_heads,
        n_context_layers=n_context_layers,
        curvature=curvature,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_cardinality=max_cardinality,
        n_queries_by_level=n_queries_by_level,
        max_cardinality_by_level=max_cardinality_by_level,
        encoder_mode="heterogeneous" if config.heterogeneous_adapters else "flat",
        use_contextual_encoder=config.contextual_euclidean,
        use_relation_bias=config.relation_attention,
        use_hyperbolic_relation_refinement=config.hyperbolic_relation_attention,
        canonical_pion_first_level=config.canonical_pion_first_level,
    )


__all__ = [
    "ABLATIONS",
    "EXPERIMENT_VARIANTS",
    "ALL_ABLATIONS",
    "AblationConfig",
    "build_ablation_model",
]
