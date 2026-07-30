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


ABLATIONS: dict[str, AblationConfig] = {
    "flat_baseline": AblationConfig("flat_baseline", False, False, False, False, False, False),
    "heterogeneous_only": AblationConfig("heterogeneous_only", True, False, False, False, False, False),
    "hyperbolic_lca_parent": AblationConfig("hyperbolic_lca_parent", True, True, False, False, False, False),
    "plus_radius_depth": AblationConfig("plus_radius_depth", True, True, True, False, False, False),
    "plus_variance_covariance": AblationConfig("plus_variance_covariance", True, True, True, True, False, False),
    "plus_channel": AblationConfig("plus_channel", True, True, True, True, True, False),
    "plus_relation_attention": AblationConfig("plus_relation_attention", True, True, True, True, True, True),
    "full_revised": AblationConfig("full_revised", True, True, True, True, True, True),
}


def build_ablation_model(
    name: str,
    *,
    n_features: int,
    n_types: int,
    hidden_dim: int = 32,
    hyper_dim: int = 16,
    n_queries: int = 8,
) -> LevelAutoregressiveReconstructor:
    config = ABLATIONS[name]
    return LevelAutoregressiveReconstructor(
        n_features=n_features,
        n_types=n_types,
        hidden_dim=hidden_dim,
        hyper_dim=hyper_dim,
        n_queries=n_queries,
        n_heads=4,
        n_context_layers=1,
        encoder_mode="heterogeneous" if config.heterogeneous_adapters else "flat",
        use_relation_bias=config.relation_attention,
    )


__all__ = ["ABLATIONS", "AblationConfig", "build_ablation_model"]
