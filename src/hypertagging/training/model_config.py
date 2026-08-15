"""Serializable model architecture presets for reconstruction production."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelArchitecture:
    preset: str
    d_model: int
    hyper_dim: int
    n_heads: int
    n_context_layers: int
    ffn_dim: int
    dropout: float
    curvature: float
    n_queries: int
    n_queries_by_level: tuple[tuple[int, int], ...] = ()
    max_cardinality: int = 6
    max_cardinality_by_level: tuple[tuple[int, int], ...] = ()
    tangent_variance_target: float = 0.1
    hyper_projection_init_scale: float = 0.05
    tangent_scale_mode: str = "fixed"
    max_tangent_norm: float | None = None
    hyperbolic_scale_contract_version: str = "dimension-aware-tangent-radius-v2"
    tree_geometry_contract_version: str = "retained-tree-exact-edges-v2"
    tree_distance_contract_version: str = "exact-edge-log-fixed-scale-v2"
    relation_feature_contract_version: str = "physical-relations-overlap-aware-v3"
    hyperbolic_level_encoding: str = "learned_euclidean"
    type_conditioned_daughter_relation_bias: bool = False
    capacity_report_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelArchitecture":
        payload = dict(values)
        for key in ("n_queries_by_level", "max_cardinality_by_level"):
            payload[key] = tuple(
                (int(level), int(value)) for level, value in payload.get(key, ())
            )
        return cls(**payload)


MODEL_PRESETS: dict[str, ModelArchitecture] = {
    "tiny_cpu": ModelArchitecture(
        "tiny_cpu", 32, 8, 4, 2, 64, 0.0, 1.0, 8,
        tangent_variance_target=0.12,
        hyper_projection_init_scale=0.08,
    ),
    "gpu_debug": ModelArchitecture(
        "gpu_debug", 64, 16, 4, 3, 192, 0.05, 1.0, 12,
        max_cardinality=8,
        tangent_variance_target=0.08,
        hyper_projection_init_scale=0.065,
    ),
    "small_candidate": ModelArchitecture(
        "small_candidate", 128, 32, 4, 4, 512, 0.1, 1.0, 32,
        # Full train_035k+validation complete-only scan (2026-08-12) proves an
        # exact observed maximum of 16; 12 overflowed 106 eligible mothers.
        max_cardinality=16,
        tangent_variance_target=0.06,
        hyper_projection_init_scale=0.055,
        capacity_report_required=True,
    ),
    "production_baseline": ModelArchitecture(
        "production_baseline", 256, 64, 8, 6, 1024, 0.1, 1.0, 32,
        n_queries_by_level=((1, 32), (2, 24), (3, 16)),
        max_cardinality=12,
        max_cardinality_by_level=((1, 8), (2, 10), (3, 12)),
        tangent_variance_target=0.05,
        hyper_projection_init_scale=0.05,
    ),
}


def resolve_model_architecture(preset: str, **overrides: Any) -> ModelArchitecture:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"unknown model preset: {preset}")
    values = MODEL_PRESETS[preset].to_dict()
    values.update({key: value for key, value in overrides.items() if value is not None})
    values["preset"] = preset
    return ModelArchitecture.from_dict(values)


__all__ = ["MODEL_PRESETS", "ModelArchitecture", "resolve_model_architecture"]
