"""Serializable reconstruction constraints shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from hypertagging.preprocessing.pid_filter import (
    MOTHER_ONTOLOGY_VERSION,
    PDG_TOKENS,
    STATIC_MOTHER_TOKENS,
)
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID


_UNIT_CHARGE_PDGS = {
    -11: 1.0, 321: 1.0, 211: 1.0, -13: 1.0, 2212: 1.0,
    3222: 1.0, 411: 1.0, 431: 1.0, 4122: 1.0, 413: 1.0,
    521: 1.0,
    11: -1.0, -321: -1.0, -211: -1.0, 13: -1.0, -2212: -1.0,
    -3222: -1.0, -411: -1.0, -431: -1.0, -4122: -1.0,
    -413: -1.0, -521: -1.0,
}
REDUCED_TOKEN_CHARGE: tuple[float, ...] = tuple(
    _UNIT_CHARGE_PDGS.get(pdg, 0.0) for pdg in PDG_TOKENS
)


@dataclass(frozen=True)
class ReconstructionConstraintPolicy:
    """One policy object for teacher contexts, rollout, validation and export."""

    version: str = "reconstruction-constraints-v2"
    mother_ontology_version: str = MOTHER_ONTOLOGY_VERSION
    static_allowed_mother_tokens: tuple[int, ...] = STATIC_MOTHER_TOKENS
    allowed_mother_types_by_level: tuple[tuple[int, tuple[int, ...]], ...] = ()
    empirical_type_prior_mode: str = "soft"  # hard | soft | off
    empirical_type_soft_penalty: float = 2.0
    valid_leaf_node_kinds: tuple[int, ...] = (
        NODE_KIND_TO_ID["track"],
        NODE_KIND_TO_ID["ecl_cluster"],
        NODE_KIND_TO_ID["other"],
    )
    valid_composite_node_kinds: tuple[int, ...] = (NODE_KIND_TO_ID["composite"],)
    require_lower_level_context: bool = True
    reject_recursive_source_conflicts: bool = True
    minimum_pointer_probability: float = 0.5
    daughter_cardinality_policy: str = "predicted"  # predicted | threshold
    minimum_daughters: int = 2
    cardinality_insufficient_policy: str = "invalid"
    mother_charge_compatibility: str = "soft_train_hard_rollout"  # off | soft | hard | soft_train_hard_rollout
    mother_charge_tolerance: float = 1e-6
    mother_charge_soft_weight: float = 0.1
    allow_fixed_hypothesis_unknown_kind: bool = True
    loose_physical_constraints: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.mother_ontology_version != MOTHER_ONTOLOGY_VERSION:
            raise ValueError(
                "unsupported static mother ontology version: "
                f"{self.mother_ontology_version}"
            )
        static_tokens = set(self.static_allowed_mother_tokens)
        if 0 in static_tokens:
            raise ValueError("unknown token 0 cannot be a reconstructed mother")
        if not static_tokens or min(static_tokens) < 0 or max(static_tokens) >= len(PDG_TOKENS):
            raise ValueError("static mother token mask is invalid")
        if self.empirical_type_prior_mode not in {"hard", "soft", "off"}:
            raise ValueError("empirical_type_prior_mode must be hard, soft, or off")
        if self.daughter_cardinality_policy not in {"predicted", "threshold"}:
            raise ValueError("invalid daughter_cardinality_policy")
        if self.cardinality_insufficient_policy not in {"invalid", "reduce"}:
            raise ValueError("invalid cardinality_insufficient_policy")
        if self.mother_charge_compatibility not in {
            "off", "soft", "hard", "soft_train_hard_rollout"
        }:
            raise ValueError("invalid mother_charge_compatibility")
        supported_physical = {
            "minimum_mother_energy", "minimum_mother_mass",
            "maximum_mother_mass", "maximum_mother_momentum",
        }
        unknown_physical = {
            name for name, _value in self.loose_physical_constraints
        } - supported_physical
        if unknown_physical:
            raise ValueError(
                f"unknown loose physical constraint(s): {sorted(unknown_physical)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReconstructionConstraintPolicy":
        values = dict(payload)
        for name in (
            "allowed_mother_types_by_level", "static_allowed_mother_tokens",
            "valid_leaf_node_kinds",
            "valid_composite_node_kinds", "loose_physical_constraints",
        ):
            if name in values:
                if name == "allowed_mother_types_by_level":
                    values[name] = tuple(
                        (int(level), tuple(int(token) for token in tokens))
                        for level, tokens in values[name]
                    )
                elif name == "loose_physical_constraints":
                    values[name] = tuple((str(key), float(value)) for key, value in values[name])
                else:
                    values[name] = tuple(int(value) for value in values[name])
        return cls(**values)

    def observed_types(self, level: int) -> tuple[int, ...]:
        return dict(self.allowed_mother_types_by_level).get(int(level), ())

    def type_constraints(
        self, level: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = self.observed_types(level)
        allowed = torch.zeros(len(PDG_TOKENS), dtype=torch.bool, device=device)
        allowed[list(self.static_allowed_mother_tokens)] = True
        bias = torch.zeros(len(PDG_TOKENS), dtype=torch.float32, device=device)
        if observed and self.empirical_type_prior_mode == "hard":
            observed_mask = torch.zeros_like(allowed)
            observed_mask[list(observed)] = True
            allowed &= observed_mask
        elif observed and self.empirical_type_prior_mode == "soft":
            seen = torch.zeros_like(allowed)
            seen[list(observed)] = True
            bias[allowed & ~seen] = -float(self.empirical_type_soft_penalty)
        if not bool(allowed.any()):
            raise ValueError(
                f"mother constraints reject every static ontology type at level {level}"
            )
        return allowed, bias

    def pointer_validity_mask(
        self, batch: Mapping[str, torch.Tensor], target_level: int
    ) -> torch.Tensor:
        valid = batch["node_mask"].bool().clone()
        if self.require_lower_level_context:
            valid &= batch["level_ids"] < int(target_level)
        kinds = batch["node_kind_ids"]
        allowed_kinds = set(self.valid_leaf_node_kinds) | set(self.valid_composite_node_kinds)
        kind_valid = torch.zeros_like(valid)
        for kind in allowed_kinds:
            kind_valid |= kinds == int(kind)
        if self.allow_fixed_hypothesis_unknown_kind and "leaf_kinematics_mode_ids" in batch:
            kind_valid |= (
                batch["leaf_kinematics_mode_ids"]
                == LEAF_MODE_TO_ID["fixed_hypothesis_candidate"]
            )
        return valid & kind_valid

    def expected_charge(self, token: int) -> float:
        return REDUCED_TOKEN_CHARGE[int(token)]

    def rollout_physical_valid(self, mother_p4: torch.Tensor) -> bool:
        """Apply optional deliberately loose hard rollout checks."""

        configured = dict(self.loose_physical_constraints)
        if not configured:
            return True
        momentum = mother_p4[:3].square().sum().sqrt()
        mass = (mother_p4[3].square() - momentum.square()).clamp_min(0).sqrt()
        checks = {
            "minimum_mother_energy": mother_p4[3] >= configured.get(
                "minimum_mother_energy", float("-inf")
            ),
            "minimum_mother_mass": mass >= configured.get(
                "minimum_mother_mass", float("-inf")
            ),
            "maximum_mother_mass": mass <= configured.get(
                "maximum_mother_mass", float("inf")
            ),
            "maximum_mother_momentum": momentum <= configured.get(
                "maximum_mother_momentum", float("inf")
            ),
        }
        return all(bool(checks[name]) for name in configured)


__all__ = ["REDUCED_TOKEN_CHARGE", "ReconstructionConstraintPolicy"]
