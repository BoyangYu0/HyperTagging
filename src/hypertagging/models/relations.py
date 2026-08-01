"""Relation-aware attention bias features."""

from __future__ import annotations

import torch
from torch import nn

from hypertagging.models.hyperbolic import (
    hyperbolic_pairwise_distance,
    logmap0,
    radius,
)
from hypertagging.preprocessing.schema_v2 import NODE_KINDS


PHYSICAL_RELATION_SCALING_VERSION = "physical-relations-overlap-aware-v3"
PHYSICAL_RELATION_FEATURE_NAMES = (
    "directed_level_difference",
    "same_level",
    "charge_sum",
    "disjoint_pair_mass",
    "disjoint_pair_energy",
    "momentum_dot",
    "momentum_dot_available",
    "same_node_kind",
    "recursive_source_overlap",
    "ancestor_descendant_relation",
    "disjoint_source_pair",
    "same_reco_source",
    "copied_source_conflict",
    "pair_mass_energy_available",
)

# Versioned provenance categories.  Only the first two may enter contextual
# attention, and the second must be built from nodes that already exist in the
# current reconstruction state.  The third category remains loss/diagnostic
# supervision only.
RELATION_FEATURE_PROVENANCE = {
    "inference_physical_relation_features": (
        "p4",
        "charge",
        "level_ids",
        "node_kind_ids",
        "reco_ids",
    ),
    "current_reconstructed_tree_state_features": (
        "copied",
        "source_node_ids",
        "recursive_leaf_source_mask",
        "current_reconstructed_ancestor_descendant_relation",
    ),
    "truth_target_only_relation_features": (
        "parent_ids",
        "ancestor_descendant_relation",
        "lca_node_id",
        "exact_tree_path_distance",
        "b_side",
    ),
}


def _signed_log_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Compress dimensional physical values under an explicit fixed contract."""

    return torch.sign(value) * torch.log1p(value.abs() / float(scale))


def _physical_features(
    *,
    p4: torch.Tensor,
    charge: torch.Tensor,
    level_ids: torch.Tensor,
    node_kind_ids: torch.Tensor | None,
    copied: torch.Tensor | None,
    source_node_ids: torch.Tensor | None,
    recursive_leaf_source_mask: torch.Tensor | None = None,
    parent_ids: torch.Tensor | None = None,
    ancestor_descendant_relation: torch.Tensor | None = None,
    reco_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    level_diff = (level_ids[:, :, None] - level_ids[:, None, :]).float()
    same_level = (level_ids[:, :, None] == level_ids[:, None, :]).float()
    charge_sum = charge[:, :, None] + charge[:, None, :]
    pair_p4 = p4[:, :, None, :] + p4[:, None, :, :]
    mass2 = pair_p4[..., 3] ** 2 - (pair_p4[..., :3] ** 2).sum(dim=-1)
    mass = torch.sqrt(torch.clamp(mass2, min=0.0))
    momentum_dot = torch.einsum("bif,bjf->bij", p4[..., :3], p4[..., :3])
    same_kind = (
        torch.zeros_like(level_diff)
        if node_kind_ids is None
        else (node_kind_ids[:, :, None] == node_kind_ids[:, None, :]).float()
    )
    copied_conflict = torch.zeros_like(level_diff)
    if copied is not None and source_node_ids is not None:
        same_source = source_node_ids[:, :, None] == source_node_ids[:, None, :]
        copied_conflict = (same_source & (copied[:, :, None] | copied[:, None, :])).float()
    recursive_overlap = torch.zeros_like(level_diff, dtype=torch.bool)
    source_available = torch.zeros_like(level_ids, dtype=torch.bool)
    if recursive_leaf_source_mask is not None:
        source_available = recursive_leaf_source_mask.any(dim=-1)
        recursive_overlap = torch.einsum(
            "bns,bms->bnm",
            recursive_leaf_source_mask.to(torch.int32),
            recursive_leaf_source_mask.to(torch.int32),
        ) > 0
    source_pair_available = source_available[:, :, None] & source_available[:, None, :]
    disjoint_source_pair = source_pair_available & ~recursive_overlap
    ancestor_descendant = (
        ancestor_descendant_relation.bool()
        if ancestor_descendant_relation is not None
        else torch.zeros_like(recursive_overlap)
    )
    if ancestor_descendant_relation is None and parent_ids is not None:
        if parent_ids.is_cuda:
            raise RuntimeError(
                "ancestor_descendant_relation must be precomputed during CPU "
                "collation before relation attention on CUDA"
            )
        for batch_index in range(parent_ids.shape[0]):
            for node in range(parent_ids.shape[1]):
                seen: set[int] = set()
                parent = int(parent_ids[batch_index, node])
                while 0 <= parent < parent_ids.shape[1] and parent not in seen:
                    ancestor_descendant[batch_index, node, parent] = True
                    ancestor_descendant[batch_index, parent, node] = True
                    seen.add(parent)
                    parent = int(parent_ids[batch_index, parent])
    same_reco_source = torch.zeros_like(recursive_overlap)
    if reco_ids is not None:
        same_reco_source = (
            (reco_ids[:, :, None] >= 0)
            & (reco_ids[:, None, :] >= 0)
            & (reco_ids[:, :, None] == reco_ids[:, None, :])
        )
    physical_mass = torch.where(disjoint_source_pair, mass, torch.zeros_like(mass))
    physical_energy = torch.where(
        disjoint_source_pair, pair_p4[..., 3], torch.zeros_like(pair_p4[..., 3])
    )
    physical_momentum_dot = torch.where(
        disjoint_source_pair, momentum_dot, torch.zeros_like(momentum_dot)
    )
    return torch.stack(
        [
            level_diff / 8.0,
            same_level,
            charge_sum / 4.0,
            _signed_log_scale(physical_mass, 1.0),
            _signed_log_scale(physical_energy, 1.0),
            _signed_log_scale(physical_momentum_dot, 1.0),
            disjoint_source_pair.float(),
            same_kind,
            recursive_overlap.float(),
            ancestor_descendant.float(),
            disjoint_source_pair.float(),
            same_reco_source.float(),
            copied_conflict,
            disjoint_source_pair.float(),
        ],
        dim=-1,
    )


class PhysicalRelationBias(nn.Module):
    """Stage-A relation bias using only data-compatible physical inputs.

    Continuous GeV-valued features follow
    ``physical-relations-overlap-aware-v3``: level and charge use fixed
    divisors; mass and energy are exposed only for disjoint recursive-source
    pairs and carry an explicit availability flag. Binary overlap/provenance
    indicators stay in {0,1}. Node kinds use a collision-free symmetric pair
    embedding rather than a summed scalar code.
    """

    def __init__(self, hidden_dim: int = 32, *, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.scaling_version = PHYSICAL_RELATION_SCALING_VERSION
        pair_dim = max(4, hidden_dim // 4)
        self.pair_kind_embedding = nn.Embedding(len(NODE_KINDS) ** 2, pair_dim)
        self.net = nn.Sequential(
            nn.Linear(len(PHYSICAL_RELATION_FEATURE_NAMES) + pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        p4: torch.Tensor,
        charge: torch.Tensor,
        level_ids: torch.Tensor,
        node_mask: torch.Tensor,
        node_kind_ids: torch.Tensor | None = None,
        copied: torch.Tensor | None = None,
        source_node_ids: torch.Tensor | None = None,
        recursive_leaf_source_mask: torch.Tensor | None = None,
        parent_ids: torch.Tensor | None = None,
        ancestor_descendant_relation: torch.Tensor | None = None,
        reco_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = _physical_features(
            p4=p4,
            charge=charge,
            level_ids=level_ids,
            node_kind_ids=node_kind_ids,
            copied=copied,
            source_node_ids=source_node_ids,
            recursive_leaf_source_mask=recursive_leaf_source_mask,
            parent_ids=parent_ids,
            ancestor_descendant_relation=ancestor_descendant_relation,
            reco_ids=reco_ids,
        )
        if node_kind_ids is None:
            pair_embedding = features.new_zeros((*features.shape[:-1], self.pair_kind_embedding.embedding_dim))
        else:
            left = node_kind_ids[:, :, None].clamp(0, len(NODE_KINDS) - 1)
            right = node_kind_ids[:, None, :].clamp(0, len(NODE_KINDS) - 1)
            lower = torch.minimum(left, right)
            upper = torch.maximum(left, right)
            pair_embedding = self.pair_kind_embedding(lower * len(NODE_KINDS) + upper)
        combined = torch.cat([features, pair_embedding], dim=-1)
        bias = self.net(combined).squeeze(-1) if self.enabled else features[..., 0] * 0.0
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        return bias.masked_fill(~valid, -1e4)


class HyperbolicRelationBias(nn.Module):
    """Optional Stage-B bias after contextual hyperbolic projection."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        enabled: bool = True,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.curvature = curvature
        self.net = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, *, z_hyperbolic: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        d_h = hyperbolic_pairwise_distance(
            z_hyperbolic,
            node_mask,
            curvature=self.curvature,
        )
        radii = radius(z_hyperbolic, curvature=self.curvature)
        tangent = logmap0(z_hyperbolic, curvature=self.curvature)
        tangent_dot = torch.einsum("bif,bjf->bij", tangent, tangent)
        features = torch.stack(
            [
                torch.log1p(d_h),
                torch.log1p(radii[:, :, None].expand_as(d_h)),
                torch.log1p(radii[:, None, :].expand_as(d_h)),
                _signed_log_scale(tangent_dot, 1.0),
            ],
            dim=-1,
        )
        bias = self.net(features).squeeze(-1) if self.enabled else d_h * 0.0
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        return bias.masked_fill(~valid, -1e4)


class RelationBias(nn.Module):
    """Learn a pairwise attention bias from level, charge, p4, and hyperbolic distance."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        enabled: bool = True,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.curvature = curvature
        self.physical = PhysicalRelationBias(hidden_dim, enabled=enabled)
        self.hyperbolic = HyperbolicRelationBias(
            hidden_dim,
            enabled=enabled,
            curvature=curvature,
        )

    def forward(
        self,
        *,
        p4: torch.Tensor,
        charge: torch.Tensor,
        level_ids: torch.Tensor,
        z_hyperbolic: torch.Tensor,
        node_mask: torch.Tensor,
        node_kind_ids: torch.Tensor | None = None,
        copied: torch.Tensor | None = None,
        source_node_ids: torch.Tensor | None = None,
        recursive_leaf_source_mask: torch.Tensor | None = None,
        parent_ids: torch.Tensor | None = None,
        ancestor_descendant_relation: torch.Tensor | None = None,
        reco_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.physical(
            p4=p4,
            charge=charge,
            level_ids=level_ids,
            node_mask=node_mask,
            node_kind_ids=node_kind_ids,
            copied=copied,
            source_node_ids=source_node_ids,
            recursive_leaf_source_mask=recursive_leaf_source_mask,
            parent_ids=parent_ids,
            ancestor_descendant_relation=ancestor_descendant_relation,
            reco_ids=reco_ids,
        ) + self.hyperbolic(z_hyperbolic=z_hyperbolic, node_mask=node_mask)


__all__ = [
    "HyperbolicRelationBias",
    "PHYSICAL_RELATION_SCALING_VERSION",
    "PHYSICAL_RELATION_FEATURE_NAMES",
    "PhysicalRelationBias",
    "RELATION_FEATURE_PROVENANCE",
    "RelationBias",
]
