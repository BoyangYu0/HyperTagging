"""Level-autoregressive reconstruction batch containers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LevelEvent:
    """One levelized reconstruction event before padding."""

    event_id: int
    node_features: torch.Tensor
    p4: torch.Tensor
    charge: torch.Tensor
    pid_labels: torch.Tensor
    level_ids: torch.Tensor
    parent_ids: torch.Tensor
    daughter_adjacency: torch.Tensor
    active: torch.Tensor
    copied: torch.Tensor
    copied_from: torch.Tensor
    raw_pdg: torch.Tensor | None = None
    node_kind_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class LevelBatch:
    """Padded batch for level-autoregressive set reconstruction."""

    node_features: torch.Tensor
    node_mask: torch.Tensor
    level_ids: torch.Tensor
    pid_labels: torch.Tensor
    p4: torch.Tensor
    charge: torch.Tensor
    daughter_adjacency: torch.Tensor
    parent_ids: torch.Tensor
    active: torch.Tensor
    copied: torch.Tensor
    copied_from: torch.Tensor
    same_mother: torch.Tensor
    same_branch: torch.Tensor
    lca_depth: torch.Tensor
    lca_node_id: torch.Tensor
    edges_to_lca_from_i: torch.Tensor
    edges_to_lca_from_j: torch.Tensor
    exact_tree_path_distance: torch.Tensor
    depth_from_retained_root: torch.Tensor
    distance_to_nearest_retained_root: torch.Tensor
    query_mask: torch.Tensor
    event_ids: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return the batch as a model-friendly dictionary."""

        return {
            "node_features": self.node_features,
            "node_mask": self.node_mask,
            "level_ids": self.level_ids,
            "pid_labels": self.pid_labels,
            "p4": self.p4,
            "charge": self.charge,
            "daughter_adjacency": self.daughter_adjacency,
            "parent_ids": self.parent_ids,
            "active": self.active,
            "copied": self.copied,
            "copied_from": self.copied_from,
            "same_mother": self.same_mother,
            "same_branch": self.same_branch,
            "lca_depth": self.lca_depth,
            "lca_node_id": self.lca_node_id,
            "edges_to_lca_from_i": self.edges_to_lca_from_i,
            "edges_to_lca_from_j": self.edges_to_lca_from_j,
            "exact_tree_path_distance": self.exact_tree_path_distance,
            "depth_from_retained_root": self.depth_from_retained_root,
            "distance_to_nearest_retained_root": self.distance_to_nearest_retained_root,
            "query_mask": self.query_mask,
            "event_ids": self.event_ids,
        }
