"""Exact retained-forest geometry derived only from parent links.

Reconstruction levels are tree heights used by autoregressive generation.  They
are deliberately absent from this module: an edge is always one edge, including
in unbalanced retained trees.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


EXACT_TREE_GEOMETRY_CONTRACT_VERSION = "retained-tree-exact-edges-v2"


@dataclass(frozen=True)
class ExactTreeGeometry:
    lca_node_id: torch.Tensor
    edges_to_lca_from_i: torch.Tensor
    edges_to_lca_from_j: torch.Tensor
    exact_tree_path_distance: torch.Tensor
    depth_from_retained_root: torch.Tensor
    distance_to_nearest_retained_root: torch.Tensor


def build_exact_tree_geometry(parent_ids: torch.Tensor) -> ExactTreeGeometry:
    """Return exact edge geometry for a retained rooted forest.

    ``parent_ids`` contains positions, with ``-1`` denoting a retained root.
    Pairs in different retained components have LCA and pair distances ``-1``.
    Invalid parent positions and cycles are rejected rather than guessed.
    """

    if parent_ids.ndim != 1:
        raise ValueError("parent_ids must be one-dimensional")
    count = int(parent_ids.numel())
    device = parent_ids.device
    paths: list[list[int]] = []
    for node in range(count):
        path = [node]
        seen = {node}
        parent = int(parent_ids[node])
        while parent >= 0:
            if parent >= count:
                raise ValueError(f"parent position {parent} is outside a {count}-node forest")
            if parent in seen:
                raise ValueError(f"cycle detected while building exact geometry for node {node}")
            path.append(parent)
            seen.add(parent)
            parent = int(parent_ids[parent])
        paths.append(path)

    lca = torch.full((count, count), -1, dtype=torch.long, device=device)
    from_i = torch.full_like(lca, -1)
    from_j = torch.full_like(lca, -1)
    distance = torch.full_like(lca, -1)
    for left in range(count):
        right_positions = {node: edge_count for edge_count, node in enumerate(paths[left])}
        for right in range(count):
            common = next(
                (
                    (node, right_positions[node], right_edge_count)
                    for right_edge_count, node in enumerate(paths[right])
                    if node in right_positions
                ),
                None,
            )
            if common is None:
                continue
            node, left_edges, right_edges = common
            lca[left, right] = node
            from_i[left, right] = left_edges
            from_j[left, right] = right_edges
            distance[left, right] = left_edges + right_edges

    depth = torch.tensor(
        [len(path) - 1 for path in paths], dtype=torch.long, device=device
    )
    return ExactTreeGeometry(
        lca_node_id=lca,
        edges_to_lca_from_i=from_i,
        edges_to_lca_from_j=from_j,
        exact_tree_path_distance=distance,
        depth_from_retained_root=depth,
        distance_to_nearest_retained_root=depth.clone(),
    )


__all__ = [
    "EXACT_TREE_GEOMETRY_CONTRACT_VERSION",
    "ExactTreeGeometry",
    "build_exact_tree_geometry",
]
