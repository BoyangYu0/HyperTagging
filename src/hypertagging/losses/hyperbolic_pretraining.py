"""Multiscale hyperbolic topology losses and anti-collapse diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hypertagging.data.tree_geometry import (
    EXACT_TREE_GEOMETRY_CONTRACT_VERSION,
    build_exact_tree_geometry,
)
from hypertagging.models.hyperbolic import distance, logmap0, radius


N_TREE_RELATIONS = 6
TREE_RELATION_NAMES = (
    "same_node",
    "same_immediate_mother",
    "same_local_branch",
    "same_b_branch",
    "different_b_same_event",
    "unrelated_or_unknown",
)
TREE_DISTANCE_CONTRACT_VERSION = "exact-edge-log-fixed-scale-v2"
LEVEL_HEIGHT_DISTANCE_ABLATION_VERSION = "reconstruction-height-distance-v1"
HYPERBOLIC_SCALE_CONTRACT_VERSION = "dimension-aware-tangent-radius-v2"


def dimension_aware_tangent_variance_target(
    hyper_dim: int,
    *,
    target_tangent_norm: float = 0.5,
) -> float:
    """Per-coordinate floor implied by a fixed RMS tangent-vector norm."""

    if hyper_dim <= 0 or target_tangent_norm < 0:
        raise ValueError("hyper_dim must be positive and target_tangent_norm non-negative")
    return float(target_tangent_norm) / hyper_dim**0.5


@dataclass(frozen=True)
class HyperbolicLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


def same_pair_bce(pair_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = mask[:, :, None] & mask[:, None, :]
    if not pair_mask.any():
        return pair_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(pair_logits[pair_mask], labels.float()[pair_mask])


def build_tree_relation_targets(
    *,
    parent_ids: torch.Tensor,
    lca_depth: torch.Tensor,
    level_ids: torch.Tensor,
    node_mask: torch.Tensor,
    b_side: torch.Tensor | None = None,
    lca_node_id: torch.Tensor | None = None,
    edges_to_lca_from_i: torch.Tensor | None = None,
    edges_to_lca_from_j: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the documented six-class LCA relation convention.

    Local branch means a retained common ancestor no more than two exact edges
    from either node. Same-B and different-B relations use explicit B-side
    labels. Pairs without a retained common ancestor or B label are class 5.
    """

    batch_size, n_nodes = parent_ids.shape
    targets = torch.full(
        (batch_size, n_nodes, n_nodes),
        N_TREE_RELATIONS - 1,
        dtype=torch.long,
        device=parent_ids.device,
    )
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    identity = torch.eye(n_nodes, dtype=torch.bool, device=parent_ids.device)[None]
    targets[valid & identity] = 0
    same_parent = (
        (parent_ids[:, :, None] == parent_ids[:, None, :])
        & (parent_ids[:, :, None] >= 0)
        & ~identity
    )
    targets[valid & same_parent] = 1
    if lca_node_id is None or edges_to_lca_from_i is None or edges_to_lca_from_j is None:
        if parent_ids.is_cuda:
            raise RuntimeError(
                "exact tree geometry must be precomputed on CPU before CUDA training; "
                "the fallback builder is only for tiny CPU/legacy callers"
            )
        lca_node_id = torch.full_like(lca_depth, -1)
        edges_to_lca_from_i = torch.full_like(lca_depth, -1)
        edges_to_lca_from_j = torch.full_like(lca_depth, -1)
        for batch_index in range(batch_size):
            geometry = build_exact_tree_geometry(parent_ids[batch_index])
            lca_node_id[batch_index] = geometry.lca_node_id
            edges_to_lca_from_i[batch_index] = geometry.edges_to_lca_from_i
            edges_to_lca_from_j[batch_index] = geometry.edges_to_lca_from_j
    local = (
        (lca_node_id >= 0)
        & (edges_to_lca_from_i <= 2)
        & (edges_to_lca_from_j <= 2)
        & ~same_parent
        & ~identity
    )
    targets[valid & local] = 2
    if b_side is not None:
        valid_side = (b_side[:, :, None] >= 0) & (b_side[:, None, :] >= 0)
        same_side = b_side[:, :, None] == b_side[:, None, :]
        targets[valid & valid_side & same_side & ~local & ~same_parent & ~identity] = 3
        targets[valid & valid_side & ~same_side] = 4
    return targets, valid


def build_topology_safe_parent_negative_mask(
    tree_relation_targets: torch.Tensor,
    node_mask: torch.Tensor,
    ancestor_descendant_relation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized [B, child, candidate] mask for the directed parent loss.

    Explicit relation class 4 (the other B branch) is preferred per child;
    class 5 is used only when class 4 is unavailable. Classes 0--3 contain the
    child/family/near-positive relations and are never negatives.
    """

    valid_pairs = node_mask[:, :, None] & node_mask[:, None, :]
    if ancestor_descendant_relation is not None:
        if ancestor_descendant_relation.shape != valid_pairs.shape:
            raise ValueError("ancestor_descendant_relation must have shape [B, N, N]")
        valid_pairs &= ~ancestor_descendant_relation
    different_b = valid_pairs & (tree_relation_targets == 4)
    unrelated = valid_pairs & (tree_relation_targets == 5)
    has_different_b = different_b.any(dim=-1, keepdim=True)
    return torch.where(has_different_b, different_b, unrelated)


def balanced_tree_relation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Give each present relation class equal weight."""

    losses: list[torch.Tensor] = []
    for relation in range(logits.shape[-1]):
        selected = pair_mask & (targets == relation)
        if selected.any():
            losses.append(F.cross_entropy(logits[selected], targets[selected]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def parent_child_margin_loss(
    z: torch.Tensor,
    parent_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.2,
    hard_negative: bool = True,
    curvature: float = 1.0,
    lca_depth: torch.Tensor | None = None,
    tree_relation_targets: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
    parent_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directed parent ranking against explicitly topology-safe negatives.

    This is distinct from the undirected tree-distance geometry objective.  A
    negative may not be the child, its parent, any ancestor/descendant,
    immediate sibling, or an LCA relation class declared positive/near-positive
    (classes 0--3). Different-B nodes are preferred, followed by unrelated
    retained roots / explicit negative LCA classes.
    """

    if parent_negative_mask is not None:
        if parent_negative_mask.shape != (*parent_ids.shape, parent_ids.shape[1]):
            raise ValueError("parent_negative_mask must have shape [B, N, N]")
        safe_parent = parent_ids.clamp_min(0)
        parent_valid = (parent_ids >= 0) & mask
        parent_valid &= mask.gather(1, safe_parent)
        eligible = parent_negative_mask & mask[:, :, None] & mask[:, None, :]
        active = parent_valid & eligible.any(dim=-1)
        parent_z = z.gather(
            1, safe_parent.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        )
        positive = distance(z, parent_z, curvature=curvature)
        all_negative = distance(
            z[:, :, None, :], z[:, None, :, :], curvature=curvature
        )
        if hard_negative:
            negative = all_negative.masked_fill(~eligible, float("inf")).amin(dim=-1)
        else:
            # A deterministic vectorized fallback: the first eligible candidate.
            first = eligible.to(torch.int64).argmax(dim=-1)
            negative = all_negative.gather(2, first.unsqueeze(-1)).squeeze(-1)
        ranked = F.relu(positive - negative + margin)
        weights = active.to(ranked.dtype)
        return (ranked * weights).sum() / weights.sum().clamp_min(1.0)

    if parent_ids.is_cuda:
        raise RuntimeError(
            "parent_negative_mask is required on CUDA; Python topology traversal "
            "is restricted to tiny CPU/legacy callers"
        )
    losses: list[torch.Tensor] = []
    for batch_index in range(z.shape[0]):
        valid = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        for child in valid.tolist():
            parent = int(parent_ids[batch_index, child])
            if parent < 0 or parent >= z.shape[1] or not bool(mask[batch_index, parent]):
                continue
            eligible = topology_safe_parent_negative_mask(
                parent_ids[batch_index],
                mask[batch_index],
                child,
                lca_depth=(lca_depth[batch_index] if lca_depth is not None else None),
                tree_relation_targets=(
                    tree_relation_targets[batch_index]
                    if tree_relation_targets is not None else None
                ),
                b_side=(b_side[batch_index] if b_side is not None else None),
            )
            negatives = torch.nonzero(eligible, as_tuple=False).flatten()
            if negatives.numel() == 0:
                continue
            positive_distance = distance(
                z[batch_index, child],
                z[batch_index, parent],
                curvature=curvature,
            )
            negative_distances = distance(
                z[batch_index, child].expand_as(z[batch_index, negatives]),
                z[batch_index, negatives],
                curvature=curvature,
            )
            negative_distance = (
                negative_distances.min()
                if hard_negative
                else negative_distances[
                    (child * 1103515245 + parent * 12345) % negatives.numel()
                ]
            )
            losses.append(F.relu(positive_distance - negative_distance + margin))
    return torch.stack(losses).mean() if losses else z.sum() * 0.0


def topology_safe_parent_negative_mask(
    parent_ids: torch.Tensor,
    node_mask: torch.Tensor,
    child: int,
    *,
    lca_depth: torch.Tensor | None = None,
    tree_relation_targets: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return candidates allowed by the directed parent-ranking contract.

    Explicit relation labels are authoritative: classes 0--3 are excluded and
    classes 4--5 are negative.  In particular, a class-4 pair is not removed
    merely because both B branches meet at the retained Upsilon root.
    """

    if parent_ids.is_cuda:
        raise RuntimeError(
            "topology_safe_parent_negative_mask fallback is CPU-only; pass the "
            "precomputed vectorized parent_negative_mask on CUDA"
        )

    valid = node_mask.bool().clone()
    parent = int(parent_ids[child])
    valid[child] = False
    if 0 <= parent < valid.numel():
        valid[parent] = False

    def ancestors(node: int) -> set[int]:
        result: set[int] = set()
        current = int(parent_ids[node])
        while 0 <= current < parent_ids.numel() and current not in result:
            result.add(current)
            current = int(parent_ids[current])
        return result

    child_ancestors = ancestors(child)
    fallback_geometry = (
        build_exact_tree_geometry(parent_ids)
        if tree_relation_targets is None
        else None
    )
    for candidate in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        candidate_ancestors = ancestors(candidate)
        if candidate in child_ancestors or child in candidate_ancestors:
            valid[candidate] = False
            continue
        if parent >= 0 and int(parent_ids[candidate]) == parent:
            valid[candidate] = False
            continue
        if tree_relation_targets is not None:
            if int(tree_relation_targets[child, candidate]) <= 3:
                valid[candidate] = False
            # Do not apply the fallback LCA rule after explicit classes exist.
            continue
        assert fallback_geometry is not None
        left_edges = int(fallback_geometry.edges_to_lca_from_i[child, candidate])
        right_edges = int(fallback_geometry.edges_to_lca_from_j[child, candidate])
        close_local_branch = (
            int(fallback_geometry.lca_node_id[child, candidate]) >= 0
            and left_edges <= 2
            and right_edges <= 2
        )
        same_explicit_branch = (
            b_side is not None
            and int(b_side[child]) >= 0
            and int(b_side[candidate]) == int(b_side[child])
        )
        if close_local_branch or same_explicit_branch:
            valid[candidate] = False

    if not valid.any():
        return valid
    if tree_relation_targets is not None:
        different_b = valid & (tree_relation_targets[child] == 4)
        if different_b.any():
            return different_b
        unrelated = valid & (tree_relation_targets[child] == 5)
        if unrelated.any():
            return unrelated
        return torch.zeros_like(valid)
    if b_side is not None and int(b_side[child]) >= 0:
        other_branch = valid & (b_side >= 0) & (b_side != b_side[child])
        if other_branch.any():
            return other_branch
    return valid


@torch.no_grad()
def parent_negative_coverage_statistics(
    parent_ids: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    lca_depth: torch.Tensor | None = None,
    tree_relation_targets: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
    parent_negative_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Count the exact denominator of the directed parent objective."""

    if parent_negative_mask is not None:
        safe_parent = parent_ids.clamp_min(0)
        children = (parent_ids >= 0) & node_mask
        children &= node_mask.gather(1, safe_parent)
        eligible_children = children & parent_negative_mask.any(dim=-1)
        different_b_children = eligible_children & (
            parent_negative_mask & (tree_relation_targets == 4)
        ).any(dim=-1) if tree_relation_targets is not None else torch.zeros_like(children)
        reference = node_mask.sum().to(torch.float32) * 0.0
        total = children.sum().to(torch.float32) + reference
        eligible = eligible_children.sum().to(torch.float32) + reference
        different_b = different_b_children.sum().to(torch.float32) + reference
        return {
            "parent_children_with_eligible_negative": eligible,
            "parent_children_with_different_b_negative": different_b,
            "parent_children_with_no_negative": total - eligible,
            "parent_loss_active_fraction": eligible / total.clamp_min(1.0),
            "parent_ranking_accuracy_denominator": eligible,
        }
    if parent_ids.is_cuda:
        raise RuntimeError("parent_negative_mask is required for CUDA diagnostics")
    eligible_children = different_b_children = total_children = 0
    for batch_index in range(parent_ids.shape[0]):
        for child in torch.nonzero(node_mask[batch_index], as_tuple=False).flatten().tolist():
            parent = int(parent_ids[batch_index, child])
            if parent < 0 or parent >= parent_ids.shape[1] or not bool(node_mask[batch_index, parent]):
                continue
            total_children += 1
            eligible = topology_safe_parent_negative_mask(
                parent_ids[batch_index],
                node_mask[batch_index],
                child,
                lca_depth=lca_depth[batch_index] if lca_depth is not None else None,
                tree_relation_targets=(
                    tree_relation_targets[batch_index]
                    if tree_relation_targets is not None
                    else None
                ),
                b_side=b_side[batch_index] if b_side is not None else None,
            )
            if eligible.any():
                eligible_children += 1
            if tree_relation_targets is not None and (
                eligible & (tree_relation_targets[batch_index, child] == 4)
            ).any():
                different_b_children += 1
            elif b_side is not None and int(b_side[batch_index, child]) >= 0 and (
                eligible
                & (b_side[batch_index] >= 0)
                & (b_side[batch_index] != b_side[batch_index, child])
            ).any():
                different_b_children += 1
    reference = node_mask.sum().to(torch.float32) * 0.0
    total = reference + float(total_children)
    eligible = reference + float(eligible_children)
    different_b = reference + float(different_b_children)
    return {
        "parent_children_with_eligible_negative": eligible,
        "parent_children_with_different_b_negative": different_b,
        "parent_children_with_no_negative": total - eligible,
        "parent_loss_active_fraction": eligible / total.clamp_min(1.0),
        "parent_ranking_accuracy_denominator": eligible,
    }


@torch.no_grad()
def parent_child_ranking_accuracy(
    z: torch.Tensor,
    parent_ids: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    curvature: float = 1.0,
    lca_depth: torch.Tensor | None = None,
    tree_relation_targets: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
    parent_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of directed parents closer than every eligible safe negative."""

    if parent_negative_mask is not None:
        safe_parent = parent_ids.clamp_min(0)
        children = (parent_ids >= 0) & node_mask
        children &= node_mask.gather(1, safe_parent)
        eligible = parent_negative_mask & node_mask[:, :, None] & node_mask[:, None, :]
        active = children & eligible.any(dim=-1)
        parent_z = z.gather(1, safe_parent.unsqueeze(-1).expand_as(z))
        positive = distance(z, parent_z, curvature=curvature)
        negative = distance(
            z[:, :, None, :], z[:, None, :, :], curvature=curvature
        ).masked_fill(~eligible, float("inf")).amin(dim=-1)
        correct = (positive < negative).to(z.dtype)
        weights = active.to(z.dtype)
        return (correct * weights).sum() / weights.sum().clamp_min(1.0)
    if parent_ids.is_cuda:
        raise RuntimeError("parent_negative_mask is required for CUDA diagnostics")
    outcomes: list[torch.Tensor] = []
    for batch_index in range(z.shape[0]):
        for child in torch.nonzero(node_mask[batch_index], as_tuple=False).flatten().tolist():
            parent = int(parent_ids[batch_index, child])
            if parent < 0 or not bool(node_mask[batch_index, parent]):
                continue
            eligible = topology_safe_parent_negative_mask(
                parent_ids[batch_index], node_mask[batch_index], child,
                lca_depth=lca_depth[batch_index] if lca_depth is not None else None,
                tree_relation_targets=(
                    tree_relation_targets[batch_index]
                    if tree_relation_targets is not None else None
                ),
                b_side=b_side[batch_index] if b_side is not None else None,
            )
            negatives = torch.nonzero(eligible, as_tuple=False).flatten()
            if not negatives.numel():
                continue
            positive = distance(z[batch_index, child], z[batch_index, parent], curvature=curvature)
            negative = distance(
                z[batch_index, child].expand_as(z[batch_index, negatives]),
                z[batch_index, negatives], curvature=curvature,
            ).min()
            outcomes.append((positive < negative).to(z.dtype))
    return torch.stack(outcomes).mean() if outcomes else z.sum() * 0.0


def radius_targets(
    level_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    r_min: float = 0.1,
    r_max: float = 1.5,
    full_event_max_level: torch.Tensor | None = None,
) -> torch.Tensor:
    """Leaves are farthest out; roots at each event's maximum level are inward."""

    safe_levels = level_ids.float().clamp_min(0)
    if full_event_max_level is None:
        masked_levels = torch.where(mask, safe_levels, torch.zeros_like(safe_levels))
        max_level = masked_levels.max(dim=-1, keepdim=True).values
    else:
        max_level = full_event_max_level.float()
        if max_level.ndim == level_ids.ndim:
            max_level = max_level.max(dim=-1, keepdim=True).values
        elif max_level.ndim == level_ids.ndim - 1:
            max_level = max_level.unsqueeze(-1)
        else:
            raise ValueError("full_event_max_level has an invalid shape")
    fraction = (max_level - safe_levels) / max_level.clamp_min(1.0)
    return r_min + (r_max - r_min) * fraction


def radius_depth_loss(
    z: torch.Tensor,
    level_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    r_min: float = 0.1,
    r_max: float = 1.5,
    scale: float | None = None,
    curvature: float = 1.0,
    full_event_max_level: torch.Tensor | None = None,
    target_mode: str = "generation_height_radius",
    depth_from_retained_root: torch.Tensor | None = None,
    distance_to_nearest_retained_root: torch.Tensor | None = None,
) -> torch.Tensor:
    # ``scale`` is retained only as an old-call compatibility alias.
    if scale is not None:
        r_max = max(r_min, float(scale) * max(int(level_ids.max().item()) + 1, 1))
    prediction = radius(z, curvature=curvature)
    if target_mode == "generation_height_radius":
        target = radius_targets(
            level_ids, mask, r_min=r_min, r_max=r_max,
            full_event_max_level=full_event_max_level,
        )
    elif target_mode == "exact_root_depth_radius":
        exact_depth = (
            depth_from_retained_root
            if depth_from_retained_root is not None
            else distance_to_nearest_retained_root
        )
        if exact_depth is None:
            raise ValueError("exact_root_depth_radius requires precomputed root depth")
        safe_depth = exact_depth.float().clamp_min(0)
        maximum = torch.where(mask, safe_depth, torch.zeros_like(safe_depth)).max(
            dim=-1, keepdim=True
        ).values
        target = r_min + (r_max - r_min) * safe_depth / maximum.clamp_min(1.0)
    elif target_mode == "weak_or_learned_radius":
        penalty = F.relu(r_min - prediction) + F.relu(prediction - r_max)
        weights = mask.to(penalty.dtype)
        return (penalty * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        raise ValueError(f"unknown radius target mode: {target_mode}")
    weights = mask.to(prediction.dtype)
    return ((prediction - target).square() * weights).sum() / weights.sum().clamp_min(1.0)


def variance_regularization(
    tangent: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float | None = None,
    target_tangent_norm: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    valid = tangent[mask]
    if valid.shape[0] < 2:
        return tangent.sum() * 0.0
    if gamma is None:
        # For curvature one, d_H(0, exp_0(u)) = 2 ||u||.  A fixed RMS tangent
        # norm therefore requires a 1/sqrt(dimension) per-coordinate target.
        gamma = dimension_aware_tangent_variance_target(
            tangent.shape[-1], target_tangent_norm=target_tangent_norm
        )
    if gamma < 0:
        raise ValueError("tangent variance target must be non-negative")
    standard_deviation = torch.sqrt(valid.var(dim=0, unbiased=False) + eps)
    return F.relu(float(gamma) - standard_deviation).mean()


def covariance_regularization(tangent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = tangent[mask]
    if valid.shape[0] < 2:
        return tangent.sum() * 0.0
    centered = valid - valid.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(valid.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / covariance.shape[0]


def balanced_tangent_sample(
    tangent: torch.Tensor,
    mask: torch.Tensor,
    *,
    level_ids: torch.Tensor | None = None,
    node_kind_ids: torch.Tensor | None = None,
    event_ids: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
    group_by: tuple[str, ...] = ("level", "node_kind"),
    max_per_group: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically cap requested level/kind/event/branch groups."""

    if not group_by:
        return tangent, mask
    selected = torch.zeros_like(mask)
    group = torch.zeros_like(mask, dtype=torch.long)
    if "event" in group_by:
        event_group = (
            event_ids[:, None].expand_as(group)
            if event_ids is not None
            else torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(group)
        )
        group = group + event_group * 1_000_000
    if "branch" in group_by and b_side is not None:
        group = group + (b_side + 1).clamp_min(0) * 10_000
    if "level" in group_by and level_ids is not None:
        group = group + level_ids.clamp_min(0) * 32
    if "node_kind" in group_by and node_kind_ids is not None:
        group = group + node_kind_ids.clamp_min(0)
    flat_valid = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if flat_valid.numel() == 0:
        return tangent, selected
    valid_groups = group.reshape(-1)[flat_valid]
    order = torch.argsort(valid_groups, stable=True)
    sorted_groups = valid_groups[order]
    positions = torch.arange(order.numel(), device=mask.device)
    group_start = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=mask.device),
            sorted_groups[1:] != sorted_groups[:-1],
        ]
    )
    starts = torch.where(group_start, positions, torch.zeros_like(positions))
    starts = torch.cummax(starts, dim=0).values
    rank_within_group = positions - starts
    kept_flat = flat_valid[order[rank_within_group < max_per_group]]
    selected.reshape(-1)[kept_flat] = True
    return tangent, selected


def pool_b_branch_embeddings(
    node_embeddings: torch.Tensor,
    b_side: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    mode: str = "mean_all",
    level_ids: torch.Tensor | None = None,
    attention_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool two unordered branch sets with an explicit ablation contract."""

    if mode not in {"mean_all", "fsp_only", "b_root", "learned_attention", "level_weighted"}:
        raise ValueError(f"unknown channel pooling mode: {mode}")
    if mode in {"b_root", "level_weighted"} and level_ids is None:
        raise ValueError(f"{mode} channel pooling requires level_ids")
    if mode == "learned_attention" and attention_logits is None:
        raise ValueError("learned_attention channel pooling requires attention logits")

    pooled = []
    available = []
    for side in (0, 1):
        membership = node_mask & (b_side == side)
        if mode == "mean_all":
            weights = membership.to(node_embeddings.dtype)
        elif mode == "fsp_only":
            if level_ids is None:
                raise ValueError("fsp_only channel pooling requires level_ids")
            weights = (membership & (level_ids == 0)).to(node_embeddings.dtype)
        elif mode == "b_root":
            branch_levels = torch.where(
                membership, level_ids, torch.full_like(level_ids, -1)
            )
            root_level = branch_levels.max(dim=-1, keepdim=True).values
            weights = (membership & (level_ids == root_level)).to(node_embeddings.dtype)
        elif mode == "level_weighted":
            weights = membership.to(node_embeddings.dtype) * (
                level_ids.clamp_min(0) + 1
            ).to(node_embeddings.dtype)
        else:
            masked_logits = attention_logits.masked_fill(~membership, -1e4)
            weights = torch.softmax(masked_logits, dim=-1) * membership.to(
                node_embeddings.dtype
            )
        pooled.append(
            torch.einsum("bn,bnd->bd", weights, node_embeddings)
            / weights.sum(dim=-1, keepdim=True).clamp_min(1)
        )
        available.append(weights.sum(dim=-1) > 0)
    return torch.stack(pooled, dim=1), torch.stack(available, dim=1)


def channel_metric_loss(
    branch_embeddings: torch.Tensor,
    branch_mask: torch.Tensor,
    channel_ids: torch.Tensor,
    structured_similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regress cosine similarity to exact/structured channel similarity."""

    left = F.normalize(branch_embeddings[:, 0], dim=-1)
    right = F.normalize(branch_embeddings[:, 1], dim=-1)
    prediction = (left * right).sum(dim=-1)
    exact = (
        (channel_ids[:, 0] > 0)
        & (channel_ids[:, 1] > 0)
        & (channel_ids[:, 0] == channel_ids[:, 1])
    ).float()
    target = exact if structured_similarity is None else structured_similarity.clamp(0, 1)
    target = 2 * target - 1
    valid = branch_mask.all(dim=-1)
    return F.mse_loss(prediction[valid], target[valid]) if valid.any() else branch_embeddings.sum() * 0.0


def cross_event_channel_metric_loss(
    branch_embeddings: torch.Tensor,
    branch_mask: torch.Tensor,
    full_truth_channel_ids: torch.Tensor,
    reconstructable_channel_ids: torch.Tensor | None = None,
    branch_count_arrays: torch.Tensor | None = None,
    *,
    temperature: float = 0.2,
    structured_positive_threshold: float = 0.75,
    structured_regression_weight: float = 0.25,
    memory_embeddings: torch.Tensor | None = None,
    memory_full_truth_channel_ids: torch.Tensor | None = None,
    memory_reconstructable_channel_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervised contrastive channel learning over all B branches in a batch.

    Branches from the same event are treated exactly like any other pair:
    they are positive only when an explicit channel identity agrees.
    """

    flat_embeddings = branch_embeddings.reshape(-1, branch_embeddings.shape[-1])
    flat_mask = branch_mask.reshape(-1)
    full_ids = full_truth_channel_ids.reshape(-1)
    reco_ids = (
        reconstructable_channel_ids.reshape(-1)
        if reconstructable_channel_ids is not None
        else torch.zeros_like(full_ids)
    )
    valid_indices = flat_mask.nonzero(as_tuple=False).flatten()
    zero = branch_embeddings.sum() * 0.0
    if valid_indices.numel() < 2:
        return zero, {
            "channel_positive_pairs": zero,
            "channel_negative_pairs": zero,
            "channel_active_anchors": zero,
            "channel_total_anchors": valid_indices.numel() + zero,
        }
    embedding = F.normalize(flat_embeddings[valid_indices], dim=-1)
    selected_full = full_ids[valid_indices]
    selected_reco = reco_ids[valid_indices]
    candidate_embedding = embedding
    candidate_full = selected_full
    candidate_reco = selected_reco
    if memory_embeddings is not None and memory_embeddings.numel():
        if memory_full_truth_channel_ids is None:
            raise ValueError("memory channel embeddings require full-truth channel IDs")
        candidate_embedding = torch.cat(
            [embedding, F.normalize(memory_embeddings.to(embedding), dim=-1)],
            dim=0,
        )
        candidate_full = torch.cat(
            [selected_full, memory_full_truth_channel_ids.to(selected_full)],
            dim=0,
        )
        memory_reco = (
            memory_reconstructable_channel_ids.to(selected_reco)
            if memory_reconstructable_channel_ids is not None
            else torch.zeros_like(memory_full_truth_channel_ids).to(selected_reco)
        )
        candidate_reco = torch.cat([selected_reco, memory_reco], dim=0)
    similarity = embedding @ candidate_embedding.T / temperature
    identity = torch.zeros_like(similarity, dtype=torch.bool)
    identity[:, : embedding.shape[0]] = torch.eye(
        embedding.shape[0], dtype=torch.bool, device=similarity.device
    )
    exact_full = (
        (selected_full[:, None] > 0)
        & (selected_full[:, None] == candidate_full[None, :])
    )
    exact_reco = (
        (selected_reco[:, None] > 0)
        & (selected_reco[:, None] == candidate_reco[None, :])
    )
    structured_similarity = torch.zeros_like(similarity)
    structured_pairs = torch.zeros_like(similarity, dtype=torch.bool)
    if branch_count_arrays is not None:
        flat_counts = branch_count_arrays.reshape(
            -1, branch_count_arrays.shape[-1]
        )[valid_indices].to(similarity)
        intersection = torch.minimum(
            flat_counts[:, None, :], flat_counts[None, :, :]
        ).sum(dim=-1)
        union = torch.maximum(
            flat_counts[:, None, :], flat_counts[None, :, :]
        ).sum(dim=-1)
        current_similarity = torch.where(
            union > 0, intersection / union.clamp_min(1e-6), torch.zeros_like(union)
        )
        structured_similarity[:, : embedding.shape[0]] = current_similarity
        structured_pairs[:, : embedding.shape[0]] = (
            current_similarity >= structured_positive_threshold
        ) & (union > 0)
    positives = (exact_full | exact_reco | structured_pairs) & ~identity
    pairs = ~identity
    positive_count = positives.sum(dim=-1)
    active_anchor = positive_count > 0
    log_denominator = torch.logsumexp(
        similarity.masked_fill(~pairs, float("-inf")), dim=-1
    )
    positive_log_probability = (
        (similarity - log_denominator[:, None]).masked_fill(~positives, 0.0).sum(dim=-1)
        / positive_count.clamp_min(1)
    )
    contrastive = (
        -positive_log_probability[active_anchor].mean()
        if active_anchor.any()
        else zero
    )
    current_pairs = (
        ~torch.eye(embedding.shape[0], dtype=torch.bool, device=embedding.device)
        if branch_count_arrays is not None
        else torch.zeros(
            (embedding.shape[0], embedding.shape[0]),
            dtype=torch.bool,
            device=embedding.device,
        )
    )
    structured_regression = (
        F.smooth_l1_loss(
            (embedding @ embedding.T)[current_pairs],
            (2 * structured_similarity[:, : embedding.shape[0]] - 1)[current_pairs],
        )
        if current_pairs.any()
        else zero
    )
    loss = contrastive + structured_regression_weight * structured_regression
    return loss, {
        "channel_positive_pairs": positives.sum().to(similarity.dtype) / 2,
        "channel_negative_pairs": (pairs & ~positives).sum().to(similarity.dtype) / 2,
        "channel_active_anchors": active_anchor.sum().to(similarity.dtype),
        "channel_total_anchors": similarity.new_tensor(similarity.shape[0]),
        "channel_structured_regression": structured_regression,
    }


def level_height_tree_distance_targets_v1(
    *,
    lca_depth: torch.Tensor,
    level_ids: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible ablation using reconstruction height differences."""

    left = level_ids[:, :, None].float()
    right = level_ids[:, None, :].float()
    raw = (lca_depth.float() - left).clamp_min(0) + (lca_depth.float() - right).clamp_min(0)
    raw = torch.where(lca_depth >= 0, raw, torch.zeros_like(raw))
    maximum = torch.where(pair_mask, raw, torch.zeros_like(raw)).flatten(1).max(
        dim=-1, keepdim=True
    ).values.clamp_min(1.0)
    return raw / maximum[:, None]


def tree_distance_targets(
    *,
    exact_tree_path_distance: torch.Tensor,
    pair_mask: torch.Tensor,
    target_scale_edges: float = 8.0,
) -> torch.Tensor:
    """Robust fixed-scale target for exact retained-tree edge distance.

    ``log1p(distance) / log1p(target_scale_edges)`` has a versioned, event-
    independent scale.  Adding a distant outlier therefore cannot rescale all
    existing target pairs.
    """

    if target_scale_edges <= 0:
        raise ValueError("target_scale_edges must be positive")
    raw = exact_tree_path_distance.to(torch.float32).clamp_min(0)
    target = torch.log1p(raw) / torch.log1p(
        raw.new_tensor(float(target_scale_edges))
    )
    return torch.where(
        pair_mask & (exact_tree_path_distance >= 0),
        target,
        torch.zeros_like(target),
    )


def hyperbolic_tree_distance_loss(
    z: torch.Tensor,
    *,
    exact_tree_path_distance: torch.Tensor,
    pair_mask: torch.Tensor,
    curvature: float = 1.0,
    target_scale_edges: float = 8.0,
    prediction_scale: float = 3.0,
) -> torch.Tensor:
    """Regress fixed-scale Poincare distance to exact retained-tree distance."""

    target = tree_distance_targets(
        exact_tree_path_distance=exact_tree_path_distance,
        pair_mask=pair_mask,
        target_scale_edges=target_scale_edges,
    )
    prediction = distance(
        z[:, :, None, :],
        z[:, None, :, :],
        curvature=curvature,
    )
    if prediction_scale <= 0:
        raise ValueError("prediction_scale must be positive")
    prediction = prediction / float(prediction_scale)
    selected = pair_mask & (exact_tree_path_distance >= 0)
    return F.smooth_l1_loss(prediction[selected], target[selected]) if selected.any() else z.sum() * 0.0


def collapse_diagnostics(
    z: torch.Tensor,
    mask: torch.Tensor,
    *,
    level_ids: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
    boundary_threshold: float = 0.95,
    curvature: float = 1.0,
) -> dict[str, torch.Tensor]:
    tangent = logmap0(z, curvature=curvature)
    valid = tangent[mask]
    zero = z.sum() * 0.0
    if valid.shape[0] < 2:
        return {
            "mean_dimension_std": zero,
            "min_dimension_std": zero,
            "covariance_off_diagonal_norm": zero,
            "effective_rank": zero,
            "boundary_fraction": zero,
            "radius_level_correlation": zero,
            "angular_separation_by_branch": zero,
        }
    std = valid.std(dim=0, unbiased=False)
    centered = valid - valid.mean(dim=0)
    covariance = centered.T @ centered / max(valid.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-8)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-8).log()).sum())
    norm = torch.linalg.norm(z[mask], dim=-1)
    radius_values = radius(z, curvature=curvature)[mask]
    correlation = zero
    if level_ids is not None:
        levels = level_ids[mask].float()
        if levels.std(unbiased=False) > 0 and radius_values.std(unbiased=False) > 0:
            correlation = torch.corrcoef(torch.stack([levels, radius_values]))[0, 1]
    angular = zero
    if b_side is not None:
        branch_valid = mask & (b_side >= 0)
        if branch_valid.sum() > 1:
            directions = F.normalize(tangent[branch_valid], dim=-1)
            sides = b_side[branch_valid]
            cross = sides[:, None] != sides[None, :]
            if cross.any():
                angular = torch.acos(
                    (directions @ directions.T).clamp(-1 + 1e-6, 1 - 1e-6)
                )[cross].mean()
    return {
        "mean_dimension_std": std.mean(),
        "min_dimension_std": std.min(),
        "covariance_off_diagonal_norm": torch.linalg.matrix_norm(off_diagonal),
        "effective_rank": effective_rank,
        "boundary_fraction": (
            curvature**0.5 * norm >= boundary_threshold
        ).float().mean(),
        "radius_level_correlation": correlation,
        "angular_separation_by_branch": angular,
    }


def relation_distance_diagnostics(
    z: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor,
    curvature: float = 1.0,
) -> dict[str, torch.Tensor]:
    distances = distance(
        z[:, :, None, :],
        z[:, None, :, :],
        curvature=curvature,
    )
    positive = pair_mask & (targets <= 3) & (targets > 0)
    negative = pair_mask & (targets >= 4)
    zero = z.sum() * 0.0
    return {
        "mean_positive_relation_distance": distances[positive].mean() if positive.any() else zero,
        "mean_negative_relation_distance": distances[negative].mean() if negative.any() else zero,
    }


def hyperbolic_pretraining_loss(
    *,
    z: torch.Tensor,
    parent_ids: torch.Tensor,
    level_ids: torch.Tensor,
    node_mask: torch.Tensor,
    tree_relation_logits: torch.Tensor | None = None,
    tree_relation_targets: torch.Tensor | None = None,
    tree_relation_mask: torch.Tensor | None = None,
    lca_depth: torch.Tensor | None = None,
    exact_tree_path_distance: torch.Tensor | None = None,
    parent_negative_mask: torch.Tensor | None = None,
    b_side: torch.Tensor | None = None,
    node_kind_ids: torch.Tensor | None = None,
    event_ids: torch.Tensor | None = None,
    anti_collapse_group_by: tuple[str, ...] = ("level", "node_kind", "event", "branch"),
    channel_embeddings: torch.Tensor | None = None,
    channel_mask: torch.Tensor | None = None,
    channel_ids: torch.Tensor | None = None,
    structured_channel_similarity: torch.Tensor | None = None,
    full_truth_channel_ids: torch.Tensor | None = None,
    reconstructable_channel_ids: torch.Tensor | None = None,
    channel_branch_count_arrays: torch.Tensor | None = None,
    channel_memory_embeddings: torch.Tensor | None = None,
    channel_memory_full_truth_ids: torch.Tensor | None = None,
    channel_memory_reconstructable_ids: torch.Tensor | None = None,
    same_mother_logits: torch.Tensor | None = None,
    same_branch_logits: torch.Tensor | None = None,
    same_mother: torch.Tensor | None = None,
    same_branch: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
    curvature: float = 1.0,
    full_event_max_level: torch.Tensor | None = None,
    tangent_variance_target: float | None = None,
    radius_target_mode: str = "generation_height_radius",
    depth_from_retained_root: torch.Tensor | None = None,
    distance_to_nearest_retained_root: torch.Tensor | None = None,
) -> HyperbolicLossOutput:
    """Principal LCA, parent, depth, channel, variance and covariance objective."""

    weights = {
        "lca": 1.0,
        "parent": 1.0,
        "depth": 0.2,
        "channel": 0.2,
        "tree_distance": 1.0,
        "var": 0.1,
        "cov": 0.01,
        "same_mother": 0.0,
        "same_branch": 0.0,
        **(weights or {}),
    }
    if tree_relation_targets is None and lca_depth is not None:
        tree_relation_targets, tree_relation_mask = build_tree_relation_targets(
            parent_ids=parent_ids,
            lca_depth=lca_depth,
            level_ids=level_ids,
            node_mask=node_mask,
            b_side=b_side,
        )
    zero = z.sum() * 0.0
    components: dict[str, torch.Tensor] = {
        "lca": (
            balanced_tree_relation_loss(
                tree_relation_logits,
                tree_relation_targets,
                tree_relation_mask
                if tree_relation_mask is not None
                else node_mask[:, :, None] & node_mask[:, None, :],
            )
            if tree_relation_logits is not None and tree_relation_targets is not None
            else zero
        ),
        "parent": parent_child_margin_loss(
            z,
            parent_ids,
            node_mask,
            curvature=curvature,
            lca_depth=lca_depth,
            tree_relation_targets=tree_relation_targets,
            b_side=b_side,
            parent_negative_mask=parent_negative_mask,
        ),
        "depth": radius_depth_loss(
            z,
            level_ids,
            node_mask,
            curvature=curvature,
            full_event_max_level=full_event_max_level,
            target_mode=radius_target_mode,
            depth_from_retained_root=depth_from_retained_root,
            distance_to_nearest_retained_root=distance_to_nearest_retained_root,
        ),
        "tree_distance": (
            hyperbolic_tree_distance_loss(
                z,
                exact_tree_path_distance=exact_tree_path_distance,
                pair_mask=(
                    tree_relation_mask
                    if tree_relation_mask is not None
                    else node_mask[:, :, None] & node_mask[:, None, :]
                ),
                curvature=curvature,
            )
            if exact_tree_path_distance is not None
            else zero
        ),
    }
    channel_pair_diagnostics: dict[str, torch.Tensor] = {}
    if (
        channel_embeddings is not None
        and channel_mask is not None
        and full_truth_channel_ids is not None
    ):
        components["channel"], channel_pair_diagnostics = cross_event_channel_metric_loss(
            channel_embeddings,
            channel_mask,
            full_truth_channel_ids,
            reconstructable_channel_ids,
            channel_branch_count_arrays,
            memory_embeddings=channel_memory_embeddings,
            memory_full_truth_channel_ids=channel_memory_full_truth_ids,
            memory_reconstructable_channel_ids=channel_memory_reconstructable_ids,
        )
    elif channel_embeddings is not None and channel_mask is not None and channel_ids is not None:
        components["channel"] = channel_metric_loss(
            channel_embeddings,
            channel_mask,
            channel_ids,
            structured_channel_similarity,
        )
    else:
        components["channel"] = zero
    tangent, anti_collapse_mask = balanced_tangent_sample(
        logmap0(z, curvature=curvature),
        node_mask,
        level_ids=level_ids,
        node_kind_ids=node_kind_ids,
        event_ids=event_ids,
        b_side=b_side,
        group_by=anti_collapse_group_by,
    )
    components["var"] = variance_regularization(
        tangent,
        anti_collapse_mask,
        gamma=tangent_variance_target,
    )
    components["cov"] = covariance_regularization(tangent, anti_collapse_mask)
    if (
        same_mother_logits is not None
        and same_mother is not None
        and weights["same_mother"] != 0
    ):
        components["same_mother"] = same_pair_bce(same_mother_logits, same_mother, node_mask)
    if (
        same_branch_logits is not None
        and same_branch is not None
        and weights["same_branch"] != 0
    ):
        components["same_branch"] = same_pair_bce(same_branch_logits, same_branch, node_mask)
    total = sum(components[name] * weights[name] for name in components)
    diagnostics = collapse_diagnostics(
        z,
        node_mask,
        level_ids=level_ids,
        b_side=b_side,
        curvature=curvature,
    )
    diagnostics.update(
        parent_negative_coverage_statistics(
            parent_ids,
            node_mask,
            lca_depth=lca_depth,
            tree_relation_targets=tree_relation_targets,
            b_side=b_side,
            parent_negative_mask=parent_negative_mask,
        )
    )
    diagnostics.update(channel_pair_diagnostics)
    active_pairs = (
        tree_relation_mask
        if tree_relation_mask is not None
        else node_mask[:, :, None] & node_mask[:, None, :]
    )
    diagnostics.update(
        {
            "active_denominator_lca": active_pairs.sum().to(z.dtype),
            "active_denominator_tree_distance": (
                active_pairs & (exact_tree_path_distance >= 0)
            ).sum().to(z.dtype)
            if exact_tree_path_distance is not None
            else zero,
            "active_denominator_radius": node_mask.sum().to(z.dtype),
            "active_denominator_variance": anti_collapse_mask.sum().to(z.dtype),
            "active_denominator_covariance": anti_collapse_mask.sum().to(z.dtype),
        }
    )
    if tree_relation_targets is not None:
        diagnostics.update(
            relation_distance_diagnostics(
                z,
                tree_relation_targets,
                tree_relation_mask
                if tree_relation_mask is not None
                else node_mask[:, :, None] & node_mask[:, None, :],
                curvature=curvature,
            )
        )
    return HyperbolicLossOutput(total=total, components=components, diagnostics=diagnostics)


__all__ = [
    "HYPERBOLIC_SCALE_CONTRACT_VERSION",
    "LEVEL_HEIGHT_DISTANCE_ABLATION_VERSION",
    "TREE_DISTANCE_CONTRACT_VERSION",
    "HyperbolicLossOutput",
    "N_TREE_RELATIONS",
    "TREE_RELATION_NAMES",
    "balanced_tree_relation_loss",
    "build_tree_relation_targets",
    "build_topology_safe_parent_negative_mask",
    "channel_metric_loss",
    "cross_event_channel_metric_loss",
    "collapse_diagnostics",
    "covariance_regularization",
    "dimension_aware_tangent_variance_target",
    "parent_negative_coverage_statistics",
    "level_height_tree_distance_targets_v1",
    "hyperbolic_pretraining_loss",
    "parent_child_margin_loss",
    "parent_child_ranking_accuracy",
    "topology_safe_parent_negative_mask",
    "hyperbolic_tree_distance_loss",
    "pool_b_branch_embeddings",
    "radius_depth_loss",
    "radius_targets",
    "same_pair_bce",
    "tree_distance_targets",
    "variance_regularization",
]
