"""Multiscale hyperbolic topology losses and anti-collapse diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the documented six-class LCA relation convention.

    Local branch means a retained common ancestor no more than two levels above
    the deeper node. Same-B and different-B relations use explicit B-side
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
    max_child_level = torch.maximum(level_ids[:, :, None], level_ids[:, None, :])
    local = (lca_depth >= 0) & ((lca_depth - max_child_level) <= 2) & ~same_parent & ~identity
    targets[valid & local] = 2
    if b_side is not None:
        valid_side = (b_side[:, :, None] >= 0) & (b_side[:, None, :] >= 0)
        same_side = b_side[:, :, None] == b_side[:, None, :]
        targets[valid & valid_side & same_side & ~local & ~same_parent & ~identity] = 3
        targets[valid & valid_side & ~same_side] = 4
    return targets, valid


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
) -> torch.Tensor:
    """Rank true parents with actual Poincare distance."""

    losses: list[torch.Tensor] = []
    for batch_index in range(z.shape[0]):
        valid = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        for child in valid.tolist():
            parent = int(parent_ids[batch_index, child])
            if parent < 0 or parent >= z.shape[1] or not bool(mask[batch_index, parent]):
                continue
            negatives = valid[(valid != parent) & (valid != child)]
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
) -> torch.Tensor:
    # ``scale`` is retained only as an old-call compatibility alias.
    if scale is not None:
        r_max = max(r_min, float(scale) * max(int(level_ids.max().item()) + 1, 1))
    target = radius_targets(
        level_ids,
        mask,
        r_min=r_min,
        r_max=r_max,
        full_event_max_level=full_event_max_level,
    )
    prediction = radius(z, curvature=curvature)
    return F.mse_loss(prediction[mask], target[mask]) if mask.any() else z.sum() * 0.0


def variance_regularization(
    tangent: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    valid = tangent[mask]
    if valid.shape[0] < 2:
        return tangent.sum() * 0.0
    standard_deviation = torch.sqrt(valid.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - standard_deviation).mean()


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
    for value in torch.unique(group[mask]).tolist():
        indices = torch.nonzero(mask & (group == value), as_tuple=False)
        indices = indices[:max_per_group]
        if indices.numel():
            selected[indices[:, 0], indices[:, 1]] = True
    return tangent, selected


def pool_b_branch_embeddings(
    node_embeddings: torch.Tensor,
    b_side: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool two unordered branch sets without node-level channel prediction."""

    pooled = []
    available = []
    for side in (0, 1):
        membership = node_mask & (b_side == side)
        weights = membership.to(node_embeddings.dtype)
        pooled.append(
            torch.einsum("bn,bnd->bd", weights, node_embeddings)
            / weights.sum(dim=-1, keepdim=True).clamp_min(1)
        )
        available.append(membership.any(dim=-1))
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
    per_anchor: list[torch.Tensor] = []
    for anchor in range(similarity.shape[0]):
        if not positives[anchor].any():
            continue
        denominator = torch.logsumexp(similarity[anchor][pairs[anchor]], dim=0)
        per_anchor.append(
            -(similarity[anchor][positives[anchor]] - denominator).mean()
        )
    contrastive = torch.stack(per_anchor).mean() if per_anchor else zero
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
        "channel_structured_regression": structured_regression,
    }


def tree_distance_targets(
    *,
    lca_depth: torch.Tensor,
    level_ids: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalized retained-tree path distance derived from LCA level."""

    left = level_ids[:, :, None].float()
    right = level_ids[:, None, :].float()
    raw = (lca_depth.float() - left).clamp_min(0) + (lca_depth.float() - right).clamp_min(0)
    raw = torch.where(lca_depth >= 0, raw, torch.zeros_like(raw))
    maximum = torch.where(pair_mask, raw, torch.zeros_like(raw)).flatten(1).max(
        dim=-1, keepdim=True
    ).values.clamp_min(1.0)
    return raw / maximum[:, None]


def hyperbolic_tree_distance_loss(
    z: torch.Tensor,
    *,
    lca_depth: torch.Tensor,
    level_ids: torch.Tensor,
    pair_mask: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Directly regress normalized Poincare distance to retained-tree distance."""

    target = tree_distance_targets(
        lca_depth=lca_depth,
        level_ids=level_ids,
        pair_mask=pair_mask,
    )
    prediction = distance(
        z[:, :, None, :],
        z[:, None, :, :],
        curvature=curvature,
    )
    scale = torch.where(pair_mask, prediction, torch.zeros_like(prediction)).flatten(1).max(
        dim=-1, keepdim=True
    ).values.clamp_min(1e-6)
    prediction = prediction / scale[:, None]
    selected = pair_mask & (lca_depth >= 0)
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
        "boundary_fraction": (norm >= boundary_threshold).float().mean(),
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
        ),
        "depth": radius_depth_loss(
            z,
            level_ids,
            node_mask,
            curvature=curvature,
            full_event_max_level=full_event_max_level,
        ),
        "tree_distance": (
            hyperbolic_tree_distance_loss(
                z,
                lca_depth=lca_depth,
                level_ids=level_ids,
                pair_mask=(
                    tree_relation_mask
                    if tree_relation_mask is not None
                    else node_mask[:, :, None] & node_mask[:, None, :]
                ),
                curvature=curvature,
            )
            if lca_depth is not None
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
    components["var"] = variance_regularization(tangent, anti_collapse_mask)
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
    diagnostics.update(channel_pair_diagnostics)
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
    "HyperbolicLossOutput",
    "N_TREE_RELATIONS",
    "TREE_RELATION_NAMES",
    "balanced_tree_relation_loss",
    "build_tree_relation_targets",
    "channel_metric_loss",
    "cross_event_channel_metric_loss",
    "collapse_diagnostics",
    "covariance_regularization",
    "hyperbolic_pretraining_loss",
    "parent_child_margin_loss",
    "hyperbolic_tree_distance_loss",
    "pool_b_branch_embeddings",
    "radius_depth_loss",
    "radius_targets",
    "same_pair_bce",
    "tree_distance_targets",
    "variance_regularization",
]
