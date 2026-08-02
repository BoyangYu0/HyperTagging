"""Losses for level-autoregressive mother set reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy

from hypertagging.losses.physics import charge_consistency_loss, p4_sum_consistency_loss
from hypertagging.losses.set_matching import hungarian_assignment, matching_cost
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.models.mother_pointer import source_conflict_penalty
from hypertagging.preprocessing.pid_filter import validate_pid_tokens


@dataclass(frozen=True)
class LevelLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    matches: list[list[tuple[int, int]]]
    confidence_targets: torch.Tensor | None = None


def targets_for_level(
    batch: dict[str, torch.Tensor],
    target_level: int,
    *,
    min_daughters: int = 2,
    target_policy: str = "complete_only",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    target_types = []
    target_masks = []
    target_p4 = []
    target_charge = []
    for batch_index in range(batch["node_features"].shape[0]):
        if target_policy not in {
            "complete_only",
            "reconstructable_partial",
            "diagnostic_all",
        }:
            raise ValueError(f"unknown reconstruction target policy: {target_policy}")
        eligible = batch["node_mask"][batch_index] & (
            batch["level_ids"][batch_index] == target_level
        )
        if target_policy != "diagnostic_all" and "valid_reconstruction_target" in batch:
            eligible &= batch["valid_reconstruction_target"][batch_index]
        if target_policy == "complete_only":
            if "recursive_reconstructable_complete" in batch:
                eligible &= batch["recursive_reconstructable_complete"][batch_index]
            elif "complete_reconstructable_decay" in batch:
                eligible &= batch["complete_reconstructable_decay"][batch_index]
        nodes = eligible.nonzero(as_tuple=False).flatten()
        context = batch["node_mask"][batch_index] & (batch["level_ids"][batch_index] < target_level)
        context_ids = context.nonzero(as_tuple=False).flatten()
        masks = torch.zeros((nodes.numel(), context_ids.numel()), dtype=torch.bool, device=batch["node_features"].device)
        for row, node_id in enumerate(nodes.tolist()):
            daughters = batch["daughter_adjacency"][batch_index, node_id, context_ids]
            masks[row] = daughters
        if min_daughters > 0 and nodes.numel():
            daughter_counts = batch["daughter_adjacency"][batch_index, nodes][:, context_ids].sum(dim=-1)
            nodes = nodes[daughter_counts >= min_daughters]
            masks = masks[daughter_counts >= min_daughters]
        labels = batch.get("pid_target_labels", batch["pid_labels"])[batch_index, nodes]
        validate_pid_tokens(labels, name=f"target level {target_level} PID labels")
        target_types.append(labels)
        target_masks.append(masks)
        target_p4.append(batch["p4"][batch_index, nodes])
        target_charge.append(batch["charge"][batch_index, nodes])
    return target_types, target_masks, target_p4, target_charge


def level_reconstruction_loss(
    output: MotherPointerOutput,
    batch: dict[str, torch.Tensor],
    *,
    target_level: int,
    weights: dict[str, float] | None = None,
    pointer_positive_weight: float = 4.0,
    pointer_focal_gamma: float = 1.0,
    object_positive_weight: float = 2.0,
    object_focal_gamma: float = 1.0,
    min_daughters: int = 2,
    target_policy: str = "complete_only",
    matching_production: bool = False,
    physics_component_scales: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    target_override: tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]
    | None = None,
    constraint_policy: ReconstructionConstraintPolicy | None = None,
    unrepresentable_target_counts: list[int] | None = None,
) -> LevelLossOutput:
    weights = {
        "object": 1.0,
        "type": 1.0,
        "pointer": 1.0,
        "cardinality": 0.2,
        "confidence": 0.2,
        "physics": 0.1,
        "source_conflict": 0.1,
        "mother_charge": 1.0,
        "query_repulsion": 0.0,
        **(weights or {}),
    }
    if target_override is None:
        target_types, target_masks, target_p4, target_charge = targets_for_level(
            batch,
            target_level,
            min_daughters=min_daughters,
            target_policy=target_policy,
        )
    else:
        target_types, target_masks, target_p4, target_charge = target_override
    for event_index, types in enumerate(target_types):
        if types.numel() > output.object_logits.shape[1]:
            raise OverflowError(
                f"event {event_index} target level {target_level} has {types.numel()} truth "
                f"mothers but decoder has {output.object_logits.shape[1]} queries"
            )
    object_targets = torch.zeros_like(output.object_logits)
    matched_pointer_targets = torch.zeros_like(
        output.pointer_logits, dtype=torch.bool
    )
    object_loss_mask = torch.ones_like(output.object_logits, dtype=torch.bool)
    confidence_targets = torch.zeros_like(output.confidence_logits)
    type_losses = []
    pointer_losses = []
    cardinality_losses = []
    p4_losses = []
    charge_losses = []
    mother_charge_losses = []
    all_matches: list[list[tuple[int, int]]] = []
    for batch_index, types in enumerate(target_types):
        context = batch["node_mask"][batch_index] & (batch["level_ids"][batch_index] < target_level)
        context_p4 = batch["p4"][batch_index, context]
        context_charge = batch["charge"][batch_index, context]
        if types.numel() == 0:
            all_matches.append([])
            continue
        cost = matching_cost(
            type_logits=output.type_logits[batch_index],
            pointer_logits=output.pointer_logits[batch_index, :, context],
            target_types=types,
            target_masks=target_masks[batch_index],
            object_logits=output.object_logits[batch_index],
            cardinality_logits=output.cardinality_logits[batch_index],
        )
        matches = hungarian_assignment(
            cost,
            production=matching_production,
            allow_bruteforce=not matching_production,
        )
        all_matches.append(matches)
        for query_id, target_id in matches:
            object_targets[batch_index, query_id] = 1.0
            matched_pointer_targets[batch_index, query_id, context] = (
                target_masks[batch_index][target_id].bool()
            )
            type_losses.append(F.cross_entropy(output.type_logits[batch_index, query_id][None], types[target_id][None]))
            pointer_losses.append(
                focal_binary_cross_entropy_with_logits(
                    output.pointer_logits[batch_index, query_id, context],
                    target_masks[batch_index][target_id].float(),
                    positive_weight=pointer_positive_weight,
                    gamma=pointer_focal_gamma,
                )
            )
            predicted_pointer = torch.sigmoid(
                output.pointer_logits[batch_index, query_id, context]
            )
            truth_pointer = target_masks[batch_index][target_id].float()
            hard_pointer = predicted_pointer >= 0.5
            truth_bool = truth_pointer.bool()
            intersection = (hard_pointer & truth_bool).sum().to(predicted_pointer.dtype)
            union = (hard_pointer | truth_bool).sum().clamp_min(1).to(
                predicted_pointer.dtype
            )
            type_correct = (
                output.type_logits[batch_index, query_id].argmax() == types[target_id]
            ).to(predicted_pointer.dtype)
            structurally_valid = predicted_pointer.new_tensor(1.0)
            conflict = batch.get("source_conflict_matrix")
            if conflict is not None and hard_pointer.any():
                selected_context = context.nonzero(as_tuple=False).flatten()[hard_pointer]
                selected_conflict = conflict[batch_index][
                    selected_context[:, None], selected_context[None, :]
                ]
                structurally_valid = (~torch.triu(selected_conflict, diagonal=1).any()).to(
                    predicted_pointer.dtype
                )
            confidence_targets[batch_index, query_id] = (
                intersection / union * type_correct * structurally_valid
            ).detach()
            cardinality = int(target_masks[batch_index][target_id].sum())
            if cardinality >= output.cardinality_logits.shape[-1]:
                raise OverflowError(
                    f"truth daughter cardinality {cardinality} exceeds decoder "
                    f"capacity {output.cardinality_logits.shape[-1] - 1}"
                )
            cardinality_losses.append(
                F.cross_entropy(
                    output.cardinality_logits[batch_index, query_id][None],
                    target_masks[batch_index][target_id].sum().long()[None],
                )
            )
            p4_losses.append(
                p4_sum_consistency_loss(
                    output.pointer_logits[batch_index, query_id, context][None, None],
                    context_p4[None],
                    target_p4[batch_index][target_id][None, None],
                    component_scales=physics_component_scales,
                )
            )
            charge_losses.append(
                charge_consistency_loss(
                    output.pointer_logits[batch_index, query_id, context][None, None],
                    context_charge[None],
                    target_charge[batch_index][target_id][None, None],
                )
            )
            if constraint_policy is not None and constraint_policy.mother_charge_compatibility in {
                "soft", "soft_train_hard_rollout"
            }:
                expected = context_charge.new_tensor(
                    constraint_policy.expected_charge(int(types[target_id]))
                )
                predicted_charge = (
                    torch.sigmoid(output.pointer_logits[batch_index, query_id, context])
                    * context_charge
                ).sum()
                mother_charge_losses.append(
                    (predicted_charge - expected).abs()
                    * float(constraint_policy.mother_charge_soft_weight)
                )
    zero = output.object_logits.sum() * 0.0
    if unrepresentable_target_counts is not None:
        if len(unrepresentable_target_counts) != output.object_logits.shape[0]:
            raise ValueError("one unrepresentable target count is required per event")
        for batch_index, missing in enumerate(unrepresentable_target_counts):
            if missing <= 0:
                continue
            unmatched = (~object_targets[batch_index].bool()).nonzero(
                as_tuple=False
            ).flatten()
            count = min(int(missing), int(unmatched.numel()))
            if count:
                uncertain = output.object_logits[batch_index, unmatched].topk(count).indices
                object_loss_mask[batch_index, unmatched[uncertain]] = False
    components = {
        "object": focal_binary_cross_entropy_with_logits(
            output.object_logits,
            object_targets,
            positive_weight=object_positive_weight,
            gamma=object_focal_gamma,
            mask=object_loss_mask,
        ),
        "type": torch.stack(type_losses).mean() if type_losses else zero,
        "pointer": torch.stack(pointer_losses).mean() if pointer_losses else zero,
        "cardinality": torch.stack(cardinality_losses).mean() if cardinality_losses else zero,
        "confidence": F.binary_cross_entropy_with_logits(
            output.confidence_logits,
            confidence_targets,
        ),
        "physics": (torch.stack(p4_losses).mean() + torch.stack(charge_losses).mean()) if p4_losses else zero,
        "source_conflict": (
            source_conflict_penalty(
                output.pointer_logits,
                batch["source_conflict_matrix"],
                object_logits=output.object_logits,
                query_mask=object_loss_mask,
            )
            if "source_conflict_matrix" in batch
            and (
                constraint_policy is None
                or constraint_policy.reject_recursive_source_conflicts
            )
            else zero
        ),
        "mother_charge": (
            torch.stack(mother_charge_losses).mean() if mother_charge_losses else zero
        ),
        "query_repulsion": query_proposal_repulsion_loss(
            output.pointer_logits,
            active_query_mask=object_targets.bool(),
            matched_pointer_targets=matched_pointer_targets,
        ),
    }
    total = sum(components[name] * weights[name] for name in components)
    return LevelLossOutput(
        total=total,
        components=components,
        matches=all_matches,
        confidence_targets=confidence_targets,
    )


def query_proposal_repulsion_loss(
    pointer_logits: torch.Tensor,
    *,
    active_query_mask: torch.Tensor,
    matched_pointer_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize duplicate disjoint proposals without imposing slot ordering."""

    if pointer_logits.shape[1] < 2:
        return pointer_logits.sum() * 0.0
    if active_query_mask.shape != pointer_logits.shape[:2]:
        raise ValueError("active_query_mask must have shape [batch, queries]")
    proposals = torch.sigmoid(pointer_logits)
    normalized = F.normalize(proposals, dim=-1, eps=1e-8)
    similarity = torch.einsum("bqn,bkn->bqk", normalized, normalized)
    off_diagonal = ~torch.eye(
        pointer_logits.shape[1], dtype=torch.bool, device=pointer_logits.device
    ).unsqueeze(0)
    selected = (
        off_diagonal
        & active_query_mask[:, :, None]
        & active_query_mask[:, None, :]
    )
    if matched_pointer_targets is not None:
        if matched_pointer_targets.shape != pointer_logits.shape:
            raise ValueError("matched_pointer_targets must match pointer_logits")
        target_overlap = torch.einsum(
            "bqn,bkn->bqk",
            matched_pointer_targets.to(pointer_logits.dtype),
            matched_pointer_targets.to(pointer_logits.dtype),
        ) > 0
        selected &= ~target_overlap
    denominator = selected.sum()
    normalized_loss = similarity.masked_select(selected).sum() / denominator.clamp_min(1).to(
        similarity.dtype
    )
    return torch.where(
        denominator > 0,
        normalized_loss,
        pointer_logits.sum() * 0.0,
    )


def focal_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    gamma: float = 0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted focal BCE normalized per proposal/mother collection."""

    targets = targets.to(logits.dtype)
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    class_weight = 1 + (positive_weight - 1) * targets
    loss = base * (1 - p_t).pow(gamma) * class_weight
    if mask is None:
        return loss.mean()
    selected = mask.bool()
    return loss[selected].mean() if selected.any() else logits.sum() * 0.0


def confidence_calibration_metrics(
    confidence_logits: torch.Tensor,
    confidence_targets: torch.Tensor,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    probability = torch.sigmoid(confidence_logits).detach()
    target = confidence_targets.detach()
    brier = (probability - target).square().mean()
    ece = probability.new_tensor(0.0)
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        selected = (probability >= lower) & (
            probability <= upper if index == n_bins - 1 else probability < upper
        )
        if selected.any():
            ece = ece + selected.float().mean() * (
                probability[selected].mean() - target[selected].mean()
            ).abs()
    return {"brier_score": float(brier.cpu()), "expected_calibration_error": float(ece.cpu())}
