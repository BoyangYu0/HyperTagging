"""Losses for level-autoregressive mother set reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hypertagging.losses.physics import charge_consistency_loss, p4_sum_consistency_loss
from hypertagging.losses.set_matching import hungarian_assignment, matching_cost
from hypertagging.models.mother_pointer import MotherPointerOutput
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
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    target_types = []
    target_masks = []
    target_p4 = []
    target_charge = []
    for batch_index in range(batch["node_features"].shape[0]):
        nodes = (batch["node_mask"][batch_index] & (batch["level_ids"][batch_index] == target_level)).nonzero(as_tuple=False).flatten()
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
    matching_production: bool = False,
    physics_component_scales: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> LevelLossOutput:
    weights = {
        "object": 1.0,
        "type": 1.0,
        "pointer": 1.0,
        "cardinality": 0.2,
        "confidence": 0.2,
        "physics": 0.1,
        **(weights or {}),
    }
    target_types, target_masks, target_p4, target_charge = targets_for_level(
        batch,
        target_level,
        min_daughters=min_daughters,
    )
    for event_index, types in enumerate(target_types):
        if types.numel() > output.object_logits.shape[1]:
            raise OverflowError(
                f"event {event_index} target level {target_level} has {types.numel()} truth "
                f"mothers but decoder has {output.object_logits.shape[1]} queries"
            )
    object_targets = torch.zeros_like(output.object_logits)
    confidence_targets = torch.zeros_like(output.confidence_logits)
    type_losses = []
    pointer_losses = []
    cardinality_losses = []
    p4_losses = []
    charge_losses = []
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
            intersection = torch.minimum(predicted_pointer, truth_pointer).sum()
            union = torch.maximum(predicted_pointer, truth_pointer).sum().clamp_min(1e-6)
            type_probability = torch.softmax(
                output.type_logits[batch_index, query_id],
                dim=-1,
            )[types[target_id]]
            confidence_targets[batch_index, query_id] = (
                intersection / union * type_probability.detach()
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
    zero = output.object_logits.sum() * 0.0
    components = {
        "object": focal_binary_cross_entropy_with_logits(
            output.object_logits,
            object_targets,
            positive_weight=object_positive_weight,
            gamma=object_focal_gamma,
        ),
        "type": torch.stack(type_losses).mean() if type_losses else zero,
        "pointer": torch.stack(pointer_losses).mean() if pointer_losses else zero,
        "cardinality": torch.stack(cardinality_losses).mean() if cardinality_losses else zero,
        "confidence": F.binary_cross_entropy_with_logits(
            output.confidence_logits,
            confidence_targets,
        ),
        "physics": (torch.stack(p4_losses).mean() + torch.stack(charge_losses).mean()) if p4_losses else zero,
    }
    total = sum(components[name] * weights[name] for name in components)
    return LevelLossOutput(
        total=total,
        components=components,
        matches=all_matches,
        confidence_targets=confidence_targets,
    )


def focal_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Weighted focal BCE normalized per proposal/mother collection."""

    targets = targets.to(logits.dtype)
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    class_weight = 1 + (positive_weight - 1) * targets
    loss = base * (1 - p_t).pow(gamma) * class_weight
    return loss.mean()


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
