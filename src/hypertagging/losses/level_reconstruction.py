"""Losses for level-autoregressive mother set reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hypertagging.losses.physics import charge_consistency_loss, p4_sum_consistency_loss
from hypertagging.losses.set_matching import hungarian_or_greedy, matching_cost
from hypertagging.models.mother_pointer import MotherPointerOutput


@dataclass(frozen=True)
class LevelLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    matches: list[list[tuple[int, int]]]


def targets_for_level(batch: dict[str, torch.Tensor], target_level: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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
        target_types.append(batch["pid_labels"][batch_index, nodes].abs().clamp(max=4095))
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
) -> LevelLossOutput:
    weights = {"object": 1.0, "type": 1.0, "pointer": 1.0, "cardinality": 0.2, "physics": 0.1, **(weights or {})}
    target_types, target_masks, target_p4, target_charge = targets_for_level(batch, target_level)
    object_targets = torch.zeros_like(output.object_logits)
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
        )
        matches = hungarian_or_greedy(cost)
        all_matches.append(matches)
        for query_id, target_id in matches:
            object_targets[batch_index, query_id] = 1.0
            type_losses.append(F.cross_entropy(output.type_logits[batch_index, query_id][None], types[target_id][None]))
            pointer_losses.append(
                F.binary_cross_entropy_with_logits(
                    output.pointer_logits[batch_index, query_id, context],
                    target_masks[batch_index][target_id].float(),
                )
            )
            cardinality_losses.append(
                F.cross_entropy(
                    output.cardinality_logits[batch_index, query_id][None],
                    target_masks[batch_index][target_id].sum().long().clamp(max=output.cardinality_logits.shape[-1] - 1)[None],
                )
            )
            p4_losses.append(
                p4_sum_consistency_loss(
                    output.pointer_logits[batch_index, query_id, context][None, None],
                    context_p4[None],
                    target_p4[batch_index][target_id][None, None],
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
        "object": F.binary_cross_entropy_with_logits(output.object_logits, object_targets),
        "type": torch.stack(type_losses).mean() if type_losses else zero,
        "pointer": torch.stack(pointer_losses).mean() if pointer_losses else zero,
        "cardinality": torch.stack(cardinality_losses).mean() if cardinality_losses else zero,
        "physics": (torch.stack(p4_losses).mean() + torch.stack(charge_losses).mean()) if p4_losses else zero,
    }
    total = sum(components[name] * weights[name] for name in components)
    return LevelLossOutput(total=total, components=components, matches=all_matches)
