"""Configurable FSP-to-predicted-composite pretraining curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from hypertagging.preprocessing.pid_filter import PDG_TOKENS


class PretrainingStage(str, Enum):
    FSP_ONLY = "fsp_only"
    TRUTH_GUIDED_MULTILEVEL = "truth_guided_multilevel"
    CORRUPTED_COMPOSITES = "corrupted_composites"


@dataclass(frozen=True)
class CurriculumBatch:
    batch: dict[str, torch.Tensor]
    stage: PretrainingStage
    corrupted_node_mask: torch.Tensor
    corruption_code: torch.Tensor
    hard_negative_pairs: torch.Tensor


def build_curriculum_batch(
    batch: dict[str, torch.Tensor],
    stage: PretrainingStage | str,
    *,
    seed: int = 0,
    corruption_probability: float = 0.5,
) -> CurriculumBatch:
    """Build one curriculum view without exposing truth composites in Stage 1."""

    stage = PretrainingStage(stage)
    output = {name: value.clone() for name, value in batch.items()}
    original_mask = output["node_mask"].bool()
    corrupted = torch.zeros_like(original_mask)
    corruption_code = torch.zeros_like(output["level_ids"])
    if stage is PretrainingStage.FSP_ONLY:
        output["node_mask"] = original_mask & (output["level_ids"] == 0)
        output["active"] = output["node_mask"].clone()
    elif stage is PretrainingStage.CORRUPTED_COMPOSITES:
        generator = torch.Generator(device=output["p4"].device)
        generator.manual_seed(seed)
        candidates = original_mask & (output["level_ids"] > 0)
        draws = torch.rand(candidates.shape, generator=generator, device=candidates.device)
        corrupted = candidates & (draws < corruption_probability)
        _apply_composite_corruptions(output, corrupted, generator)
        # Codes: 1 missing daughter, 2 wrong daughter, 3 wrong type,
        # 4 shared-leaf conflict.  Deterministic cycling gives tiny batches
        # coverage without relying on chance.
        indices = torch.nonzero(corrupted, as_tuple=False)
        for ordinal, (batch_index, node_index) in enumerate(indices.tolist()):
            corruption_code[batch_index, node_index] = ordinal % 4 + 1
    hard_negatives = relation_aware_hard_negative_pairs(output)
    return CurriculumBatch(
        batch=output,
        stage=stage,
        corrupted_node_mask=corrupted,
        corruption_code=corruption_code,
        hard_negative_pairs=hard_negatives,
    )


def _apply_composite_corruptions(
    batch: dict[str, torch.Tensor],
    corrupted: torch.Tensor,
    generator: torch.Generator,
) -> None:
    adjacency = batch["daughter_adjacency"]
    for batch_index, node_index in torch.nonzero(corrupted, as_tuple=False).tolist():
        daughters = adjacency[batch_index, node_index].nonzero(as_tuple=False).flatten()
        context = (
            batch["node_mask"][batch_index]
            & (batch["level_ids"][batch_index] < batch["level_ids"][batch_index, node_index])
        ).nonzero(as_tuple=False).flatten()
        mode = (batch_index + node_index) % 4
        if mode == 0 and daughters.numel():
            adjacency[batch_index, node_index, daughters[-1]] = False
        elif mode == 1 and context.numel():
            alternatives = context[
                ~torch.isin(context, daughters)
            ]
            if alternatives.numel():
                if daughters.numel():
                    adjacency[batch_index, node_index, daughters[0]] = False
                adjacency[batch_index, node_index, alternatives[0]] = True
        elif mode == 2:
            current = int(batch["pid_labels"][batch_index, node_index])
            batch["pid_labels"][batch_index, node_index] = (
                current + 1
            ) % len(PDG_TOKENS)
        elif mode == 3 and context.numel():
            # Reuse a source already present in a sibling hypothesis when one
            # is available. Bool adjacency cannot represent duplicate direct
            # pointers, so this is the explicit shared-leaf conflict variant.
            sibling_rows = adjacency[batch_index].any(dim=-1).nonzero(as_tuple=False).flatten()
            for sibling in sibling_rows.tolist():
                if sibling == node_index:
                    continue
                shared = adjacency[batch_index, sibling].nonzero(as_tuple=False).flatten()
                if shared.numel():
                    adjacency[batch_index, node_index, shared[0]] = True
                    break


def relation_aware_hard_negative_pairs(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Select nearest-p4 non-sibling pairs among valid contextual nodes."""

    mask = batch["node_mask"].bool()
    p3 = batch["p4"][..., :3]
    parent = batch["parent_ids"]
    pairs: list[list[int]] = []
    for batch_index in range(mask.shape[0]):
        valid = mask[batch_index].nonzero(as_tuple=False).flatten()
        for left in valid.tolist():
            candidates = valid[
                (valid != left)
                & (
                    (parent[batch_index, valid] != parent[batch_index, left])
                    | (parent[batch_index, left] < 0)
                )
            ]
            if candidates.numel() == 0:
                continue
            distances = torch.linalg.vector_norm(
                p3[batch_index, candidates] - p3[batch_index, left],
                dim=-1,
            )
            right = int(candidates[distances.argmin()])
            pairs.append([batch_index, left, right])
    return torch.tensor(pairs, dtype=torch.long, device=mask.device).reshape(-1, 3)


__all__ = [
    "CurriculumBatch",
    "PretrainingStage",
    "build_curriculum_batch",
    "relation_aware_hard_negative_pairs",
]
