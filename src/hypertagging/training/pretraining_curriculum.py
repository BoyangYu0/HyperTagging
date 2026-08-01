"""Configurable FSP-to-predicted-composite pretraining curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.pid_state import (
    COMPOSITE_TYPE_SOURCE_TO_ID,
    hard_daughter_pid_histograms,
)


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
    hard_negative_relation_classes: torch.Tensor
    structural_positive_mask: torch.Tensor
    corruption_objective: str
    relation_input_policy: str


def build_curriculum_batch(
    batch: dict[str, torch.Tensor],
    stage: PretrainingStage | str,
    *,
    seed: int = 0,
    corruption_probability: float = 0.5,
    corruption_objective: str = "invalid_candidate",
    truth_guided_structural_relation_inputs: bool = False,
) -> CurriculumBatch:
    """Build one curriculum view without exposing truth composites in Stage 1."""

    stage = PretrainingStage(stage)
    if corruption_objective not in {"denoising", "invalid_candidate"}:
        raise ValueError("corruption_objective must be denoising or invalid_candidate")
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
        corruption_code, corrupted = _apply_composite_corruptions(
            output, corrupted, generator
        )
        output["runtime_composite_type_source_ids"] = torch.where(
            corrupted,
            torch.full_like(
                output["level_ids"],
                COMPOSITE_TYPE_SOURCE_TO_ID["corrupted"],
            ),
            output.get(
                "runtime_composite_type_source_ids",
                torch.full_like(
                    output["level_ids"],
                    COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"],
                ),
            ),
        )
        _rebuild_corrupted_derived_fields(output)
    output["curriculum_attention_mask"] = curriculum_attention_mask(output, stage)
    relation_input_policy = "inference_physical_relation_features"
    if (
        truth_guided_structural_relation_inputs
        and stage is PretrainingStage.TRUTH_GUIDED_MULTILEVEL
    ):
        visible = output["node_mask"].bool()
        output["current_reconstructed_ancestor_descendant_relation"] = (
            output["ancestor_descendant_relation"].bool()
            & visible[:, :, None]
            & visible[:, None, :]
        )
        relation_input_policy = "current_reconstructed_tree_state_features"
    else:
        count = output["node_mask"].shape[1]
        output["current_reconstructed_ancestor_descendant_relation"] = torch.zeros(
            (output["node_mask"].shape[0], count, count),
            dtype=torch.bool,
            device=output["node_mask"].device,
        )
    hard_negatives, hard_negative_classes = hard_negative_pairs_with_classes(output)
    return CurriculumBatch(
        batch=output,
        stage=stage,
        corrupted_node_mask=corrupted,
        corruption_code=corruption_code,
        hard_negative_pairs=hard_negatives,
        hard_negative_relation_classes=hard_negative_classes,
        structural_positive_mask=(
            output["node_mask"]
            if corruption_objective == "denoising"
            else output["node_mask"] & ~corrupted
        ),
        corruption_objective=corruption_objective,
        relation_input_policy=relation_input_policy,
    )


def curriculum_attention_mask(
    batch: dict[str, torch.Tensor],
    stage: PretrainingStage | str,
) -> torch.Tensor:
    """FSP-full or level-causal mask; lower nodes never see future parents."""

    stage = PretrainingStage(stage)
    valid = batch["node_mask"].bool()
    if stage is PretrainingStage.FSP_ONLY:
        leaves = valid & (batch["level_ids"] == 0)
        return leaves[:, :, None] & leaves[:, None, :]
    levels = batch["level_ids"]
    return (
        valid[:, :, None]
        & valid[:, None, :]
        & (levels[:, None, :] <= levels[:, :, None])
    )


def _apply_composite_corruptions(
    batch: dict[str, torch.Tensor],
    corrupted: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjacency = batch["daughter_adjacency"]
    codes = torch.zeros_like(batch["level_ids"])
    applied = torch.zeros_like(corrupted)
    for batch_index, node_index in torch.nonzero(corrupted, as_tuple=False).tolist():
        before_adjacency = adjacency[batch_index, node_index].clone()
        before_type = batch["pid_labels"][batch_index, node_index].clone()
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
        changed = (
            not torch.equal(before_adjacency, adjacency[batch_index, node_index])
            or not torch.equal(before_type, batch["pid_labels"][batch_index, node_index])
        )
        if changed:
            applied[batch_index, node_index] = True
            codes[batch_index, node_index] = mode + 1
    return codes, applied


def _rebuild_corrupted_derived_fields(batch: dict[str, torch.Tensor]) -> None:
    """Recompute every model-input field derived from corrupted adjacency/type."""

    adjacency = batch["daughter_adjacency"].bool()
    node_mask = batch["node_mask"].bool()
    p4 = batch["p4"].clone()
    charge = batch["charge"].clone()
    source_mask = batch["recursive_leaf_source_mask"].clone()
    for level in sorted(
        {
            int(value)
            for value in batch["level_ids"][node_mask].detach().cpu().tolist()
            if int(value) > 0
        }
    ):
        mothers = node_mask & (batch["level_ids"] == level)
        summed_p4 = torch.einsum("bmn,bnf->bmf", adjacency.float(), p4)
        summed_charge = torch.einsum("bmn,bn->bm", adjacency.float(), charge)
        p4 = torch.where(mothers.unsqueeze(-1), summed_p4, p4)
        charge = torch.where(mothers, summed_charge, charge)
        union = torch.einsum(
            "bmn,bns->bms", adjacency.to(torch.int32), source_mask.to(torch.int32)
        ) > 0
        source_mask = torch.where(mothers.unsqueeze(-1), union, source_mask)
    batch["p4"] = p4
    batch["charge"] = charge
    batch["recursive_leaf_source_mask"] = source_mask
    input_tokens = batch["pid_labels"]
    histogram = hard_daughter_pid_histograms(input_tokens, adjacency)
    batch["daughter_input_pid_histogram"] = histogram
    batch["daughter_pid_histogram"] = histogram
    available = adjacency.any(dim=-1) & node_mask
    batch["daughter_input_pid_histogram_available"] = available
    batch["daughter_pid_histogram_available"] = available
    composite = batch["composite_features"].clone()
    composite[..., :4] = torch.einsum("bmn,bnf->bmf", adjacency.float(), p4)
    if composite.shape[-1] > 4:
        composite[..., 4] = torch.einsum("bmn,bn->bm", adjacency.float(), charge)
    if composite.shape[-1] > 5:
        composite[..., 5] = adjacency.sum(dim=-1)
    batch["composite_features"] = composite
    common = batch["common_features"].clone()
    common[..., :4] = p4
    mass2 = p4[..., 3].square() - p4[..., :3].square().sum(dim=-1)
    common[..., 4] = mass2.clamp_min(0).sqrt()
    common[..., 5] = charge
    batch["common_features"] = common
    overlap = torch.einsum(
        "bns,bms->bnm", source_mask.to(torch.int32), source_mask.to(torch.int32)
    ) > 0
    diagonal = torch.eye(overlap.shape[-1], dtype=torch.bool, device=overlap.device)
    batch["source_conflict_matrix"] = overlap & ~diagonal.unsqueeze(0)


def relation_aware_hard_negative_pairs(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return hard_negative_pairs_with_classes(batch)[0]


def hard_negative_pairs_with_classes(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select plausible pairs only from explicit negative tree relations."""

    mask = batch["node_mask"].bool()
    p3 = batch["p4"][..., :3]
    parent = batch["parent_ids"]
    pairs: list[list[int]] = []
    classes: list[int] = []
    for batch_index in range(mask.shape[0]):
        valid = mask[batch_index].nonzero(as_tuple=False).flatten()
        for left in valid.tolist():
            eligible = []
            relation_classes = []
            left_ancestors = _ancestor_positions(parent[batch_index], left)
            for right in valid.tolist():
                if right == left:
                    continue
                right_ancestors = _ancestor_positions(parent[batch_index], right)
                if right in left_ancestors or left in right_ancestors:
                    continue
                if (
                    int(parent[batch_index, left]) >= 0
                    and parent[batch_index, left] == parent[batch_index, right]
                ):
                    continue
                left_side = int(batch["b_side"][batch_index, left])
                right_side = int(batch["b_side"][batch_index, right])
                if left_side >= 0 and right_side >= 0 and left_side != right_side:
                    eligible.append(right)
                    relation_classes.append(1)  # different B side
                    continue
                if not (left_ancestors & right_ancestors):
                    eligible.append(right)
                    relation_classes.append(2)  # unrelated retained roots
            candidates = torch.tensor(eligible, device=valid.device, dtype=torch.long)
            if candidates.numel() == 0:
                continue
            distances = torch.linalg.vector_norm(
                p3[batch_index, candidates] - p3[batch_index, left],
                dim=-1,
            )
            right = int(candidates[distances.argmin()])
            pairs.append([batch_index, left, right])
            classes.append(relation_classes[int(distances.argmin())])
    return (
        torch.tensor(pairs, dtype=torch.long, device=mask.device).reshape(-1, 3),
        torch.tensor(classes, dtype=torch.long, device=mask.device),
    )


def _ancestor_positions(parent: torch.Tensor, node: int) -> set[int]:
    ancestors: set[int] = set()
    current = int(parent[node])
    while current >= 0 and current not in ancestors:
        ancestors.add(current)
        current = int(parent[current])
    return ancestors


__all__ = [
    "CurriculumBatch",
    "PretrainingStage",
    "build_curriculum_batch",
    "curriculum_attention_mask",
    "relation_aware_hard_negative_pairs",
    "hard_negative_pairs_with_classes",
]
