"""Runtime, truth-free PID state used by every reconstruction context mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from hypertagging.preprocessing.pid_filter import PDG_TOKENS, TOKENIZE_DICT
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.reconstruction.kinematics import (
    PARTICLE_CHARGES,
    hard_track_p4_from_pid_token,
    soft_track_p4_from_pid_logits,
    track_p4_from_pid_probabilities,
)


@dataclass(frozen=True)
class RuntimePIDState:
    probabilities: torch.Tensor
    current_tokens: torch.Tensor
    available: torch.Tensor
    p4: torch.Tensor
    daughter_input_histograms: torch.Tensor
    daughter_histogram_available: torch.Tensor


COMPOSITE_TYPE_SOURCES = (
    "input_fixed",
    "truth_teacher_forced",
    "predicted",
    "corrupted",
    "unknown",
)
COMPOSITE_TYPE_SOURCE_TO_ID = {
    name: index for index, name in enumerate(COMPOSITE_TYPE_SOURCES)
}


def charge_compatible_pid_mask(charge: torch.Tensor) -> torch.Tensor:
    """Return charged-stable reduced-token support for each reconstructed charge."""

    output = torch.zeros((*charge.shape, len(PDG_TOKENS)), dtype=torch.bool, device=charge.device)
    for pdg, particle_charge in PARTICLE_CHARGES.items():
        token = TOKENIZE_DICT.get(pdg)
        if token is not None:
            output[..., token] = torch.where(
                charge == float(particle_charge),
                torch.ones_like(charge, dtype=torch.bool),
                output[..., token],
            )
    return output


def soft_daughter_pid_histograms(
    probabilities: torch.Tensor,
    daughter_adjacency: torch.Tensor,
) -> torch.Tensor:
    """Count-valued differentiable daughter PID histogram."""

    if probabilities.shape[:-1] != daughter_adjacency.shape[:2]:
        raise ValueError("PID probabilities and daughter adjacency node axes differ")
    if probabilities.shape[-1] != len(PDG_TOKENS):
        raise ValueError("PID probability width must equal the reduced vocabulary")
    return torch.einsum(
        "bmn,bnc->bmc", daughter_adjacency.to(probabilities.dtype), probabilities
    )


def hard_daughter_pid_histograms(
    tokens: torch.Tensor,
    daughter_adjacency: torch.Tensor,
) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(
        tokens, num_classes=len(PDG_TOKENS)
    ).to(torch.float32)
    return soft_daughter_pid_histograms(one_hot, daughter_adjacency)


def rebuild_runtime_pid_state(
    batch: Mapping[str, torch.Tensor],
    leaf_pid_logits: torch.Tensor,
    *,
    hard: bool = False,
    refine_fixed_hypothesis: bool = False,
    mode: str = "soft_expectation",
    temperature: float = 1.0,
) -> RuntimePIDState:
    """Build current PID/p4/histograms without reading any truth target field."""

    if hard:
        mode = "hard"
    if mode not in {
        "soft_expectation", "temperature_softmax", "straight_through_hard", "hard"
    }:
        raise ValueError(f"unknown PID kinematics mode: {mode}")
    if temperature <= 0:
        raise ValueError("PID softmax temperature must be positive")
    node_mask = batch["node_mask"].bool()
    modes = batch["leaf_kinematics_mode_ids"]
    input_tokens = batch["pid_labels"]
    one_hot = torch.nn.functional.one_hot(
        input_tokens, num_classes=len(PDG_TOKENS)
    ).to(leaf_pid_logits.dtype)
    probabilities = one_hot.clone()
    source_ids = batch.get(
        "runtime_composite_type_source_ids",
        torch.full_like(input_tokens, COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"]),
    )
    composite = node_mask & (
        modes == LEAF_MODE_TO_ID["composite"]
    )
    teacher = composite & (
        source_ids == COMPOSITE_TYPE_SOURCE_TO_ID["truth_teacher_forced"]
    )
    if teacher.any():
        if "pid_target_labels" not in batch:
            raise ValueError("teacher-forced composite types require pid_target_labels")
        probabilities = torch.where(
            teacher.unsqueeze(-1),
            torch.nn.functional.one_hot(
                batch["pid_target_labels"], num_classes=len(PDG_TOKENS)
            ).to(probabilities.dtype),
            probabilities,
        )
    generated = composite & (
        (source_ids == COMPOSITE_TYPE_SOURCE_TO_ID["predicted"])
        | (source_ids == COMPOSITE_TYPE_SOURCE_TO_ID["corrupted"])
    )
    if generated.any():
        generated_tokens = batch.get("current_pid_tokens", input_tokens)
        probabilities = torch.where(
            generated.unsqueeze(-1),
            torch.nn.functional.one_hot(
                generated_tokens, num_classes=len(PDG_TOKENS)
            ).to(probabilities.dtype),
            probabilities,
        )
    unknown = composite & (
        source_ids == COMPOSITE_TYPE_SOURCE_TO_ID["unknown"]
    )
    if unknown.any():
        unknown_one_hot = torch.zeros_like(probabilities)
        unknown_one_hot[..., 0] = 1
        probabilities = torch.where(unknown.unsqueeze(-1), unknown_one_hot, probabilities)
    raw = node_mask & (
        modes == LEAF_MODE_TO_ID["raw_track_predicted_pid"]
    )
    compatible = charge_compatible_pid_mask(batch["charge"])
    if raw.any() and (~compatible.any(dim=-1) & raw).any():
        bad = torch.nonzero(~compatible.any(dim=-1) & raw, as_tuple=False).tolist()
        raise ValueError(
            "raw track has no charge-compatible stable PID species at positions "
            f"{bad}"
        )
    raw_logits = leaf_pid_logits.masked_fill(~compatible, -1e4)
    effective_temperature = temperature if mode == "temperature_softmax" else 1.0
    soft_probabilities = torch.softmax(raw_logits / effective_temperature, dim=-1)
    hard_probabilities = torch.nn.functional.one_hot(
        soft_probabilities.argmax(dim=-1), num_classes=len(PDG_TOKENS)
    ).to(soft_probabilities.dtype)
    if mode == "straight_through_hard":
        raw_probabilities = (
            hard_probabilities + soft_probabilities - soft_probabilities.detach()
        )
    elif mode == "hard":
        raw_probabilities = hard_probabilities
    else:
        raw_probabilities = soft_probabilities
    probabilities = torch.where(raw.unsqueeze(-1), raw_probabilities, probabilities)
    if refine_fixed_hypothesis:
        fixed = node_mask & (
            modes == LEAF_MODE_TO_ID["fixed_hypothesis_candidate"]
        )
        probabilities = torch.where(
            fixed.unsqueeze(-1), torch.softmax(leaf_pid_logits, dim=-1), probabilities
        )
    current_tokens = probabilities.argmax(dim=-1)
    available = node_mask & (probabilities.sum(dim=-1) > 0)
    p4 = batch["p4"].clone()
    if raw.any():
        if mode == "hard":
            p4[raw] = hard_track_p4_from_pid_token(
                batch["p4"][raw, :3], current_tokens[raw]
            )
        elif mode == "straight_through_hard":
            p4[raw] = track_p4_from_pid_probabilities(
                batch["p4"][raw, :3], raw_probabilities[raw]
            )
        else:
            for sign in (-1, 1):
                selected = raw & (
                    (batch["charge"] < 0) if sign < 0 else (batch["charge"] > 0)
                )
                if not selected.any():
                    continue
                allowed = tuple(
                    TOKENIZE_DICT[pdg]
                    for pdg, particle_charge in PARTICLE_CHARGES.items()
                    if particle_charge == sign
                )
                p4[selected] = soft_track_p4_from_pid_logits(
                    batch["p4"][selected, :3],
                    raw_logits[selected] / effective_temperature,
                    allowed_tokens=allowed,
                )
    levels = sorted(
        {
            int(value)
            for value in batch["level_ids"][node_mask].detach().cpu().tolist()
            if int(value) > 0
        }
    )
    adjacency = batch["daughter_adjacency"].to(p4.dtype)
    for level in levels:
        mothers = node_mask & (batch["level_ids"] == level)
        summed = torch.einsum("bmn,bnf->bmf", adjacency, p4)
        p4 = torch.where(mothers.unsqueeze(-1), summed, p4)
    histograms = soft_daughter_pid_histograms(
        probabilities, batch["daughter_adjacency"]
    )
    histogram_available = batch["daughter_adjacency"].any(dim=-1) & node_mask
    return RuntimePIDState(
        probabilities=probabilities,
        current_tokens=current_tokens,
        available=available,
        p4=p4,
        daughter_input_histograms=histograms,
        daughter_histogram_available=histogram_available,
    )


__all__ = [
    "RuntimePIDState",
    "charge_compatible_pid_mask",
    "hard_daughter_pid_histograms",
    "rebuild_runtime_pid_state",
    "soft_daughter_pid_histograms",
    "COMPOSITE_TYPE_SOURCES",
    "COMPOSITE_TYPE_SOURCE_TO_ID",
]
