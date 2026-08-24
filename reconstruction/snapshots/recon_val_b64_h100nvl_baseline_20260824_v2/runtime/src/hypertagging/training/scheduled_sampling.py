"""Scheduled-sampling utilities for level reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ScheduledSamplingConfig:
    start: float = 0.0
    end: float = 0.5
    warmup_steps: int = 1000


def scheduled_sampling_probability(step: int, config: ScheduledSamplingConfig | None = None) -> float:
    config = config or ScheduledSamplingConfig()
    if config.warmup_steps <= 0:
        return config.end
    frac = min(max(step, 0) / config.warmup_steps, 1.0)
    return config.start + (config.end - config.start) * frac


@dataclass(frozen=True)
class TeacherForcingSchedule:
    kind: str = "linear"
    start_probability: float = 1.0
    end_probability: float = 0.2
    duration_steps: int = 1000
    inverse_sigmoid_k: float = 100.0

    def probability(self, step: int) -> float:
        if step >= self.duration_steps > 0 and self.kind != "constant":
            return float(self.end_probability)
        if self.duration_steps <= 0:
            return float(self.end_probability)
        progress = min(max(step, 0) / self.duration_steps, 1.0)
        if self.kind == "constant":
            value = self.start_probability
        elif self.kind == "linear":
            value = self.start_probability + (
                self.end_probability - self.start_probability
            ) * progress
        elif self.kind == "cosine":
            weight = 0.5 * (1 + math.cos(math.pi * progress))
            value = self.end_probability + (
                self.start_probability - self.end_probability
            ) * weight
        elif self.kind == "inverse_sigmoid":
            raw = self.inverse_sigmoid_k / (
                self.inverse_sigmoid_k + math.exp(progress * 10)
            )
            start_raw = self.inverse_sigmoid_k / (self.inverse_sigmoid_k + 1)
            end_raw = self.inverse_sigmoid_k / (
                self.inverse_sigmoid_k + math.exp(10)
            )
            scaled = (raw - end_raw) / max(start_raw - end_raw, 1e-12)
            value = self.end_probability + (
                self.start_probability - self.end_probability
            ) * scaled
        else:
            raise ValueError(f"unknown teacher forcing schedule: {self.kind}")
        return float(min(max(value, 0.0), 1.0))

    def sample(
        self,
        count: int,
        *,
        step: int,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed + step)
        return (
            torch.rand(count, generator=generator, device=device)
            < self.probability(step)
        )


def combine_sampled_context_losses(
    teacher_losses: torch.Tensor,
    predicted_losses: torch.Tensor,
    choose_teacher: torch.Tensor,
    *,
    auxiliary_teacher_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Choose one primary loss per event, with optional auxiliary teacher loss."""

    if (
        teacher_losses.shape != predicted_losses.shape
        or choose_teacher.shape != teacher_losses.shape
    ):
        raise ValueError("context losses and choices must have identical shapes")
    primary = torch.where(choose_teacher, teacher_losses, predicted_losses)
    auxiliary = (
        teacher_losses[~choose_teacher].mean()
        if (~choose_teacher).any()
        else teacher_losses.sum() * 0.0
    )
    total = primary.mean() + float(auxiliary_teacher_weight) * auxiliary
    return total, {
        "sampled_teacher_count": int(choose_teacher.sum()),
        "sampled_predicted_count": int((~choose_teacher).sum()),
        "primary_teacher_loss": float(
            teacher_losses[choose_teacher].detach().mean().cpu()
            if choose_teacher.any()
            else 0.0
        ),
        "primary_predicted_loss": float(
            predicted_losses[~choose_teacher].detach().mean().cpu()
            if (~choose_teacher).any()
            else 0.0
        ),
        "auxiliary_teacher_loss": float(auxiliary.detach().cpu()),
    }


@dataclass(frozen=True)
class ContextAlignment:
    predicted_to_truth: torch.Tensor
    representable: torch.Tensor
    source_jaccard: torch.Tensor


@dataclass(frozen=True)
class AlignedLevelTargets:
    target_override: tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]
    truth_target_count: int
    representable_count: int


@dataclass(frozen=True)
class UnrepresentableTargetDecision:
    use_teacher_context: bool = False
    skip_event_level: bool = False
    use_representable_subset: bool = False
    add_recovery_objective: bool = False


def resolve_unrepresentable_target_policy(
    policy: str,
    *,
    truth_target_count: int,
    representable_target_count: int,
) -> UnrepresentableTargetDecision:
    """Resolve missing-context targets without ever inventing no-object labels."""

    if representable_target_count >= truth_target_count:
        return UnrepresentableTargetDecision(use_representable_subset=True)
    if policy == "fallback_teacher":
        return UnrepresentableTargetDecision(use_teacher_context=True)
    if policy == "skip_event_level":
        return UnrepresentableTargetDecision(skip_event_level=True)
    if policy == "masked_representable_only":
        return UnrepresentableTargetDecision(use_representable_subset=True)
    if policy == "recovery_objective":
        return UnrepresentableTargetDecision(
            use_representable_subset=True, add_recovery_objective=True
        )
    raise ValueError(f"unknown unrepresentable target policy: {policy}")


def align_context_by_recursive_sources(
    predicted_sources: torch.Tensor,
    truth_sources: torch.Tensor,
) -> ContextAlignment:
    """Align without node IDs; exact source sets define representability."""

    predicted = predicted_sources.bool()
    truth = truth_sources.bool()
    intersection = (predicted[:, None] & truth[None]).sum(dim=-1).float()
    union = (predicted[:, None] | truth[None]).sum(dim=-1).clamp_min(1).float()
    jaccard = intersection / union
    best_score, best_truth = jaccard.max(dim=-1)
    exact = best_score == 1
    mapping = torch.where(exact, best_truth, torch.full_like(best_truth, -1))
    truth_representable = torch.zeros(truth.shape[0], dtype=torch.bool, device=truth.device)
    if exact.any():
        truth_representable[best_truth[exact]] = True
    return ContextAlignment(mapping, truth_representable, best_score)


def aligned_level_targets(
    truth_batch: dict[str, torch.Tensor],
    predicted_context: dict[str, torch.Tensor],
    *,
    target_level: int,
    target_policy: str = "complete_only",
    min_daughters: int = 2,
) -> AlignedLevelTargets:
    """Build source-aligned truth targets on an arbitrary predicted context."""

    if truth_batch["node_mask"].shape[0] != 1 or predicted_context["node_mask"].shape[0] != 1:
        raise ValueError("scheduled target alignment is a per-event micro-rollout")
    eligible = truth_batch["node_mask"][0] & (
        truth_batch["level_ids"][0] == target_level
    )
    if target_policy != "diagnostic_all":
        eligible &= truth_batch["valid_reconstruction_target"][0]
    if target_policy == "complete_only":
        eligible &= truth_batch["recursive_reconstructable_complete"][0]
    elif target_policy not in {"reconstructable_partial", "diagnostic_all"}:
        raise ValueError(f"unknown reconstruction target policy: {target_policy}")
    truth_nodes = eligible.nonzero(as_tuple=False).flatten()
    predicted_mask = predicted_context["node_mask"][0]
    predicted_sources = predicted_context["recursive_leaf_source_mask"][0][predicted_mask]
    predicted_positions = predicted_mask.nonzero(as_tuple=False).flatten()
    target_types = []
    target_masks = []
    target_p4 = []
    target_charge = []
    representable = 0
    for mother in truth_nodes.tolist():
        daughters = truth_batch["daughter_adjacency"][0, mother].nonzero(
            as_tuple=False
        ).flatten()
        if daughters.numel() < min_daughters:
            continue
        positions: list[int] = []
        for daughter in daughters.tolist():
            sources = truth_batch["recursive_leaf_source_mask"][0, daughter]
            exact = (
                (predicted_sources == sources.unsqueeze(0)).all(dim=-1)
                if predicted_sources.shape[-1] == sources.shape[-1]
                else torch.zeros(
                    predicted_sources.shape[0],
                    dtype=torch.bool,
                    device=predicted_sources.device,
                )
            )
            candidates = exact.nonzero(as_tuple=False).flatten()
            if not candidates.numel():
                positions = []
                break
            positions.append(int(predicted_positions[int(candidates[0])]))
        if len(positions) != len(daughters) or len(set(positions)) != len(positions):
            continue
        mask = torch.zeros(
            predicted_context["node_mask"].shape[1],
            dtype=torch.bool,
            device=predicted_context["node_mask"].device,
        )
        mask[positions] = True
        target_types.append(
            truth_batch.get("pid_target_labels", truth_batch["pid_labels"])[0, mother]
        )
        target_masks.append(mask)
        target_p4.append(predicted_context["p4"][0, mask].sum(dim=0))
        target_charge.append(predicted_context["charge"][0, mask].sum())
        representable += 1
    device = predicted_context["p4"].device
    types_tensor = (
        torch.stack(target_types).long()
        if target_types
        else torch.empty(0, dtype=torch.long, device=device)
    )
    masks_tensor = (
        torch.stack(target_masks)
        if target_masks
        else torch.zeros(
            (0, predicted_context["node_mask"].shape[1]),
            dtype=torch.bool,
            device=device,
        )
    )
    p4_tensor = (
        torch.stack(target_p4)
        if target_p4
        else torch.zeros((0, 4), dtype=predicted_context["p4"].dtype, device=device)
    )
    charge_tensor = (
        torch.stack(target_charge)
        if target_charge
        else torch.zeros(
            0, dtype=predicted_context["charge"].dtype, device=device
        )
    )
    return AlignedLevelTargets(
        ([types_tensor], [masks_tensor], [p4_tensor], [charge_tensor]),
        int(truth_nodes.numel()),
        representable,
    )


__all__ = [
    "ContextAlignment",
    "AlignedLevelTargets",
    "ScheduledSamplingConfig",
    "TeacherForcingSchedule",
    "align_context_by_recursive_sources",
    "aligned_level_targets",
    "scheduled_sampling_probability",
    "combine_sampled_context_losses",
]
