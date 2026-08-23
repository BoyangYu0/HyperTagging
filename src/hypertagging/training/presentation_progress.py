"""Fail-closed presentation accounting for exact mid-run batch migration.

The phase-3 source run used 16 presentations per optimizer update.  This
module keeps that unit explicit when a replacement run uses a larger device
batch.  The optimizer update counter remains an implementation/runtime
counter; scientific progress is the integer number of 16-presentation virtual
steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


PRESENTATION_PROGRESS_VERSION = "presentation-progress-v1"
VIRTUAL_STEP_PRESENTATIONS = 16
PHASE3_TOTAL_PRESENTATIONS = 1_730_048
PHASE3_RESUME_STEP = 54_064
PHASE3_RESUME_PRESENTATIONS = PHASE3_RESUME_STEP * VIRTUAL_STEP_PRESENTATIONS
PHASE3_VALIDATION_VIRTUAL_STEPS = (
    13_516,
    27_032,
    40_548,
    54_064,
    67_580,
    81_096,
    94_612,
    108_128,
)
PHASE3_VALIDATION_PRESENTATIONS = tuple(
    value * VIRTUAL_STEP_PRESENTATIONS for value in PHASE3_VALIDATION_VIRTUAL_STEPS
)
PHASE3_PHASE_VIRTUAL_DURATIONS = (27_032, 27_032, 27_032, 27_032)
PHASE3_PHASE_PRESENTATION_DURATIONS = tuple(
    value * VIRTUAL_STEP_PRESENTATIONS for value in PHASE3_PHASE_VIRTUAL_DURATIONS
)


@dataclass(frozen=True)
class PresentationProgress:
    """A validated scientific progress position."""

    presentations: int
    virtual_step: int
    optimizer_steps: int
    batch_size: int
    unit_presentations: int = VIRTUAL_STEP_PRESENTATIONS

    def contract(self) -> dict[str, object]:
        return {
            "version": PRESENTATION_PROGRESS_VERSION,
            "unit": "virtual_step",
            "unit_presentations": self.unit_presentations,
            "presentations_completed": self.presentations,
            "virtual_step": self.virtual_step,
            "optimizer_steps_completed": self.optimizer_steps,
            "device_batch_size": self.batch_size,
        }


def validate_batch_profile(
    batch_size: int,
    *,
    unit_presentations: int = VIRTUAL_STEP_PRESENTATIONS,
    total_presentations: int | None = None,
    milestone_presentations: Sequence[int] = (),
) -> None:
    """Validate a batch size before it can participate in scientific resume."""

    batch_size = int(batch_size)
    unit_presentations = int(unit_presentations)
    if batch_size <= 0 or unit_presentations <= 0:
        raise ValueError("batch_size and unit_presentations must be positive")
    if batch_size % unit_presentations:
        raise ValueError(
            "scientific batch migration requires batch_size to be divisible by "
            f"the {unit_presentations}-presentation virtual unit"
        )
    if total_presentations is not None:
        total_presentations = int(total_presentations)
        if total_presentations <= 0 or total_presentations % batch_size:
            raise ValueError(
                "total presentations must be positive and divisible by batch_size"
            )
    milestones = tuple(int(value) for value in milestone_presentations)
    if any(value <= 0 or value % batch_size for value in milestones):
        raise ValueError(
            "every validation/checkpoint presentation milestone must be divisible "
            "by the candidate batch size"
        )


def optimizer_updates_for_presentations(
    presentations: int, batch_size: int
) -> int:
    presentations = int(presentations)
    batch_size = int(batch_size)
    if presentations < 0 or batch_size <= 0 or presentations % batch_size:
        raise ValueError("presentations must be non-negative and batch-divisible")
    return presentations // batch_size


def progress_from_presentations(
    presentations: int,
    *,
    optimizer_steps: int,
    batch_size: int,
    unit_presentations: int = VIRTUAL_STEP_PRESENTATIONS,
) -> PresentationProgress:
    presentations = int(presentations)
    unit_presentations = int(unit_presentations)
    if presentations < 0 or presentations % unit_presentations:
        raise ValueError("presentations must be an exact virtual-step boundary")
    validate_batch_profile(batch_size, unit_presentations=unit_presentations)
    return PresentationProgress(
        presentations=presentations,
        virtual_step=presentations // unit_presentations,
        optimizer_steps=int(optimizer_steps),
        batch_size=int(batch_size),
        unit_presentations=unit_presentations,
    )


def progress_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    target_batch_size: int,
    unit_presentations: int = VIRTUAL_STEP_PRESENTATIONS,
    total_presentations: int = PHASE3_TOTAL_PRESENTATIONS,
    milestone_presentations: Sequence[int] = PHASE3_VALIDATION_PRESENTATIONS,
) -> PresentationProgress:
    """Read explicit progress or migrate a legacy step-16 checkpoint.

    Legacy phase-3 checkpoints have no presentation record.  Their optimizer
    step is authoritative only because the source batch size is bound to 16;
    the migration therefore uses ``step * 16`` and never trusts the old
    runtime batch cursor as scientific progress.
    """

    validate_batch_profile(
        target_batch_size,
        unit_presentations=unit_presentations,
        total_presentations=total_presentations,
        milestone_presentations=milestone_presentations,
    )
    state = payload.get("training_state", {})
    stored = state.get("presentation_progress") if isinstance(state, Mapping) else None
    if stored:
        if stored.get("version") != PRESENTATION_PROGRESS_VERSION:
            raise ValueError("unsupported checkpoint presentation-progress contract")
        if int(stored.get("unit_presentations", -1)) != unit_presentations:
            raise ValueError("checkpoint presentation unit differs from target")
        presentations = int(stored["presentations_completed"])
        optimizer_steps = int(stored.get("optimizer_steps_completed", payload.get("step", 0)))
    else:
        source_config = payload.get("config", {})
        source_batch_size = int(source_config.get("batch_size", unit_presentations))
        if source_batch_size != unit_presentations:
            raise ValueError(
                "legacy checkpoint migration is only authorized from the "
                f"{unit_presentations}-presentation source batch"
            )
        optimizer_steps = int(payload.get("step", 0))
        presentations = optimizer_steps * unit_presentations
    if presentations < 0 or presentations > total_presentations:
        raise ValueError("checkpoint presentation progress is outside the total budget")
    if presentations % unit_presentations:
        raise ValueError("checkpoint presentation progress is not virtual-step aligned")
    if presentations % target_batch_size:
        raise ValueError("checkpoint progress is not divisible by target batch size")
    return progress_from_presentations(
        presentations,
        optimizer_steps=optimizer_steps,
        batch_size=target_batch_size,
        unit_presentations=unit_presentations,
    )


def presentation_schedule_contract(
    *,
    total_presentations: int,
    warmup_presentations: int,
    min_lr_ratio: float,
    base_lrs: Sequence[float],
    unit_presentations: int = VIRTUAL_STEP_PRESENTATIONS,
) -> dict[str, object]:
    """Build the presentation-equivalent LR schedule metadata."""

    if total_presentations <= 0 or total_presentations % unit_presentations:
        raise ValueError("total presentations must be virtual-step divisible")
    if warmup_presentations < 0 or warmup_presentations >= total_presentations:
        raise ValueError("warmup presentations must be inside the total budget")
    if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must lie in [0, 1]")
    return {
        "version": PRESENTATION_PROGRESS_VERSION,
        "unit": "virtual_step",
        "unit_presentations": int(unit_presentations),
        "total_presentations": int(total_presentations),
        "total_virtual_steps": int(total_presentations // unit_presentations),
        "warmup_presentations": int(warmup_presentations),
        "warmup_virtual_steps": int(warmup_presentations // unit_presentations),
        "min_lr_ratio": float(min_lr_ratio),
        "base_lrs": [float(value) for value in base_lrs],
    }


def phase_index_for_virtual_step(
    virtual_step: int, durations: Sequence[int] = PHASE3_PHASE_VIRTUAL_DURATIONS
) -> int:
    virtual_step = int(virtual_step)
    if virtual_step < 0 or not durations or any(int(value) <= 0 for value in durations):
        raise ValueError("virtual step and phase durations must be positive")
    cursor = 0
    for index, duration in enumerate(durations):
        cursor += int(duration)
        if virtual_step < cursor:
            return index
    return len(durations) - 1


def crossed_milestones(
    previous_presentations: int,
    current_presentations: int,
    milestones: Sequence[int] = PHASE3_VALIDATION_PRESENTATIONS,
) -> tuple[int, ...]:
    previous_presentations = int(previous_presentations)
    current_presentations = int(current_presentations)
    if current_presentations < previous_presentations:
        raise ValueError("presentation progress cannot move backwards")
    return tuple(
        int(value)
        for value in milestones
        if previous_presentations < int(value) <= current_presentations
    )


__all__ = [
    "PHASE3_PHASE_PRESENTATION_DURATIONS",
    "PHASE3_PHASE_VIRTUAL_DURATIONS",
    "PHASE3_RESUME_PRESENTATIONS",
    "PHASE3_RESUME_STEP",
    "PHASE3_TOTAL_PRESENTATIONS",
    "PHASE3_VALIDATION_PRESENTATIONS",
    "PHASE3_VALIDATION_VIRTUAL_STEPS",
    "PRESENTATION_PROGRESS_VERSION",
    "PresentationProgress",
    "VIRTUAL_STEP_PRESENTATIONS",
    "crossed_milestones",
    "optimizer_updates_for_presentations",
    "phase_index_for_virtual_step",
    "presentation_schedule_contract",
    "progress_from_checkpoint",
    "progress_from_presentations",
    "validate_batch_profile",
]
