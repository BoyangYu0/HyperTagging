"""Versioned optimizer-step learning-rate schedules."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


LR_SCHEDULE_VERSION = "linear-warmup-cosine-v1"


def learning_rate_schedule_contract(
    *,
    total_steps: int,
    warmup_fraction: float = 0.05,
    warmup_steps: int | None = None,
    max_warmup_steps: int = 10_000,
    min_lr_ratio: float = 0.0,
    base_lrs: Sequence[float] = (),
) -> dict[str, object]:
    if total_steps <= 0:
        raise ValueError("learning-rate total_steps must be positive")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must lie in [0, 1]")
    if max_warmup_steps < 0:
        raise ValueError("max_warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must lie in [0, 1]")
    resolved = (
        min(int(warmup_steps), max_warmup_steps)
        if warmup_steps is not None
        else min(
            int(math.ceil(total_steps * warmup_fraction)),
            max_warmup_steps,
            max(total_steps - 1, 0),
        )
    )
    if resolved < 0 or resolved >= total_steps:
        if not (total_steps == 1 and resolved == 0):
            raise ValueError("warmup_steps must be non-negative and smaller than total_steps")
    return {
        "version": LR_SCHEDULE_VERSION,
        "unit": "optimizer_step",
        "total_steps": int(total_steps),
        "warmup_steps": int(resolved),
        "warmup_fraction": float(warmup_fraction),
        "max_warmup_steps": int(max_warmup_steps),
        "min_lr_ratio": float(min_lr_ratio),
        "base_lrs": [float(value) for value in base_lrs],
    }


def lr_multiplier(optimizer_step: int, contract: Mapping[str, object]) -> float:
    """Multiplier used by optimizer step ``optimizer_step`` (zero based)."""

    total = int(contract["total_steps"])
    warmup = int(contract["warmup_steps"])
    minimum = float(contract["min_lr_ratio"])
    step = max(int(optimizer_step), 0)
    if warmup and step < warmup:
        return (step + 1) / warmup
    decay_steps = total - warmup
    if decay_steps <= 1:
        return minimum if step >= total else 1.0
    progress = min(max((step - warmup) / (decay_steps - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (1.0 - minimum) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    contract: Mapping[str, object],
) -> torch.optim.lr_scheduler.LambdaLR:
    if contract.get("version") != LR_SCHEDULE_VERSION:
        raise ValueError("unsupported learning-rate schedule contract version")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, contract),
    )


def resolve_resume_schedule_contract(
    *,
    resume_payload: Mapping[str, Any] | None,
    configured_total_steps: int | None,
    run_max_steps: int,
    warmup_fraction: float,
    warmup_steps: int | None,
    max_warmup_steps: int,
    min_lr_ratio: float,
    base_lrs: Sequence[float],
) -> dict[str, object]:
    """Resolve a fresh contract or require the exact serialized resume contract."""

    if resume_payload is None:
        return learning_rate_schedule_contract(
            total_steps=(configured_total_steps or run_max_steps),
            warmup_fraction=warmup_fraction,
            warmup_steps=warmup_steps,
            max_warmup_steps=max_warmup_steps,
            min_lr_ratio=min_lr_ratio,
            base_lrs=base_lrs,
        )
    stored = resume_payload.get("training_state", {}).get("lr_schedule_contract")
    if not stored:
        raise ValueError(
            "legacy checkpoint has no versioned learning-rate schedule contract; "
            "refusing to silently change its LR trajectory"
        )
    if stored.get("version") != LR_SCHEDULE_VERSION:
        raise ValueError("checkpoint uses an unsupported learning-rate schedule contract")
    expected_base_lrs = [float(value) for value in base_lrs]
    if list(stored.get("base_lrs", [])) != expected_base_lrs:
        raise ValueError("resume learning-rate base values differ from the checkpoint")
    if configured_total_steps is not None and int(stored["total_steps"]) != configured_total_steps:
        raise ValueError("resume learning-rate total step budget differs from the checkpoint")
    for name, configured in (
        ("warmup_fraction", float(warmup_fraction)),
        ("max_warmup_steps", int(max_warmup_steps)),
        ("min_lr_ratio", float(min_lr_ratio)),
    ):
        if stored.get(name) != configured:
            raise ValueError(f"resume learning-rate {name} differs from the checkpoint")
    if warmup_steps is not None and int(stored["warmup_steps"]) != warmup_steps:
        raise ValueError("resume warmup_steps differs from the checkpoint")
    return dict(stored)


__all__ = [
    "LR_SCHEDULE_VERSION",
    "build_warmup_cosine_scheduler",
    "learning_rate_schedule_contract",
    "lr_multiplier",
    "resolve_resume_schedule_contract",
]
