"""Scheduled-sampling utilities for level reconstruction."""

from __future__ import annotations

from dataclasses import dataclass


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
