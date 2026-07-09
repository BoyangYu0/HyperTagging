"""Checkpoint helpers for new level-autoregressive training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    schema_version: str = "level-reconstruction-v1",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "config": config or {},
        "metrics": metrics or {},
        "preprocessing_schema_version": schema_version,
    }
    torch.save(payload, output)
    return output


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
