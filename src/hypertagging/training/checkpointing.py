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
    normalizer_state: dict[str, Any] | None = None,
    schema_version: str = "direct-mdst-tree-v2",
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
        "normalizer_state": normalizer_state or {},
        "preprocessing_schema_version": schema_version,
    }
    torch.save(payload, output)
    return output


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore model/training state and return the complete checkpoint metadata."""

    payload = load_training_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
