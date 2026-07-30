"""Checkpoint helpers for new level-autoregressive training."""

from __future__ import annotations

from pathlib import Path
import os
import random
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v3 import SCHEMA_VERSION_V3, feature_spec_v3


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
    schema_version: str = SCHEMA_VERSION_V3,
    encoder: torch.nn.Module | None = None,
    scaler: Any | None = None,
    split_manifest_hash: str = "",
    confidence_head_trained: bool = False,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": (
            encoder.state_dict()
            if encoder is not None
            else (
                getattr(model, "encoder").state_dict()
                if hasattr(model, "encoder")
                else {}
            )
        ),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "config": config or {},
        "metrics": metrics or {},
        "normalizer_state": normalizer_state or {},
        "preprocessing_schema_version": schema_version,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_specification": feature_spec_v3(),
        "split_manifest_hash": split_manifest_hash,
        "git_commit": _git_commit(),
        "confidence_head_trained": bool(confidence_head_trained),
        "random_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Checkpoints contain optimizer and RNG metadata in addition to tensors;
    # callers must only load trusted experiment artifacts.
    return torch.load(path, map_location=map_location, weights_only=False)


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


def _git_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"
