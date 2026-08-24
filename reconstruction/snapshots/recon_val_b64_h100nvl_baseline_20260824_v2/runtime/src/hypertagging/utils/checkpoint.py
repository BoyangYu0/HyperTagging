"""Checkpoint helpers for historical HyperTagging checkpoint dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


Checkpoint = Mapping[str, Any]


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = "cpu",
    **torch_load_kwargs: Any,
) -> Any:
    """Load a checkpoint with CPU-safe defaults.

    Historical scripts commonly read dictionaries containing ``epoch`` and
    ``model_state_dict``. The default ``map_location='cpu'`` supports local
    debugging without a GPU while still allowing callers to request another
    device explicitly.
    """

    return torch.load(path, map_location=map_location, **torch_load_kwargs)


def get_model_state_dict(
    checkpoint: Checkpoint,
    key: str = "model_state_dict",
) -> Mapping[str, Any]:
    """Return the model state dictionary using the historical key."""

    if key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain {key!r}.")
    state_dict = checkpoint[key]
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint entry {key!r} is not a mapping.")
    return state_dict


def get_epoch(checkpoint: Checkpoint, default: int | None = None) -> int | None:
    """Return the historical ``epoch`` checkpoint entry if present."""

    epoch = checkpoint.get("epoch", default)
    if epoch is None:
        return None
    return int(epoch)


def load_model_state(
    model: torch.nn.Module,
    path: str | Path,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    key: str = "model_state_dict",
    **torch_load_kwargs: Any,
) -> tuple[Any, torch.nn.modules.module._IncompatibleKeys]:
    """Load a historical checkpoint state dict into ``model``."""

    checkpoint = load_checkpoint(path, map_location=map_location, **torch_load_kwargs)
    state_dict = get_model_state_dict(checkpoint, key=key)
    result = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, result


def save_checkpoint(
    path: str | Path,
    model_state_dict: Mapping[str, Any],
    epoch: int | None = None,
    **extra: Any,
) -> None:
    """Write the minimal historical checkpoint dictionary format."""

    payload: dict[str, Any] = dict(extra)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    payload["model_state_dict"] = dict(model_state_dict)
    torch.save(payload, path)
