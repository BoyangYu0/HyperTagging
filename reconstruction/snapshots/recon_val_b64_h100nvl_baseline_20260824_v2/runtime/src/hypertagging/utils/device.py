"""Device selection helpers for CPU-first debugging."""

from __future__ import annotations

import torch


def resolve_device(
    device: str | torch.device | None = None,
    *,
    prefer_cuda: bool = False,
) -> torch.device:
    """Resolve a torch device, defaulting to CPU for reproducible smoke tests."""

    if device is not None:
        return torch.device(device)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cpu_device() -> torch.device:
    """Return the CPU torch device."""

    return torch.device("cpu")
