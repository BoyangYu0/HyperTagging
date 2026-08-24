"""Held-out validation helpers shared by production trainers."""

from __future__ import annotations

import torch

from hypertagging.losses.hyperbolic_pretraining import collapse_diagnostics
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder


@torch.no_grad()
def validate_contextual_geometry(
    encoder: HeterogeneousNodeEncoder,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    curvature: float,
) -> dict[str, float]:
    """Evaluate contextual geometry on a held-out normalized event batch."""

    encoder.eval()
    device_batch = {name: value.to(device) for name, value in batch.items()}
    encoded = encoder(device_batch)
    diagnostics = collapse_diagnostics(
        encoded.hyperbolic_embeddings,
        device_batch["node_mask"],
        level_ids=device_batch["level_ids"],
        node_kind_ids=device_batch.get("node_kind_ids"),
        b_side=device_batch.get("b_side"),
        curvature=curvature,
    )
    encoder.train()
    return {
        f"validation_{name}": float(value.detach().cpu())
        for name, value in diagnostics.items()
    }


__all__ = ["validate_contextual_geometry"]
