"""CPU-testable level-autoregressive reconstruction training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.utils.seeds import seed_everything


@dataclass(frozen=True)
class LevelReconstructionSummary:
    steps: int
    loss: float
    component_losses: dict[str, float]
    matches: int
    device: str


def run_level_reconstruction_dry_run(
    *,
    device: str | torch.device = "cpu",
    max_steps: int = 2,
    batch_size: int = 2,
    seed: int = 11,
    target_level: int = 1,
) -> LevelReconstructionSummary:
    seed_everything(seed)
    device = torch.device(device)
    batch = collate_level_events(tiny_level_events()[:batch_size], max_query_slots=4).to_dict()
    batch = {key: value.to(device) for key, value in batch.items()}
    model = LevelAutoregressiveReconstructor(
        n_features=batch["node_features"].shape[-1],
        n_types=4096,
        hidden_dim=24,
        hyper_dim=8,
        n_queries=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    last = None
    for _step in range(max_steps):
        optimizer.zero_grad()
        output = model(batch, target_level=target_level)
        last = level_reconstruction_loss(output.pointer, batch, target_level=target_level)
        last.total.backward()
        optimizer.step()
    assert last is not None
    return LevelReconstructionSummary(
        steps=max_steps,
        loss=float(last.total.detach().cpu()),
        component_losses={key: float(value.detach().cpu()) for key, value in last.components.items()},
        matches=sum(len(matches) for matches in last.matches),
        device=device.type,
    )
