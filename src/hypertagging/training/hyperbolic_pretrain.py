"""CPU-testable hyperbolic pretraining loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.hyperbolic_pretraining import hyperbolic_pretraining_loss
from hypertagging.models.hyperbolic import HyperbolicNodeEncoder
from hypertagging.utils.seeds import seed_everything


@dataclass(frozen=True)
class HyperbolicPretrainSummary:
    steps: int
    loss: float
    component_losses: dict[str, float]
    device: str


class PairwiseTopologyHeads(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.same_mother = torch.nn.Linear(hidden_dim * 2, 1)
        self.same_branch = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.cat([h[:, :, None, :].expand(-1, -1, h.shape[1], -1), h[:, None, :, :].expand(-1, h.shape[1], -1, -1)], dim=-1)
        return self.same_mother(pair).squeeze(-1), self.same_branch(pair).squeeze(-1)


def run_hyperbolic_pretrain_dry_run(
    *,
    device: str | torch.device = "cpu",
    max_steps: int = 2,
    batch_size: int = 2,
    seed: int = 7,
) -> HyperbolicPretrainSummary:
    seed_everything(seed)
    device = torch.device(device)
    batch = collate_level_events(tiny_level_events()[:batch_size]).to_dict()
    batch = {key: value.to(device) for key, value in batch.items()}
    encoder = HyperbolicNodeEncoder(n_features=batch["node_features"].shape[-1], hidden_dim=24, hyper_dim=8).to(device)
    heads = PairwiseTopologyHeads(24).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), lr=1e-3)
    last_components: dict[str, torch.Tensor] = {}
    loss = torch.tensor(0.0, device=device)
    for _step in range(max_steps):
        optimizer.zero_grad()
        h, z = encoder(batch["node_features"], batch["pid_labels"], batch["level_ids"], batch["charge"])
        same_mother_logits, same_branch_logits = heads(h)
        loss_out = hyperbolic_pretraining_loss(
            z=z,
            same_mother_logits=same_mother_logits,
            same_branch_logits=same_branch_logits,
            same_mother=batch["same_mother"],
            same_branch=batch["same_branch"],
            parent_ids=batch["parent_ids"],
            level_ids=batch["level_ids"],
            node_mask=batch["node_mask"],
        )
        loss = loss_out.total
        loss.backward()
        optimizer.step()
        last_components = loss_out.components
    return HyperbolicPretrainSummary(
        steps=max_steps,
        loss=float(loss.detach().cpu()),
        component_losses={key: float(value.detach().cpu()) for key, value in last_components.items()},
        device=device.type,
    )
