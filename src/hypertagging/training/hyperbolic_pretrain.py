"""CPU-testable hyperbolic pretraining loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.hyperbolic_pretraining import (
    N_TREE_RELATIONS,
    build_topology_safe_parent_negative_mask,
    build_tree_relation_targets,
    hyperbolic_pretraining_loss,
    pool_b_branch_embeddings,
)
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.ablation import ALL_ABLATIONS
from hypertagging.training.checkpointing import restore_training_checkpoint
from hypertagging.evaluation.hierarchical_metrics import (
    parent_ranking_accuracy,
    tree_relation_accuracy,
)
from hypertagging.utils.seeds import seed_everything


@dataclass(frozen=True)
class HyperbolicPretrainSummary:
    steps: int
    loss: float
    component_losses: dict[str, float]
    diagnostics: dict[str, float]
    metrics: dict[str, float]
    device: str


class PairwiseTopologyHeads(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.same_mother = torch.nn.Linear(hidden_dim * 2, 1)
        self.same_branch = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.cat([h[:, :, None, :].expand(-1, -1, h.shape[1], -1), h[:, None, :, :].expand(-1, h.shape[1], -1, -1)], dim=-1)
        return self.same_mother(pair).squeeze(-1), self.same_branch(pair).squeeze(-1)


class TreeRelationHead(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, N_TREE_RELATIONS),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        left = h[:, :, None, :].expand(-1, -1, h.shape[1], -1)
        right = h[:, None, :, :].expand(-1, h.shape[1], -1, -1)
        return self.net(torch.cat([left, right, left - right, left * right], dim=-1))


def run_hyperbolic_pretrain_dry_run(
    *,
    device: str | torch.device = "cpu",
    max_steps: int = 2,
    batch_size: int = 2,
    seed: int = 7,
    ablation: str = "full_revised",
    resume: str | None = None,
) -> HyperbolicPretrainSummary:
    seed_everything(seed)
    device = torch.device(device)
    events = tiny_level_events()[:batch_size]
    topology_batch = collate_level_events(events).to_dict()
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(event) for event in events]
    )
    for name in (
        "same_mother",
        "same_branch",
        "lca_depth",
        "lca_node_id",
        "edges_to_lca_from_i",
        "edges_to_lca_from_j",
        "exact_tree_path_distance",
        "depth_from_retained_root",
        "distance_to_nearest_retained_root",
        "ancestor_descendant_relation",
    ):
        batch[name] = topology_batch[name]
    batch = {key: value.to(device) for key, value in batch.items()}
    encoder = HeterogeneousNodeEncoder(d_model=24, hyper_dim=8).to(device)
    head = TreeRelationHead(24).to(device)
    checkpoint_model = torch.nn.ModuleDict({"encoder": encoder, "relation_head": head})
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=1e-3)
    if resume is not None:
        restore_training_checkpoint(
            resume,
            model=checkpoint_model,
            optimizer=optimizer,
            map_location=device,
        )
    ablation_config = ALL_ABLATIONS[ablation]
    weights = {
        "lca": float(ablation_config.lca_parent),
        "parent": float(ablation_config.lca_parent),
        "depth": 0.2 * float(ablation_config.radius_depth),
        "channel": 0.2 * float(ablation_config.channel_supervision),
        "var": 0.1 * float(ablation_config.variance_covariance),
        "cov": 0.01 * float(ablation_config.variance_covariance),
    }
    last_components: dict[str, torch.Tensor] = {}
    last_diagnostics: dict[str, torch.Tensor] = {}
    last_metrics: dict[str, float] = {}
    loss = torch.tensor(0.0, device=device)
    for _step in range(max_steps):
        optimizer.zero_grad()
        encoded = encoder(batch)
        relation_logits = head(encoded.tree_projection)
        relation_targets, relation_mask = build_tree_relation_targets(
            parent_ids=batch["parent_ids"],
            lca_depth=batch["lca_depth"],
            level_ids=batch["level_ids"],
            node_mask=batch["node_mask"],
            b_side=batch["b_side"],
            lca_node_id=batch["lca_node_id"],
            edges_to_lca_from_i=batch["edges_to_lca_from_i"],
            edges_to_lca_from_j=batch["edges_to_lca_from_j"],
        )
        parent_negative_mask = build_topology_safe_parent_negative_mask(
            relation_targets, batch["node_mask"],
            batch["ancestor_descendant_relation"],
        )
        branch_embeddings, branch_mask = pool_b_branch_embeddings(
            encoded.channel_projection,
            batch["b_side"],
            batch["node_mask"],
        )
        loss_out = hyperbolic_pretraining_loss(
            z=encoded.hyperbolic_embeddings,
            tree_relation_logits=relation_logits,
            tree_relation_targets=relation_targets,
            tree_relation_mask=relation_mask,
            lca_depth=batch["lca_depth"],
            exact_tree_path_distance=batch["exact_tree_path_distance"],
            parent_negative_mask=parent_negative_mask,
            parent_ids=batch["parent_ids"],
            level_ids=batch["level_ids"],
            node_mask=batch["node_mask"],
            b_side=batch["b_side"],
            node_kind_ids=batch["node_kind_ids"],
            event_ids=batch["event_ids"],
            channel_embeddings=branch_embeddings,
            channel_mask=branch_mask,
            channel_ids=torch.stack([batch["b1_channel_ids"], batch["b2_channel_ids"]], dim=-1),
            structured_channel_similarity=batch["channel_similarity"],
            weights=weights,
        )
        loss = loss_out.total
        loss.backward()
        optimizer.step()
        last_components = loss_out.components
        last_diagnostics = loss_out.diagnostics
        last_metrics = {
            "lca_tree_relation_accuracy": tree_relation_accuracy(
                relation_logits.detach(),
                relation_targets,
                relation_mask,
            ),
            "parent_ranking_accuracy": parent_ranking_accuracy(
                encoded.hyperbolic_embeddings.detach(),
                batch["parent_ids"],
                batch["node_mask"],
            ),
        }
    return HyperbolicPretrainSummary(
        steps=max_steps,
        loss=float(loss.detach().cpu()),
        component_losses={key: float(value.detach().cpu()) for key, value in last_components.items()},
        diagnostics={key: float(value.detach().cpu()) for key, value in last_diagnostics.items()},
        metrics=last_metrics,
        device=device.type,
    )
