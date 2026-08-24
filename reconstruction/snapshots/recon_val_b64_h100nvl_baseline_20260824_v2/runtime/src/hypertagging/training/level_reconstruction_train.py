"""CPU-testable level-autoregressive reconstruction training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.ablation import build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import RolloutConfig, level_rollout
from hypertagging.evaluation.hierarchical_metrics import next_level_metrics
from hypertagging.training.checkpointing import restore_training_checkpoint
from hypertagging.utils.seeds import seed_everything


@dataclass(frozen=True)
class LevelReconstructionSummary:
    steps: int
    loss: float
    component_losses: dict[str, float]
    matches: int
    teacher_forced_steps: int
    predicted_stop_reason: str
    metrics: dict[str, float]
    device: str


def run_level_reconstruction_dry_run(
    *,
    device: str | torch.device = "cpu",
    max_steps: int = 2,
    batch_size: int = 2,
    seed: int = 11,
    target_level: int = 1,
    ablation: str = "full_revised",
    resume: str | None = None,
) -> LevelReconstructionSummary:
    seed_everything(seed)
    device = torch.device(device)
    batch = collate_level_events(tiny_level_events()[:batch_size], max_query_slots=4).to_dict()
    batch = {key: value.to(device) for key, value in batch.items()}
    model = build_ablation_model(
        ablation,
        n_features=batch["node_features"].shape[-1],
        n_types=len(PDG_TOKENS),
        hidden_dim=24,
        hyper_dim=8,
        n_queries=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if resume is not None:
        restore_training_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
    last = None
    for _step in range(max_steps):
        optimizer.zero_grad()
        output = model(batch, target_level=target_level)
        last = level_reconstruction_loss(output.pointer, batch, target_level=target_level)
        last.total.backward()
        optimizer.step()
    assert last is not None
    with torch.no_grad():
        metric_output = model(batch, target_level=target_level)
        metric_loss = level_reconstruction_loss(
            metric_output.pointer,
            batch,
            target_level=target_level,
        )
        metrics = next_level_metrics(
            metric_output.pointer,
            batch,
            metric_loss.matches,
            target_level=target_level,
        )
    rollout_config = RolloutConfig(
        max_level=4,
        root_types=(),
        exclusive_final=False,
        object_threshold=0.5,
    )
    rollout_batch = collate_level_events(
        [tiny_level_events()[0]],
        max_query_slots=4,
    ).to_dict()
    rollout_batch = {key: value.to(device) for key, value in rollout_batch.items()}
    teacher_forced = level_rollout(
        model,
        rollout_batch,
        mode="teacher_forced",
        config=rollout_config,
    )
    predicted = level_rollout(
        model,
        rollout_batch,
        mode="predicted",
        config=rollout_config,
    )
    return LevelReconstructionSummary(
        steps=max_steps,
        loss=float(last.total.detach().cpu()),
        component_losses={key: float(value.detach().cpu()) for key, value in last.components.items()},
        matches=sum(len(matches) for matches in last.matches),
        teacher_forced_steps=len(teacher_forced.steps),
        predicted_stop_reason=predicted.stop_reason,
        metrics=metrics,
        device=device.type,
    )
