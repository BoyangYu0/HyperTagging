"""Link-stage training entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hypertagging.losses.link_losses import link_metrics, transfer_link_metrics
from hypertagging.models import EmbLinker, linearLinker
from hypertagging.training.dry_run import link_batch
from hypertagging.training.loops import build_optimizer, run_link_dry_run


LinkTrainingMode = Literal["ground_truth", "reconstructed_mother"]
LinkModelVariant = Literal["standard", "embedding"]


@dataclass(frozen=True)
class LinkStepResult:
    """One link-prediction training/evaluation step."""

    mode: LinkTrainingMode
    logits: torch.Tensor
    loss: torch.Tensor
    accuracy: torch.Tensor
    model_input: dict[str, torch.Tensor]


@dataclass(frozen=True)
class LinkDryRunSummary:
    """Summary of a one-batch link-prediction dry run."""

    stage: str
    mode: LinkTrainingMode
    model_variant: LinkModelVariant
    device: str
    model_class: str
    optimizer_class: str
    loss: float
    accuracy: float
    logits_shape: tuple[int, ...]
    parameter_count: int
    backward_ran: bool


def build_link_model_input(
    batch: dict[str, torch.Tensor],
    *,
    mode: LinkTrainingMode = "ground_truth",
    reconstructed_mother: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build link-model input for ground-truth or reconstructed-mother mode."""

    if mode == "ground_truth":
        return {
            "pdg_x": batch["pdg_x"],
            "pdg_y": batch["pdg_y"],
            "feature_x": batch["feature_x"],
            "feature_y": batch["feature_y"],
            "padding_mask": batch["padding_mask"],
        }
    if mode != "reconstructed_mother":
        raise ValueError(f"Unknown link training mode: {mode}")
    if reconstructed_mother is None:
        raise ValueError("reconstructed_mother is required for reconstructed_mother mode.")
    pdg_y = reconstructed_mother.get("pdg_y", reconstructed_mother.get("pdg"))
    feature_y = reconstructed_mother.get("feature_y", reconstructed_mother.get("feature"))
    if pdg_y is None or feature_y is None:
        raise KeyError("reconstructed_mother must contain pdg/feature or pdg_y/feature_y.")
    return {
        "pdg_x": batch["pdg_x"],
        "pdg_y": pdg_y,
        "feature_x": batch["feature_x"],
        "feature_y": feature_y,
        "padding_mask": batch["padding_mask"],
    }


def link_prediction_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    mode: LinkTrainingMode = "ground_truth",
    reconstructed_mother: dict[str, torch.Tensor] | None = None,
    teacher_logits: torch.Tensor | None = None,
) -> LinkStepResult:
    """Run one link-prediction step and compute the historical loss."""

    model_input = build_link_model_input(
        batch,
        mode=mode,
        reconstructed_mother=reconstructed_mother,
    )
    logits = model(model_input)
    if teacher_logits is None:
        loss, accuracy = link_metrics(logits, batch["links"], batch["padding_mask"])
    else:
        loss, accuracy = transfer_link_metrics(logits, teacher_logits, batch["padding_mask"])
    return LinkStepResult(
        mode=mode,
        logits=logits,
        loss=loss,
        accuracy=accuracy,
        model_input=model_input,
    )


def run_link_prediction_dry_run(
    *,
    mode: LinkTrainingMode = "ground_truth",
    model_variant: LinkModelVariant = "standard",
    device: str | torch.device = "cpu",
    backward: bool = True,
) -> LinkDryRunSummary:
    """Build a link model, one batch, loss, and optimizer for CPU dry-run."""

    device = torch.device(device)
    batch = link_batch(device)
    if model_variant == "standard":
        model: torch.nn.Module = linearLinker(
            n_features=4,
            link_width=8,
            link_n_head=1,
            link_n_layers=1,
            link_fc=16,
            pdg_emb=2,
            num_pdg=8,
            device=device,
        )
    elif model_variant == "embedding":
        model = EmbLinker(
            n_features=4,
            link_width=8,
            link_n_head=1,
            link_n_layers=1,
            link_fc=16,
            device=device,
        )
        batch = {
            "emb_x": batch["feature_x"],
            "emb_y": batch["feature_y"],
            "links": batch["links"],
            "padding_mask": batch["padding_mask"],
        }
    else:
        raise ValueError(f"Unknown link model variant: {model_variant}")

    reconstructed = None
    if mode == "reconstructed_mother":
        reconstructed = {
            "pdg": batch["pdg_y"] if "pdg_y" in batch else None,
            "feature": (batch["feature_y"] + 0.05) if "feature_y" in batch else None,
        }
    if model_variant == "embedding":
        logits = model(batch)
        loss, accuracy = link_metrics(logits, batch["links"], batch["padding_mask"])
    else:
        result = link_prediction_step(
            model,
            batch,
            mode=mode,
            reconstructed_mother=reconstructed,
        )
        logits = result.logits
        loss = result.loss
        accuracy = result.accuracy

    optimizer = build_optimizer(model, stage="link")
    if backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return LinkDryRunSummary(
        stage="link",
        mode=mode,
        model_variant=model_variant,
        device=device.type,
        model_class=model.__class__.__name__,
        optimizer_class=optimizer.__class__.__name__,
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        logits_shape=tuple(logits.shape),
        parameter_count=sum(p.numel() for p in model.parameters()),
        backward_ran=backward,
    )


__all__ = [
    "LinkDryRunSummary",
    "LinkModelVariant",
    "LinkStepResult",
    "LinkTrainingMode",
    "build_link_model_input",
    "link_prediction_step",
    "run_link_dry_run",
    "run_link_prediction_dry_run",
]
