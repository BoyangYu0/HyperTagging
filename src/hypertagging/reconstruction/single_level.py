"""Single-level reconstruction helpers migrated from historical scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from hypertagging.losses.link_losses import link_metrics, transfer_link_metrics
from hypertagging.losses.reconstruction_losses import (
    embedding_cosine_distance,
    embedding_mse_distance,
    momentum_metrics,
    pdg_metrics,
    plain_momentum_metrics,
    recover_pdg,
)


ReconstructionVariant = Literal["grafei_reduced", "grafei", "toy_mc"]


@dataclass(frozen=True)
class SingleLevelReconstruction:
    """Outputs and loss terms from one reconstruction step."""

    pdg_logits: torch.Tensor
    feature: torch.Tensor
    recovered_pdg: torch.Tensor
    reconstructed: dict[str, torch.Tensor]
    reconstructed_link: dict[str, torch.Tensor] | None
    losses: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]

    @property
    def total_loss(self) -> torch.Tensor:
        return self.losses["total"]


def sort_energy(
    pdg: torch.Tensor,
    feat: torch.Tensor,
    mask: torch.Tensor,
    *,
    recover: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort reconstructed particles by descending reconstructed energy."""

    masked_feat = torch.where(
        mask[..., None].repeat((1, 1, feat.shape[-1])),
        feat,
        torch.tensor(float("-inf"), device=feat.device),
    )
    masked_pdg = torch.where(mask, recover_pdg(pdg), 0)
    indices = torch.argsort(masked_feat[:, :, -1], descending=True)
    sorted_feat = torch.gather(masked_feat, 1, indices.unsqueeze(-1).expand(-1, -1, feat.size(-1)))
    if recover:
        sorted_pdg = torch.gather(masked_pdg, 1, indices)
    else:
        sorted_pdg = torch.gather(pdg, 1, indices.unsqueeze(-1).expand(-1, -1, pdg.size(-1)))
    return sorted_pdg, torch.nan_to_num(sorted_feat, neginf=0)


def build_reconstructed_batch(
    pdg_logits: torch.Tensor,
    feature: torch.Tensor,
    mask: torch.Tensor,
    *,
    variant: ReconstructionVariant = "grafei_reduced",
) -> dict[str, torch.Tensor]:
    """Build the reconstructed HTR-style batch used by historical losses."""

    if variant == "grafei":
        pdg = recover_pdg(pdg_logits)
        feat = feature * (pdg > 0)[..., None]
    else:
        pdg = recover_pdg(pdg_logits) * mask
        feat = feature * mask[..., None]
    return {
        "pdg": pdg,
        "feature": feat,
        "padding_mask": mask,
    }


def build_goal_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build the historical target HTR-style batch from a reconstruction batch."""

    return {
        "pdg": batch["pdg_y"],
        "feature": batch["feature_y"],
        "padding_mask": batch["padding_mask"],
    }


def build_reconstructed_link_batch(
    batch: dict[str, torch.Tensor],
    reconstructed: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build the historical link-model input for reconstructed mothers."""

    return {
        "pdg_x": batch["pdg_x"],
        "pdg_y": reconstructed["pdg"],
        "feature_x": batch["feature_x"],
        "feature_y": reconstructed["feature"],
        "padding_mask": batch["padding_mask"],
    }


def single_level_reconstruction_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    variant: ReconstructionVariant = "grafei_reduced",
    sort_by_energy: bool = False,
    embedding_model: torch.nn.Module | None = None,
    link_model: torch.nn.Module | None = None,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> SingleLevelReconstruction:
    """Run one historical reconstruction step and compute available losses."""

    pdg_logits, feature = model(batch)
    mask = batch["padding_mask"]
    if sort_by_energy:
        pdg_logits, feature = sort_energy(pdg_logits, feature, mask, recover=False)

    reconstructed = build_reconstructed_batch(pdg_logits, feature, mask, variant=variant)
    reconstructed_link = build_reconstructed_link_batch(batch, reconstructed)

    pdg_loss, pdg_acc = pdg_metrics(pdg_logits, batch["pdg_y"], mask)
    if variant == "toy_mc":
        feat_loss, feat_err = plain_momentum_metrics(feature, batch["feature_y"], mask)
    else:
        spatial_weight = 3 if variant == "grafei" else 1
        feat_loss, feat_err = momentum_metrics(feature, batch["feature_y"], mask, spatial_weight=spatial_weight)

    zero = pdg_loss.new_tensor(0.0)
    emb_loss = zero
    link_loss = zero
    link_acc = zero

    if embedding_model is not None:
        if variant == "grafei":
            emb_loss = embedding_cosine_distance(embedding_model(reconstructed), batch["emb"])
        elif variant == "grafei_reduced":
            emb_loss = embedding_cosine_distance(embedding_model(reconstructed), embedding_model(build_goal_batch(batch)))
        else:
            emb_loss = embedding_mse_distance(embedding_model(reconstructed), batch["emb_y"])

    if link_model is not None:
        if variant == "grafei":
            link_loss, link_acc = transfer_link_metrics(link_model(reconstructed_link), link_model(batch), mask)
        else:
            link_loss, link_acc = link_metrics(link_model(reconstructed_link), batch["links"], mask)

    total = weights[0] * pdg_loss + weights[1] * feat_loss + weights[2] * emb_loss + weights[3] * link_loss
    return SingleLevelReconstruction(
        pdg_logits=pdg_logits,
        feature=feature,
        recovered_pdg=reconstructed["pdg"],
        reconstructed=reconstructed,
        reconstructed_link=reconstructed_link,
        losses={
            "pdg": pdg_loss,
            "feature": feat_loss,
            "embedding": emb_loss,
            "link": link_loss,
            "total": total,
        },
        metrics={
            "pdg_acc": pdg_acc,
            "feature_err": feat_err,
            "link_acc": link_acc,
        },
    )


def reconstructed_embedding_distance(
    reconstructed_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    variant: ReconstructionVariant = "grafei_reduced",
) -> torch.Tensor:
    """Tensor-level embedding distance used by single-level reconstruction."""

    if variant == "toy_mc":
        return embedding_mse_distance(reconstructed_embedding, target_embedding)
    return embedding_cosine_distance(reconstructed_embedding, target_embedding)
