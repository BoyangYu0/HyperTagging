"""Embedding losses migrated from historical HyperTagging scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from hypertagging.utils.tensor_contractions import boolean_matmul


EPSILON = 1e-6


def build_angle_matrix(
    vectors: torch.Tensor,
    amplifier: float | None = None,
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Historical cosine-angle affinity matrix."""

    left = vectors.unsqueeze(1).repeat(1, len(vectors), 1)
    right = vectors.unsqueeze(0).repeat(len(vectors), 1, 1)
    similarity = F.cosine_similarity(left, right, dim=-1, eps=epsilon)
    if not amplifier:
        return torch.exp(similarity)
    return torch.exp(similarity / amplifier), torch.exp(similarity * amplifier)


def build_distance_matrix(vectors: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """Historical Poincare-distance affinity matrix."""

    scale = 1 / (1 - torch.norm(vectors, dim=1) ** 2)
    distance = torch.norm(vectors[:, None] - vectors, dim=2) ** 2
    return torch.exp(-torch.acosh(1 + 2 * scale * distance * scale[:, None] + epsilon))


def grafei_radius_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """GraFEI SampleEmbedding radius loss using mass tokens."""

    r_euclidean = torch.norm(vectors, dim=-1) ** 2
    r_poincare = torch.acosh(1 + 2 * r_euclidean / (1 - r_euclidean - epsilon))
    r_goal = 0.6 * torch.sqrt(1 - dataset["mass"].float().to(vectors.device) / 100) + 0.3
    return F.mse_loss(input=r_poincare, target=r_goal)


def toy_mc_radius_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """HyperTagging/noR SampleEmbedding radius loss using E_Rec."""

    r_euclidean = torch.norm(vectors, dim=-1) ** 2
    r_poincare = torch.acosh(1 + 2 * r_euclidean / (1 - r_euclidean - epsilon))
    r_goal = 0.6 * torch.sqrt(1 - dataset["E_Rec"].float().to(vectors.device)) + 0.3
    return F.mse_loss(input=r_poincare, target=r_goal)


def colab_radius_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    metric: str = "hyperbolic",
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """HyperTaggingColab masked radius loss variant."""

    mask = dataset["padding_mask"].to(vectors.device)
    masked_vectors = vectors[mask]
    masses = dataset["mass"].to(vectors.device)[mask]
    radius = torch.norm(masked_vectors, dim=-1)
    if metric == "hyperbolic":
        radius = torch.acosh(1 + 2 * radius**2 / (1 - radius**2 - epsilon))
    r_goal = 0.9 * torch.sqrt(1 - masses / 100) + 0.1
    return F.l1_loss(input=radius[..., None], target=r_goal)


def grafei_intra_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    amplifier: float = 1,
    epsilon: float = EPSILON,
    event_key: str = "evtNums",
) -> torch.Tensor:
    """Historical intra-event embedding concentration loss."""

    affinities_narrow, affinities_wide = build_angle_matrix(vectors, amplifier=amplifier, epsilon=epsilon)
    events = dataset[event_key].to(vectors.device)
    mask = events[:, None] == events
    norm = torch.einsum("ij,ij->j", affinities_narrow, (~mask).float()) + epsilon
    return (-torch.log(affinities_wide / norm)[mask]).mean()


def grafei_inter_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Historical GraFEI inter-pattern embedding separation loss."""

    pattern = dataset["pattern"].float().to(vectors.device)
    left = pattern.unsqueeze(1).repeat(1, len(pattern), 1)
    right = pattern.unsqueeze(0).repeat(len(pattern), 1, 1)
    pattern_similarity = F.cosine_similarity(left, right, dim=-1, eps=epsilon).to(vectors.device)
    angle = build_angle_matrix(vectors, epsilon=epsilon)
    norm_angle = angle.sum(dim=-1)
    distance = build_distance_matrix(vectors, epsilon=epsilon)
    norm_distance = distance.sum(dim=-1)
    return (pattern_similarity * (-torch.log(angle / norm_angle) - torch.log(distance / norm_distance))).mean()


def toy_mc_inter_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Historical HyperTagging/noR inter-channel embedding loss."""

    channel = dataset["channel"].to(vectors.device)
    mask = channel[:, None] == channel
    angle = build_angle_matrix(vectors, epsilon=epsilon)
    norm_angle = torch.einsum("ij,ij->j", angle, (~mask).float()) + epsilon
    distance = build_distance_matrix(vectors, epsilon=epsilon)
    norm_distance = torch.einsum("ij,ij->j", distance, (~mask).float()) + epsilon
    return ((-torch.log(angle / norm_angle) - torch.log(distance / norm_distance))[mask]).mean()


def colab_intra_loss(vectors: torch.Tensor, dataset: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """HyperTaggingColab masked grouped-variance intra loss."""

    mask = dataset["padding_mask"].to(vectors.device)
    masked_emb = vectors * mask.unsqueeze(-1)
    mask_weight = mask.shape[-1] / mask.sum(dim=-1)
    grouped_var = (
        (masked_emb - masked_emb.mean(dim=-2, keepdim=True) * mask_weight[:, None, None]) ** 2
        * mask.unsqueeze(-1)
        * mask_weight[:, None, None]
    ).mean()
    return grouped_var


def vicreg_loss(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    weights: Sequence[float] = (1, 10, 10),
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Historical VICReg-style embedding loss from Colab/RegEmbedding variants."""

    pattern = dataset["pattern"].to(vectors.device).float()
    left = pattern.unsqueeze(1).repeat(1, len(pattern), 1)
    right = pattern.unsqueeze(0).repeat(len(pattern), 1, 1)
    similarity = F.cosine_similarity(left, right, dim=-1, eps=epsilon)
    norm = similarity.sum(dim=0).unsqueeze(-1)
    repeated = vectors.unsqueeze(1).repeat(1, len(vectors), 1)
    sim_loss = (build_distance_matrix(vectors, epsilon=epsilon) * similarity).mean()
    centered = repeated - (torch.einsum("ijk,ij->jk", repeated, similarity) / norm).unsqueeze(0)
    std = torch.sqrt(torch.einsum("ijk,ij->jk", centered**2, similarity) / (norm - 1) + epsilon)
    std_loss = torch.mean(F.relu(1 - std))
    eye = torch.eye(centered.size(-1), device=vectors.device)
    cov = (centered.transpose(-1, -2) @ (centered * similarity.unsqueeze(-1)) - eye) / (
        norm.unsqueeze(-1) - 1
    )
    cov_loss = cov.pow(2).sum() / centered.size(-1)
    return weights[0] * sim_loss + weights[1] * std_loss + weights[2] * cov_loss


def connection_loss_from_embeddings(
    vectors: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """DoubleEmbedding connection loss from pairwise embedding cosine similarity."""

    left = vectors.unsqueeze(2).repeat(1, 1, vectors.shape[1], 1)
    right = vectors.unsqueeze(1).repeat(1, vectors.shape[1], 1, 1)
    predictions = (F.cosine_similarity(left, right, dim=-1) + 1) / 2
    return connection_loss_from_predictions(predictions, dataset, already_scaled=True)


def connection_loss_from_predictions(
    predictions: torch.Tensor,
    dataset: Mapping[str, torch.Tensor],
    *,
    already_scaled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ParticleCombination connection loss for pairwise connection predictions."""

    if not already_scaled:
        predictions = (predictions + 1) / 2
    mask = dataset["padding_mask"].to(predictions.device)
    index = dataset["links"].to(predictions.device)
    padding_mtx = boolean_matmul(mask.unsqueeze(-1), mask.unsqueeze(-2))
    connection_mtx = torch.stack([(idx[..., None] == idx).int() for idx in index]) * padding_mtx
    n_connections = connection_mtx.sum(dim=(-2, -1))[:, None, None]
    n_total = padding_mtx.sum(dim=(-2, -1))[:, None, None]
    weights = n_total * (padding_mtx / n_connections + ~padding_mtx / torch.clamp(n_total - n_connections, min=1))
    acc = (torch.eq(connection_mtx, predictions > 0.6) * padding_mtx).sum() / padding_mtx.sum()
    loss = F.binary_cross_entropy(predictions, connection_mtx.float(), weight=weights)
    return loss, acc


class CompositeLoss:
    """Historical Colab weighted loss combiner."""

    def __init__(self, losses: Mapping[str, tuple[Any, float]] | None = None):
        self.losses = dict(losses or {})

    def add_loss(self, name: str, loss_fn: Any, weight: float = 1.0) -> None:
        self.losses[name] = (loss_fn, weight)

    def update_weight(self, name: str, weight: float) -> None:
        if name not in self.losses:
            raise ValueError(f"Loss '{name}' not found. Please add it first with add_loss().")
        loss_fn, _ = self.losses[name]
        self.losses[name] = (loss_fn, weight)

    def __call__(self, output: Any, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        loss_dict = {}
        total_loss: torch.Tensor | float = 0.0
        for name, (loss_fn, weight) in self.losses.items():
            loss_value = loss_fn(output, batch)
            loss_dict[name] = loss_value.detach().cpu().numpy()
            total_loss = total_loss + weight * loss_value
        assert isinstance(total_loss, torch.Tensor)
        return total_loss, loss_dict
