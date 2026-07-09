"""Stable Poincare-ball utilities and a small hyperbolic node encoder."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def project(x: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Project points inside the Poincare ball."""

    max_norm = (1.0 - eps) / curvature**0.5
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def expmap0(v: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map at the origin of the Poincare ball."""

    sqrt_c = curvature**0.5
    norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return project(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), curvature=curvature)


def logmap0(x: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Logarithmic map at the origin of the Poincare ball."""

    x = project(x, curvature=curvature)
    sqrt_c = curvature**0.5
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return torch.atanh((sqrt_c * norm).clamp(max=1 - eps)) * x / (sqrt_c * norm)


def distance(x: torch.Tensor, y: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise Poincare distance with broadcasting support."""

    x = project(x, curvature=curvature)
    y = project(y, curvature=curvature)
    sqrt_c = curvature**0.5
    diff2 = ((x - y) ** 2).sum(dim=-1)
    x2 = (x**2).sum(dim=-1).clamp(max=(1 - eps) / curvature)
    y2 = (y**2).sum(dim=-1).clamp(max=(1 - eps) / curvature)
    denom = ((1 - curvature * x2) * (1 - curvature * y2)).clamp_min(eps)
    z = 1 + 2 * curvature * diff2 / denom
    return torch.acosh(z.clamp_min(1 + eps)) / sqrt_c


def radius(x: torch.Tensor, *, curvature: float = 1.0) -> torch.Tensor:
    """Distance from the origin."""

    return distance(torch.zeros_like(x), x, curvature=curvature)


class HyperbolicNodeEncoder(nn.Module):
    """Encode levelized node features into Euclidean and hyperbolic embeddings."""

    def __init__(
        self,
        *,
        n_features: int,
        n_pid: int = 4096,
        hidden_dim: int = 32,
        hyper_dim: int = 16,
        max_level: int = 16,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.curvature = curvature
        self.pid_embedding = nn.Embedding(n_pid, hidden_dim)
        self.level_embedding = nn.Embedding(max_level + 2, hidden_dim)
        self.feature_projection = nn.Linear(n_features + 1, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.hyper_projection = nn.Linear(hidden_dim, hyper_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        pid_labels: torch.Tensor,
        level_ids: torch.Tensor,
        charge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level_safe = level_ids.clamp(min=0, max=self.level_embedding.num_embeddings - 1)
        pid_safe = pid_labels.abs().clamp(max=self.pid_embedding.num_embeddings - 1)
        base = self.feature_projection(torch.cat([node_features, charge.unsqueeze(-1)], dim=-1))
        h = self.mlp(torch.cat([base, self.pid_embedding(pid_safe), self.level_embedding(level_safe)], dim=-1))
        z = expmap0(self.hyper_projection(h), curvature=self.curvature)
        return h, z


def hyperbolic_pairwise_distance(z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return [B, N, N] hyperbolic distances."""

    d = distance(z[:, :, None, :], z[:, None, :, :])
    if mask is not None:
        d = d.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.0)
    return d
