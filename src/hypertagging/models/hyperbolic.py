"""Stable Poincare-ball utilities and a small hyperbolic node encoder."""

from __future__ import annotations

import torch
from torch import nn

from hypertagging.preprocessing.pid_filter import PDG_TOKENS, validate_pid_tokens


HYPERBOLIC_SCALE_CONTRACT_VERSION = "dimension-aware-tangent-radius-v2"


def initialize_hyper_projection(linear: nn.Linear, *, output_std: float) -> None:
    """Initialize tangent coordinates at a dimension-independent scale."""

    if output_std <= 0:
        raise ValueError("hyper_projection_init_scale must be positive")
    nn.init.normal_(linear.weight, std=float(output_std) / linear.in_features**0.5)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class BoundedTangentScale(nn.Module):
    """Optional bounded scalar on tangent vectors before ``expmap0``."""

    def __init__(
        self,
        *,
        mode: str = "fixed",
        initial: float = 1.0,
        minimum: float = 0.25,
        maximum: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in {"fixed", "learned_bounded"}:
            raise ValueError(f"unknown tangent scale mode: {mode}")
        if not 0 < minimum <= initial <= maximum:
            raise ValueError("tangent scale bounds must contain the positive initial value")
        self.mode = mode
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        fraction = (float(initial) - minimum) / max(maximum - minimum, 1e-12)
        logit = torch.logit(torch.tensor(fraction).clamp(1e-6, 1 - 1e-6))
        if mode == "learned_bounded":
            self.logit = nn.Parameter(logit)
        else:
            self.register_buffer("logit", logit)

    def value(self) -> torch.Tensor:
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(self.logit)

    def forward(self, tangent: torch.Tensor) -> torch.Tensor:
        return tangent * self.value().to(tangent)


def project(x: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Project points inside the Poincare ball in explicit FP32."""

    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = x.float()
        max_norm = (1.0 - eps) / curvature**0.5
        norm = torch.linalg.norm(x32, dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x32 * scale


def expmap0(v: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map at the origin of the Poincare ball in explicit FP32."""

    with torch.autocast(device_type=v.device.type, enabled=False):
        v32 = v.float()
        sqrt_c = curvature**0.5
        norm = torch.linalg.norm(v32, dim=-1, keepdim=True).clamp_min(eps)
        return project(
            torch.tanh(sqrt_c * norm) * v32 / (sqrt_c * norm),
            curvature=curvature,
        )


def logmap0(x: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Logarithmic map at the origin of the Poincare ball in explicit FP32."""

    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = project(x.float(), curvature=curvature)
        sqrt_c = curvature**0.5
        norm = torch.linalg.norm(x32, dim=-1, keepdim=True).clamp_min(eps)
        return (
            torch.atanh((sqrt_c * norm).clamp(max=1 - eps))
            * x32
            / (sqrt_c * norm)
        )


def distance(x: torch.Tensor, y: torch.Tensor, *, curvature: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise Poincare distance with broadcasting support in FP32."""

    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = project(x.float(), curvature=curvature)
        y32 = project(y.float(), curvature=curvature)
        sqrt_c = curvature**0.5
        diff_norm = torch.linalg.vector_norm(x32 - y32, dim=-1)
        x2 = (x32**2).sum(dim=-1).clamp(max=(1 - eps) / curvature)
        y2 = (y32**2).sum(dim=-1).clamp(max=(1 - eps) / curvature)
        denom = ((1 - curvature * x2) * (1 - curvature * y2)).clamp_min(eps)
        # The equivalent asinh form has finite derivatives at coincident points;
        # acosh(1 + delta) has an infinite derivative at delta=0.
        argument = sqrt_c * diff_norm / torch.sqrt(denom)
        return 2.0 * torch.asinh(argument) / sqrt_c


def radius(x: torch.Tensor, *, curvature: float = 1.0) -> torch.Tensor:
    """Distance from the origin."""

    return distance(torch.zeros_like(x), x, curvature=curvature)


class HyperbolicNodeEncoder(nn.Module):
    """Encode levelized node features into Euclidean and hyperbolic embeddings."""

    def __init__(
        self,
        *,
        n_features: int,
        n_pid: int = len(PDG_TOKENS),
        hidden_dim: int = 32,
        hyper_dim: int = 16,
        max_level: int = 16,
        curvature: float = 1.0,
        hyper_projection_init_scale: float = 0.05,
        tangent_scale_mode: str = "fixed",
    ) -> None:
        super().__init__()
        self.curvature = curvature
        if n_pid != len(PDG_TOKENS):
            n_pid = len(PDG_TOKENS)
        self.pid_embedding = nn.Embedding(n_pid, hidden_dim)
        self.level_embedding = nn.Embedding(max_level + 2, hidden_dim)
        self.feature_projection = nn.Linear(n_features + 1, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.hyper_projection = nn.Linear(hidden_dim, hyper_dim)
        initialize_hyper_projection(
            self.hyper_projection, output_std=hyper_projection_init_scale
        )
        self.tangent_scale = BoundedTangentScale(mode=tangent_scale_mode)

    def forward(
        self,
        node_features: torch.Tensor,
        pid_labels: torch.Tensor,
        level_ids: torch.Tensor,
        charge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level_safe = level_ids.clamp(min=0, max=self.level_embedding.num_embeddings - 1)
        validate_pid_tokens(pid_labels, name="flat encoder PID labels")
        pid_safe = pid_labels
        base = self.feature_projection(torch.cat([node_features, charge.unsqueeze(-1)], dim=-1))
        h = self.mlp(torch.cat([base, self.pid_embedding(pid_safe), self.level_embedding(level_safe)], dim=-1))
        z = expmap0(
            self.tangent_scale(self.hyper_projection(h)),
            curvature=self.curvature,
        )
        return h, z


def hyperbolic_pairwise_distance(
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Return [B, N, N] hyperbolic distances."""

    d = distance(z[:, :, None, :], z[:, None, :, :], curvature=curvature)
    if mask is not None:
        d = d.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.0)
    return d
