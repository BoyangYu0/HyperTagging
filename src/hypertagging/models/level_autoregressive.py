"""Level-autoregressive set reconstruction model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hypertagging.models.hyperbolic import HyperbolicNodeEncoder
from hypertagging.models.mother_pointer import MotherPointerDecoder, MotherPointerOutput
from hypertagging.models.relations import RelationBias
from hypertagging.models.stair_masks import context_mask_for_level


@dataclass(frozen=True)
class LevelReconstructionOutput:
    target_level: int
    pointer: MotherPointerOutput
    node_embeddings: torch.Tensor
    hyperbolic_embeddings: torch.Tensor
    context_mask: torch.Tensor


class LevelAutoregressiveReconstructor(nn.Module):
    """Tree-biased encoder plus pointer-based mother set decoder."""

    def __init__(
        self,
        *,
        n_features: int,
        n_types: int,
        hidden_dim: int = 32,
        hyper_dim: int = 16,
        n_queries: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = HyperbolicNodeEncoder(
            n_features=n_features,
            n_pid=max(n_types, 4096),
            hidden_dim=hidden_dim,
            hyper_dim=hyper_dim,
        )
        self.relation_bias = RelationBias(hidden_dim=hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = MotherPointerDecoder(hidden_dim=hidden_dim, n_types=n_types, n_queries=n_queries)

    def forward(self, batch: dict[str, torch.Tensor], *, target_level: int = 1) -> LevelReconstructionOutput:
        h, z = self.encoder(batch["node_features"], batch["pid_labels"], batch["level_ids"], batch["charge"])
        context_mask = context_mask_for_level(batch["level_ids"], batch["node_mask"], target_level)
        _bias = self.relation_bias(
            p4=batch["p4"],
            charge=batch["charge"],
            level_ids=batch["level_ids"],
            z_hyperbolic=z,
            node_mask=batch["node_mask"],
        )
        # The compact CPU implementation keeps relation bias available for loss
        # diagnostics while using a simple projected context for decoding.
        h = self.context_projection(h)
        pointer = self.decoder(h, context_mask)
        return LevelReconstructionOutput(target_level, pointer, h, z, context_mask)


def construct_mother_p4(pointer_logits: torch.Tensor, p4: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
    """Construct mother p4 from daughter pointers, never from MC mother p4."""

    weights = (pointer_logits > 0).float() if hard else torch.sigmoid(pointer_logits)
    return torch.einsum("bqn,bnf->bqf", weights, p4)
