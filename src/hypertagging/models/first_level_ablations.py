"""Wired, disabled-by-default first-level ambiguity helpers.

The former whole-set scorer and iterative-pointer helpers were deliberately
removed: neither had a training target or a model/decoder consumer.  Their
scientific designs remain documented in ``docs/deferred_model_ablations.md``.
"""

from __future__ import annotations

import torch


def type_conditioned_relation_bias(
    type_probabilities: torch.Tensor,
    type_relation_table: torch.Tensor,
    node_pid_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Expected soft mother-type/candidate-PID compatibility bias.

    No PID or mother type is pruned.  Both distributions remain differentiable,
    and the learned table contributes directly to pointer logits.
    """

    return torch.einsum(
        "bqt,tu,bnu->bqn",
        type_probabilities,
        type_relation_table,
        node_pid_probabilities,
    )


__all__ = [
    "type_conditioned_relation_bias",
]
