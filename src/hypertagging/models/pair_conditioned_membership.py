"""Conditional development decoder: one semantic message-to-refinement connection.

Native detached merge selection remains unchanged, including native index-based
merge tie breaks. Semantic edges/messages are permutation equivariant. No truth
is accepted by forward; semantic supervision is a separate post-forward loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hypertagging.models.assembly_development import AssemblyMembershipDecoder


class PairConditionedMembershipDecoder(AssemblyMembershipDecoder):
    def __init__(self, *, connection_enabled: bool, addition_seed: int):
        if type(connection_enabled) is not bool or type(addition_seed) is not int:
            raise ValueError(
                "Explicit boolean connection and integer addition seed required"
            )
        super().__init__()
        self.connection_enabled = connection_enabled
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(addition_seed)
            self.semantic_edges = nn.Sequential(
                nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 3)
            )
            self.message = nn.Sequential(
                nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 256)
            )
            nn.init.zeros_(self.message[-1].weight)
            nn.init.zeros_(self.message[-1].bias)

    def semantic_message(self, features, sources):
        if (
            features.ndim != 2
            or features.shape[1] != 128
            or not 1 <= len(features) <= 256
        ):
            raise ValueError("Context128 and1..256 detector nodes required")
        if not torch.isfinite(features).all():
            raise ValueError("Finite detector features required")
        if (
            sources.ndim != 2
            or len(sources) != len(features)
            or sources.dtype != torch.bool
            or sources.device != features.device
        ):
            raise ValueError("Boolean detector source masks on feature device required")
        pairs = torch.triu_indices(
            len(features), len(features), 1, device=features.device
        )
        valid = (
            sources[pairs[0]].any(-1)
            & sources[pairs[1]].any(-1)
            & ~(sources[pairs[0]] & sources[pairs[1]]).any(-1)
        )
        pairs = pairs[:, valid]
        a, b = features[pairs[0]], features[pairs[1]]
        edge_input = torch.cat(
            ((a - b).abs(), a * b, features.mean(0).expand(len(a), -1)), -1
        )
        logits = self.semantic_edges(edge_input)
        probability = logits.softmax(-1)
        h = self.interface(features)
        sums = h.new_zeros((len(h), 3, 256))
        weights = h.new_zeros((len(h), 3))
        for left, right in ((pairs[0], pairs[1]), (pairs[1], pairs[0])):
            sums = sums.index_add(0, left, probability[..., None] * h[right, None])
            weights = weights.index_add(0, left, probability)
        aggregates = sums / weights.clamp_min(1e-6)[..., None]
        message = self.message(aggregates.reshape(len(h), 768))
        # Biases must never fabricate neighbor information on unsupported nodes.
        message = message * (weights.sum(-1) > 0)[:, None]
        return message, logits, pairs

    def forward(self, features, sources, *, group_conditioning=True):
        message, edge_logits, edge_pairs = self.semantic_message(features, sources)
        h = self.interface(features)
        mask = torch.ones((1, len(h)), dtype=torch.bool, device=h.device)
        proposal_logits, proposal_objects = self.proposal(h[None], mask)
        probability = proposal_logits[0].softmax(-1)[:, 1:]
        representations = (
            probability.T @ h / probability.sum(0).clamp_min(1e-6)[:, None]
        )
        predicted_context = probability @ representations
        states = self.generate(h, sources)
        partial_context = torch.zeros_like(h)
        for group in states[-1]["groups"]:
            positions = list(group)
            partial_context[positions] = h[positions].mean(0)
        if not self.partial_state_conditioning:
            partial_context = torch.zeros_like(partial_context)
        if not group_conditioning:
            predicted_context = torch.zeros_like(predicted_context)
            partial_context = torch.zeros_like(partial_context)
        conditioned = h + self.condition(
            torch.cat((h, predicted_context, partial_context), -1)
        )
        if self.connection_enabled:
            conditioned = conditioned + message
        logits, objects = self.refinement(conditioned[None], mask)
        return dict(
            logits=logits,
            objects=objects,
            proposal_logits=proposal_logits,
            proposal_objects=proposal_objects,
            states=states,
            edge_logits=edge_logits,
            edge_pairs=edge_pairs,
        )


def semantic_pair_loss(logits, pairs, targets):
    """Mean over supported class CE means; source masks applied before generation."""
    if (
        targets.ndim != 1
        or targets.dtype != torch.long
        or not ((targets >= -1) & (targets <= 2)).all()
    ):
        raise ValueError(
            "Membership targets must be unknown/background/two unordered B slots"
        )
    if (
        pairs.ndim != 2
        or pairs.shape[0] != 2
        or pairs.dtype != torch.long
        or logits.shape != (pairs.shape[1], 3)
        or pairs.device != targets.device
        or logits.device != targets.device
    ):
        raise ValueError("Edge output contract changed")
    if (
        pairs.numel()
        and (pairs.min() < 0 or pairs.max() >= len(targets))
        or not (pairs[0] < pairs[1]).all()
        or torch.unique(pairs, dim=1).shape[1] != pairs.shape[1]
    ):
        raise ValueError("Expected unique unordered valid detector pairs")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite edge logits")
    a, b = targets[pairs[0]], targets[pairs[1]]
    known = (a >= 0) & (b >= 0)
    labels = torch.where((a == 0) | (b == 0), 2, (a != b).long())
    possible = len(targets) * (len(targets) - 1) // 2
    counts = dict(
        possible_pairs=possible,
        source_excluded_pairs=possible - pairs.shape[1],
        unknown_pairs=int((~known).sum()),
        eligible_pairs=pairs.shape[1],
    )
    values = []
    for c, name in enumerate(("same_b", "cross_b", "background")):
        mask = known & (labels == c)
        counts[name] = int(mask.sum())
        if mask.any():
            values.append(F.cross_entropy(logits[mask].float(), labels[mask]))
    return (torch.stack(values).mean() if values else logits.sum() * 0), counts
