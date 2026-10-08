"""One-sided relation-gradient projection preserving membership and decoder losses."""

from __future__ import annotations
import torch


def relation_correction(membership, relation):
    """Return encoder correction only; absent gradients behave as zero vectors."""
    pairs = [
        (a, b) for a, b in zip(membership, relation) if a is not None and b is not None
    ]
    dot = sum(float((a.detach().float() * b.detach().float()).sum()) for a, b in pairs)
    norm2 = sum(
        float(a.detach().float().square().sum()) for a in membership if a is not None
    )
    coefficient = -min(dot, 0.0) / max(norm2, 1e-20)
    correction = [None if a is None else a.detach() * coefficient for a in membership]
    return correction, {
        "dot_before": dot,
        "dot_after": dot + coefficient * norm2,
        "conflict": int(dot < 0),
        "membership_norm2": norm2,
    }


def encoder_correction(member, relation, parameters):
    a = torch.autograd.grad(member, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(relation, parameters, retain_graph=True, allow_unused=True)
    return relation_correction(a, b)
