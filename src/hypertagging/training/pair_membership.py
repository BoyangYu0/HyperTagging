"""Training-only permutation-invariant B partition supervision.

No targets enter the decoder. Pair categories are same B, cross B, or at least
one unassigned/background node. Shared detector sources and unknown targets are
excluded explicitly; they are not background examples.
"""

from __future__ import annotations

import torch


def pair_membership_loss(logits, targets, sources):
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("Expected [nodes, unassigned+B1+B2] logits")
    if targets.ndim != 1 or len(targets) != len(logits) or len(sources) != len(logits):
        raise ValueError("Pair source/target axis mismatch")
    if not ((targets >= -1) & (targets <= 2)).all():
        raise ValueError("Unknown target encoding")
    if any(not s for s in sources):
        raise ValueError("Missing detector-source support")
    source_sets = [set(s) for s in sources]
    pairs, labels = [], []
    support = {
        "same_b": 0,
        "cross_b": 0,
        "background": 0,
        "unknown_pairs": 0,
        "shared_source_pairs": 0,
    }
    truth = targets.detach().cpu().tolist()
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if source_sets[i] & source_sets[j]:
                support["shared_source_pairs"] += 1
                continue
            if truth[i] < 0 or truth[j] < 0:
                support["unknown_pairs"] += 1
                continue
            label = 2 if min(truth[i], truth[j]) == 0 else int(truth[i] != truth[j])
            pairs.append((i, j))
            labels.append(label)
            support[("same_b", "cross_b", "background")[label]] += 1
    if not pairs:
        return logits.float().sum() * 0, support
    indices = torch.tensor(pairs, device=logits.device)
    labels = torch.tensor(labels, device=logits.device)
    logp = logits.float().log_softmax(-1)
    a, b = logp[indices[:, 0]], logp[indices[:, 1]]
    same = torch.logaddexp(a[:, 1] + b[:, 1], a[:, 2] + b[:, 2])
    cross = torch.logaddexp(a[:, 1] + b[:, 2], a[:, 2] + b[:, 1])
    # Disjoint cases: left background, or left B and right background.
    background = torch.logaddexp(a[:, 0], a[:, 1:].logsumexp(-1) + b[:, 0])
    nll = -torch.stack((same, cross, background), -1).gather(1, labels[:, None])[:, 0]
    # Event mean over available relation classes. No global B-slot identity.
    loss = torch.stack(
        [nll[labels == c].mean() for c in range(3) if (labels == c).any()]
    ).mean()
    return loss, support


def proposal_refinement_pair_loss(result, targets, sources):
    proposal, support = pair_membership_loss(
        result["proposal_logits"], targets, sources
    )
    refinement, other = pair_membership_loss(result["logits"], targets, sources)
    assert support == other
    return (proposal + refinement) / 2, support
