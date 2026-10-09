"""Development-only latent assembly; no physical mothers or hierarchy claims.

Generated states are partitions of detector FSP indices. Every merge is selected
by model scores with detector-source disjointness, never by targets. Their pooled
representations condition the shared membership decoder. No teacher composites,
truth labels, kinematics, PID or geometry are accepted by this module.
"""

from __future__ import annotations
from itertools import combinations
import torch
from torch import nn
from torch.nn import functional as F
from hypertagging.models.direct_membership import DirectMembershipHead


class AssemblyMembershipDecoder(nn.Module):
    width = 256

    def __init__(self, *, partial_state_conditioning=True):
        super().__init__()
        if type(partial_state_conditioning) is not bool:
            raise ValueError("partial_state_conditioning must be boolean")
        self.partial_state_conditioning = partial_state_conditioning
        self.proposal = DirectMembershipHead(256, 256)
        self.relations = nn.Sequential(
            nn.Linear(770, 256), nn.GELU(), nn.Linear(256, 3)
        )
        self.condition = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 256)
        )
        self.refinement = DirectMembershipHead(256, 256)

    @staticmethod
    def interface(features):
        if features.ndim != 2 or features.shape[-1] not in (128, 256):
            raise ValueError("Only128/256 contextual features are admitted")
        return F.pad(features, (0, 256 - features.shape[-1]))

    def pair_scores(self, h, groups, sources):
        pooled = torch.stack([h[list(group)].mean(0) for group in groups])
        masks = [sources[list(group)].any(0) for group in groups]
        pairs = [
            (i, j)
            for i, j in combinations(range(len(groups)), 2)
            if not bool((masks[i] & masks[j]).any())
        ]
        if not pairs:
            return h.new_zeros((0, 3)) + h.sum() * 0, pairs
        i, j = torch.tensor(pairs, device=h.device).T
        left, right = pooled[i], pooled[j]
        sizes = h.new_tensor(
            [[len(groups[a]), len(groups[b])] for a, b in pairs]
        ).log1p()
        values = torch.cat(
            (left + right, (left - right).abs(), left * right, sizes), -1
        )
        return self.relations(values), pairs

    def generate(self, h, sources):
        if sources.ndim != 2 or len(sources) != len(h):
            raise ValueError("Detector provenance shape mismatch")
        groups = [frozenset([i]) for i in range(len(h))]
        states = []
        for step in range(3):
            scores, pairs = self.pair_scores(h, groups, sources)
            states.append(
                {"groups": tuple(groups), "pairs": tuple(pairs), "logits": scores}
            )
            if step == 2 or not pairs:
                break
            # Fixed two rounds, at most four disjoint merges per round. This is
            # a latent partial-membership proposal, not a physical mother.
            ranking = sorted(
                range(len(pairs)),
                key=lambda k: (-float(scores[k, 0].detach()), pairs[k]),
            )
            used, merged = set(), []
            for k in ranking:
                i, j = pairs[k]
                if i in used or j in used:
                    continue
                merged.append(groups[i] | groups[j])
                used.update((i, j))
                if len(merged) == 4:
                    break
            groups = [g for i, g in enumerate(groups) if i not in used] + merged
        return states

    def forward(self, features, sources, *, group_conditioning=True):
        h = self.interface(features)
        if not len(h) or len(h) > 256:
            raise ValueError("Development FSP capacity is1..256; never truncate")
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
        logits, objects = self.refinement(conditioned[None], mask)
        return {
            "logits": logits,
            "objects": objects,
            "proposal_logits": proposal_logits,
            "proposal_objects": proposal_objects,
            "states": states,
        }


def relation_targets(groups, pairs, supervision):
    """Post-generation labels: exact two-child parent, partial siblings, same-B other.

    Legitimate relatives are discriminated categorically, never globally repelled.
    Ambiguous source-set mappings and unknown B membership are unavailable.
    """
    sides = supervision["b_groups"]
    node_sets = supervision["node_sets"]
    parents = supervision["parents"]
    result = []
    for i, j in pairs:
        a, b = groups[i], groups[j]
        same_b = any(a <= side and b <= side for side in sides)
        if not same_b:
            result.append(-1)
            continue
        ai = [k for k, s in enumerate(node_sets) if s == a]
        bi = [k for k, s in enumerate(node_sets) if s == b]
        if len(ai) != 1 or len(bi) != 1:
            result.append(-1)
            continue
        pa, pb = parents[ai[0]], parents[bi[0]]
        if pa >= 0 and pa == pb:
            result.append(0 if node_sets[pa] == a | b else 1)
        else:
            result.append(2)
    return result


def assembly_relation_loss(states, supervision):
    values, counts = [], [0, 0, 0]
    for state in states:
        logits = state["logits"]
        target = torch.tensor(
            relation_targets(state["groups"], state["pairs"], supervision),
            device=logits.device,
        )
        for label in range(3):
            mask = target == label
            counts[label] += int(mask.sum())
            if bool(mask.any()):
                values.append(F.cross_entropy(logits[mask].float(), target[mask]))
    zero = states[0]["logits"].sum() * 0
    return (torch.stack(values).mean() if values else zero), counts
