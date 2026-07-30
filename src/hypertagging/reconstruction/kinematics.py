"""Authoritative reconstructed four-momentum construction.

Truth labels never enter this module.  Track momentum is a reconstructed fit
measurement, cluster four-momentum is reconstructed calorimeter information,
and every composite is an exact daughter sum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from hypertagging.preprocessing.pid_filter import (
    DETOKENIZE_DICT,
    PDG_TOKENS,
    TOKENIZE_DICT,
    validate_pid_token,
    validate_pid_tokens,
)


CHARGED_STABLE_PDGS: tuple[int, ...] = (11, 13, 211, 321, 2212)
CHARGED_STABLE_NAMES: tuple[str, ...] = ("electron", "muon", "pion", "kaon", "proton")
PARTICLE_MASSES_GEV: Mapping[int, float] = {
    11: 0.00051099895,
    13: 0.1056583755,
    211: 0.13957039,
    321: 0.493677,
    2212: 0.93827208816,
}
PARTICLE_CHARGES: Mapping[int, int] = {
    11: -1,
    -11: 1,
    13: -1,
    -13: 1,
    211: 1,
    -211: -1,
    321: 1,
    -321: -1,
    2212: 1,
    -2212: -1,
}
CANONICAL_TRACK_HYPOTHESIS = "pion"
CANONICAL_TRACK_TOKEN = TOKENIZE_DICT[211]


def _as_p3(p3: torch.Tensor) -> torch.Tensor:
    if p3.shape[-1] != 3:
        raise ValueError(f"track p3 must end in dimension 3, got {tuple(p3.shape)}")
    if not torch.isfinite(p3).all():
        raise ValueError("track p3 contains NaN or infinite values")
    return p3


def track_energy_hypotheses(p3: torch.Tensor) -> torch.Tensor:
    """Return deterministic ``(e, mu, pi, K, p)`` energies for one reco p3."""

    p3 = _as_p3(p3)
    masses = p3.new_tensor([PARTICLE_MASSES_GEV[pdg] for pdg in CHARGED_STABLE_PDGS])
    momentum2 = p3.square().sum(dim=-1, keepdim=True)
    return torch.sqrt(momentum2 + masses.square())


def canonical_track_p4(p3: torch.Tensor) -> torch.Tensor:
    """Return the data-independent default pion-hypothesis track p4."""

    p3 = _as_p3(p3)
    energy = track_energy_hypotheses(p3)[..., 2:3]
    return torch.cat((p3, energy), dim=-1)


def _mass_by_reduced_token(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    masses = []
    for token in range(len(PDG_TOKENS)):
        pdg = abs(DETOKENIZE_DICT[token])
        masses.append(PARTICLE_MASSES_GEV.get(pdg, float("nan")))
    return torch.tensor(masses, device=device, dtype=dtype)


def hard_track_p4_from_pid_token(p3: torch.Tensor, pid_token: torch.Tensor | int) -> torch.Tensor:
    """Build hard track p4 using a predicted charged-stable reduced token."""

    p3 = _as_p3(p3)
    tokens = torch.as_tensor(pid_token, device=p3.device, dtype=torch.long)
    validate_pid_tokens(tokens, name="hard track PID token")
    masses = _mass_by_reduced_token(device=p3.device, dtype=p3.dtype)[tokens]
    if not torch.isfinite(masses).all():
        invalid = tokens[~torch.isfinite(masses)].detach().cpu().tolist()
        raise ValueError(f"non charged-stable track PID token(s): {invalid}")
    target_shape = p3.shape[:-1]
    masses = torch.broadcast_to(masses, target_shape)
    energy = torch.sqrt(p3.square().sum(dim=-1) + masses.square()).unsqueeze(-1)
    return torch.cat((p3, energy), dim=-1)


def soft_track_p4_from_pid_logits(
    p3: torch.Tensor,
    pid_logits: torch.Tensor,
    *,
    allowed_tokens: Sequence[int] | None = None,
) -> torch.Tensor:
    """Differentiable mixture of charged-stable energy hypotheses.

    The default support contains both charge signs of e/mu/pi/K/p.  Momentum
    and charge are not averaged; only the mass-dependent energy is mixed.
    """

    p3 = _as_p3(p3)
    if pid_logits.shape[:-1] != p3.shape[:-1] or pid_logits.shape[-1] != len(PDG_TOKENS):
        raise ValueError(
            "pid_logits must have shape p3.shape[:-1] + "
            f"({len(PDG_TOKENS)},), got {tuple(pid_logits.shape)}"
        )
    if allowed_tokens is None:
        allowed_tokens = tuple(
            token
            for pdg, token in TOKENIZE_DICT.items()
            if abs(pdg) in PARTICLE_MASSES_GEV
        )
    checked = [validate_pid_token(token, name="allowed track PID token") for token in allowed_tokens]
    if not checked:
        raise ValueError("allowed track PID token set is empty")
    selected_logits = pid_logits[..., checked]
    probabilities = torch.softmax(selected_logits, dim=-1)
    masses = _mass_by_reduced_token(device=p3.device, dtype=p3.dtype)[checked]
    energies = torch.sqrt(p3.square().sum(dim=-1, keepdim=True) + masses.square())
    energy = (probabilities * energies).sum(dim=-1, keepdim=True)
    return torch.cat((p3, energy), dim=-1)


def cluster_reco_p4(
    *,
    energy: torch.Tensor,
    direction: torch.Tensor | None = None,
    p3: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct a reconstructed massless cluster p4.

    A measured p3 may be supplied directly.  Otherwise a reconstructed
    direction is normalized and multiplied by the measured cluster energy.
    """

    if p3 is None:
        if direction is None or direction.shape[-1] != 3:
            raise ValueError("cluster_reco_p4 requires p3 or a 3-vector direction")
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if torch.any(norm <= 0):
            raise ValueError("cluster direction must have nonzero norm")
        p3 = direction / norm * energy.unsqueeze(-1) if energy.ndim == direction.ndim - 1 else direction / norm * energy
    p3 = _as_p3(p3)
    energy_column = energy.unsqueeze(-1) if energy.ndim == p3.ndim - 1 else energy
    if energy_column.shape != p3.shape[:-1] + (1,):
        raise ValueError("cluster energy shape is incompatible with p3")
    return torch.cat((p3, energy_column), dim=-1)


def composite_p4_from_daughters(
    daughter_p4: torch.Tensor,
    daughter_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact daughter sum along the penultimate dimension."""

    if daughter_p4.shape[-1] != 4:
        raise ValueError("daughter p4 must end in dimension 4")
    if daughter_mask is not None:
        if daughter_mask.shape != daughter_p4.shape[:-1]:
            raise ValueError("daughter mask shape does not match daughter p4")
        daughter_p4 = daughter_p4 * daughter_mask.to(daughter_p4.dtype).unsqueeze(-1)
    return daughter_p4.sum(dim=-2)


def soft_reconstructed_p4_from_leaf_pid(
    batch: Mapping[str, torch.Tensor],
    pid_logits: torch.Tensor,
) -> torch.Tensor:
    """Differentiably rebuild raw-track energies and every composite p4."""

    return _reconstructed_p4_from_leaf_pid(batch, pid_logits, hard=False)


def hard_reconstructed_p4_from_leaf_pid(
    batch: Mapping[str, torch.Tensor],
    pid_logits: torch.Tensor,
) -> torch.Tensor:
    """Hard-decode charge-compatible raw-track PID and recursively rebuild p4."""

    return _reconstructed_p4_from_leaf_pid(batch, pid_logits, hard=True)


def _reconstructed_p4_from_leaf_pid(
    batch: Mapping[str, torch.Tensor],
    pid_logits: torch.Tensor,
    *,
    hard: bool,
) -> torch.Tensor:
    p4 = batch["p4"].clone()
    raw_track = (
        batch["node_mask"].bool()
        & (batch["node_kind_ids"] == 1)
        & (batch["level_ids"] == 0)
        & (batch["pid_labels"] == 0)
        & (batch["charge"] != 0)
    )
    all_stable_pdgs = CHARGED_STABLE_PDGS + tuple(
        -pdg for pdg in CHARGED_STABLE_PDGS
    )
    for charge_sign in (-1, 1):
        selected = raw_track & (
            (batch["charge"] < 0) if charge_sign < 0 else (batch["charge"] > 0)
        )
        if not selected.any():
            continue
        allowed = tuple(
            TOKENIZE_DICT[pdg]
            for pdg in all_stable_pdgs
            if PARTICLE_CHARGES[pdg] == charge_sign
        )
        if hard:
            selected_logits = pid_logits[selected]
            masked_logits = selected_logits.new_full(selected_logits.shape, -1e4)
            masked_logits[:, allowed] = selected_logits[:, allowed]
            p4[selected] = hard_track_p4_from_pid_token(
                batch["p4"][selected, :3],
                masked_logits.argmax(dim=-1),
            )
        else:
            p4[selected] = soft_track_p4_from_pid_logits(
                batch["p4"][selected, :3],
                pid_logits[selected],
                allowed_tokens=allowed,
            )
    levels = sorted(
        {
            int(level)
            for level in batch["level_ids"][batch["node_mask"]]
            .detach()
            .cpu()
            .tolist()
            if int(level) > 0
        }
    )
    adjacency = batch["daughter_adjacency"].to(p4.dtype)
    for level in levels:
        mothers = batch["node_mask"] & (batch["level_ids"] == level)
        summed = torch.einsum("bmn,bnf->bmf", adjacency, p4)
        p4 = torch.where(mothers.unsqueeze(-1), summed, p4)
    return p4


def validate_recursive_p4_closure(
    node_p4: Mapping[int, Sequence[float]],
    daughter_ids: Mapping[int, Sequence[int]],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> dict[str, float | int]:
    """Validate exact recursive closure for a Python tree representation."""

    visiting: set[int] = set()
    visited: set[int] = set()
    maximum = 0.0

    def visit(node_id: int) -> None:
        nonlocal maximum
        if node_id in visiting:
            raise ValueError(f"cycle detected at node {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        children = list(daughter_ids.get(node_id, ()))
        for child_id in children:
            if child_id not in node_p4:
                raise ValueError(f"node {node_id} references missing daughter {child_id}")
            visit(child_id)
        if children:
            expected = [sum(float(node_p4[child][axis]) for child in children) for axis in range(4)]
            stored = [float(value) for value in node_p4[node_id]]
            for actual, target in zip(stored, expected):
                residual = abs(actual - target)
                maximum = max(maximum, residual)
                if not math.isclose(actual, target, abs_tol=atol, rel_tol=rtol):
                    raise ValueError(
                        f"mother {node_id} p4 is not its daughter sum: {stored} != {expected}"
                    )
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_p4:
        visit(node_id)
    return {"nodes": len(visited), "max_abs_p4_residual": maximum}
