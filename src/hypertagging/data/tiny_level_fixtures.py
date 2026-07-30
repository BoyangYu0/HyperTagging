"""Tiny levelized trees for CPU tests and dry-runs."""

from __future__ import annotations

import torch

from hypertagging.data.level_batch import LevelEvent
from hypertagging.preprocessing.pid_filter import tokenize_pdg, validate_pid_tokens
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID


def tiny_level_events() -> list[LevelEvent]:
    """Return synthetic variable-size events with copied/shared and skipped-like nodes."""

    return [_gamma_gamma_pi0_event(), _b_to_dstar_event(), _copied_shared_event()]


def _event(
    event_id: int,
    *,
    pdg: list[int],
    charge: list[float],
    p4: list[tuple[float, float, float, float]],
    levels: list[int],
    parents: list[int],
    daughters: list[list[int]],
    copied_from: list[int] | None = None,
) -> LevelEvent:
    n_nodes = len(pdg)
    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)
    for parent_id, child_ids in enumerate(daughters):
        for child_id in child_ids:
            adjacency[parent_id, child_id] = True
    p4_tensor = torch.tensor(p4, dtype=torch.float32)
    mass2 = p4_tensor[:, 3] ** 2 - (p4_tensor[:, :3] ** 2).sum(dim=-1)
    mass = torch.sqrt(torch.clamp(mass2, min=0.0))
    raw_pdg = torch.tensor(pdg, dtype=torch.long)
    pid = torch.tensor([tokenize_pdg(value) for value in pdg], dtype=torch.long)
    validate_pid_tokens(pid, name="tiny fixture PID labels")
    charge_tensor = torch.tensor(charge, dtype=torch.float32)
    node_features = torch.stack(
        [
            pid.float() / max(1, 40),
            charge_tensor,
            p4_tensor[:, 0],
            p4_tensor[:, 1],
            p4_tensor[:, 2],
            p4_tensor[:, 3],
            mass,
            torch.tensor(levels, dtype=torch.float32),
        ],
        dim=-1,
    )
    copied_from_tensor = torch.tensor(copied_from or [-1] * n_nodes, dtype=torch.long)
    return LevelEvent(
        event_id=event_id,
        node_features=node_features,
        p4=p4_tensor,
        charge=charge_tensor,
        pid_labels=pid,
        level_ids=torch.tensor(levels, dtype=torch.long),
        parent_ids=torch.tensor(parents, dtype=torch.long),
        daughter_adjacency=adjacency,
        active=torch.ones(n_nodes, dtype=torch.bool),
        copied=copied_from_tensor >= 0,
        copied_from=copied_from_tensor,
        raw_pdg=raw_pdg,
        node_kind_ids=torch.tensor(
            [
                NODE_KIND_TO_ID["composite"]
                if daughters[index]
                else (
                    NODE_KIND_TO_ID["ecl_cluster"]
                    if abs(pdg[index]) == 22
                    else NODE_KIND_TO_ID["track"]
                )
                for index in range(n_nodes)
            ],
            dtype=torch.long,
        ),
    )


def _gamma_gamma_pi0_event() -> LevelEvent:
    gamma1 = (0.10, 0.00, 0.00, 0.10)
    gamma2 = (-0.02, 0.07, 0.00, 0.08)
    pi0 = tuple(gamma1[i] + gamma2[i] for i in range(4))
    return _event(
        100,
        pdg=[22, 22, 111],
        charge=[0.0, 0.0, 0.0],
        p4=[gamma1, gamma2, pi0],
        levels=[0, 0, 1],
        parents=[2, 2, -1],
        daughters=[[], [], [0, 1]],
    )


def _b_to_dstar_event() -> LevelEvent:
    k = (0.30, 0.00, 0.05, 0.58)
    pi = (-0.10, 0.20, 0.02, 0.27)
    slow_pi = (0.05, -0.03, 0.01, 0.15)
    bachelor = (-0.08, 0.02, 0.03, 0.18)
    d0 = tuple(k[i] + pi[i] for i in range(4))
    dstar = tuple(d0[i] + slow_pi[i] for i in range(4))
    b = tuple(dstar[i] + bachelor[i] for i in range(4))
    return _event(
        101,
        pdg=[321, -211, 211, -211, 421, 413, 521],
        charge=[1, -1, 1, -1, 0, 1, 0],
        p4=[k, pi, slow_pi, bachelor, d0, dstar, b],
        levels=[0, 0, 0, 0, 1, 2, 3],
        parents=[4, 4, 5, 6, 5, 6, -1],
        daughters=[[], [], [], [], [0, 1], [4, 2], [5, 3]],
    )


def _copied_shared_event() -> LevelEvent:
    g = (0.05, 0.01, 0.00, 0.06)
    g_copy = g
    g2 = (-0.01, 0.03, 0.00, 0.04)
    g3 = (0.02, -0.02, 0.01, 0.04)
    pi0_a = tuple(g[i] + g2[i] for i in range(4))
    pi0_b = tuple(g_copy[i] + g3[i] for i in range(4))
    return _event(
        102,
        pdg=[22, 22, 22, 22, 111, 111],
        charge=[0, 0, 0, 0, 0, 0],
        p4=[g, g_copy, g2, g3, pi0_a, pi0_b],
        levels=[0, 0, 0, 0, 1, 1],
        parents=[4, 5, 4, 5, -1, -1],
        daughters=[[], [], [], [], [0, 2], [1, 3]],
        copied_from=[-1, 0, -1, -1, -1, -1],
    )
