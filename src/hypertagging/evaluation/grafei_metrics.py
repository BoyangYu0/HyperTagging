"""GraFEI full-reconstruction evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReconstructionEvaluationRow:
    """One event-level row matching the historical evaluation dataframe."""

    nleaves: int
    depth: int
    pdgAcc: float
    featErr: np.ndarray
    perfect: bool
    failed: bool
    sigProb: float

    def as_dict(self) -> dict[str, object]:
        return {
            "nleaves": self.nleaves,
            "depth": self.depth,
            "pdgAcc": self.pdgAcc,
            "featErr": self.featErr,
            "perfect": self.perfect,
            "failed": self.failed,
            "sigProb": self.sigProb,
        }


def accuracy_for_level(
    out_pdg: np.ndarray | torch.Tensor,
    out_feat: np.ndarray | torch.Tensor,
    truth_pdg: np.ndarray,
    truth_feat: np.ndarray,
    level: int,
    padding_size: int,
) -> tuple[int, np.ndarray]:
    """Return historical padded PDG correctness and summed feature error."""

    out_pdg_np = _to_numpy(out_pdg)
    out_feat_np = _to_numpy(out_feat)
    if level < len(truth_pdg):
        padding_size = max(padding_size, len(truth_pdg[level]))

    pred_pdg = np.zeros(padding_size)
    goal_pdg = np.zeros(padding_size)
    pred_pdg[: len(out_pdg_np)] = out_pdg_np
    if level < len(truth_pdg):
        goal_pdg[: len(truth_pdg[level])] = truth_pdg[level]
    correct_pdg = int(np.sum((pred_pdg - goal_pdg) == 0))

    pred_feat = np.zeros((padding_size, 4))
    goal_feat = np.zeros((padding_size, 4))
    pred_feat[: len(out_feat_np), :] = out_feat_np
    if level < len(truth_feat):
        goal_feat[: len(truth_feat[level]), :] = truth_feat[level]
    err_feat = np.sum(np.abs(pred_feat - goal_feat), axis=0)

    return correct_pdg, err_feat


def perfect_lca(pred_lca: np.ndarray, goal_lca: np.ndarray) -> bool:
    """Return the historical exact-LCA success flag."""

    return bool(np.sum(np.abs(goal_lca - pred_lca)) == 0)


def build_evaluation_row(
    *,
    nleaves: int,
    depth: int,
    pred_lca: np.ndarray | None,
    goal_lca: np.ndarray | None,
    accuracy: dict[str, object],
    signal_probability: torch.Tensor | float,
    failed: bool,
) -> ReconstructionEvaluationRow:
    """Build one event-level evaluation row matching ``whole_eva.py`` columns."""

    if failed or pred_lca is None or goal_lca is None:
        return ReconstructionEvaluationRow(
            nleaves=nleaves,
            depth=depth,
            pdgAcc=0.0,
            featErr=np.zeros(4),
            perfect=False,
            failed=True,
            sigProb=0.0,
        )

    sig_prob = signal_probability.item() if isinstance(signal_probability, torch.Tensor) else float(signal_probability)
    return ReconstructionEvaluationRow(
        nleaves=nleaves,
        depth=depth,
        pdgAcc=float(accuracy["pdg"]),
        featErr=np.asarray(accuracy["feat"]),
        perfect=perfect_lca(pred_lca, goal_lca),
        failed=False,
        sigProb=sig_prob,
    )


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
