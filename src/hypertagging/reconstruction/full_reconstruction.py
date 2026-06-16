"""Full GraFEI reconstruction evaluation loop.

This preserves the event-level behavior from ``graFEI/whole_eva.py``:
argmax PDG recovery, empty-mother link remapping, daughter-feature aggregation,
iterative LCA construction, and root-token stopping on PDG ``13``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from hypertagging.evaluation.grafei_metrics import accuracy_for_level, build_evaluation_row


class ReconstructionModel(Protocol):
    def __call__(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class LinkModel(Protocol):
    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class PredictionStep:
    """One historical full-reconstruction prediction step."""

    pdg: torch.Tensor
    feature: torch.Tensor
    link: torch.Tensor
    signal_probability: torch.Tensor


@dataclass(frozen=True)
class FullReconstructionResult:
    """Event-level full-reconstruction output."""

    lca: np.ndarray
    signal_probability: torch.Tensor
    accuracy: dict[str, float | np.ndarray]
    steps: tuple[PredictionStep, ...]
    failed: bool = False
    failure_reason: str | None = None


def recover(pred: torch.Tensor, *, return_value: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Recover argmax classes, optionally returning historical max values."""

    values, parsed = torch.max(pred, dim=-1)
    if return_value:
        return parsed, values
    return parsed


def remap_links_around_empty(out_link: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the historical empty-mother index remapping."""

    remapped = out_link.clone()
    empty_idx = (~mask).nonzero()
    if empty_idx.numel() == 0 or remapped.numel() == 0:
        return remapped
    remapped -= (~(empty_idx.repeat(1, len(remapped)) > remapped[:, None].repeat(1, len(empty_idx)).T)).sum(0)
    return remapped


def aggregate_features_by_link(in_feat: torch.Tensor, out_link: torch.Tensor) -> torch.Tensor:
    """Aggregate daughter features into mother slots using recovered links."""

    rec_feats = torch.zeros_like(in_feat)
    for daughter, mother in enumerate(out_link):
        rec_feats[mother] += in_feat[daughter]
    return rec_feats


def prediction_step(
    in_pdg: torch.Tensor,
    in_feat: torch.Tensor,
    generator: ReconstructionModel,
    linker: LinkModel,
) -> PredictionStep:
    """Run one historical full-reconstruction prediction step on CPU tensors."""

    pdg_x = in_pdg.unsqueeze(0)
    feature_x = in_feat.unsqueeze(0)
    padding_mask = torch.ones_like(pdg_x).bool()
    out_pdg_logits, out_feat = generator(
        {
            "pdg_x": pdg_x,
            "feature_x": feature_x,
            "padding_mask": padding_mask,
        }
    )
    out_pdg = recover(out_pdg_logits)
    assert isinstance(out_pdg, torch.Tensor)
    mask = out_pdg > 0
    out_feat = out_feat * mask[..., None]

    reconstructed_link = {
        "pdg_x": pdg_x,
        "pdg_y": out_pdg,
        "feature_x": feature_x,
        "feature_y": out_feat,
        "padding_mask": padding_mask,
    }
    recovered = recover(linker(reconstructed_link), return_value=True)
    assert isinstance(recovered, tuple)
    out_link, likelihood = recovered

    out_link = out_link.squeeze()
    out_pdg = out_pdg.squeeze()
    out_feat = out_feat.squeeze()
    mask = mask.squeeze()
    if out_link.ndim == 0:
        out_link = out_link.unsqueeze(0)
    if out_pdg.ndim == 0:
        out_pdg = out_pdg.unsqueeze(0)
        out_feat = out_feat.unsqueeze(0)
        mask = mask.unsqueeze(0)

    if mask.sum() > 1:
        out_link = remap_links_around_empty(out_link, mask)
        signal_probability = torch.prod(likelihood.squeeze()[mask]) ** (1 / mask.sum())
        out_feat = aggregate_features_by_link(in_feat, out_link)
        return PredictionStep(
            pdg=out_pdg[mask],
            feature=out_feat[mask],
            link=out_link,
            signal_probability=signal_probability,
        )

    return PredictionStep(
        pdg=out_pdg[mask],
        feature=out_feat[mask],
        link=torch.zeros_like(in_pdg),
        signal_probability=in_feat.new_tensor(1.0),
    )


def build_pred_lca_from_pairs(
    pairs: np.ndarray,
    generator: ReconstructionModel,
    linker: LinkModel,
    *,
    max_steps: int = 20,
) -> FullReconstructionResult:
    """Build the predicted LCA matrix from a GraFEI ``pairs`` event array."""

    pairs_np = np.asarray(pairs)
    level = 0
    truth_pdg = pairs_np[:, 1, :, 0][::-1]
    truth_feat = pairs_np[:, 1, :, 1:][::-1]
    in_pdg = torch.as_tensor(pairs_np[-1, 0, :, 0], dtype=torch.int64)
    in_feat = torch.as_tensor(pairs_np[-1, 0, :, 1:], dtype=torch.float32)

    first_step = prediction_step(in_pdg, in_feat, generator, linker)
    correct_pdg, err_feat = accuracy_for_level(
        first_step.pdg,
        first_step.feature,
        truth_pdg,
        truth_feat,
        level,
        padding_size=len(in_pdg),
    )
    accuracy_total = {
        "pdg": correct_pdg,
        "feat": err_feat,
        "total": len(in_pdg),
    }
    sig_probs = [first_step.signal_probability]
    steps = [first_step]
    idx_list = np.stack((np.arange(len(in_pdg)), first_step.link.detach().cpu().numpy())).T
    out_pdg = first_step.pdg
    out_feat = first_step.feature
    out_link = first_step.link

    while 13 not in out_pdg and len(out_pdg) > 1:
        padding_size = len(out_pdg)
        level += 1
        step = prediction_step(out_pdg, out_feat, generator, linker)
        steps.append(step)
        sig_probs.append(step.signal_probability)
        correct_pdg, err_feat = accuracy_for_level(
            step.pdg,
            step.feature,
            truth_pdg,
            truth_feat,
            level,
            padding_size=padding_size,
        )
        accuracy_total["pdg"] += correct_pdg
        accuracy_total["feat"] += err_feat
        accuracy_total["total"] += padding_size

        out_pdg = step.pdg
        out_feat = step.feature
        out_link = step.link
        if not out_link.shape:
            break
        idx_list[..., -1] = idx_list[..., -1].clip(max=len(out_link) - 1)
        idx_list = np.column_stack((idx_list, out_link.detach().cpu().numpy()[idx_list[..., -1]][..., None]))
        if len(sig_probs) > max_steps:
            raise RuntimeError("Unable to find the tree")

    lca = np.argmin(np.abs((idx_list[None, ...] - idx_list[:, None, :])), axis=-1)
    signal_probability = torch.prod(torch.tensor(sig_probs)) ** len(sig_probs)
    accuracy = {
        "pdg": float(accuracy_total["pdg"] / accuracy_total["total"]),
        "feat": accuracy_total["feat"] / accuracy_total["total"],
    }
    return FullReconstructionResult(
        lca=lca,
        signal_probability=signal_probability,
        accuracy=accuracy,
        steps=tuple(steps),
    )


def build_pred_lca(
    event: object,
    generator: ReconstructionModel,
    linker: LinkModel,
    *,
    max_steps: int = 20,
) -> FullReconstructionResult:
    """Build predicted LCA from an event object exposing historical ``pairs``."""

    return build_pred_lca_from_pairs(event.pairs, generator, linker, max_steps=max_steps)


def evaluate_event(
    event: object,
    generator: ReconstructionModel,
    linker: LinkModel,
    *,
    goal_lca: np.ndarray | None = None,
    max_steps: int = 20,
) -> dict[str, object]:
    """Evaluate one event and return columns compatible with ``whole_eva.py``."""

    pairs = np.asarray(event.pairs)
    nleaves = len(pairs[-1, 0, :, 0])
    depth = len(pairs)
    if goal_lca is None and hasattr(event, "lcas"):
        goal_lca = np.asarray(event.lcas[0])

    try:
        result = build_pred_lca_from_pairs(pairs, generator, linker, max_steps=max_steps)
        row = build_evaluation_row(
            nleaves=nleaves,
            depth=depth,
            pred_lca=result.lca,
            goal_lca=goal_lca,
            accuracy=result.accuracy,
            signal_probability=result.signal_probability,
            failed=False,
        )
    except Exception:
        row = build_evaluation_row(
            nleaves=nleaves,
            depth=depth,
            pred_lca=None,
            goal_lca=goal_lca,
            accuracy={"pdg": 0.0, "feat": np.zeros(4)},
            signal_probability=0.0,
            failed=True,
        )
    return row.as_dict()
