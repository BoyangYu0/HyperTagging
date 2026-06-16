"""CPU dry-run fixtures for migrated training loops."""

from __future__ import annotations

import torch


def embedding_batch(device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Tiny GraFEI-style embedding batch."""

    return {
        "pdg": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long, device=device),
        "feature": torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4) / 50,
        "padding_mask": torch.tensor([[True, True, False], [True, True, True]], device=device),
        "mass": torch.tensor([1, 5], dtype=torch.long, device=device),
        "pattern": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32, device=device),
        "evtNums": torch.tensor([1001, 1002], dtype=torch.long, device=device),
    }


def link_batch(device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Tiny GraFEI-style link-prediction batch."""

    base = embedding_batch(device)
    return {
        "pdg_x": base["pdg"],
        "pdg_y": torch.tensor([[2, 1, 0], [4, 3, 5]], dtype=torch.long, device=device),
        "feature_x": base["feature"],
        "feature_y": base["feature"] + 0.1,
        "padding_mask": base["padding_mask"],
        "links": torch.tensor([[0, 0, -1], [1, 1, 2]], dtype=torch.long, device=device),
    }


def reconstruction_batch(device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Tiny GraFEI-style reconstruction batch."""

    batch = link_batch(device)
    batch["pdg_y"] = torch.tensor([[2, 1, 0], [4, 3, 5]], dtype=torch.long, device=device)
    return batch


def gpt_batch(device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Tiny GPT-like reconstruction batch."""

    src_mask = torch.zeros((2, 4, 4), dtype=torch.float32, device=device)
    return {
        "emb": torch.arange(32, dtype=torch.float32, device=device).reshape(2, 4, 4) / 100,
        "target": torch.arange(32, 64, dtype=torch.float32, device=device).reshape(2, 4, 4) / 100,
        "src_mask": src_mask,
        "lvl_code": torch.tensor([[1.0, 1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.0]], device=device),
        "links": torch.tensor([[0, 1, -1, -1], [1, -1, 2, -1]], dtype=torch.long, device=device),
        "mass": torch.tensor([[1, 2, 0, 0], [5, 0, 10, 0]], dtype=torch.float32, device=device),
    }
