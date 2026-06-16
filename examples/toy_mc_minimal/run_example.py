#!/usr/bin/env python
"""Run a minimal CPU-only Toy-MC example."""

from __future__ import annotations

import json

import torch

from hypertagging.data import DEFAULT_TOY_MC_INPUT_ROOT, prepare_toy_mc, tiny_toy_mc_batch, validate_batch
from hypertagging.losses import toy_mc_inter_loss


def main() -> None:
    batch = tiny_toy_mc_batch()
    validate_batch("toy_mc", batch)
    vectors = torch.tensor([[0.10, 0.03], [0.04, 0.11]], dtype=torch.float32)
    loss = toy_mc_inter_loss(vectors, {"channel": torch.as_tensor(batch["channel"])})
    plan = prepare_toy_mc(0, dry_run=True)
    print(
        json.dumps(
            {
                "example": "toy_mc_minimal",
                "contract": "toy_mc",
                "input_root": str(DEFAULT_TOY_MC_INPUT_ROOT),
                "pdg_shape": list(batch["pdg"].shape),
                "toy_inter_loss": float(loss.detach().cpu()),
                "preprocess_dry_run": " ".join(plan.command),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
