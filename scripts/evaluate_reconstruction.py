#!/usr/bin/env python
"""Dry-run GraFEI full-reconstruction evaluation on a tiny CPU event."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hypertagging.reconstruction.full_reconstruction import build_pred_lca_from_pairs


class TinyFullGenerator(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        n_particles = batch["pdg_x"].shape[1]
        logits = torch.full((1, n_particles, 14), -10.0, dtype=torch.float32)
        feature = torch.zeros((1, n_particles, 4), dtype=torch.float32)
        pdg = [4, 5, 0] if n_particles == 3 else [13] + [0] * (n_particles - 1)
        for index, pdg_id in enumerate(pdg[:n_particles]):
            logits[0, index, pdg_id] = 10.0
        return logits, feature


class TinyFullLinker(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        n_daughters = batch["pdg_x"].shape[1]
        n_mothers = batch["pdg_y"].shape[1]
        logits = torch.zeros((1, n_daughters, n_mothers), dtype=torch.float32)
        links = [0, 0, 1] if n_daughters == 3 else [0] * n_daughters
        for daughter, mother in enumerate(links[:n_daughters]):
            logits[0, daughter, mother] = float(2 + daughter)
        return logits


def tiny_pairs() -> np.ndarray:
    pairs = np.zeros((2, 2, 3, 5), dtype=np.float32)
    pairs[-1, 0, :, 0] = [1, 2, 3]
    pairs[-1, 0, :, 1:] = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    pairs[-1, 1, :, 0] = [4, 5, 0]
    pairs[-1, 1, :, 1:] = [
        [1.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    pairs[0, 1, :, 0] = [13, 0, 0]
    pairs[0, 1, :, 1:] = [
        [1.0, 1.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Run a tiny CPU-only fixture.")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="Phase 10 smoke test is CPU-only.")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("Phase 10 only supports --dry-run. Full-data evaluation remains HPC-only.")

    result = build_pred_lca_from_pairs(tiny_pairs(), TinyFullGenerator(), TinyFullLinker())
    print(
        json.dumps(
            {
                "lca": result.lca.tolist(),
                "pdgAcc": result.accuracy["pdg"],
                "featErr": np.asarray(result.accuracy["feat"]).tolist(),
                "sigProb": float(result.signal_probability.item()),
                "nSteps": len(result.steps),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
