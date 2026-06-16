#!/usr/bin/env python
"""Run a minimal CPU-only GraFEI example."""

from __future__ import annotations

from dataclasses import asdict
import json

from hypertagging.data import DEFAULT_GRAFEI_INPUT_ROOT, prepare_grafei, tiny_grafei_combined_batch, validate_batch
from hypertagging.training import run_reconstruction_dry_run


def main() -> None:
    batch = tiny_grafei_combined_batch()
    validate_batch("grafei_combined", batch)
    summary = run_reconstruction_dry_run(device="cpu", backward=False)
    plan = prepare_grafei(0, dry_run=True)
    payload = asdict(summary)
    payload.update(
        {
            "example": "grafei_minimal",
            "contract": "grafei_combined",
            "input_root": str(DEFAULT_GRAFEI_INPUT_ROOT),
            "batch_particles": int(batch["pdg_x"].shape[1]),
            "preprocess_dry_run": " ".join(plan.command),
        }
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
