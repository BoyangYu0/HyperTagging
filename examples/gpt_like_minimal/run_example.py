#!/usr/bin/env python
"""Run a minimal CPU-only GPT-like/autoregressive example."""

from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np

from hypertagging.data import (
    DEFAULT_GRAFEI_INPUT_ROOT,
    collate_gpt_reconstruction,
    prepare_gpt_like,
    validate_gpt_reconstruction_batch,
)
from hypertagging.training import run_multi_gpt_dry_run


def main() -> None:
    collated = collate_gpt_reconstruction(
        [
            {
                "emb": np.arange(16, dtype=np.float32).reshape(4, 4) / 100,
                "links": np.array([0, 1], dtype=np.int_),
                "mass": np.array([1, 2, 3, 4], dtype=np.int_),
                "shape": np.array([4, 2], dtype=np.int_),
            }
        ]
    )
    validate_gpt_reconstruction_batch(collated)
    summary = run_multi_gpt_dry_run(device="cpu", backward=False)
    plan = prepare_gpt_like(0, dry_run=True)
    payload = asdict(summary)
    payload.update(
        {
            "example": "gpt_like_minimal",
            "contract": "gpt_reconstruction_flattened",
            "input_root": str(DEFAULT_GRAFEI_INPUT_ROOT),
            "collated_emb_shape": list(collated["emb"].shape),
            "preprocess_dry_run": " ".join(plan.command),
        }
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
