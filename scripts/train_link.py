"""Dry-run the link-prediction training stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from hypertagging.training.train_link import run_link_prediction_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["ground_truth", "reconstructed_mother"],
        default="ground_truth",
    )
    parser.add_argument(
        "--model-variant",
        choices=["standard", "embedding"],
        default="standard",
    )
    args = parser.parse_args()

    summary = run_link_prediction_dry_run(
        mode=args.mode,
        model_variant=args.model_variant,
        device=args.device,
        backward=not args.no_backward,
    )
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
