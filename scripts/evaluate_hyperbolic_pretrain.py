#!/usr/bin/env python3
"""Read-only evaluation of a pretraining checkpoint on the fixed 2,000-event validation cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.evaluation.pretraining_validation import (  # noqa: E402
    evaluate_pretraining_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-git-sha", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, default=3282)
    parser.add_argument(
        "--allow-comparison-checkpoint",
        action="store_true",
        help="Permit a hash-bound non-best checkpoint while retaining read-only validation guards.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_pretraining_checkpoint(
        checkpoint=args.checkpoint,
        data=args.data,
        dataset_index=args.dataset_index,
        output=args.output,
        device=args.device,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_source_git_sha=args.expected_source_git_sha,
        expected_checkpoint_step=args.expected_checkpoint_step,
        require_selected_minimum=not args.allow_comparison_checkpoint,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint_step": result["checkpoint"]["step"],
                "selected_minimum_required": result["checkpoint"][
                    "selected_minimum_required"
                ],
                "validation_events": result["metrics"]["validation_events"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
