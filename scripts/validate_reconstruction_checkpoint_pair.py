#!/usr/bin/env python
"""Validate pretraining/reconstruction checkpoint lineage on CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# This script is deliberately CPU-only and hides accelerators before importing
# torch through the project package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-checkpoint", required=True)
    parser.add_argument("--reconstruction-checkpoint", required=True)
    parser.add_argument(
        "--allow-finetuned-encoder",
        action="store_true",
        help="Require compatible encoder keys/contracts but allow changed tensors.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON receipt path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_checkpoint_pair(
        args.pretraining_checkpoint,
        args.reconstruction_checkpoint,
        require_exact_frozen_encoder=not args.allow_finetuned_encoder,
    ).as_dict()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
