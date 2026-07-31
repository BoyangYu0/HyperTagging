#!/usr/bin/env python
"""Build the one-pass HyperTagging dataset index used by real trainers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.dataset_index import (
    build_dataset_index,
    build_dataset_index_from_sidecars,
)
from hypertagging.training.data_module import resolve_data_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-policy", default="complete_only")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--from-sidecars",
        action="store_true",
        help="Merge published shard statistics without reading event payloads.",
    )
    args = parser.parse_args()
    builder = (
        build_dataset_index_from_sidecars
        if args.from_sidecars
        else build_dataset_index
    )
    kwargs = {"target_policy": args.target_policy}
    if not args.from_sidecars:
        kwargs["max_events"] = args.max_events
    path = builder(resolve_data_paths(args.data), args.output, **kwargs)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
