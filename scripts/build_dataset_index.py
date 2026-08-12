#!/usr/bin/env python
"""Build the one-pass HyperTagging dataset index used by real trainers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.dataset_index import (  # noqa: E402
    build_dataset_index,
    build_dataset_index_from_sidecars,
)
from hypertagging.training.data_module import resolve_data_paths  # noqa: E402
from hypertagging.data.training_selection import load_training_selection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument(
        "--selection-manifest",
        help="Immutable source-role manifest; mutually exclusive with --data.",
    )
    parser.add_argument(
        "--include-splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
        help="Manifest roles whose Parquet payloads may be opened (default excludes sealed test).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-policy", default="complete_only")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--scientific-mode",
        action="store_true",
        help="Fail closed unless immutable source roles replace raw max_events selection.",
    )
    parser.add_argument(
        "--from-sidecars",
        action="store_true",
        help="Merge published shard statistics without reading event payloads.",
    )
    args = parser.parse_args()
    if bool(args.data) == bool(args.selection_manifest):
        parser.error("provide exactly one of --data or --selection-manifest")
    if args.scientific_mode and not args.selection_manifest:
        parser.error("--scientific-mode requires --selection-manifest")
    if args.scientific_mode and args.from_sidecars:
        parser.error("--scientific-mode requires full event records, not sidecars")
    if args.scientific_mode and tuple(args.include_splits) != ("train", "validation"):
        parser.error("--scientific-mode permits exactly train and validation splits")
    if args.selection_manifest and args.max_events is not None:
        parser.error("--selection-manifest cannot be combined with --max-events")
    builder = (
        build_dataset_index_from_sidecars if args.from_sidecars else build_dataset_index
    )
    kwargs = {"target_policy": args.target_policy}
    if args.selection_manifest:
        selection = load_training_selection(
            args.selection_manifest, include_splits=args.include_splits
        )
        paths = list(selection.paths)
        kwargs.update(
            {
                "source_split_overrides": selection.source_split_overrides,
                "selection_manifest_hash": selection.manifest_hash,
                "selection_included_splits": selection.included_splits,
                "source_expectations": selection.source_expectations,
            }
        )
    else:
        paths = resolve_data_paths(args.data)
    if not args.from_sidecars:
        kwargs["max_events"] = args.max_events
        kwargs["require_event_identity_validation"] = args.scientific_mode
    path = builder(paths, args.output, **kwargs)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
