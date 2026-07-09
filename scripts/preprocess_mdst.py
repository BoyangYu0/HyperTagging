#!/usr/bin/env python
"""Preprocess generic mDST files into HyperTagging direct-tree parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Input generic mDST ROOT file(s).")
    parser.add_argument("--output", required=True, help="Output parquet file.")
    parser.add_argument("--max-events", type=int, default=None, help="Maximum number of events.")
    parser.add_argument("--event-index", type=int, default=None, help="Alias for --debug-event.")
    parser.add_argument("--debug-event", type=int, default=None, help="Only process one EventMetaData event number.")
    parser.add_argument("--config", default=None, help="Optional config file path recorded for reproducibility.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose/debug preprocessing behavior.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned preprocessing inputs and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    parser.add_argument(
        "--particle-array",
        action="append",
        default=None,
        help="DataStore Particle array to read directly. Can be repeated. Default: Particles.",
    )
    parser.add_argument("--no-tracks", action="store_true", help="Do not read Tracks StoreArray.")
    parser.add_argument("--no-ecl-clusters", action="store_true", help="Do not read ECLClusters StoreArray.")
    parser.add_argument(
        "--allow-mc-leaf-kinematics-for-debug",
        action="store_true",
        help="Use MC leaf p4 only for debug when no reco objects are present. Never use this for training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = parse_args(argv)
    output_path = Path(args.output)
    if args.dry_run:
        print(
            {
                "input": args.input,
                "output": str(output_path),
                "max_events": args.max_events,
                "event_index": args.event_index,
                "config": args.config,
            }
        )
        return 0
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")
    from hypertagging.preprocessing.basf2_mdst import Basf2PreprocessConfig, run_basf2_preprocessing

    config = Basf2PreprocessConfig(
        input_files=tuple(args.input),
        output=output_path,
        max_events=args.max_events,
        debug_event=args.debug_event if args.debug_event is not None else args.event_index,
        particle_arrays=tuple(args.particle_array or ["Particles"]),
        include_tracks=not args.no_tracks,
        include_ecl_clusters=not args.no_ecl_clusters,
        allow_mc_leaf_kinematics_for_debug=args.allow_mc_leaf_kinematics_for_debug,
    )
    output = run_basf2_preprocessing(config)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
