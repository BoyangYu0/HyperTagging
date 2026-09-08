#!/usr/bin/env python3
"""Export a trained hierarchical full-decay model for basf2 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hypertagging.deployment.export_onnx import (  # noqa: E402
    DEFAULT_BASF2_RELEASES,
    DEFAULT_LEVELS,
    ExportConfiguration,
    export_checkpoint_bundle,
    inspect_checkpoint_for_export,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trusted schema-v4 reconstruction checkpoint as fixed-shape "
            "per-level ONNX graphs and a hash-checked manifest."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="trusted PyTorch checkpoint")
    parser.add_argument(
        "output_directory",
        type=Path,
        help="new/empty destination for manifest.json and level-XX.onnx files",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_LEVELS),
        metavar="LEVEL",
        help="contiguous target levels starting at 1 (default: 1 2 3 4 5 6)",
    )
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--max-sources", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-proposals", type=int, default=12)
    parser.add_argument("--object-threshold", type=float, default=0.5)
    parser.add_argument(
        "--pointer-threshold",
        type=float,
        default=None,
        help="default: checkpoint reconstruction policy",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--type-probability-threshold", type=float, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--basf2-release",
        action="append",
        dest="basf2_releases",
        default=None,
        help=(
            "allowed BELLE2_RELEASE value; repeat for multiple releases "
            f"(default: {', '.join(DEFAULT_BASF2_RELEASES)})"
        ),
    )
    confidence = parser.add_mutually_exclusive_group()
    confidence.add_argument(
        "--use-learned-confidence",
        dest="use_learned_confidence",
        action="store_true",
        help="require and use the trained confidence head for beam scores",
    )
    confidence.add_argument(
        "--derive-confidence",
        dest="use_learned_confidence",
        action="store_false",
        help="derive confidence from object/type/pointer probabilities",
    )
    parser.set_defaults(use_learned_confidence=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only an existing, structurally valid HyperTagging bundle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the checkpoint/configuration and print the export plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configuration = ExportConfiguration(
        levels=tuple(args.levels),
        max_nodes=args.max_nodes,
        max_sources=args.max_sources,
        beam_width=args.beam_width,
        max_proposals_per_hypothesis=args.max_proposals,
        object_threshold=args.object_threshold,
        pointer_threshold=args.pointer_threshold,
        confidence_threshold=args.confidence_threshold,
        type_probability_threshold=args.type_probability_threshold,
        opset_version=args.opset,
        basf2_releases=tuple(args.basf2_releases or DEFAULT_BASF2_RELEASES),
        use_learned_confidence=args.use_learned_confidence,
    )
    if args.dry_run:
        plan = inspect_checkpoint_for_export(
            args.checkpoint, configuration=configuration
        )
        plan["output_directory"] = str(args.output_directory.expanduser().resolve())
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    manifest = export_checkpoint_bundle(
        args.checkpoint,
        args.output_directory,
        configuration=configuration,
        force=args.force,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
