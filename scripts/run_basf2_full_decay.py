#!/usr/bin/env python3
"""Run the additive ONNX full-decay module on existing ParticleLists.

Example (after setting up the compatible CVMFS release)::

    basf2 scripts/run_basf2_full_decay.py -- \
      --manifest bundle/manifest.json --input input.udst.root \
      --input-list pi+:HyperTaggingFSP --input-list gamma:HyperTaggingFSP \
      --events 1

No output file is written by default.  ``--output-udst`` must name a new file;
this runner deliberately refuses to overwrite any existing file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import basf2 as b2  # type: ignore[import-not-found]  # noqa: E402
import modularAnalysis as ma  # type: ignore[import-not-found]  # noqa: E402

from hypertagging.basf2_integration.module import (  # noqa: E402
    HyperTaggingFullDecayModule,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ONNX-guided HyperTagging without changing offline lists."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Input mDST/uDST; repeat for multiple files.",
    )
    parser.add_argument(
        "--input-list",
        action="append",
        required=True,
        help="Existing final-state ParticleList; repeat for multiple lists.",
    )
    parser.add_argument(
        "--output-list",
        default="Upsilon(4S):HyperTagging",
        help="Unique additive output list name.",
    )
    parser.add_argument(
        "--output-udst",
        type=Path,
        help="Optionally persist the new list to a new uDST file.",
    )
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.events < 1:
        raise ValueError("--events must be positive")
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"ONNX bundle manifest does not exist: {manifest}")
    input_files = tuple(path.expanduser().resolve() for path in args.input)
    missing = [path for path in input_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input file(s) do not exist: {missing}")

    output = None
    if args.output_udst is not None:
        output = args.output_udst.expanduser().resolve()
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output file: {output}"
            )
        if output in input_files:
            raise ValueError("output uDST must be distinct from every input file")
        output.parent.mkdir(parents=True, exist_ok=True)

    path = b2.create_path()
    ma.inputMdstList(
        filelist=[str(filename) for filename in input_files],
        environmentType="default",
        path=path,
    )
    path.add_module(
        HyperTaggingFullDecayModule(
            manifest_path=manifest,
            input_particle_lists=tuple(args.input_list),
            output_particle_list=args.output_list,
            write_out=output is not None,
            intra_op_threads=args.intra_op_threads,
        )
    )
    if output is not None:
        ma.outputUdst(
            filename=str(output),
            particleLists=[args.output_list],
            path=path,
        )
    b2.process(path=path, max_event=args.events)
    print(b2.statistics)


if __name__ == "__main__":
    main()
