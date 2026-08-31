#!/usr/bin/env python3
"""Render one inert, CAS-named metadata repromotion package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hypertagging.data.selection_repromotion import (
    build_repromotion_package,
    collect_git_provenance,
    load_repromotion_contract,
    publish_repromotion_package,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / (
    "configs/training_selection/repromotion/train_035k_16key_repromotion.contract.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--implementation-tag", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        parser.error("--output-root must be an absolute fresh namespace")
    contract = load_repromotion_contract(args.contract)
    provenance = collect_git_provenance(
        ROOT,
        implementation_tag=args.implementation_tag,
        contract=contract,
    )
    package = build_repromotion_package(
        contract,
        ROOT,
        implementation_provenance=provenance,
        output_namespace=str(args.output_root),
    )
    paths = publish_repromotion_package(package, args.output_root)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
