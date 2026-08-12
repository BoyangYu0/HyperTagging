#!/usr/bin/env python
"""Validate training provenance and optionally enforce the scientific gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.training_selection import load_hashed_manifest  # noqa: E402


DEFAULT_STATUS = (
    ROOT
    / "configs"
    / "training_selection"
    / "production_1m_20260812"
    / "provenance_status.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--require-scientific-slurm-ready", action="store_true")
    args = parser.parse_args()
    status = load_hashed_manifest(
        args.status,
        expected_version="hypertagging-training-provenance-status-v1",
    )
    ready = bool(status.get("scientific_slurm_submission_allowed", False))
    result = {
        "status": "valid",
        "cpu_implementation_allowed": bool(status["cpu_implementation_allowed"]),
        "scientific_slurm_submission_allowed": ready,
        "blockers": status.get("scientific_submission_blockers", []),
        "manifest_hash": status["manifest_hash"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_scientific_slurm_ready and not ready:
        print(
            "scientific Slurm submission blocked by provenance status",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
