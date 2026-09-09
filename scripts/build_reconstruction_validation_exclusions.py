#!/usr/bin/env python3
"""Build a canonical fixed-validation exclusion manifest from prior results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hypertagging.training.fixed_validation import (  # noqa: E402
    excluded_event_uids_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(ROOT.resolve())
    return resolved


def _checkpoint_validation_uids(path: Path) -> tuple[str, ...]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    selection = payload.get("validation_selection")
    if not isinstance(selection, dict):
        raise RuntimeError(f"checkpoint has no validation selection: {path}")
    values = selection.get("event_uids")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"checkpoint has no validation event UIDs: {path}")
    return tuple(str(value) for value in values)


def _confirmation_uids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("selection", {}).get("event_uids")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"confirmation result has no event UIDs: {path}")
    if payload.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError(f"confirmation result does not deny sealed-test access: {path}")
    if payload.get("training_performed") is not False:
        raise RuntimeError(f"confirmation result unexpectedly performed training: {path}")
    return tuple(str(value) for value in values)


def _source(path: Path, kind: str, values: tuple[str, ...]) -> dict[str, Any]:
    relative = str(path.relative_to(ROOT))
    return {
        "kind": kind,
        "path": relative,
        "sha256": sha256(path),
        "event_uid_count": len(values),
        **excluded_event_uids_contract(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--confirmation-result", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.confirmation_result) != 2:
        raise RuntimeError("exactly two confirmation results are required")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite exclusion manifest: {output}")

    checkpoint = _repo_path(args.checkpoint)
    confirmations = tuple(_repo_path(path) for path in args.confirmation_result)
    source_values = (
        _checkpoint_validation_uids(checkpoint),
        *(_confirmation_uids(path) for path in confirmations),
    )
    source_paths = (checkpoint, *confirmations)
    source_kinds = (
        "checkpoint_validation_event_uids",
        "confirmation_result_event_uids",
        "confirmation_result_event_uids",
    )
    source_sets = tuple(set(values) for values in source_values)
    overlaps: list[dict[str, Any]] = []
    for left in range(len(source_sets)):
        for right in range(left + 1, len(source_sets)):
            overlap = source_sets[left] & source_sets[right]
            overlaps.append(
                {
                    "left_source_index": left,
                    "right_source_index": right,
                    "event_uid_count": len(overlap),
                }
            )
    event_uids = tuple(sorted(set().union(*source_sets)))
    identity = excluded_event_uids_contract(event_uids)
    payload = {
        "manifest_version": "hypertagging-validation-exclusions-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "validation",
        "reason": (
            "exclude prior reconstruction rollout and paired-confirmation "
            "cohorts from phase35 checkpoint selection"
        ),
        "sealed_test_role_access": "forbidden",
        "sources": [
            _source(path, kind, values)
            for path, kind, values in zip(
                source_paths, source_kinds, source_values, strict=True
            )
        ],
        "pairwise_source_overlaps": overlaps,
        **identity,
        "event_uids": list(event_uids),
    }
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                **identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
