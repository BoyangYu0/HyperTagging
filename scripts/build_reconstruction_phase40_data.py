#!/usr/bin/env python3
"""Build the exact doubled-data reconstruction selection and scientific index."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.dataset_index import build_dataset_index  # noqa: E402
from hypertagging.data.training_selection import (  # noqa: E402
    INVENTORY_VERSION,
    ROLE_MANIFEST_VERSION,
    SELECTION_MANIFEST_VERSION,
    build_training_selection,
    load_hashed_manifest,
    load_training_selection,
    write_hashed_manifest,
)


SOURCE = ROOT / "configs/training_selection/production_1m_20260812"
TRAINING_QUOTAS = {
    "ccbar": 2,
    "charged": 2,
    "ddbar": 2,
    "mixed": 2,
    "ssbar": 2,
    "taupair": 2,
    "uubar": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()
    selection_output = args.selection_output.resolve()
    index_output = args.index_output.resolve()
    if selection_output.exists() or index_output.exists():
        raise RuntimeError("refusing to overwrite phase40 data products")

    inventory = load_hashed_manifest(
        SOURCE / "inventory.json", expected_version=INVENTORY_VERSION
    )
    roles = load_hashed_manifest(
        SOURCE / "roles.json", expected_version=ROLE_MANIFEST_VERSION
    )
    selection = build_training_selection(
        inventory,
        roles,
        selection_name="train_070k_phase40",
        training_quotas=TRAINING_QUOTAS,
        include_test=True,
    )
    write_hashed_manifest(selection, selection_output)
    checked = load_hashed_manifest(
        selection_output, expected_version=SELECTION_MANIFEST_VERSION
    )
    loaded = load_training_selection(
        selection_output, include_splits=("train", "validation")
    )
    if checked["split_counts"] != {
        "train": 70_000,
        "validation": 50_000,
        "test": 50_000,
    }:
        raise RuntimeError("phase40 selection is not the exact 70k/50k/50k design")
    train_categories = Counter(
        entry["category"] for entry in checked["entries"] if entry["split"] == "train"
    )
    if train_categories != Counter(TRAINING_QUOTAS):
        raise RuntimeError("phase40 doubled selection lost balanced category coverage")

    build_dataset_index(
        list(loaded.paths),
        index_output,
        target_policy="complete_only",
        require_event_identity_validation=True,
        source_split_overrides=loaded.source_split_overrides,
        selection_manifest_hash=loaded.manifest_hash,
        selection_included_splits=loaded.included_splits,
        source_expectations=loaded.source_expectations,
    )
    index = json.loads(index_output.read_text(encoding="utf-8"))
    if (
        index.get("split_counts")
        not in (
            {"train": 70_000, "validation": 50_000},
            {"train": 70_000, "validation": 50_000, "test": 0},
        )
        or index.get("selection_contract", {}).get("included_splits")
        != ["train", "validation"]
        or index.get("event_identity_validation", {}).get("sealed_test_opened")
        is not False
    ):
        raise RuntimeError("phase40 index did not prove train/validation-only access")
    print(
        json.dumps(
            {
                "selection": str(selection_output),
                "selection_manifest_hash": checked["manifest_hash"],
                "index": str(index_output),
                "index_hash": index["index_hash"],
                "train_events": 70_000,
                "validation_events": 50_000,
                "sealed_test_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
