#!/usr/bin/env python
"""Validate the fail-closed 865k/H100 pretraining configuration without training."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.dataset_index import load_dataset_index  # noqa: E402
from hypertagging.data.training_selection import load_hashed_manifest  # noqa: E402
from hypertagging.training.learning_rate import (  # noqa: E402
    learning_rate_schedule_contract,
)


EXPECTED = {
    "batch_size": 16,
    "max_steps": 108128,
    "lr_schedule_total_steps": 108128,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 0.5,
    "warmup_fraction": 0.1,
    "max_warmup_steps": 10000,
    "min_lr_ratio": 0.1,
    "checkpoint_every": 13516,
    "validate_every": 13516,
    "validation_events": 5000,
    "validation_batches": 313,
    "curriculum_phase_steps": [27032, 27032, 27032, 27032],
    "amp_dtype": "bfloat16",
    "channel_memory_size": 4096,
    "channel_zero_positive_action": "fail",
    "pilot_objective_violation_action": "fail",
    "scientific_mode": True,
    "num_workers": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/slurm/pretrain_1m_h100_20260821.yaml",
    )
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("full-scale config must be a YAML mapping")
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise ValueError(
                f"full-scale config {key}={config.get(key)!r} does not equal {expected!r}"
            )
    if config.get("max_steps", 0) * config["batch_size"] != 1_730_048:
        raise ValueError("presentation count is not 1,730,048")
    if sum(config["curriculum_phase_steps"]) != config["max_steps"]:
        raise ValueError("curriculum phase steps do not cover max_steps exactly")
    lr_contract = learning_rate_schedule_contract(
        total_steps=config["lr_schedule_total_steps"],
        warmup_fraction=config["warmup_fraction"],
        max_warmup_steps=config["max_warmup_steps"],
        min_lr_ratio=config["min_lr_ratio"],
        base_lrs=[config["learning_rate"]],
    )
    if lr_contract["warmup_steps"] != 10000:
        raise ValueError("full-scale warmup cap did not resolve to 10000 steps")
    validation_events = int(config["validation_events"])
    batch_size = int(config["batch_size"])
    expected_batches = ceil(validation_events / batch_size)
    if expected_batches != config["validation_batches"]:
        raise ValueError("validation_batches does not use exact-event ceiling semantics")
    validation_remainder = validation_events % batch_size

    selection_path = ROOT / config["data"]
    selection = load_hashed_manifest(
        selection_path, expected_version="hypertagging-training-selection-v1"
    )
    if selection["selection_name"] != "train_865k":
        raise ValueError("full-scale config is not bound to train_865k")
    if selection["selection_includes_test"] is not False:
        raise ValueError("full-scale selection includes sealed test")
    if selection["split_counts"] != {
        "test": 0,
        "train": 865_000,
        "validation": 50_000,
    }:
        raise ValueError("full-scale selection counts are not canonical")
    if {entry["split"] for entry in selection["entries"]} != {
        "train",
        "validation",
    }:
        raise ValueError("full-scale selection contains a forbidden role")

    index_path = ROOT / config["dataset_index"]
    # Validate the index's internal content hash and schema without rescanning
    # payloads; the build step already performed the event-level identity gate.
    index = load_dataset_index(index_path, verify_sources=False)
    if index.get("selection_contract", {}).get("selection_manifest_hash") != selection[
        "manifest_hash"
    ]:
        raise ValueError("index is not bound to the full-scale selection hash")
    if index.get("selection_contract", {}).get("included_splits") != [
        "train",
        "validation",
    ]:
        raise ValueError("index includes a role other than train and validation")
    identity = index.get("event_identity_validation", {})
    if identity.get("status") != "passed" or identity.get("sealed_test_opened") is not False:
        raise ValueError("index identity gate is not passed or claims sealed-test access")
    index_counts = index.get("split_counts", {})
    if (
        index_counts.get("train") != 865_000
        or index_counts.get("validation") != 50_000
        or index_counts.get("test", 0) != 0
        or set(index_counts) - {"train", "validation", "test"}
    ):
        raise ValueError("index split counts are not canonical")

    result = {
        "status": "valid",
        "config": str(args.config),
        "selection_manifest_hash": selection["manifest_hash"],
        "index_hash": index.get("index_hash"),
        "train_events": 865_000,
        "validation_events": validation_events,
        "validation_batches": expected_batches,
        "validation_final_partial_batch_events": validation_remainder,
        "optimizer_steps": config["max_steps"],
        "presentations": config["max_steps"] * config["batch_size"],
        "two_pool_presentations": 2 * 865_000,
        "presentation_excess_over_two_pools": 48,
        "curriculum_phase_steps": config["curriculum_phase_steps"],
        "lr_schedule_contract": lr_contract,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
