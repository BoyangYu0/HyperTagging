#!/usr/bin/env python
"""Build or validate immutable source-role training-selection manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.training_selection import (  # noqa: E402
    INVENTORY_VERSION,
    ROLE_MANIFEST_VERSION,
    SELECTION_MANIFEST_VERSION,
    SUMMARY_VERSION,
    assign_source_roles,
    build_training_selection,
    inventory_publications,
    load_hashed_manifest,
    load_training_selection,
    validate_nested_selections,
    write_hashed_manifest,
)


DEFAULT_SEED = 20260812
EXPECTED_CAMPAIGN = "campaign-fb070c6c9805-f4e54df23b5c"
EXPECTED_SOURCE_COMMIT = "f4e54df23b5c60115e475c5d68df4651899d678e"
EXPECTED_SOURCE_TREE = "b6e3a4118b960e3a4676a61af9601438d56cef96"
EXPECTED_SCHEMA = "direct-mdst-tree-v4"
EXPECTED_CATEGORY_SHARDS = {
    "ccbar": 20,
    "charged": 18,
    "ddbar": 16,
    "mixed": 64,
    "ssbar": 16,
    "taupair": 21,
    "uubar": 45,
}
HELD_OUT_QUOTAS = {
    "validation": {
        "ccbar": 2,
        "charged": 1,
        "ddbar": 1,
        "mixed": 1,
        "ssbar": 1,
        "taupair": 2,
        "uubar": 2,
    },
    "test": {
        "ccbar": 2,
        "charged": 1,
        "ddbar": 1,
        "mixed": 1,
        "ssbar": 1,
        "taupair": 2,
        "uubar": 2,
    },
    "stress": {
        "ccbar": 1,
        "charged": 1,
        "ddbar": 1,
        "mixed": 1,
        "ssbar": 1,
        "taupair": 1,
        "uubar": 1,
    },
}
TRAINING_QUOTAS = {
    "train_035k": {
        "ccbar": 1,
        "charged": 1,
        "ddbar": 1,
        "mixed": 1,
        "ssbar": 1,
        "taupair": 1,
        "uubar": 1,
    },
    "train_100k": {
        "ccbar": 4,
        "charged": 3,
        "ddbar": 2,
        "mixed": 2,
        "ssbar": 2,
        "taupair": 3,
        "uubar": 4,
    },
    "train_250k": {
        "ccbar": 11,
        "charged": 7,
        "ddbar": 4,
        "mixed": 6,
        "ssbar": 4,
        "taupair": 8,
        "uubar": 10,
    },
    # The full production training role is the complete 173-shard pool.
    # These quotas are derived from the canonical role allocation and keep
    # selection generation deterministic without hand-authoring shard paths.
    "train_865k": {
        "ccbar": 15,
        "charged": 15,
        "ddbar": 13,
        "mixed": 61,
        "ssbar": 13,
        "taupair": 16,
        "uubar": 40,
    },
}

SELECTION_INCLUDES_TEST = {
    "train_035k": True,
    "train_100k": True,
    "train_250k": True,
    "train_865k": False,
}


def build(
    data_root: Path,
    output_dir: Path,
    seed: int,
    *,
    reuse_manifests: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if reuse_manifests:
        inventory = load_hashed_manifest(
            output_dir / "inventory.json", expected_version=INVENTORY_VERSION
        )
        roles = load_hashed_manifest(
            output_dir / "roles.json", expected_version=ROLE_MANIFEST_VERSION
        )
    else:
        inventory = inventory_publications(data_root)
        inventory_path = write_hashed_manifest(inventory, output_dir / "inventory.json")
        inventory = load_hashed_manifest(
            inventory_path, expected_version=INVENTORY_VERSION
        )
        roles = assign_source_roles(
            inventory,
            seed=seed,
            validation_quotas=HELD_OUT_QUOTAS["validation"],
            test_quotas=HELD_OUT_QUOTAS["test"],
            stress_quotas=HELD_OUT_QUOTAS["stress"],
        )
        roles_path = write_hashed_manifest(roles, output_dir / "roles.json")
        roles = load_hashed_manifest(
            roles_path, expected_version=ROLE_MANIFEST_VERSION
        )
    _validate_reduced_campaign(inventory)
    if roles["inventory_hash"] != inventory["manifest_hash"]:
        raise ValueError("roles manifest does not reference the inventory hash")
    if int(roles["selection_seed"]) != int(seed):
        raise ValueError("canonical roles manifest was generated with a different seed")
    selections = []
    selection_hashes = {}
    for name, quotas in TRAINING_QUOTAS.items():
        selection = build_training_selection(
            inventory,
            roles,
            selection_name=name,
            training_quotas=quotas,
            include_test=SELECTION_INCLUDES_TEST[name],
        )
        path = write_hashed_manifest(selection, output_dir / f"{name}.json")
        checked = load_hashed_manifest(
            path, expected_version=SELECTION_MANIFEST_VERSION
        )
        load_training_selection(path, include_splits=("train", "validation"))
        selections.append(checked)
        selection_hashes[name] = checked["manifest_hash"]
    validate_nested_selections(
        selection
        for selection in selections
        if selection.get("selection_includes_test", True)
    )
    roles_by_name = {entry["role"]: [] for entry in roles["entries"]}
    for entry in roles["entries"]:
        roles_by_name[entry["role"]].append(entry)
    training_pool_categories = Counter(
        entry["category"] for entry in roles_by_name["training_pool"]
    )
    summary = {
        "manifest_version": SUMMARY_VERSION,
        "generated_by": "scripts/build_training_selection.py",
        "data_root": inventory["data_root"],
        "selection_seed": seed,
        "inventory_hash": inventory["manifest_hash"],
        "roles_hash": roles["manifest_hash"],
        "selection_hashes": selection_hashes,
        "inventory_shards": inventory["shard_count"],
        "inventory_events": inventory["event_count"],
        "inventory_category_shards": inventory["category_shard_counts"],
        "reduced_campaign_contract": "validated",
        "role_shards": roles["role_shard_counts"],
        "role_events": roles["role_event_counts"],
        "training_pool_category_shards": dict(sorted(training_pool_categories.items())),
        "nested_training_selections": [
            {
                "name": selection["selection_name"],
                "train_shards": selection["split_shard_counts"]["train"],
                "train_events": selection["split_counts"]["train"],
                "validation_events": selection["split_counts"]["validation"],
                "test_events": selection["split_counts"]["test"],
                "category_train_shards": {
                    category: values.get("train", 0)
                    for category, values in selection[
                        "category_split_shard_counts"
                    ].items()
                },
            }
            for selection in selections
        ],
        "nested_source_sets": "validated",
        "fixed_validation_test_sets": "validated",
        "sealed_test_selection": "excluded_from_train_865k",
        "uid_validation": "pending_full_index_build",
        "category_representativeness_note": (
            "The 1M reduced campaign is source-task sampled and skewed, especially "
            "toward mixed and uubar. The explicit quotas improve small-run coverage "
            "but do not make 500k or 865k category representative."
        ),
    }
    write_hashed_manifest(summary, output_dir / "summary.json")


def validate(output_dir: Path) -> None:
    inventory = load_hashed_manifest(
        output_dir / "inventory.json", expected_version=INVENTORY_VERSION
    )
    _validate_reduced_campaign(inventory)
    roles = load_hashed_manifest(
        output_dir / "roles.json", expected_version=ROLE_MANIFEST_VERSION
    )
    if roles["inventory_hash"] != inventory["manifest_hash"]:
        raise ValueError("roles manifest does not reference the inventory hash")
    selections = []
    for name in TRAINING_QUOTAS:
        path = output_dir / f"{name}.json"
        selection = load_hashed_manifest(
            path, expected_version=SELECTION_MANIFEST_VERSION
        )
        if selection["inventory_hash"] != inventory["manifest_hash"]:
            raise ValueError(f"{name} does not reference the inventory hash")
        if selection["roles_hash"] != roles["manifest_hash"]:
            raise ValueError(f"{name} does not reference the roles hash")
        # Do not open sealed-test payloads during manifest validation.  The
        # production selection excludes that role; historical smaller
        # selections retain their old schema but are checked only on the
        # train/validation roles here.
        load_training_selection(path, include_splits=("train", "validation"))
        if name == "train_865k":
            if selection.get("selection_includes_test") is not False:
                raise ValueError("train_865k must exclude the sealed-test role")
            if selection.get("split_counts", {}).get("train") != 865_000:
                raise ValueError("train_865k must contain exactly 865000 train events")
            if selection.get("split_counts", {}).get("validation") != 50_000:
                raise ValueError("train_865k must contain exactly 50000 validation events")
            if selection.get("split_counts", {}).get("test", 0) != 0:
                raise ValueError("train_865k contains sealed-test events")
            if set(selection.get("excluded_roles", [])) != {"stress", "test"}:
                raise ValueError("train_865k excluded-role contract is incomplete")
        elif selection.get("selection_includes_test", True) is not True:
            raise ValueError(f"{name} must retain its historical test-role schema")
        selections.append(selection)
    validate_nested_selections(
        selection
        for selection in selections
        if selection.get("selection_includes_test", True)
    )
    summary = load_hashed_manifest(
        output_dir / "summary.json", expected_version=SUMMARY_VERSION
    )
    if summary["inventory_hash"] != inventory["manifest_hash"]:
        raise ValueError("summary does not reference the inventory hash")
    if summary["roles_hash"] != roles["manifest_hash"]:
        raise ValueError("summary does not reference the roles hash")


def _validate_reduced_campaign(inventory: dict) -> None:
    expected_tasks = list(range(9, 2000, 10))
    entries = inventory["entries"]
    checks = {
        "campaign": inventory.get("campaigns") == [EXPECTED_CAMPAIGN],
        "source commit": inventory.get("source_git_commits")
        == [EXPECTED_SOURCE_COMMIT],
        "source tree": inventory.get("source_git_trees") == [EXPECTED_SOURCE_TREE],
        "schema": inventory.get("schema_versions") == [EXPECTED_SCHEMA],
        "shard count": int(inventory.get("shard_count", -1)) == 200,
        "event count": int(inventory.get("event_count", -1)) == 1_000_000,
        "category composition": inventory.get("category_shard_counts")
        == EXPECTED_CATEGORY_SHARDS,
        "task schedule": sorted(int(entry["task_id"]) for entry in entries)
        == expected_tasks,
        "events per shard": {int(entry["event_count"]) for entry in entries} == {5_000},
        "KLM scope": {str(entry["klm_training_scope"]) for entry in entries}
        == {"included"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "data root does not match the canonical reduced 1M campaign: "
            + ", ".join(failed)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/project/agkuhr/users/boyang/data/HyperTagging_uni"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--reuse-manifests",
        action="store_true",
        help=(
            "reuse the validated inventory.json and roles.json; this is the "
            "sealed-test-safe path for deterministic selection generation"
        ),
    )
    args = parser.parse_args()
    if args.validate_only:
        validate(args.output_dir)
    else:
        build(
            args.data_root,
            args.output_dir,
            args.seed,
            reuse_manifests=args.reuse_manifests,
        )
        validate(args.output_dir)
    print(json.dumps({"status": "valid", "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
