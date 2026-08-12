import json
import hashlib

import pytest
import torch

from hypertagging.data.training_selection import (
    INVENTORY_VERSION,
    SELECTION_MANIFEST_VERSION,
    assign_source_roles,
    build_training_selection,
    canonical_manifest_hash,
    inventory_publications,
    load_hashed_manifest,
    load_training_selection,
    validate_nested_selections,
    write_hashed_manifest,
)


def _inventory(tmp_path, *, categories=("a", "b", "c"), shards_per_category=6):
    entries = []
    task_id = 0
    for category in categories:
        for _ in range(shards_per_category):
            path = tmp_path / f"shard_{task_id:03d}.parquet"
            path.touch()
            sidecar = path.with_suffix(path.suffix + ".metadata.json")
            marker = path.with_suffix(path.suffix + ".complete")
            sidecar.write_text(json.dumps({"task_id": task_id}) + "\n")
            marker.write_text(json.dumps({"parquet_sha256": "c" * 64}) + "\n")
            sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            marker_hash = hashlib.sha256(marker.read_bytes()).hexdigest()
            entry = {
                "path": path.name,
                "schema_version": "direct-mdst-tree-v4",
                "campaign_id": "campaign-test",
                "source_git_commit": "a" * 40,
                "source_git_tree": "b" * 40,
                "task_id": task_id,
                "task_record_hash": f"{task_id:064x}",
                "source_file": f"source_{task_id:03d}.root",
                "category": category,
                "event_count": 5,
                "parquet_sha256_reference": "c" * 64,
                "sidecar_sha256": sidecar_hash,
                "completion_marker_sha256": marker_hash,
            }
            entry["inventory_entry_hash"] = canonical_manifest_hash(entry)
            entries.append(entry)
            task_id += 1
    inventory = {
        "manifest_version": INVENTORY_VERSION,
        "data_root": str(tmp_path),
        "entries": entries,
    }
    inventory["manifest_hash"] = canonical_manifest_hash(inventory)
    return inventory


def _roles(inventory, seed=20260812):
    quotas = {category: 1 for category in ("a", "b", "c")}
    roles = assign_source_roles(
        inventory,
        seed=seed,
        validation_quotas=quotas,
        test_quotas=quotas,
        stress_quotas=quotas,
    )
    roles["manifest_hash"] = canonical_manifest_hash(roles)
    return roles


def test_role_selection_is_deterministic_and_meets_category_quotas(tmp_path):
    inventory = _inventory(tmp_path)
    first = _roles(inventory)
    second = _roles(inventory)
    assert first == second
    assert first["role_shard_counts"] == {
        "stress": 3,
        "test": 3,
        "training_pool": 9,
        "validation": 3,
    }
    for counts in first["category_role_shard_counts"].values():
        assert counts == {
            "stress": 1,
            "test": 1,
            "training_pool": 3,
            "validation": 1,
        }


def test_sources_and_tasks_are_isolated_and_training_sets_are_nested(tmp_path):
    inventory = _inventory(tmp_path)
    roles = _roles(inventory)
    small = build_training_selection(
        inventory,
        roles,
        selection_name="small",
        training_quotas={"a": 1, "b": 1, "c": 1},
    )
    large = build_training_selection(
        inventory,
        roles,
        selection_name="large",
        training_quotas={"a": 2, "b": 2, "c": 2},
    )
    validate_nested_selections((small, large))
    for selection in (small, large):
        source_splits = {}
        task_splits = {}
        for entry in selection["entries"]:
            assert (
                source_splits.setdefault(entry["source_file"], entry["split"])
                == entry["split"]
            )
            assert (
                task_splits.setdefault(entry["task_id"], entry["split"])
                == entry["split"]
            )
    assert small["split_counts"] == {"test": 15, "train": 15, "validation": 15}
    assert large["split_counts"] == {"test": 15, "train": 30, "validation": 15}


def test_loader_rejects_manifest_tampering_and_raw_prefix(tmp_path):
    from hypertagging.training.data_module import build_real_data_module

    inventory = _inventory(tmp_path)
    roles = _roles(inventory)
    selection = build_training_selection(
        inventory,
        roles,
        selection_name="small",
        training_quotas={"a": 1, "b": 1, "c": 1},
    )
    path = write_hashed_manifest(selection, tmp_path / "selection.json")
    loaded = load_training_selection(path)
    assert loaded.split_counts == {"test": 15, "train": 15, "validation": 15}
    with pytest.raises(ValueError, match="max_events prefixes"):
        build_real_data_module(path, max_events=1)
    payload = json.loads(path.read_text())
    payload["selection_name"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_hashed_manifest(path, expected_version=SELECTION_MANIFEST_VERSION)


def test_scientific_mode_rejects_raw_data_before_scanning(tmp_path):
    raw = tmp_path / "raw.parquet"
    raw.touch()
    from hypertagging.training.data_module import build_real_data_module

    with pytest.raises(ValueError, match="promoted full-record dataset index"):
        build_real_data_module(raw, scientific_mode=True)


def test_selection_handoff_fits_normalizers_on_training_role_only(tmp_path):
    from hypertagging.data.notebook_fixtures import notebook_fixture_trees
    from hypertagging.preprocessing.schema_v4 import export_trees_v4
    from hypertagging.training.data_module import build_real_data_module

    entries = []
    for task_id, split in enumerate(("train", "validation", "test")):
        source_file = f"source_{task_id}.root"
        trees = notebook_fixture_trees()
        for tree in trees:
            tree.metadata["source_file"] = source_file
            tree.metadata["source_category"] = "fixture"
            tree.metadata["event_uid"] = f"fixture:{task_id}:{tree.event_id}:0"
        parquet = export_trees_v4(
            trees,
            tmp_path / f"source_{task_id}.parquet",
            metadata={
                "source_file": source_file,
                "category": "fixture",
                "task_id": task_id,
                "task_record_hash": f"{task_id + 10:064x}",
            },
            event_buffer_size=1,
            row_group_size=1,
        )
        sidecar = parquet.with_suffix(parquet.suffix + ".metadata.json")
        marker = parquet.with_suffix(parquet.suffix + ".complete")
        completion = json.loads(marker.read_text())
        entries.append(
            {
                "inventory_entry_hash": f"{task_id + 1:064x}",
                "path": parquet.name,
                "schema_version": "direct-mdst-tree-v4",
                "campaign_id": "fixture",
                "source_git_commit": "a" * 40,
                "source_git_tree": "b" * 40,
                "task_id": task_id,
                "task_record_hash": f"{task_id + 10:064x}",
                "source_file": source_file,
                "category": "fixture",
                "event_count": 2,
                "parquet_sha256_reference": completion["parquet_sha256"],
                "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                "completion_marker_sha256": hashlib.sha256(
                    marker.read_bytes()
                ).hexdigest(),
                "split": split,
            }
        )
    selection = {
        "manifest_version": SELECTION_MANIFEST_VERSION,
        "selection_name": "normalizer-handoff",
        "data_root": str(tmp_path),
        "inventory_hash": "1" * 64,
        "roles_hash": "2" * 64,
        "selection_seed": 20260812,
        "selection_mode": "explicit_whole_shard_source_roles",
        "max_events_prefix_allowed": False,
        "normalizer_scope": "train_split_only",
        "uid_validation": {
            "status": "pending_full_index_build",
            "gate": "required_before_scientific_training",
        },
        "split_counts": {"test": 2, "train": 2, "validation": 2},
        "split_shard_counts": {"test": 1, "train": 1, "validation": 1},
        "entries": entries,
    }
    manifest = write_hashed_manifest(selection, tmp_path / "selection.json")
    module = build_real_data_module(
        manifest,
        scientific_mode=False,
        required_splits=("train", "validation", "test"),
    )
    train_events = list(module.iter_events("train"))
    all_events = (
        train_events
        + list(module.iter_events("validation"))
        + list(module.iter_events("test"))
    )
    assert [
        len(list(module.iter_events(name))) for name in ("train", "validation", "test")
    ] == [2, 2, 2]
    expected_train_count = sum(
        (event.track_availability.sum(dim=0) for event in train_events),
        start=torch.zeros_like(module.normalizers["track"].count),
    )
    all_count = sum(
        (event.track_availability.sum(dim=0) for event in all_events),
        start=torch.zeros_like(module.normalizers["track"].count),
    )
    assert torch.equal(module.normalizers["track"].count, expected_train_count)
    assert bool((all_count > module.normalizers["track"].count).any())
    loaded = load_training_selection(
        manifest, include_splits=("train", "validation")
    )
    assert module.selection_manifest_hash == loaded.manifest_hash

    from hypertagging.data.dataset_index import build_dataset_index

    index = build_dataset_index(
        loaded.paths,
        tmp_path / "index.json",
        source_split_overrides=loaded.source_split_overrides,
        selection_manifest_hash=loaded.manifest_hash,
        selection_included_splits=loaded.included_splits,
        source_expectations=loaded.source_expectations,
        require_event_identity_validation=True,
    )
    indexed_module = build_real_data_module(
        manifest,
        scientific_mode=True,
        required_splits=("train", "validation"),
        dataset_index=index,
    )
    assert indexed_module.split_counts == {"train": 2, "validation": 2, "test": 0}
    assert indexed_module.dataset_index["normalizer_scope"] == "train"

    from hypertagging.data.dataset_index import _index_hash

    original = json.loads(index.read_text())
    for name, mutate, message in (
        (
            "test-role",
            lambda payload: payload["selection_contract"].update(
                included_splits=["train", "validation", "test"]
            ),
            "exactly train and validation",
        ),
        (
            "legacy-index",
            lambda payload: payload.pop("event_identity_validation"),
            "identity/task-binding gate",
        ),
        (
            "sidecar-index",
            lambda payload: payload.update(event_identity_validation={}),
            "identity/task-binding gate",
        ),
    ):
        payload = json.loads(json.dumps(original))
        mutate(payload)
        payload["index_hash"] = _index_hash(payload)
        bad_index = tmp_path / f"{name}.json"
        bad_index.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=message):
            build_real_data_module(
                manifest,
                scientific_mode=True,
                required_splits=("train", "validation"),
                dataset_index=bad_index,
            )


def test_inventory_rejects_sidecar_tampering_without_hashing_parquet(tmp_path):
    from hypertagging.data.notebook_fixtures import notebook_fixture_trees
    from hypertagging.preprocessing.schema_v4 import export_trees_v4

    trees = notebook_fixture_trees()
    for tree in trees:
        tree.metadata["source_file"] = "inventory.root"
        tree.metadata["source_category"] = "fixture"
    shard = export_trees_v4(
        trees,
        tmp_path / "inventory.parquet",
        metadata={
            "source_file": "inventory.root",
            "source_git_commit": "a" * 40,
            "source_git_tree": "b" * 40,
            "task_id": 7,
            "task_record_hash": "c" * 64,
            "campaign_id": "fixture-campaign",
            "physics_category": "fixture",
        },
    )
    inventory = inventory_publications(tmp_path)
    assert inventory["event_count"] == 2
    assert inventory["content_validation_scope"]["parquet_payload"].endswith(
        "not_rehashed"
    )
    sidecar = shard.with_suffix(shard.suffix + ".metadata.json")
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(ValueError, match="sidecar digest mismatch"):
        inventory_publications(tmp_path)
