import json
import hashlib
from copy import deepcopy
import inspect
from pathlib import Path, PurePosixPath

import pytest
import torch

from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.data.training_selection import (
    INVENTORY_VERSION,
    SELECTION_MANIFEST_VERSION,
    assign_source_roles,
    build_training_selection,
    canonical_manifest_hash,
    inventory_publications,
    load_hashed_manifest,
    load_training_selection,
    validate_training_selection_metadata,
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
                "campaign_config_digest": "d" * 64,
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


def _metadata_only_selection(tmp_path, splits=("train", "validation", "test")):
    entries = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_shard_counts = {"train": 0, "validation": 0, "test": 0}
    for task_id, split in enumerate(splits):
        entries.append(
            {
                "inventory_entry_hash": f"{task_id + 1:064x}",
                "path": f"shard_{task_id}.parquet",
                "schema_version": "direct-mdst-tree-v4",
                "campaign_id": "fixture",
                "campaign_config_digest": "d" * 64,
                "source_git_commit": "a" * 40,
                "source_git_tree": "b" * 40,
                "task_id": task_id,
                "task_record_hash": f"{task_id + 10:064x}",
                "source_file": f"source_{task_id}.root",
                "category": "fixture",
                "event_count": task_id + 1,
                "parquet_sha256_reference": f"{task_id + 20:064x}",
                "sidecar_sha256": f"{task_id + 30:064x}",
                "completion_marker_sha256": f"{task_id + 40:064x}",
                "split": split,
            }
        )
        split_counts[split] += task_id + 1
        split_shard_counts[split] += 1
    entries.sort(key=lambda entry: (entry["split"], entry["category"], entry["task_id"]))
    payload = {
        "manifest_version": SELECTION_MANIFEST_VERSION,
        "data_root": str(tmp_path),
        "selection_name": "metadata-only-fixture",
        "inventory_hash": "1" * 64,
        "roles_hash": "2" * 64,
        "selection_seed": 20260812,
        "training_category_shard_quotas": {"fixture": 1},
        "selection_mode": "explicit_whole_shard_source_roles",
        "max_events_prefix_allowed": False,
        "normalizer_scope": "train_split_only",
        "uid_validation": {
            "status": "pending_full_index_build",
            "gate": "required_before_scientific_training",
        },
        "source_split_isolation": "validated",
        "task_split_isolation": "validated",
        "split_counts": split_counts,
        "split_shard_counts": split_shard_counts,
        "category_split_shard_counts": {
            "fixture": {
                split: split_shard_counts[split]
                for split in splits
                if split_shard_counts[split]
            }
        },
        "entries": entries,
    }
    if "test" not in splits:
        payload["selection_includes_test"] = False
        payload["excluded_roles"] = ["stress", "test"]
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    return payload


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


def test_full_training_selection_excludes_sealed_test_by_construction(tmp_path):
    inventory = _inventory(tmp_path)
    roles = _roles(inventory)
    selection = build_training_selection(
        inventory,
        roles,
        selection_name="train_865k",
        training_quotas={"a": 3, "b": 3, "c": 3},
        include_test=False,
    )
    assert selection["selection_includes_test"] is False
    assert selection["excluded_roles"] == ["stress", "test"]
    assert {entry["split"] for entry in selection["entries"]} == {
        "train",
        "validation",
    }
    assert selection["split_counts"]["test"] == 0
    assert selection["split_shard_counts"]["test"] == 0


def test_true_test_builder_rejects_empty_test_role(tmp_path):
    inventory = _inventory(tmp_path)
    roles = _roles(inventory)
    for entry in roles["entries"]:
        if entry["role"] == "test":
            entry["role"] = "stress"

    with pytest.raises(ValueError, match="positive test counts and shards"):
        build_training_selection(
            inventory,
            roles,
            selection_name="missing-test",
            training_quotas={"a": 1, "b": 1, "c": 1},
            include_test=True,
        )


def test_selection_variants_and_projected_entry_shape_are_exact(tmp_path):
    inventory = _inventory(tmp_path)
    roles = _roles(inventory)
    with_test = build_training_selection(
        inventory,
        roles,
        selection_name="with-test",
        training_quotas={"a": 1, "b": 1, "c": 1},
    )
    without_test = build_training_selection(
        inventory,
        roles,
        selection_name="without-test",
        training_quotas={"a": 1, "b": 1, "c": 1},
        include_test=False,
    )
    assert "selection_includes_test" not in with_test
    assert "excluded_roles" not in with_test
    assert without_test["selection_includes_test"] is False
    assert without_test["excluded_roles"] == ["stress", "test"]
    expected_entry_keys = {
        "inventory_entry_hash",
        "path",
        "schema_version",
        "campaign_id",
        "campaign_config_digest",
        "source_git_commit",
        "source_git_tree",
        "task_id",
        "task_record_hash",
        "source_file",
        "category",
        "event_count",
        "parquet_sha256_reference",
        "sidecar_sha256",
        "completion_marker_sha256",
        "split",
    }
    assert all(set(entry) == expected_entry_keys for entry in with_test["entries"])
    assert PurePosixPath(with_test["data_root"]).is_absolute()
    assert str(PurePosixPath(with_test["data_root"])) == with_test["data_root"]
    assert all(
        PurePosixPath(entry["path"]).name == entry["path"]
        and "/" not in entry["path"]
        and "\\" not in entry["path"]
        and entry["path"].endswith(".parquet")
        for entry in with_test["entries"]
    )
    assert with_test["entries"] == sorted(
        with_test["entries"],
        key=lambda entry: (
            entry["split"],
            entry["category"],
            entry["task_id"],
        ),
    )
    roundtrip = load_hashed_manifest(
        write_hashed_manifest(with_test, tmp_path / "with-test.json"),
        expected_version=SELECTION_MANIFEST_VERSION,
    )
    assert validate_training_selection_metadata(roundtrip) == (
        "train",
        "validation",
        "test",
    )
    assert roundtrip["entries"] == with_test["entries"]

    bad_true = deepcopy(with_test)
    bad_true["selection_includes_test"] = True
    bad_true["manifest_hash"] = canonical_manifest_hash(bad_true)
    with pytest.raises(ValueError, match="top-level schema"):
        validate_training_selection_metadata(bad_true)

    bad_false = deepcopy(without_test)
    bad_false["excluded_roles"] = ["test"]
    bad_false["manifest_hash"] = canonical_manifest_hash(bad_false)
    with pytest.raises(ValueError, match="false-test selection variant"):
        validate_training_selection_metadata(bad_false)

    false_without_excluded_count = deepcopy(without_test)
    false_without_excluded_count["split_counts"].pop("test")
    false_without_excluded_count["manifest_hash"] = canonical_manifest_hash(
        false_without_excluded_count
    )
    assert validate_training_selection_metadata(false_without_excluded_count) == (
        "train",
        "validation",
    )


@pytest.mark.parametrize(
    "manifest_name",
    ("train_035k.json", "train_100k.json", "train_250k.json", "train_865k.json"),
)
def test_tracked_authoritative_inventory_projects_exact_entry_schema_and_order(
    monkeypatch, manifest_name
):
    import hypertagging.data.training_selection as training_selection_module

    source = Path(
        "configs/training_selection/production_1m_20260812"
    ) / manifest_name
    payload = json.loads(source.read_text())
    inventory = json.loads(
        source.with_name("inventory.json").read_text()
    )
    inventory_by_hash = {
        entry["inventory_entry_hash"]: entry for entry in inventory["entries"]
    }
    for entry in payload["entries"]:
        entry["campaign_config_digest"] = inventory_by_hash[
            entry["inventory_entry_hash"]
        ]["campaign_config_digest"]
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    expected_entry_keys = {
        "inventory_entry_hash",
        "path",
        "schema_version",
        "campaign_id",
        "campaign_config_digest",
        "source_git_commit",
        "source_git_tree",
        "task_id",
        "task_record_hash",
        "source_file",
        "category",
        "event_count",
        "parquet_sha256_reference",
        "sidecar_sha256",
        "completion_marker_sha256",
        "split",
    }

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached during authoritative metadata check")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)
    assert all(set(entry) == expected_entry_keys for entry in payload["entries"])
    assert payload["entries"] == sorted(
        payload["entries"],
        key=lambda entry: (entry["split"], entry["category"], entry["task_id"]),
    )
    validate_training_selection_metadata(payload)


@pytest.mark.campaign_artifacts('artifacts/experiment_readiness/production_1m_20260812/train_035k/train_035k.complete_only.index.json')
def test_promoted_manifest_index_preflight_resolves_only_index_paths(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    tracked_manifest = Path(
        "configs/training_selection/production_1m_20260812/train_035k.json"
    )
    tracked_index = Path(
        "artifacts/experiment_readiness/production_1m_20260812/train_035k/"
        "train_035k.complete_only.index.json"
    )
    manifest_payload = json.loads(tracked_manifest.read_text())
    inventory = json.loads(tracked_manifest.with_name("inventory.json").read_text())
    inventory_by_hash = {
        entry["inventory_entry_hash"]: entry for entry in inventory["entries"]
    }
    for entry in manifest_payload["entries"]:
        entry["campaign_config_digest"] = inventory_by_hash[
            entry["inventory_entry_hash"]
        ]["campaign_config_digest"]
    manifest = write_hashed_manifest(manifest_payload, tmp_path / "train_035k.json")
    index_payload = json.loads(tracked_index.read_text())
    index_payload["selection_contract"]["selection_manifest_hash"] = json.loads(
        manifest.read_text()
    )["manifest_hash"]
    index_payload["selection_contract"]["fingerprint"] = (
        dataset_index_module._selection_fingerprint_from_strings(
            index_payload["paths"],
            mode="source_role_manifest",
            max_events=None,
            selection_manifest_hash=index_payload["selection_contract"][
                "selection_manifest_hash"
            ],
        )
    )
    index_payload["index_hash"] = dataset_index_module._index_hash(index_payload)
    index = tmp_path / "train_035k.index.json"
    index.write_text(json.dumps(index_payload))
    expected_resolve_inputs = list(index_payload["paths"]) + [
        shard["path"] for shard in index_payload["shards"]
    ]
    resolve_inputs = []
    original_resolve = dataset_index_module.Path.resolve

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached during promoted metadata preflight")

    def observed_resolve(path, *args, **kwargs):
        resolve_inputs.append(str(path))
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(dataset_index_module.Path, "resolve", observed_resolve)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(dataset_index_module, "_sha256_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)
    monkeypatch.setattr(training_selection_module.pq, "ParquetFile", bomb)
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)

    metadata = data_module.preflight_dataset_index_data_binding(
        manifest,
        index,
        required_splits=("train", "validation"),
    )
    assert metadata["index_hash"] == index_payload["index_hash"]
    assert metadata["selection_contract"]["selection_manifest_hash"] == json.loads(
        manifest.read_text()
    )["manifest_hash"]
    assert resolve_inputs == expected_resolve_inputs


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.pop("selection_mode"), "top-level schema"),
        (
            lambda payload: payload.update(max_events_prefix_allowed=0),
            "max-events gate",
        ),
        (
            lambda payload: payload.update(normalizer_scope="all_splits"),
            "normalizer scope",
        ),
        (
            lambda payload: payload.pop("source_split_isolation"),
            "top-level schema",
        ),
        (
            lambda payload: payload.update(task_split_isolation=True),
            "task-split isolation",
        ),
        (
            lambda payload: payload["uid_validation"].update(status="passed"),
            "UID-validation gate",
        ),
        (
            lambda payload: payload["uid_validation"].pop("gate"),
            "UID-validation gate",
        ),
        (lambda payload: payload.update(inventory_hash="A" * 64), "inventory_hash"),
        (lambda payload: payload.update(roles_hash=[]), "roles_hash"),
        (lambda payload: payload.update(entries=[]), "no entries"),
        (lambda payload: payload.update(entries=[[]]), "must be an object"),
        (
            lambda payload: payload["entries"][0].update(path="../escape"),
            "parent traversal",
        ),
        (lambda payload: payload["entries"][0].update(source_file=[]), "source_file"),
        (lambda payload: payload["entries"][0].update(task_id=True), "task_id"),
        (
            lambda payload: payload["entries"][0].update(event_count=False),
            "event_count",
        ),
        (
            lambda payload: payload["entries"][0].update(inventory_entry_hash="F" * 64),
            "inventory_entry_hash",
        ),
        (
            lambda payload: payload["entries"][0].update(task_record_hash={}),
            "task_record_hash",
        ),
        (
            lambda payload: payload["entries"][0].update(
                campaign_config_digest="A" * 64
            ),
            "campaign_config_digest",
        ),
        (
            lambda payload: payload["entries"][0].update(unexpected="field"),
            "builder metadata",
        ),
        (
            lambda payload: payload.update(entries=list(reversed(payload["entries"]))),
            "not sorted",
        ),
        (
            lambda payload: payload["split_counts"].pop("test"),
            "split_counts",
        ),
        (
            lambda payload: payload["split_shard_counts"].update(train=True),
            "split_shard_counts",
        ),
        (
            lambda payload: payload["split_counts"].update(train=999),
            "split_counts disagree",
        ),
        (
            lambda payload: payload["category_split_shard_counts"]["fixture"].update(
                train=999
            ),
            "category_split_shard_counts disagree",
        ),
        (
            lambda payload: payload["training_category_shard_quotas"].update(
                fixture=2
            ),
            "category quotas disagree",
        ),
        (
            lambda payload: payload.update(training_category_shard_quotas={}),
            "category quotas are invalid",
        ),
        (
            lambda payload: payload.update(selection_includes_test=False),
            "top-level schema",
        ),
        (
            lambda payload: payload["entries"].append(deepcopy(payload["entries"][0])),
            "duplicate lexical paths",
        ),
        (
            lambda payload: payload["entries"][1].update(
                source_file=payload["entries"][0]["source_file"]
            ),
            "duplicate source_file",
        ),
        (
            lambda payload: payload["entries"][1].update(
                task_id=payload["entries"][0]["task_id"]
            ),
            "duplicate task_id",
        ),
    ),
)
def test_full_loader_rejects_manifest_metadata_before_source_effects(
    tmp_path, monkeypatch, mutation, message
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path)
    mutation(payload)
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before manifest metadata binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)

    with pytest.raises(ValueError, match=message):
        load_training_selection(manifest)


@pytest.mark.parametrize(
    "invalid_path",
    (
        "/absolute.parquet",
        "",
        ".",
        "./shard.parquet",
        "nested/shard.parquet",
        "nested//shard.parquet",
        "nested/./shard.parquet",
        "..",
        "nested/../shard.parquet",
        "shard.parquet/",
        "shard.txt",
        "nul\x00shard.parquet",
        "control\x7fshard.parquet",
        r"backslash\shard.parquet",
    ),
)
def test_entry_paths_are_strict_builder_relative_before_source_effects(
    tmp_path, monkeypatch, invalid_path
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path)
    payload["entries"][0]["path"] = invalid_path
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before lexical path binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="entry path"):
        load_training_selection(manifest)


@pytest.mark.parametrize(
    "invalid_root",
    (
        "relative/data",
        "{root}/",
        "{root}//nested",
        "{root}/./nested",
        "{root}/../nested",
        "{root}/bad\\component",
        "{root}/control\x7fcomponent",
    ),
)
def test_data_root_is_absolute_byte_canonical_before_source_effects(
    tmp_path, monkeypatch, invalid_root
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path)
    payload["data_root"] = invalid_root.format(root=tmp_path)
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before data_root binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="data_root"):
        load_training_selection(manifest)


def test_direct_child_paths_and_canonical_root_are_retained_by_pure_validation(
    tmp_path,
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path)
    validated = training_selection_module._validated_training_selection_metadata(
        payload
    )

    assert validated.data_root == str(tmp_path)
    assert validated.entry_paths == tuple(entry["path"] for entry in payload["entries"])
    assert all("/" not in path and "\\" not in path for path in validated.entry_paths)
    assert all(path.endswith(".parquet") for path in validated.entry_paths)


def test_full_loader_consumes_the_pure_validator_canonical_path(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    canonical = root / "canonical.parquet"
    canonical.touch()
    canonical.with_suffix(canonical.suffix + ".metadata.json").touch()
    canonical.with_suffix(canonical.suffix + ".complete").touch()
    manifest = write_hashed_manifest(
        _metadata_only_selection(root, splits=("train",)),
        tmp_path / "selection.json",
    )
    original_validator = training_selection_module._validate_entry_relative_path
    publications = []

    def canonicalizing_validator(value):
        validated = original_validator(value)
        if validated == "shard_0.parquet":
            return canonical.name
        return validated

    def publication_reached(path, _entry):
        publications.append(path)
        raise AssertionError("canonical publication reached")

    monkeypatch.setattr(
        training_selection_module,
        "_validate_entry_relative_path",
        canonicalizing_validator,
    )
    monkeypatch.setattr(
        training_selection_module,
        "_validate_selection_publication",
        publication_reached,
    )

    with pytest.raises(AssertionError, match="canonical publication reached"):
        load_training_selection(manifest)
    assert publications == [canonical.resolve()]


def test_lexical_path_uniqueness_covers_excluded_roles_before_resolution(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path)
    train_entry = next(entry for entry in payload["entries"] if entry["split"] == "train")
    test_entry = next(entry for entry in payload["entries"] if entry["split"] == "test")
    test_entry["path"] = train_entry["path"]
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before all-role lexical uniqueness")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="duplicate lexical paths"):
        load_training_selection(manifest, include_splits=("train",))


def test_excluded_role_metadata_is_validated_before_included_path_resolution(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path)
    test_entry = next(entry for entry in payload["entries"] if entry["split"] == "test")
    test_entry["path"] = "/excluded-test.parquet"
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("included path resolved before excluded role validation")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="builder-relative"):
        load_training_selection(manifest, include_splits=("train",))


def test_excluded_and_sealed_roles_are_never_resolved_or_statted(tmp_path, monkeypatch):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    train = root / "shard_0.parquet"
    train.write_bytes(b"train")
    train.with_suffix(train.suffix + ".metadata.json").write_bytes(b"sidecar")
    train.with_suffix(train.suffix + ".complete").write_bytes(b"marker")
    manifest = write_hashed_manifest(
        _metadata_only_selection(root), tmp_path / "selection.json"
    )
    original_resolve = training_selection_module.Path.resolve
    original_stat = training_selection_module.Path.stat
    resolved: set[str] = set()
    statted: set[str] = set()

    def observed_resolve(path, *args, **kwargs):
        if path.name.startswith("shard_"):
            resolved.add(path.name)
        return original_resolve(path, *args, **kwargs)

    def observed_stat(path, *args, **kwargs):
        if path.name.startswith("shard_"):
            statted.add(path.name)
        return original_stat(path, *args, **kwargs)

    def publication_reached(*_args, **_kwargs):
        assert resolved == {"shard_0.parquet"}
        assert "shard_0.parquet" in statted
        assert not any(
            name.startswith(("shard_1.parquet", "shard_2.parquet"))
            for name in statted
        )
        raise AssertionError("included publication reached")

    monkeypatch.setattr(training_selection_module.Path, "resolve", observed_resolve)
    monkeypatch.setattr(training_selection_module.Path, "stat", observed_stat)
    monkeypatch.setattr(
        training_selection_module,
        "_validate_selection_publication",
        publication_reached,
    )
    with pytest.raises(AssertionError, match="included publication reached"):
        load_training_selection(manifest, include_splits=("train",))


def test_resolved_selection_paths_must_remain_under_resolved_data_root(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.parquet"
    outside.touch()
    (root / "escape.parquet").symlink_to(outside)
    payload = _metadata_only_selection(root, splits=("train",))
    payload["entries"][0]["path"] = "escape.parquet"
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source read occurred before containment validation")

    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="outside data_root"):
        load_training_selection(manifest)


def test_resolved_aliases_are_rejected_before_file_or_publication_reads(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    target = root / "target.parquet"
    target.touch()
    (root / "alias.parquet").symlink_to(target)
    payload = _metadata_only_selection(root, splits=("train", "validation"))
    payload["entries"][0]["path"] = "target.parquet"
    payload["entries"][1]["path"] = "alias.parquet"
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source read occurred before alias validation")

    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="duplicate resolved parquet paths"):
        load_training_selection(manifest)


def test_hardlink_aliases_fail_after_batch_stat_before_publication(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    first = root / "shard_0.parquet"
    second = root / "shard_1.parquet"
    first.write_bytes(b"same-inode")
    second.hardlink_to(first)
    manifest = write_hashed_manifest(
        _metadata_only_selection(root, splits=("train", "validation")),
        tmp_path / "selection.json",
    )
    original_stat = training_selection_module.Path.stat
    statted: set[Path] = set()

    def observed_stat(path, *args, **kwargs):
        if path in {first, second}:
            statted.add(path)
        return original_stat(path, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("publication digest reached before inode uniqueness")

    monkeypatch.setattr(training_selection_module.Path, "stat", observed_stat)
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    with pytest.raises(ValueError, match="hardlink"):
        load_training_selection(manifest)
    assert statted == {first, second}


def test_public_loaders_expose_no_payload_injection_escape_hatches():
    from hypertagging.data.dataset_index import (
        load_dataset_index,
        load_dataset_index_metadata,
    )
    from hypertagging.training.data_module import (
        build_real_data_module,
        preflight_dataset_index_data_binding,
    )

    forbidden = {
        "payload",
        "_payload",
        "metadata_payload",
        "_metadata_payload",
        "manifest_payload",
        "_manifest_payload",
        "index_payload",
        "_index_payload",
    }
    public_functions = (
        load_training_selection,
        load_dataset_index,
        load_dataset_index_metadata,
        build_real_data_module,
        preflight_dataset_index_data_binding,
    )
    for function in public_functions:
        assert not (set(inspect.signature(function).parameters) & forbidden)
    for function in (
        load_training_selection,
        load_dataset_index,
        load_dataset_index_metadata,
    ):
        with pytest.raises(ValueError, match="path"):
            function({})


def test_manifest_shaped_json_never_falls_back_to_generic_resolution(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    payload = _metadata_only_selection(tmp_path)
    payload["manifest_version"] = "wrong-selection-version"
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("generic or source resolution reached")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    with pytest.raises(ValueError, match="unsupported training selection"):
        data_module.resolve_data_paths(manifest)


@pytest.mark.parametrize(
    ("mutation", "rehash", "message"),
    (
        (
            lambda payload: payload.update(selection_name="tampered-without-rehash"),
            False,
            "manifest hash mismatch",
        ),
        (
            lambda payload: payload.update(manifest_version="wrong-selection-version"),
            True,
            "unsupported training selection",
        ),
        (
            lambda payload: payload.pop("selection_mode"),
            True,
            "top-level schema",
        ),
        (
            lambda payload: payload.update(unexpected_authenticated_field=True),
            True,
            "top-level schema",
        ),
    ),
)
def test_invalid_authenticated_looking_selection_never_reaches_generic_fallback(
    tmp_path, monkeypatch, mutation, rehash, message
):
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    manifest = write_hashed_manifest(
        _metadata_only_selection(tmp_path), tmp_path / "selection.json"
    )
    payload = json.loads(manifest.read_text())
    mutation(payload)
    if rehash:
        payload["manifest_hash"] = canonical_manifest_hash(payload)
    manifest.write_text(json.dumps(payload))
    original_read_text = training_selection_module.Path.read_text
    manifest_reads = 0

    def observed_read_text(path, *args, **kwargs):
        nonlocal manifest_reads
        if path == manifest:
            manifest_reads += 1
            if manifest_reads > 1:
                raise AssertionError("invalid selection reached generic JSON reread")
        return original_read_text(path, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("invalid selection reached a filesystem fallback effect")

    monkeypatch.setattr(training_selection_module.Path, "read_text", observed_read_text)
    monkeypatch.setattr(training_selection_module.Path, "is_dir", bomb)
    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(training_selection_module.Path, "exists", bomb)

    with pytest.raises(ValueError, match=message):
        data_module.resolve_data_paths(manifest)
    assert manifest_reads == 1


def test_non_selection_generic_json_fallback_remains_supported(tmp_path):
    import hypertagging.training.data_module as data_module

    shard = tmp_path / "events.parquet"
    shard.touch()
    manifest = tmp_path / "generic.json"
    manifest.write_text(json.dumps({"shards": [{"path": shard.name}]}))

    assert data_module.resolve_data_paths(manifest) == [shard.resolve()]


@pytest.mark.parametrize(
    ("caller_update", "message"),
    (
        ({"max_events": 1}, "raw max_events prefixes"),
        ({"pilot_split_repair": True}, "cannot use pilot_split_repair"),
    ),
)
def test_manifest_only_caller_modes_reject_after_one_auth_before_full_load(
    tmp_path, monkeypatch, caller_update, message
):
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    manifest = write_hashed_manifest(
        _metadata_only_selection(tmp_path, splits=("train",)),
        tmp_path / "selection.json",
    )
    original_read_text = training_selection_module.Path.read_text
    manifest_reads = 0

    def observed_read_text(path, *args, **kwargs):
        nonlocal manifest_reads
        if path == manifest:
            manifest_reads += 1
        return original_read_text(path, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("manifest caller mode reached full load or publication")

    monkeypatch.setattr(training_selection_module.Path, "read_text", observed_read_text)
    monkeypatch.setattr(training_selection_module, "_load_training_selection_bound", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    with pytest.raises(ValueError, match=message):
        data_module.build_real_data_module(manifest, **caller_update)
    assert manifest_reads == 1


def test_all_included_paths_resolve_before_first_file_or_publication_read(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    selected_names = {"shard_0.parquet", "shard_1.parquet"}
    for name in selected_names:
        (root / name).touch()
    payload = _metadata_only_selection(root, splits=("train", "validation"))
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")
    original_resolve = training_selection_module.Path.resolve
    resolved_selected: set[str] = set()

    def observed_resolve(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        if path.name in selected_names:
            resolved_selected.add(path.name)
        return resolved

    def first_file_read(*_args, **_kwargs):
        assert resolved_selected == selected_names
        raise AssertionError("first file read reached after complete resolve batch")

    monkeypatch.setattr(training_selection_module.Path, "resolve", observed_resolve)
    monkeypatch.setattr(training_selection_module.Path, "is_file", first_file_read)
    monkeypatch.setattr(
        training_selection_module,
        "_validate_selection_publication",
        first_file_read,
    )

    with pytest.raises(
        AssertionError, match="first file read reached after complete resolve batch"
    ):
        load_training_selection(manifest)


def test_missing_later_included_file_causes_zero_publication_calls(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    first = root / "shard_0.parquet"
    later = root / "shard_1.parquet"
    first.touch()
    manifest = write_hashed_manifest(
        _metadata_only_selection(root, splits=("train", "validation")),
        tmp_path / "selection.json",
    )
    original_is_file = training_selection_module.Path.is_file
    checked = []
    publications = []

    def observed_is_file(path):
        checked.append(path)
        return original_is_file(path)

    def observed_publication(*args, **kwargs):
        publications.append((args, kwargs))

    monkeypatch.setattr(training_selection_module.Path, "is_file", observed_is_file)
    monkeypatch.setattr(
        training_selection_module,
        "_validate_selection_publication",
        observed_publication,
    )

    with pytest.raises(FileNotFoundError, match="missing selected parquet shard"):
        load_training_selection(manifest)
    assert {first, later}.issubset(checked)
    assert publications == []


def test_all_shard_and_publication_file_checks_precede_first_digest(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    expected: set = set()
    for task_id in (0, 1):
        parquet = root / f"shard_{task_id}.parquet"
        sidecar = parquet.with_suffix(parquet.suffix + ".metadata.json")
        marker = parquet.with_suffix(parquet.suffix + ".complete")
        for path in (parquet, sidecar, marker):
            path.write_bytes(b"fixture")
            expected.add(path)
    manifest = write_hashed_manifest(
        _metadata_only_selection(root, splits=("train", "validation")),
        tmp_path / "selection.json",
    )
    original_is_file = training_selection_module.Path.is_file
    checked: set = set()

    def observed_is_file(path):
        checked.add(path)
        return original_is_file(path)

    def first_digest(*_args, **_kwargs):
        assert expected.issubset(checked)
        raise AssertionError("first digest reached after complete file-check batch")

    monkeypatch.setattr(training_selection_module.Path, "is_file", observed_is_file)
    monkeypatch.setattr(training_selection_module, "_sha256_file", first_digest)
    with pytest.raises(
        AssertionError, match="first digest reached after complete file-check batch"
    ):
        load_training_selection(manifest)


def test_selection_checks_every_publication_path_even_after_missing_sidecar(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    root = tmp_path / "data"
    root.mkdir()
    parquets = [root / f"shard_{position}.parquet" for position in range(2)]
    for parquet in parquets:
        parquet.touch()
    manifest = write_hashed_manifest(
        _metadata_only_selection(root, splits=("train", "validation")),
        tmp_path / "selection.json",
    )
    publication_paths = {
        path
        for parquet in parquets
        for path in (
            parquet.with_suffix(parquet.suffix + ".metadata.json"),
            parquet.with_suffix(parquet.suffix + ".complete"),
        )
    }
    missing = parquets[0].with_suffix(parquets[0].suffix + ".metadata.json")
    checked: set = set()

    def observed_is_file(path):
        checked.add(path)
        return path != missing

    monkeypatch.setattr(training_selection_module.Path, "is_file", observed_is_file)
    with pytest.raises(ValueError, match="publication is incomplete"):
        load_training_selection(manifest)
    assert publication_paths.issubset(checked)


def test_mutated_private_manifest_binding_is_rejected_before_source_effects(
    tmp_path, monkeypatch
):
    import hypertagging.data.training_selection as training_selection_module

    manifest = write_hashed_manifest(
        _metadata_only_selection(tmp_path), tmp_path / "selection.json"
    )
    binding = training_selection_module._load_selection_manifest_binding(manifest)
    binding.payload["selection_name"] = "mutated-after-authentication"

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached with mutated provenance token")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    with pytest.raises(ValueError, match="provenance binding was mutated"):
        training_selection_module._load_training_selection_bound(binding)


@pytest.mark.parametrize(
    "requested",
    (
        (),
        ("train", "train"),
        ("validation", "train"),
        ("invalid",),
        (False,),
        (1,),
        ([],),
        ({},),
        False,
        1,
        {"train": True},
    ),
)
def test_requested_selection_splits_fail_closed_before_source_effects(
    tmp_path, monkeypatch, requested
):
    import hypertagging.data.training_selection as training_selection_module

    manifest = write_hashed_manifest(
        _metadata_only_selection(tmp_path), tmp_path / "selection.json"
    )

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before requested-split binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError):
        load_training_selection(manifest, include_splits=requested)


def test_required_and_included_splits_require_positive_manifest_roles(tmp_path):
    payload = _metadata_only_selection(tmp_path, splits=("train",))

    with pytest.raises(ValueError, match="no positive"):
        validate_training_selection_metadata(
            payload, include_splits=("train", "validation")
        )
    with pytest.raises(ValueError, match="not included"):
        validate_training_selection_metadata(
            payload,
            include_splits=("train",),
            required_splits=("train", "validation"),
        )


@pytest.mark.parametrize("splits", (("train", "validation", "test"), ("train",)))
@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_training_selection_top_level_schema_is_exact_before_source_effects(
    tmp_path, monkeypatch, splits, mutation
):
    import hypertagging.data.training_selection as training_selection_module

    payload = _metadata_only_selection(tmp_path, splits=splits)
    if mutation == "extra":
        payload["authenticated_but_unsupported"] = {"value": True}
    else:
        payload.pop("category_split_shard_counts")
    manifest = write_hashed_manifest(payload, tmp_path / "selection.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before exact manifest schema binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)
    monkeypatch.setattr(training_selection_module.pq, "ParquetFile", bomb)

    with pytest.raises(ValueError, match="top-level schema"):
        load_training_selection(manifest)


@pytest.mark.parametrize(
    "field",
    (
        "inventory_entry_hash",
        "task_record_hash",
        "parquet_sha256_reference",
        "sidecar_sha256",
        "completion_marker_sha256",
    ),
)
def test_every_selection_entry_digest_requires_lowercase_sha256(tmp_path, field):
    payload = _metadata_only_selection(tmp_path)
    payload["entries"][0][field] = "A" * 64
    payload["manifest_hash"] = canonical_manifest_hash(payload)

    with pytest.raises(ValueError, match=field):
        validate_training_selection_metadata(payload)


@pytest.mark.parametrize(
    "required",
    (
        (),
        ("train", "train"),
        ("validation", "train"),
        ("invalid",),
        (False,),
        ([],),
        False,
        {"train": True},
    ),
)
def test_full_loader_rejects_invalid_required_splits_before_source_effects(
    tmp_path, monkeypatch, required
):
    import hypertagging.data.training_selection as training_selection_module

    manifest = write_hashed_manifest(
        _metadata_only_selection(tmp_path), tmp_path / "selection.json"
    )

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before required-split binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError):
        load_training_selection(manifest, required_splits=required)


def test_scientific_mode_rejects_raw_data_before_scanning(tmp_path):
    raw = tmp_path / "raw.parquet"
    raw.touch()
    from hypertagging.training.data_module import build_real_data_module

    with pytest.raises(ValueError, match="promoted full-record dataset index"):
        build_real_data_module(raw, scientific_mode=True)


def test_selection_handoff_fits_normalizers_on_training_role_only(
    tmp_path, monkeypatch
):
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
                "campaign_config_digest": "d" * 64,
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
    entries.sort(key=lambda entry: (entry["split"], entry["category"], entry["task_id"]))
    selection = {
        "manifest_version": SELECTION_MANIFEST_VERSION,
        "selection_name": "normalizer-handoff",
        "data_root": str(tmp_path),
        "inventory_hash": "1" * 64,
        "roles_hash": "2" * 64,
        "selection_seed": 20260812,
        "training_category_shard_quotas": {"fixture": 1},
        "selection_mode": "explicit_whole_shard_source_roles",
        "max_events_prefix_allowed": False,
        "normalizer_scope": "train_split_only",
        "uid_validation": {
            "status": "pending_full_index_build",
            "gate": "required_before_scientific_training",
        },
        "split_counts": {"test": 2, "train": 2, "validation": 2},
        "split_shard_counts": {"test": 1, "train": 1, "validation": 1},
        "category_split_shard_counts": {
            "fixture": {"test": 1, "train": 1, "validation": 1}
        },
        "source_split_isolation": "validated",
        "task_split_isolation": "validated",
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
    all_roles = load_training_selection(manifest)
    assert set(all_roles.source_expectations) == {
        "source_0.root",
        "source_1.root",
        "source_2.root",
    }

    indexed_module = build_real_data_module(
        manifest,
        scientific_mode=True,
        required_splits=("train", "validation"),
        dataset_index=index,
        seed=20260812,
    )
    assert indexed_module.split_counts == {"train": 2, "validation": 2, "test": 0}
    assert indexed_module.dataset_index["normalizer_scope"] == "train"
    assert indexed_module.dataset_index["source_groups"] == {
        "source_0.root": "train",
        "source_1.root": "validation",
    }
    assert "source_2.root" not in indexed_module.dataset_index["source_groups"]
    assert indexed_module.seed == 20260812
    assert indexed_module.split_config.__dict__ == json.loads(index.read_text())[
        "split_config"
    ]
    assert indexed_module.split_config.seed == 20260730

    with pytest.raises(ValueError, match="split configuration mismatch"):
        build_real_data_module(
            manifest,
            scientific_mode=True,
            required_splits=("train", "validation"),
            dataset_index=index,
            split_config=SourceAwareSplitConfig(seed=20260812),
        )

    from hypertagging.data.dataset_index import _index_hash

    original = json.loads(index.read_text())
    for name, mutate, message in (
        (
            "test-role",
            lambda payload: payload["selection_contract"].update(
                included_splits=["train", "validation", "test"]
            ),
            "shard paths do not match",
        ),
        (
            "legacy-index",
            lambda payload: payload.pop("event_identity_validation"),
            "top-level schema",
        ),
        (
            "sidecar-index",
            lambda payload: payload.update(event_identity_validation={}),
            "identity/task-binding gate",
        ),
        (
            "wrong-source-role",
            lambda payload: payload["source_groups"].update(
                {"source_0.root": "validation"}
            ),
            "source groups disagree",
        ),
        (
            "excluded-test-source",
            lambda payload: payload["source_groups"].update(
                {"source_2.root": "test"}
            ),
            "source groups disagree",
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
