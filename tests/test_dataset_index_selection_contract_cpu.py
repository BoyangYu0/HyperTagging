import json
import hashlib

import pytest

from hypertagging.data.dataset_index import build_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.training.data_module import build_real_data_module


def _write_single_shard_selection(tmp_path, shard):
    from hypertagging.data.training_selection import (
        SELECTION_MANIFEST_VERSION,
        write_hashed_manifest,
    )

    sidecar = shard.with_suffix(shard.suffix + ".metadata.json")
    marker = shard.with_suffix(shard.suffix + ".complete")
    completion = json.loads(marker.read_text())
    return write_hashed_manifest(
        {
            "manifest_version": SELECTION_MANIFEST_VERSION,
            "data_root": str(tmp_path),
            "selection_name": "single-shard-fixture",
            "inventory_hash": "1" * 64,
            "roles_hash": "2" * 64,
            "selection_seed": 20260812,
            "training_category_shard_quotas": {"fixture": 1},
            "selection_mode": "explicit_whole_shard_source_roles",
            "selection_includes_test": False,
            "excluded_roles": ["stress", "test"],
            "max_events_prefix_allowed": False,
            "normalizer_scope": "train_split_only",
            "uid_validation": {
                "status": "pending_full_index_build",
                "gate": "required_before_scientific_training",
            },
            "source_split_isolation": "validated",
            "task_split_isolation": "validated",
            "split_counts": {"train": 2, "validation": 0, "test": 0},
            "split_shard_counts": {"train": 1, "validation": 0, "test": 0},
            "category_split_shard_counts": {"fixture": {"train": 1}},
            "entries": [
                {
                    "inventory_entry_hash": "3" * 64,
                    "path": shard.name,
                    "schema_version": "direct-mdst-tree-v4",
                    "campaign_id": "fixture",
                    "campaign_config_digest": "d" * 64,
                    "source_git_commit": "a" * 40,
                    "source_git_tree": "b" * 40,
                    "task_id": 0,
                    "task_record_hash": "4" * 64,
                    "source_file": "fixture.root",
                    "category": "fixture",
                    "event_count": 2,
                    "parquet_sha256_reference": completion["parquet_sha256"],
                    "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                    "completion_marker_sha256": hashlib.sha256(
                        marker.read_bytes()
                    ).hexdigest(),
                    "split": "train",
                }
            ],
        },
        tmp_path / "selection.json",
    )


def _source_role_index_fixture(tmp_path):
    from hypertagging.data.training_selection import load_training_selection

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    return shard, manifest, index_path


def _install_index_metadata_effect_bombs(monkeypatch):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before index metadata binding")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_selection_fingerprint", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(
        training_selection_module, "_load_selection_manifest_binding", bomb
    )
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )


def test_false_test_source_role_binding_may_omit_excluded_test_count(tmp_path):
    from hypertagging.data.dataset_index import load_dataset_index_metadata
    from hypertagging.data.training_selection import (
        canonical_manifest_hash,
        load_training_selection,
    )

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    payload = json.loads(manifest.read_text())
    payload["split_counts"].pop("test")
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    manifest.write_text(json.dumps(payload))
    selection = load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    index = load_dataset_index_metadata(index_path)
    assert "test" not in index["split_counts"]
    module = build_real_data_module(manifest, dataset_index=index_path)
    assert module.split_counts == {"train": 2, "validation": 0, "test": 0}


def test_sidecar_builder_emits_validator_required_track_fit_policy(tmp_path):
    from hypertagging.data.dataset_index import (
        build_dataset_index_from_sidecars,
        load_dataset_index_metadata,
    )
    from hypertagging.preprocessing.schema_v4 import (
        ParquetEventWriter,
        iter_event_records_v4,
    )

    source = write_notebook_fixture_v4(tmp_path / "source.parquet")
    production = tmp_path / "production.parquet"
    with ParquetEventWriter(
        production,
        event_buffer_size=1,
        metadata={"source_file": "input.root", "category": "fixture"},
    ) as writer:
        for event in iter_event_records_v4(source):
            writer.write_event(event)
    sidecar = json.loads(
        production.with_suffix(".parquet.metadata.json").read_text()
    )
    index_path = build_dataset_index_from_sidecars(
        [production],
        tmp_path / "sidecar-index.json",
        split_config=SourceAwareSplitConfig(
            train_fraction=1.0,
            validation_fraction=0.0,
            test_fraction=0.0,
        ),
    )
    index = load_dataset_index_metadata(index_path)

    expected_policy = str(sidecar.get("track_fit_policy", ""))
    assert "track_fit_policy" in index["shards"][0]
    assert index["shards"][0]["track_fit_policy"] == expected_policy
    assert index["track_fit_policies"] == [expected_policy]


def test_self_built_legacy_index_retains_derived_schema_without_sidecars(tmp_path):
    from hypertagging.data.dataset_index import load_dataset_index_metadata
    from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3

    shard = write_notebook_fixture_v3(tmp_path / "legacy.parquet")
    index_path = build_dataset_index([shard], tmp_path / "legacy-index.json")
    index = load_dataset_index_metadata(index_path)

    assert index["schema_versions"] == ["direct-mdst-tree-v3"]
    assert index["shards"][0]["schema"] == "direct-mdst-tree-v3"
    assert index["shards"][0]["sidecar_hash"] == ""
    assert index["shards"][0]["completion_marker_hash"] == ""


def test_builder_rejects_unloadable_v4_index_before_publication(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "unpublished-v4.parquet")
    shard.with_suffix(shard.suffix + ".metadata.json").unlink()
    shard.with_suffix(shard.suffix + ".complete").unlink()
    output = tmp_path / "must-not-exist.index.json"

    with pytest.raises(ValueError, match="v4 shard sidecar_hash"):
        build_dataset_index([shard], output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("selection_update", "message"),
    (
        ({"mode": "unknown"}, "selection mode"),
        ({"fingerprint": "not-a-sha256"}, "selection fingerprint"),
        (
            {
                "mode": "source_role_manifest",
                "selection_manifest_hash": "bad",
            },
            "source-role hash contract",
        ),
    ),
)
def test_dataset_index_metadata_rejects_invalid_selection_schema(
    tmp_path, selection_update, message
):
    from hypertagging.data.dataset_index import (
        _index_hash,
        load_dataset_index_metadata,
    )

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].update(selection_update)
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_dataset_index_metadata(index_path)


def test_dataset_index_metadata_requires_json_object(tmp_path):
    from hypertagging.data.dataset_index import load_dataset_index_metadata

    index_path = tmp_path / "index.json"
    index_path.write_text("[]")

    with pytest.raises(ValueError, match="JSON object"):
        load_dataset_index_metadata(index_path)


@pytest.mark.parametrize(
    "included_splits",
    (
        [],
        ["train", "train"],
        ["validation", "train"],
        ["invalid"],
        [False],
        [1],
        [[]],
        [{}],
        False,
        1,
        {"train": True},
    ),
)
def test_source_role_index_included_splits_fail_closed_metadata_only(
    tmp_path, monkeypatch, included_splits
):
    from hypertagging.data.dataset_index import _index_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].update(
        mode="source_role_manifest",
        max_events=None,
        selection_manifest_hash="a" * 64,
        included_splits=included_splits,
    )
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before index metadata binding")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_selection_fingerprint", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)

    with pytest.raises(ValueError):
        data_module.preflight_dataset_index_data_binding(shard, index_path)


@pytest.mark.parametrize("mode", ("all", "ordered_prefix"))
def test_raw_index_modes_require_exact_empty_included_splits(tmp_path, mode):
    from hypertagging.data.dataset_index import _index_hash, load_dataset_index_metadata

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].update(
        mode=mode,
        max_events=1 if mode == "ordered_prefix" else None,
        included_splits=["train"],
    )
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="raw-selection hash contract"):
        load_dataset_index_metadata(index_path)


@pytest.mark.parametrize("field", ("fingerprint", "selection_manifest_hash"))
def test_index_selection_digests_require_exact_lowercase_sha256(tmp_path, field):
    from hypertagging.data.dataset_index import _index_hash, load_dataset_index_metadata

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].update(
        mode="source_role_manifest",
        max_events=None,
        selection_manifest_hash="a" * 64,
        included_splits=["train"],
    )
    payload["selection_contract"][field] = "A" * 64
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_dataset_index_metadata(index_path)


def test_wrong_but_shaped_selection_fingerprint_fails_before_manifest_or_sources(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["selection_contract"]["fingerprint"] = "0" * 64
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match="selection fingerprint mismatch"):
        build_real_data_module(manifest, dataset_index=index_path)


@pytest.mark.parametrize(
    "missing_key",
    (
        "mode",
        "max_events",
        "selection_manifest_hash",
        "included_splits",
        "fingerprint",
    ),
)
def test_selection_contract_requires_every_key_before_manifest_or_sources(
    tmp_path, monkeypatch, missing_key
):
    from hypertagging.data.dataset_index import _index_hash

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].pop(missing_key)
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match="selection contract keys"):
        build_real_data_module(manifest, dataset_index=index_path)


@pytest.mark.parametrize(
    "bad_path",
    (
        "",
        "relative/events.parquet",
        "/tmp/./events.parquet",
        "/tmp/../events.parquet",
        "//tmp/events.parquet",
        "/tmp/events\\bad.parquet",
        "/tmp/events\x00bad.parquet",
        "/tmp/events\x7fbad.parquet",
        1,
        [],
        {},
    ),
)
def test_index_paths_fail_lexically_before_manifest_or_sources(
    tmp_path, monkeypatch, bad_path
):
    from hypertagging.data.dataset_index import _index_hash

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["paths"][0] = bad_path
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match="dataset index path"):
        build_real_data_module(manifest, dataset_index=index_path)


def test_duplicate_index_paths_fail_before_manifest_or_sources(tmp_path, monkeypatch):
    from hypertagging.data.dataset_index import _index_hash

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["paths"].append(payload["paths"][0])
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match="duplicate canonical paths"):
        build_real_data_module(manifest, dataset_index=index_path)


def test_reordered_index_paths_fail_fingerprint_before_any_source_effect(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash, load_dataset_index_metadata
    from hypertagging.data.notebook_fixtures import notebook_fixture_trees
    from hypertagging.preprocessing.schema_v4 import export_trees_v4

    shards = []
    for position in range(2):
        trees = notebook_fixture_trees()
        for tree in trees:
            tree.metadata["event_uid"] = f"reorder:{position}:{tree.event_id}:0"
            tree.metadata["source_file"] = f"reorder_{position}.root"
        shards.append(
            export_trees_v4(trees, tmp_path / f"events_{position}.parquet")
        )
    index_path = build_dataset_index(shards, tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    payload["paths"].reverse()
    payload["shards"].reverse()
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match="selection fingerprint mismatch"):
        load_dataset_index_metadata(index_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.update(pid_vocabulary_version="wrong"),
            "PID vocabulary",
        ),
        (
            lambda payload: payload.update(schema_versions="direct-mdst-tree-v4"),
            "unsupported schemas",
        ),
        (
            lambda payload: payload.update(schema_versions=["unknown-schema"]),
            "unsupported schemas",
        ),
        (
            lambda payload: payload.update(supported_schema_set=[]),
            "supported-schema contract",
        ),
        (
            lambda payload: payload.update(feature_spec_revision="wrong"),
            "feature-spec revision",
        ),
        (
            lambda payload: payload.update(feature_spec_hash="0" * 64),
            "feature-spec hash",
        ),
    ),
)
def test_static_index_contracts_fail_before_manifest_or_source_effects(
    tmp_path, monkeypatch, mutation, message
):
    from hypertagging.data.dataset_index import _index_hash

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    mutation(payload)
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match=message):
        build_real_data_module(manifest, dataset_index=index_path)


def test_source_role_bound_index_rejects_raw_parquet_before_source_reads(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash, _selection_fingerprint
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    selection_hash = "a" * 64
    payload["selection_contract"].update(
        mode="source_role_manifest",
        selection_manifest_hash=selection_hash,
        included_splits=["train", "validation"],
        fingerprint=_selection_fingerprint(
            [shard.resolve()],
            None,
            selection_manifest_hash=selection_hash,
        ),
    )
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source data was touched before binding preflight")

    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(dataset_index_module, "_selection_fingerprint", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)

    with pytest.raises(ValueError, match="exact immutable training-selection"):
        build_real_data_module(
            [shard],
            dataset_index=index_path,
        )


def test_source_role_bound_index_rejects_wrong_manifest_before_source_reads(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash, _selection_fingerprint
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module
    from hypertagging.data.training_selection import (
        SELECTION_MANIFEST_VERSION,
        write_hashed_manifest,
    )

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    selection_hash = "a" * 64
    payload["selection_contract"].update(
        mode="source_role_manifest",
        selection_manifest_hash=selection_hash,
        included_splits=["train", "validation"],
        fingerprint=_selection_fingerprint(
            [shard.resolve()],
            None,
            selection_manifest_hash=selection_hash,
        ),
    )
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    wrong_manifest = write_hashed_manifest(
        {
            "manifest_version": SELECTION_MANIFEST_VERSION,
            "data_root": str(tmp_path),
            "entries": [],
        },
        tmp_path / "wrong-selection.json",
    )

    def bomb(*_args, **_kwargs):
        raise AssertionError("source data was touched before binding preflight")

    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(dataset_index_module, "_selection_fingerprint", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module, "load_training_selection", bomb)
    monkeypatch.setattr(training_selection_module, "_validate_selection_publication", bomb)

    with pytest.raises(ValueError, match="training-selection hash mismatch"):
        build_real_data_module(
            wrong_manifest,
            dataset_index=index_path,
        )


def test_matching_hash_malformed_manifest_fails_binding_before_source_effects(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash, _selection_fingerprint
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module
    from hypertagging.data.training_selection import write_hashed_manifest

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload.pop("selection_mode")
    write_hashed_manifest(manifest_payload, manifest)
    manifest_hash = json.loads(manifest.read_text())["manifest_hash"]
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].update(
        mode="source_role_manifest",
        max_events=None,
        selection_manifest_hash=manifest_hash,
        included_splits=["train"],
        fingerprint=_selection_fingerprint(
            [shard.resolve()], None, selection_manifest_hash=manifest_hash
        ),
    )
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect occurred before exact manifest binding")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_selection_fingerprint", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(training_selection_module, "load_training_selection", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)

    with pytest.raises(ValueError, match="top-level schema"):
        build_real_data_module(manifest, dataset_index=index_path)


def test_exact_metadata_binding_reaches_full_index_verification(tmp_path, monkeypatch):
    from hypertagging.data.dataset_index import _index_hash, _selection_fingerprint
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    manifest_hash = json.loads(manifest.read_text())["manifest_hash"]
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    payload = json.loads(index_path.read_text())
    payload["selection_contract"].update(
        mode="source_role_manifest",
        max_events=None,
        selection_manifest_hash=manifest_hash,
        included_splits=["train"],
        fingerprint=_selection_fingerprint(
            [shard.resolve()], None, selection_manifest_hash=manifest_hash
        ),
    )
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    calls = {"loader": 0, "fingerprint": 0}
    original_loader = training_selection_module._load_training_selection_bound
    original_fingerprint = dataset_index_module._selection_fingerprint

    def observed_loader(*args, **kwargs):
        calls["loader"] += 1
        return original_loader(*args, **kwargs)

    def observed_fingerprint(*args, **kwargs):
        calls["fingerprint"] += 1
        return original_fingerprint(*args, **kwargs)

    def full_verification_reached(*_args, **_kwargs):
        assert calls == {"loader": 1, "fingerprint": 0}
        raise AssertionError("full indexed-shard verification reached")

    monkeypatch.setattr(
        training_selection_module, "_load_training_selection_bound", observed_loader
    )
    monkeypatch.setattr(
        dataset_index_module, "_selection_fingerprint", observed_fingerprint
    )
    monkeypatch.setattr(
        dataset_index_module, "_verify_indexed_shards", full_verification_reached
    )

    with pytest.raises(AssertionError, match="full indexed-shard verification reached"):
        build_real_data_module(manifest, dataset_index=index_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["source_groups"].update(
                {"fixture.root": "validation"}
            ),
            "source groups disagree",
        ),
        (
            lambda payload: (
                payload["shards"][0].update(source_digest="f" * 64),
                payload["shards"][0]["completion_marker_content"].update(
                    parquet_sha256="f" * 64
                ),
            ),
            "parquet digest disagrees",
        ),
        (
            lambda payload: payload["split_counts"].update(train=3),
            "split counts disagree",
        ),
    ),
)
def test_cross_document_mutations_fail_before_any_resolve_or_source_read(
    tmp_path, monkeypatch, mutation, message
):
    from hypertagging.data.dataset_index import _index_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    payload = json.loads(index_path.read_text())
    mutation(payload)
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before cross-document binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    with pytest.raises(ValueError, match=message):
        build_real_data_module(manifest, dataset_index=index_path)


def test_pure_manifest_absolute_paths_bind_before_symlink_resolution(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import (
        _index_hash,
        _selection_fingerprint_from_strings,
    )
    from hypertagging.data.training_selection import canonical_manifest_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    manifest_metadata = json.loads(manifest.read_text())
    manifest_metadata["data_root"] = "/different/canonical/root"
    manifest_metadata["manifest_hash"] = canonical_manifest_hash(manifest_metadata)
    manifest.write_text(json.dumps(manifest_metadata))
    index_metadata = json.loads(index_path.read_text())
    contract = index_metadata["selection_contract"]
    contract["selection_manifest_hash"] = manifest_metadata["manifest_hash"]
    contract["fingerprint"] = _selection_fingerprint_from_strings(
        index_metadata["paths"],
        mode=contract["mode"],
        max_events=contract["max_events"],
        selection_manifest_hash=contract["selection_manifest_hash"],
    )
    index_metadata["index_hash"] = _index_hash(index_metadata)
    index_path.write_text(json.dumps(index_metadata))

    def bomb(*_args, **_kwargs):
        raise AssertionError("path resolution preceded pure absolute-path binding")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match="shard paths do not match"):
        build_real_data_module(manifest, dataset_index=index_path)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("parquet_sha256_reference", "parquet digest disagrees"),
        ("sidecar_sha256", "sidecar digest disagrees"),
        ("completion_marker_sha256", "completion-marker digest disagrees"),
    ),
)
def test_self_hashed_manifest_index_publication_mismatches_are_pure(
    tmp_path, monkeypatch, field, message
):
    from hypertagging.data.dataset_index import (
        _index_hash,
        _selection_fingerprint_from_strings,
    )
    from hypertagging.data.training_selection import canonical_manifest_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["entries"][0][field] = "f" * 64
    manifest_payload["manifest_hash"] = canonical_manifest_hash(manifest_payload)
    manifest.write_text(json.dumps(manifest_payload))
    index_payload = json.loads(index_path.read_text())
    contract = index_payload["selection_contract"]
    contract["selection_manifest_hash"] = manifest_payload["manifest_hash"]
    contract["fingerprint"] = _selection_fingerprint_from_strings(
        index_payload["paths"],
        mode=contract["mode"],
        max_events=contract["max_events"],
        selection_manifest_hash=contract["selection_manifest_hash"],
    )
    index_payload["index_hash"] = _index_hash(index_payload)
    index_path.write_text(json.dumps(index_payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before publication cross-binding")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    with pytest.raises(ValueError, match=message):
        build_real_data_module(manifest, dataset_index=index_path)


def test_full_index_cannot_back_truncated_pilot(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    with pytest.raises(ValueError, match="max-events"):
        build_real_data_module(
            shard, dataset_index=index_path, max_events=1, pilot_split_repair=True
        )


@pytest.mark.parametrize(
    "configuration",
    (
        {"seed": 20260812},
        {"split_config": SourceAwareSplitConfig(seed=20260812)},
    ),
)
def test_raw_index_still_requires_exact_split_configuration(
    tmp_path, configuration
):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")

    with pytest.raises(ValueError, match="split configuration mismatch"):
        build_real_data_module(
            shard,
            dataset_index=index_path,
            **configuration,
        )


def test_caller_index_gates_precede_every_raw_source_effect(tmp_path, monkeypatch):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index(
        [shard], tmp_path / "index.json", target_policy="diagnostic_all"
    )

    def bomb(*_args, **_kwargs):
        raise AssertionError("raw source effect reached before caller/index gates")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)
    with pytest.raises(ValueError, match="target policy"):
        build_real_data_module(shard, dataset_index=index_path)


@pytest.mark.parametrize(
    ("index_update", "caller_update", "message"),
    (
        ({}, {"target_policy": "diagnostic_all"}, "target policy"),
        (
            {},
            {"split_config": SourceAwareSplitConfig(seed=20260812)},
            "split configuration mismatch",
        ),
        ({}, {"max_events": 1}, "max-events fingerprint mismatch"),
        ({}, {"required_splits": ("train",)}, "empty required split"),
        ({"legacy_fraction": 0.5}, {}, "legacy-conflated nodes"),
        ({}, {"allow_legacy_conflated": 1}, "must be boolean"),
        (
            {},
            {"scientific_mode": True, "required_splits": ("validation",)},
            "identity/task-binding gate",
        ),
    ),
)
def test_authenticated_preflight_binds_every_raw_caller_semantic_before_effects(
    tmp_path, monkeypatch, index_update, caller_update, message
):
    from hypertagging.data.dataset_index import _index_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    if index_update:
        payload = json.loads(index_path.read_text())
        payload.update(index_update)
        payload["index_hash"] = _index_hash(payload)
        index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before raw caller preflight")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)

    with pytest.raises(ValueError, match=message):
        build_real_data_module(shard, dataset_index=index_path, **caller_update)


def test_public_preflight_accepts_and_binds_caller_contract_inputs(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")

    def bomb(*_args, **_kwargs):
        raise AssertionError("path/source effect reached before public caller preflight")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)

    with pytest.raises(ValueError, match="target policy"):
        data_module.preflight_dataset_index_data_binding(
            shard,
            index_path,
            target_policy="diagnostic_all",
            max_events=None,
            split_config=None,
            seed=20260730,
            allow_legacy_conflated=False,
            scientific_mode=False,
        )


@pytest.mark.parametrize(
    ("caller_update", "message"),
    (
        ({"target_policy": "diagnostic_all"}, "target policy"),
        ({"max_events": 1}, "cannot use max_events"),
        (
            {"split_config": SourceAwareSplitConfig(seed=20260812)},
            "split configuration mismatch",
        ),
        ({"allow_legacy_conflated": 1}, "must be boolean"),
        ({"required_splits": ("validation",)}, "positive entries/counts/shards"),
        ({"pilot_split_repair": True}, "cannot use pilot_split_repair"),
    ),
)
def test_source_role_caller_semantics_precede_manifest_path_and_publication_effects(
    tmp_path, monkeypatch, caller_update, message
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)

    def bomb(*_args, **_kwargs):
        raise AssertionError("manifest/source effect reached before caller preflight")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match=message):
        build_real_data_module(manifest, dataset_index=index_path, **caller_update)


def test_indexed_pilot_repair_rejects_before_manifest_or_publication_effects(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)

    def bomb(*_args, **_kwargs):
        raise AssertionError("indexed pilot reached manifest or source effects")

    monkeypatch.setattr(
        training_selection_module, "_load_selection_manifest_binding", bomb
    )
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    with pytest.raises(ValueError, match="cannot use pilot_split_repair"):
        build_real_data_module(
            manifest,
            dataset_index=index_path,
            pilot_split_repair=True,
        )


@pytest.mark.parametrize(
    ("identity_update", "message"),
    (
        ({"status": "pending"}, "identity/task-binding gate"),
        ({"task_binding": "not_requested_legacy_index"}, "identity/task-binding gate"),
        ({"sealed_test_opened": True}, "identity/task-binding gate"),
    ),
)
def test_scientific_identity_task_and_sealed_gates_precede_manifest_effects(
    tmp_path, monkeypatch, identity_update, message
):
    from hypertagging.data.dataset_index import _index_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    _shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["event_identity_validation"].update(identity_update)
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("manifest/source effect reached before scientific gates")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )

    with pytest.raises(ValueError, match=message):
        build_real_data_module(
            manifest,
            dataset_index=index_path,
            scientific_mode=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.update(event_count=payload["event_count"] + 1),
            "event count disagrees",
        ),
        (
            lambda payload: payload.update(schema_versions=["direct-mdst-tree-v3"]),
            "schema_versions disagree",
        ),
        (
            lambda payload: payload.update(feature_spec_hashes=[]),
            "feature_spec_hashes disagree",
        ),
        (
            lambda payload: payload.update(track_fit_policies=["forged-policy"]),
            "track_fit_policies disagree",
        ),
        (
            lambda payload: payload["split_counts"].update(
                {
                    next(iter(payload["split_counts"])): next(
                        iter(payload["split_counts"].values())
                    )
                    + 1
                }
            ),
            "split counts disagree",
        ),
        (
            lambda payload: payload["category_counts"].update(
                {
                    next(iter(payload["category_counts"])): next(
                        iter(payload["category_counts"].values())
                    )
                    + 1
                }
            ),
            "category counts disagree",
        ),
        (
            lambda payload: payload.update(
                normalizer_scope=(
                    "all_events_no_train_split_diagnostic"
                    if payload["normalizer_scope"] == "train"
                    else "train"
                )
            ),
            "normalizer scope disagrees",
        ),
        (
            lambda payload: payload["normalizer_state"]["common"].update(count=[0.0]),
            "normalizer shapes",
        ),
        (
            lambda payload: payload["normalizer_state"].update(extra={}),
            "normalizer blocks",
        ),
        (
            lambda payload: payload["normalizer_state"]["track"].update(extra=[]),
            "normalizer blocks",
        ),
        (
            lambda payload: payload["normalizer_state"]["common"]["count"].__setitem__(
                0, -1.0
            ),
            "normalizer values",
        ),
        (
            lambda payload: payload["normalizer_state"]["track"]["mean"].__setitem__(
                0, float("nan")
            ),
            "normalizer values",
        ),
        (
            lambda payload: payload["normalizer_state"]["cluster"]["m2"].__setitem__(
                0, -1.0
            ),
            "normalizer values",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                validated_events=payload["event_count"] + 1
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                unique_event_uids=payload["event_count"] - 1
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                duplicate_event_uids=1
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                source_mismatches=1
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                category_mismatches=1
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                event_uid_stream_sha256="A" * 64
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                validation_scope="partial"
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                task_binding="forged"
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["event_identity_validation"].update(
                sealed_test_opened=True
            ),
            "event-identity metadata",
        ),
        (
            lambda payload: payload["shards"][0].update(source_entry_range=[0, 0]),
            "source-entry range",
        ),
        (
            lambda payload: payload["shards"][0].update(event_count=True),
            "event_count is invalid",
        ),
        (
            lambda payload: payload["shards"][0].update(unexpected={}),
            "descriptor 0 is invalid",
        ),
        (
            lambda payload: payload["shards"][0][
                "completion_marker_content"
            ].update(event_count=True),
            "completion marker metadata",
        ),
        (
            lambda payload: payload["shards"][0][
                "completion_marker_content"
            ].update(entry_stop_exclusive=0),
            "completion marker source range",
        ),
        (
            lambda payload: payload["shards"][0][
                "completion_marker_content"
            ].update(track_fit_policy="forged-policy"),
            "completion marker policy",
        ),
    ),
)
def test_index_aggregate_and_descriptor_invariants_are_pure_before_effects(
    tmp_path, monkeypatch, mutation, message
):
    from hypertagging.data.dataset_index import _index_hash, load_dataset_index_metadata
    import hypertagging.data.dataset_index as dataset_index_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    mutation(payload)
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before pure index invariants")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(dataset_index_module, "_selection_fingerprint", bomb)
    with pytest.raises(ValueError, match=message):
        load_dataset_index_metadata(index_path)


@pytest.mark.parametrize(
    ("allowed_types", "message"),
    (
        ([], "mapping"),
        ({"not-an-int": ["not-a-token"]}, "level key"),
        ({"01": [1]}, "level key"),
        ({"0": [1]}, "level key"),
        ({"33": [1]}, "level key"),
        ({"1": ["1"]}, "token vocabulary"),
        ({"1": [True]}, "token vocabulary"),
        ({"1": [-1]}, "token vocabulary"),
        ({"1": [len(PDG_TOKENS)]}, "token vocabulary"),
        ({"1": [1, 1]}, "ordered and unique"),
        ({"1": [2, 1]}, "ordered and unique"),
        ({"2": [1], "1": [1]}, "levels are not ordered"),
        ({"7": [1]}, "cross-fields"),
    ),
)
def test_allowed_types_are_purely_bound_before_manifest_or_source_effects(
    tmp_path, monkeypatch, allowed_types, message
):
    from hypertagging.data.dataset_index import _index_hash

    shard, manifest, index_path = _source_role_index_fixture(tmp_path)
    payload = json.loads(index_path.read_text())
    payload["allowed_types_by_level"] = allowed_types
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match=message):
        build_real_data_module(manifest, dataset_index=index_path)


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_dataset_index_top_level_schema_is_exact_before_all_effects(
    tmp_path, monkeypatch, mutation
):
    from hypertagging.data.dataset_index import _index_hash, load_dataset_index_metadata

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    payload = json.loads(index_path.read_text())
    if mutation == "extra":
        payload["authenticated_but_unsupported"] = {"value": True}
    else:
        payload.pop("category_counts")
    payload["index_hash"] = _index_hash(payload)
    index_path.write_text(json.dumps(payload))
    _install_index_metadata_effect_bombs(monkeypatch)

    with pytest.raises(ValueError, match="top-level schema"):
        load_dataset_index_metadata(index_path)


def test_index_source_file_and_stat_batches_precede_first_digest(tmp_path, monkeypatch):
    import hypertagging.data.dataset_index as dataset_index_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    sidecar = shard.with_suffix(shard.suffix + ".metadata.json")
    marker = shard.with_suffix(shard.suffix + ".complete")
    original_is_file = dataset_index_module.Path.is_file
    original_stat = dataset_index_module.Path.stat
    checked: set = set()
    statted: set = set()

    def observed_is_file(path):
        checked.add(path)
        return original_is_file(path)

    def observed_stat(path, *args, **kwargs):
        statted.add(path)
        return original_stat(path, *args, **kwargs)

    def first_digest(*_args, **_kwargs):
        assert {shard, sidecar, marker}.issubset(checked)
        assert shard in statted
        raise AssertionError("first digest reached after file/stat batches")

    monkeypatch.setattr(dataset_index_module.Path, "is_file", observed_is_file)
    monkeypatch.setattr(dataset_index_module.Path, "stat", observed_stat)
    monkeypatch.setattr(dataset_index_module, "_sha256_file", first_digest)
    with pytest.raises(
        AssertionError, match="first digest reached after file/stat batches"
    ):
        dataset_index_module.load_dataset_index(index_path)


def test_index_checks_every_publication_path_even_after_missing_sidecar(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    from hypertagging.data.notebook_fixtures import notebook_fixture_trees
    from hypertagging.preprocessing.schema_v4 import export_trees_v4

    shards = []
    for position in range(2):
        trees = notebook_fixture_trees()
        for tree in trees:
            tree.metadata["event_uid"] = (
                f"fixture:{position}:{tree.event_id}:0"
            )
            tree.metadata["source_file"] = f"fixture_{position}.root"
        shards.append(
            export_trees_v4(trees, tmp_path / f"events_{position}.parquet")
        )
    index_path = build_dataset_index(shards, tmp_path / "index.json")
    publication_paths = {
        path
        for shard in shards
        for path in (
            shard.with_suffix(shard.suffix + ".metadata.json"),
            shard.with_suffix(shard.suffix + ".complete"),
        )
    }
    missing = shards[0].with_suffix(shards[0].suffix + ".metadata.json")
    original_is_file = dataset_index_module.Path.is_file
    checked: set = set()

    def observed_is_file(path):
        checked.add(path)
        if path == missing:
            return False
        return original_is_file(path)

    monkeypatch.setattr(dataset_index_module.Path, "is_file", observed_is_file)
    with pytest.raises(ValueError, match="incomplete indexed v4 shard"):
        dataset_index_module.load_dataset_index(index_path)
    assert publication_paths.issubset(checked)


@pytest.mark.parametrize(
    ("configuration", "message"),
    (
        ({"required_splits": ("validation", "train")}, "order-preserving"),
        ({"max_events": True}, "positive integer"),
        ({"data_relative": True}, "canonical absolute parquet"),
    ),
)
def test_raw_caller_metadata_gates_fail_before_resolve_or_source_verification(
    tmp_path, monkeypatch, configuration, message
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    data = shard.name if configuration.pop("data_relative", False) else shard

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached before raw caller binding")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(data_module, "resolve_data_paths", bomb)
    with pytest.raises(ValueError, match=message):
        build_real_data_module(data, dataset_index=index_path, **configuration)


def test_raw_resolved_caller_equality_precedes_publication_and_source_verification(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    index_payload = json.loads(index_path.read_text())
    required_split = next(
        split for split, count in index_payload["split_counts"].items() if count > 0
    )
    original_resolve = dataset_index_module.Path.resolve
    resolve_calls = 0

    def drift_after_index_resolution(path, *args, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        if resolve_calls == 3:
            return tmp_path / "symlink-drift.parquet"
        return original_resolve(path, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect preceded resolved caller equality")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", drift_after_index_resolution)
    monkeypatch.setattr(data_module, "_require_complete_v4_publications", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)

    with pytest.raises(ValueError, match="resolved dataset index paths"):
        build_real_data_module(
            shard, dataset_index=index_path, required_splits=(required_split,)
        )
    assert resolve_calls == 3


def test_mutated_private_index_binding_is_rejected_before_source_effects(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    binding = dataset_index_module._load_dataset_index_binding(index_path)
    binding.payload["target_policy"] = "diagnostic_all"

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached with mutated provenance token")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    with pytest.raises(ValueError, match="provenance binding was mutated"):
        dataset_index_module._load_dataset_index_bound(binding)


def test_full_data_module_reads_each_bound_metadata_file_once(tmp_path, monkeypatch):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    original_read_text = dataset_index_module.Path.read_text
    original_selection_loader = (
        data_module._load_training_selection_from_dataset_data_binding
    )
    original_index_loader = data_module._load_dataset_index_from_dataset_data_binding
    reads = {manifest: 0, index_path: 0}
    private_binding_ids = []

    def observed_read_text(path, *args, **kwargs):
        if path in reads:
            reads[path] += 1
        return original_read_text(path, *args, **kwargs)

    def observed_selection_loader(binding, **kwargs):
        data_module._require_resolved_dataset_data_binding(binding)
        private_binding_ids.append(id(binding))
        return original_selection_loader(binding, **kwargs)

    def observed_index_loader(binding):
        data_module._require_resolved_dataset_data_binding(binding)
        private_binding_ids.append(id(binding))
        return original_index_loader(binding)

    monkeypatch.setattr(dataset_index_module.Path, "read_text", observed_read_text)
    monkeypatch.setattr(
        data_module,
        "_load_training_selection_from_dataset_data_binding",
        observed_selection_loader,
    )
    monkeypatch.setattr(
        data_module,
        "_load_dataset_index_from_dataset_data_binding",
        observed_index_loader,
    )
    module = build_real_data_module(manifest, dataset_index=index_path)
    assert module.dataset_index["selection_contract"]["selection_manifest_hash"] == selection.manifest_hash
    assert reads == {manifest: 1, index_path: 1}
    assert len(private_binding_ids) == 2
    assert len(set(private_binding_ids)) == 1


def test_post_preflight_index_and_manifest_swaps_cannot_change_pinned_data(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash
    from hypertagging.data.training_selection import canonical_manifest_hash
    import hypertagging.data.training_selection as training_selection_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    expected_index_hash = json.loads(index_path.read_text())["index_hash"]
    forged_manifest = json.loads(manifest.read_text())
    forged_manifest["selection_name"] = "forged-after-preflight"
    forged_manifest["manifest_hash"] = canonical_manifest_hash(forged_manifest)
    forged_index = json.loads(index_path.read_text())
    forged_index["target_policy"] = "diagnostic_all"
    forged_index["index_hash"] = _index_hash(forged_index)
    import hypertagging.training.data_module as data_module

    original_bound_loader = data_module._load_dataset_index_from_dataset_data_binding
    swaps = 0

    def swap_after_preflight(binding):
        nonlocal swaps
        swaps += 1
        manifest.write_text(json.dumps(forged_manifest) + "\n")
        index_path.write_text(json.dumps(forged_index) + "\n")
        return original_bound_loader(binding)

    monkeypatch.setattr(
        data_module,
        "_load_dataset_index_from_dataset_data_binding",
        swap_after_preflight,
    )
    module = build_real_data_module(manifest, dataset_index=index_path)
    assert swaps == 1
    assert module.selection_manifest_hash == selection.manifest_hash
    assert module.dataset_index["index_hash"] == expected_index_hash


def test_post_preflight_metadata_swaps_are_pinned_before_source_phase(
    tmp_path, monkeypatch
):
    from hypertagging.data.dataset_index import _index_hash
    from hypertagging.data.training_selection import canonical_manifest_hash
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    original_read_text = dataset_index_module.Path.read_text
    reads = {manifest: 0, index_path: 0}

    def observed_read_text(path, *args, **kwargs):
        if path in reads:
            reads[path] += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(dataset_index_module.Path, "read_text", observed_read_text)
    binding = data_module._preflight_dataset_index_data_binding(
        manifest, index_path, required_splits=("train",)
    )
    resolved_binding = data_module._resolve_dataset_index_data_binding(binding)
    forged_manifest = json.loads(manifest.read_text())
    forged_manifest["selection_name"] = "forged-after-preflight"
    forged_manifest["manifest_hash"] = canonical_manifest_hash(forged_manifest)
    forged_index = json.loads(index_path.read_text())
    forged_index["target_policy"] = "diagnostic_all"
    forged_index["index_hash"] = _index_hash(forged_index)
    baseline_reads = dict(reads)
    manifest.write_text(json.dumps(forged_manifest))
    index_path.write_text(json.dumps(forged_index))

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached after metadata preflight swap")

    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    with pytest.raises(ValueError, match="required selection splits are not included"):
        data_module._load_training_selection_from_dataset_data_binding(
            resolved_binding,
            include_splits=("train",),
            required_splits=("validation",),
        )
    assert reads == baseline_reads


def test_forged_mapping_cannot_replace_private_combined_binding(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module
    import hypertagging.training.data_module as data_module

    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    manifest = _write_single_shard_selection(tmp_path, shard)
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    index = dataset_index_module.load_dataset_index_metadata(index_path)
    manifest_payload = json.loads(manifest.read_text())

    def bomb(*_args, **_kwargs):
        raise AssertionError("forged mapping reached a path or source effect")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(training_selection_module.Path, "resolve", bomb)
    monkeypatch.setattr(training_selection_module.Path, "is_file", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    with pytest.raises(ValueError, match="provenance binding is invalid"):
        data_module._resolve_dataset_index_data_binding(
            {"index": index, "manifest": manifest_payload}
        )


def test_index_symlink_drift_fails_before_manifest_or_source_effects(
    tmp_path, monkeypatch
):
    import hypertagging.data.dataset_index as dataset_index_module
    import hypertagging.data.training_selection as training_selection_module

    published = tmp_path / "published"
    published.mkdir()
    shard = write_notebook_fixture_v4(published / "events.parquet")
    nested_manifest = _write_single_shard_selection(published, shard)
    manifest = nested_manifest.rename(tmp_path / "selection.json")
    selection = training_selection_module.load_training_selection(manifest)
    index_path = build_dataset_index(
        selection.paths,
        tmp_path / "index.json",
        source_split_overrides=selection.source_split_overrides,
        selection_manifest_hash=selection.manifest_hash,
        selection_included_splits=selection.included_splits,
    )
    stored_path = json.loads(index_path.read_text())["paths"][0]
    original = published.rename(tmp_path / "original-publication")
    alternate = tmp_path / "alternate-publication"
    alternate.mkdir()
    published.symlink_to(alternate, target_is_directory=True)
    assert original.is_dir()

    original_resolve = dataset_index_module.Path.resolve
    resolve_inputs = []

    def observed_resolve(path, *args, **kwargs):
        resolve_inputs.append(str(path))
        return original_resolve(path, *args, **kwargs)

    def bomb(*_args, **_kwargs):
        raise AssertionError("source effect reached after index symlink drift")

    monkeypatch.setattr(dataset_index_module.Path, "resolve", observed_resolve)
    monkeypatch.setattr(dataset_index_module.Path, "is_file", bomb)
    monkeypatch.setattr(dataset_index_module, "_verify_indexed_shards", bomb)
    monkeypatch.setattr(
        training_selection_module, "_validate_selection_publication", bomb
    )
    monkeypatch.setattr(training_selection_module, "_sha256_file", bomb)

    with pytest.raises(ValueError, match="resolved selection fingerprint"):
        build_real_data_module(manifest, dataset_index=index_path)
    assert resolve_inputs == [stored_path, stored_path]
