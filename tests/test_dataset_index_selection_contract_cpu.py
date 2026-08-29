import json

import pytest

from hypertagging.data.dataset_index import build_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.training.data_module import build_real_data_module


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
