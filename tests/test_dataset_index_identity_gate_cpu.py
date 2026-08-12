import json

import pytest

from hypertagging.data.dataset_index import build_dataset_index
from hypertagging.data.notebook_fixtures import notebook_fixture_trees
from hypertagging.preprocessing.schema_v4 import export_trees_v4
from hypertagging.training.model_config import MODEL_PRESETS


def _shard(tmp_path, number, uid):
    source = f"source-{number}.root"
    trees = notebook_fixture_trees()[:1]
    trees[0].metadata.update(
        {"source_file": source, "source_category": "fixture", "event_uid": uid}
    )
    task_hash = f"{number + 10:064x}"
    path = export_trees_v4(
        trees,
        tmp_path / f"shard-{number}.parquet",
        metadata={
            "source_file": source,
            "category": "fixture",
            "task_id": number,
            "task_record_hash": task_hash,
        },
    )
    return path, source, {
        "category": "fixture",
        "task_id": number,
        "task_record_hash": task_hash,
        "path": str(path.resolve()),
    }


def test_index_records_global_uid_source_task_gate(tmp_path):
    path, source, expectation = _shard(tmp_path, 1, "unique-event")
    output = build_dataset_index(
        [path],
        tmp_path / "index.json",
        source_split_overrides={source: "train"},
        selection_manifest_hash="a" * 64,
        selection_included_splits=("train", "validation"),
        source_expectations={source: expectation},
    )
    gate = json.loads(output.read_text())["event_identity_validation"]
    assert gate["status"] == "passed"
    assert gate["validated_events"] == gate["unique_event_uids"] == 1
    assert gate["sealed_test_opened"] is False


def test_index_rejects_duplicate_uid_across_shards(tmp_path):
    first = _shard(tmp_path, 1, "duplicate-event")
    second = _shard(tmp_path, 2, "duplicate-event")
    overrides = {first[1]: "train", second[1]: "validation"}
    expectations = {first[1]: first[2], second[1]: second[2]}
    with pytest.raises(ValueError, match="duplicate event_uid"):
        build_dataset_index(
            [first[0], second[0]],
            tmp_path / "index.json",
            source_split_overrides=overrides,
            selection_manifest_hash="a" * 64,
            selection_included_splits=("train", "validation"),
            source_expectations=expectations,
        )


def test_small_candidate_preserves_evidence_based_capacity_contract():
    architecture = MODEL_PRESETS["small_candidate"]
    assert architecture.n_queries == 32
    assert architecture.max_cardinality == 16
    assert architecture.capacity_report_required is True


def test_scientific_identity_gate_requires_selection_expectations(tmp_path):
    path, source, _expectation = _shard(tmp_path, 3, "unique-scientific-event")
    with pytest.raises(ValueError, match="requires selection source expectations"):
        build_dataset_index(
            [path],
            tmp_path / "index.json",
            source_split_overrides={source: "train"},
            selection_manifest_hash="b" * 64,
            require_event_identity_validation=True,
        )
