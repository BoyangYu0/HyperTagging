import importlib.util
import json
from pathlib import Path

import pytest

from hypertagging.data.notebook_fixtures import (
    write_notebook_fixture,
    write_notebook_fixture_v1,
    write_notebook_fixture_v3,
)
from hypertagging.preprocessing.schema_v3 import SCHEMA_VERSION_V3, load_payload_v3


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mdst_batch_production",
    ROOT / "scripts" / "mdst_batch_production.py",
)
assert SPEC and SPEC.loader
production = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(production)


def test_v1_v2_load_and_v3_round_trip(tmp_path):
    for writer, name in (
        (write_notebook_fixture_v1, "v1"),
        (write_notebook_fixture, "v2"),
        (write_notebook_fixture_v3, "v3"),
    ):
        payload = load_payload_v3(writer(tmp_path / f"{name}.parquet"))
        assert payload["schema_version"] == SCHEMA_VERSION_V3
    node = load_payload_v3(tmp_path / "v3.parquet")["events"][0]["nodes"][0]
    for field in (
        "raw_pdg",
        "input_pid_token",
        "pid_target_token",
        "reco_charge",
        "truth_charge",
        "recursive_leaf_source_ids",
    ):
        assert field in node


def test_worker_command_and_manifest_scientific_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "root_event_count", lambda _path: 5)
    records, _ = production.build_manifest_records(
        [tmp_path / "input.root"],
        output_root=tmp_path,
        target_events=5,
        events_per_task=5,
    )
    record = records[0]
    assert record["schema_version"] == SCHEMA_VERSION_V3
    assert record["pid_vocabulary_version"]
    assert record["feature_spec_hash"]
    assert record["leaf_kinematics_mode"] == "raw_track_predicted_pid"


def test_global_validator_catches_duplicate_uids_and_schema_mismatch(tmp_path):
    shard = write_notebook_fixture_v3(tmp_path / "one.parquet")
    payload = load_payload_v3(shard)
    manifest = tmp_path / "manifest.jsonl"
    event_count = len(payload["events"])
    record = {
        "task_id": 0,
        "input_file": "a.root",
        "physics_category": "charged",
        "entry_start": 0,
        "entry_stop_exclusive": event_count,
        "planned_events": event_count,
        "output_file": str(shard),
        "schema_version": payload["schema_version"],
        "pid_vocabulary_version": payload["pid_vocabulary_version"],
        "feature_spec_hash": payload["feature_spec_hash"],
        "charge_conjugate_normalization": False,
        "leaf_kinematics_mode": "raw_track_predicted_pid",
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    summary = production.validate_production_manifest(manifest)
    assert summary["unique_event_uids"] == event_count
    bad = dict(record, schema_version="direct-mdst-tree-v2")
    manifest.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        production.validate_production_manifest(manifest)
