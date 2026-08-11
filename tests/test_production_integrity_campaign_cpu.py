import hashlib
import json
from pathlib import Path

import pytest

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.preprocessing.schema_v4 import ParquetEventWriter, iter_event_records_v4
from scripts import mdst_batch_production as production


def _record(tmp_path: Path, monkeypatch, *, leaf_mode="raw_track_predicted_pid"):
    source = tmp_path / "charged" / "mdst" / "sub00" / "input.root"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"stable-root-identity")
    monkeypatch.setattr(production, "root_event_count", lambda _path: 2)
    records, _ = production.build_manifest_records(
        [source],
        output_root=tmp_path / "output",
        target_events=2,
        events_per_task=2,
        source_git_commit="a" * 40,
        source_git_tree="b" * 40,
        source_state="clean",
        campaign_id="campaign-test",
        leaf_kinematics_mode=leaf_mode,
    )
    return records[0]


def _write_bound_shard(record, tmp_path, *, modes=None):
    seed = write_notebook_fixture_v4(tmp_path / "seed.parquet")
    events = list(iter_event_records_v4(seed))
    if modes is not None:
        for event in events:
            for node in event["nodes"]:
                if not node.get("daughter_ids") and node.get("node_kind") == "track":
                    node["leaf_kinematics_mode"] = modes[0]
            if len(modes) > 1:
                track = next(
                    node for node in event["nodes"]
                    if not node.get("daughter_ids") and node.get("node_kind") == "track"
                )
                track["leaf_kinematics_mode"] = modes[1]
    output = Path(record["output_file"])
    provenance = production._task_provenance(record)
    writer = ParquetEventWriter(output, metadata=provenance)
    writer.metadata["preprocessing_configuration"] = {
        "track_fit_policy": record["track_fit_policy"]
    }
    for event in events:
        writer.write_event(event)
    writer.close()
    return output


def _classify(record):
    return production.classify_shard(
        Path(record["output_file"]), **production._validation_kwargs(record)
    )["classification"]


def _rehash_marker(output: Path):
    marker = output.with_suffix(output.suffix + ".complete")
    payload = json.loads(marker.read_text())
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    payload["parquet_sha256"] = production._sha256_path(output)
    payload["sidecar_sha256"] = production._sha256_path(sidecar)
    marker.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_manifest_records_full_immutable_contract(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    required = set(production.PROVENANCE_FIELDS) | {
        "source_git_tree", "source_state", "task_record_hash",
        "input_file_sha256", "campaign_config_digest",
    }
    assert required <= set(record)
    assert record["input_file_size"] > 0
    assert record["input_file_mtime_ns"] > 0
    assert record["task_record_hash"] == production.task_record_hash(record)
    assert record["campaign_id"] in record["output_file"]


@pytest.mark.parametrize("window", ["parquet_only", "parquet_sidecar"])
def test_interrupted_publication_windows_are_incomplete(tmp_path, monkeypatch, window):
    record = _record(tmp_path, monkeypatch)
    output = _write_bound_shard(record, tmp_path)
    marker = output.with_suffix(output.suffix + ".complete")
    marker.unlink()
    if window == "parquet_only":
        output.with_suffix(output.suffix + ".metadata.json").unlink()
    assert _classify(record) == "INCOMPLETE_NO_MARKER"


def test_stale_and_corrupt_marker_hashes_are_rejected(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    output = _write_bound_shard(record, tmp_path)
    marker = output.with_suffix(output.suffix + ".complete")
    original = json.loads(marker.read_text())
    for key in ("parquet_sha256", "sidecar_sha256"):
        payload = dict(original)
        payload[key] = "0" * 64
        marker.write_text(json.dumps(payload) + "\n")
        assert _classify(record) == "CORRUPT_HASH"
    marker.write_text(json.dumps(original) + "\n")
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    sidecar.write_text(sidecar.read_text() + " ")
    assert _classify(record) == "CORRUPT_HASH"


def test_sidecar_and_provenance_from_another_task_are_rejected(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    output = _write_bound_shard(record, tmp_path)
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    payload = json.loads(sidecar.read_text())
    payload["task_id"] = 99
    sidecar.write_text(json.dumps(payload, sort_keys=True) + "\n")
    _rehash_marker(output)
    assert _classify(record) in {"METADATA_MISMATCH", "PROVENANCE_MISMATCH"}


def test_valid_completed_shard_and_manifest_binding(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    _write_bound_shard(record, tmp_path)
    result = production.validate_shard(
        Path(record["output_file"]), **production._validation_kwargs(record)
    )
    assert result["classification"] == "COMPLETE_VALID"
    assert result["campaign_id"] == record["campaign_id"]
    assert result["task_record_hash"] == record["task_record_hash"]

    changed = dict(record, campaign_id="other")
    changed["task_record_hash"] = production.task_record_hash(changed)
    assert production.classify_shard(
        Path(record["output_file"]), **production._validation_kwargs(changed)
    )["classification"] == "PROVENANCE_MISMATCH"
    for field, value in (
        ("input_file", "different.root"),
        ("entry_start", 1),
        ("entry_stop_exclusive", 1),
    ):
        mismatched = dict(record, **{field: value})
        mismatched["task_record_hash"] = production.task_record_hash(mismatched)
        assert production.classify_shard(
            Path(record["output_file"]), **production._validation_kwargs(mismatched)
        )["classification"] == "PROVENANCE_MISMATCH"

    manifest = tmp_path / "valid-manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    global_result = production.validate_production_manifest(manifest)
    assert global_result["all_completion_markers_valid"] is True
    assert global_result["validated_events"] == 2
    assert global_result["campaign_id"] == record["campaign_id"]


def test_leaf_mode_and_track_fit_validator_branches(tmp_path, monkeypatch):
    fixed = _record(tmp_path, monkeypatch, leaf_mode="fixed_hypothesis_candidate")
    output = _write_bound_shard(fixed, tmp_path, modes=("fixed_hypothesis_candidate",))
    assert _classify(fixed) == "COMPLETE_VALID"

    raw_mixed = dict(fixed)
    raw_mixed["output_file"] = str(tmp_path / "mixed.parquet")
    raw_mixed["task_record_hash"] = production.task_record_hash(raw_mixed)
    _write_bound_shard(
        raw_mixed, tmp_path, modes=("fixed_hypothesis_candidate", "raw_track_predicted_pid")
    )
    assert _classify(raw_mixed) == "METADATA_MISMATCH"

    missing_fixed = dict(fixed)
    missing_fixed["output_file"] = str(tmp_path / "missing-fixed.parquet")
    missing_fixed["task_record_hash"] = production.task_record_hash(missing_fixed)
    _write_bound_shard(missing_fixed, tmp_path, modes=("truth_topology_only",))
    assert _classify(missing_fixed) == "METADATA_MISMATCH"

    wrong_policy = dict(fixed, track_fit_policy="canonical_pion_closest_mass-v1")
    wrong_policy["task_record_hash"] = production.task_record_hash(wrong_policy)
    assert production.classify_shard(
        output, **production._validation_kwargs(wrong_policy)
    )["classification"] == "METADATA_MISMATCH"


def test_incomplete_shard_is_quarantined_before_retry(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    output = Path(record["output_file"])
    output.parent.mkdir(parents=True)
    output.write_bytes(b"interrupted")

    monkeypatch.setattr(production, "verify_worker_source", lambda *_args: {})
    monkeypatch.setattr(production.shutil, "which", lambda _name: "/fake/basf2")
    dependency_root = tmp_path / "basf2-site"
    dependency_root.mkdir()
    monkeypatch.setattr(production, "DEFAULT_BASF2_PYTHON_SITE", dependency_root)

    seed = write_notebook_fixture_v4(tmp_path / "retry-seed.parquet")
    events = list(iter_event_records_v4(seed))

    def fake_run(command, **_kwargs):
        temporary = Path(command[command.index("--output") + 1])
        provenance_path = Path(
            command[command.index("--production-provenance-json") + 1]
        )
        writer = ParquetEventWriter(
            temporary, metadata=json.loads(provenance_path.read_text())
        )
        writer.metadata["preprocessing_configuration"] = {
            "track_fit_policy": record["track_fit_policy"]
        }
        for event in events:
            writer.write_event(event)
        writer.close()

    monkeypatch.setattr(production.subprocess, "run", fake_run)
    result = production.run_task(
        manifest=manifest, task_id=0, repo_root=tmp_path, overwrite=False
    )
    assert result["classification"] == "COMPLETE_VALID"
    quarantined = list((output.parent.parent / "quarantine" / "task-00000").glob("*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / output.name).read_bytes() == b"interrupted"
    worker_result = json.loads(
        output.with_suffix(output.suffix + ".result.json").read_text()
    )
    assert worker_result["campaign_id"] == record["campaign_id"]
    assert worker_result["task_record_hash"] == record["task_record_hash"]


def test_valid_output_refuses_implicit_destructive_overwrite(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    _write_bound_shard(record, tmp_path)
    monkeypatch.setattr(production, "verify_worker_source", lambda *_args: {})
    monkeypatch.setattr(
        production.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("complete shard must not rerun"),
    )
    result = production.run_task(
        manifest=manifest, task_id=0, repo_root=tmp_path, overwrite=False
    )
    assert result["status"] == "already-complete"


def test_tampered_task_hash_is_refused(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    record["entry_stop_exclusive"] = 1
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(production.ShardValidationError, match="task_record_hash"):
        production.read_manifest_record(manifest, 0)


def test_status_missing_and_targeted_resubmit_are_non_submitting(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    manifest = tmp_path / "missing-manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    status = production.production_status(manifest)
    assert status["classifications"] == {"MISSING": 1}
    assert production.list_missing_tasks(manifest) == [0]
    monkeypatch.setattr(
        production.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("render must not submit"),
    )
    rendered = production.render_resubmit(manifest, repo_root=tmp_path)
    assert "queue task_id in (0)" in rendered
    environment_line = next(
        line for line in rendered.splitlines() if line.startswith("environment = ")
    )
    assert ";" not in environment_line
    assert f"MANIFEST={manifest.resolve()}" in environment_line
    assert f"REPO_ROOT={tmp_path.resolve()}" in environment_line
    assert f"OUTPUT_ROOT={manifest.resolve().parent.parent}" in environment_line


def test_source_preflight_failure_writes_structured_failure(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    manifest = tmp_path / "source-failure.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        production,
        "verify_worker_source",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("dirty checkout")),
    )
    with pytest.raises(RuntimeError, match="dirty checkout"):
        production.run_task(manifest=manifest, task_id=0, repo_root=tmp_path)
    failure = json.loads(
        Path(record["output_file"])
        .with_suffix(".parquet.failure.json")
        .read_text()
    )
    assert failure["schema_version"] == production.FAILURE_SCHEMA_VERSION
    assert failure["task_record_hash"] == record["task_record_hash"]
    assert failure["exception_type"] == "RuntimeError"


def test_changed_scientific_configuration_gets_new_campaign_namespace(tmp_path, monkeypatch):
    first = _record(tmp_path / "first", monkeypatch)
    second_root = tmp_path / "second"
    second = _record(second_root, monkeypatch)
    # Rebuild the second record with a different scientific policy and no
    # operator-forced campaign ID so the deterministic namespace must change.
    source = Path(second["input_file"])
    records, _ = production.build_manifest_records(
        [source],
        output_root=second_root / "output",
        target_events=2,
        events_per_task=2,
        source_git_commit="a" * 40,
        source_git_tree="b" * 40,
        source_state="clean",
        track_fit_policy="canonical_pion_closest_mass-v1",
    )
    changed = records[0]
    assert changed["campaign_config_digest"] != first["campaign_config_digest"]
    assert changed["campaign_id"] != first["campaign_id"]
    assert changed["campaign_id"] in changed["output_file"]
