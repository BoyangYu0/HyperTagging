import json

from scripts import mdst_batch_production as production


def test_global_uid_validation_is_folded_into_each_shard_pass(monkeypatch, tmp_path):
    calls = []
    records = []
    for task_id in range(2):
        output = tmp_path / f"shard-{task_id}.parquet"
        output.touch()
        records.append({
            "task_id": task_id,
            "input_file": f"input-{task_id}.root",
            "physics_category": "signal",
            "entry_start": 0,
            "entry_stop_exclusive": 1,
            "planned_events": 1,
            "output_file": str(output),
            "schema_version": "direct-mdst-tree-v4-event-row",
            "pid_vocabulary_version": "pid-vocabulary-v1",
            "feature_spec_hash": "feature",
            "charge_conjugate_normalization": False,
            "leaf_kinematics_mode": "raw_track_predicted_pid",
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    def fake_validate(path, *, uid_callback, **_kwargs):
        calls.append(path)
        uid_callback(f"uid:{path.name}")
        return {"events": 1}

    monkeypatch.setattr(production, "validate_shard", fake_validate)
    result = production.validate_production_manifest(manifest)
    assert len(calls) == 2
    assert result["validated_events"] == result["unique_event_uids"] == 2
    assert result["global_uid_validation_passes"] == 1

