import json
from pathlib import Path

import pytest

from scripts import mdst_batch_production as production


def test_manifest_records_are_exact_non_overlapping_ranges(monkeypatch, tmp_path):
    files = [
        tmp_path / "ccbar" / "mdst" / "sub00" / "first.root",
        tmp_path / "charged" / "mdst" / "sub00" / "second.root",
    ]
    counts = {files[0]: 12, files[1]: 9}
    monkeypatch.setattr(production, "root_event_count", lambda path: counts[path])

    records, categories = production.build_manifest_records(
        files,
        output_root=tmp_path / "output",
        target_events=17,
        events_per_task=5,
    )

    assert [record["entry_sequence"] for record in records] == ["0:4", "5:9", "10:11", "0:4"]
    assert [record["planned_events"] for record in records] == [5, 5, 2, 5]
    assert sum(record["planned_events"] for record in records) == 17
    assert categories == {"ccbar": 12, "charged": 5}
    assert len({record["output_file"] for record in records}) == len(records)
    assert {
        record["track_fit_policy"] for record in records
    } == {production.TRACK_FIT_POLICY_MAX_P_VALUE_V1}

    with pytest.raises(ValueError, match="unknown track_fit_policy"):
        production.build_manifest_records(
            files,
            output_root=tmp_path / "invalid",
            target_events=1,
            events_per_task=1,
            track_fit_policy="truth_best",
        )


def test_manifest_refuses_insufficient_input(monkeypatch, tmp_path):
    input_file = tmp_path / "ccbar" / "mdst" / "sub00" / "small.root"
    monkeypatch.setattr(production, "root_event_count", lambda _path: 3)

    with pytest.raises(RuntimeError, match="below target"):
        production.build_manifest_records(
            [input_file],
            output_root=tmp_path / "output",
            target_events=4,
            events_per_task=2,
        )


def test_write_and_read_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    output_root = tmp_path / "output"
    records = [
        {
            "task_id": 0,
            "input_file": "/input.root",
            "physics_category": "ccbar",
            "source_entries": 10,
            "entry_start": 0,
            "entry_stop_exclusive": 10,
            "entry_sequence": "0:9",
            "planned_events": 10,
            "output_file": str(output_root / "shards" / "mdst_00000.parquet"),
        }
    ]

    summary = production.write_manifest(
        records,
        manifest=manifest,
        input_root=Path("/input"),
        output_root=output_root,
        target_events=10,
        events_per_task=10,
        category_events={"ccbar": 10},
        overwrite=False,
    )

    assert production.read_manifest_record(manifest, 0) == records[0]
    assert summary["planned_events"] == 10
    assert json.loads(manifest.with_suffix(".summary.json").read_text())["tasks"] == 1
