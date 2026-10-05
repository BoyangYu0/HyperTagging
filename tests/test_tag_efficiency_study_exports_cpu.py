from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path
import sys

import pytest


def _tagging():
    return {
        "version": "tag-efficiency-study-v1",
        "replaces_preregistered_metrics": False,
        "summaries": {
            "full/greedy": {
                "channels": [
                    {
                        "channel": "B0 -> (D- -> K+ pi- pi-) pi+",
                        "correct": 1,
                        "evaluated": 2,
                        "unavailable": 0,
                    }
                ],
                "continuum_type": "ccbar",
            }
        },
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "run_full_reconstruction_evaluation_suite",
        "run_reconstruction_phase69_full_decay",
        "run_reconstruction_phase70_full_decay",
    ],
)
def test_repeat_payload_preserves_tagging_and_accepts_historical_absence(module_name):
    module = importlib.import_module(f"scripts.{module_name}")
    historical = {
        "retained_tree_checks": {},
        "summaries": {},
        "summaries_by_source_category": {},
        "summaries_by_target_shape": {},
        "events": [],
        "beam_search": {},
    }
    assert "tag_efficiency" not in module._decision_payload(historical)
    report = {**historical, "tag_efficiency": _tagging()}
    first = module._decision_payload(report)
    changed = json.loads(json.dumps(report))
    changed["tag_efficiency"]["summaries"]["full/greedy"]["channels"][0]["correct"] = 0
    assert first["tag_efficiency"] == report["tag_efficiency"]
    assert first != module._decision_payload(changed)


@pytest.mark.parametrize("include_tagging", [False, True])
def test_native_export_preserves_readable_tagging_without_changing_public_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_tagging: bool
):
    from scripts import export_phase69_native_metrics as export

    arm, job = "fixture", "123"
    source = tmp_path / "source"
    output = tmp_path / "export"
    run = source / f"artifacts/runs/ht-reconstruction-phase69-20261005/{arm}/{job}"
    reports = run / "full-decay-reports"
    reports.mkdir(parents=True)
    training = run / "training"
    training.mkdir()
    (training / "metrics.jsonl").write_text('{"loss": 1.0}\n')
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    result = {
        "pretraining_refinement": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": export.digest(checkpoint),
            "additional_pretraining_steps": 0,
            "mode": "frozen_checkpoint_reuse_no_training",
        },
        "transfer_report": {
            name: []
            for name in (
                "missing_keys", "unexpected_keys", "shape_mismatches",
                "leaf_pid_missing_keys", "leaf_pid_unexpected_keys",
                "leaf_pid_shape_mismatches", "loaded_keys", "leaf_pid_loaded_keys",
            )
        },
        "checkpoints": {},
    }
    (run / "result.json").write_text(json.dumps(result))
    report = {
        "summaries": {},
        "summaries_by_source_category": {},
        "summaries_by_target_shape": {},
        "retained_tree_checks": {"summaries": {}},
    }
    if include_tagging:
        report["tag_efficiency"] = _tagging()
    (reports / "primary.json").write_text(json.dumps(report))
    monkeypatch.setattr(export, "ARMS", {arm: job})
    monkeypatch.setattr(
        sys, "argv", ["export", "--source", str(source), "--output", str(output)]
    )
    export.main()

    public = json.loads((output / "public-primary.json").read_text())
    assert set(public) == {"version", "view", "rows"}
    assert public["rows"] == []
    sidecar = output / f"{arm}-tag-efficiency.json"
    manifest = json.loads((output / "export-manifest.json").read_text())
    assert sidecar.exists() is include_tagging
    assert any(row["name"] == sidecar.name for row in manifest["files"]) is include_tagging
    if include_tagging:
        payload = json.loads(sidecar.read_text())
        assert payload["reports"]["full-decay-reports/primary.json"] == _tagging()
        with gzip.open(output / f"{arm}-all-native-scalars.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        assert any(
            row["metric"] == "tag_efficiency.summaries.full/greedy.channels.0.correct"
            and row["value"] == 1
            for row in rows
        )
