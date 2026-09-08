from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_full_reconstruction_evaluation_suite as suite


def _manifest(event_uids: list[str]) -> dict[str, object]:
    return {
        "manifest_version": suite.MANIFEST_VERSION,
        "role": "validation",
        "sealed_test_role_access": "forbidden",
        "event_uid_count": len(event_uids),
        "event_uids_sha256": suite._uid_sequence_sha256(event_uids),
        "event_uids": event_uids,
    }


def _summary() -> dict[str, object]:
    ratio = {"numerator": 2, "denominator": 2, "value": 1.0}
    return {
        "decay_metrics": {
            "source_recall": ratio,
            "source_precision": ratio,
            "lcag_pair_accuracy": ratio,
            "perfect_lcag": ratio,
            "mother_pid_coverage": ratio,
            "root_pid_accuracy": ratio,
        },
        "inference": {
            "configured_root_completion": ratio,
            "inference_structurally_valid": ratio,
            "p4_closure": ratio,
            "recursive_detector_sources_disjoint": ratio,
            "forest_root_sources_disjoint": ratio,
        },
    }


def _report(*, uids: list[str], scope: str, topology: str, beam: bool) -> dict:
    scopes = ["full", "half"] if scope == "both" else [scope]
    summaries = {name: _summary() for name in scopes}
    beam_summary = {
        ranking: _summary()["decay_metrics"]
        for ranking in suite.MODEL_ONLY_BEAM_RANKINGS
    }
    return {
        "checkpoint_pair": {"compatible": True},
        "context": {"evaluated_event_uids": uids},
        "evaluator_code_provenance": {"git_head": "fixture"},
        "timing": {"event_count": len(uids)},
        "configuration": {
            "evaluated_scopes": scopes,
            "requested_scope": scope,
            "truth_topology_mode": topology,
            "pointer_threshold": 0.35,
        },
        "summaries": summaries,
        "summaries_by_source_category": {},
        "summaries_by_target_shape": {name: {} for name in scopes},
        "events": [{"event_uid": uid} for uid in uids],
        "beam_search": {
            "event_count": len(uids) if beam else 0,
            "top1_summaries_by_model_only_ranking": beam_summary if beam else {},
            "oracle_at_k_summary": _summary()["decay_metrics"] if beam else {},
            "events": [],
        },
    }


def test_manifest_is_validation_only_and_identity_bound(tmp_path: Path) -> None:
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(_manifest(["a", "b"])), encoding="utf-8")
    assert suite._load_validation_manifest(path, 2)["event_uids"] == ["a", "b"]

    payload = _manifest(["a", "b"])
    payload["role"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation-only"):
        suite._load_validation_manifest(path, 2)


def test_full_suite_runs_every_component_and_writes_one_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    pretraining = tmp_path / "pretrain.pt"
    reconstruction = tmp_path / "reconstruction.pt"
    data = tmp_path / "selection.json"
    index = tmp_path / "index.json"
    manifest = tmp_path / "cohort.json"
    for path in (python, pretraining, reconstruction, data, index):
        path.write_bytes(path.name.encode())
    manifest.write_text(json.dumps(_manifest(["one", "two"])), encoding="utf-8")
    output = tmp_path / "suite"
    calls: list[dict] = []

    def fake_run_component(**kwargs):
        calls.append(kwargs)
        report = _report(
            uids=list(kwargs["expected_uids"]),
            scope=kwargs["expected_scope"],
            topology=kwargs["expected_topology"],
            beam=kwargs["expect_beam"],
        )
        kwargs["output"].write_text(json.dumps(report), encoding="utf-8")
        kwargs["log"].write_text("{}\n", encoding="utf-8")
        return report

    monkeypatch.setattr(suite, "_run_component", fake_run_component)
    result = suite.main(
        [
            "--python",
            str(python),
            "--pretraining-checkpoint",
            str(pretraining),
            "--reconstruction-checkpoint",
            str(reconstruction),
            "--data",
            str(data),
            "--dataset-index",
            str(index),
            "--event-uid-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--max-events",
            "2",
            "--beam-max-events",
            "1",
        ]
    )
    assert result == 0
    assert [call["name"] for call in calls] == [
        "strict-checkpoint-direct",
        "strict-checkpoint-direct-repeat",
        "contracted-topology-diagnostic",
        "full-depth-beam-search",
    ]
    assert calls[-1]["expected_uids"] == ["one"]
    assert "--beam-width" in calls[-1]["command"]
    assert "--deterministic-algorithms" in calls[0]["command"]

    receipt = json.loads(
        (output / "full-evaluation-suite.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "completed"
    assert receipt["configuration"]["resolved_pointer_threshold"] == 0.35
    assert receipt["validation_checks"][
        "strict_repeat_scientifically_identical"
    ] is True
    assert receipt["results"]["strict_checkpoint_direct"]["summaries"]["full"]
    assert receipt["results"]["contracted_topology_diagnostic"]["summaries"][
        "half"
    ]
    assert receipt["results"]["beam_search"]["event_count"] == 1
    assert "average_link_probability" in receipt["metric_catalog"]
    assert receipt["metric_catalog"]["oracle_at_k"].startswith("Truth-ranked")
    assert receipt["checkpoint_pair"]["compatible"] is True
    assert receipt["component_timing"]["full_depth_beam_search"][
        "event_count"
    ] == 1


def test_repeat_mismatch_fails_closed(tmp_path: Path) -> None:
    report = _report(
        uids=["one"], scope="both", topology="checkpoint_direct", beam=False
    )
    changed = json.loads(json.dumps(report))
    changed["events"][0]["event_uid"] = "different"
    assert suite._decision_payload(report) != suite._decision_payload(changed)


def test_environment_python_symlink_is_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base-python"
    base.write_bytes(b"python")
    environment = tmp_path / "environment"
    environment.mkdir()
    entrypoint = environment / "python"
    entrypoint.symlink_to(base)
    assert entrypoint.absolute() == entrypoint
    assert entrypoint.resolve() == base
