"""Generic-B dashboard publication keeps static, learned, and operational semantics distinct."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "docs" / "_ext" / "wiki_generic_b.py"
SPEC = importlib.util.spec_from_file_location("wiki_generic_b_test", MODULE)
assert SPEC and SPEC.loader
generic_b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generic_b)


def _regenerate_binding(root):
    source = root / generic_b.SOURCE
    raw = (source / "summary.json").read_bytes()
    binding = json.loads((source / "binding.json").read_text())
    binding["source_bytes"] = len(raw)
    binding["source_sha256"] = hashlib.sha256(raw).hexdigest()
    (source / "binding.json").write_text(json.dumps(binding))


def test_generic_b_aggregate_preserves_static_and_unavailable_semantics(tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    value = generic_b.generate(ROOT, output)
    gb001, gb002, cr054 = value["gb001"], value["gb002"], value["cr054"]
    assert gb001["learned_model_evaluation"] is False
    assert gb001["static_compatibility"]["accepted_membership"] == {
        "compatible": 34819, "available": 40000, "nominal": 40000, "unavailable": 0,
    }
    assert gb001["static_compatibility"]["recursive_topology_pid"]["compatible"] == 6378
    assert gb001["population"]["b_collisions"] == 20000
    assert gb001["population"]["continuum_collisions"] == 8000
    assert all(metric is None for metric in gb001["learned_metrics"].values())
    assert gb002["training_completed"] is False and gb002["full_evaluation_completed"] is False
    assert gb002["status"] == "ACTIVE_OPERATIONAL_REPAIR_NO_LEARNED_EVALUATION"
    assert gb002["snapshot_recorded_state"] == "OPERATIONAL_PREPARATION_ADMITTED"
    assert gb002["current_state"] == "CONTINUOUS_SCIENCE_LOOP_RESUMED_OPERATIONAL_REPAIR_ACTIVE"
    assert gb002["snapshot_scope"] == "AS_OF_REVIEWED_SCIENCE_LOOP_RESUMPTION_BOUNDARY"
    assert gb002["snapshot_is_live_monitor"] is False
    assert gb002["local_data"] == {
        "transfer_status": "COMPLETE_HASH_VERIFIED",
        "transfer_scope": "OPAQUE_BYTES_ONLY",
        "scientific_payload_authentication_complete": False,
    }
    assert gb002["resource_policy"]["project_cpu_time_limit_seconds"] is None
    assert gb002["resource_policy"]["project_gpu_time_limit_hours"] is None
    assert gb002["runtime_attempt_at_snapshot"]["runtime_admission_passed"] is True
    assert gb002["runtime_attempt_at_snapshot"]["authenticated"] == 0
    assert gb002["runtime_attempt_at_snapshot"]["failed"] == 1
    assert gb002["replacement_transition"]["status"] == "FAILED_REVIEW_REQUIRED"
    assert gb002["replacement_transition"]["payload_execution_authorized"] is False
    assert gb002["science_loop"]["status"] == "RESUMED_UNDER_REVIEWED_ASTRA_SOL_ROUTING"
    assert gb002["science_loop"]["native_baseline_status"] == "UNMEASURED"
    assert gb002["next_scientific_plan"]["status"] == "SEALED_PROSPECTIVE_PLAN_NOT_A_RESULT"
    assert all(metric is None for metric in gb002["learned_metrics"].values())
    assert cr054["arms"]["control"]["native"]["exact_b_recovery"]["correct"] == 0
    assert cr054["arms"]["control"]["native"]["inclusive_b_recovery"]["nominal"] == 256
    assert cr054["arms"]["control"]["b_channel_coverage"] == {
        "evaluated_channel_groups": 256, "evaluated_b_trials": 256, "exact_successes": 0,
        "successful_channel_groups": 0, "unavailable": 0,
    }
    rendered = " ".join(generic_b.render(value))
    assert "generic-sample static target-contract census" in rendered
    assert "20,000 B and 8,000 continuum collisions" in rendered
    assert "not a learned-model evaluation" in rendered
    assert "not reconstruction efficiency" in rendered
    assert "needed provenance is unavailable in the census inputs" in rendered
    assert "original-mDST recovery was not assessed" in rendered
    assert "GB002: local transfer complete; scientific authentication unresolved" in rendered
    assert "historical and now-superseded operational snapshot" in rendered
    assert "as-of snapshot at the reviewed science-loop resumption boundary" in rendered
    assert "not a live job monitor" in rendered
    assert "opaque-byte identity only" in rendered
    assert "0 authenticated, 1 failed and 3,071 unreached" in rendered
    assert "event-number and source-entry identities were conflated" in rendered
    assert "FAILED_REVIEW_REQUIRED" in rendered
    assert "reviewed Astra/Sol science loop has since resumed" in rendered
    assert "payload decoding and the native baseline remain pending" in rendered
    assert "two Generic B training sample sizes by two network widths" in rendered
    assert "do not constitute execution approval" in rendered
    assert "Every learned" in rendered and "UNAVAILABLE" in rendered
    assert "Latest learned full reconstruction evaluation — CR054" in rendered
    assert "Channel coverage" in rendered and "256 signed retained-channel groups" in rendered
    assert "12/758" in rendered and "parton ancestry reconstruction is UNAVAILABLE" in rendered
    assert "1/256 versus 0/256" in rendered and "not full reconstruction" in rendered
    published = (output / "generic-b-status-summary.json").read_text()
    assert '"exact_b_recovery": null' in published
    assert '"current_state": "CONTINUOUS_SCIENCE_LOOP_RESUMED_OPERATIONAL_REPAIR_ACTIVE"' in published
    private_prefix = "/" + "project/"
    operational_markers = ("allocation" + "-", "stage" + "-000")
    assert private_prefix not in published
    assert all(marker not in published for marker in operational_markers)


def test_gb002_zero_cannot_replace_unmeasured_value(tmp_path):
    root = tmp_path / "repository"
    source = root / generic_b.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / generic_b.SOURCE, source)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gb002"]["learned_metrics"]["exact_b_recovery"] = 0
    summary_path.write_text(json.dumps(summary))
    _regenerate_binding(root)
    output = tmp_path / "generated"
    output.mkdir()
    with pytest.raises(ValueError, match="GB002 unmeasured status"):
        generic_b.generate(root, output)


def test_gb002_snapshot_cannot_be_presented_as_current_approval(tmp_path):
    root = tmp_path / "repository"
    source = root / generic_b.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / generic_b.SOURCE, source)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gb002"]["current_state"] = "EXECUTION_APPROVED"
    summary_path.write_text(json.dumps(summary))
    _regenerate_binding(root)
    output = tmp_path / "generated"
    output.mkdir()
    with pytest.raises(ValueError, match="GB002 unmeasured status"):
        generic_b.generate(root, output)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("local_data", "scientific_payload_authentication_complete", True),
    ("resource_policy", "project_cpu_time_limit_seconds", 1),
    ("runtime_attempt_at_snapshot", "authenticated", 1),
    ("replacement_transition", "status", "ACTIVATED"),
])
def test_gb002_readiness_cannot_overstate_operational_evidence(
        tmp_path, section, field, value):
    root = tmp_path / "repository"
    source = root / generic_b.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / generic_b.SOURCE, source)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gb002"][section][field] = value
    summary_path.write_text(json.dumps(summary))
    _regenerate_binding(root)
    output = tmp_path / "generated"
    output.mkdir()
    with pytest.raises(ValueError, match="GB002 unmeasured status"):
        generic_b.generate(root, output)


def test_gb001_count_cannot_be_relabelled_as_efficiency(tmp_path):
    root = tmp_path / "repository"
    source = root / generic_b.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / generic_b.SOURCE, source)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gb001"]["reconstruction_efficiency"] = 34819 / 40000
    summary_path.write_text(json.dumps(summary))
    _regenerate_binding(root)
    output = tmp_path / "generated"
    output.mkdir()
    with pytest.raises(ValueError, match="GB001 field allowlist"):
        generic_b.generate(root, output)
