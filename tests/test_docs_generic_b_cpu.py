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
    assert gb002["status"] == "HISTORICAL_SUPERSEDED_OPERATIONAL_SNAPSHOT"
    assert gb002["snapshot_recorded_state"] == "OPERATIONAL_PREPARATION_ADMITTED"
    assert gb002["current_state"] == "VERSIONED_B_ONLY_REPLANNING"
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
    assert "GB002: superseded snapshot and current replanning status" in rendered
    assert "historical and now-superseded operational snapshot" in rendered
    assert "versioned B-only replanning" in rendered
    assert "not an execution approval" in rendered
    assert "Every learned" in rendered and "UNAVAILABLE" in rendered
    assert "Latest learned full reconstruction evaluation — CR054" in rendered
    assert "Channel coverage" in rendered and "256 signed retained-channel groups" in rendered
    assert "12/758" in rendered and "parton ancestry reconstruction is UNAVAILABLE" in rendered
    assert "1/256 versus 0/256" in rendered and "not full reconstruction" in rendered
    published = (output / "generic-b-status-summary.json").read_text()
    assert '"exact_b_recovery": null' in published
    assert '"current_state": "VERSIONED_B_ONLY_REPLANNING"' in published
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
