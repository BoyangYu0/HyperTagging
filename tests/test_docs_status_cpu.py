"""Status publication is a field allowlist, never a source/evidence mirror."""
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
_MODULE = Path(__file__).resolve().parents[1] / "docs" / "_ext" / "wiki_status.py"
_SPEC = importlib.util.spec_from_file_location("wiki_status", _MODULE)
assert _SPEC and _SPEC.loader
status = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(status)


def write_source(root, key, document):
    path = root / status.SOURCE_PATHS[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")


def source_info(manifest, key):
    return manifest["sources"][status.SOURCE_IDS[key]]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr(status, "_git", lambda *_args: None)
    documents = {
        "reconstruction_phase56": {"reserved": True},
        "reconstruction_phase56_retained": {"reserved": True},
        "reconstruction_phase56_aggregation": {"reserved": True},
        "reconstruction_phase56_pretraining": {"reserved": True},
        "reconstruction_phase57_submission": {"reserved": True},

        "reconstruction_phase48": {"reserved_for_phase48_closeout": True},
        "reconstruction_phase48_retained": {"reserved_for_phase48_retained": True},
        "reconstruction_phase48_aggregation": {"reserved_for_phase48_aggregation": True},
        "reconstruction_phase49_submission": {"reserved_for_phase49": True},
        "reconstruction_phase46": {"reserved_for_phase46_closeout": True},
        "reconstruction_phase46_retained": {"reserved_for_phase46_retained": True},
        "reconstruction_phase47_submission": {"reserved_for_phase47": True},
        "reconstruction_phase47": {"reserved_for_phase47_closeout": True},
        "reconstruction_phase47_retained": {"reserved_for_phase47_retained": True},
        "reconstruction_phase48_submission": {"reserved_for_phase48": True},
        "reconstruction_phase47_aggregation": {"reserved_for_phase47_aggregation": True},
        "reconstruction_phase44": {"reserved_for_phase44": True},
        "reconstruction_phase44_retained": {"reserved_for_phase44": True},
        "reconstruction_phase45_submission": {"reserved_for_phase45": True},
        "reconstruction_phase45": {"reserved_for_phase45_closeout": True},
        "reconstruction_phase45_retained": {"reserved_for_phase45_retained": True},
        "reconstruction_phase46_submission": {"reserved_for_phase46": True},
        "reconstruction_phase42": {"reserved_for_phase42": True},
        "reconstruction_phase43": {"reserved_for_phase43": True},
        "reconstruction_phase44_submission": {"reserved_for_phase44": True},
        "reconstruction_phase43_submission": {"reserved_for_phase43": True},
        "reconstruction_phase41": {"reserved_for_phase41": True},
        "reconstruction_phase42_submission": {"reserved_for_phase42": True},
        "current_status": "# Current status\n\n## Recommendation: NO-GO\n\nNo current real pilot.\n\n## Older result\n\n999 passed.\n",
        "issue_ledger": {"audit_generated_at": "2026-08-01T12:00:00+00:00", "audited_code_sha": "a" * 40,
                         "items": [{"id": "A", "current_status": "FIXED_AND_TESTED"}, {"id": "B", "current_status": "PARTIAL"}]},
        "verification_runs": {"runs": [
            {"date": "2026-08-01", "pytest": {"passed": 999}},
            {"date": "2026-08-02", "source_sha": "b" * 40, "pytest": {"result": "PASS", "passed": 3, "skipped": 1},
             "human_visual_review_status": "NOT_REVIEWED", "ci_run": "NOT_RUN",
             "notebooks": {"default_fixture": {"result": "NOT_RUN", "count": 0},
                           "real_mdst_pilot": {"result": "NOT_RUN", "count": 0},
                           "trained_physics_validation": {"result": "NOT_RUN", "count": 0}}}]},
        "notebook_registry": {"visual_review_status": "NOT_REVIEWED", "verification_base_head": "a" * 40,
                              "notebooks": [{"group": "CORE_CONTRACT", "fixture_or_real": "fixture", "default_smoke": True},
                                            {"group": "EXTERNAL_SCIENTIFIC", "fixture_or_real": "real_only", "last_verified_sha": "NOT_RUN"}]},
        "pretraining_contract": {"scientific_contract": {"total_presentations": 1730048, "virtual_step_presentations": 16,
                                                            "resume_step": 54064, "validation_milestones_virtual_steps": [54064, 108128]}},
        "pretraining_selection": {"selected_profile": None, "production_submission_authorized": False,
                                  "submission_performed": False, "candidates": {
            "private-device-one": {"status": "calibration_pending"}, "private-device-two": {"status": "calibration_pending"}}},
        "transfer_preregistration": {"study_date": "2026-08-04", "pretraining_success_gate_passed": False,
                                     "pretraining_contract_exception": {"authorized_checkpoint_step": 54064, "authorized_checkpoint": "/never/read/private-state.pt"}},
        "reconstruction_terminal": {"audit_version": "2026-08-27.stage-a.v1", "complete": True,
                                    "recommendation": "DO_NOT_PROMOTE", "exploratory": True,
                                    "jobs": {"12345678": {"state": "COMPLETED"}, "87654321": {"state": "COMPLETED"}},
                                    "validation": {"delta_relbias_minus_q32": {"edge_f1": 0.008}, "required_edge_f1_delta": 0.010,
                                                   "paired_event_evidence": False, "guards": "not_established", "evidence_class": "aggregate_only_nonpromotable"}},
        "reconstruction_phase40r1": {
            "audit_version": "2026-09-09.phase40r1-closeout.v1",
            "status": "COMPLETED", "strict_event_count": 100,
            "beam_event_count": 20, "primary_repeat_identical": True,
            "sealed_test_accessed": False, "promotion_authorized": False,
            "longer_run_authorized": False,
            "control": {"selected_step": 4000,
                        "micro_complete_target_efficiency": 0.045,
                        "full_root_completion_numerator": 1,
                        "full_root_completion_denominator": 100,
                        "all_gates_passed": False},
            "query_scale": {"selected_step": 2000,
                            "micro_complete_target_efficiency": 0.035,
                            "full_root_completion_numerator": 2,
                            "full_root_completion_denominator": 100,
                            "all_gates_passed": False},
            "control_beam": {"average_link_source_recall": 0.25,
                             "average_link_source_precision": 0.68,
                             "average_link_lcag_numerator": 1,
                             "average_link_lcag_denominator": 329,
                             "oracle_source_recall": 0.29,
                             "oracle_lcag_numerator": 1,
                             "oracle_lcag_denominator": 329,
                             "oracle_is_diagnostic_only": True},
            "decision": {"winning_arm": "CONTROL",
                         "query_scale_continuation": "STOP",
                         "longer_budget": "NOT_AUTHORIZED",
                         "next_study": "LEVEL1_POINTER_BALANCE",
                         "next_study_status": "PREREGISTRATION_IN_PROGRESS"}},
        "cpu_workflow": {"jobs": {"unit": {"steps": [{"run": "python -m pytest -q | tee private-log.txt"}]}}},
    }
    for key in ("reconstruction_phase49", "reconstruction_phase49_retained", "reconstruction_phase49_aggregation", "reconstruction_phase50_submission", "reconstruction_phase50", "reconstruction_phase50_retained", "reconstruction_phase50_aggregation", "reconstruction_phase51_submission", "reconstruction_phase51", "reconstruction_phase51_retained", "reconstruction_phase51_aggregation", "reconstruction_phase52_submission", "reconstruction_phase52", "reconstruction_phase52_retained", "reconstruction_phase52_aggregation", "reconstruction_phase53_submission", "reconstruction_phase53", "reconstruction_phase53_retained", "reconstruction_phase53_aggregation", "reconstruction_phase54_submission", "reconstruction_phase54", "reconstruction_phase54_retained", "reconstruction_phase54_aggregation", "reconstruction_phase55_submission", "reconstruction_phase55", "reconstruction_phase55_retained", "reconstruction_phase55_aggregation", "reconstruction_phase55_pretraining", "reconstruction_phase56_submission"):
        documents[key] = {"reserved_for_phase49_review": True}
    recent = documents["reconstruction_phase40r1"]
    recent.update({"audit_version": "2026-09-09.phase40r1-closeout.v3",
                   "current_best_arm": "CONTROL",
                   "metric_contract_version": "reconstruction-current-best-complete-v2",
                   "metric_completeness": "COMPLETE"})
    complete_arm_defaults = {
        "predicted_depth_fraction": 2.0, "tree_validity": 1.0,
        "full_source_recall": 0.2, "full_source_precision": 0.7,
        "half_source_recall": 0.21, "half_source_precision": 0.45,
        "full_lcag_numerator": 1, "full_lcag_denominator": 2518,
        "exact_mother_coverage_numerator": 1, "exact_mother_coverage_denominator": 149,
        "half_lcag_numerator": 15, "half_lcag_denominator": 1724,
        "half_perfect_lcag_numerator": 1, "half_perfect_lcag_denominator": 164,
    }
    for arm in ("control", "query_scale"):
        recent[arm].update(complete_arm_defaults)
    metric_point = {"numerator": 1, "denominator": 10, "value": 0.1}
    recent["beam_rankings"] = {
        ranking: {metric: dict(metric_point) for metric in (
            "source_recall", "source_precision", "lcag_pair_accuracy",
            "mother_pid_coverage", "perfect_lcag")}
        for ranking in ("greedy", "average_link_probability", "learned_confidence_mean",
                        "learned_confidence_sum", "normalized_joint_log_probability", "oracle_at_k")
    }
    recent["beam_half_rankings"] = json.loads(json.dumps(recent["beam_rankings"]))
    for key, document in documents.items():
        write_source(root, key, document)
    return root


def test_status_is_deterministic_and_preserves_record_scope(evidence, tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1788825600")
    first, second = tmp_path / "first", tmp_path / "second"
    before_imports = set(sys.modules)
    manifest = status.generate_status(evidence, first)
    assert status.generate_status(evidence, second) == manifest
    files = lambda directory: {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert files(first) == files(second)
    assert manifest["audit"]["recommendation"] == "NO_GO"
    assert manifest["audit"]["ledger_status_counts"] == {"FIXED_AND_TESTED": 1, "PARTIAL": 1}
    assert manifest["verification"]["latest_record"]["pytest"]["passed"] == 3
    assert manifest["verification"]["record_count"] == 2
    assert manifest["verification"]["current_build_test_result"] == "NOT_RUN_BY_DASHBOARD"
    assert manifest["science"]["live_scheduler_status"] == "UNKNOWN_NOT_QUERIED"
    assert manifest["science"]["real_pilot"] == manifest["science"]["trained_physics"] == "NOT_RUN"
    assert manifest["pretraining"]["production_submission_authorized"] is False
    assert manifest["pretraining"]["selected_profile_state"] == "NONE_SELECTED"
    assert manifest["pretraining"]["submission_performed"] is False
    assert manifest["pretraining"]["pretraining_success_gate_passed"] is False
    assert manifest["provenance"]["tracked_repository_artifact_inputs_opened"] == 60
    assert manifest["provenance"]["external_filesystem_or_network_artifacts_opened"] is False
    assert source_info(manifest, "issue_ledger")["freshness"]["status"] == "stale"
    assert source_info(manifest, "current_status")["freshness"]["status"] == "unknown"
    assert not ({"torch", "basf2", "hypertagging"} & (set(sys.modules) - before_imports))
    assert "not live authorization" in (first / "index.rst").read_text()


def test_sources_are_hashed_but_never_copied_or_named(evidence, tmp_path):
    output = tmp_path / "generated"
    manifest = status.generate_status(evidence, output)
    published = "\n".join(path.read_text() for path in output.iterdir())
    for key, path in status.SOURCE_PATHS.items():
        source = source_info(manifest, key)
        assert source["sha256"] == hashlib.sha256((evidence / path).read_bytes()).hexdigest()
        assert "path" not in source and "download" not in source
        assert path not in published
        assert (evidence / path).read_text() not in published
    assert {path.name for path in output.iterdir()} == {"index.rst", "status.json"}
    assert manifest["provenance"]["as_of"] is None
    assert manifest["audit"]["revision_match"] == "unknown"
    assert json.loads((output / "status.json").read_text()) == manifest


def test_unknown_and_nested_fields_cannot_publish_private_identifiers(evidence, tmp_path):
    secrets = ["/home/private-user/key", "internal-node.example.invalid", "GPU-12345678-dead-beef-cafe-000000000001",
               "scheduler-job-87654321", "checkpoint-secret.pt", "sk-secret-DO-NOT-PUBLISH", "PRIVATE-AUTHORITY-PATH"]
    payload = " | ".join(secrets)
    for key, path in status.SOURCE_PATHS.items():
        source = evidence / path
        if source.suffix == ".md":
            source.write_text(source.read_text() + "\n\n" + payload)
            continue
        document = json.loads(source.read_text())
        def inject(value):
            if isinstance(value, dict):
                for nested in list(value.values()):
                    inject(nested)
                value["private_unknown_field"] = {"secret": payload}
            elif isinstance(value, list):
                for nested in value:
                    inject(nested)
        inject(document)
        source.write_text(json.dumps(document))
    output = tmp_path / "generated"
    manifest = status.generate_status(evidence, output)
    published = "\n".join(path.read_text() for path in output.iterdir())
    for secret in secrets:
        assert secret not in published
    assert "private_unknown_field" not in published
    assert manifest["verification"]["latest_record"]["pytest"]["passed"] == 3
    assert manifest["pretraining"]["recorded_step"] == 54064


def test_bad_known_scalar_values_are_dropped_not_stringified(evidence, tmp_path):
    secret = "INTERNAL_HOST_SECRET"
    write_source(evidence, "issue_ledger", {"audited_code_sha": secret, "audit_generated_at": "2026-08-01 " + secret,
                                          "items": [{"current_status": secret}]})
    write_source(evidence, "verification_runs", {"runs": [{"date": "2026-08-01", "source_sha": secret,
                                                          "pytest": {"result": secret, "passed": secret, "duration_seconds": secret}}]})
    output = tmp_path / "generated"
    manifest = status.generate_status(evidence, output)
    assert secret not in "\n".join(path.read_text() for path in output.iterdir())
    assert manifest["audit"]["ledger_status_counts"] == {"UNKNOWN": 1}
    assert manifest["verification"]["latest_record"]["pytest"]["passed"] is None


def test_missing_and_invalid_evidence_never_implies_success(evidence, tmp_path):
    write_source(evidence, "current_status", "No recommendation available")
    write_source(evidence, "issue_ledger", "[broken:")
    write_source(evidence, "pretraining_selection", "{broken")
    manifest = status.generate_status(evidence, tmp_path / "generated")
    assert source_info(manifest, "issue_ledger")["availability"] == "invalid"
    assert source_info(manifest, "pretraining_selection")["availability"] == "invalid"
    assert manifest["audit"]["recommendation"] == "UNKNOWN"
    assert manifest["audit"]["ledger_item_count"] is None
    assert manifest["pretraining"]["production_submission_authorized"] is None
    assert manifest["pretraining"]["selected_profile_state"] == "UNKNOWN"


def test_source_symlink_is_not_followed(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    source = root / status.SOURCE_PATHS["pretraining_selection"]
    source.parent.mkdir(parents=True)
    forbidden = tmp_path / "external-state.pt"
    forbidden.write_bytes(b"SECRET-STATE")
    source.symlink_to(forbidden)
    monkeypatch.setattr(status, "_git", lambda *_args: None)
    manifest = status.generate_status(root, tmp_path / "generated")
    assert source_info(manifest, "pretraining_selection")["availability"] == "unsafe_path"
    assert source_info(manifest, "pretraining_selection")["sha256"] is None


def test_latest_record_respects_timezone_offsets(evidence, tmp_path):
    write_source(evidence, "verification_runs", {"runs": [
        {"date": "2026-08-01T10:00:00+00:00", "pytest": {"passed": 2}},
        {"date": "2026-08-01T11:00:00+02:00", "pytest": {"passed": 9}}]})
    manifest = status.generate_status(evidence, tmp_path / "generated")
    assert manifest["verification"]["latest_record"]["pytest"]["passed"] == 2


def test_metadata_cannot_introduce_rst_directives(evidence, tmp_path):
    write_source(evidence, "current_status", "# Audit\n\n## Recommendation: NO-GO\n\n.. include:: /private/SECRET\n\n.. raw:: html\n\n   <script>SECRET</script>\n")
    output = tmp_path / "generated"
    status.generate_status(evidence, output)
    assert "SECRET" not in (output / "index.rst").read_text()
    assert ".. include::" not in (output / "index.rst").read_text()
    assert "<script>" not in (output / "index.rst").read_text()


@pytest.mark.parametrize("contents", ['{"invalid": NaN}', '{"invalid": Infinity}'])
def test_nonfinite_metadata_is_invalid(evidence, tmp_path, contents):
    write_source(evidence, "pretraining_selection", contents)
    manifest = status.generate_status(evidence, tmp_path / "generated")
    assert source_info(manifest, "pretraining_selection")["availability"] == "invalid"
    assert manifest["pretraining"]["production_submission_authorized"] is None


def test_revision_warnings_and_changed_sources_are_disclosed(evidence, tmp_path, monkeypatch):
    def git(_root, *arguments):
        if arguments[:1] == ("log",):
            return ("c" * 40 + "\n2026-08-02T00:00:00+00:00\n").encode()
        if arguments[:1] == ("show",):
            return b"previous bytes"
        return None
    monkeypatch.setattr(status, "_git", git)
    output = tmp_path / "generated"
    manifest = status.generate_status(evidence, output)
    assert manifest["provenance"]["as_of_basis"] == "git_commit_time"
    assert manifest["audit"]["revision_match"] == "different_revision"
    assert manifest["verification"]["latest_record"]["revision_match"] == "different_revision"
    assert "STALE REVISION EVIDENCE" in (output / "index.rst").read_text()
    assert source_info(manifest, "current_status")["worktree_state"] == "modified_from_head"
    assert source_info(manifest, "issue_ledger")["freshness"]["status"] == "recent"
    assert source_info(manifest, "transfer_preregistration")["freshness"]["status"] == "after_reference_date"


def test_dashboard_has_accessible_performance_cards_and_nonhtml_fallback(evidence, tmp_path):
    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = []
        def handle_starttag(self, tag, attrs):
            self.elements.append((tag, dict(attrs)))
    output = tmp_path / "generated"
    manifest = status.generate_status(evidence, output)
    document = (output / "index.rst").read_text()
    parser = Elements()
    parser.feed(document)
    articles = [attrs for tag, attrs in parser.elements if tag == "article"]
    assert len(articles) == 3
    assert all("aria-labelledby" in attrs for attrs in articles)
    assert ".. only:: not html" in document
    assert document.index("Reconstruction edge F1") < document.index("Training and software context")
    assert "DO_NOT_PROMOTE" in document and "+0.008" in document and "+0.010" in document
    assert "UNAVAILABLE" in document and "NOT_RUN" in document
    assert manifest["pretraining"]["progress_fraction"] == 0.5
    assert manifest["reconstruction"]["completion"] == "COMPLETED"


def test_failed_pytest_record_is_rendered_as_blocking_evidence(evidence, tmp_path):
    write_source(evidence, "verification_runs", {"runs": [{
        "date": "2026-08-03",
        "source_sha": "b" * 40,
        "pytest": {
            "result": "FAIL", "passed": 10, "failed": 2,
            "skipped": 1, "warnings": 3, "duration_seconds": 4.5,
        },
    }]})
    output = tmp_path / "generated"
    status.generate_status(evidence, output)
    document = (output / "index.rst").read_text()
    assert status._literal("FAIL: 10 passed") in document
    assert "2 failed" in document
    assert "CPU fixtures" in document


@pytest.mark.parametrize("shell,command,expected", [
    (None, "python -m pytest | tee result.txt", 1),
    ("bash", "python -m pytest | tee result.txt", 0),
    (None, "set -euo pipefail\npython -m pytest | tee result.txt", 0),
])
def test_cpu_ci_tee_limitation_is_derived_from_shell_settings(shell, command, expected):
    step = {"run": command}
    if shell:
        step["shell"] = shell
    result = status._ci_projection({"jobs": {"unit": {"steps": [step]}}})
    assert result["without_explicit_pipefail"] == expected


def test_current_repository_dashboard_surfaces_recorded_acceptance_values(tmp_path):
    root = _MODULE.parents[2]
    manifest = status.generate_status(root, tmp_path / "generated")
    assert manifest["audit"]["recommendation"] == "NO_GO"
    assert manifest["audit"]["ledger_status_counts"] == {"FIXED_AND_TESTED": 85, "IMPLEMENTED_NOT_REAL_VERIFIED": 6,
        "INTENTIONALLY_DEFERRED_SCIENCE": 4, "OBSOLETE_OR_DUPLICATE": 1, "PARTIAL": 9}
    assert manifest["verification"]["latest_record"]["pytest"]["passed"] == 463
    assert manifest["notebooks"]["total"] == 18 and manifest["notebooks"]["default_smoke"] == 15
    assert manifest["notebooks"]["input_modes"] == {"fixture": 16, "real_only": 2}
    assert manifest["notebooks"]["visual_review_status"] == "NOT_REVIEWED"
    assert manifest["pretraining"]["recorded_step"] == 54064 and manifest["pretraining"]["planned_steps"] == 108128
    assert manifest["pretraining"]["calibration_status"] == "PENDING"
    rendered_status = (tmp_path / "generated" / "index.rst").read_text().replace("\\-", "-")
    for required_label in (
        "half-tree source recall", "half-tree source precision", "half-tree LCAG",
        "perfect half-tree LCAG", "average link probability", "learned confidence mean",
        "learned confidence sum", "normalized joint log probability", "oracle at k",
    ):
        assert required_label in rendered_status
    assert manifest["pretraining"]["production_submission_authorized"] is False
    assert manifest["pretraining"]["selected_profile_state"] == "NONE_SELECTED"
    assert manifest["pretraining"]["submission_performed"] is False
    assert manifest["reconstruction"]["completion"] == "COMPLETED"
    assert manifest["reconstruction"]["recommendation"] == "DO_NOT_PROMOTE"
    assert manifest["reconstruction"]["edge_f1_delta"] == 0.008
    assert manifest["reconstruction"]["metrics"]["relbias"]["edge_f1"] == 0.039
    assert manifest["reconstruction"]["metrics"]["q32"]["edge_f1"] == 0.031
    assert manifest["reconstruction"]["cohorts"]["relbias"] == {"validation_events": 2000, "rollout_events": 1000}
    phase40r1 = manifest["reconstruction"]["phase40r1"]
    assert phase40r1["status"] == "COMPLETED"
    assert phase40r1["arms"]["control"]["micro_complete_target_efficiency"] == pytest.approx(0.045015189174261255)
    assert phase40r1["arms"]["control"]["full_root_completion_numerator"] == 1
    assert phase40r1["arms"]["query_scale"]["full_root_completion_numerator"] == 2
    assert phase40r1["arms"]["control"]["all_gates_passed"] is False
    assert phase40r1["arms"]["control"]["half_source_recall"] == pytest.approx(0.2123076923076923)
    assert phase40r1["arms"]["control"]["half_perfect_lcag_numerator"] == 1
    assert phase40r1["metric_completeness"] == "COMPLETE"
    assert set(phase40r1["beam_rankings_by_scope"]) == {"full", "half"}
    for scope in ("full", "half"):
        assert set(phase40r1["beam_rankings_by_scope"][scope]) == {
            "greedy", "average_link_probability", "learned_confidence_mean",
            "learned_confidence_sum", "normalized_joint_log_probability", "oracle_at_k",
        }
    assert phase40r1["beam_rankings_by_scope"]["half"]["greedy"][
        "lcag_pair_accuracy"
    ] == {"numerator": 3, "denominator": 213}
    assert phase40r1["decision"]["winning_arm"] == "CONTROL"
    assert phase40r1["decision"]["next_study_status"] == "SUBMITTED"
    assert phase40r1["sealed_test_accessed"] is False
    assert manifest["science"]["real_pilot"] == manifest["science"]["trained_physics"] == "NOT_RUN"
    assert manifest["cpu_ci"]["without_explicit_pipefail"] == 0


def test_current_best_metric_contract_rejects_missing_half_or_beam_metrics(evidence, tmp_path):
    source = evidence / status.SOURCE_PATHS["reconstruction_phase40r1"]
    payload = json.loads(source.read_text())
    del payload["control"]["half_source_recall"]
    write_source(evidence, "reconstruction_phase40r1", payload)
    with pytest.raises(ValueError, match="metric contract is incomplete"):
        status.generate_status(evidence, tmp_path / "missing-half")

    payload["control"]["half_source_recall"] = 0.21
    del payload["beam_rankings"]["learned_confidence_mean"]["perfect_lcag"]
    write_source(evidence, "reconstruction_phase40r1", payload)
    with pytest.raises(ValueError, match="metric contract is incomplete"):
        status.generate_status(evidence, tmp_path / "missing-beam")

    payload = json.loads(source.read_text())
    payload["beam_rankings"]["learned_confidence_mean"]["perfect_lcag"] = {
        "numerator": 1, "denominator": 10, "value": 0.1
    }
    del payload["beam_half_rankings"]["average_link_probability"][
        "lcag_pair_accuracy"
    ]
    write_source(evidence, "reconstruction_phase40r1", payload)
    with pytest.raises(ValueError, match="metric contract is incomplete"):
        status.generate_status(evidence, tmp_path / "missing-half-beam")


@pytest.mark.parametrize("value", ["yesterday", "-1", "1.5", "99999999999999999999999999999"])
def test_invalid_reproducible_reference_is_rejected(evidence, tmp_path, monkeypatch, value):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        status.generate_status(evidence, tmp_path / "generated")


def test_output_cannot_alias_repository_symlink_or_previous_raw_artifact(evidence, tmp_path):
    with pytest.raises(ValueError, match="dedicated"):
        status.generate_status(evidence, evidence)
    link = tmp_path / "link"
    link.symlink_to(evidence, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        status.generate_status(evidence, link)
    output = tmp_path / "generated"
    (output / "evidence").mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected files"):
        status.generate_status(evidence, output)


def test_metric_projection_and_target_gap(evidence, tmp_path):
    path = evidence / status.SOURCE_PATHS["reconstruction_terminal"]
    payload = json.loads(path.read_text())
    payload["validation"]["relbias"] = {"edge_f1": 0.039, "validation_loss": 2.46,
                                      "validation_events": 2000, "private_extra": "SECRET"}
    payload["validation"]["q32"] = {"edge_f1": 0.031, "validation_loss": "SECRET"}
    write_source(evidence, "reconstruction_terminal", payload)
    output = tmp_path / "generated"
    result = status.generate_status(evidence, output)["reconstruction"]
    assert result["metrics"]["relbias"]["edge_f1"] == 0.039
    assert result["metrics"]["q32"]["validation_loss"] is None
    assert result["metrics"]["relbias"]["pointer_recall"] is None
    assert result["cohorts"]["relbias"]["validation_events"] == 2000
    assert result["target_gap"] == pytest.approx(-0.002)
    assert "SECRET" not in (output / "index.rst").read_text()


def test_missing_metric_source_remains_unknown(evidence, tmp_path):
    (evidence / status.SOURCE_PATHS["reconstruction_terminal"]).unlink()
    output = tmp_path / "generated"
    manifest = status.generate_status(evidence, output)
    assert source_info(manifest, "reconstruction_terminal")["availability"] == "missing"
    result = manifest["reconstruction"]
    assert result["target_gap"] is None
    assert all(value is None for value in result["metrics"]["relbias"].values())
    assert "UNKNOWN" in (output / "index.rst").read_text()


def test_phase41_current_evidence_has_every_registered_metric_and_both_beam_scopes():
    source = ROOT / status.SOURCE_PATHS['reconstruction_phase41']
    assert source.stat().st_size <= status._MAX_SOURCE_BYTES
    payload = json.loads(source.read_text())
    projected = status._phase41_projection(payload)
    assert len(projected['metric_rows']) == len(payload['metric_rows'])
    for arm in ('pointer32_control', 'level1_pointer24'):
        record = projected['arms'][arm]
        assert record['endpoints']['full_root_completion']['numerator'] == 0
        assert record['endpoints']['full_root_completion']['denominator'] == 100
        assert set(record['beam']) == {'full', 'half'}
        assert len(record['beam']['half']) == 6
        assert record['all_gates_passed'] is False


def test_phase41_rejects_missing_metric_or_beam_statistic():
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase41']).read_text())
    removed = payload['metric_rows'].pop()
    with pytest.raises(ValueError, match='registry is incomplete'):
        status._phase41_projection(payload)
    payload['metric_rows'].append(removed)
    del payload['arms']['level1_pointer24']['beam']['half']['oracle_at_k']['source_recall']['denominator']
    with pytest.raises(ValueError, match='count is missing'):
        status._phase41_projection(payload)


def test_phase41_arbitrary_labels_and_nested_strings_do_not_publish():
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase41']).read_text())
    payload['metric_rows'].append({'arm': 'pointer32_control', 'view': 'private-host', 'metric': '/private/checkpoint.pt', 'value': 1})
    payload['arms']['pointer32_control']['private'] = '/private/checkpoint.pt'
    assert 'private' not in json.dumps(status._phase41_projection(payload))



def test_phase42_metrics_are_complete_and_keep_counts():
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase42']).read_text())
    result = status._phase41_projection(payload, phase=42, labels=('pretrain81096_control', 'pretrain108128'))
    assert len(result['metric_rows']) == len(payload['metric_rows']) == 16306
    assert result['arms']['pretrain81096_control']['endpoints']['full_lcag']['numerator'] == 2
    assert result['arms']['pretrain108128']['endpoints']['full_lcag']['numerator'] == 1
    assert set(result['arms']['pretrain108128']['beam']) == {'full', 'half'}
    assert (ROOT / status.SOURCE_PATHS['reconstruction_phase42']).stat().st_size < status._MAX_SOURCE_BYTES


@pytest.mark.parametrize('missing', ['metric', 'ranking'])
def test_phase42_incomplete_metrics_fail_closed(missing):
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase42']).read_text())
    if missing == 'metric':
        payload['metric_rows'].pop()
    else:
        del payload['arms']['pretrain108128']['beam']['half']['greedy']
    with pytest.raises(ValueError):
        status._phase41_projection(payload, phase=42, labels=('pretrain81096_control', 'pretrain108128'))



def test_complete_metric_download_stays_within_publication_limit(tmp_path):
    output = tmp_path / 'complete-status'
    manifest = status.generate_status(ROOT, output)
    download = output / 'status.json'
    assert download.stat().st_size < 10 * 1024 * 1024
    assert json.loads(download.read_text()) == manifest
    binding = manifest['reconstruction']['phase41']['metric_download']
    assert binding['metric_count'] == 21467
    metric_bytes = (output/binding['filename']).read_bytes()
    assert hashlib.sha256(metric_bytes).hexdigest() == binding['sha256']
    assert len(metric_bytes) == binding['bytes']
    original = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase41']).read_text())
    assert json.loads(metric_bytes)['metric_rows'] == original['metric_rows']
    assert manifest['reconstruction']['phase41']['metric_rows'] == [r for r in original['metric_rows'] if r['view'] == 'training_best']
    assert len(manifest['reconstruction']['phase42']['metric_rows']) == 16306
    assert len(manifest['reconstruction']['phase43']['metric_rows']) == 16405
    assert manifest['reconstruction']['phase41']['next_study_status'] == 'COMPLETED'
    assert manifest['reconstruction']['phase42']['next_study_status'] == 'COMPLETED'
    assert manifest['reconstruction']['phase43']['next_study_status'] == 'FAILED_JOBS_DIAGNOSTICALLY_RECOVERED'
    assert manifest['reconstruction']['phase44']['status'] == 'RECOVERED_DIAGNOSTIC'
    assert manifest['reconstruction']['phase44']['metric_rows'] == json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase44']).read_text())['metric_rows']


def test_phase43_metrics_are_complete_and_keep_counts():
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase43']).read_text())
    result = status._phase41_projection(payload, phase=43, labels=('late_adaptation_control', 'early_adaptation'))
    assert len(result['metric_rows']) == len(payload['metric_rows']) == 16405
    assert result['arms']['late_adaptation_control']['endpoints']['full_lcag']['numerator'] == 1
    assert result['arms']['early_adaptation']['endpoints']['full_lcag']['numerator'] == 2
    assert set(result['arms']['early_adaptation']['beam']) == {'full', 'half'}
    assert (ROOT / status.SOURCE_PATHS['reconstruction_phase43']).stat().st_size < status._MAX_SOURCE_BYTES


@pytest.mark.parametrize('missing', ['metric', 'ranking'])
def test_phase43_incomplete_metrics_fail_closed(missing):
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase43']).read_text())
    if missing == 'metric':
        payload['metric_rows'].pop()
    else:
        del payload['arms']['early_adaptation']['beam']['half']['greedy']
    with pytest.raises(ValueError):
        status._phase41_projection(payload, phase=43, labels=('late_adaptation_control', 'early_adaptation'))


def test_phase44_recovery_is_explicit_and_metrics_are_lossless():
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase44']).read_text())
    result = status._phase41_projection(payload, phase=44, labels=('late_adaptation_control', 'early_adaptation'))
    assert result['status'] == 'RECOVERED_DIAGNOSTIC'
    assert result['original_job_status'] == 'FAILED'
    assert result['recovery_classification'] == 'POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION'
    assert result['metric_rows'] == payload['metric_rows']
    for arm in ('late_adaptation_control', 'early_adaptation'):
        assert result['arms'][arm]['endpoints'] == payload['arms'][arm]['endpoints']
        assert set(result['arms'][arm]['beam']) == {'full', 'half'}
    assert (ROOT / status.SOURCE_PATHS['reconstruction_phase44']).stat().st_size < status._MAX_SOURCE_BYTES


@pytest.mark.parametrize('change', ['status', 'original_job_status', 'recovery_classification', 'metric', 'ranking'])
def test_phase44_incomplete_or_misclassified_recovery_cannot_publish(change):
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase44']).read_text())
    if change == 'metric':
        payload['metric_rows'].pop()
    elif change == 'ranking':
        del payload['arms']['early_adaptation']['beam']['half']['greedy']
    else:
        payload[change] = 'COMPLETED'
    with pytest.raises(ValueError):
        status._phase41_projection(payload, phase=44, labels=('late_adaptation_control', 'early_adaptation'))


def test_phase44_retained_download_is_complete_and_separately_indexed(tmp_path):
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase44_retained']).read_text())
    projected = status._phase44_retained_projection(payload)
    assert projected['metric_rows'] == payload['metric_rows']
    output = tmp_path / 'retained-status'
    manifest = status.generate_status(ROOT, output)
    record = manifest['reconstruction']['phase44']['retained_tree_checks']
    binding = record['metric_download']
    data = (output / binding['filename']).read_bytes()
    import hashlib
    assert hashlib.sha256(data).hexdigest() == binding['sha256']
    assert len(data) == binding['bytes'] < 10 * 1024 * 1024
    assert json.loads(data)['metric_rows'] == payload['metric_rows']
    assert binding['metric_count'] == len(payload['metric_rows'])
    assert 'metric_rows' not in record
    assert (ROOT / status.SOURCE_PATHS['reconstruction_phase44_retained']).stat().st_size < status._MAX_SOURCE_BYTES


@pytest.mark.parametrize('change', ['missing_row', 'duplicate_row', 'beam_unchecked', 'legacy_changed'])
def test_phase44_retained_incomplete_metrics_cannot_publish(change):
    payload = json.loads((ROOT / status.SOURCE_PATHS['reconstruction_phase44_retained']).read_text())
    if change == 'missing_row': payload['metric_rows'].pop()
    if change == 'duplicate_row': payload['metric_rows'].append(payload['metric_rows'][0])
    if change == 'beam_unchecked': payload['all_returned_beam_candidates_checked'] = False
    if change == 'legacy_changed': payload['legacy_metrics_unchanged'] = False
    with pytest.raises(ValueError):
        status._phase44_retained_projection(payload)


def test_phase45_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase45']
    assert record['source_boundary'] == 'PRE_SCIENTIFIC_AUDIT_FIXES'
    for source_key, projected in [('reconstruction_phase45',record),('reconstruction_phase45_retained',record['retained_tree_checks'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['encoder_lr005_control']['beam_candidate_count'] == 37
    assert record['retained_tree_checks']['arms']['encoder_lr010']['beam_candidate_count'] == 35
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase45_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase45_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase45_retained_projection(raw)


def test_phase45_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase45']).read_text())
    raw['source_boundary'] = 'corrected_scientific_audit_source'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=45,labels=('encoder_lr005_control','encoder_lr010'))


def test_phase46_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase46']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase46',record),('reconstruction_phase46_retained',record['retained_tree_checks'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 40
    assert record['retained_tree_checks']['arms']['frozen_encoder']['beam_candidate_count'] == 39
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase46_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase46_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase46_retained_projection(raw)


def test_phase46_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase46']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=46,labels=('late_adaptation_control','frozen_encoder'))


def test_phase47_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase47']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase47',record),('reconstruction_phase47_retained',record['retained_tree_checks']),('reconstruction_phase47_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 42
    assert record['retained_tree_checks']['arms']['frozen_encoder']['beam_candidate_count'] == 42
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase47_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase47_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase47_retained_projection(raw)


def test_phase47_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase47']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=47,labels=('late_adaptation_control','frozen_encoder'))


def test_phase48_all_tree_and_beam_publication(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path/'phase48-status')
    phase = manifest['reconstruction']['phase48']
    assert phase['status'] == 'COMPLETED'
    assert phase['metric_download']['metric_count'] == 15936
    retained = phase['retained_tree_checks']
    assert retained['metric_download']['metric_count'] == 19316
    assert retained['tree_metric_scalar_rows'] == 8953020
    assert sum(a['beam_candidate_count'] for a in retained['arms'].values()) == 92
    assert manifest['reconstruction']['phase47']['next_study_status'] == 'COMPLETED'
    page = (tmp_path/'phase48-status/index.rst').read_text()
    assert 'PID adaptation did not execute' in page
    assert 'not evidence that effective PID adaptation has no benefit' in page


def test_phase49_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase49']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase49',record),('reconstruction_phase49_retained',record['retained_tree_checks']),('reconstruction_phase49_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 61
    assert record['retained_tree_checks']['arms']['late_pid_adaptation']['beam_candidate_count'] == 60
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase49_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase49_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase49_retained_projection(raw)


def test_phase49_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase49']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=49,labels=('late_adaptation_control','late_pid_adaptation'))


def test_phase50_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase50']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase50',record),('reconstruction_phase50_retained',record['retained_tree_checks']),('reconstruction_phase50_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 65
    assert record['retained_tree_checks']['arms']['late_pid_adaptation']['beam_candidate_count'] == 64
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase50_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase50_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase50_retained_projection(raw)


def test_phase50_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase50']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=50,labels=('late_adaptation_control','late_pid_adaptation'))


def test_phase51_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase51']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase51',record),('reconstruction_phase51_retained',record['retained_tree_checks']),('reconstruction_phase51_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 65
    assert record['retained_tree_checks']['arms']['late_pid_adaptation']['beam_candidate_count'] == 59
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase51_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase51_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase51_retained_projection(raw)


def test_phase51_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase51']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=51,labels=('late_adaptation_control','late_pid_adaptation'))


def test_phase52_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase52']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase52',record),('reconstruction_phase52_retained',record['retained_tree_checks']),('reconstruction_phase52_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 58
    assert record['retained_tree_checks']['arms']['stronger_recovery']['beam_candidate_count'] == 60
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase52_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase52_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase52_retained_projection(raw)


def test_phase52_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase52']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=52,labels=('late_adaptation_control','stronger_recovery'))


def test_phase53_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase53']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase53',record),('reconstruction_phase53_retained',record['retained_tree_checks']),('reconstruction_phase53_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 72
    assert record['retained_tree_checks']['arms']['weaker_recovery']['beam_candidate_count'] == 69
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase53_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase53_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase53_retained_projection(raw)


def test_phase53_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase53']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=53,labels=('late_adaptation_control','weaker_recovery'))


def test_status_compact_numbers_preserve_values_and_negative_zero():
    value = {'counts': [0.0, 17.0, -2.0, 1.25, -0.0, float(2**54)]}
    result = status._compact_exact_numbers(value)
    assert result == value
    assert result['counts'][:3] == [0, 17, -2]
    assert all(type(x) is int for x in result['counts'][:3])
    assert json.dumps(result['counts'][4]) == '-0.0'
    assert type(result['counts'][5]) is float


def test_phase54_full_metric_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT, tmp_path / 'status')
    record = manifest['reconstruction']['phase54']
    assert record['source_boundary'] == 'CORRECTED_SCIENTIFIC_AUDIT_SOURCE'
    for source_key, projected in [('reconstruction_phase54',record),('reconstruction_phase54_retained',record['retained_tree_checks']),('reconstruction_phase54_aggregation',record['aggregation_supplement'])]:
        raw = json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding = projected['metric_download']
        content = (tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == binding['sha256']
        assert len(content) == binding['bytes']
        assert json.loads(content)['metric_rows'] == raw['metric_rows']
        assert binding['metric_count'] == len(raw['metric_rows'])
    assert record['retained_tree_checks']['arms']['late_adaptation_control']['beam_candidate_count'] == 65
    assert record['retained_tree_checks']['arms']['weaker_recovery']['beam_candidate_count'] == 59
    assert (tmp_path/'status/status.json').stat().st_size < 10 * 1024 * 1024


@pytest.mark.parametrize('change',['missing','duplicate','beam'])
def test_phase54_incomplete_retained_metrics_cannot_publish(change):
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase54_retained']).read_text())
    if change == 'missing': raw['metric_rows'].pop()
    if change == 'duplicate': raw['metric_rows'].append(raw['metric_rows'][0])
    if change == 'beam': raw['all_returned_beam_candidates_checked'] = False
    with pytest.raises(ValueError):
        status._phase54_retained_projection(raw)


def test_phase54_source_boundary_cannot_be_silently_relabelled():
    raw = json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase54']).read_text())
    raw['source_boundary'] = 'pre_scientific_audit_fixes'
    with pytest.raises(ValueError, match='source boundary'):
        status._phase41_projection(raw,phase=54,labels=('late_adaptation_control','weaker_recovery'))



def test_phase55_recovered_metrics_and_pretraining_downloads_are_lossless(tmp_path):
    manifest = status.generate_status(ROOT,tmp_path/'status')
    record = manifest['reconstruction']['phase55']
    assert record['original_job_status'] == 'FAILED_POST_TRAINING_EVALUATION'
    assert manifest['reconstruction']['phase54']['next_study_status'] == 'COMPLETED'
    for source_key, projected in [('reconstruction_phase55',record),('reconstruction_phase55_retained',record['retained_tree_checks']),('reconstruction_phase55_aggregation',record['aggregation_supplement']),('reconstruction_phase55_pretraining',record['pretraining_metrics'])]:
        raw=json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        binding=projected['metric_download'];content=(tmp_path/'status'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest()==binding['sha256']
        assert len(content)==binding['bytes']
        assert json.loads(content)['metric_rows']==raw['metric_rows']
        assert binding['metric_count']==len(raw['metric_rows'])
    assert (tmp_path/'status/status.json').stat().st_size < 10*1024*1024


@pytest.mark.parametrize('change',['missing','duplicate','beam','job_status'])
def test_phase55_incomplete_or_relabelled_recovery_cannot_publish(change):
    raw=json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase55_retained']).read_text())
    if change=='missing':raw['metric_rows'].pop()
    if change=='duplicate':raw['metric_rows'].append(raw['metric_rows'][0])
    if change=='beam':raw['all_returned_beam_candidates_checked']=False
    if change=='job_status':raw['original_job_status']='COMPLETED'
    with pytest.raises(ValueError):status._phase55_retained_projection(raw)


def test_phase55_original_source_boundary_cannot_be_relabelled():
    raw=json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase55']).read_text())
    raw['source_boundary']='corrected_scientific_audit_source'
    with pytest.raises(ValueError,match='source boundary'):
        status._phase41_projection(raw,phase=55,labels=('late_adaptation_control','stronger_parent_pretraining'))


def test_phase55_missing_pretraining_metric_cannot_publish():
    raw=json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase55_pretraining']).read_text())
    raw['metric_rows'].pop()
    with pytest.raises(ValueError):status._phase55_pretraining_projection(raw)


def test_phase56_available_metrics_are_lossless_and_missing_control_is_null(tmp_path):
    manifest=status.generate_status(ROOT,tmp_path/'phase56')
    record=manifest['reconstruction']['phase56']
    assert record['status']=='INCOMPLETE_COMPARISON' and record['comparison_available'] is False
    assert record['arms']['late_adaptation_control']['endpoints'] is None
    for source_key,projected in [('reconstruction_phase56',record),('reconstruction_phase56_retained',record['retained_tree_checks']),('reconstruction_phase56_aggregation',record['aggregation_supplement']),('reconstruction_phase56_pretraining',record['pretraining_metrics'])]:
        binding=projected['metric_download'];content=(tmp_path/'phase56'/binding['filename']).read_bytes()
        assert hashlib.sha256(content).hexdigest()==binding['sha256']
        raw=json.loads((ROOT/status.SOURCE_PATHS[source_key]).read_text())
        assert json.loads(content)['metric_rows']==raw['metric_rows']


@pytest.mark.parametrize('change',['complete','comparison','fabricated_control','missing_row'])
def test_phase56_rejects_false_comparison_or_incomplete_export(change):
    raw=json.loads((ROOT/status.SOURCE_PATHS['reconstruction_phase56']).read_text())
    if change=='complete':raw['status']='COMPLETED'
    if change=='comparison':raw['comparison_available']=True
    if change=='fabricated_control':raw['arms']['late_adaptation_control']['endpoints']={}
    if change=='missing_row':raw['metric_rows'].pop()
    with pytest.raises(ValueError):status._phase56_projection(raw)
