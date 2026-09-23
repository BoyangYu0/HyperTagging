"""Project selected local metadata into a deterministic, privacy-safe dashboard.

All published fields are validated scalars, fixed labels, counts, or hashes.
Unknown fields and source locations/bodies are never published or copied.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import string
import subprocess
from typing import Any

import yaml

# Private input-only addresses. Public provenance uses opaque SOURCE_IDS.
SOURCE_PATHS = {
    "current_status": "docs/audits/current_status.md",
    "issue_ledger": "docs/audits/issue_ledger.yaml",
    "verification_runs": "docs/audits/verification_runs.yaml",
    "notebook_registry": "notebooks/index.yaml",
    "pretraining_contract": "configs/batch_efficiency/ht_pretraining_1m_phase3_device_profiles_v1.json",
    "pretraining_selection": "artifacts/codex/ht_pretraining_1m_phase3_batch_efficiency_selection_20260823.json",
    "transfer_preregistration": "configs/reconstruction/ht_reconstruction_transfer_preregistration_20260824.json",
    "reconstruction_terminal": "artifacts/codex/joint_optimization_20260824/reconstruction_stage_a_paired_relbias_16036157_terminal_audit_20260827.json",
    "reconstruction_phase40r1": "artifacts/codex/reconstruction_phase40r1_closeout_20260909.json",
    "cpu_workflow": ".github/workflows/cpu-tests.yml",
    "reconstruction_phase41": "artifacts/codex/reconstruction_phase41_closeout_20260909.json",
    "reconstruction_phase42_submission": "artifacts/codex/reconstruction_phase42_submission_20260909.json",
}
SOURCE_PATHS.update({
    "reconstruction_phase42": "artifacts/codex/reconstruction_phase42_closeout_20260910.json",
    "reconstruction_phase43_submission": "artifacts/codex/reconstruction_phase43_submission_20260910.json",
    "reconstruction_phase43": "artifacts/codex/reconstruction_phase43_closeout_20260911.json",
    "reconstruction_phase44_submission": "artifacts/codex/reconstruction_phase44_submission_20260911.json",
})
SOURCE_PATHS.update({
    "reconstruction_phase44": "artifacts/codex/reconstruction_phase44_closeout_20260911.json",
    "reconstruction_phase45_submission": "artifacts/codex/reconstruction_phase45_submission_20260911.json",
    "reconstruction_phase44_retained": "artifacts/codex/reconstruction_phase44_retained_metrics_20260911.json",
})
SOURCE_PATHS.update({
    "reconstruction_phase45": "artifacts/codex/reconstruction_phase45_closeout_20260912.json",
    "reconstruction_phase45_retained": "artifacts/codex/reconstruction_phase45_retained_metrics_20260912.json",
    "reconstruction_phase46_submission": "artifacts/codex/reconstruction_phase46_submission_20260912.json",
    "reconstruction_phase46": "artifacts/codex/reconstruction_phase46_closeout_20260913.json",
    "reconstruction_phase46_retained": "artifacts/codex/reconstruction_phase46_retained_metrics_20260913.json",
    "reconstruction_phase47_submission": "artifacts/codex/reconstruction_phase47_submission_20260913.json",
    "reconstruction_phase47": "artifacts/codex/reconstruction_phase47_closeout_20260914.json",
    "reconstruction_phase47_retained": "artifacts/codex/reconstruction_phase47_retained_metrics_20260914.json",
    "reconstruction_phase48_submission": "artifacts/codex/reconstruction_phase48_submission_20260914.json",
    "reconstruction_phase47_aggregation": "artifacts/codex/reconstruction_phase47_aggregation_20260914.json",
    "reconstruction_phase48": "artifacts/codex/reconstruction_phase48_closeout_20260914.json",
    "reconstruction_phase48_retained": "artifacts/codex/reconstruction_phase48_retained_metrics_20260914.json",
    "reconstruction_phase48_aggregation": "artifacts/codex/reconstruction_phase48_aggregation_20260914.json",
    "reconstruction_phase49_submission": "artifacts/codex/reconstruction_phase49_submission_20260914.json",
    "reconstruction_phase49": "artifacts/codex/reconstruction_phase49_closeout_20260915.json",
    "reconstruction_phase49_retained": "artifacts/codex/reconstruction_phase49_retained_metrics_20260915.json",
    "reconstruction_phase49_aggregation": "artifacts/codex/reconstruction_phase49_aggregation_20260915.json",
    "reconstruction_phase50_submission": "artifacts/codex/reconstruction_phase50_submission_20260915.json",
    "reconstruction_phase50": "artifacts/codex/reconstruction_phase50_closeout_20260916.json",
    "reconstruction_phase50_retained": "artifacts/codex/reconstruction_phase50_retained_metrics_20260916.json",
    "reconstruction_phase50_aggregation": "artifacts/codex/reconstruction_phase50_aggregation_20260916.json",
    "reconstruction_phase51_submission": "artifacts/codex/reconstruction_phase51_submission_20260916.json",
    "reconstruction_phase51": "artifacts/codex/reconstruction_phase51_closeout_20260917.json",
    "reconstruction_phase51_retained": "artifacts/codex/reconstruction_phase51_retained_metrics_20260917.json",
    "reconstruction_phase51_aggregation": "artifacts/codex/reconstruction_phase51_aggregation_20260917.json",
    "reconstruction_phase52_submission": "artifacts/codex/reconstruction_phase52_submission_20260917.json",
    "reconstruction_phase52": "artifacts/codex/reconstruction_phase52_closeout_20260917.json",
    "reconstruction_phase52_retained": "artifacts/codex/reconstruction_phase52_retained_metrics_20260917.json",
    "reconstruction_phase52_aggregation": "artifacts/codex/reconstruction_phase52_aggregation_20260917.json",
    "reconstruction_phase53_submission": "artifacts/codex/reconstruction_phase53_submission_20260917.json",
    "reconstruction_phase53": "artifacts/codex/reconstruction_phase53_closeout_20260918.json",
    "reconstruction_phase53_retained": "artifacts/codex/reconstruction_phase53_retained_metrics_20260918.json",
    "reconstruction_phase53_aggregation": "artifacts/codex/reconstruction_phase53_aggregation_20260918.json",
    "reconstruction_phase54_submission": "artifacts/codex/reconstruction_phase54_submission_20260918.json",
    "reconstruction_phase54": "artifacts/codex/reconstruction_phase54_closeout_20260920.json",
    "reconstruction_phase54_retained": "artifacts/codex/reconstruction_phase54_retained_metrics_20260920.json",
    "reconstruction_phase54_aggregation": "artifacts/codex/reconstruction_phase54_aggregation_20260920.json",
    "reconstruction_phase55_submission": "artifacts/codex/reconstruction_phase55_submission_20260920.json",
    "reconstruction_phase55": "artifacts/codex/reconstruction_phase55_closeout_20260921.json",
    "reconstruction_phase55_retained": "artifacts/codex/reconstruction_phase55_retained_metrics_20260921.json",
    "reconstruction_phase55_aggregation": "artifacts/codex/reconstruction_phase55_aggregation_20260921.json",
    "reconstruction_phase55_pretraining": "artifacts/codex/reconstruction_phase55_pretraining_20260921.json",
    "reconstruction_phase56_submission": "artifacts/codex/reconstruction_phase56_submission_20260921.json",
    "reconstruction_phase56": "artifacts/codex/reconstruction_phase56_closeout_20260922.json",
    "reconstruction_phase56_retained": "artifacts/codex/reconstruction_phase56_retained_metrics_20260922.json",
    "reconstruction_phase56_aggregation": "artifacts/codex/reconstruction_phase56_aggregation_20260922.json",
    "reconstruction_phase56_pretraining": "artifacts/codex/reconstruction_phase56_pretraining_20260922.json",
    "reconstruction_phase57_submission": "artifacts/codex/reconstruction_phase57_submission_20260922.json",
    "reconstruction_phase57": "artifacts/codex/reconstruction_phase57_closeout_20260923.json",
    "reconstruction_phase57_retained": "artifacts/codex/reconstruction_phase57_retained_metrics_20260923.json",
    "reconstruction_phase57_aggregation": "artifacts/codex/reconstruction_phase57_aggregation_20260923.json",
    "reconstruction_phase57_pretraining": "artifacts/codex/reconstruction_phase57_pretraining_20260923.json",
    "reconstruction_phase58_submission": "artifacts/codex/reconstruction_phase58_submission_20260923.json",
})
SOURCE_IDS = {key: f"source-{index:02d}" for index, key in enumerate(SOURCE_PATHS, 1)}
FRESHNESS_DAYS = 30
_MAX_SOURCE_BYTES = 5_000_000
_ISSUE_STATUSES = {"FIXED_AND_TESTED", "IMPLEMENTED_NOT_REAL_VERIFIED", "INTENTIONALLY_DEFERRED_SCIENCE", "OBSOLETE_OR_DUPLICATE", "PARTIAL"}
_NOTEBOOK_GROUPS = {"CORE_CONTRACT", "EXTENDED_ENGINEERING", "HISTORICAL_COMPATIBILITY", "DIAGNOSTIC", "EXTERNAL_SCIENTIFIC"}
_VERIFICATION_GROUPS = ("generated_consistency", "default_fixture", "first_level_diagnostic", "focused_production_notebooks", "real_mdst_pilot", "trained_physics_validation")


def _git(root: Path, *arguments: str) -> bytes | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *arguments], check=False, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=10, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _enum(value: Any, allowed: set[str], default: str = "UNKNOWN") -> str:
    return value if isinstance(value, str) and value in allowed else default


def _integer(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1_000_000_000 else None


def _boolean(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _sha(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", value) else None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, (date, datetime)):
        value = value.isoformat()
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?)?", value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _date_text(value: Any) -> str | None:
    parsed = _timestamp(value)
    return parsed.isoformat() if parsed is not None else None


def _freshness(recorded: Any, reference: Any) -> dict[str, Any]:
    recorded_at, reference_at = _timestamp(recorded), _timestamp(reference)
    if recorded_at is None or reference_at is None:
        return {"status": "unknown", "age_days": None}
    age = (reference_at.date() - recorded_at.date()).days
    return {"status": "after_reference_date" if age < 0 else "stale" if age > FRESHNESS_DAYS else "recent", "age_days": age}


def _recorded_date(payload: Any) -> tuple[str | None, str]:
    if isinstance(payload, dict):
        for key in ("status_date", "audit_generated_at", "study_date", "date"):
            if value := _date_text(payload.get(key)):
                return value, "explicit_date_field"
        dates = [_date_text(run.get("date")) for run in _list(payload.get("runs")) if isinstance(run, dict)]
        if dates := [value for value in dates if value]:
            return max(dates), "latest_dated_record"
        match = re.match(r"^(\d{4}-\d{2}-\d{2})\.", str(payload.get("audit_version", "")))
        if match:
            return _date_text(match.group(1)), "audit_version_date"
    elif isinstance(payload, str):
        dates = [_date_text(value) for value in re.findall(r"(?m)^#+[^\n]*\b(\d{4}-\d{2}-\d{2})\b", payload)]
        if dates := [value for value in dates if value]:
            return max(dates), "latest_heading_date_not_whole_document_verification"
    return None, "unknown"


def _json_value(value: Any) -> Any:
    # Private parsing/finite-value check only; never publish source subtrees.
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _write(path: Path, payload: bytes) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Refusing symlink documentation output")
    if path.exists() and path.stat().st_nlink != 1:
        raise ValueError("Refusing multiply linked documentation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)


def _load_source(root: Path, key: str, reference: str | None) -> tuple[dict[str, Any], Any]:
    path = SOURCE_PATHS[key]
    source = root / path
    info = {"source_id": SOURCE_IDS[key], "availability": "missing", "sha256": None, "recorded_date": None,
            "recorded_date_basis": "unknown", "freshness": {"status": "unknown", "age_days": None},
            "last_commit": None, "last_commit_date": None, "worktree_state": "unknown"}
    if any(part.is_symlink() for part in (source, *source.parents)):
        info["availability"] = "unsafe_path"
        return info, None
    if not source.is_file():
        return info, None
    try:
        if source.stat().st_size > _MAX_SOURCE_BYTES:
            raise ValueError("Evidence metadata exceeds size limit")
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        payload = json.loads(text) if source.suffix == ".json" else yaml.safe_load(text) if source.suffix in {".yaml", ".yml"} else text
        payload = _json_value(payload)
        if not isinstance(payload, str if source.suffix == ".md" else dict):
            raise ValueError("Evidence must be a document or mapping")
        json.dumps(payload, allow_nan=False)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, yaml.YAMLError):
        info["availability"] = "invalid"
        return info, None
    recorded, basis = _recorded_date(payload)
    digest = hashlib.sha256(raw).hexdigest()
    info.update(availability="present", sha256=digest, recorded_date=recorded, recorded_date_basis=basis,
                freshness=_freshness(recorded, reference))
    committed = _git(root, "show", f"HEAD:{path}")
    if committed is not None:
        info["worktree_state"] = "same_as_head" if hashlib.sha256(committed).hexdigest() == digest else "modified_from_head"
    elif _git(root, "rev-parse", "--verify", "HEAD"):
        info["worktree_state"] = "untracked_at_head"
    history = _git(root, "log", "-1", "--format=%H%n%cI", "--", path)
    lines = history.decode("utf-8", errors="replace").splitlines() if history else []
    if len(lines) >= 2:
        info.update(last_commit=_sha(lines[0]), last_commit_date=_date_text(lines[1]))
    return info, payload


def _revision_match(recorded: Any, current: str | None) -> str:
    recorded = _sha(recorded)
    return "unknown" if recorded is None or current is None else "same_revision" if recorded == current else "different_revision"


def _refs(*keys: str) -> list[str]:
    return [SOURCE_IDS[key] for key in keys]


def _notebook_record(value: Any) -> dict[str, Any]:
    record = _mapping(value)
    return {"result": _enum(record.get("result"), {"PASS", "FAIL", "NOT_RUN"}), "count": _integer(record.get("count"))}


def _ci_projection(payload: Any) -> dict[str, Any]:
    workflow = _mapping(payload)
    unprotected, pipelines = 0, 0
    default_shell = _mapping(_mapping(workflow.get("defaults")).get("run")).get("shell")
    for job in _mapping(workflow.get("jobs")).values():
        job = _mapping(job)
        job_shell = _mapping(_mapping(job.get("defaults")).get("run")).get("shell", default_shell)
        for step in _list(job.get("steps")):
            step = _mapping(step)
            command = step.get("run")
            if not isinstance(command, str) or not re.search(r"\|\s*tee\b", command):
                continue
            pipelines += 1
            shell = step.get("shell", job_shell)
            # GitHub's explicit bash template has pipefail; implicit bash does not.
            protected = shell == "bash" or (isinstance(shell, str) and "pipefail" in shell)
            protected = protected or bool(re.search(r"\bset\s+-[a-z]*o\s+pipefail\b", command))
            unprotected += not protected
    return {"source_ids": _refs("cpu_workflow"), "tee_pipeline_steps": pipelines if workflow else None,
            "without_explicit_pipefail": unprotected if workflow else None,
            "evidence_limitation": "TEE_WITHOUT_PIPEFAIL" if unprotected else "NONE_DETECTED" if workflow else "UNKNOWN",
            "live_ci_result": "NOT_QUERIED"}


def _collect(payloads: dict[str, Any], revision: str | None) -> dict[str, Any]:
    current = payloads.get("current_status")
    recommendation = re.search(
        r"(?m)^## (?:Recommendation|Historical production recommendation at the generated audit boundary):\s*(NO[-_]GO|GO)\s*$",
        current,
    ) if isinstance(current, str) else None
    ledger = _mapping(payloads.get("issue_ledger"))
    items = [item for item in _list(ledger.get("items")) if isinstance(item, dict)]
    counts = Counter(_enum(item.get("current_status"), _ISSUE_STATUSES) for item in items)
    audit = {"source_ids": _refs("current_status", "issue_ledger"),
             "recommendation": recommendation.group(1).replace("-", "_") if recommendation else "UNKNOWN",
             "audited_source_sha": _sha(ledger.get("audited_code_sha")),
             "revision_match": _revision_match(ledger.get("audited_code_sha"), revision),
             "recorded_at": _date_text(ledger.get("audit_generated_at")),
             "ledger_item_count": len(items) if isinstance(ledger.get("items"), list) else None,
             "ledger_status_counts": dict(sorted(counts.items()))}
    run_payload = _mapping(payloads.get("verification_runs"))
    runs = [run for run in _list(run_payload.get("runs")) if isinstance(run, dict)]
    dated = [run for run in runs if _timestamp(run.get("date")) is not None]
    latest = max(dated, key=lambda run: _timestamp(run["date"])) if dated else {}
    pytest, notebook_runs = _mapping(latest.get("pytest")), _mapping(latest.get("notebooks"))
    verification = {"source_ids": _refs("verification_runs"),
                    "record_count": len(runs) if isinstance(run_payload.get("runs"), list) else None,
                    "latest_record": {"date": _date_text(latest.get("date")), "source_sha": _sha(latest.get("source_sha")),
                        "revision_match": _revision_match(latest.get("source_sha"), revision),
                        "pytest": {"result": _enum(pytest.get("result"), {"PASS", "FAIL"}),
                            **{key: _integer(pytest.get(key)) for key in ("passed", "failed", "skipped", "warnings")},
                            "duration_seconds": _number(pytest.get("duration_seconds"))},
                        "notebooks": {key: _notebook_record(notebook_runs.get(key)) for key in _VERIFICATION_GROUPS if key in notebook_runs},
                        "human_visual_review_status": _enum(latest.get("human_visual_review_status"), {"NOT_REVIEWED", "REVIEWED"})},
                    "evidence_scope": "RECORDED_CPU_SOFTWARE_FIXTURES", "current_build_test_result": "NOT_RUN_BY_DASHBOARD"}
    registry = _mapping(payloads.get("notebook_registry"))
    entries = [item for item in _list(registry.get("notebooks")) if isinstance(item, dict)]
    groups = Counter(_enum(item.get("group"), _NOTEBOOK_GROUPS) for item in entries)
    modes = Counter(_enum(item.get("fixture_or_real"), {"fixture", "real_only"}) for item in entries)
    notebooks = {"source_ids": _refs("notebook_registry"), "total": len(entries) if isinstance(registry.get("notebooks"), list) else None,
                 "default_smoke": sum(item.get("default_smoke") is True for item in entries) if registry else None,
                 "groups": dict(sorted(groups.items())), "input_modes": dict(sorted(modes.items())),
                 "not_run": sum(item.get("last_verified_sha") == "NOT_RUN" for item in entries) if registry else None,
                 "visual_review_status": _enum(registry.get("visual_review_status"), {"NOT_REVIEWED", "REVIEWED"}),
                 "revision_match": _revision_match(registry.get("verification_base_head"), revision)}
    contract = _mapping(_mapping(payloads.get("pretraining_contract")).get("scientific_contract"))
    transfer = _mapping(payloads.get("transfer_preregistration"))
    step = _integer(contract.get("resume_step"))
    milestones = [_integer(value) for value in _list(contract.get("validation_milestones_virtual_steps"))]
    milestones_valid = bool(milestones) and all(value is not None for value in milestones)
    milestones_valid = milestones_valid and milestones == sorted(set(milestones))
    total_presentations, step_presentations = _integer(contract.get("total_presentations")), _integer(contract.get("virtual_step_presentations"))
    derived_total = total_presentations // step_presentations if total_presentations is not None and step_presentations and total_presentations % step_presentations == 0 else None
    milestone_total = milestones[-1] if milestones_valid else None
    total_steps = milestone_total if milestone_total is not None and milestone_total == derived_total else None
    fraction = step / total_steps if step is not None and total_steps and step <= total_steps else None
    selection = _mapping(payloads.get("pretraining_selection"))
    candidates = list(_mapping(selection.get("candidates")).values())
    pending = sum(_mapping(item).get("status") == "calibration_pending" for item in candidates)
    selected_profile = selection.get("selected_profile")
    selected_profile_state = (
        "UNKNOWN" if "selected_profile" not in selection
        else "NONE_SELECTED" if selected_profile is None
        else "SELECTED_REDACTED" if isinstance(selected_profile, str)
        else "UNKNOWN"
    )
    pretraining = {"source_ids": _refs("pretraining_contract", "pretraining_selection", "transfer_preregistration"),
                   "recorded_step": step, "planned_steps": total_steps, "progress_fraction": fraction,
                   "progress_basis": "SCIENTIFIC_CONTRACT_RESUME_STEP_VS_FINAL_VALIDATION_MILESTONE",
                   "calibration_status": "PENDING" if pending else "UNKNOWN", "calibration_candidates": len(candidates) if candidates else None,
                   "calibration_pending": pending if candidates else None,
                   "selected_profile_state": selected_profile_state,
                   "production_submission_authorized": _boolean(selection.get("production_submission_authorized")),
                   "submission_performed": _boolean(selection.get("submission_performed")),
                   "pretraining_success_gate_passed": _boolean(transfer.get("pretraining_success_gate_passed"))}
    terminal = _mapping(payloads.get("reconstruction_terminal"))
    complete = _boolean(terminal.get("complete"))
    validation = _mapping(terminal.get("validation"))
    reconstruction = {"source_ids": _refs("reconstruction_terminal"), "stage": "A",
                      "completion": "COMPLETED" if complete is True else "INCOMPLETE" if complete is False else "UNKNOWN",
                      "recommendation": _enum(terminal.get("recommendation"), {"DO_NOT_PROMOTE", "PROMOTE"}),
                      "exploratory": _boolean(terminal.get("exploratory")), "submission_authorized": _boolean(terminal.get("submission_authorized")),
                      "evidence_class": _enum(validation.get("evidence_class"), {"aggregate_only_nonpromotable"}),
                      "edge_f1_delta": _number(_mapping(validation.get("delta_relbias_minus_q32")).get("edge_f1")),
                      "required_edge_f1_delta": _number(validation.get("required_edge_f1_delta")),
                      "paired_event_evidence": _boolean(validation.get("paired_event_evidence")),
                      "guards": _enum(validation.get("guards"), {"not_established", "established"})}
    # Fixed metric names and scalar validation keep operational receipt fields private.
    metric_names = ("edge_f1", "full_tree_exact_match", "tree_validity", "canonical_subtree",
                    "pointer_precision", "pointer_recall", "validation_loss", "query_utilization")
    reconstruction["metrics"] = {
        variant: {name: _number(_mapping(validation.get(variant)).get(name))
                  for name in metric_names}
        for variant in ("relbias", "q32")
    }
    reconstruction["cohorts"] = {
        variant: {name: _integer(_mapping(validation.get(variant)).get(name))
                  for name in ("validation_events", "rollout_events")}
        for variant in ("relbias", "q32")
    }
    reconstruction["target_gap"] = (
        reconstruction["edge_f1_delta"] - reconstruction["required_edge_f1_delta"]
        if reconstruction["edge_f1_delta"] is not None and reconstruction["required_edge_f1_delta"] is not None
        else None
    )
    recent = _mapping(payloads.get("reconstruction_phase40r1"))
    recent_metric_names = (
        "micro_complete_target_efficiency", "predicted_depth_fraction",
        "tree_validity", "full_source_recall", "full_source_precision",
        "half_source_recall", "half_source_precision",
    )
    recent_count_names = (
        "full_root_completion_numerator", "full_root_completion_denominator",
        "full_lcag_numerator", "full_lcag_denominator",
        "exact_mother_coverage_numerator", "exact_mother_coverage_denominator",
        "half_lcag_numerator", "half_lcag_denominator",
        "half_perfect_lcag_numerator", "half_perfect_lcag_denominator",
    )
    beam_ranking_names = (
        "greedy", "average_link_probability", "learned_confidence_mean",
        "learned_confidence_sum", "normalized_joint_log_probability", "oracle_at_k",
    )
    beam_metric_names = (
        "source_recall", "source_precision", "lcag_pair_accuracy",
        "mother_pid_coverage", "perfect_lcag",
    )
    recent_arms = {
        arm: {
            "selected_step": _integer(_mapping(recent.get(arm)).get("selected_step")),
            "all_gates_passed": _boolean(_mapping(recent.get(arm)).get("all_gates_passed")),
            **{name: _number(_mapping(recent.get(arm)).get(name)) for name in recent_metric_names},
            **{name: _integer(_mapping(recent.get(arm)).get(name)) for name in recent_count_names},
        }
        for arm in ("control", "query_scale")
    }
    raw_rankings_by_scope = {
        "full": _mapping(recent.get("beam_rankings")),
        "half": _mapping(recent.get("beam_half_rankings")),
    }
    beam_rankings_by_scope = {
        scope: {
            ranking: {
                metric: {
                    "numerator": _integer(_mapping(_mapping(raw_rankings.get(ranking)).get(metric)).get("numerator")),
                    "denominator": _integer(_mapping(_mapping(raw_rankings.get(ranking)).get(metric)).get("denominator")),
                }
                for metric in beam_metric_names
            }
            for ranking in beam_ranking_names
        }
        for scope, raw_rankings in raw_rankings_by_scope.items()
    }
    complete_arm_metrics = all(
        recent_arms[arm][name] is not None
        for arm in recent_arms
        for name in ("selected_step", "all_gates_passed", *recent_metric_names, *recent_count_names)
    )
    complete_beam_metrics = all(
        beam_rankings_by_scope[scope][ranking][metric][field] is not None
        for scope in beam_rankings_by_scope
        for ranking in beam_rankings_by_scope[scope]
        for metric in beam_rankings_by_scope[scope][ranking]
        for field in ("numerator", "denominator")
    )
    if recent and (
        recent.get("metric_contract_version") != "reconstruction-current-best-complete-v2"
        or recent.get("metric_completeness") != "COMPLETE"
        or not complete_arm_metrics
        or not complete_beam_metrics
    ):
        raise ValueError("current-best reconstruction metric contract is incomplete")
    reconstruction["phase40r1"] = {
        "source_ids": _refs("reconstruction_phase40r1"),
        "status": _enum(recent.get("status"), {"COMPLETED"}),
        "strict_event_count": _integer(recent.get("strict_event_count")),
        "beam_event_count": _integer(recent.get("beam_event_count")),
        "primary_repeat_identical": _boolean(recent.get("primary_repeat_identical")),
        "sealed_test_accessed": _boolean(recent.get("sealed_test_accessed")),
        "promotion_authorized": _boolean(recent.get("promotion_authorized")),
        "longer_run_authorized": _boolean(recent.get("longer_run_authorized")),
        "current_best_arm": _enum(recent.get("current_best_arm"), {"CONTROL", "QUERY_SCALE"}),
        "metric_contract_version": _enum(recent.get("metric_contract_version"), {"reconstruction-current-best-complete-v2"}),
        "metric_completeness": _enum(recent.get("metric_completeness"), {"COMPLETE"}),
        "arms": recent_arms,
        "beam_rankings_by_scope": beam_rankings_by_scope,
        "decision": {
            "winning_arm": _enum(_mapping(recent.get("decision")).get("winning_arm"), {"CONTROL", "QUERY_SCALE"}),
            "query_scale_continuation": _enum(_mapping(recent.get("decision")).get("query_scale_continuation"), {"STOP", "CONTINUE"}),
            "longer_budget": _enum(_mapping(recent.get("decision")).get("longer_budget"), {"NOT_AUTHORIZED", "AUTHORIZED"}),
            "next_study": _enum(_mapping(recent.get("decision")).get("next_study"), {"LEVEL1_POINTER_BALANCE"}),
            "next_study_status": _enum(_mapping(recent.get("decision")).get("next_study_status"), {"PREREGISTRATION_IN_PROGRESS", "SUBMITTED", "RUNNING", "COMPLETED"}),
            "phase41_task_count": _integer(_mapping(recent.get("decision")).get("phase41_task_count")),
            "phase41_pointer_threshold": _number(_mapping(recent.get("decision")).get("phase41_pointer_threshold")),
            "phase41_object_threshold": _number(_mapping(recent.get("decision")).get("phase41_object_threshold")),
            "phase41_threshold_role": _enum(_mapping(recent.get("decision")).get("phase41_threshold_role"), {"PREREGISTERED_FRESH_COHORT_CANDIDATE_ONLY"}),
        },
    }
    reconstruction["phase41"] = _phase41_projection(payloads.get("reconstruction_phase41"))
    reconstruction["phase42"] = _phase41_projection(payloads.get("reconstruction_phase42"), phase=42, labels=("pretrain81096_control", "pretrain108128"))
    reconstruction["phase43"] = _phase41_projection(payloads.get("reconstruction_phase43"), phase=43, labels=("late_adaptation_control", "early_adaptation"))
    phase44 = _mapping(payloads.get("reconstruction_phase44_submission"))
    if reconstruction["phase43"] and phase44.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
        reconstruction["phase43"]["next_study_status"] = phase44["status"]
        reconstruction["phase43"]["submission_task_count"] = _integer(phase44.get("task_count"))
        reconstruction["phase43"]["submission_source_revision"] = _sha(phase44.get("source_revision"))
    reconstruction["phase44"] = _phase41_projection(payloads.get("reconstruction_phase44"), phase=44, labels=("late_adaptation_control", "early_adaptation"))
    phase45 = _mapping(payloads.get("reconstruction_phase45_submission"))
    if reconstruction["phase44"]:
        reconstruction["phase44"]["retained_tree_checks"] = _phase44_retained_projection(payloads.get("reconstruction_phase44_retained"))
        if reconstruction["phase43"]:
            reconstruction["phase43"]["next_study_status"] = "FAILED_JOBS_DIAGNOSTICALLY_RECOVERED"
        if phase45.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            reconstruction["phase44"]["next_study_status"] = phase45["status"]
            reconstruction["phase44"]["submission_task_count"] = _integer(phase45.get("task_count"))
            reconstruction["phase44"]["submission_source_revision"] = _sha(phase45.get("source_revision"))
    phase43 = _mapping(payloads.get("reconstruction_phase43_submission"))
    if reconstruction["phase42"] and phase43.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
        reconstruction["phase42"]["next_study_status"] = phase43["status"]
        reconstruction["phase42"]["submission_task_count"] = _integer(phase43.get("task_count"))
        reconstruction["phase42"]["submission_source_revision"] = _sha(phase43.get("source_revision"))
    if reconstruction["phase43"] and reconstruction["phase42"]:
        reconstruction["phase42"]["next_study_status"] = "COMPLETED"
    submission = _mapping(payloads.get("reconstruction_phase42_submission"))
    if reconstruction["phase41"] and submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
        reconstruction["phase41"]["next_study_status"] = submission["status"]
        reconstruction["phase41"]["submission_task_count"] = _integer(submission.get("task_count"))
        reconstruction["phase41"]["submission_source_revision"] = _sha(submission.get("source_revision"))
    if reconstruction["phase42"] and reconstruction["phase41"]:
        reconstruction["phase41"]["next_study_status"] = "COMPLETED"
    reconstruction["phase45"] = _phase41_projection(payloads.get("reconstruction_phase45"), phase=45, labels=("encoder_lr005_control", "encoder_lr010"))
    if reconstruction["phase45"]:
        record = reconstruction["phase45"]
        record["source_boundary"] = "PRE_SCIENTIFIC_AUDIT_FIXES"
        record["retained_tree_checks"] = _phase45_retained_projection(payloads.get("reconstruction_phase45_retained"))
        if reconstruction["phase44"]:
            reconstruction["phase44"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase46_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase46"] = _phase41_projection(payloads.get("reconstruction_phase46"), phase=46, labels=("late_adaptation_control", "frozen_encoder"))
    if reconstruction["phase46"]:
        record = reconstruction["phase46"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase46_retained_projection(payloads.get("reconstruction_phase46_retained"))
        if reconstruction["phase45"]:
            reconstruction["phase45"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase47_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase47"] = _phase41_projection(payloads.get("reconstruction_phase47"), phase=47, labels=("late_adaptation_control", "frozen_encoder"))
    if reconstruction["phase47"]:
        record = reconstruction["phase47"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase47_retained_projection(payloads.get("reconstruction_phase47_retained"))
        record["aggregation_supplement"] = _phase47_aggregation_projection(payloads.get("reconstruction_phase47_aggregation"))
        if reconstruction["phase46"]:
            reconstruction["phase46"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase48_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase48"] = _phase41_projection(payloads.get("reconstruction_phase48"), phase=48, labels=("late_adaptation_control", "late_pid_adaptation"))
    if reconstruction["phase48"]:
        record = reconstruction["phase48"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase48_retained_projection(payloads.get("reconstruction_phase48_retained"))
        record["aggregation_supplement"] = _phase48_aggregation_projection(payloads.get("reconstruction_phase48_aggregation"))
        if reconstruction["phase47"]:
            reconstruction["phase47"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase49_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase49"] = _phase41_projection(payloads.get("reconstruction_phase49"), phase=49, labels=("late_adaptation_control", "late_pid_adaptation"))
    if reconstruction["phase49"]:
        record = reconstruction["phase49"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase49_retained_projection(payloads.get("reconstruction_phase49_retained"))
        record["aggregation_supplement"] = _phase49_aggregation_projection(payloads.get("reconstruction_phase49_aggregation"))
        if reconstruction["phase48"]:
            reconstruction["phase48"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase50_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase50"] = _phase41_projection(payloads.get("reconstruction_phase50"), phase=50, labels=("late_adaptation_control", "late_pid_adaptation"))
    if reconstruction["phase50"]:
        record = reconstruction["phase50"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase50_retained_projection(payloads.get("reconstruction_phase50_retained"))
        record["aggregation_supplement"] = _phase50_aggregation_projection(payloads.get("reconstruction_phase50_aggregation"))
        if reconstruction["phase49"]:
            reconstruction["phase49"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase51_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase51"] = _phase41_projection(payloads.get("reconstruction_phase51"), phase=51, labels=("late_adaptation_control", "late_pid_adaptation"))
    if reconstruction["phase51"]:
        record = reconstruction["phase51"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase51_retained_projection(payloads.get("reconstruction_phase51_retained"))
        record["aggregation_supplement"] = _phase51_aggregation_projection(payloads.get("reconstruction_phase51_aggregation"))
        if reconstruction["phase50"]:
            reconstruction["phase50"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase52_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase52"] = _phase41_projection(payloads.get("reconstruction_phase52"), phase=52, labels=("late_adaptation_control", "stronger_recovery"))
    if reconstruction["phase52"]:
        record = reconstruction["phase52"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase52_retained_projection(payloads.get("reconstruction_phase52_retained"))
        record["aggregation_supplement"] = _phase52_aggregation_projection(payloads.get("reconstruction_phase52_aggregation"))
        if reconstruction["phase51"]:
            reconstruction["phase51"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase53_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase53"] = _phase41_projection(payloads.get("reconstruction_phase53"), phase=53, labels=("late_adaptation_control", "weaker_recovery"))
    if reconstruction["phase53"]:
        record = reconstruction["phase53"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase53_retained_projection(payloads.get("reconstruction_phase53_retained"))
        record["aggregation_supplement"] = _phase53_aggregation_projection(payloads.get("reconstruction_phase53_aggregation"))
        if reconstruction["phase52"]:
            reconstruction["phase52"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase54_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase54"] = _phase41_projection(payloads.get("reconstruction_phase54"), phase=54, labels=("late_adaptation_control", "weaker_recovery"))
    if reconstruction["phase54"]:
        record = reconstruction["phase54"]
        record["source_boundary"] = "CORRECTED_SCIENTIFIC_AUDIT_SOURCE"
        record["retained_tree_checks"] = _phase54_retained_projection(payloads.get("reconstruction_phase54_retained"))
        record["aggregation_supplement"] = _phase54_aggregation_projection(payloads.get("reconstruction_phase54_aggregation"))
        if reconstruction["phase53"]:
            reconstruction["phase53"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase55_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction["phase55"] = _phase41_projection(payloads.get("reconstruction_phase55"), phase=55, labels=("late_adaptation_control", "stronger_parent_pretraining"))
    if reconstruction["phase55"]:
        record = reconstruction["phase55"]
        record["source_boundary"] = "AUTHENTICATED_REFINED_CHECKPOINT_EVALUATION_RECOVERY"
        record["pretraining_metrics"] = _phase55_pretraining_projection(payloads.get("reconstruction_phase55_pretraining"))
        record["original_job_status"] = "FAILED_POST_TRAINING_EVALUATION"
        record["retained_tree_checks"] = _phase55_retained_projection(payloads.get("reconstruction_phase55_retained"))
        record["aggregation_supplement"] = _phase55_aggregation_projection(payloads.get("reconstruction_phase55_aggregation"))
        if reconstruction["phase54"]:
            reconstruction["phase54"]["next_study_status"] = "COMPLETED"
        submission = _mapping(payloads.get("reconstruction_phase56_submission"))
        if submission.get("status") in {"PREPARED", "SUBMITTED", "RUNNING"}:
            record["next_study_status"] = submission["status"]
            record["submission_task_count"] = _integer(submission.get("task_count"))
            record["submission_source_revision"] = _sha(submission.get("source_revision"))
    reconstruction['phase56'] = _phase56_projection(payloads.get('reconstruction_phase56'))
    if reconstruction['phase56']:
        record = reconstruction['phase56']
        record['retained_tree_checks'] = _phase56_retained_projection(payloads.get('reconstruction_phase56_retained'))
        record['aggregation_supplement'] = _phase56_aggregation_projection(payloads.get('reconstruction_phase56_aggregation'))
        record['pretraining_metrics'] = _phase56_pretraining_projection(payloads.get('reconstruction_phase56_pretraining'))
        if reconstruction['phase55']:
            reconstruction['phase55']['next_study_status'] = 'INCOMPLETE_COMPARISON'
        submission = _mapping(payloads.get('reconstruction_phase57_submission'))
        if submission.get('status') in {'PREPARED', 'SUBMITTED', 'RUNNING'}:
            record['next_study_status'] = submission['status']
            record['submission_task_count'] = _integer(submission.get('task_count'))
            record['submission_source_revision'] = _sha(submission.get('source_revision'))
    reconstruction['phase57'] = _phase41_projection(payloads.get('reconstruction_phase57'), phase=57, labels=('pretraining_balance_control','lower_late_pid_pretraining'))
    if reconstruction['phase57']:
        record=reconstruction['phase57']
        record['retained_tree_checks']=_phase57_retained_projection(payloads.get('reconstruction_phase57_retained'))
        record['aggregation_supplement']=_phase57_aggregation_projection(payloads.get('reconstruction_phase57_aggregation'))
        record['pretraining_metrics']=_phase57_pretraining_projection(payloads.get('reconstruction_phase57_pretraining'))
        if reconstruction['phase56']:reconstruction['phase56']['next_study_status']='DIAGNOSTIC_COMPLETE'
        submission=_mapping(payloads.get('reconstruction_phase58_submission'))
        if submission.get('status') in {'PREPARED','SUBMITTED','RUNNING'}:
            record['next_study_status']=submission['status']
            record['submission_task_count']=_integer(submission.get('task_count'))
            record['submission_source_revision']=_sha(submission.get('source_revision'))
    science = {"source_ids": _refs("verification_runs", "notebook_registry"),
               "real_pilot": _notebook_record(notebook_runs.get("real_mdst_pilot"))["result"],
               "trained_physics": _notebook_record(notebook_runs.get("trained_physics_validation"))["result"],
               "live_scheduler_status": "UNKNOWN_NOT_QUERIED", "live_training_status": "UNKNOWN_NOT_QUERIED"}
    return {"audit": audit, "verification": verification, "notebooks": notebooks, "pretraining": pretraining,
            "reconstruction": reconstruction, "science": science, "cpu_ci": _ci_projection(payloads.get("cpu_workflow"))}



def _phase41_projection(raw: Any, *, phase=41, labels=("pointer32_control", "level1_pointer24")) -> dict[str, Any]:
    raw = _mapping(raw)
    if raw.get("audit_version") != f"phase{phase}-closeout-v1":
        return {}
    if raw.get("metric_completeness") != "COMPLETE":
        raise ValueError("Phase41 full/half metric set is incomplete")
    endpoints = ("exact_mother_coverage", "full_lcag", "full_root_completion",
                 "full_source_precision", "full_source_recall", "half_lcag",
                 "half_perfect_lcag", "half_root_pid_accuracy", "half_source_precision", "half_source_recall")
    rankings = ("greedy", "average_link_probability", "learned_confidence_mean",
                "learned_confidence_sum", "normalized_joint_log_probability", "oracle_at_k")
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag")
    def point(raw_point):
        point = _mapping(raw_point)
        num, den = _number(point.get("numerator")), _number(point.get("denominator"))
        if num is None or den is None or num < 0 or den < 0 or num > den:
            raise ValueError("Phase41 count is missing or invalid")
        return {"numerator": int(num) if num.is_integer() else num, "denominator": int(den) if den.is_integer() else den, "value": num / den if den else None}
    result = {"source_ids": _refs(f"reconstruction_phase{phase}"), "status": "COMPLETED",
              "next_study_status": _enum(raw.get("next_study_status"), {"PREPARED", "SUBMITTED", "RUNNING"}),
              "train_events": _integer(raw.get("train_events")),
              "strict_event_count": _integer(raw.get("strict_event_count")),
              "beam_event_count": _integer(raw.get("beam_event_count")),
              "sealed_test_accessed": _boolean(raw.get("sealed_test_accessed")),
              "source_hashes": [_sha(value) for value in _list(raw.get("source_hashes"))], "arms": {}}
    if phase == 57:
        if (raw.get('status')!='DIAGNOSTIC_COMPLETE' or raw.get('independent_validation') is not False
            or raw.get('strict_selection_overlap')!=100 or raw.get('source_boundary')!='selection_contaminated_diagnostic_recovery'
            or raw.get('original_job_status')!='FAILED_POST_TRAINING_COHORT_AUDIT' or raw.get('sealed_test_accessed') is not False):
            raise ValueError('Phase57 selection contamination must remain explicit')
        if any(_mapping(v).get('all_gates_passed') is not False for v in _mapping(raw.get('arms')).values()):
            raise ValueError('Phase57 cannot certify independent gate passage')
        result.update(status='DIAGNOSTIC_COMPLETE',metric_completeness='COMPLETE',independent_validation=False,strict_selection_overlap=100,
            source_boundary='selection_contaminated_diagnostic_recovery',original_job_status='FAILED_POST_TRAINING_COHORT_AUDIT')
    if phase in (46, 47, 48, 49, 50, 51, 52, 53, 54):
        if (raw.get("status") != "COMPLETED" or raw.get("source_boundary") != "corrected_scientific_audit_source"
            or raw.get("sealed_test_accessed") is not False):
            raise ValueError("Phase46 corrected source boundary is missing")
    if phase == 55:
        if (raw.get("status") != "COMPLETED" or raw.get("source_boundary") != "authenticated_refined_checkpoint_evaluation_recovery"
            or raw.get("original_job_status") != "FAILED_POST_TRAINING_EVALUATION" or raw.get("sealed_test_accessed") is not False):
            raise ValueError("Phase55 recovered evaluation source boundary is missing")
    if phase == 45:
        if (raw.get("status") != "COMPLETED" or raw.get("source_boundary") != "pre_scientific_audit_fixes"
            or raw.get("sealed_test_accessed") is not False):
            raise ValueError("Phase45 immutable source boundary is missing")
    if phase == 44:
        if (raw.get("status") != "RECOVERED_DIAGNOSTIC"
            or raw.get("original_job_status") != "FAILED"
            or raw.get("recovery_classification") != "POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION"):
            raise ValueError("Phase44 recovery classification is missing or invalid")
        for key in ("training_history_scalar_rows", "training_history_log_records", "training_history_checkpoint_records"):
            result[key] = _integer(raw.get(key))
            if result[key] is None or result[key] <= 0:
                raise ValueError("Phase44 training history export is missing")
        result.update(status="RECOVERED_DIAGNOSTIC", original_job_status="FAILED",
                      recovery_classification="POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION")
    for arm in labels:
        record = _mapping(_mapping(raw.get("arms")).get(arm))
        selected = _mapping(record.get("selected"))
        result["arms"][arm] = {
            "optimizer_steps": _integer(record.get("optimizer_steps")),
            "training_elapsed_seconds": _number(record.get("training_elapsed_seconds")),
            "gates": {key: _boolean(_mapping(record.get("gates")).get(key)) for key in (
                "minimum_complete_target_efficiency", "minimum_depth_fraction", "minimum_full_source_precision",
                "minimum_full_source_recall", "minimum_half_lcag", "minimum_half_perfect_lcag",
                "minimum_half_root_pid_accuracy", "minimum_half_source_precision", "minimum_half_source_recall",
                "nonzero_exact_mother_coverage", "nonzero_full_lcag", "nonzero_full_root_completion",
                "primary_repeat_identical", "structural_guardrails")},
            "selected": {key: _number(selected.get(key)) for key in ("step", "micro_complete_target_efficiency", "predicted_depth_fraction", "predicted_tree_validity_rate")},
            "all_gates_passed": _boolean(record.get("all_gates_passed")),
            "primary_repeat_identical": _boolean(record.get("primary_repeat_identical")),
            "endpoints": {key: point(_mapping(record.get("endpoints")).get(key)) for key in endpoints},
            "beam": {scope: {rank: {metric: point(_mapping(_mapping(_mapping(_mapping(record.get("beam")).get(scope)).get(rank)).get(metric)))
                                      for metric in metrics} for rank in rankings} for scope in ("full", "half")}}
    # Exact authored vocabulary; arbitrary source strings and injected fields cannot publish.
    registry = json.loads(Path(__file__).with_name(f"phase{phase}_metric_registry.json").read_text())
    allowed = {tuple(row) for row in registry}
    rows = []
    seen = set()
    for row in _list(raw.get("metric_rows")):
        identity = (row.get("arm"), row.get("view"), row.get("metric"))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError("Duplicate Phase41 metric")
        seen.add(identity)
        value = row.get("value")
        if value is not None and _number(value) is None and type(value) is not bool:
            raise ValueError("Invalid Phase41 metric value")
        rows.append({"arm": identity[0], "view": identity[1], "metric": identity[2], "value": value})
    if seen != allowed:
        raise ValueError("Phase41 metric registry is incomplete")
    result["metric_rows"] = rows
    return result


def _render_phase41(record: dict[str, Any]) -> list[str]:
    if not record:
        return []
    arms = record["arms"]
    labels = ("pointer32_control", "level1_pointer24")
    lines = ["Phase41: completed, no promotion", "--------------------------------", "",
             "The control remains the reference. Lowering the level-1 pointer-positive weight",
             "trades recall for precision without improving exact hierarchy reconstruction.",
             "The single full-tree topology-matched mother has incorrect PID in both arms (0/1 PID accuracy).", "",
             "Both arms used 70,000 training events and 4,376 optimization steps. Checkpoint",
             "selection used a fresh 2,000-event cohort (1,000 rollout events); strict evaluation",
             "used a separate 100-event cohort, and beam search its fixed 20-event subset.", "",
             "All registered aggregate metrics, checkpoint tracks, calibration, PID confusion,",
             "and full/half beam rankers are included in the metric download above.", ""]
    rows = [[key, *[arms[arm]["selected"][key] for arm in labels]] for key in arms[labels[0]]["selected"]]
    rows += [[key, *[_metric_point(arms[arm]["endpoints"][key]) for arm in labels]] for key in arms[labels[0]]["endpoints"]]
    rows += [["all hierarchy gates passed", *[arms[arm]["all_gates_passed"] for arm in labels]]]
    lines += _table(["Metric", "Pointer32 control", "Level1 pointer24"], rows)
    if record.get("metric_download"):
        lines += [":download:`Complete Phase41 metric values <phase41-metrics.json>`.", ""]
    selected_metrics = {(row["arm"], row["metric"]): row["value"] for row in record["metric_rows"] if row["view"] == "training_best"}
    lines += ["Checkpoint-selection diagnostics", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "These micro teacher-forced pointer metrics use aggregated numerator/denominator",
              "counts and diagnose daughter association separately from strict rollout topology.", ""]
    lines += _table(["Level", "Control precision", "Control recall", "Pointer24 precision", "Pointer24 recall"], [
        [level, *[selected_metrics.get((arm, f"micro_level_{level}_pointer_{metric}")) for arm in labels for metric in ("precision", "recall")]]
        for level in range(1, 7)])
    lines += ["Phase41 beam comparison", "~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Model-only top-1 rankers and diagnostic oracle share the same cohort. Available-target",
              "denominators differ from event counts; unavailable targets are retained separately in the download.", ""]
    lines += _table(["Arm", "Scope", "Ranking", "Recall", "Precision", "LCAG", "Mother coverage", "Perfect LCAG"], [
        [arm, scope, rank, *[_metric_point(point) for point in points.values()]]
        for arm in labels for scope, rankings in arms[arm]["beam"].items() for rank, points in rankings.items()])
    lines += ["Decision for the next training", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Hold the dataset at 70,000 events. The earlier 35,000-to-70,000 comparison also",
              "changed optimization budget and validation cohort, so it does not isolate data scaling.",
              "Test pretraining transfer before a larger data campaign: compare pretrained steps",
              "81,096 and 108,128 with the current decoder, fixed data/budget, and a fresh cohort.",
              "Earlier studies did not establish a reliable downstream benefit from later pretraining.",
              "This study tests that hypothesis; lower pretraining loss alone is insufficient.", "",
              f"Phase42 submission snapshot: {_literal(record['next_study_status'])}. Two bounded training arms; no automatic promotion.", "",
              "Physical mother momentum resolution remains unavailable: retained truth mother",
              "four-vectors are absent. Daughter-sum closure measures an implementation invariant.",
              "Small validation cohorts and a single training seed limit conclusions.", ""]
    return lines

def _render_phase42(record: dict[str, Any]) -> list[str]:
    if not record:
        return []
    labels = ("pretrain81096_control", "pretrain108128")
    arms = record["arms"]
    lines = ["Phase42: later pretraining did not improve the primary", "------------------------------------------------------", "",
             "Both arms completed 4,376 reconstruction steps on 70,000 training events.",
             "The 81,096-step pretrained control selected 141/3,938 complete targets (3.58%);",
             "the 108,128-step candidate selected 136/3,938 (3.45%). Neither passes all hierarchy gates.",
             "These are checkpoint-selection metrics, distinct from the zero full roots out of 100 strict events.", "",
             "Selection uses 2,000 validation events, including 1,000 rollout events; strict evaluation",
             "uses a separate 100-event cohort, and beam evaluation its fixed 20-event subset.",
             "Available targets are 19/100 full units and 149/200 half units. Strict repeats match exactly.",
             "Full mother PID accuracy is 2/2 for control and 1/1 for the candidate, conditional on",
             "topology alignment; these tiny denominators do not establish high overall PID performance.", "",
             "The download above contains all registered aggregate metrics, checkpoint tracks,",
             "calibration, PID confusion counts, contracted diagnostics, and full/half beam rankings.", ""]
    rows = [[key, *[arms[a]["selected"][key] for a in labels]] for key in arms[labels[0]]["selected"]]
    rows += [[key, *[_metric_point(arms[a]["endpoints"][key]) for a in labels]] for key in arms[labels[0]]["endpoints"]]
    rows += [["all hierarchy gates passed", *[arms[a]["all_gates_passed"] for a in labels]]]
    lines += _table(["Metric", "Pretrained 81,096", "Pretrained 108,128"], rows)
    values = {(r["arm"], r["metric"]): r["value"] for r in record["metric_rows"] if r["view"] == "training_best"}
    lines += ["Phase42 pointer diagnostics", "~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Micro teacher-forced daughter association, separate from strict rollout topology.", ""]
    lines += _table(["Level", "Control precision", "Control recall", "Candidate precision", "Candidate recall"], [
        [level, *[values.get((arm, f"micro_level_{level}_pointer_{metric}")) for arm in labels for metric in ("precision", "recall")]] for level in range(1, 7)])
    lines += ["Phase42 beam comparison", "~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Oracle is a truth-assisted evaluation diagnostic, never a deployable ranking.",
              "It uses a lexicographic topology ranking, not a separate maximum for each metric.", ""]
    lines += _table(["Arm", "Scope", "Ranking", "Recall", "Precision", "LCAG", "Mother coverage", "Perfect LCAG"], [
        [arm, scope, rank, *[_metric_point(p) for p in points.values()]]
        for arm in labels for scope, rankings in arms[arm]["beam"].items() for rank, points in rankings.items()])
    lines += ["Phase43 allocation decision", "~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Hold 70,000 training events and the 81,096-step pretrained checkpoint. Do not extend",
              "pretraining duration now. Compare encoder adaptation from step zero against the",
              "current 2,188-step freeze, using identical decoder settings and 4,376 total steps.",
              "This is an exploratory test of task-specific adaptation, not an established improvement.",
              "Encoder update counts differ by design; exact training FLOPs and wall time need not match.",
              f"Phase43 submission snapshot: {_literal(record['next_study_status'])}. Two bounded arms; no automatic promotion.", "",
              "Earlier data scaling changed both data and optimization budget. Neither data limitation",
              "nor a benefit from more pretraining is established. Small cohorts and one seed limit inference.",
              "Daughter-sum closure is an implementation invariant; physical mother momentum resolution",
              "remains unavailable because retained truth mother four-vectors are absent.", "",
              "See :doc:`../../phase42` for the full study synthesis and next-training rationale.", ""]
    return lines


def _render_phase43(record: dict[str, Any]) -> list[str]:
    if not record:
        return []
    labels = ("late_adaptation_control", "early_adaptation")
    arms = record["arms"]
    lines = ["Phase43: small adaptation signal, no promotion", "------------------------------------------------", "",
             "Both arms completed 4,376 steps on 70,000 training events with the 81,096-step pretrained checkpoint.",
             "The late-adaptation control selected 136/3,592 complete targets (3.79%); early adaptation selected",
             "140/3,592 (3.90%). A four-target difference in one seed does not establish a reliable benefit.",
             "Both completed zero full roots out of 100 strict events. Control failed six gates; early adaptation failed three.",
             "Both failed full-root completion and full/half source recall. Neither is promoted.", "",
             "Selection uses 2,000 validation events, including 1,000 rollout events; strict evaluation",
             "uses a separate 100-event cohort, and beam evaluation its fixed 20-event subset.",
             "Strict repeats match exactly. All seven evaluation tracks and all aggregate metrics are in the download.", ""]
    rows = [[key, *[arms[a]["selected"][key] for a in labels]] for key in arms[labels[0]]["selected"]]
    rows += [[key, *[_metric_point(arms[a]["endpoints"][key]) for a in labels]] for key in arms[labels[0]]["endpoints"]]
    rows += [["all hierarchy gates passed", *[arms[a]["all_gates_passed"] for a in labels]]]
    lines += _table(["Metric", "Late adaptation", "Early adaptation"], rows)
    values = {(r["arm"], r["metric"]): r["value"] for r in record["metric_rows"] if r["view"] == "training_best"}
    lines += ["Phase43 pointer diagnostics", "~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Micro teacher-forced daughter association, separate from strict rollout topology.", ""]
    lines += _table(["Level", "Control precision", "Control recall", "Candidate precision", "Candidate recall"], [
        [level, *[values.get((arm, f"micro_level_{level}_pointer_{metric}")) for arm in labels for metric in ("precision", "recall")]] for level in range(1, 7)])
    lines += ["Phase43 beam comparison", "~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Oracle is a truth-assisted evaluation diagnostic, never a deployable ranking.",
              "It uses a lexicographic topology ranking, not a separate maximum for each metric.", ""]
    lines += _table(["Arm", "Scope", "Ranking", "Recall", "Precision", "LCAG", "Mother coverage", "Perfect LCAG"], [
        [arm, scope, rank, *[_metric_point(p) for p in points.values()]]
        for arm in labels for scope, rankings in arms[arm]["beam"].items() for rank, points in rankings.items()])
    lines += ["Phase44 allocation decision", "~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Hold 70,000 events and the 81,096-step pretrained checkpoint. Replicate the same encoder",
              "freeze comparison (2,188 versus zero steps) with seed 20260911 and an untouched cohort.",
              "Both arms use 4,376 total steps and unchanged strict acceptance gates. More encoder updates",
              "in the early arm are intentional; this does not guarantee equal FLOPs.",
              "Phase44 outcome: original jobs failed; diagnostic evaluation recovered." if record["next_study_status"] == "FAILED_JOBS_DIAGNOSTICALLY_RECOVERED" else f"Phase44 submission snapshot: {_literal(record['next_study_status'])}. Two bounded arms; no automatic promotion.", "",
              "Prior dataset scaling confounded data and compute; Phase42 did not support longer pretraining.",
              "Better representation learning remains a hypothesis. Replicate the adaptation signal before scaling.",
              "Physical mother momentum resolution is unavailable; daughter-sum closure is an invariant.",
              "See :doc:`../../phase43` for metric populations, study synthesis, and the next-training rationale.", ""]
    return lines




def _phase44_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase44-retained-tree-export-v1"
        or raw.get("legacy_metrics_unchanged") is not True
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "FAILED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase44 retained-tree coverage is incomplete")
    result = {"version": "phase44-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase44_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "legacy_metrics_unchanged": True, "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "early_adaptation"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase44_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result

def _phase45_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase45-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase45 retained-tree coverage is incomplete")
    result = {"version": "phase45-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase45_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("encoder_lr005_control", "encoder_lr010"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase45_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase46_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase46-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase46 retained-tree coverage is incomplete")
    result = {"version": "phase46-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase46_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "frozen_encoder"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase46_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase47_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase47-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase47 retained-tree coverage is incomplete")
    result = {"version": "phase47-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase47_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "frozen_encoder"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase47_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase47_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase47-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase47 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase47_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase47 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase47 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'frozen_encoder')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase47_aggregation')}


def _phase48_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase48-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase48 retained-tree coverage is incomplete")
    result = {"version": "phase48-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase48_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "late_pid_adaptation"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase48_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase48_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase48-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase48 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase48_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase48 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase48 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'late_pid_adaptation')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase48_aggregation')}


def _phase55_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase55-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "FAILED_POST_TRAINING_EVALUATION" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase55 retained-tree coverage is incomplete")
    result = {"version": "phase55-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase55_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "stronger_parent_pretraining"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase55_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase55_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase55-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase55 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase55_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase55 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase55 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'stronger_parent_pretraining')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase55_aggregation')}


def _phase54_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase54-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase54 retained-tree coverage is incomplete")
    result = {"version": "phase54-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase54_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "weaker_recovery"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase54_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase54_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase54-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase54 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase54_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase54 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase54 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'weaker_recovery')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase54_aggregation')}


def _render_phase54(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'weaker_recovery')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase54: Recovery-dose replication does not establish a joint benefit', '-' * 80, '',
        'Primary source-set plus mother-PID recovery is 243/3783 control versus 261/3783 weaker recovery; this is not recursive topology efficiency.',
        'Recovery weights 2 and 1 are verified; both PID heads stayed frozen with unchanged weights.',
        'Both arms fail full-source-recall and full-root-construction gates. Gate passage does not establish promotion-grade physics.', '',
        ':download:`Complete Phase54 original metric values <phase54-metrics.json>`;',
        ':download:`Complete Phase54 retained-tree metric values <phase54-retained-metrics.json>`;',
        ':download:`Phase54 micro, macro and exact source/PID/topology counts <phase54-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Recovery weight 2', 'Recovery weight 1'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Recovery weight 2', 'Recovery weight 1'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase54 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Recovery weight 2', 'Recovery weight 1'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase54 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase54 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase54 decision and uncertainty', '~' * 40, '',
        'Primary recovery rises by 18 source sets, but retained exact components fall from 12 to 11. Each arm has one coherent forest on a different continuum event; the control event has no mothers and the candidate has one shallow mother. No exact component has depth greater than one.',
        'Paired event-bootstrap intervals for retained LCAG gains include zero; they are conditional on the selected models, not training-seed uncertainty.',
        'Keep 70,000 events. Stop recovery-dose tuning and test pretraining quality with matched compute.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase55 compares parent-ranking weights 1 and 2 during 2,188-step pretraining refinement, followed by identical 4,376-step reconstruction. Both arms use 70,000 events and the same initial parameters. Pretraining quality is a hypothesis, not an established benefit.',
        f"Phase55 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'After Phase55 only 1,491 untouched validation UIDs remain; plan validation budgeting separately from training-data growth.',
        'See :doc:`../../phase54` for full interpretation and historical synthesis.', '']
    return lines

def _phase53_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase53-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase53 retained-tree coverage is incomplete")
    result = {"version": "phase53-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase53_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "weaker_recovery"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase53_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase53_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase53-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase53 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase53_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase53 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase53 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'weaker_recovery')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase53_aggregation')}


def _render_phase53(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'weaker_recovery')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase53: Lower recovery has mixed component and forest results', '-' * 70, '',
        'Primary source-set plus mother-PID recovery is 278/3807 control versus 279/3807 weaker recovery; this is not recursive topology efficiency.',
        'Recovery weights 2 and 1 are verified; both PID heads stayed frozen with unchanged weights.',
        'Control passes all original gates; weaker recovery misses full-root construction. Gate passage does not establish promotion-grade physics.', '',
        ':download:`Complete Phase53 original metric values <phase53-metrics.json>`;',
        ':download:`Complete Phase53 retained-tree metric values <phase53-retained-metrics.json>`;',
        ':download:`Phase53 micro, macro and exact source/PID/topology counts <phase53-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Recovery weight 2', 'Recovery weight 1'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Recovery weight 2', 'Recovery weight 1'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase53 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Recovery weight 2', 'Recovery weight 1'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase53 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase53 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase53 decision and uncertainty', '~' * 40, '',
        'Primary recovery rises by one source set; exact nontrivial components rise from 7 to 9, but coherent forests fall from 1 to 0. Both recover the same depth-two component.',
        'Paired event-bootstrap intervals for retained LCAG gains include zero; they are conditional on the selected models, not training-seed uncertainty.',
        'Keep 70,000 events. Replicate the mixed lower-dose result before increasing data or pretraining duration.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase54 replicates recovery-objective weights 2 and 1, with PID frozen and the encoder schedule and budget fixed.',
        f"Phase54 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase53` for full interpretation and historical synthesis.', '']
    return lines

def _phase52_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase52-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase52 retained-tree coverage is incomplete")
    result = {"version": "phase52-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase52_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "stronger_recovery"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase52_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase52_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase52-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase52 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase52_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase52 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase52 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'stronger_recovery')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase52_aggregation')}


def _render_phase52(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'stronger_recovery')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase52: Stronger recovery does not improve retained topology', '-' * 65, '',
        'Primary source-set plus mother-PID recovery is 279/3619 control versus 282/3619 stronger recovery; this is not recursive topology efficiency.',
        'Recovery weights 2 and 4 are verified; both PID heads stayed frozen with unchanged weights.',
        'Both arms pass the original gates. Gate passage does not establish promotion-grade physics.', '',
        ':download:`Complete Phase52 original metric values <phase52-metrics.json>`;',
        ':download:`Complete Phase52 retained-tree metric values <phase52-retained-metrics.json>`;',
        ':download:`Phase52 micro, macro and exact source/PID/topology counts <phase52-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Recovery weight 2', 'Recovery weight 4'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Recovery weight 2', 'Recovery weight 4'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase52 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Recovery weight 2', 'Recovery weight 4'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase52 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase52 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase52 decision and uncertainty', '~' * 40, '',
        'Primary recovery rises by three source sets; exact nontrivial components decline from 10 to 9, and coherent forests remain zero.',
        'Paired event-bootstrap intervals for retained LCAG gains include zero; they are conditional on the selected models, not training-seed uncertainty.',
        'Keep 70,000 events. Test a lower recovery dose before increasing data or pretraining duration.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase53 compares recovery-objective weights 2 and 1, with PID frozen and the encoder schedule and budget fixed.',
        f"Phase53 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase52` for full interpretation and historical synthesis.', '']
    return lines

def _phase51_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase51-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase51 retained-tree coverage is incomplete")
    result = {"version": "phase51-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase51_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "late_pid_adaptation"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase51_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase51_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase51-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase51 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase51_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase51 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase51 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'late_pid_adaptation')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase51_aggregation')}


def _render_phase51(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'late_pid_adaptation')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase51: Lower-rate PID shows no clear benefit', '-' * 65, '',
        'Primary source-set plus mother-PID recovery is 275/3597 control versus 280/3597 adapted; this is not recursive topology efficiency.',
        'The candidate PID head records 2,188 optimizer updates and changed weights; the control remains frozen.',
        'Both arms miss the full-root construction gate. Gate passage does not establish promotion-grade physics.', '',
        ':download:`Complete Phase51 original metric values <phase51-metrics.json>`;',
        ':download:`Complete Phase51 retained-tree metric values <phase51-retained-metrics.json>`;',
        ':download:`Phase51 micro, macro and exact source/PID/topology counts <phase51-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase51 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Late adaptation', 'Late PID adaptation'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase51 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase51 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase51 decision and uncertainty', '~' * 40, '',
        'Primary recovery increases slightly; retained LCAG and original half-root PID decline.',
        'Paired event-bootstrap intervals for retained LCAG gains include zero; they are conditional on the selected models, not training-seed uncertainty.',
        'Keep 70,000 events. Test recovery-objective weight before increasing data or pretraining duration.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase52 compares recovery-objective weights 2 and 4, with PID frozen and the encoder schedule and budget fixed.',
        f"Phase52 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase51` for full interpretation and historical synthesis.', '']
    return lines

def _phase50_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase50-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase50 retained-tree coverage is incomplete")
    result = {"version": "phase50-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase50_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "late_pid_adaptation"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase50_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase50_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase50-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase50 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase50_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase50 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase50 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'late_pid_adaptation')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase50_aggregation')}


def _render_phase50(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'late_pid_adaptation')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase50: PID replication shows mixed effects', '-' * 65, '',
        'Primary source-set plus mother-PID recovery is 285/3588 control versus 280/3588 adapted; this is not recursive topology efficiency.',
        'The candidate PID head records 2,188 optimizer updates and changed weights; the control remains frozen.',
        'The control passes all original gates; the candidate misses full-root construction. Gate passage does not establish promotion-grade physics.', '',
        ':download:`Complete Phase50 original metric values <phase50-metrics.json>`;',
        ':download:`Complete Phase50 retained-tree metric values <phase50-retained-metrics.json>`;',
        ':download:`Phase50 micro, macro and exact source/PID/topology counts <phase50-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase50 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Late adaptation', 'Late PID adaptation'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase50 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase50 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase50 decision and uncertainty', '~' * 40, '',
        'Retained LCAG favors adaptation slightly; primary recovery, original source recall and half-root PID decline.',
        'Paired event-bootstrap intervals for retained LCAG gains include zero; they are conditional on the selected models, not training-seed uncertainty.',
        'Keep 70,000 events. Test gentler PID adaptation before increasing data or pretraining duration.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase51 compares frozen PID with late PID adaptation at learning-rate multiplier 0.1, keeping the encoder schedule and budget fixed.',
        f"Phase51 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase50` for full interpretation and historical synthesis.', '']
    return lines

def _phase49_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase49-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("original_job_status") != "COMPLETED" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase49 retained-tree coverage is incomplete")
    result = {"version": "phase49-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase49_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("late_adaptation_control", "late_pid_adaptation"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase49_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase49_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase49-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase49 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase49_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase49 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase49 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('late_adaptation_control', 'late_pid_adaptation')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase49_aggregation')}


def _render_phase49(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'late_pid_adaptation')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase49: effective PID adaptation, mixed scientific results', '-' * 65, '',
        'Both arms have the same primary exact source-set plus mother-PID recovery; this is not recursive topology efficiency.',
        'The candidate PID head records 2,188 optimizer updates and changed weights; the control remains frozen.',
        'The candidate passes all original gates; the control misses full-source recall. Gate passage does not establish promotion-grade physics.', '',
        ':download:`Complete Phase49 original metric values <phase49-metrics.json>`;',
        ':download:`Complete Phase49 retained-tree metric values <phase49-retained-metrics.json>`;',
        ':download:`Phase49 micro, macro and exact source/PID/topology counts <phase49-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase49 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Late adaptation', 'Late PID adaptation'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase49 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase49 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase49 decision and uncertainty', '~' * 40, '',
        'All-retained LCAG and mother coverage favor adaptation in this cohort; original-policy full LCAG and half-root PID accuracy decline.',
        'Paired event-bootstrap intervals for retained LCAG gains exclude zero, conditional on the two selected models; training-seed uncertainty is unmeasured.',
        'Keep 70,000 events and replicate effective PID adaptation before increasing data or pretraining duration.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase50 repeats the same control and PID adaptation contracts with a new seed and untouched validation cohort.',
        f"Phase50 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase49` for full interpretation and historical synthesis.', '']
    return lines

def _render_phase48(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'late_pid_adaptation')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase48: PID adaptation did not execute', '-' * 48, '',
        'Both selected step 4,000 with 115/3,670 exact source sets plus mother PID (3.13%); this does not require recursive topology.',
        'Final model tensors are identical. Neither PID head has optimizer state: the intended adaptation was ineffective.',
        'Mixed-precision weight caching after no-gradient rollout suppresses later gradients; a CPU model regression reproduces and repairs the path.',
        'These results are valid observations of the saved models, not evidence that effective PID adaptation has no benefit.',
        'Neither arm passes the full-root gate. Coherent forest agreement is zero; no promotion.', '',
        ':download:`Complete Phase48 original metric values <phase48-metrics.json>`;',
        ':download:`Complete Phase48 retained-tree metric values <phase48-retained-metrics.json>`;',
        ':download:`Phase48 micro, macro and exact source/PID/topology counts <phase48-aggregation-metrics.json>`.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views and strict repeats are checked.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Late adaptation', 'Late PID adaptation'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase48 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Late adaptation', 'Late PID adaptation'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase48 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase48 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase48 decision and uncertainty', '~' * 40, '',
        'Paired retained count differences are exactly zero. Degenerate bootstrap intervals do not imply zero population uncertainty.',
        'All exact components are depth one; no hierarchical reconstruction breakthrough is established.',
        'Hold 70,000 events. Repair gradient execution before spending on more data or longer pretraining.',
        'Corrected pretraining quality remains unmeasured; prior data and compute scaling were confounded.',
        'Phase49 repeats control versus late PID adaptation with gradient-safe mixed precision in both arms and explicit PID gradient checks.',
        f"Phase49 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase48` for full interpretation and historical synthesis.', '']
    return lines


def _render_phase47(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'frozen_encoder')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase47: primary advantage repeats; strict topology ties', '-' * 64, '',
        'Late adaptation selected 139/3,738 complete source sets plus mother PID (3.72%); frozen selected 133/3,738 (3.56%).',
        'This primary does not require recursive topology. Both recover three exact nontrivial components per retained scope.',
        'Neither arm passes all strict gates; coherent forest agreement is zero. No promotion.', '',
        ':download:`Complete Phase47 original metric values <phase47-metrics.json>`;',
        ':download:`Complete Phase47 retained-tree metric values <phase47-retained-metrics.json>`;',
        ':download:`Phase47 micro, macro and exact source/PID/topology counts <phase47-aggregation-metrics.json>`.', '',
        'Micro pools counts; unit and event macros average defined ratios equally and report their available denominators.',
        'Exact topology alone does not imply every leaf PID is correct; the supplement reports that stricter conjunction separately.', '',
        'Both arms completed 4,376 steps on 70,000 events. All seven views, exact repeats and checkpoint finiteness are verified.',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.', '']
    lines += _table(['Original policy metric', 'Late adaptation', 'Frozen encoder'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Late adaptation', 'Frozen encoder'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase47 all retained full and half trees', '~' * 48, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Late adaptation', 'Frozen encoder'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    aggregation = record['aggregation_supplement']['primary']
    lines += ['Phase47 micro and macro populations', '~' * 40, '',
        'Macro means exclude undefined zero-denominator ratios, count that unavailability, and retain defined failures as zeros.',
        'Exact source and PID columns require representability and structural validity. Nontrivial units have at least two sources.', '']
    lines += _table(['Arm', 'Scope', 'Metric', 'Micro', 'Unit macro', 'Event macro'],
        [[arm, scope, key, *[f"{aggregation[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if aggregation[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]]
         for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm', 'Scope', 'Nontrivial exact criterion', 'Numerator', 'Denominator'],
        [[arm, scope, key, *[aggregation[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]]
         for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase47 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Phase47 decision and uncertainty', '~' * 40, '',
        'Retained LCAG and mother-coverage event counts match exactly between arms. Paired bootstrap differences are zero;',
        'degenerate intervals do not imply zero population uncertainty. Source precision/recall intervals include zero.',
        'The Phase46 frozen topology advantage did not repeat. Cross-phase seed and cohort differ.',
        'Hold 70,000 events. Prior scaling confounded data and compute; longer pretraining has no established benefit.',
        'Phase48 tests late PID-head adaptation against the unchanged late-adaptation control, with equal 4,376-step budgets.',
        'Only PID-head freeze duration differs. Corrected pretraining quality remains unmeasured.',
        f"Phase48 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase47` for all populations, uncertainty, downloads and historical synthesis.', '']
    return lines


def _render_phase46(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'frozen_encoder')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase46: primary and topology favor different arms', '-' * 60, '',
        'Both corrected-source arms completed 4,376 steps on 70,000 events. Late adaptation selected 120/3,854',
        'complete target source sets plus mother PID (3.11%); frozen selected 110/3,854 (2.85%).',
        'This primary does not require exact recursive topology. Frozen instead recovers nine exact nontrivial components',
        'versus six for late adaptation in both scopes. Neither arm passes every strict gate. No promotion.', '',
        ':download:`Complete Phase46 original metric values <phase46-metrics.json>`;',
        ':download:`Complete Phase46 retained-tree metric values <phase46-retained-metrics.json>`.', '',
        'Selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.',
        'All seven views per arm and exact repeats are verified. Historical pre-audit results are separate populations.', '']
    lines += _table(['Original policy metric', 'Late adaptation', 'Frozen encoder'],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Late adaptation', 'Frozen encoder'],
        [[key, *[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['All retained full and half trees', '~' * 40, '',
        'Every retained root is checked, including incompatible targets as failed trials. Isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses explicit B partitions or labelled component fallbacks; missing roots or hemispheres are never invented.',
        'Source coverage is dominated by preserved inputs and is not hierarchy efficiency.', '']
    lines += _table(['Scope', 'Metric', 'Late adaptation', 'Frozen encoder'],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full', 'half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population', 'Full', 'Half/component'],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    lines += ['Phase46 returned beam candidates', '~' * 40, '',
        'All returned width-four candidates receive full and half checks. Model-only rankings remain distinct from the coherent',
        'post-inference oracle. Per-unit Oracle@K maxima are bounds and do not describe one coherent event.', '']
    lines += _table(['Arm', 'Returned candidates'], [[a, retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm', 'Scope', 'Ranking', 'LCAG', 'Mother coverage', 'Perfect component', 'Coherent forest'],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ['Decision and uncertainty', '~' * 40, '',
        'Frozen-minus-control retained half LCAG is +0.223 percentage points (paired event-cluster 95% interval +0.052 to +0.439).',
        'Full LCAG is +0.078 points (0.000 to +0.201). These 10,000-resample intervals condition on the trained models;',
        'they do not measure training-seed uncertainty. Primary selection and strict topology favor different arms.',
        'Hold 70,000 events. Prior scaling confounded data and compute; Phase42 did not support longer pretraining.',
        'Corrected pretraining quality remains unmeasured. Phase47 repeats the corrected adaptation comparison with a second seed',
        'and a fresh disjoint cohort, using the same authenticated train-only statistics and historical pretrained weights.',
        f"Phase47 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        'Physical mother momentum resolution is unavailable because truth mother four-vectors were not retained.',
        'Daughter-sum closure is an implementation invariant. See :doc:`../../phase46` for the complete review and historical synthesis.', '']
    return lines


def _render_phase45(record):
    if not record:
        return []
    labels = ("encoder_lr005_control", "encoder_lr010")
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ["Phase45: no benefit from doubling encoder learning rate", "-" * 60, "",
        "Both preregistered arms completed 4,376 steps and selected 148/3,805 complete target source sets plus mother PID (3.89%).",
        "This primary metric does not require exact recursive topology. Neither arm passes all original strict gates.",
        "Both full/half repeats are exact. These frozen-source results precede the independent scientific audit fixes;",
        "historical checkpoints and reported values are preserved, and do not validate the corrected implementation.", "",
        ":download:`Complete Phase45 original metric values <phase45-metrics.json>`;",
        ":download:`Complete Phase45 retained-tree metric values <phase45-retained-metrics.json>`.", "",
        "Selection uses 2,000 events (1,000 rollout); strict scoring uses 100 disjoint events and beam its 20-event subset.", ""]
    lines += _table(["Original policy metric", "Encoder LR 0.05", "Encoder LR 0.10"],
        [[key, *[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += ["All retained trees, including incompatible targets", "~" * 60, "",
        "All explicit roots are checked. Full scope includes isolated leaves; half scope uses two explicit B roots where available,",
        "otherwise labelled retained components. Missing initial roots and B partitions are never invented.",
        "Source coverage is dominated by preserved isolated inputs and is not hierarchy efficiency.", ""]
    lines += _table(["Scope", "Metric", "Encoder LR 0.05", "Encoder LR 0.10"],
        [[scope, key, *[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]]
         for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest')])
    lines += _table(["Reference population", "Full", "Half/component"],
        [[key, *[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]]
         for key in retained['arms'][labels[0]]['primary_structure']['full']])
    lines += ["Every returned beam candidate", "~" * 60, "",
        "Beam width is four; the returned candidate counts are shown below. Every candidate receives full and half checks.",
        "Greedy finds 2/37 full and 2/39 half exact nontrivial components. Average-link ranking and the coherent oracle find 3/37 and 3/39",
        "in both arms, without increasing the five correct LCAG pairs or recovering a coherent forest (0/20).",
        "Oracle uses truth only after search. Per-unit Oracle@K maxima are bounds, not one reconstructed event.", ""]
    lines += _table(["Arm", "Returned candidates"], [[a,retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(["Arm", "Scope", "Ranking", "LCAG", "Mother coverage", "Perfect component", "Coherent forest"],
        [[a, scope, rank, *[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]]
         for a in labels for scope, rankings in retained['arms'][a]['beam'].items() for rank, points in rankings.items()])
    lines += ["Decision and measurement limits", "~" * 60, "",
        "Hold 70,000 events. Earlier data scaling confounded data and compute, and Phase42 did not support longer pretraining.",
        "The priority is corrected supervision and train-only statistics. Phase46 compares late adaptation against a frozen",
        "historical encoder under the same corrected code. This tests transfer/adaptation, not a new pretraining objective.",
        f"Phase46 submission snapshot: {_literal(record['next_study_status'])}. No automatic promotion or sealed-test access.",
        "Physical mother momentum resolution is unavailable because truth mother four-vectors were not retained;",
        "daughter-sum closure is an implementation invariant. Single-seed estimates and small strict cohorts limit inference.",
        "See :doc:`../../phase45` for all-study synthesis, source limitations, and complete export counts.", ""]
    return lines


def _render_phase44(record: dict[str, Any]) -> list[str]:
    if not record:
        return []
    labels = ("late_adaptation_control", "early_adaptation")
    arms = record["arms"]
    lines = ["Phase44: recovered diagnostics, no promotion", "---------------------------------------------", "",
             "Both original jobs failed after completing 4,376 optimizer steps. Their nested replay seed",
             "was stale (20260910), while training used the intended top-level seed (20260911).",
             "Independent replay and checkpoint audits support diagnostic evaluation of the preserved checkpoints.",
             "These are post-hoc measurements with a preregistration deviation; the original jobs remain FAILED.",
             "Original policy metrics: full topology has 18 eligible truth-root units (82 unavailable); 4/18 are directly representable.",
             "Original policy metrics: half scope has 145 available units (62 unavailable); 102/145 are directly representable.",
             "Root construction uses all 100 processed events and is not a truth-validated success count.",
             "Unavailable topology measurements are not zero scores; alternative-checkpoint details appear in the review.",
             "No model is promoted. Selection uses 1,000 rollout events from a 2,000-event selection cohort;",
             "strict evaluation uses a separate 100 events and beam evaluation its fixed 20-event subset.",
             "All seven reports per arm, exact repeats, and every aggregate metric are retained in the download.",
             f"The separate local training-history export retains {record['training_history_scalar_rows']:,} scalar entries from all saved checkpoints and optimizer logs.", ""]
    retained = record["retained_tree_checks"]
    lines += ["Complete retained-tree checks", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Every explicit retained root is checked, including roots outside the training policy and isolated FSPs.",
              "Full scope checks the retained forest. Half scope uses explicit B halves where available and otherwise",
              "labels the retained-component fallback. No missing initial-state root or B hemisphere is invented.",
              "Component LCAG excludes trivial singletons; coherent agreement requires the whole retained forest in one candidate.",
              "These additional denominators do not replace the original eligibility-based metrics or promotion gates.",
              ":download:`Complete retained-tree metrics and source hashes <phase44-retained-metrics.json>`.", ""]
    lines += _table(["Arm", "Scope", "Checked units", "Unavailable", "LCAG pairs", "Mother coverage", "Perfect components", "Coherent forest"], [
        [arm, scope, value["available_unit_count"], value["unavailable_unit_count"], *[_metric_point(value[key]) for key in ("lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag", "coherent_retained_forest")]]
        for arm in labels for scope, value in retained["arms"][arm]["primary"].items()])
    lines += ["Retained reference composition (same for both arms)", "~" * 60, ""]
    lines += _table(["Scope", "Isolated leaves", "Single-source composites", "LCAG-eligible components", "Representable LCAG components"], [
        [scope, counts["isolated_leaf_units"], counts["single_source_composite_units"], counts["nontrivial_topology_units"], counts["representable_nontrivial_units"]]
        for scope, counts in retained["arms"][labels[0]]["primary_structure"].items()])
    lines += ["Source coverage includes isolated inputs; it is not hierarchy reconstruction efficiency.", ""]
    lines += ["Retained-tree beam checks", "~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Every returned candidate is scored. Model-only rankings remain separate from the truth-only oracle.",
              "The table oracle selects one coherent candidate per scope. The download also contains per-unit",
              "Oracle@K bounds and a separate coherent-event check; unit bounds are not one reconstructed event.", ""]
    lines += _table(["Arm", "Scope", "Ranking", "LCAG pairs", "Mother coverage", "Perfect components", "Coherent forest"], [
        [arm, scope, ranking, *[_metric_point(value[key]) for key in ("lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag", "coherent_retained_forest")]]
        for arm in labels for scope, rankings in retained["arms"][arm]["beam"].items() for ranking, value in rankings.items()])
    lines += ["Original policy-eligible metrics", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    rows = [[key, *[arms[a]["selected"][key] for a in labels]] for key in arms[labels[0]]["selected"]]
    rows += [[key, *[_metric_point(arms[a]["endpoints"][key]) for a in labels]] for key in arms[labels[0]]["endpoints"]]
    rows += [["all hierarchy gates passed", *[arms[a]["all_gates_passed"] for a in labels]]]
    lines += _table(["Metric", "Late adaptation", "Early adaptation"], rows)
    values = {(r["arm"], r["metric"]): r["value"] for r in record["metric_rows"] if r["view"] == "training_best"}
    lines += ["Phase44 pointer diagnostics", "~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Micro teacher-forced daughter association, separate from strict rollout topology.", ""]
    lines += _table(["Level", "Control precision", "Control recall", "Candidate precision", "Candidate recall"], [
        [level, *[values.get((arm, f"micro_level_{level}_pointer_{metric}")) for arm in labels for metric in ("precision", "recall")]] for level in range(1, 7)])
    lines += ["Phase44 beam comparison", "~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Oracle is a truth-assisted evaluation diagnostic, never a deployable ranking.",
              "It uses a lexicographic topology ranking, not a separate maximum for each metric.", ""]
    lines += _table(["Arm", "Scope", "Ranking", "Recall", "Precision", "LCAG", "Mother coverage", "Perfect LCAG"], [
        [arm, scope, rank, *[_metric_point(p) for p in points.values()]]
        for arm in labels for scope, rankings in arms[arm]["beam"].items() for rank, points in rankings.items()])
    lines += ["Phase45 allocation decision", "~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
              "Hold 70,000 events, pretrained checkpoint 81,096, and the 2,188-step encoder freeze.",
              "Test encoder learning-rate multipliers 0.05 versus 0.10 at 4,376 total steps, with",
              "seed 20260912 and a fresh cohort. Training, replay, and cohort seeds must agree before submission.",
              f"Phase45 submission snapshot: {_literal(record['next_study_status'])}. Two bounded arms; no automatic promotion.", "",
              "Increasing data is not justified by a controlled learning curve. Longer pretraining was not supported",
              "by Phase42. Stronger task-specific adaptation is an exploratory representation-learning hypothesis.",
              "Physical mother momentum resolution is unavailable; daughter-sum closure is an invariant.",
              "See :doc:`../../phase44` for the recovery limitation, complete evidence, and study synthesis.", ""]
    return lines


def _literal(value: Any) -> str:
    text = "UNKNOWN" if value is None else str(value)
    return " ".join("".join("\\" + char if char in string.punctuation else char for char in text).split())


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No records available.", ""]
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - " + _literal(headers[0])]
    lines.extend("     - " + _literal(header) for header in headers[1:])
    for row in rows:
        lines.append("   * - " + _literal(row[0]))
        lines.extend("     - " + _literal(cell) for cell in row[1:])
    return [*lines, ""]


def _display(value: Any) -> str:
    return "UNKNOWN" if value is None else str(value)


def _metric_point(value: dict[str, Any]) -> str:
    numerator, denominator = value.get("numerator"), value.get("denominator")
    ratio = numerator / denominator if numerator is not None and denominator else None
    shown = "UNKNOWN" if ratio is None else f"{ratio:.6g}"
    return f"{shown} ({_display(numerator)}/{_display(denominator)})"


def _signed(value: Any) -> str:
    return "UNKNOWN" if value is None else f"{value:+.3f}"


def _card(title: str, value: str, paragraphs: list[str], sources: list[str], kind: str, index: int, progress: tuple[int, int] | None = None) -> str:
    label = f"status-card-{index}"
    content = [f'<article class="status-card status-card--{kind}" aria-labelledby="{label}">',
               f'<h3 id="{label}">{html.escape(title)}</h3>', f'<p class="status-value">{html.escape(value)}</p>']
    if progress:
        step, total = progress
        content += ['<label for="status-pretraining-progress">Recorded pretraining progress</label>',
                    f'<progress class="status-progress" id="status-pretraining-progress" value="{step}" max="{total}">{step / total:.0%}</progress>']
    content += [f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs]
    content += [f'<p class="status-provenance">Provenance: {html.escape(", ".join(sources))}.</p>', "</article>"]
    return "\n".join(content)


def _render(manifest: dict[str, Any]) -> str:
    reconstruction = manifest["reconstruction"]
    science = manifest["science"]
    source = manifest["sources"][SOURCE_IDS["reconstruction_terminal"]]
    recent_source = manifest["sources"][SOURCE_IDS["reconstruction_phase40r1"]]
    recent = reconstruction["phase40r1"]
    control = recent["arms"]["control"]
    query_scale = recent["arms"]["query_scale"]
    beam_rankings_by_scope = recent["beam_rankings_by_scope"]
    delta, required = reconstruction["edge_f1_delta"], reconstruction["required_edge_f1_delta"]
    cards = [
        ("Reconstruction edge F1", _display(reconstruction["metrics"]["relbias"]["edge_f1"]),
         [f"q32 comparison: {_display(reconstruction['metrics']['q32']['edge_f1'])}.",
          "Historical Stage A aggregate rollout; exploratory, not a current physics benchmark."], "info"),
        ("Edge F1 improvement", _signed(delta),
         [f"Required improvement {_signed(required)}; gap to target {_signed(reconstruction['target_gap'])}.",
          f"Recommendation: {reconstruction['recommendation']}. Paired event evidence: {_display(reconstruction['paired_event_evidence'])}."], "warning"),
        ("Phase40r1 strict reconstruction", recent["status"],
         [f"Control complete-target efficiency: {_display(control['micro_complete_target_efficiency'])}; strict full roots: {_display(control['full_root_completion_numerator'])}/{_display(control['full_root_completion_denominator'])}.",
          f"Control half-tree recall / precision: {_display(control['half_source_recall'])} / {_display(control['half_source_precision'])}; perfect half LCAG: {_display(control['half_perfect_lcag_numerator'])}/{_display(control['half_perfect_lcag_denominator'])}.",
          f"Query-scale complete-target efficiency: {_display(query_scale['micro_complete_target_efficiency'])}; all gates passed: {_display(query_scale['all_gates_passed'])}.",
          "Historical Phase40r1 snapshot; see the Phase41 closeout below for the subsequent decision.",
          f"Recorded real pilot: {science['real_pilot']}; pretraining validation objectives remain UNAVAILABLE."], "warning"),
    ]
    if reconstruction.get("phase41"):
        phase41 = reconstruction["phase41"]
        best = phase41["arms"]["pointer32_control"]
        cards.insert(0, ("Phase41 reconstruction (historical)", "NO PROMOTION",
            [f"Control complete-target efficiency: {best['selected']['micro_complete_target_efficiency']:.2%}; full roots: 0/100.",
             "Historical Phase41 decision; the completed Phase42 comparison appears above.",
             f"Phase42: {phase41['next_study_status']}."], "warning"))
    if reconstruction.get("phase42"):
        record = reconstruction["phase42"]
        cards.insert(0, ("Phase42 pretraining transfer (historical)", "NO PROMOTION",
            ["Complete-target efficiency: 3.58% control versus 3.45% later checkpoint; both 0/100 full roots.",
             "Hold 70,000 events and the 81,096-step checkpoint. Test earlier task-specific encoder adaptation.",
             f"Phase43: {record['next_study_status']} (two bounded arms)."], "warning"))
    if reconstruction.get("phase43"):
        record = reconstruction["phase43"]
        cards.insert(0, ("Phase43 encoder adaptation (historical)", "NO PROMOTION",
            ["Complete-target efficiency: 3.79% late versus 3.90% early adaptation; both 0/100 full roots.",
             "Hold 70,000 events and the 81,096-step checkpoint. Replicate the small adaptation signal.",
             "Phase44: failed jobs; diagnostic evaluation recovered." if record["next_study_status"] == "FAILED_JOBS_DIAGNOSTICALLY_RECOVERED" else f"Phase44: {record['next_study_status']} (two bounded arms)."], "warning"))
    if reconstruction.get("phase44"):
        record = reconstruction["phase44"]
        cards.insert(0, ("Phase44 recovered reconstruction review", "DIAGNOSTIC / NO PROMOTION",
            ["Both original jobs failed a stale replay-seed contract check after training; preserved checkpoints were audited.",
             "Post-hoc metrics are complete. Hold 70,000 events and pretrained checkpoint 81,096.",
             f"Phase45: {record['next_study_status']} (two bounded learning-rate arms)."], "warning"))
    if reconstruction.get("phase45"):
        record = reconstruction["phase45"]
        cards.insert(0, ("Phase45 learning-rate review", "NO PROMOTION",
            ["Both arms recover 148/3,805 complete target source sets plus mother PID; strict coherent forest agreement remains zero.",
             "All retained roots and every beam candidate are checked. Phase45 predates the scientific audit fixes.",
             f"Phase46: {record['next_study_status']}; corrected late adaptation versus a frozen encoder at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase46"):
        record = reconstruction["phase46"]
        cards.insert(0, ("Phase46 corrected adaptation review", "NO PROMOTION",
            ["Primary recovery: 120/3,854 late adaptation versus 110/3,854 frozen; exact components favor frozen, 9 versus 6.",
             "All retained roots and returned beam candidates checked. Neither arm passes all strict gates; coherent forest agreement is zero.",
             f"Phase47: {record['next_study_status']}; second-seed replication at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase47"):
        record = reconstruction["phase47"]
        cards.insert(0, ("Phase47 corrected adaptation replication", "NO PROMOTION",
            ["Primary recovery: 139/3,738 late adaptation versus 133/3,738 frozen; both recover three exact nontrivial components.",
             "All fourteen views and every returned candidate checked. Coherent retained forests: 0/100; neither passes all gates.",
             f"Phase48: {record['next_study_status']}; one bounded late PID-head adaptation comparison at 70,000 events."], "warning"))
    if reconstruction.get("phase48"):
        record = reconstruction["phase48"]
        cards.insert(0, ("Phase48 PID adaptation review", "GRADIENT REPAIR REQUIRED",
            ["Both final models are identical: 115/3,670 primary recovery and zero coherent forests.",
             "PID weights received no optimizer updates. This does not test effective PID adaptation; all 92 beam candidates are scored.",
             f"Phase49: {record['next_study_status']}; gradient-safe repeat at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase49"):
        record = reconstruction["phase49"]
        cards.insert(0, ("Phase49 effective PID adaptation", "REPLICATION NEEDED",
            ["PID adaptation executed: 2,188 updates versus frozen control. Primary recovery ties; all-retained topology improves modestly.",
             "Candidate passes original gates, with mixed PID and topology outcomes. All full/half beam candidates checked; no promotion.",
             f"Phase50: {record['next_study_status']}; independent-seed replication at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase50"):
        record = reconstruction["phase50"]
        cards.insert(0, ("Phase50 effective PID adaptation", "MIXED / NO PROMOTION",
            ["PID adaptation executed: 2,188 updates versus frozen control. Primary recovery decreases; retained topology gains are small.",
             "Control passes original gates; candidate misses full-root construction. All full/half beam candidates checked; no promotion.",
             f"Phase51: {record['next_study_status']}; lower-rate PID adaptation at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase51"):
        record = reconstruction["phase51"]
        cards.insert(0, ("Phase51 lower-rate PID adaptation", "MIXED / NO PROMOTION",
            ["PID adaptation executed: 2,188 updates versus frozen control. Primary recovery rises slightly; retained topology declines slightly.",
             "Both arms miss full-root construction. All full/half beam candidates checked; no promotion.",
             f"Phase52: {record['next_study_status']}; recovery-objective weight 2 versus 4 at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase52"):
        record = reconstruction["phase52"]
        cards.insert(0, ("Phase52 stronger recovery objective", "MIXED / NO PROMOTION",
            ["Recovery weights 2 and 4 executed. Primary recovery rises slightly; retained exact components decline 10 to 9.",
             "Both arms pass the original gates. All full/half beam candidates checked; no promotion.",
             f"Phase53: {record['next_study_status']}; recovery-objective weight 2 versus 1 at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase53"):
        record = reconstruction["phase53"]
        cards.insert(0, ("Phase53 lower recovery objective", "MIXED / NO PROMOTION",
            ["Recovery weights 2 and 1 executed. Exact components rise 7 to 9; coherent forests fall 1 to 0.",
             "Weaker recovery misses full-root construction. All full/half beam candidates checked; no promotion.",
             f"Phase54: {record['next_study_status']}; replicate recovery-objective weight 2 versus 1 at fixed 70,000 events."], "warning"))
    if reconstruction.get("phase54"):
        record = reconstruction["phase54"]
        cards.insert(0, ("Phase54 recovery-dose replication", "MIXED / NO PROMOTION",
            ["Primary recovery rises 243/3783 to 261/3783; retained exact components fall 12 to 11.",
             "Both arms fail full-recall and root-construction gates. All 124 full/half beam candidates checked.",
             f"Phase55: {record['next_study_status']}; matched parent-ranking pretraining weights 1 versus 2, then identical reconstruction."], "warning"))
    if reconstruction.get("phase55"):
        record = reconstruction["phase55"]
        cards.insert(0, ("Phase55 parent-ranking pretraining", "MIXED / NO PROMOTION",
            ["Training completed; the evaluation checkpoint handoff was repaired and all views regenerated.",
             "Primary recovery rises 265/3548 to 271/3548; exact retained components rise 5 to 6, but coherent forests fall 1 to 0.",
             f"Phase56: {record['next_study_status']}; one matched replication at 70,000 training events."], "warning"))
    if reconstruction.get('phase56'):
        record = reconstruction['phase56']
        cards.insert(0, ('Phase56 parent-weight replication', 'INCOMPLETE COMPARISON / NO PROMOTION',
            ['Control failed pretraining objective preflight at attempted step 1643; reconstruction is unavailable.',
             'Completed arm fails full-root and source-recall gates. All retained full/half beam candidates checked.',
             f"Phase57: {record['next_study_status']}; late leaf-PID balance test, fixed 70,000 training events."], 'warning'))
    if reconstruction.get('phase57'):
        record=reconstruction['phase57']
        cards.insert(0,('Phase57 pretraining balance','SELECTION-CONTAMINATED DIAGNOSTICS',
            ['Both arms completed training, but all 100 reserved strict events entered checkpoint selection.',
             'All available full/half and beam metrics are diagnostic; no independent gate passage or promotion.',
             f"Phase58: {record['next_study_status']}; corrected cohort enforcement, fixed 70,000 training events."],'warning'))
    lines = ["Model performance and scientific status", "=======================================", "",
             "Recorded measurements from tracked evidence. Missing measurements are UNAVAILABLE;",
             "NOT_RUN describes a recorded evaluation status. Historical results do not verify",
             "the current model. See :doc:`../../evaluation` for metric definitions and populations.", "",
             ":download:`Metric values and source hashes <status.json>`.", "",
             f"Phase40r1 observation: {_literal(recent_source['recorded_date'])}; freshness: {_literal(recent_source['freshness']['status'])}.",
             f"Historical Stage A observation: {_literal(source['recorded_date'])}; freshness: {_literal(source['freshness']['status'])}.",
             "Historical cards use different cohorts; cross-phase differences are not controlled effects.", "",
             ".. raw:: html", "",
             '   <section class="status-dashboard" aria-label="Recorded model performance">']
    for index, (title, value, paragraphs, kind) in enumerate(cards):
        refs = reconstruction["phase57"]["source_ids"] if title.startswith("Phase57") else reconstruction["phase56"]["source_ids"] if title.startswith("Phase56") else reconstruction["phase55"]["source_ids"] if title.startswith("Phase55") else reconstruction["phase54"]["source_ids"] if title.startswith("Phase54") else reconstruction["phase53"]["source_ids"] if title.startswith("Phase53") else reconstruction["phase52"]["source_ids"] if title.startswith("Phase52") else reconstruction["phase51"]["source_ids"] if title.startswith("Phase51") else reconstruction["phase50"]["source_ids"] if title.startswith("Phase50") else reconstruction["phase49"]["source_ids"] if title.startswith("Phase49") else reconstruction["phase48"]["source_ids"] if title.startswith("Phase48") else reconstruction["phase47"]["source_ids"] if title.startswith("Phase47") else reconstruction["phase46"]["source_ids"] if title.startswith("Phase46") else reconstruction["phase45"]["source_ids"] if title.startswith("Phase45") else reconstruction["phase44"]["source_ids"] if title.startswith("Phase44") else reconstruction["phase43"]["source_ids"] if title.startswith("Phase43") else reconstruction["phase42"]["source_ids"] if title.startswith("Phase42") else reconstruction["phase41"]["source_ids"] if title.startswith("Phase41") else recent["source_ids"] if title.startswith("Phase40") else reconstruction["source_ids"]
        lines.extend("   " + line for line in _card(title, value, paragraphs, refs, kind, index).splitlines())
    lines += ["   </section>", "", ".. only:: not html", ""]
    for title, value, paragraphs, kind in cards:
        lines += [f"   **{_literal(title)}: {_literal(value)}**", ""]
        lines.extend(f"   {_literal(paragraph)}" for paragraph in paragraphs)
        lines += [""]
    lines += _render_phase57(reconstruction.get("phase57", {}))
    lines += _render_phase55(reconstruction.get("phase55", {}))
    lines += _render_phase56(reconstruction.get("phase56", {}))
    lines += _render_phase54(reconstruction.get("phase54", {}))
    lines += _render_phase53(reconstruction.get("phase53", {}))
    lines += _render_phase52(reconstruction.get("phase52", {}))
    lines += _render_phase51(reconstruction.get("phase51", {}))
    lines += _render_phase50(reconstruction.get("phase50", {}))
    lines += _render_phase49(reconstruction.get("phase49", {}))
    lines += _render_phase48(reconstruction.get("phase48", {}))
    lines += _render_phase47(reconstruction.get("phase47", {}))
    lines += _render_phase46(reconstruction.get("phase46", {}))
    lines += _render_phase45(reconstruction.get("phase45", {}))
    lines += _render_phase44(reconstruction["phase44"])
    lines += _render_phase43(reconstruction["phase43"])
    lines += _render_phase42(reconstruction["phase42"])
    lines += _render_phase41(reconstruction["phase41"])
    lines += ["Phase40r1 strict full-decay comparison", "----------------------------------------", "",
              "Both arms used the same untouched 100-event validation cohort. Values are",
              "strict checkpoint-direct reconstruction metrics; neither arm passed every",
              "preregistered hierarchy gate and neither is authorized for promotion.", ""]
    lines += _table(["Metric", "Control", "Query scale"], [
        ["selected step", control["selected_step"], query_scale["selected_step"]],
        ["micro complete-target efficiency", control["micro_complete_target_efficiency"], query_scale["micro_complete_target_efficiency"]],
        ["predicted depth fraction", control["predicted_depth_fraction"], query_scale["predicted_depth_fraction"]],
        ["tree validity", control["tree_validity"], query_scale["tree_validity"]],
        ["full root completion", f"{_display(control['full_root_completion_numerator'])}/{_display(control['full_root_completion_denominator'])}", f"{_display(query_scale['full_root_completion_numerator'])}/{_display(query_scale['full_root_completion_denominator'])}"],
        ["full LCAG", f"{_display(control['full_lcag_numerator'])}/{_display(control['full_lcag_denominator'])}", f"{_display(query_scale['full_lcag_numerator'])}/{_display(query_scale['full_lcag_denominator'])}"],
        ["exact mother coverage", f"{_display(control['exact_mother_coverage_numerator'])}/{_display(control['exact_mother_coverage_denominator'])}", f"{_display(query_scale['exact_mother_coverage_numerator'])}/{_display(query_scale['exact_mother_coverage_denominator'])}"],
        ["full source recall", control["full_source_recall"], query_scale["full_source_recall"]],
        ["full source precision", control["full_source_precision"], query_scale["full_source_precision"]],
        ["half-tree source recall", control["half_source_recall"], query_scale["half_source_recall"]],
        ["half-tree source precision", control["half_source_precision"], query_scale["half_source_precision"]],
        ["half-tree LCAG", f"{_display(control['half_lcag_numerator'])}/{_display(control['half_lcag_denominator'])}", f"{_display(query_scale['half_lcag_numerator'])}/{_display(query_scale['half_lcag_denominator'])}"],
        ["perfect half-tree LCAG", f"{_display(control['half_perfect_lcag_numerator'])}/{_display(control['half_perfect_lcag_denominator'])}", f"{_display(query_scale['half_perfect_lcag_numerator'])}/{_display(query_scale['half_perfect_lcag_denominator'])}"],
        ["all gates passed", control["all_gates_passed"], query_scale["all_gates_passed"]],
    ])
    lines += [f"Current best arm: {_literal(recent['current_best_arm'])}. Metric contract:",
              f"{_literal(recent['metric_contract_version'])} ({_literal(recent['metric_completeness'])}).",
              "The contract rejects dashboard generation if any strict, half-tree, full-scope beam, or half-scope beam metric is absent.", "",
              "Current-best beam-search reconstruction", "---------------------------------------", "",
              f"Beam search used {_literal(recent['beam_event_count'])} validation events. Every registered model-only ranker is shown; oracle-at-k is diagnostic only.", ""]
    beam_labels = {
        "greedy": "greedy",
        "average_link_probability": "average link probability",
        "learned_confidence_mean": "learned confidence mean",
        "learned_confidence_sum": "learned confidence sum",
        "normalized_joint_log_probability": "normalized joint log probability",
        "oracle_at_k": "oracle at k (diagnostic)",
    }
    lines += _table(["Scope", "Ranking", "source recall", "source precision", "LCAG pair accuracy", "mother coverage", "perfect LCAG"], [
        [scope, beam_labels[name], *[_metric_point(beam_rankings_by_scope[scope][name][metric]) for metric in (
            "source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag")]]
        for scope in ("full", "half")
        for name in beam_labels
    ])
    lines += [
              f"Phase41 preregisters pointer/object thresholds {_literal(recent['decision']['phase41_pointer_threshold'])}/{_literal(recent['decision']['phase41_object_threshold'])} on a fresh cohort; this is not a post-hoc phase40 promotion setting.", "",
              "Stage A validation comparison", "-----------------------------", "",
              "Values are copied from the terminal receipt. Validation loss and pointer metrics",
              "describe the validation view; edge/tree results describe rollout. The receipt",
              "does not retain per-metric sufficient statistics or paired event uncertainties.",
              "The canonical subtree field is retained under its receipt name.", ""]
    lines += _table(["Metric", "Relation bias", "q32"], [
        [name, reconstruction["metrics"]["relbias"][name], reconstruction["metrics"]["q32"][name]]
        for name in reconstruction["metrics"]["relbias"]])
    lines += _table(["Cohort", "Relation bias", "q32"], [
        [name, reconstruction["cohorts"]["relbias"][name], reconstruction["cohorts"]["q32"][name]]
        for name in ("validation_events", "rollout_events")])
    lines += ["Unavailable values in this table appear as UNKNOWN. Smaller validation loss is",
              "better only under comparable objective weights and target populations; it is not",
              "a full-tree reconstruction efficiency. Tree validity does not imply correct topology.", "",
              "Training and software context", "-----------------------------", ""]
    pretraining, audit = manifest["pretraining"], manifest["audit"]
    latest = manifest["verification"]["latest_record"]
    pytest = latest["pytest"]
    lines += _table(["Recorded context", "Value"], [
        ["Pretraining resume / planned steps", f"{_display(pretraining['recorded_step'])} / {_display(pretraining['planned_steps'])}"],
        ["Stage A completion", reconstruction["completion"]],
        ["Current status recommendation", audit["recommendation"]],
        ["CPU fixtures", f"{pytest['result']}: {_display(pytest['passed'])} passed; {_display(pytest['failed'])} failed"],
        ["Notebook registry", manifest["notebooks"]["total"]],
        ["Live deployment / CI", "NOT_QUERIED"],
    ])
    if audit["revision_match"] == "different_revision" or latest["revision_match"] == "different_revision":
        lines += ["STALE REVISION EVIDENCE: audit or software verification describes another revision.", ""]
    lines += ["The current-status record owns the recommendation. Recorded metadata is not live authorization.",
              "CPU fixtures measure software behavior. Step counts do not measure convergence.", "",
              "Evidence provenance", "-------------------", "",
              f"Reference time: {_literal(manifest['provenance']['as_of'])}; basis: {_literal(manifest['provenance']['as_of_basis'])}.",
              f"Freshness threshold: {FRESHNESS_DAYS} days. Unknown dates remain unknown; commit time is not experiment time.",
              "Full hashes and revision details are available in the metric download.", ""]
    lines += _table(["Source", "Observation date", "Availability", "Freshness"], [
        [item["source_id"], item["recorded_date"], item["availability"], item["freshness"]["status"]]
        for item in manifest["sources"].values()])
    return "\n".join(lines)


def _compact_exact_numbers(value):
    """Shorten exactly integral JSON numbers without changing metric values.

    Keep negative zero and integers beyond the exact binary64 range unchanged.
    Historical metric downloads retain their original byte serialization.
    """
    if type(value) is float and abs(value) <= 2**53 and value.is_integer():
        if value != 0 or math.copysign(1, value) > 0:
            return int(value)
    if isinstance(value, dict):
        return {key: _compact_exact_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_exact_numbers(item) for item in value]
    return value


def generate_status(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write the dashboard and allowlisted metric downloads; copy no raw evidence.

    Git is optional. Builds never sample wall-clock time, import scientific
    dependencies, read external run artifacts, or execute tests/training.
    """
    root, requested = Path(repo_root).resolve(), Path(output_dir).absolute()
    output = requested.resolve()
    if output == root or output in root.parents or any(part.is_symlink() for part in (requested, *requested.parents)):
        raise ValueError("Status output must be a dedicated non-symlink directory")
    if output.exists() and any(path.name not in {"index.rst", "status.json", "phase44-retained-metrics.json", "phase45-metrics.json", "phase45-retained-metrics.json", "phase46-metrics.json", "phase46-retained-metrics.json", "phase47-metrics.json", "phase47-retained-metrics.json", "phase47-aggregation-metrics.json", "phase48-metrics.json", "phase48-retained-metrics.json", "phase48-aggregation-metrics.json", "phase49-metrics.json", "phase49-retained-metrics.json", "phase49-aggregation-metrics.json", "phase50-metrics.json", "phase50-retained-metrics.json", "phase50-aggregation-metrics.json", "phase51-metrics.json", "phase51-retained-metrics.json", "phase51-aggregation-metrics.json", "phase52-metrics.json", "phase52-retained-metrics.json", "phase52-aggregation-metrics.json", "phase53-metrics.json", "phase53-retained-metrics.json", "phase53-aggregation-metrics.json", "phase54-metrics.json", "phase54-retained-metrics.json", "phase54-aggregation-metrics.json", "phase41-metrics.json", "phase55-metrics.json", "phase55-retained-metrics.json", "phase55-aggregation-metrics.json", "phase55-pretraining-metrics.json", "phase56-metrics.json", "phase56-retained-metrics.json", "phase56-aggregation-metrics.json", "phase56-pretraining-metrics.json", "phase57-metrics.json", "phase57-retained-metrics.json", "phase57-aggregation-metrics.json", "phase57-pretraining-metrics.json"} for path in output.iterdir()):
        raise ValueError("Status output contains unexpected files; use a fresh dedicated directory")
    history = _git(root, "log", "-1", "--format=%H%n%cI")
    lines = history.decode("utf-8", errors="replace").splitlines() if history else []
    revision, committed_at = (_sha(lines[0]), _date_text(lines[1])) if len(lines) >= 2 else (None, None)
    reference, basis = committed_at, "git_commit_time" if committed_at else "unknown_no_git_history"
    if (epoch := os.environ.get("SOURCE_DATE_EPOCH")) is not None:
        if not epoch.isdigit():
            raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer")
        try:
            reference = datetime.fromtimestamp(int(epoch), timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError) as error:
            raise ValueError("SOURCE_DATE_EPOCH is outside the supported range") from error
        basis = "SOURCE_DATE_EPOCH"
    sources, payloads = {}, {}
    for key in SOURCE_PATHS:
        source, payloads[key] = _load_source(root, key, reference)
        sources[SOURCE_IDS[key]] = source
    source_set = [{key: source[key] for key in ("source_id", "sha256", "availability")} for source in sources.values()]
    digest = hashlib.sha256(json.dumps(source_set, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"schema_version": "hypertagging-wiki-status-v2", "provenance": {
        "git_revision": revision, "git_commit_date": committed_at, "as_of": reference, "as_of_basis": basis,
        "freshness_threshold_days": FRESHNESS_DAYS, "source_set_sha256": digest,
        "tracked_repository_artifact_inputs_opened": sum(path.startswith("artifacts/") for path in SOURCE_PATHS.values()),
        "external_filesystem_or_network_artifacts_opened": False,
        "dashboard_executes_tests_or_training": False, "publication_policy": "ALLOWLISTED_SCALARS_COUNTS_DATES_HASHES_ONLY",
    }, "sources": sources, **_collect(payloads, revision)}
    phase44 = manifest.get("reconstruction", {}).get("phase44", {})
    retained = phase44.get("retained_tree_checks")
    if retained:
        metric_rows = retained.pop("metric_rows")
        download = {"version": retained["version"], "evaluator_revision": retained["evaluator_revision"],
                    "source_hashes": retained["source_hashes"], "metric_rows": metric_rows}
        encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
        _write(output / "phase44-retained-metrics.json", encoded)
        retained["metric_download"] = {"filename": "phase44-retained-metrics.json", "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase45 = manifest.get("reconstruction", {}).get("phase45", {})
    for record, filename in ((phase45, "phase45-metrics.json"), (phase45.get("retained_tree_checks", {}), "phase45-retained-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase46 = manifest.get("reconstruction", {}).get("phase46", {})
    for record, filename in ((phase46, "phase46-metrics.json"), (phase46.get("retained_tree_checks", {}), "phase46-retained-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase47 = manifest.get("reconstruction", {}).get("phase47", {})
    for record, filename in ((phase47, "phase47-metrics.json"), (phase47.get("retained_tree_checks", {}), "phase47-retained-metrics.json"), (phase47.get("aggregation_supplement", {}), "phase47-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase48 = manifest.get("reconstruction", {}).get("phase48", {})
    for record, filename in ((phase48, "phase48-metrics.json"), (phase48.get("retained_tree_checks", {}), "phase48-retained-metrics.json"), (phase48.get("aggregation_supplement", {}), "phase48-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase49 = manifest.get("reconstruction", {}).get("phase49", {})
    for record, filename in ((phase49, "phase49-metrics.json"), (phase49.get("retained_tree_checks", {}), "phase49-retained-metrics.json"), (phase49.get("aggregation_supplement", {}), "phase49-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase50 = manifest.get("reconstruction", {}).get("phase50", {})
    for record, filename in ((phase50, "phase50-metrics.json"), (phase50.get("retained_tree_checks", {}), "phase50-retained-metrics.json"), (phase50.get("aggregation_supplement", {}), "phase50-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase51 = manifest.get("reconstruction", {}).get("phase51", {})
    for record, filename in ((phase51, "phase51-metrics.json"), (phase51.get("retained_tree_checks", {}), "phase51-retained-metrics.json"), (phase51.get("aggregation_supplement", {}), "phase51-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase52 = manifest.get("reconstruction", {}).get("phase52", {})
    for record, filename in ((phase52, "phase52-metrics.json"), (phase52.get("retained_tree_checks", {}), "phase52-retained-metrics.json"), (phase52.get("aggregation_supplement", {}), "phase52-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase53 = manifest.get("reconstruction", {}).get("phase53", {})
    for record, filename in ((phase53, "phase53-metrics.json"), (phase53.get("retained_tree_checks", {}), "phase53-retained-metrics.json"), (phase53.get("aggregation_supplement", {}), "phase53-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase54 = manifest.get("reconstruction", {}).get("phase54", {})
    for record, filename in ((phase54, "phase54-metrics.json"), (phase54.get("retained_tree_checks", {}), "phase54-retained-metrics.json"), (phase54.get("aggregation_supplement", {}), "phase54-aggregation-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase55 = manifest.get("reconstruction", {}).get("phase55", {})
    for record, filename in ((phase55, "phase55-metrics.json"), (phase55.get("retained_tree_checks", {}), "phase55-retained-metrics.json"), (phase55.get("aggregation_supplement", {}), "phase55-aggregation-metrics.json"), (phase55.get("pretraining_metrics", {}), "phase55-pretraining-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase56 = manifest.get("reconstruction", {}).get("phase56", {})
    for record, filename in ((phase56, "phase56-metrics.json"), (phase56.get("retained_tree_checks", {}), "phase56-retained-metrics.json"), (phase56.get("aggregation_supplement", {}), "phase56-aggregation-metrics.json"), (phase56.get("pretraining_metrics", {}), "phase56-pretraining-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    phase57 = manifest.get("reconstruction", {}).get("phase57", {})
    for record, filename in ((phase57, "phase57-metrics.json"), (phase57.get("retained_tree_checks", {}), "phase57-retained-metrics.json"), (phase57.get("aggregation_supplement", {}), "phase57-aggregation-metrics.json"), (phase57.get("pretraining_metrics", {}), "phase57-pretraining-metrics.json")):
        if record:
            metric_rows = record.pop("metric_rows")
            download = {"source_hashes": record["source_hashes"], "metric_rows": metric_rows, "independent_validation": False, "strict_selection_overlap": 100, "scientific_classification": "SELECTION_CONTAMINATED_DIAGNOSTICS", "classification_scope": "phase57_reconstruction_evaluation"}
            encoded = (json.dumps(download, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            _write(output / filename, encoded)
            record["metric_download"] = {"filename": filename, "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(metric_rows)}
    # Keep the historical diagnostics needed to render Phase41 inline; publish
    # every bulk row in a hash-bound download to keep status.json below 10 MiB.
    phase41 = manifest.get("reconstruction", {}).get("phase41", {})
    if phase41:
        rows = phase41["metric_rows"]
        download = _compact_exact_numbers({"source_hashes": phase41["source_hashes"], "metric_rows": rows})
        encoded = (json.dumps(download, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        _write(output / "phase41-metrics.json", encoded)
        phase41["metric_download"] = {"filename": "phase41-metrics.json", "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "metric_count": len(rows)}
        phase41["metric_rows"] = [row for row in rows if row["view"] == "training_best"]
    manifest = _compact_exact_numbers(manifest)
    _write(output / "status.json", (json.dumps(manifest, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode())
    _write(output / "index.rst", _render(manifest).encode())
    return manifest


def _phase55_pretraining_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase55-pretraining-export-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Phase55 pretraining evidence is incomplete')
    registry = json.loads(Path(__file__).with_name('phase55_pretraining_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and type(row.get('value')) is not bool and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase55 pretraining scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase55 pretraining metric registry')
    return {'metric_rows': rows, 'scalar_rows': _integer(raw.get('scalar_rows')),
            'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'final_validation': {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'validation_step_2188'} for arm in ('late_adaptation_control', 'stronger_parent_pretraining')},
            'source_ids': _refs('reconstruction_phase55_pretraining')}


def _render_phase55(record):
    if not record:
        return []
    labels = ('late_adaptation_control', 'stronger_parent_pretraining')
    arms, retained = record['arms'], record['retained_tree_checks']
    lines = ['Phase55: Parent-ranking pretraining has mixed reconstruction effects', '-' * 80, '',
        'Both original jobs failed in post-training evaluation, after completing 2,188 pretraining and 4,376 reconstruction steps.',
        'The evaluator was given the original encoder instead of the refined checkpoint. Its lineage check correctly rejected the frozen early checkpoint.',
        'Evaluation was regenerated using authenticated arm-specific refined checkpoints. Original failed receipts and all trained models are preserved.',
        'Two control and four candidate event records differ across original-node and recovery-host execution despite matching checkpoint bytes, scoring code and inputs. Recovered same-host repeats control this review; small-count effects remain sensitive.', '',
        'Primary source-set plus mother-PID recovery rises 265/3548 to 271/3548; this is not recursive topology efficiency.',
        'Control fails full-source recall; stronger parent pretraining fails full-root construction. Neither arm passes every gate.',
        'The control coherent forest contains 17 leaves and no mothers; it is not a successful deep hierarchy. Primary exact components in both arms have depth one.', '',
        ':download:`Complete Phase55 original metrics <phase55-metrics.json>`;',
        ':download:`Complete Phase55 retained-tree metrics <phase55-retained-metrics.json>`;',
        ':download:`Phase55 micro, macro and exact counts <phase55-aggregation-metrics.json>`;',
        ':download:`Phase55 pretraining validation and execution metrics <phase55-pretraining-metrics.json>`.', '',
        'Phase55 pretraining execution', '~' * 40, '',
        'Parent-ranking weights 1 and 2 executed with nonzero objective gradients. Both arms processed exactly 70,000 training events in 2,188 steps; the last batch has 16 events.',
        'Both initialize 129 parameters from step 81096, with fresh train-only normalization, optimizer, scheduler, RNG and memory.',
        'Reconstruction transfers each fixed final refined checkpoint and freezes its PID head. This study compares objective weights, not refinement versus no refinement.', '']
    pre = record['pretraining_metrics']['final_validation']
    lines += _table(['Recorded final pretraining validation metric', 'Weight 1', 'Weight 2'],
        [[key, *[f"{pre[a][key]:.6g}" for a in labels]] for key in ('validation_parent_ranking_accuracy','validation_loss_parent','validation_relation_accuracy','validation_leaf_pid_accuracy','validation_effective_rank')])
    lines += ['These are recorded validation aggregates over the same 1,000 events and four views; batch-mean diagnostics are not reconstructed micro efficiencies.', '',
        'Phase55 original-policy gates', '~' * 40, '',
        'Checkpoint selection uses 2,000 validation events (1,000 rollout); strict scoring uses 100 disjoint events, and beam uses its first 20.', '']
    lines += _table(['Original policy metric', 'Weight 1', 'Weight 2'], [[key,*[_metric_point(arms[a]['endpoints'][key]) for a in labels]] for key in arms[labels[0]]['endpoints']])
    lines += _table(['Strict gate', 'Weight 1', 'Weight 2'], [[key,*[arms[a]['gates'][key] for a in labels]] for key in arms[labels[0]]['gates']])
    lines += ['Phase55 all retained full and half trees', '~' * 48, '',
        'All retained roots are checked. Incompatible targets remain failed primary trials, and isolated leaves do not earn trivial LCAG successes.',
        'Half scope uses B partitions where defined and explicit component fallbacks otherwise. Source coverage is dominated by preserved inputs.', '']
    lines += _table(['Scope','Metric','Weight 1','Weight 2'], [[scope,key,*[_metric_point(retained['arms'][a]['primary'][scope][key]) for a in labels]] for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','mother_pid_accuracy','perfect_lcag','coherent_retained_forest','target_representable','root_pid_accuracy')])
    lines += _table(['Reference population','Full','Half/component'], [[key,*[retained['arms'][labels[0]]['primary_structure'][scope][key] for scope in ('full','half')]] for key in retained['arms'][labels[0]]['primary_structure']['full']])
    agg = record['aggregation_supplement']['primary']
    lines += ['Phase55 micro and macro populations','~' * 40,'','Undefined zero-denominator ratios remain unavailable; defined failures remain zeros. Nontrivial units contain at least two sources.','']
    lines += _table(['Arm','Scope','Metric','Micro','Unit macro','Event macro'], [[arm,scope,key,*[f"{agg[arm][f'{scope}.greedy.{key}.{kind}']:.6g}" if agg[arm][f'{scope}.greedy.{key}.{kind}'] is not None else 'UNAVAILABLE' for kind in ('micro','unit_macro','event_macro')]] for arm in labels for scope in ('full','half') for key in ('source_recall','source_precision','lcag_pair_accuracy','perfectLCAG')])
    lines += _table(['Arm','Scope','Nontrivial exact criterion','Numerator','Denominator'], [[arm,scope,key,*[agg[arm][f'{scope}.greedy.exact.nontrivial.{key}.{part}'] for part in ('numerator','denominator')]] for arm in labels for scope in ('full','half') for key in ('source','source_leaf_pid','source_topology','source_topology_all_pid')])
    lines += ['Phase55 returned beam candidates','~' * 40,'','Every returned candidate receives full and half checks. Model-only rankings and coherent post-inference oracle metrics remain separate; per-unit Oracle@K maxima are bounds, not one event hypothesis.','']
    lines += _table(['Arm','Returned candidates'],[[a,retained['arms'][a]['beam_candidate_count']] for a in labels])
    lines += _table(['Arm','Scope','Ranking','LCAG','Mother coverage','Perfect component','Coherent forest'],[[a,scope,rank,*[_metric_point(points[k]) for k in ('lcag_pair_accuracy','mother_pid_coverage','perfect_lcag','coherent_retained_forest')]] for a in labels for scope,rankings in retained['arms'][a]['beam'].items() for rank,points in rankings.items()])
    lines += ['Phase55 decision','~' * 40,'',
        'Hold 70,000 training events. The accumulated studies do not establish an immediate data-growth benefit or a benefit from longer pretraining alone.',
        'Improved pretraining quality remains a testable hypothesis: component improvements coexist with a coherent-forest regression. No promotion is justified.',
        'Exploratory recovered half-LCAG and full-recall intervals exclude zero, but their original-node intervals include zero. Half mother coverage and half recall have positive intervals in both executions. None measures training-seed uncertainty or adjusts for multiple comparisons.',
        'Phase56 repeats parent-ranking weights 1 and 2 once with a fresh seed. Both arms retain matched 2,188-step refinement and 4,376-step reconstruction.',
        'Only 1,491 untouched validation events remain before Phase56. Its matched selection cohort is reduced to 1,000 events, with 100 untouched strict events and 20 beam events; 391 remain afterward. This reduces selection precision and prevents causal cross-phase rate comparisons.',
        f"Phase56 submission snapshot: {_literal(record['next_study_status'])}. No automatic campaign chain or sealed-test access.",
        'Physical mother momentum resolution remains unavailable; daughter-sum closure is an implementation invariant.',
        'See :doc:`../../phase55` for uncertainty, depth checks, historical synthesis and recovery provenance.','']
    return lines


def _phase56_projection(raw: Any) -> dict[str, Any]:
    phase, labels = 56, ("stronger_parent_pretraining",)
    raw = _mapping(raw)
    if raw.get("audit_version") != f"phase{phase}-closeout-v1":
        return {}
    if raw.get("metric_completeness") != "COMPLETE_AVAILABLE_ARTIFACTS":
        raise ValueError("Phase41 full/half metric set is incomplete")
    endpoints = ("exact_mother_coverage", "full_lcag", "full_root_completion",
                 "full_source_precision", "full_source_recall", "half_lcag",
                 "half_perfect_lcag", "half_root_pid_accuracy", "half_source_precision", "half_source_recall")
    rankings = ("greedy", "average_link_probability", "learned_confidence_mean",
                "learned_confidence_sum", "normalized_joint_log_probability", "oracle_at_k")
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage", "perfect_lcag")
    def point(raw_point):
        point = _mapping(raw_point)
        num, den = _number(point.get("numerator")), _number(point.get("denominator"))
        if num is None or den is None or num < 0 or den < 0 or num > den:
            raise ValueError("Phase41 count is missing or invalid")
        return {"numerator": int(num) if num.is_integer() else num, "denominator": int(den) if den.is_integer() else den, "value": num / den if den else None}
    result = {"source_ids": _refs(f"reconstruction_phase{phase}"), "status": "INCOMPLETE_COMPARISON",
              "next_study_status": _enum(raw.get("next_study_status"), {"PREPARED", "SUBMITTED", "RUNNING"}),
              "train_events": _integer(raw.get("train_events")),
              "strict_event_count": _integer(raw.get("strict_event_count")),
              "beam_event_count": _integer(raw.get("beam_event_count")),
              "sealed_test_accessed": _boolean(raw.get("sealed_test_accessed")),
              "source_hashes": [_sha(value) for value in _list(raw.get("source_hashes"))], "arms": {}}
    control = _mapping(_mapping(raw.get('arms')).get('late_adaptation_control'))
    candidate = _mapping(_mapping(raw.get('arms')).get('stronger_parent_pretraining'))
    if (raw.get('status') != 'INCOMPLETE_COMPARISON' or raw.get('comparison_available') is not False
        or raw.get('source_boundary') != 'native_phase56_one_arm_pretraining_failure'
        or raw.get('sealed_test_accessed') is not False
        or control.get('reconstruction_status') != 'UNAVAILABLE_NOT_TRAINED'
        or control.get('endpoints') is not None or control.get('gates') is not None
        or control.get('selected') is not None or control.get('all_gates_passed') is not None
        or candidate.get('status') != 'COMPLETED'):
        raise ValueError('Phase56 incomplete comparison boundary is invalid')
    result.update(comparison_available=False, metric_completeness='COMPLETE_AVAILABLE_ARTIFACTS',
                  source_boundary='native_phase56_one_arm_pretraining_failure')
    result['arms']['late_adaptation_control'] = {'status':'FAILED', 'reconstruction_status':'UNAVAILABLE_NOT_TRAINED',
        'endpoints':None, 'gates':None, 'selected':None, 'all_gates_passed':None,
        'completed_pretraining_steps':1642, 'attempted_pretraining_step':1643}
    for arm in labels:
        record = _mapping(_mapping(raw.get("arms")).get(arm))
        selected = _mapping(record.get("selected"))
        result["arms"][arm] = {
            "optimizer_steps": _integer(record.get("optimizer_steps")),
            "training_elapsed_seconds": _number(record.get("training_elapsed_seconds")),
            "gates": {key: _boolean(_mapping(record.get("gates")).get(key)) for key in (
                "minimum_complete_target_efficiency", "minimum_depth_fraction", "minimum_full_source_precision",
                "minimum_full_source_recall", "minimum_half_lcag", "minimum_half_perfect_lcag",
                "minimum_half_root_pid_accuracy", "minimum_half_source_precision", "minimum_half_source_recall",
                "nonzero_exact_mother_coverage", "nonzero_full_lcag", "nonzero_full_root_completion",
                "primary_repeat_identical", "structural_guardrails")},
            "selected": {key: _number(selected.get(key)) for key in ("step", "micro_complete_target_efficiency", "predicted_depth_fraction", "predicted_tree_validity_rate")},
            "all_gates_passed": _boolean(record.get("all_gates_passed")),
            "primary_repeat_identical": _boolean(record.get("primary_repeat_identical")),
            "endpoints": {key: point(_mapping(record.get("endpoints")).get(key)) for key in endpoints},
            "beam": {scope: {rank: {metric: point(_mapping(_mapping(_mapping(_mapping(record.get("beam")).get(scope)).get(rank)).get(metric)))
                                      for metric in metrics} for rank in rankings} for scope in ("full", "half")}}
    # Exact authored vocabulary; arbitrary source strings and injected fields cannot publish.
    registry = json.loads(Path(__file__).with_name(f"phase{phase}_metric_registry.json").read_text())
    allowed = {tuple(row) for row in registry}
    rows = []
    seen = set()
    for row in _list(raw.get("metric_rows")):
        identity = (row.get("arm"), row.get("view"), row.get("metric"))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError("Duplicate Phase41 metric")
        seen.add(identity)
        value = row.get("value")
        if value is not None and _number(value) is None and type(value) is not bool:
            raise ValueError("Invalid Phase41 metric value")
        rows.append({"arm": identity[0], "view": identity[1], "metric": identity[2], "value": value})
    if seen != allowed:
        raise ValueError("Phase41 metric registry is incomplete")
    result["metric_rows"] = rows
    return result


def _phase56_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase56-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("source_boundary") != "native_phase56_one_arm_pretraining_failure" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase56 retained-tree coverage is incomplete")
    result = {"version": "phase56-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase56_retained"),
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("stronger_parent_pretraining",):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase56_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase56_aggregation_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase56-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE':
        raise ValueError('Missing Phase56 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase56_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase56 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase56 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('stronger_parent_pretraining',)}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase56_aggregation')}


def _phase56_pretraining_projection(value):
    raw = _mapping(value)
    if raw.get('version') != 'phase56-pretraining-export-v1' or raw.get('status') != 'COMPLETE_AVAILABLE_ARTIFACTS':
        raise ValueError('Phase56 pretraining evidence is incomplete')
    registry = json.loads(Path(__file__).with_name('phase56_pretraining_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and type(row.get('value')) is not bool and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase56 pretraining scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase56 pretraining metric registry')
    return {'metric_rows': rows, 'scalar_rows': _integer(raw.get('scalar_rows')),
            'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'final_validation': {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'validation_step_2188'} for arm in ('stronger_parent_pretraining',)},
            'source_ids': _refs('reconstruction_phase56_pretraining')}


def _render_phase56(record):
    if not record:
        return []
    arm = record['arms']['stronger_parent_pretraining']
    retained = record['retained_tree_checks']['arms']['stronger_parent_pretraining']
    lines = ['Phase56: incomplete comparison, all available metrics checked', '-' * 64, '',
        'The control failed pretraining at attempted step 1643: weighted leaf-PID/LCA gradient dominance 20.1546 exceeded the unchanged limit 20.',
        'It completed 1642 optimizer steps and 52,544 presentations. It has no reconstruction checkpoints: reconstruction metrics are UNAVAILABLE_NOT_TRAINED, never zero.',
        'The stronger-parent arm completed 2188 refinement and 4376 reconstruction steps. No paired contrast or paired bootstrap is available.',
        'It fails full-root completion and full-source recall. No exact retained component deeper than one generation occurs across any view or beam candidate.',
        'One distinct primary continuum event has a coherent forest of 27 components, 26 isolated leaves and one depth-one mother; this is not deep hierarchy success.', '',
        ':download:`Phase56 original metrics <phase56-metrics.json>`; :download:`Phase56 retained full/half and beam metrics <phase56-retained-metrics.json>`;',
        ':download:`Phase56 micro/macro and exact-count metrics <phase56-aggregation-metrics.json>`; :download:`Phase56 pretraining metrics <phase56-pretraining-metrics.json>`.', '',
        'See :doc:`../../phase56` for the complete review, limitations and training receipts.', '']
    lines += _table(['Original policy metric', 'Completed stronger-parent arm'], [[k,_metric_point(v)] for k,v in arm['endpoints'].items()])
    lines += _table(['Strict gate', 'Completed arm'], [[k,v] for k,v in arm['gates'].items()])
    lines += ['Retained full and half metrics', '~' * 32, '', f"Every returned beam candidate was checked: {retained['beam_candidate_count']} coherent candidates on 20 strict events.", '']
    keys=('source_precision','source_recall','lcag_pair_accuracy','perfect_lcag','coherent_retained_forest','mother_pid_coverage','root_pid_accuracy')
    lines += _table(['Retained metric','Full','Half'], [[key, _metric_point(retained['primary']['full'][key]), _metric_point(retained['primary']['half'][key])] for key in keys])
    for scope in ('full','half'):
        lines += [f'{scope.title()} retained beam rankings', '~' * 32, '']
        lines += _table(['Ranking','LCAG','Exact components','Coherent forest'], [[rank,*[_metric_point(points[key]) for key in ('lcag_pair_accuracy','perfect_lcag','coherent_retained_forest')]] for rank,points in retained['beam'][scope].items()])
    lines += ['', 'Oracle values are truth-scored diagnostics after generation, not deployable rankings. Isolated leaves do not count as exact nontrivial LCAG components.',
        'Hold 70,000 training events: no controlled size study establishes that growth is currently better. Stop parent-dose tuning and test late pretraining leaf-PID balance (0.4 versus 0.2), with parent weight 2 and unchanged dominance fail guard.',
        'Phase57 reuses exactly Phase56 selection (1000 events), reserves 100 fresh strict events and leaves 291 untouched. This is not independent cross-phase selection replication. Validation capacity needs planning separately from training dataset growth.', '']
    return lines


def _phase57_retained_projection(value):
    raw = _mapping(value)
    if (raw.get("status") != "COMPLETE" or raw.get("version") != "phase57-retained-tree-export-v1"
        or raw.get("all_returned_beam_candidates_checked") is not True
        or raw.get("independent_validation") is not False or raw.get("strict_selection_overlap") != 100
        or raw.get("original_job_status") != "FAILED_POST_TRAINING_COHORT_AUDIT" or raw.get("sealed_test_accessed") is not False):
        raise ValueError("Phase57 retained-tree coverage is incomplete")
    result = {"version": "phase57-retained-tree-export-v1", "status": "COMPLETE",
              "source_ids": _refs("reconstruction_phase57_retained"), "independent_validation": False, "strict_selection_overlap": 100,
              "evaluator_revision": _sha(raw.get("evaluator_revision")),
              "source_hashes": [_sha(item) for item in _list(raw.get("source_hashes"))],
              "all_returned_beam_candidates_checked": True,
              "aggregate_metric_count": _integer(raw.get("aggregate_metric_count")),
              "detailed_metric_count": _integer(raw.get("detailed_metric_count")),
              "tree_metric_scalar_rows": _integer(raw.get("tree_metric_scalar_rows")),
              "tree_metric_export_sha256": _sha(raw.get("tree_metric_export_sha256")), "arms": {}}
    metrics = ("source_recall", "source_precision", "lcag_pair_accuracy", "mother_pid_coverage",
               "mother_pid_accuracy", "perfect_lcag", "coherent_retained_forest", "target_representable", "root_pid_accuracy")
    def points(value):
        output = {}
        value = _mapping(value)
        for key in metrics:
            point = _mapping(value.get(key))
            num, den = _number(point.get("numerator")), _number(point.get("denominator"))
            if num is None or den is None or num < 0 or den < num:
                raise ValueError("Invalid retained-tree metric counts")
            output[key] = {"numerator": num, "denominator": den, "value": num / den if den else None}
        for key in ("unit_count", "available_unit_count", "unavailable_unit_count", "truth_mother_count", "matched_mother_count"):
            output[key] = _integer(value.get(key))
            if output[key] is None or output[key] < 0:
                raise ValueError("Missing retained-tree coverage count")
        if output['unit_count'] != output['available_unit_count'] + output['unavailable_unit_count']:
            raise ValueError("Inconsistent retained-tree coverage count")
        output['unit_semantics_counts'] = {key: _integer(_mapping(value.get('unit_semantics_counts')).get(key, 0)) for key in ('retained_full_forest', 'retained_b_halves', 'retained_explicit_components_no_b_partition')}
        return output
    for arm in ("pretraining_balance_control", "lower_late_pid_pretraining"):
        record = _mapping(_mapping(raw.get('arms')).get(arm))
        if record.get('primary_repeat_identical') is not True:
            raise ValueError('Retained-tree repeat was not verified')
        result['arms'][arm] = {'primary_repeat_identical': True, 'beam_candidate_count': _integer(record.get('beam_candidate_count')),
            'primary_structure': {scope: {key: _integer(_mapping(_mapping(record.get('primary_structure')).get(scope)).get(key)) for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units', 'representable_nontrivial_units', 'events_without_flagged_target_incompatibility')} for scope in ('full', 'half')},
            'primary': {scope: points(_mapping(record.get('primary')).get(scope)) for scope in ('full', 'half')},
            'beam': {scope: {ranking: points(_mapping(_mapping(record.get('beam')).get(scope)).get(ranking)) for ranking in ('average_link_probability', 'learned_confidence_mean', 'learned_confidence_sum', 'normalized_joint_log_probability', 'oracle_diagnostic')} for scope in ('full', 'half')}}
    for arm in result['arms'].values():
        for scope, counts in arm['primary_structure'].items():
            if any(value is None for value in counts.values()):
                raise ValueError('Missing retained-reference composition count')
            if sum(counts[key] for key in ('isolated_leaf_units', 'single_source_composite_units', 'nontrivial_topology_units', 'source_empty_units')) != arm['primary'][scope]['unit_count']:
                raise ValueError('Retained-reference composition does not cover all units')
            if counts['representable_nontrivial_units'] > counts['nontrivial_topology_units']:
                raise ValueError('Invalid retained-reference representability count')
    registry = json.loads(Path(__file__).with_name('phase57_retained_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen:
            raise ValueError('Duplicate retained-tree metric')
        number = row.get('value')
        if number is not None and _number(number) is None and type(number) is not bool:
            raise ValueError('Invalid retained-tree scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=number))
    if seen != allowed or len(rows) != result['aggregate_metric_count']:
        raise ValueError('Incomplete retained-tree metric registry')
    result['metric_rows'] = rows
    return result


def _phase57_aggregation_projection(value):
    raw = _mapping(value)
    if (raw.get('version') != 'phase57-aggregation-supplement-v1' or raw.get('status') != 'COMPLETE'
        or raw.get('independent_validation') is not False or raw.get('strict_selection_overlap') != 100):
        raise ValueError('Missing Phase57 aggregation supplement')
    registry = json.loads(Path(__file__).with_name('phase57_aggregation_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase57 aggregation scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase57 aggregation registry')
    primary = {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'primary_complete_target_direct'} for arm in ('pretraining_balance_control', 'lower_late_pid_pretraining')}
    return {'metric_rows': rows, 'primary': primary, 'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'source_ids': _refs('reconstruction_phase57_aggregation')}


def _phase57_pretraining_projection(value):
    raw = _mapping(value)
    if (raw.get('version') != 'phase57-pretraining-export-v1' or raw.get('status') != 'COMPLETE'
        or raw.get('independent_validation') is not False):
        raise ValueError('Phase57 pretraining evidence is incomplete')
    registry = json.loads(Path(__file__).with_name('phase57_pretraining_metric_registry.json').read_text())
    allowed, seen, rows = set(map(tuple, registry)), set(), []
    for row in _list(raw.get('metric_rows')):
        identity = (row.get('arm'), row.get('view'), row.get('metric'))
        if identity not in allowed:
            continue
        if identity in seen or (row.get('value') is not None and type(row.get('value')) is not bool and _number(row.get('value')) is None):
            raise ValueError('Invalid Phase57 pretraining scalar')
        seen.add(identity)
        rows.append(dict(arm=identity[0], view=identity[1], metric=identity[2], value=row.get('value')))
    if seen != allowed:
        raise ValueError('Incomplete Phase57 pretraining metric registry')
    return {'metric_rows': rows, 'scalar_rows': _integer(raw.get('scalar_rows')),
            'source_hashes': [_sha(x) for x in _list(raw.get('source_hashes'))],
            'final_validation': {arm: {r['metric']: r['value'] for r in rows if r['arm'] == arm and r['view'] == 'validation_step_2188'} for arm in ('pretraining_balance_control', 'lower_late_pid_pretraining')},
            'source_ids': _refs('reconstruction_phase57_pretraining')}


def _render_phase57(record):
    if not record:return []
    labels=('pretraining_balance_control','lower_late_pid_pretraining');arms=record['arms'];retained=record['retained_tree_checks']['arms']
    lines=['Phase57: complete diagnostics, invalid independent validation','-'*64,'',
        'Both arms finished 2,188 refinement and 4,376 reconstruction steps, then failed the ordered-cohort audit. Original failed receipts remain unchanged.',
        'Reconstruction selected 734 intended events plus 266 unintended events, including all 100 reserved strict events. Both arms used the same actual sequence.',
        'Recovered evaluations are in-sample diagnostics. Numerical threshold checks do not certify independent gate passage; neither model is promoted.',
        'Selection recovery is 289/3533 versus 283/3533 (late leaf-PID weights 0.4 versus 0.2). This is not a held-out effect.', '',
        ':download:`Phase57 original-policy diagnostics <phase57-metrics.json>`; :download:`Phase57 all retained full/half and beam metrics <phase57-retained-metrics.json>`;',
        ':download:`Phase57 micro/macro and exact combinations <phase57-aggregation-metrics.json>`; :download:`Phase57 pretraining metrics <phase57-pretraining-metrics.json>`.',
        'See :doc:`../../phase57` for cohort failure, detailed review and corrected next-study design.','']
    lines+=_table(['Original-policy diagnostic','Late PID 0.4','Late PID 0.2'],[[k,*[_metric_point(arms[a]['endpoints'][k]) for a in labels]] for k in arms[labels[0]]['endpoints']])
    lines+=_table(['Numerical threshold check (diagnostic only)','0.4','0.2'],[[k,*[arms[a]['gates'][k] for a in labels]] for k in arms[labels[0]]['gates']])
    for scope in ('full','half'):
        lines += [f'Phase57 retained {scope} diagnostics','~'*48,'']
        keys=('source_precision','source_recall','lcag_pair_accuracy','perfect_lcag','coherent_retained_forest','mother_pid_coverage','root_pid_accuracy','target_representable')
        lines+=_table(['Retained diagnostic','0.4','0.2'],[[k,*[_metric_point(retained[a]['primary'][scope][k]) for a in labels]] for k in keys])
        lines+=_table(['Beam ranking','0.4 LCAG','0.2 LCAG','0.4 exact','0.2 exact'],[[rank,*[_metric_point(retained[a]['beam'][scope][rank]['lcag_pair_accuracy']) for a in labels],*[_metric_point(retained[a]['beam'][scope][rank]['perfect_lcag']) for a in labels]] for rank in retained[labels[0]]['beam'][scope]])
    lines += ['Phase57 pretraining execution', '~'*48, '',
        'Both refinements completed 2,188 steps and 70,000 presentations; models are identical at step 1,094. Validation below uses the reused selection cohort and is not independent reconstruction evidence.', '']
    pre=record['pretraining_metrics']['final_validation']
    keys=('validation_leaf_pid_accuracy','validation_parent_ranking_accuracy','validation_relation_accuracy','validation_fsp_only_relation_accuracy_separate','validation_effective_rank','validation_loss_parent','validation_loss_lca')
    lines+=_table(['Pretraining selection metric','Late PID 0.4','Late PID 0.2'],[[k,*[pre[a][k] for a in labels]] for k in keys])
    lines+=_table(['Arm','Returned beam candidates'],[[a,retained[a]['beam_candidate_count']] for a in labels])
    lines += [f"Phase58 submission snapshot: {_literal(record['next_study_status'])}; two bounded jobs, no automatic chain.", '']
    lines+=['','All returned beam candidates are checked in both scopes, including incompatible retained targets. Oracle values are post-generation truth diagnostics, not deployable rankings.','',
        'Hold 70,000 training events; no controlled learning curve supports immediate growth. Phase57 cannot establish a pretraining benefit independently. Repeat the intended balance comparison with corrected selection enforcement.',
        'Validation-data expansion is now a priority: only 125 events remained untouched after actual Phase57 selection. Phase58 reserves 100 fresh strict events, leaves 25, and excludes every non-selection validation UID before training. The sealed test remains closed.','']
    return lines
