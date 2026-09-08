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
    "cpu_workflow": ".github/workflows/cpu-tests.yml",
}
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
    recommendation = re.search(r"(?m)^## Recommendation:\s*(NO[-_]GO|GO)\s*$", current) if isinstance(current, str) else None
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
    science = {"source_ids": _refs("verification_runs", "notebook_registry"),
               "real_pilot": _notebook_record(notebook_runs.get("real_mdst_pilot"))["result"],
               "trained_physics": _notebook_record(notebook_runs.get("trained_physics_validation"))["result"],
               "live_scheduler_status": "UNKNOWN_NOT_QUERIED", "live_training_status": "UNKNOWN_NOT_QUERIED"}
    return {"audit": audit, "verification": verification, "notebooks": notebooks, "pretraining": pretraining,
            "reconstruction": reconstruction, "science": science, "cpu_ci": _ci_projection(payloads.get("cpu_workflow"))}


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
    provenance, audit = manifest["provenance"], manifest["audit"]
    verification, notebooks = manifest["verification"], manifest["notebooks"]
    latest, pretraining, reconstruction = verification["latest_record"], manifest["pretraining"], manifest["reconstruction"]
    science, ci, pytest = manifest["science"], manifest["cpu_ci"], latest["pytest"]
    pytest_result = pytest["result"]
    pytest_kind = "blocked" if pytest_result == "FAIL" else "info" if pytest_result == "PASS" else "warning"
    pytest_failed = (
        "failed count UNKNOWN"
        if pytest["failed"] is None
        else f"{pytest['failed']} failed"
    )
    fraction = pretraining["progress_fraction"]
    progress_text = "UNKNOWN" if fraction is None else f"{fraction:.0%}"
    progress = (pretraining["recorded_step"], pretraining["planned_steps"]) if fraction is not None else None
    delta, required = reconstruction["edge_f1_delta"], reconstruction["required_edge_f1_delta"]
    comparison = "below" if delta is not None and required is not None and delta < required else "compared with"
    cards = [
        ("Production recommendation", audit["recommendation"], ["Recorded audit recommendation; this build grants no production authorization.",
            f"Audit revision: {audit['revision_match']}. Verification revision: {latest['revision_match']}."], audit["source_ids"], "blocked", None),
        ("Latest recorded CPU pytest", f"{pytest_result}: {_display(pytest['passed'])} passed", [
            f"{pytest_failed}; {_display(pytest['skipped'])} skipped; {_display(pytest['warnings'])} warnings; {_display(pytest['duration_seconds'])} seconds.",
            f"Recorded {latest['date'] or 'UNKNOWN'}. CPU software/fixture evidence; no inference of trained physics quality."], verification["source_ids"], pytest_kind, None),
        ("Notebook registry", f"{_display(notebooks['total'])} registered", [
            f"{_display(notebooks['default_smoke'])} default smoke; {notebooks['input_modes'].get('fixture', 0)} fixture; {notebooks['input_modes'].get('real_only', 0)} real only.",
            f"Registry NOT_RUN: {_display(notebooks['not_run'])}. Human visual review: {notebooks['visual_review_status']}."], notebooks["source_ids"], "warning", None),
        ("Pretraining", progress_text, [
            f"{_display(pretraining['recorded_step'])} / {_display(pretraining['planned_steps'])} steps = {progress_text}; scientific-contract resume step against its final validation milestone.",
            f"Calibration {pretraining['calibration_status']}: {_display(pretraining['calibration_pending'])} candidates pending.",
            f"Selected profile: {pretraining['selected_profile_state']}; production authorization: {_display(pretraining['production_submission_authorized'])}; submission performed: {_display(pretraining['submission_performed'])}.",
            "This is a metadata snapshot, not live training progress or evidence of convergence."], pretraining["source_ids"], "warning", progress),
        ("Reconstruction Stage A", reconstruction["completion"], [f"Recommendation: {reconstruction['recommendation']}.",
            f"Edge F1 delta {_signed(delta)} {comparison} required {_signed(required)}.",
            "Recorded aggregate exploratory evidence; paired event evidence and complete promotion guards are not established." if reconstruction["paired_event_evidence"] is False else "Projected evidence fields do not establish scientific validity."], reconstruction["source_ids"], "blocked", None),
        ("Real data and trained physics", f"Pilot {science['real_pilot']}", [f"Trained physics validation {science['trained_physics']}.",
            "Latest recorded validation statuses; bounded training receipts do not replace these scientific evaluations.",
            "Schedulers and live training are not queried."], science["source_ids"], "warning", None),
        ("Existing CPU CI evidence", ci["evidence_limitation"], [f"{_display(ci['without_explicit_pipefail'])} of {_display(ci['tee_pipeline_steps'])} tee pipeline steps lack explicit pipefail protection.",
            "An upstream test failure can be masked by a successful tee exit; an apparent green job alone is insufficient evidence." if ci['evidence_limitation'] == 'TEE_WITHOUT_PIPEFAIL' else "Static workflow inspection only; no CI run was queried."], ci["source_ids"], "warning", None),
    ]
    lines = ["Development and training status", "===============================", "",
             "This static dashboard projects selected recorded metadata. It does not run tests,",
             "train a model, query a scheduler, or inspect data or saved model state.",
             "CPU fixture evidence and unverified claims about real data and trained physics",
             "remain separate. Recorded permissions are not live authorization.", "",
             ":download:`Redacted status projection and hashes <status.json>`.", ""]
    if audit["revision_match"] == "different_revision" or latest["revision_match"] == "different_revision":
        lines += [".. warning::", "", "   STALE REVISION EVIDENCE: the audit and/or latest verification record describes",
                  "   another source revision. Recorded passes do not verify this checkout.",
                  f"   Audit: {audit['revision_match']}. Verification: {latest['revision_match']}.", ""]
    lines += [".. raw:: html", "", '   <section class="status-dashboard" aria-label="Recorded development and training dashboard">']
    for index, (title, value, paragraphs, refs, kind, card_progress) in enumerate(cards):
        lines.extend("   " + line for line in _card(title, value, paragraphs, refs, kind, index, card_progress).splitlines())
    lines += ["   </section>", "", ".. only:: not html", ""]
    for title, value, paragraphs, refs, kind, card_progress in cards:
        lines += [f"   **{_literal(title)}: {_literal(value)}**", ""]
        lines.extend(f"   {_literal(paragraph)}" for paragraph in paragraphs)
        lines += [f"   Provenance: {_literal(', '.join(refs))}.", ""]
    lines += ["Issue distribution", "------------------", "", f"{_literal(audit['ledger_item_count'])} recorded issues. Counts describe the audit scope,",
              "not the current checkout's test results or scientific readiness.", ""]
    lines += _table(["Recorded status", "Count"], [[key, value] for key, value in audit["ledger_status_counts"].items()])
    lines += ["Notebook evidence", "-----------------", "", "Registry counts describe available notebooks. Execution records describe runs;",
              "NOT_RUN and NOT_REVIEWED remain distinct from failure and pass.", ""]
    lines += _table(["Registry group", "Count"], [[key, value] for key, value in notebooks["groups"].items()])
    lines += _table(["Latest recorded notebook check", "Result", "Count"], [[key, value["result"], value["count"]] for key, value in latest["notebooks"].items()])
    lines += ["Freshness and provenance", "------------------------", "",
              f"* Checkout revision: {_literal(provenance['git_revision'])}.",
              f"* Reproducible reference time: {_literal(provenance['as_of'])}; basis: {_literal(provenance['as_of_basis'])}.",
              f"* Evidence-set SHA-256: {_literal(provenance['source_set_sha256'])}.", "",
              f"Age freshness uses a {FRESHNESS_DAYS}-day editorial threshold against the reference time.",
              "The default reference is checkout commit time; SOURCE_DATE_EPOCH can provide an",
              "explicit reproducible reference. Wall-clock and file modification times are not",
              "used. Age freshness never overrides revision mismatch warnings. Missing dates",
              "or history produce unknown freshness. after_reference_date means recorded evidence",
              "postdates the reference. File commit dates describe metadata history, not an",
              "experiment date. A heading date does not establish whole-document verification.", "",
              "Opaque source IDs map the cards to SHA-256 hashes of exact local metadata inputs.",
              "These field-level projections omit source bodies, operational identifiers, and",
              "source locations. A hash binds input bytes; it does not validate their claims.", ""]
    for source in manifest["sources"].values():
        label = source["source_id"]
        lines += [label, "~" * len(label), ""]
        lines += _table(["Property", "Value"], [[key, source.get(key)] for key in (
            "availability", "sha256", "recorded_date", "recorded_date_basis", "last_commit", "last_commit_date", "worktree_state"
        )] + [["age_freshness", source["freshness"]["status"]], ["age_days", source["freshness"]["age_days"]]])
    return "\n".join(lines)


def generate_status(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write only ``index.rst`` and allowlisted ``status.json``; copy no evidence.

    Git is optional. Builds never sample wall-clock time, import scientific
    dependencies, read external run artifacts, or execute tests/training.
    """
    root, requested = Path(repo_root).resolve(), Path(output_dir).absolute()
    output = requested.resolve()
    if output == root or output in root.parents or any(part.is_symlink() for part in (requested, *requested.parents)):
        raise ValueError("Status output must be a dedicated non-symlink directory")
    if output.exists() and any(path.name not in {"index.rst", "status.json"} for path in output.iterdir()):
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
    _write(output / "status.json", (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    _write(output / "index.rst", _render(manifest).encode())
    return manifest
