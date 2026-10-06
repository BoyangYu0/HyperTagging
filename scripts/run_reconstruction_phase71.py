#!/usr/bin/env python3
"""Execute one fail-closed phase-71 reconstruction training contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (SRC, ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.run_reconstruction_phase35 import (  # noqa: E402
    _training_config,
    _validation_selection_audit,
    atomic_json,
    finite_checkpoint,
    first_twenty_gate,
    replay_slot_audit,
    sha256,
)
from hypertagging.training.fixed_validation import (  # noqa: E402
    excluded_event_uids_contract,
)
from hypertagging.training.reconstruction_trainer import (  # noqa: E402
    train_level_reconstruction,
)


CONTRACT_VERSION = "hypertagging-reconstruction-phase71-contract-v1"
STUDY_ID = "phase71-auxiliary-teacher-objective-20261006"
COHORT_STUDY_ID = STUDY_ID
ARM_ROLES = (
    "aux_teacher_050",
    "aux_teacher_100",
)
ALLOWED_SELECTION_STEPS = {1000, 2000, 3000, 4000, 4376}
CHECKPOINT_TRACKS = {
    "final": "checkpoint.pt",
    "best": "best.pt",
    "best_complete_target": "best_rollout_complete_target_efficiency.pt",
    "best_depth": "best_rollout_depth_fraction.pt",
    "best_tree_validity": "best_rollout_tree_validity.pt",
}


def canonical_contract_hash(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git command failed: {args}")
    return result.stdout.strip()


def _repo_file(value: str) -> Path:
    path = (ROOT / value).resolve(strict=True)
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"contract input escapes repository: {value}") from error
    return path


def validate_policy_plan(cohort, universe, historical, selection, diagnostic):
    """Authenticate independent reserved policy coverage before training admission."""
    binding = cohort["policy_evaluation_plan"]
    path = _repo_file(binding["path"])
    if sha256(path) != binding["sha256"]:
        raise RuntimeError("Phase71 policy plan changed")
    plan = json.loads(path.read_text())
    groups = plan["by_category"]
    categories = {"charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"}
    flat = [uid for values in groups.values() for uid in values]
    if (set(groups) != categories or any(len(set(v)) != 2000 for v in groups.values())
        or len(flat) != len(set(flat)) or len(flat) != 12000
        or not set(flat) <= universe or set(flat) & (historical | selection)
        or not diagnostic <= set(flat) or plan["sealed_test_accessed"] is not False
        or plan["status"] != "RESERVED_NOT_PROCESSED"
        or plan["required_distinct_processed_per_category"] != 2000
        or plan["scopes"] != ["full", "half"]
        or plan["checkpoint_selection_uses_primary_cohort"] is not False):
        raise RuntimeError("Phase71 category coverage or independence plan invalid")
    return {"status": "RESERVED_NOT_PROCESSED", "reserved": 12000,
            "actual_processed": 0, "final_evaluation_compliant": False}


def validation_exclusions(cohort: dict[str, Any]) -> tuple[str, ...]:
    binding = cohort["source_bindings"]["validation_universe"]
    path = _repo_file(binding["path"])
    if sha256(path) != binding["sha256"]:
        raise RuntimeError("Phase71 validation universe changed")
    universe = json.loads(path.read_text())
    all_uids = set(universe["event_uids"])
    selected = cohort["checkpoint_selection_event_uids"]
    strict = set(cohort["event_uids"])
    if (
        len(all_uids) != 160000
        or len(set(selected)) != 1000
        or not set(selected) <= all_uids
        or not strict <= all_uids
        or strict & set(selected)
    ):
        raise RuntimeError("Phase71 validation eligibility set is invalid")
    binding = cohort["source_bindings"]["previous_validation_universe"]
    old_path = _repo_file(binding["path"])
    if sha256(old_path) != binding["sha256"]:
        raise RuntimeError("Phase71 historical validation universe changed")
    historical = set(json.loads(old_path.read_text())["event_uids"])
    if (
        len(historical) != 85000
        or not historical <= all_uids
        or len(strict) != 100
        or strict & historical
        or set(selected) & historical
    ):
        raise RuntimeError(
            "Phase71 requires fresh selection and strict events from added sources"
        )
    validate_policy_plan(cohort, all_uids, historical, set(selected), strict)
    values = all_uids - set(selected)
    identity = excluded_event_uids_contract(tuple(values))
    if (
        identity["excluded_event_uid_count"] != 159000
        or identity["excluded_event_uid_count"]
        != cohort["validation_exclusion_event_uid_count"]
        or identity["excluded_event_uids_sha256"]
        != cohort["validation_exclusion_event_uids_sha256"]
    ):
        raise RuntimeError(
            "Phase71 every non-selection validation event must be excluded"
        )
    return tuple(sorted(values))


def validate_evaluation_population(evaluation, cohort):
    """Catch a stale strict/beam count before any training allocation."""
    if (
        evaluation.get("coverage_classification") != "HISTORICAL_STYLE_DIAGNOSTIC_NOT_FINAL_POLICY_COVERAGE"
        or evaluation.get("max_events") != 100
        or cohort.get("event_uid_count") != 100
        or cohort.get("evaluation_event_uid_count") != 100
        or len(cohort.get("event_uids", [])) != 100
        or cohort.get("evaluation_event_uids") != cohort.get("event_uids")
        or evaluation.get("beam_search", {}).get("max_events") != 20
    ):
        raise RuntimeError("Phase71 evaluator population differs from its bound cohort")


def preflight_selection(cohort, config, exclusions):
    """Exercise the trainer's selector against the authenticated complete UID census."""
    if config.get("scientific_mode") is not True:
        raise RuntimeError("Phase71 preflight selector requires actual scientific mode")
    from types import SimpleNamespace
    from hypertagging.training.fixed_validation import select_validation_events

    binding = cohort["source_bindings"]["validation_universe"]
    path = _repo_file(binding["path"])
    if sha256(path) != binding["sha256"]:
        raise RuntimeError("Phase71 validation universe changed")
    universe = json.loads(path.read_text())
    _events, uids, _metadata = select_validation_events(
        (SimpleNamespace(event_uid=u) for u in universe["event_uids"]),
        limit=config["max_validation_events"],
        scientific_mode=config["scientific_mode"],
        selection_manifest_hash=cohort["selection_manifest_sha256"],
        seed=config["seed"],
        excluded_event_uids=exclusions,
    )
    if tuple(cohort["checkpoint_selection_event_uids"]) != uids or set(uids) & set(
        cohort["event_uids"]
    ):
        raise RuntimeError(
            "Phase71 preflight selector differs from preregistered sequence or touches strict events"
        )
    return {
        "status": "PASS",
        "selection_events": len(uids),
        "strict_overlap": 0,
        "excluded_events": len(exclusions),
        "before_any_training": True,
    }


def validate_context_contrast(prereg):
    configs = []
    for arm, role, weight in zip(prereg["arms"], ARM_ROLES, (0.5, 1.0), strict=True):
        config = {**prereg["common_training_contract"], **arm["overrides"]}
        if arm["role"] != role or config.pop("auxiliary_teacher_weight") != weight:
            raise RuntimeError("Phase71 auxiliary teacher objective intervention differs")
        if config["type_conditioned_daughter_relation_bias"] is not True:
            raise RuntimeError("Phase71 shared decoder architecture changed")
        if config["scheduled_sampling_probability"] != 0.5 or config["freeze_pretrained_encoder_steps"] != 2188 or config["max_steps"] != 4376:
            raise RuntimeError("Phase71 context, adaptation or budget differs")
        configs.append(config)
    if configs[0] != configs[1]:
        raise RuntimeError("Phase71 changed a second scientific factor")


def audit_context_execution(log_path, arm):
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    rows = [row for row in rows if "sampled_predicted_count" in row]
    if len(rows) != 4376:
        raise RuntimeError("Phase71 incomplete exposure records")
    total = {key: sum(row[key] for row in rows) for key in (
        "sampled_predicted_count", "sampled_teacher_count", "rollout_call_count",
        "truth_target_count", "representable_target_count", "unrepresentable_target_count",
        "model_forward_count", "auxiliary_teacher_loss")}
    if total["sampled_predicted_count"] + total["sampled_teacher_count"] != 280064:
        raise RuntimeError("Phase71 exposure count differs from replay budget")
    if arm not in ARM_ROLES or total["sampled_predicted_count"] <= 0:
        raise RuntimeError("Phase71 mixed-context intervention did not execute")
    if total["auxiliary_teacher_loss"] <= 0:
        raise RuntimeError("Phase71 auxiliary teacher objective did not execute")
    weight = {"aux_teacher_050": 0.5, "aux_teacher_100": 1.0}[arm]
    return {"passed": True, "totals": total, "training_records": len(rows), "objective_weight": weight, "weighted_auxiliary_loss_sum": weight * total["auxiliary_teacher_loss"]}


def validate_closeout_basis(prereg: dict[str, Any]) -> dict[str, Any]:
    if (
        prereg.get("pilot_classification") != "TEACHER_AUXILIARY_OBJECTIVE_PILOT"
        or prereg.get("validation_budget", {}).get("strict_events") != 100
        or prereg.get("validation_budget", {}).get("remaining_untouched_after_phase71")
        != 62000
    ):
        raise RuntimeError("Phase71 requires the bounded context-exposure pilot")
    validate_context_contrast(prereg)
    if prereg["common_training_contract"]["max_cardinality"] != 17:
        raise RuntimeError("Phase71 reviewed global cardinality must be 17")
    geometry = prereg["geometry_admission"]
    path = _repo_file(geometry["path"])
    if sha256(path) != geometry["sha256"]:
        raise RuntimeError("Phase71 geometry evidence changed")
    diagnostic = json.loads(path.read_text())
    expected = {
        arm + "_" + stage + "_" + sample
        for arm in ("type_bias", "no_type_bias")
        for stage in ("refined", "selected", "final")
        for sample in ("train", "development")
    }
    if not expected <= diagnostic["models"].keys() or any(
        v["radial_saturated_fraction"] >= 0.01 or v["radial_derivative_mean"] <= 0.1
        for key in expected
        for v in diagnostic["models"][key].values()
    ):
        raise RuntimeError(
            "Phase71 requires verified persistently unsaturated geometry"
        )
    closeout = prereg["phase70_closeout_basis"]
    if prereg.get("gradient_execution_contract") != {
        "autocast_weight_cache": False,
        "trainable_pid_supervision_requires_gradient": True,
        "phase48_pid_adaptation_effect_was_not_executed": True,
    }:
        raise RuntimeError("phase71 requires gradient-safe PID execution")
    if (
        prereg.get("study_id") != STUDY_ID
        or closeout.get("classification") != "completed_pair_one_passes_original_gates"
        or closeout.get("selected_next_factor")
        != "auxiliary_teacher_weight"
        or closeout.get("sealed_test_accessed") is not False
    ):
        raise RuntimeError("phase71 closeout basis is invalid")
    return closeout


def validate_completed_basis(evidence):
    if (
        evidence.get("status") != "COMPLETED_PAIR_REVIEWED"
        or evidence.get("metric_completeness")
        != "BOTH_COMPLETED_ARMS_ALL_FOURTEEN_VIEWS"
        or evidence.get("scientific_mode") != "DECODER_RELATION_BIAS_PILOT"
        or evidence.get("strict_event_count") != 100
        or evidence.get("strict_selection_overlap") != 0
        or evidence.get("sealed_test_accessed") is not False
        or set(evidence.get("arms", {})) != {"type_bias", "no_type_bias"}
        or any(
            a.get("optimizer_steps") != 4376
            or a.get("primary_repeat_identical") is not True
            for a in evidence.get("arms", {}).values()
        )
    ):
        raise RuntimeError("Phase71 requires authenticated complete Phase70 evidence")


def validate_retained_basis(retained, retained_binding):
    validate_completed_basis(retained)
    if (
        retained.get("audit_version") != retained_binding.get("version")
        or retained.get("matched_design_verified") is not True
    ):
        raise RuntimeError("Phase71 retained-tree evidence is incomplete")
    for arm in retained["arms"].values():
        views = arm.get("view_topology_audits", {})
        if (
            len(views) != 7
            or not arm.get("retained_beam")
            or not arm.get("retained_primary")
        ):
            raise RuntimeError("Phase71 requires all prior retained views/candidates")


def validate_seed_contract(config: dict[str, Any], cohort_seed: int) -> None:
    """Reject stale replay seeds before rendering or allocating a training job."""
    if (
        type(config.get("seed")) is not int
        or config["seed"] != 20261007
        or config["seed"] != cohort_seed
        or config.get("balanced_level_replay_contract", {}).get("seed")
        != config["seed"]
    ):
        raise RuntimeError("phase71 training, replay, and cohort seeds must agree")


def verify_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("contract_version") != CONTRACT_VERSION
        or contract.get("study_id") != STUDY_ID
        or contract.get("arm_role") not in ARM_ROLES
        or contract.get("contract_sha256") != canonical_contract_hash(contract)
        or contract.get("submission_authorized") is not True
        or contract.get("submission_performed") is not False
        or contract.get("automatic_promotion") is not False
        or contract.get("sealed_test_role_access") != "forbidden"
        or contract.get("source_checkpoint_mutation") != "forbidden"
    ):
        raise RuntimeError("phase71 contract identity or authority is invalid")

    expected_sha = str(contract["expected_git_sha"])
    expected_tag = str(contract["expected_git_tag"])
    if _git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("phase71 HEAD differs from the frozen contract")
    if _git("rev-list", "-n", "1", expected_tag) != expected_sha:
        raise RuntimeError("phase71 implementation tag does not bind HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("phase71 tracked worktree is dirty")

    for binding in contract["hashed_inputs"]:
        source = _repo_file(str(binding["path"]))
        if sha256(source) != binding["sha256"]:
            raise RuntimeError(f"phase71 hashed input changed: {binding['path']}")

    prereg_path = _repo_file(contract["preregistration"]["path"])
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    closeout = validate_closeout_basis(prereg)
    evidence = _repo_file(closeout["path"])
    evidence_record = json.loads(evidence.read_text())
    if (
        sha256(evidence) != closeout["sha256"]
        or evidence_record.get("metric_completeness")
        != "BOTH_COMPLETED_ARMS_ALL_FOURTEEN_VIEWS"
        or evidence_record.get("status") != "COMPLETED_PAIR_REVIEWED"
    ):
        raise RuntimeError("phase71 closeout evidence changed")
    validate_completed_basis(evidence_record)
    expected_arm = next(
        arm for arm in prereg["arms"] if arm["role"] == contract["arm_role"]
    )
    if (
        contract["checkpoint"] != expected_arm["checkpoint"]
        or contract["checkpoint_sha256"] != expected_arm["checkpoint_sha256"]
        or contract["checkpoint_step"] != expected_arm["checkpoint_step"]
        or sha256(_repo_file(contract["checkpoint"])) != contract["checkpoint_sha256"]
        or contract["config"]
        != {**prereg["common_training_contract"], **expected_arm["overrides"]}
    ):
        raise RuntimeError("phase71 preregistered checkpoint or config changed")
    capacity = prereg.get("capacity_admission", {}).get("reports_by_arm", {})
    if set(capacity) != set(ARM_ROLES) or any(
        report.get("production_training_allowed") is not True
        or report.get("query_overflow_count") != 0
        or report.get("cardinality_overflow_count") != 0
        for report in capacity.values()
    ):
        raise RuntimeError("phase71 capacity admission is invalid")

    from scripts.run_phase71_pretraining import validate_pretraining_contract

    validate_pretraining_contract(contract, prereg)

    cohort_path = _repo_file(contract["cohort"]["path"])
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if (
        sha256(cohort_path) != contract["cohort"]["sha256"]
        or cohort.get("manifest_version")
        != "hypertagging-reconstruction-phase71-cohort-v1"
        or cohort.get("study_id") != COHORT_STUDY_ID
        or cohort.get("role") != "validation"
        or cohort.get("sealed_test_role_access") != "forbidden"
        or cohort.get("all_required_overlaps_zero") is not True
        or any(cohort.get("overlap_audit", {}).values())
        or len(cohort.get("checkpoint_selection_event_uids", [])) != 1_000
        or len(cohort.get("event_uids", [])) != 100
    ):
        raise RuntimeError(
            "phase71 fresh-selection/fresh-strict cohort contract is invalid"
        )
    validate_evaluation_population(contract["evaluation_contract"], cohort)
    preflight_selection(cohort, contract["config"], validation_exclusions(cohort))

    config = dict(contract["config"])
    validate_seed_contract(config, cohort["seed"])
    gates = dict(contract["post_training_gates"])
    retained_binding = prereg["phase70_retained_metric_basis"]
    retained_path = _repo_file(retained_binding["path"])
    if sha256(retained_path) != retained_binding["sha256"]:
        raise RuntimeError("phase71 retained-tree evidence hash changed")
    retained = json.loads(retained_path.read_text())
    validate_retained_basis(retained, retained_binding)
    evaluation_contract = dict(contract["evaluation_contract"])
    if evaluation_contract.get("retained_tree_checks") != {
        "version": "retained-direct-tree-checks-v1",
        "required": True,
        "scopes": ["full", "half"],
        "all_returned_beam_candidates": True,
        "population": "all_explicit_retained_roots_including_outside_training_policy",
        "replaces_original_gates": False,
    }:
        raise RuntimeError("phase71 requires complete retained-tree and beam checks")
    beam_contract = dict(evaluation_contract.get("beam_search", {}))
    unified_contract = dict(evaluation_contract.get("unified_full_evaluation", {}))
    if (
        config.get("max_steps") != 4_376
        or config.get("max_validation_events") != 1000
        or config.get("rollout_validation_events") != 1000
        or config.get("lr_schedule_total_steps") != 4_376
        or config.get("best_metric") != "micro_complete_target_efficiency"
        or config.get("best_mode") != "max"
        or config.get("level_sampling_mode") != "balanced_level_replay"
        or config.get("rollout_max_level") != 6
        or config.get("rollout_min_depth_fraction") != 0.0
        or config.get("rollout_min_complete_target_efficiency") != 0.0
        or float(gates.get("minimum_predicted_depth_fraction", 0.0)) <= 0.0
        or float(gates.get("minimum_complete_target_efficiency", 0.0)) <= 0.0
        or gates.get("minimum_full_root_completion_numerator") != 2
        or gates.get("minimum_full_lcag_numerator") != 2
        or gates.get("minimum_exact_mother_coverage_numerator") != 2
        or gates.get("minimum_full_source_recall") != 0.17
        or gates.get("minimum_full_source_precision") != 0.75
        or gates.get("minimum_half_source_recall") != 0.19
        or gates.get("minimum_half_source_precision") != 0.45
        or gates.get("minimum_half_lcag_numerator") != 10
        or gates.get("minimum_half_perfect_lcag_numerator") != 3
        or gates.get("minimum_half_root_pid_accuracy") != 0.02
        or gates.get("minimum_tree_validity") != 0.999
        or gates.get("minimum_p4_closure") != 1.0
        or gates.get("all_gates_required") is not True
        or evaluation_contract.get("cpu_threads") != 1
        or evaluation_contract.get("deterministic_algorithms") is not True
        or set(
            key
            for key, value in unified_contract.items()
            if key != "runner" and value is True
        )
        != {
            "strict_checkpoint_direct",
            "strict_repeat",
            "contracted_topology_diagnostic",
            "beam_search",
        }
        or unified_contract.get("runner")
        != "scripts/run_full_reconstruction_evaluation_suite.py"
        or evaluation_contract.get("primary_evaluation_repeats") != 2
        or evaluation_contract.get("require_exact_primary_repeat_equality") is not True
        or evaluation_contract.get("root_completion_stability_policy")
        != "require_at_least_two_completed_roots_in_each_identical_primary_repeat"
        or beam_contract.get("enabled") is not True
        or beam_contract.get("scope") != "full"
        or beam_contract.get("evaluated_scopes") != ["full", "half"]
        or beam_contract.get("beam_width") != 4
        or beam_contract.get("max_events") != 20
        or beam_contract.get("max_level") != 6
        or beam_contract.get("oracle_at_k_diagnostic_only") is not True
        or "average_link_probability"
        not in beam_contract.get("model_only_rankings", [])
        or contract.get("longer_run_authorized") is not False
        or contract.get("promotion_authorized") is not False
        or contract.get("sealed_test_request_authorized") is not False
    ):
        raise RuntimeError("phase71 scientific selection/gate contract changed")
    if (
        config.get("level_loss_weights")
        != [[1, 1.0], [2, 1.0], [3, 1.25], [4, 1.5], [5, 2.0], [6, 3.0]]
        or config.get("recovery_objective_weight") != 0.0
        or config.get("rollout_pointer_threshold") != 0.35
        or config.get("rollout_object_threshold") != 0.6
        or config.get("unrepresentable_target_policy") != "masked_representable_only"
        or config.get("auxiliary_teacher_weight") != {"aux_teacher_050": 0.5, "aux_teacher_100": 1.0}[contract["arm_role"]]
    ):
        raise RuntimeError("phase71 shared weighted/recovery contract changed")
    if config.get("max_cardinality_by_level") != [
        [1, 12],
        [2, 17],
        [3, 15],
        [4, 15],
        [5, 14],
        [6, 2],
    ]:
        raise RuntimeError("phase71 repaired cardinality contract changed")
    statistics_binding = prereg["fresh_statistics_binding"]
    statistics_path = _repo_file(statistics_binding["path"])
    statistics = json.loads(statistics_path.read_text())
    if (
        sha256(statistics_path) != statistics_binding["sha256"]
        or statistics.get("status") != "PASS"
        or statistics.get("normalizer_scope") != "train"
        or statistics.get("sealed_test_accessed") is not False
        or statistics.get("index_path") != contract["data"]["dataset_index"]
        or statistics.get("index_sha256")
        != sha256(_repo_file(contract["data"]["dataset_index"]))
        or any(
            sha256(_repo_file(path)) != digest
            for path, digest in statistics["source_files"].items()
        )
    ):
        raise RuntimeError("phase71 fresh corrected statistics are not bound")
    expected_freeze = {
        "aux_teacher_050": 4376,
        "aux_teacher_100": 4376,
    }[contract["arm_role"]]
    if (
        config.get("freeze_pretrained_encoder_steps")
        != {"aux_teacher_050": 2188, "aux_teacher_100": 2188}[contract["arm_role"]]
        or config.get("freeze_leaf_pid_head_steps") != expected_freeze
        or config.get("leaf_pid_lr_multiplier") != 1.0
        or config.get("encoder_lr_multiplier") != 0.05
        or config.get("resume")
        or config.get("resume_from")
    ):
        raise RuntimeError("phase71 corrected adaptation comparison changed")
    boundary = prereg.get("scientific_source_boundary", {})
    if (
        boundary.get("fresh_train_only_normalization_required") is not True
        or boundary.get("both_arms_use_corrected_training_and_inference") is not True
        or boundary.get("resume_forbidden") is not True
    ):
        raise RuntimeError("phase71 corrected-source boundary is missing")
    expected_pointer_weights = []
    if (
        config.get("pointer_positive_weight") != 32.0
        or config.get("pointer_positive_weights_by_level") != expected_pointer_weights
        or config.get("n_queries_by_level")
        != [[1, 16], [2, 8], [3, 6], [4, 4], [5, 3], [6, 2]]
        or config.get("train_split_event_count") != 70_000
        or config.get("replay_slot_budget") != 280_064
    ):
        raise RuntimeError("phase71 controlled pretraining-transfer contract changed")

    output_root = (ROOT / contract["output_root"]).resolve()
    expected_parent = (
        ROOT / "artifacts/runs/ht-reconstruction-phase71-20261006"
    ).resolve()
    if expected_parent not in output_root.parents:
        raise RuntimeError("phase71 output root escaped its namespace")
    runtime = {
        "selection_manifest": str(_repo_file(contract["data"]["selection_manifest"])),
        "dataset_index": str(_repo_file(contract["data"]["dataset_index"])),
        "checkpoint": str(_repo_file(contract["checkpoint"])),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "checkpoint_step": str(contract["checkpoint_step"]),
        "cohort": str(cohort_path),
        "output_root": str(output_root),
        "gpu_environment": str(Path(contract["gpu_environment"]).resolve(strict=True)),
    }
    return contract, runtime



def audit_decoder_intervention(training_output, enabled, auxiliary_teacher_weight):
    """Authenticate the shared architecture and actually saved objective weight."""
    import torch
    evidence = {}
    for filename in sorted(set(CHECKPOINT_TRACKS.values())):
        cp = torch.load(training_output / filename, map_location="cpu", weights_only=False)
        keys = [k for k in cp["model_state_dict"] if "type_relation_table" in k or "compatibility_query" in k]
        if cp["config"]["auxiliary_teacher_weight"] != auxiliary_teacher_weight:
            raise RuntimeError("Phase71 saved objective weight differs from treatment")
        if cp["config"]["type_conditioned_daughter_relation_bias"] is not enabled or cp["architecture"]["type_conditioned_daughter_relation_bias"] is not enabled or bool(keys) is not enabled:
            raise RuntimeError("Phase71 decoder intervention did not execute")
        expected = "type_conditioned_relation_aware" if enabled else "generic_relation_aware"
        if cp["feature_contract"]["daughter_compatibility"] != expected:
            raise RuntimeError("Phase71 decoded feature contract differs")
        evidence[filename] = {"enabled": enabled, "treatment_parameter_entries": len(keys), "step": cp["step"]}
    return evidence

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    contract, runtime = verify_contract(contract_path)
    output = args.output.resolve()
    training_output = args.training_output_dir.resolve()
    output_root = Path(runtime["output_root"])
    if output_root not in output.parents or output_root not in training_output.parents:
        raise RuntimeError("phase71 outputs escape the arm output root")
    if output.exists() or training_output.exists():
        raise RuntimeError("refusing to reuse phase71 outputs")

    cohort = json.loads(Path(runtime["cohort"]).read_text(encoding="utf-8"))
    exclusions = validation_exclusions(cohort)
    config = dict(contract["config"])
    validate_seed_contract(config, cohort["seed"])
    selection_preflight = preflight_selection(cohort, config, exclusions)
    atomic_json(
        training_output.parent / "selection-preflight.json", selection_preflight
    )
    source = Path(runtime["checkpoint"])
    source_before = sha256(source)
    started = time.perf_counter()
    from scripts.run_phase71_pretraining import run_pretraining_refinement

    refinement = run_pretraining_refinement(
        contract, runtime, cohort, training_output.parent / "pretraining"
    )
    runtime = {
        **runtime,
        "checkpoint": refinement["checkpoint"],
        "checkpoint_sha256": refinement["checkpoint_sha256"],
        "checkpoint_step": str(refinement["step"]),
    }
    result = train_level_reconstruction(
        _training_config(
            config=config,
            runtime=runtime,
            training_output=training_output,
            validation_exclusions=exclusions,
        )
    )
    elapsed = time.perf_counter() - started
    if result.steps != 4_376 or sha256(source) != source_before:
        raise RuntimeError("phase71 budget or immutable source contract failed")
    if result.transfer_report is None or result.transfer_report.coverage < 1.0:
        raise RuntimeError("phase71 encoder transfer coverage is incomplete")

    checkpoints = {
        name: finite_checkpoint((training_output / filename).resolve(strict=True))
        for name, filename in CHECKPOINT_TRACKS.items()
    }
    if checkpoints["final"]["step"] != 4_376:
        raise RuntimeError("phase71 final checkpoint is not step 4376")
    if any(
        audit["step"] not in ALLOWED_SELECTION_STEPS
        for name, audit in checkpoints.items()
        if name != "final"
    ):
        raise RuntimeError("phase71 track selected an unregistered step")

    selection_audit = _validation_selection_audit(
        training_output / CHECKPOINT_TRACKS["final"],
        exclusions=exclusions,
        expected_event_uids=tuple(cohort["checkpoint_selection_event_uids"]),
        expected_rollout_event_uids=tuple(
            cohort["checkpoint_selection_event_uids"][
                : int(config["rollout_validation_events"])
            ]
        ),
    )
    replay_audit = replay_slot_audit(
        result.log_path,
        config=config,
        trainer_contract=result.balanced_level_replay_contract,
    )
    payload = {
        "result_version": "hypertagging-reconstruction-phase71-training-v1",
        "status": "training_completed",
        "study_id": STUDY_ID,
        "task_id": contract["task_id"],
        "arm_role": contract["arm_role"],
        "contract_sha256": contract["contract_sha256"],
        "optimizer_steps": result.steps,
        "context_execution": audit_context_execution(result.log_path, contract["arm_role"]),
        "decoder_intervention": audit_decoder_intervention(training_output, config["type_conditioned_daughter_relation_bias"], config["auxiliary_teacher_weight"]),
        "selection_preflight": selection_preflight,
        "elapsed_seconds": elapsed,
        "source_checkpoint": {
            "path": str(source),
            "sha256_before": source_before,
            "sha256_after": sha256(source),
            "unchanged": True,
        },
        "pretraining_refinement": refinement,
        "checkpoints": checkpoints,
        "metrics": dict(result.metrics),
        "training_log": {
            "path": str(result.log_path.resolve(strict=True)),
            "sha256": sha256(result.log_path),
        },
        "first_20_optimizer_steps": first_twenty_gate(result.log_path),
        "balanced_level_replay": replay_audit,
        "validation_selection": selection_audit,
        "transfer_report": asdict(result.transfer_report),
        "post_training_gates": contract["post_training_gates"],
        "longer_run_authorized": False,
        "promotion_authorized": False,
        "sealed_test_request_authorized": False,
    }
    atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
