#!/usr/bin/env python3
"""Execute one fail-closed phase-41 reconstruction training contract."""

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


CONTRACT_VERSION = "hypertagging-reconstruction-phase41-contract-v1"
STUDY_ID = "phase41-level1-pointer-balance-20260909"
COHORT_STUDY_ID = STUDY_ID
ARM_ROLES = (
    "pointer32_control",
    "level1_pointer24",
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


def validation_exclusions(cohort: dict[str, Any]) -> tuple[str, ...]:
    bindings = cohort["source_bindings"]
    historical_path = _repo_file(
        bindings["historical_validation_exclusions"]["path"]
    )
    phase35_path = _repo_file(bindings["phase35_cohort"]["path"])
    phase36_path = _repo_file(bindings["phase36_cohort"]["path"])
    phase37_path = _repo_file(bindings["phase37_cohort"]["path"])
    phase38_path = _repo_file(bindings["phase38_cohort"]["path"])
    phase39_path = _repo_file(bindings["phase39_cohort"]["path"])
    phase40_path = _repo_file(bindings["phase40_cohort"]["path"])
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    phase35 = json.loads(phase35_path.read_text(encoding="utf-8"))
    phase36 = json.loads(phase36_path.read_text(encoding="utf-8"))
    phase37 = json.loads(phase37_path.read_text(encoding="utf-8"))
    phase38 = json.loads(phase38_path.read_text(encoding="utf-8"))
    phase39 = json.loads(phase39_path.read_text(encoding="utf-8"))
    phase40 = json.loads(phase40_path.read_text(encoding="utf-8"))
    values = {
        *(str(uid) for uid in historical["event_uids"]),
        *(str(uid) for uid in phase35["checkpoint_selection_event_uids"]),
        *(str(uid) for uid in phase35["event_uids"]),
        *(str(uid) for uid in phase36["checkpoint_selection_event_uids"]),
        *(str(uid) for uid in phase36["evaluation_event_uids"]),
        *(str(uid) for uid in phase37["checkpoint_selection_event_uids"]),
        *(str(uid) for uid in phase37["evaluation_event_uids"]),
        *(str(uid) for uid in phase38["checkpoint_selection_event_uids"]),
        *(str(uid) for uid in phase38["evaluation_event_uids"]),
        *(str(uid) for uid in phase39["checkpoint_selection_event_uids"]),
        *(str(uid) for uid in phase39["evaluation_event_uids"]),
        *(str(uid) for uid in phase40["checkpoint_selection_event_uids"]),
        *(str(uid) for uid in phase40["evaluation_event_uids"]),
    }
    identity = excluded_event_uids_contract(tuple(values))
    if (
        identity["excluded_event_uid_count"]
        != cohort["validation_exclusion_event_uid_count"]
        or identity["excluded_event_uids_sha256"]
        != cohort["validation_exclusion_event_uids_sha256"]
    ):
        raise RuntimeError("phase41 validation exclusion identity changed")
    return tuple(sorted(values))


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
        raise RuntimeError("phase41 contract identity or authority is invalid")

    expected_sha = str(contract["expected_git_sha"])
    expected_tag = str(contract["expected_git_tag"])
    if _git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("phase41 HEAD differs from the frozen contract")
    if _git("rev-list", "-n", "1", expected_tag) != expected_sha:
        raise RuntimeError("phase41 implementation tag does not bind HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("phase41 tracked worktree is dirty")

    for binding in contract["hashed_inputs"]:
        source = _repo_file(str(binding["path"]))
        if sha256(source) != binding["sha256"]:
            raise RuntimeError(f"phase41 hashed input changed: {binding['path']}")

    prereg_path = _repo_file(contract["preregistration"]["path"])
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    closeout = dict(prereg.get("phase40r1_closeout_basis", {}))
    if (
        prereg.get("study_id") != STUDY_ID
        or closeout.get("classification")
        != "completed_controlled_study_no_promotion"
        or closeout.get("selected_next_factor")
        != "level_1_pointer_positive_weight_32_to_24"
        or closeout.get("query_scale_continuation_authorized") is not False
        or closeout.get("longer_run_authorized") is not False
        or closeout.get("sealed_test_accessed") is not False
    ):
        raise RuntimeError("phase41 closeout basis is invalid")
    threshold_binding = dict(closeout.get("threshold_diagnostic", {}))
    threshold_diagnostic = _repo_file(threshold_binding.get("path", ""))
    if (
        sha256(threshold_diagnostic) != threshold_binding.get("sha256")
        or threshold_binding.get("phase41_policy")
        != "PREREGISTERED_FRESH_COHORT_CANDIDATE_ONLY"
    ):
        raise RuntimeError("phase41 threshold diagnostic binding changed")
    for evidence in closeout.get("evidence", []):
        receipt = _repo_file(evidence["receipt"])
        training_result = _repo_file(evidence["training_result"])
        full_decay_gates = _repo_file(evidence["full_decay_gates"])
        if (
            sha256(receipt) != evidence["receipt_sha256"]
            or sha256(training_result) != evidence["training_result_sha256"]
            or sha256(full_decay_gates) != evidence["full_decay_gates_sha256"]
            or evidence.get("optimizer_steps") != 4_376
            or evidence.get("all_gates_passed") is not False
            or evidence.get("sealed_test_accessed") is not False
        ):
            raise RuntimeError("phase41 phase40r1 evidence changed")
    capacity = prereg.get("capacity_admission", {}).get("reports_by_arm", {})
    if set(capacity) != set(ARM_ROLES) or any(
        report.get("production_training_allowed") is not True
        or report.get("query_overflow_count") != 0
        or report.get("cardinality_overflow_count") != 0
        for report in capacity.values()
    ):
        raise RuntimeError("phase41 capacity admission is invalid")

    cohort_path = _repo_file(contract["cohort"]["path"])
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if (
        sha256(cohort_path) != contract["cohort"]["sha256"]
        or cohort.get("manifest_version")
        != "hypertagging-reconstruction-phase41-cohort-v1"
        or cohort.get("study_id") != COHORT_STUDY_ID
        or cohort.get("role") != "validation"
        or cohort.get("sealed_test_role_access") != "forbidden"
        or cohort.get("all_required_overlaps_zero") is not True
        or any(cohort.get("overlap_audit", {}).values())
        or len(cohort.get("checkpoint_selection_event_uids", [])) != 2_000
        or len(cohort.get("event_uids", [])) != 100
    ):
        raise RuntimeError("phase41 untouched cohort contract is invalid")
    validation_exclusions(cohort)

    config = dict(contract["config"])
    gates = dict(contract["post_training_gates"])
    evaluation_contract = dict(contract["evaluation_contract"])
    beam_contract = dict(evaluation_contract.get("beam_search", {}))
    unified_contract = dict(
        evaluation_contract.get("unified_full_evaluation", {})
    )
    if (
        config.get("max_steps") != 4_376
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
        or evaluation_contract.get("require_exact_primary_repeat_equality")
        is not True
        or evaluation_contract.get("root_completion_stability_policy")
        != "require_at_least_two_completed_roots_in_each_identical_primary_repeat"
        or beam_contract.get("enabled") is not True
        or beam_contract.get("scope") != "full"
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
        raise RuntimeError("phase41 scientific selection/gate contract changed")
    if (
        config.get("level_loss_weights")
        != [[1, 1.0], [2, 1.0], [3, 1.25], [4, 1.5], [5, 2.0], [6, 3.0]]
        or config.get("recovery_objective_weight") != 2.0
        or config.get("rollout_pointer_threshold") != 0.35
        or config.get("rollout_object_threshold") != 0.6
        or config.get("unrepresentable_target_policy") != "recovery_objective"
        or config.get("auxiliary_teacher_weight") != 0.50
    ):
        raise RuntimeError("phase41 shared weighted/recovery contract changed")
    if config.get("max_cardinality_by_level") != [
        [1, 9], [2, 16], [3, 15], [4, 15], [5, 13], [6, 2]
    ]:
        raise RuntimeError("phase41 repaired cardinality contract changed")
    expected_pointer_weights = (
        []
        if contract["arm_role"] == "pointer32_control"
        else [[1, 24.0]]
    )
    if (
        config.get("pointer_positive_weight") != 32.0
        or config.get("pointer_positive_weights_by_level")
        != expected_pointer_weights
        or config.get("n_queries_by_level")
        != [[1, 16], [2, 8], [3, 6], [4, 4], [5, 3], [6, 2]]
        or config.get("train_split_event_count") != 70_000
        or config.get("replay_slot_budget") != 280_064
    ):
        raise RuntimeError("phase41 controlled pointer-balance contract changed")

    output_root = (ROOT / contract["output_root"]).resolve()
    expected_parent = (ROOT / "artifacts/runs/ht-reconstruction-phase41-20260909").resolve()
    if expected_parent not in output_root.parents:
        raise RuntimeError("phase41 output root escaped its namespace")
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
        raise RuntimeError("phase41 outputs escape the arm output root")
    if output.exists() or training_output.exists():
        raise RuntimeError("refusing to reuse phase41 outputs")

    cohort = json.loads(Path(runtime["cohort"]).read_text(encoding="utf-8"))
    exclusions = validation_exclusions(cohort)
    config = dict(contract["config"])
    source = Path(runtime["checkpoint"])
    source_before = sha256(source)
    started = time.perf_counter()
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
        raise RuntimeError("phase41 budget or immutable source contract failed")
    if result.transfer_report is None or result.transfer_report.coverage < 1.0:
        raise RuntimeError("phase41 encoder transfer coverage is incomplete")

    checkpoints = {
        name: finite_checkpoint((training_output / filename).resolve(strict=True))
        for name, filename in CHECKPOINT_TRACKS.items()
    }
    if checkpoints["final"]["step"] != 4_376:
        raise RuntimeError("phase41 final checkpoint is not step 4376")
    if any(
        audit["step"] not in ALLOWED_SELECTION_STEPS
        for name, audit in checkpoints.items()
        if name != "final"
    ):
        raise RuntimeError("phase41 track selected an unregistered step")

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
        "result_version": "hypertagging-reconstruction-phase41-training-v1",
        "status": "training_completed",
        "study_id": STUDY_ID,
        "task_id": contract["task_id"],
        "arm_role": contract["arm_role"],
        "contract_sha256": contract["contract_sha256"],
        "optimizer_steps": result.steps,
        "elapsed_seconds": elapsed,
        "source_checkpoint": {
            "path": str(source),
            "sha256_before": source_before,
            "sha256_after": sha256(source),
            "unchanged": True,
        },
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
