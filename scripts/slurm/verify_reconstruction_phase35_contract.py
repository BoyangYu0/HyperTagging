#!/usr/bin/env python3
"""Fail-closed verification for phase-35 reconstruction training contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    "configs/reconstruction/ht_reconstruction_phase35_improvement_20260904.json"
)
PARENT_STUDY_ID = "phase35-full-decay-reconstruction-improvement-20260904"
REPAIR_PREREGISTRATION = (
    "configs/reconstruction/ht_reconstruction_phase35_masked_repair_20260904.json"
)
REPAIR_PREREGISTRATION_VERSION = (
    "hypertagging-reconstruction-phase35-masked-repair-v1"
)
CONTRACT_VERSION = "hypertagging-reconstruction-phase35-masked-repair-contract-v1"
STUDY_ID = "phase35-masked-context-repair-20260904"
IMPLEMENTATION_TAG = "ht-reconstruction-phase35-masked-repair-20260904-v1"
OUTPUT_NAMESPACE = "artifacts/runs/ht-reconstruction-phase35-masked-repair-20260904"
WRAPPER = "scripts/slurm/run_reconstruction_phase35.sbatch"
CAMPAIGN_SUBMISSION_RECEIPT = (
    "artifacts/codex/"
    "reconstruction_phase35_masked_repair_submission_20260904.json"
)
CAMPAIGN_SUBMISSION_RECEIPT_VERSION = (
    "hypertagging-reconstruction-phase35-masked-repair-submission-v1"
)
PHASE35_TEST_RECEIPT = (
    "artifacts/codex/"
    "reconstruction_phase35_masked_repair_test_receipt_20260904.json"
)
PHASE35_SOURCE_FILES = (
    PREREGISTRATION,
    REPAIR_PREREGISTRATION,
    "scripts/build_reconstruction_validation_exclusions.py",
    "scripts/build_reconstruction_phase35_evaluation_cohort.py",
    "scripts/run_reconstruction_phase35.py",
    "scripts/run_reconstruction_phase35_full_decay.py",
    "scripts/evaluate_full_decay.py",
    "scripts/slurm/render_reconstruction_phase35_job.py",
    "scripts/slurm/submit_reconstruction_phase35_campaign.py",
    "scripts/slurm/verify_reconstruction_phase35_contract.py",
    WRAPPER,
    "scripts/slurm/finalize_reconstruction_fullscale_receipt.py",
    "scripts/slurm/preflight_gpu_environment.py",
    "scripts/slurm/monitor_gpu_telemetry.py",
    "scripts/slurm/run_with_bounded_requeue.sh",
    "src/hypertagging/training/reconstruction_trainer.py",
    "src/hypertagging/training/fixed_validation.py",
    "src/hypertagging/training/checkpoint_selection.py",
    "src/hypertagging/training/pretrained_transfer.py",
    "src/hypertagging/training/checkpointing.py",
    "src/hypertagging/training/data_module.py",
    "src/hypertagging/training/learning_rate.py",
    "src/hypertagging/training/scheduled_sampling.py",
    "src/hypertagging/evaluation/trained_context.py",
    "src/hypertagging/evaluation/full_decay_metrics.py",
    "src/hypertagging/evaluation/full_decay_runner.py",
    "src/hypertagging/reconstruction/hierarchical_inference.py",
    "src/hypertagging/models/level_autoregressive.py",
    "src/hypertagging/models/mother_pointer.py",
    "src/hypertagging/losses/level_reconstruction.py",
    "src/hypertagging/reconstruction/level_rollout.py",
    "environment/gpu/runtime-contract.json",
    "environment/gpu/requirements-cu126.lock",
)
PHASE35_ARM_ROLES = (
    "depth_balanced_masked_aux_encoder_adapt",
    "depth_balanced_masked_aux_frozen",
)
PHASE35_HIERARCHY_SUPERVISION = {
    "target_levels": [1, 2, 3, 4, 5, 6],
    "eligible_mother_counts_by_level": {
        "1": 75460,
        "2": 32095,
        "3": 20559,
        "4": 11666,
        "5": 6568,
        "6": 885,
    },
    "eligible_event_pool_counts_by_level": {
        "1": 27620,
        "2": 16160,
        "3": 12961,
        "4": 10001,
        "5": 6542,
        "6": 885,
    },
    "upsilon4s_root_count": 10000,
    "upsilon4s_roots_by_level": {
        "3": 284,
        "4": 3169,
        "5": 5662,
        "6": 885,
    },
    "continuum_forest_roots": {
        "root_count": 37200,
        "event_count": 17796,
        "continuum_event_count": 25000,
    },
    "artificial_continuum_resonance_token": False,
    "inference_seed": "detector_fsps_only",
    "truth_targets": "eligible_complete_mothers_all_levels",
}
PHASE35_CAMPAIGN_PERMISSIONS = {
    "training_authorized": True,
    "validation_payload_access_authorized": True,
    "execution_authorized": True,
    "scheduler_authorized": True,
    "submission_authorized": True,
    "scientific_validation_authorized": True,
    "source_checkpoint_mutation_authorized": False,
    "sealed_test_access_authorized": False,
    "promotion_authorized": False,
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str, *, suffix: str | None = None) -> Path:
    path = (Path(value) if Path(value).is_absolute() else ROOT / value).resolve(
        strict=True
    )
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"contract path escapes repository: {value}") from error
    if suffix is not None and path.suffix != suffix:
        raise RuntimeError(f"unexpected input suffix for {value}")
    return path


def safe_repo_output_path(
    value: str | Path,
    *,
    expected_relative: str | None = None,
    require_file: bool = False,
) -> Path:
    """Resolve a repository output without following namespace symlinks."""
    raw = Path(value)
    candidate = raw if raw.is_absolute() else ROOT / raw
    root_absolute = Path(os.path.abspath(ROOT))
    absolute = Path(os.path.abspath(candidate))
    try:
        relative = absolute.relative_to(root_absolute)
    except ValueError as error:
        raise RuntimeError(f"output path escapes repository: {value}") from error
    if expected_relative is not None and relative.as_posix() != expected_relative:
        raise RuntimeError(
            f"output path differs from reserved namespace: {relative}"
        )
    cursor = root_absolute
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise RuntimeError(f"output namespace contains a symlink: {cursor}")
    if require_file and not absolute.is_file():
        raise RuntimeError(f"required output file is missing: {absolute}")
    if absolute.exists():
        try:
            absolute.resolve(strict=True).relative_to(ROOT.resolve())
        except ValueError as error:
            raise RuntimeError(f"output path escapes repository: {value}") from error
    return absolute


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def verify_clean_worktree() -> None:
    status = git("status", "--porcelain", "--untracked-files=all")
    disallowed: list[str] = []
    for row in status.splitlines():
        if not row:
            continue
        state = row[:2]
        path = row[3:]
        if state == "??" and (
            path.startswith("runtime_inputs/") or path.startswith("artifacts/")
        ):
            continue
        disallowed.append(row)
    if disallowed:
        raise RuntimeError(
            f"tracked or executable-source worktree is dirty: {disallowed[:8]}"
        )


def canonical_contract_hash(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def phase35_contract_relative_path(arm_role: str) -> str:
    if arm_role not in PHASE35_ARM_ROLES:
        raise RuntimeError(f"phase35 campaign arm set changed: {arm_role}")
    return (
        f"artifacts/codex/reconstruction_phase35_repair_{arm_role}_"
        "contract_20260904.json"
    )


def phase35_task_id(arm_role: str) -> str:
    if arm_role not in PHASE35_ARM_ROLES:
        raise RuntimeError(f"phase35 campaign arm set changed: {arm_role}")
    return f"phase35-repair-{arm_role}-20260904"


def phase35_submission_command(
    contract: dict[str, Any], contract_path: Path
) -> list[str]:
    resources = contract["resources"]
    return [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--hold",
        "--account=others",
        "--partition=inter",
        f"--gres={resources['gres']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem={resources['memory']}",
        f"--time={resources['time']}",
        "--no-requeue",
        "--export=NIL",
        f"--job-name={contract['task_id']}",
        f"--comment=phase35:{contract['contract_sha256']}",
        WRAPPER,
        str(contract_path),
    ]


def verify_contract_hash(contract: dict[str, Any]) -> str:
    expected = str(contract.get("contract_sha256", ""))
    actual = canonical_contract_hash(contract)
    if not HEX64.fullmatch(expected) or expected != actual:
        raise RuntimeError("phase35 reconstruction contract hash mismatch")
    return actual


def verify_hashed_inputs(
    items: list[dict[str, str]], *, expected_paths: set[str]
) -> None:
    if not items:
        raise RuntimeError("phase35 contract has no hashed inputs")
    seen: set[str] = set()
    for item in items:
        value = str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        if value in seen or value not in expected_paths or Path(value).is_absolute():
            raise RuntimeError(f"duplicate hashed input: {value}")
        seen.add(value)
    if seen != expected_paths:
        raise RuntimeError(
            "phase35 hashed input allowlist changed: "
            f"missing={sorted(expected_paths - seen)} "
            f"extra={sorted(seen - expected_paths)}"
        )
    for item in items:
        value = str(item["path"])
        expected = str(item["sha256"])
        if not HEX64.fullmatch(expected) or sha256(repo_path(value)) != expected:
            raise RuntimeError(f"hashed phase35 input changed: {value}")


def load_preregistration() -> dict[str, Any]:
    payload = json.loads(repo_path(PREREGISTRATION, suffix=".json").read_text())
    if (
        payload.get("preregistration_version")
        != "hypertagging-reconstruction-phase35-improvement-v1"
        or payload.get("study_id") != PARENT_STUDY_ID
        or payload.get("study_classification")
        != "exploratory_validation_performance_screen"
    ):
        raise RuntimeError("unsupported parent phase35 preregistration")
    return payload


def load_repair_preregistration() -> dict[str, Any]:
    payload = json.loads(
        repo_path(REPAIR_PREREGISTRATION, suffix=".json").read_text(
            encoding="utf-8"
        )
    )
    if (
        payload.get("repair_preregistration_version")
        != REPAIR_PREREGISTRATION_VERSION
        or payload.get("study_id") != STUDY_ID
        or payload.get("parent_study_id") != PARENT_STUDY_ID
        or payload.get("parent_git_sha")
        != "91f1a43e8827c67b4cd4e3d8d36279321f6b9dee"
        or payload.get("parent_git_tag")
        != "ht-reconstruction-phase35-improvement-20260904-v3"
        or payload.get("parent_preregistration")
        != {
            "path": PREREGISTRATION,
            "sha256": sha256(repo_path(PREREGISTRATION)),
        }
        or payload.get("eligible_arm_roles") != list(PHASE35_ARM_ROLES)
        or payload.get("excluded_completed_arm_role")
        != "depth_balanced_fallback_frozen"
        or payload.get("excluded_completed_arm_rerun_authorized") is not False
        or payload.get("clean_restart_required") is not True
        or payload.get("resume_from_failed_attempt_authorized") is not False
        or payload.get("reuse_failed_output_authorized") is not False
        or payload.get("sealed_test_role_access") != "forbidden"
        or payload.get("automatic_promotion") is not False
        or payload.get("repair_scope")
        != {
            "checkpoint_metadata_nonfinite_fix_required": True,
            "masked_context_node_padding_fix_required": True,
            "scientific_configuration_changes_authorized": False,
        }
        or payload.get("hierarchy_supervision")
        != PHASE35_HIERARCHY_SUPERVISION
    ):
        raise RuntimeError("unsupported phase35 repair preregistration")

    expected_resources = {
        "task_count": 2,
        "gres": "gpu:h100nvl:1",
        "cpus_per_task": 8,
        "memory": "64G",
        "time": "24:00:00",
        "requeue": False,
        "maximum_restarts": 0,
        "global_concurrency": 2,
    }
    if payload.get("execution_policy") != expected_resources:
        raise RuntimeError("phase35 repair execution policy changed")

    source = payload.get("restart_source_checkpoint")
    if not isinstance(source, dict):
        raise RuntimeError("phase35 repair restart source is missing")
    source_path = repo_path(str(source.get("path", "")), suffix=".pt")
    if (
        source.get("path")
        != "runtime_inputs/reconstruction_phase34_20260904/checkpoint-step-81096.pt"
        or source.get("step") != 81096
        or source.get("sha256")
        != "98e461ad5c5d0a82ce312f4e2c6e67f6f40212d9f0df038cae315296ec990869"
        or sha256(source_path) != source.get("sha256")
    ):
        raise RuntimeError("phase35 repair restart source changed")

    evidence = payload.get("historical_attempts")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise RuntimeError("phase35 repair historical attempt set changed")
    expected_attempts = {
        "depth_balanced_fallback_frozen": {
            "job_id": "16282571",
            "task_id": "phase35-depth_balanced_fallback_frozen-20260904",
            "contract_sha256": (
                "9de3a764468f26b3e0c847866fced743af62a72f06cf4097fb4ab6a632a0c86e"
            ),
            "status": "completed",
            "terminal_stage": "full_decay_complete",
            "batch_exit_status": 0,
            "rerun_authorized": False,
        },
        "depth_balanced_masked_aux_encoder_adapt": {
            "job_id": "16282587",
            "task_id": (
                "phase35-depth_balanced_masked_aux_encoder_adapt-20260904"
            ),
            "contract_sha256": (
                "5cd05ca3b4953c84abef56ef0ec82e738f3c4b7ba099942ea03141d69a83618d"
            ),
            "status": "failed_or_nonterminal",
            "terminal_stage": "trainer_failed",
            "batch_exit_status": 1,
            "rerun_authorized": True,
        },
        "depth_balanced_masked_aux_frozen": {
            "job_id": "16282601",
            "task_id": "phase35-depth_balanced_masked_aux_frozen-20260904",
            "contract_sha256": (
                "7e1c79aec1e405178248f79ecbfb68a38bcdfbf0bf6d8f80087993ff4dca1753"
            ),
            "status": "failed_or_nonterminal",
            "terminal_stage": "trainer_failed",
            "batch_exit_status": 1,
            "rerun_authorized": True,
        },
    }
    by_role: dict[str, dict[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise RuntimeError("phase35 repair historical attempt is malformed")
        role = str(item.get("arm_role", ""))
        if role in by_role or role not in expected_attempts:
            raise RuntimeError("phase35 repair historical attempt role changed")
        by_role[role] = item
        contract_binding = item.get("contract")
        receipt_binding = item.get("attempt_receipt")
        error_binding = item.get("stderr")
        if not all(
            isinstance(binding, dict)
            for binding in (contract_binding, receipt_binding, error_binding)
        ):
            raise RuntimeError("phase35 repair evidence binding is malformed")
        # Avoid relying on assertions for fail-closed validation: optimized
        # Python removes ``assert`` statements entirely.
        contract_binding = dict(contract_binding)
        receipt_binding = dict(receipt_binding)
        error_binding = dict(error_binding)
        contract_path = repo_path(str(contract_binding.get("path", "")))
        receipt_path = repo_path(str(receipt_binding.get("path", "")))
        error_path = repo_path(str(error_binding.get("path", "")))
        if (
            sha256(contract_path) != contract_binding.get("sha256")
            or sha256(receipt_path) != receipt_binding.get("sha256")
            or sha256(error_path) != error_binding.get("sha256")
        ):
            raise RuntimeError("phase35 repair historical evidence changed")
        historical_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        attempt_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = expected_attempts[role]
        slurm = attempt_receipt.get("slurm")
        source_checkpoint = attempt_receipt.get("source_checkpoint")
        if (
            item.get("job_id") != expected["job_id"]
            or historical_contract.get("contract_sha256")
            != expected["contract_sha256"]
            or historical_contract.get("arm_role") != role
            or historical_contract.get("task_id") != expected["task_id"]
            or canonical_contract_hash(historical_contract)
            != expected["contract_sha256"]
            or attempt_receipt.get("receipt_sha256")
            != receipt_binding.get("receipt_sha256")
            or attempt_receipt.get("contract_sha256")
            != expected["contract_sha256"]
            or any(attempt_receipt.get(key) != expected[key] for key in (
                "status",
                "terminal_stage",
                "batch_exit_status",
                "task_id",
            ))
            or not isinstance(slurm, dict)
            or str(slurm.get("job_id"))
            != expected["job_id"]
            or str(slurm.get("restart_count")) != "0"
            or not isinstance(source_checkpoint, dict)
            or source_checkpoint.get("live_hash_matches_contract") is not True
            or item.get("rerun_authorized") != expected["rerun_authorized"]
        ):
            raise RuntimeError("phase35 repair historical attempt identity changed")

    if set(by_role) != set(expected_attempts):
        raise RuntimeError("phase35 repair historical attempt set changed")
    parent_submission = payload.get("parent_campaign_submission_receipt")
    if not isinstance(parent_submission, dict):
        raise RuntimeError("phase35 repair parent campaign receipt is missing")
    parent_receipt_path = repo_path(str(parent_submission.get("path", "")))
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    if (
        sha256(parent_receipt_path) != parent_submission.get("sha256")
        or parent_receipt.get("receipt_sha256")
        != parent_submission.get("receipt_sha256")
        or parent_receipt.get("status") != "submitted"
        or parent_receipt.get("study_id") != PARENT_STUDY_ID
        or parent_receipt.get("git_sha")
        != "91f1a43e8827c67b4cd4e3d8d36279321f6b9dee"
        or parent_receipt.get("git_tag")
        != "ht-reconstruction-phase35-improvement-20260904-v3"
        or parent_receipt.get("submitted_job_ids")
        != ["16282571", "16282587", "16282601"]
    ):
        raise RuntimeError("phase35 repair parent campaign receipt changed")

    expected_diagnostic_reports = [
        {
            "arm_role": "depth_balanced_masked_aux_frozen",
            "checkpoint_sha256": (
                "02db81fd0bff1a7047d263021ec02522ffc6e533ef37599bec099df08d87d622"
            ),
            "checkpoint_step": 500,
            "path": (
                "artifacts/evaluation/phase35_failed_arm_step500_clean_v3_"
                "20260904/masked_frozen_direct.json"
            ),
            "truth_topology_mode": "checkpoint_direct",
        },
        {
            "arm_role": "depth_balanced_masked_aux_encoder_adapt",
            "checkpoint_sha256": (
                "569121077774faa329dd6b620fb4d9cd6747ee3ac95ccccc7f9a408bc65b71c7"
            ),
            "checkpoint_step": 500,
            "path": (
                "artifacts/evaluation/phase35_failed_arm_step500_clean_v3_"
                "20260904/masked_adapt_direct.json"
            ),
            "truth_topology_mode": "checkpoint_direct",
        },
        {
            "arm_role": "depth_balanced_masked_aux_frozen",
            "checkpoint_sha256": (
                "02db81fd0bff1a7047d263021ec02522ffc6e533ef37599bec099df08d87d622"
            ),
            "checkpoint_step": 500,
            "path": (
                "artifacts/evaluation/phase35_failed_arm_step500_clean_v3_"
                "20260904/masked_frozen_contracted.json"
            ),
            "truth_topology_mode": "contracted_diagnostic",
        },
        {
            "arm_role": "depth_balanced_masked_aux_encoder_adapt",
            "checkpoint_sha256": (
                "569121077774faa329dd6b620fb4d9cd6747ee3ac95ccccc7f9a408bc65b71c7"
            ),
            "checkpoint_step": 500,
            "path": (
                "artifacts/evaluation/phase35_failed_arm_step500_clean_v3_"
                "20260904/masked_adapt_contracted.json"
            ),
            "truth_topology_mode": "contracted_diagnostic",
        },
    ]
    diagnostics = payload.get("step500_diagnostics")
    reports = diagnostics.get("reports") if isinstance(diagnostics, dict) else None
    if (
        not isinstance(reports, list)
        or diagnostics.get("decision") != "clean_restart_both_failed_arms"
        or len(reports) != len(expected_diagnostic_reports)
        or any(
            not isinstance(binding, dict)
            or set(binding) != {*expected, "sha256"}
            or {key: binding.get(key) for key in expected} != expected
            or not HEX64.fullmatch(str(binding.get("sha256", "")))
            for binding, expected in zip(reports, expected_diagnostic_reports)
        )
    ):
        raise RuntimeError("phase35 repair diagnostic decision changed")

    def metric_counts(
        report: dict[str, Any], scope: str, section: str, name: str
    ) -> tuple[float, float]:
        metric = report.get("summaries", {}).get(scope, {}).get(section, {}).get(name)
        if not isinstance(metric, dict):
            raise RuntimeError("phase35 repair diagnostic metric is missing")
        return float(metric.get("numerator", -1)), float(
            metric.get("denominator", -1)
        )

    parent = load_preregistration()
    exclusions = load_validation_exclusions(parent)
    cohort = load_evaluation_cohort(parent, exclusions=exclusions)
    expected_cohort = parent["evaluation_cohort"]
    for binding in reports:
        report_path = repo_path(binding["path"], suffix=".json")
        if sha256(report_path) != binding["sha256"]:
            raise RuntimeError("phase35 repair diagnostic report changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint_pair = report.get("checkpoint_pair")
        configuration = report.get("configuration")
        context = report.get("context")
        events = report.get("events")
        provenance = report.get("evaluator_code_provenance")
        if (
            report.get("report_version")
            != "hypertagging-offline-full-decay-evaluation-v3"
            or report.get("evaluation_role") != "offline_model_evaluation"
            or report.get("not_basf2_reconstruction") is not True
            or report.get("input_contract")
            != "direct-mdst-tree-v4-preprocessed-training-input"
            or not isinstance(checkpoint_pair, dict)
            or checkpoint_pair.get("compatible") is not True
            or checkpoint_pair.get("pretraining_step") != source["step"]
            or checkpoint_pair.get("pretraining_sha256") != source["sha256"]
            or checkpoint_pair.get("reconstruction_step")
            != binding["checkpoint_step"]
            or checkpoint_pair.get("reconstruction_sha256")
            != binding["checkpoint_sha256"]
            or checkpoint_pair.get("encoder_exact_fraction") != 1.0
            or checkpoint_pair.get("encoder_common_keys") != 121
            or checkpoint_pair.get("encoder_shape_compatible_keys") != 121
            or checkpoint_pair.get("encoder_dtype_compatible_keys") != 121
            or checkpoint_pair.get("encoder_exact_keys") != 121
            or checkpoint_pair.get("encoder_pretraining_keys") != 121
            or checkpoint_pair.get("encoder_reconstruction_keys") != 121
            or checkpoint_pair.get("exact_frozen_encoder_required")
            is not binding["arm_role"].endswith("_frozen")
            or not isinstance(configuration, dict)
            or configuration.get("truth_topology_mode")
            != binding["truth_topology_mode"]
            or configuration.get("max_level") != 6
            or configuration.get("target_policy") != "complete_only"
            or configuration.get("event_uid_manifest", {}).get("sha256")
            != expected_cohort["manifest_sha256"]
            or not isinstance(context, dict)
            or context.get("evaluation_split") != "validation"
            or context.get("evaluation_event_selection")
            != "explicit_uid_cohort"
            or context.get("evaluated_event_uids") != cohort["event_uids"]
            or context.get("evaluation_uid_train_overlap") != []
            or not isinstance(provenance, dict)
            or provenance.get("git_head")
            != "91f1a43e8827c67b4cd4e3d8d36279321f6b9dee"
            or provenance.get("provenance_version")
            != "git-index-cached-diff-v1"
            or provenance.get("provenance_complete") is not True
            or provenance.get("index_matches_worktree") is not True
            or provenance.get("worktree_dirty") is not False
            or provenance.get("dirty_path_count") != 0
            or provenance.get("dirty_paths") != []
            or provenance.get("untracked_path_count") != 0
            or provenance.get("untracked_paths") != []
            or not HEX64.fullmatch(
                str(provenance.get("worktree_patch_sha256", ""))
            )
            or not isinstance(events, list)
            or len(events) != 100
        ):
            raise RuntimeError("phase35 repair diagnostic identity changed")
        expected_common_counts = {
            ("full", "inference", "configured_root_completion"): (0.0, 100.0),
            ("full", "decay_metrics", "source_recall"): (32.0, 389.0),
            ("full", "decay_metrics", "source_precision"): (32.0, 42.0),
            ("full", "decay_metrics", "lcag_pair_accuracy"): (0.0, 3672.0),
            ("full", "decay_metrics", "mother_pid_coverage"): (0.0, 203.0),
            ("half", "decay_metrics", "source_recall"): (61.0, 657.0),
            ("half", "decay_metrics", "perfect_lcag"): (2.0, 148.0),
        }
        if any(
            metric_counts(report, scope, section, name) != counts
            for (scope, section, name), counts in expected_common_counts.items()
        ):
            raise RuntimeError("phase35 repair diagnostic endpoint changed")
        expected_representability = (
            ((21.0, 21.0), (148.0, 148.0))
            if binding["truth_topology_mode"] == "contracted_diagnostic"
            else ((1.0, 21.0), (97.0, 148.0))
        )
        if (
            metric_counts(report, "full", "decay_metrics", "target_representable")
            != expected_representability[0]
            or metric_counts(
                report, "half", "decay_metrics", "target_representable"
            )
            != expected_representability[1]
        ):
            raise RuntimeError("phase35 repair representability evidence changed")
    return payload


def resolved_arm_config(
    preregistration: dict[str, Any], arm_role: str
) -> dict[str, Any]:
    arms = {
        str(item["role"]): item for item in preregistration.get("arms", [])
    }
    if arm_role not in arms:
        raise RuntimeError(f"phase35 arm is not preregistered: {arm_role}")
    return {
        **dict(preregistration["common_training_contract"]),
        **dict(arms[arm_role]["overrides"]),
    }


def exclusion_identity(event_uids: list[str]) -> dict[str, Any]:
    normalized = tuple(sorted({str(uid) for uid in event_uids}))
    digest = hashlib.sha256()
    for uid in normalized:
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return {
        "event_uid_count": len(normalized),
        "event_uids_sha256": digest.hexdigest(),
        "event_uids_hash_scheme": "sha256-u64be-length-prefixed-utf8-v1",
    }


def uid_sequence_sha256(event_uids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"hypertagging-evaluation-event-uids-v1\0")
    for uid in event_uids:
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_validation_exclusions(
    preregistration: dict[str, Any],
) -> tuple[str, ...]:
    binding = preregistration.get("validation_exclusion")
    if not isinstance(binding, dict):
        raise RuntimeError("validation exclusion binding is missing")
    manifest_path = repo_path(str(binding.get("manifest", "")), suffix=".json")
    if sha256(manifest_path) != binding.get("manifest_sha256"):
        raise RuntimeError("validation exclusion manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("manifest_version") != "hypertagging-validation-exclusions-v1"
        or manifest.get("role") != "validation"
        or manifest.get("sealed_test_role_access") != "forbidden"
    ):
        raise RuntimeError("validation exclusion manifest policy changed")
    event_uids = manifest.get("event_uids")
    if not isinstance(event_uids, list) or not event_uids:
        raise RuntimeError("validation exclusion manifest contains no UIDs")
    if len(event_uids) != len(set(str(uid) for uid in event_uids)):
        raise RuntimeError("validation exclusion manifest contains duplicate UIDs")
    identity = exclusion_identity([str(uid) for uid in event_uids])
    expected = {
        "event_uid_count": binding.get("event_uid_count"),
        "event_uids_sha256": binding.get("event_uids_sha256"),
        "event_uids_hash_scheme": binding.get("event_uids_hash_scheme"),
    }
    if identity != expected:
        raise RuntimeError("validation exclusion identity changed")
    manifest_identity = {
        "event_uid_count": manifest.get("excluded_event_uid_count"),
        "event_uids_sha256": manifest.get("excluded_event_uids_sha256"),
        "event_uids_hash_scheme": manifest.get(
            "excluded_event_uids_hash_scheme"
        ),
    }
    if identity != manifest_identity:
        raise RuntimeError("validation exclusion manifest identity is invalid")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise RuntimeError("validation exclusion source bindings changed")
    for source in sources:
        if sha256(repo_path(str(source["path"]))) != source.get("sha256"):
            raise RuntimeError(
                f"validation exclusion source changed: {source['path']}"
            )
    return tuple(str(uid) for uid in event_uids)


def load_evaluation_cohort(
    preregistration: dict[str, Any], *, exclusions: tuple[str, ...]
) -> dict[str, Any]:
    binding = preregistration.get("evaluation_cohort")
    if not isinstance(binding, dict):
        raise RuntimeError("phase35 evaluation cohort binding is missing")
    manifest_path = repo_path(str(binding.get("manifest", "")), suffix=".json")
    if sha256(manifest_path) != binding.get("manifest_sha256"):
        raise RuntimeError("phase35 evaluation cohort manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_uids = [str(uid) for uid in manifest.get("event_uids", [])]
    selection_uids = [
        str(uid)
        for uid in manifest.get("checkpoint_selection_event_uids", [])
    ]
    if (
        manifest.get("manifest_version")
        != "hypertagging-reconstruction-evaluation-cohort-v1"
        or manifest.get("study_id") != PARENT_STUDY_ID
        or manifest.get("role") != "validation"
        or manifest.get("sealed_test_role_access") != "forbidden"
        or len(event_uids) != 100
        or len(set(event_uids)) != 100
        or len(selection_uids) != 2000
        or len(set(selection_uids)) != 2000
        or uid_sequence_sha256(event_uids)
        != manifest.get("event_uids_sha256")
        or uid_sequence_sha256(selection_uids)
        != manifest.get("checkpoint_selection_event_uids_sha256")
        or set(event_uids) & set(selection_uids)
        or (set(event_uids) | set(selection_uids)) & set(exclusions)
        or manifest.get("overlap_audit")
        != {
            "historical_exclusions_vs_checkpoint_selection": 0,
            "historical_exclusions_vs_evaluation": 0,
            "checkpoint_selection_vs_evaluation": 0,
        }
    ):
        raise RuntimeError("phase35 evaluation cohort contract is invalid")
    expected_binding = {
        "manifest": binding.get("manifest"),
        "manifest_sha256": binding.get("manifest_sha256"),
        "role": manifest.get("role"),
        "event_uid_count": manifest.get("event_uid_count"),
        "event_uids_sha256": manifest.get("event_uids_sha256"),
        "checkpoint_selection_event_uid_count": manifest.get(
            "checkpoint_selection_event_uid_count"
        ),
        "checkpoint_selection_event_uids_sha256": manifest.get(
            "checkpoint_selection_event_uids_sha256"
        ),
        "source_category_counts": manifest.get("source_category_counts"),
        "historical_exclusion_overlap": manifest["overlap_audit"][
            "historical_exclusions_vs_evaluation"
        ],
        "checkpoint_selection_overlap": manifest["overlap_audit"][
            "checkpoint_selection_vs_evaluation"
        ],
        "sealed_test_role_access": manifest.get("sealed_test_role_access"),
    }
    if binding != expected_binding:
        raise RuntimeError("phase35 evaluation cohort preregistration changed")
    return manifest


def verify_test_receipt(
    path: Path, *, expected_git_sha: str, expected_git_tag: str
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    stored_receipt_hash = str(receipt.get("receipt_sha256", ""))
    canonical_receipt = dict(receipt)
    canonical_receipt.pop("receipt_sha256", None)
    actual_receipt_hash = hashlib.sha256(
        json.dumps(
            canonical_receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        receipt.get("receipt_version")
        != "hypertagging-reconstruction-phase35-test-receipt-v1"
        or stored_receipt_hash != actual_receipt_hash
        or receipt.get("status") != "passed"
        or receipt.get("git_sha") != expected_git_sha
        or receipt.get("git_tag") != expected_git_tag
        or receipt.get("sealed_test_accessed") is not False
    ):
        raise RuntimeError("test receipt does not bind a passing sealed-test-free HEAD")
    tests = receipt.get("tests")
    if not isinstance(tests, list) or not tests or not all(
        item.get("passed") is True and item.get("exit_code") == 0
        for item in tests
    ):
        raise RuntimeError("test receipt does not contain an all-passing test set")
    by_name = {str(item.get("name")): item for item in tests}
    focused = by_name.get("phase35_focused_and_adjacent_cpu")
    static = by_name.get("phase35_static_validation")
    required_test_files = {
        "tests/test_reconstruction_phase35_improvement_cpu.py",
        "tests/test_reconstruction_phase35_full_decay_runner_cpu.py",
        "tests/test_balanced_level_replay_cpu.py",
        "tests/test_reconstruction_rollout_alignment_cpu.py",
        "tests/test_campaign_readiness_cpu.py",
        "tests/test_validation_batch_size_cpu.py",
        "tests/test_trainer_corrections_cpu.py",
        "tests/test_real_training_pipeline_cpu.py",
        "tests/test_full_decay_cli_cpu.py",
        "tests/test_trained_evaluation_context_cpu.py",
        "tests/test_two_pass_leaf_pid_reconstruction_cpu.py",
    }
    required_static_checks = {
        "bash syntax",
        "JSON syntax",
        "Python compilation",
        "ruff",
        "git diff whitespace",
        "tracked worktree cleanliness",
    }
    if (
        not isinstance(focused, dict)
        or set(focused.get("files", ())) != required_test_files
        or int(focused.get("passed_count", 0)) < 119
        or "uv run --no-project" not in str(focused.get("command", ""))
        or "hypertagging-gpu-cu126-v1/bin/python" not in str(
            focused.get("command", "")
        )
    ):
        raise RuntimeError("phase35 focused test evidence is incomplete")
    if (
        not isinstance(static, dict)
        or not required_static_checks.issubset(set(static.get("checks", ())))
    ):
        raise RuntimeError("phase35 static validation evidence is incomplete")
    permissions = receipt.get("permissions", {})
    required_true = (
        "training_authorized",
        "validation_payload_access_authorized",
        "execution_authorized",
        "scheduler_authorized",
        "submission_authorized",
        "scientific_validation_authorized",
    )
    required_false = (
        "source_checkpoint_mutation_authorized",
        "sealed_test_access_authorized",
        "promotion_authorized",
    )
    if any(permissions.get(key) is not True for key in required_true):
        raise RuntimeError("test receipt is missing required phase35 permission")
    if any(permissions.get(key) is not False for key in required_false):
        raise RuntimeError("test receipt enables forbidden phase35 authority")
    return receipt


def verify_attempt_receipt(
    path: Path, *, expected_contract_sha256: str
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = str(receipt.get("receipt_sha256", ""))
    canonical = dict(receipt)
    canonical.pop("receipt_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        receipt.get("receipt_version")
        != "hypertagging-reconstruction-phase35-attempt-v1"
        or receipt.get("status") != "completed"
        or receipt.get("contract_sha256") != expected_contract_sha256
        or not HEX64.fullmatch(stored_hash)
        or stored_hash != actual_hash
    ):
        raise RuntimeError("phase35 attempt receipt is not canonically healthy")
    return receipt


def verify_campaign_submission_receipt(
    *,
    job_id: str,
    contract: dict[str, Any],
    allowed_statuses: tuple[str, ...] = ("release_in_progress", "submitted"),
) -> dict[str, Any]:
    """Verify the complete, held-first two-arm repair authorization.

    ``release_in_progress`` is written and fsynced before the single multi-job
    release command.  It is therefore an authorization to start the exact
    recorded jobs, without falsely claiming that scheduler release evidence was
    already collected.  Earlier preparation states are never executable.
    """

    receipt_path = safe_repo_output_path(
        str(contract.get("campaign_submission_receipt", "")),
        expected_relative=CAMPAIGN_SUBMISSION_RECEIPT,
        require_file=True,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("phase35 campaign submission receipt is malformed")
    stored_hash = str(receipt.get("receipt_sha256", ""))
    canonical = dict(receipt)
    canonical.pop("receipt_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if not allowed_statuses or any(
        status not in {"release_in_progress", "submitted"}
        for status in allowed_statuses
    ):
        raise RuntimeError("invalid phase35 executable receipt status policy")
    expected_top_level_keys = {
        "receipt_version",
        "created_at",
        "updated_at",
        "status",
        "study_id",
        "git_sha",
        "git_tag",
        "test_receipt",
        "permissions",
        "sealed_test_accessed",
        "automatic_promotion",
        "atomic_campaign_release",
        "jobs_initially_submitted_held",
        "planned_jobs",
        "active_submission_arm",
        "jobs",
        "submitted_job_ids",
        "release_argv",
        "scheduler_records_after_release",
        "cancellation_states",
        "error",
        "receipt_sha256",
    }
    if (
        receipt.get("receipt_version")
        != CAMPAIGN_SUBMISSION_RECEIPT_VERSION
        or receipt.get("status") not in set(allowed_statuses)
        or set(receipt) != expected_top_level_keys
        or not HEX64.fullmatch(stored_hash)
        or stored_hash != actual_hash
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("git_sha") != contract.get("expected_git_sha")
        or receipt.get("git_tag") != contract.get("expected_git_tag")
        or receipt.get("test_receipt") != contract.get("test_receipt")
        or receipt.get("permissions") != PHASE35_CAMPAIGN_PERMISSIONS
        or receipt.get("sealed_test_accessed") is not False
        or receipt.get("automatic_promotion") is not False
        or receipt.get("atomic_campaign_release") is not True
        or receipt.get("jobs_initially_submitted_held") is not True
        or receipt.get("active_submission_arm") is not None
        or receipt.get("cancellation_states") != {}
        or receipt.get("error") is not None
    ):
        raise RuntimeError("phase35 job lacks its atomic campaign submission receipt")

    campaign_contracts: list[tuple[Path, dict[str, Any]]] = []
    expected_planned_jobs: list[dict[str, str]] = []
    for arm_role in PHASE35_ARM_ROLES:
        relative = phase35_contract_relative_path(arm_role)
        peer_path = safe_repo_output_path(
            relative,
            expected_relative=relative,
            require_file=True,
        )
        peer = json.loads(peer_path.read_text(encoding="utf-8"))
        if not isinstance(peer, dict):
            raise RuntimeError(
                f"phase35 campaign peer contract is malformed: {arm_role}"
            )
        peer_hash = str(peer.get("contract_sha256", ""))
        if (
            not HEX64.fullmatch(peer_hash)
            or canonical_contract_hash(peer) != peer_hash
            or peer.get("contract_version") != CONTRACT_VERSION
            or peer.get("study_id") != STUDY_ID
            or peer.get("arm_role") != arm_role
            or peer.get("task_id") != phase35_task_id(arm_role)
            or peer.get("expected_git_sha") != contract.get("expected_git_sha")
            or peer.get("expected_git_tag") != contract.get("expected_git_tag")
            or peer.get("test_receipt") != contract.get("test_receipt")
            or peer.get("campaign_submission_receipt")
            != CAMPAIGN_SUBMISSION_RECEIPT
        ):
            raise RuntimeError(
                f"phase35 campaign peer contract is invalid: {arm_role}"
            )
        campaign_contracts.append((peer_path, peer))
        expected_planned_jobs.append(
            {
                "arm_role": arm_role,
                "task_id": str(peer["task_id"]),
                "contract": relative,
                "contract_sha256": peer_hash,
                "scheduler_comment": f"phase35:{peer_hash}",
            }
        )
    current_peers = [
        peer
        for _, peer in campaign_contracts
        if peer["arm_role"] == contract.get("arm_role")
    ]
    if (
        len(current_peers) != 1
        or current_peers[0]["contract_sha256"]
        != contract.get("contract_sha256")
        or receipt.get("planned_jobs") != expected_planned_jobs
    ):
        raise RuntimeError("phase35 campaign planned-job contract changed")

    jobs = receipt.get("jobs")
    submitted_ids = receipt.get("submitted_job_ids")
    if (
        not isinstance(jobs, list)
        or len(jobs) != len(PHASE35_ARM_ROLES)
        or not all(isinstance(item, dict) for item in jobs)
        or not isinstance(submitted_ids, list)
        or len(submitted_ids) != len(PHASE35_ARM_ROLES)
    ):
        raise RuntimeError("phase35 repair campaign does not contain exactly two jobs")
    job_ids = [str(item.get("job_id", "")) for item in jobs]
    if (
        any(not value.isdigit() for value in job_ids)
        or len(set(job_ids)) != len(PHASE35_ARM_ROLES)
        or submitted_ids != job_ids
        or job_id not in job_ids
    ):
        raise RuntimeError("phase35 campaign job ID set changed")
    expected_release_argv = [
        "/opt/slurm/bin/scontrol",
        "release",
        ",".join(job_ids),
    ]
    if receipt.get("release_argv") != expected_release_argv:
        raise RuntimeError("phase35 campaign release command changed")

    base_job_keys = {
        "arm_role",
        "task_id",
        "job_id",
        "contract",
        "contract_sha256",
        "submission_argv",
        "scheduler_submit_line",
        "scheduler_record_while_held",
        "scheduler_state_while_held",
        "requeue",
        "restarts",
    }
    released_job_keys = {
        "scheduler_record_after_release",
        "scheduler_state_after_release",
    }
    for index, (item, planned, peer_entry) in enumerate(
        zip(jobs, expected_planned_jobs, campaign_contracts)
    ):
        if not isinstance(item, dict):
            raise RuntimeError("phase35 campaign job evidence is malformed")
        peer_path, peer = peer_entry
        expected_keys = (
            base_job_keys | released_job_keys
            if receipt["status"] == "submitted"
            else base_job_keys
        )
        held_record = str(item.get("scheduler_record_while_held", ""))
        command = phase35_submission_command(peer, peer_path)
        try:
            recorded_submit_argv = shlex.split(
                str(item.get("scheduler_submit_line", ""))
            )
        except ValueError as error:
            raise RuntimeError(
                "phase35 scheduler submit line is malformed"
            ) from error
        if (
            set(item) != expected_keys
            or item.get("arm_role") != planned["arm_role"]
            or item.get("task_id") != planned["task_id"]
            or item.get("contract") != planned["contract"]
            or item.get("contract_sha256") != planned["contract_sha256"]
            or str(item.get("job_id")) != job_ids[index]
            or item.get("submission_argv") != command
            or recorded_submit_argv != command
            or item.get("scheduler_state_while_held") != "PENDING"
            or _slurm_field(held_record, "JobId") != job_ids[index]
            or _slurm_field(held_record, "JobState").split("+", 1)[0]
            != "PENDING"
            or _slurm_field(held_record, "Reason") != "JobHeldUser"
            or type(item.get("requeue")) is not int
            or item.get("requeue") != 0
            or type(item.get("restarts")) is not int
            or item.get("restarts") != 0
        ):
            raise RuntimeError("phase35 held-job evidence changed")
        verify_live_slurm_record(
            held_record,
            contract=peer,
            contract_path=peer_path,
        )

    release_records = receipt.get("scheduler_records_after_release")
    if receipt["status"] == "release_in_progress":
        if release_records != {}:
            raise RuntimeError(
                "release-in-progress receipt contains premature release evidence"
            )
        # The durable intent receipt closes the release/write crash window, but
        # it is executable only after every member is demonstrably no longer on
        # its user hold (or has already started and reached accounting history).
        for (peer_path, peer), recorded_job_id in zip(
            campaign_contracts, job_ids
        ):
            verify_job_released_or_executed(
                recorded_job_id,
                contract=peer,
                contract_path=peer_path,
            )
    else:
        if not isinstance(release_records, dict) or set(release_records) != set(
            job_ids
        ):
            raise RuntimeError("phase35 post-release scheduler evidence is incomplete")
        allowed_released_states = {"PENDING", "CONFIGURING", "RUNNING"}
        for item, (peer_path, peer), recorded_job_id in zip(
            jobs, campaign_contracts, job_ids
        ):
            record = str(release_records[recorded_job_id])
            state = _slurm_field(record, "JobState").split("+", 1)[0]
            if (
                _slurm_field(record, "JobId") != recorded_job_id
                or state not in allowed_released_states
                or _slurm_field(record, "Reason").startswith("JobHeld")
                or item.get("scheduler_record_after_release") != record
                or item.get("scheduler_state_after_release") != state
            ):
                raise RuntimeError("phase35 post-release job evidence changed")
            verify_live_slurm_record(
                record,
                contract=peer,
                contract_path=peer_path,
            )

    matching = [
        item
        for item in jobs
        if str(item["job_id"]) == job_id
        and item["arm_role"] == contract.get("arm_role")
        and item["contract_sha256"] == contract.get("contract_sha256")
    ]
    if len(matching) != 1:
        raise RuntimeError("phase35 job is absent from its campaign receipt")
    return receipt


def slurm_job(job_id: str) -> str:
    result = subprocess.run(
        ("/opt/slurm/bin/scontrol", "show", "job", "-o", job_id),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    record = result.stdout.strip()
    if not record:
        raise RuntimeError(f"Slurm has no live record for job {job_id}")
    return record


def _slurm_accounting_execution(job_id: str) -> tuple[str, str]:
    result = subprocess.run(
        (
            "/opt/slurm/bin/sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,State,Start",
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    for row in result.stdout.splitlines():
        values = row.split("|")
        if len(values) >= 3 and values[0] == job_id:
            return values[1].split()[0].split("+", 1)[0], values[2].strip()
    raise RuntimeError(f"Slurm accounting has no execution record for job {job_id}")


def verify_job_released_or_executed(
    job_id: str, *, contract: dict[str, Any], contract_path: Path
) -> None:
    try:
        record = slurm_job(job_id)
    except RuntimeError:
        state, started_at = _slurm_accounting_execution(job_id)
        terminal_states = {
            "CANCELLED",
            "COMPLETED",
            "FAILED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
        if state not in terminal_states or started_at in {"", "Unknown", "N/A"}:
            raise RuntimeError(
                f"phase35 campaign job {job_id} has no proven release execution"
            )
        return
    verify_live_slurm_record(
        record,
        contract=contract,
        contract_path=contract_path,
    )
    state = _slurm_field(record, "JobState").split("+", 1)[0]
    if state not in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"} or (
        _slurm_field(record, "Reason").startswith("JobHeld")
    ):
        raise RuntimeError(
            f"phase35 campaign job {job_id} remains held or has invalid state"
        )


def _slurm_field(record: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", record)
    return "" if match is None else match.group(1)


def verify_live_slurm_record(
    record: str,
    *,
    contract: dict[str, Any],
    contract_path: Path,
) -> None:
    expected = {
        "Requeue": "0",
        "Restarts": "0",
        "JobName": str(contract["task_id"]),
        "Partition": "inter",
        "Account": "others",
        "NumCPUs": "8",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "TimeLimit": "1-00:00:00",
        "MinMemoryNode": "64G",
        "Comment": f"phase35:{contract['contract_sha256']}",
    }
    mismatches = {
        key: (_slurm_field(record, key), value)
        for key, value in expected.items()
        if _slurm_field(record, key) != value
    }
    # A pending one-node job is rendered as the exact range ``1-1`` by this
    # cluster's Slurm controller; running and terminal records use ``1``.
    # ReqTRES:node is independently required below, so accept only these two
    # equivalent single-node encodings.
    num_nodes = _slurm_field(record, "NumNodes")
    if num_nodes not in {"1", "1-1"}:
        mismatches["NumNodes"] = (num_nodes, "1 or 1-1")
    req_tres = _slurm_field(record, "ReqTRES")
    req_tres_values = {
        key: value
        for token in req_tres.split(",")
        for key, separator, value in (token.partition("="),)
        if separator
    }
    required_req_tres = {
        "cpu": "8",
        "mem": "64G",
        "node": "1",
        "gres/gpu": "1",
        "gres/gpu:h100nvl": "1",
    }
    for key, value in required_req_tres.items():
        if req_tres_values.get(key) != value:
            mismatches[f"ReqTRES:{key}"] = (
                req_tres_values.get(key, ""),
                value,
            )
    requested_gpu_tres = {
        key: value
        for key, value in req_tres_values.items()
        if key.startswith("gres/gpu")
    }
    expected_gpu_tres = {
        "gres/gpu": "1",
        "gres/gpu:h100nvl": "1",
    }
    if requested_gpu_tres != expected_gpu_tres:
        mismatches["ReqTRES:gpu_exact"] = (
            str(requested_gpu_tres),
            str(expected_gpu_tres),
        )
    tres_per_node = _slurm_field(record, "TresPerNode")
    if tres_per_node != "gres:gpu:h100nvl:1":
        mismatches["TresPerNode"] = (
            tres_per_node,
            "gres:gpu:h100nvl:1",
        )
    command = _slurm_field(record, "Command")
    expected_command = str((ROOT / WRAPPER).resolve())
    if command != expected_command:
        mismatches["Command"] = (command, expected_command)
    if _slurm_field(record, "WorkDir") != str(ROOT.resolve()):
        mismatches["WorkDir"] = (
            _slurm_field(record, "WorkDir"),
            str(ROOT.resolve()),
        )
    if mismatches:
        raise RuntimeError(f"live Slurm allocation violates phase35 contract: {mismatches}")


def verify_contract(
    path: Path, *, require_slurm: bool = True
) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.resolve(strict=True)
    path.relative_to(ROOT.resolve())
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract_hash = verify_contract_hash(contract)
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("unsupported phase35 reconstruction contract")
    if contract.get("study_id") != STUDY_ID:
        raise RuntimeError("phase35 study ID changed")
    expected_sha = str(contract.get("expected_git_sha", ""))
    expected_tag = str(contract.get("expected_git_tag", ""))
    if (
        not HEX40.fullmatch(expected_sha)
        or expected_tag != IMPLEMENTATION_TAG
        or git("rev-parse", "HEAD") != expected_sha
        or git("rev-list", "-n", "1", IMPLEMENTATION_TAG) != expected_sha
    ):
        raise RuntimeError("phase35 tested implementation identity mismatch")
    verify_clean_worktree()
    preregistration = load_preregistration()
    repair_preregistration = load_repair_preregistration()
    prereg_binding = contract.get("preregistration", {})
    if (
        prereg_binding.get("path") != PREREGISTRATION
        or prereg_binding.get("sha256") != sha256(repo_path(PREREGISTRATION))
    ):
        raise RuntimeError("phase35 preregistration binding changed")
    repair_prereg_binding = contract.get("repair_preregistration", {})
    if (
        repair_prereg_binding.get("path") != REPAIR_PREREGISTRATION
        or repair_prereg_binding.get("sha256")
        != sha256(repo_path(REPAIR_PREREGISTRATION))
    ):
        raise RuntimeError("phase35 repair preregistration binding changed")
    arm_role = str(contract.get("arm_role", ""))
    expected_task_id = phase35_task_id(arm_role)
    if (
        arm_role not in PHASE35_ARM_ROLES
        or contract.get("mode") != "production"
        or contract.get("task_id") != expected_task_id
        or contract.get("experiment") != expected_task_id
        or contract.get("submission_performed") is not False
    ):
        raise RuntimeError("phase35 contract execution identity changed")
    if contract.get("config") != resolved_arm_config(preregistration, arm_role):
        raise RuntimeError("phase35 arm configuration differs from preregistration")
    config = contract["config"]
    expected_replay_counts = {
        "1": 23339,
        "2": 23339,
        "3": 23339,
        "4": 23339,
        "5": 23338,
        "6": 23338,
    }
    replay_contract = config.get("balanced_level_replay_contract", {})
    if (
        config.get("max_steps") != 2188
        or config.get("training_budget_semantics")
        != (
            "four_train_size_equivalent_level_conditioned_replay_slots_not_"
            "four_literal_dataset_epochs"
        )
        or config.get("train_split_event_count") != 35000
        or config.get("four_train_size_equivalent_slot_target") != 140000
        or config.get("optimizer_steps_per_train_size_equivalent") != 547
        or config.get("replay_slot_budget") != 140032
        or config.get("replay_slot_counts_by_level") != expected_replay_counts
        or config.get("batch_size") != 64
        or config.get("lr_schedule_total_steps") != 2188
        or config.get("level_sampling_mode") != "balanced_level_replay"
        or replay_contract.get("version") != "balanced-level-replay-v1"
        or replay_contract.get("levels") != [1, 2, 3, 4, 5, 6]
        or replay_contract.get("seed") != 20260904
        or replay_contract.get("target_policy") != "complete_only"
        or replay_contract.get("min_daughters") != 2
        or replay_contract.get("materialized_train_event_count") != 35000
        or replay_contract.get("eligible_pool_counts_by_level")
        != {
            "1": 27620,
            "2": 16160,
            "3": 12961,
            "4": 10001,
            "5": 6542,
            "6": 885,
        }
        or replay_contract.get("planned_schedule", {}).get("slot_count")
        != 140032
        or replay_contract.get("planned_schedule", {}).get("level_counts")
        != expected_replay_counts
        or replay_contract.get("planned_schedule", {}).get(
            "max_minus_min_level_count"
        )
        != 1
        or config.get("scheduled_sampling_probability") != 0.5
        or config.get("scheduled_sampling_duration_steps") != 1094
        or config.get("scheduled_sampling_zero_based_step_range") != [0, 2187]
        or config.get(
            "nominal_expected_predicted_context_slots_before_fallback"
        )
        != 52496
        or config.get(
            "nominal_expected_predicted_context_fraction_before_fallback"
        )
        != 0.3748857404021938
        or config.get("rollout_continue_through_empty_levels") is not True
        or config.get("rollout_max_level") != 6
        or config.get("rollout_root_types") != [1]
        or config.get("rollout_exclusive_final") is not True
        or config.get("rollout_use_learned_confidence") is not True
        or config.get("target_policy") != "complete_only"
        or config.get("validation_enabled") is not True
        or config.get("scientific_mode") is not True
    ):
        raise RuntimeError("phase35 anti-collapse training controls changed")
    if (
        config.get("transfer_leaf_pid_head") is not True
        or config.get("minimum_encoder_transfer_coverage") != 1.0
        or config.get("allow_low_encoder_transfer_coverage") is not False
        or config.get("require_exact_leaf_pid_transfer") is not True
    ):
        raise RuntimeError("phase35 requires exact encoder and leaf PID transfer")
    if config.get("freeze_leaf_pid_head_steps") != config["max_steps"]:
        raise RuntimeError("phase35 leaf PID head must remain frozen")
    if arm_role.endswith("encoder_adapt"):
        if (
            config.get("freeze_pretrained_encoder_steps") != 1094
            or config.get("encoder_lr_multiplier") != 0.05
        ):
            raise RuntimeError("phase35 encoder adaptation policy changed")
    elif config.get("freeze_pretrained_encoder_steps") != config["max_steps"]:
        raise RuntimeError("phase35 frozen arm unexpectedly adapts the encoder")
    if arm_role == "depth_balanced_fallback_frozen":
        if (
            config.get("unrepresentable_target_policy") != "fallback_teacher"
            or config.get("auxiliary_teacher_weight") != 0.0
        ):
            raise RuntimeError("phase35 fallback control policy changed")
    elif (
        config.get("unrepresentable_target_policy")
        != "masked_representable_only"
        or config.get("auxiliary_teacher_weight") != 0.25
    ):
        raise RuntimeError("phase35 masked auxiliary policy changed")

    evaluation_policy = preregistration.get("selection_and_evaluation", {})
    if (
        evaluation_policy.get("checkpoint_selection")
        != {
            "checkpoint": "best.pt",
            "primary_metric": "predicted_edge_f1",
            "mode": "max",
            "eligible_rollout_steps": [500, 1000, 1500, 2000, 2188],
            "tie_break": "retain_earliest_step",
            "teacher_forced_event_uid_count": 2000,
            "rollout_event_uid_count": 1000,
            "rollout_event_selection": (
                "ordered_prefix_of_checkpoint_selection_cohort"
            ),
            "evaluation_cohort_used_for_checkpoint_selection": False,
        }
        or evaluation_policy.get("post_training_full_decay_protocol", {}).get(
            "max_level"
        )
        != 6
        or evaluation_policy.get("sealed_test_evaluation_before_selection")
        is not False
        or evaluation_policy.get("automatic_promotion") is not False
        or evaluation_policy.get("evaluation_cohort_arm_selection_authorized")
        is not False
    ):
        raise RuntimeError("phase35 selection/evaluation separation changed")

    source = preregistration["source_checkpoint"]
    if (
        contract.get("checkpoint") != source["path"]
        or contract.get("checkpoint_step") != source["step"]
        or contract.get("checkpoint_sha256") != source["sha256"]
        or repair_preregistration.get("restart_source_checkpoint")
        != {
            "path": source["path"],
            "step": source["step"],
            "sha256": source["sha256"],
        }
    ):
        raise RuntimeError("phase35 source checkpoint binding changed")
    checkpoint = repo_path(str(source["path"]), suffix=".pt")
    if sha256(checkpoint) != source["sha256"]:
        raise RuntimeError("phase35 source checkpoint binding changed")
    data = preregistration["data_binding"]
    if contract.get("data") != data:
        raise RuntimeError("phase35 data binding changed")
    selection = repo_path(str(data["selection_manifest"]), suffix=".json")
    index = repo_path(str(data["dataset_index"]), suffix=".json")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    if (
        sha256(selection) != data["selection_manifest_sha256"]
        or sha256(index) != data["dataset_index_sha256"]
        or data.get("split_counts", {}).get("test") != 0
        or data.get("opened_index_roles") != ["train", "validation"]
        or data.get("raw_manifest_contains_unopened_test_entries") is not True
        or data.get("dataset_index_sealed_test_opened") is not False
        or data.get("sealed_test_role_access") != "forbidden"
        or index_payload.get("selection_contract", {}).get(
            "included_splits"
        )
        != ["train", "validation"]
        or index_payload.get("event_identity_validation", {}).get("status")
        != "passed"
        or index_payload.get("event_identity_validation", {}).get(
            "sealed_test_opened"
        )
        is not False
        or index_payload.get("split_counts")
        != {"train": 35000, "validation": 50000}
    ):
        raise RuntimeError("phase35 train/validation data contract changed")
    exclusions = load_validation_exclusions(preregistration)
    if len(exclusions) != 3956:
        raise RuntimeError("phase35 validation exclusion cohort changed")
    if contract.get("validation_exclusion") != preregistration[
        "validation_exclusion"
    ]:
        raise RuntimeError("phase35 validation exclusion contract changed")
    evaluation_cohort = load_evaluation_cohort(
        preregistration, exclusions=exclusions
    )
    if contract.get("evaluation_cohort") != preregistration["evaluation_cohort"]:
        raise RuntimeError("phase35 evaluation cohort contract changed")
    if contract.get("hierarchy_supervision") != PHASE35_HIERARCHY_SUPERVISION:
        raise RuntimeError("phase35 all-level truth supervision contract changed")

    if contract.get("resources") != repair_preregistration["execution_policy"]:
        raise RuntimeError("phase35 resource policy changed")
    resources = contract["resources"]
    if (
        resources.get("gres") != "gpu:h100nvl:1"
        or resources.get("cpus_per_task") != 8
        or resources.get("memory") != "64G"
        or resources.get("time") != "24:00:00"
        or resources.get("requeue") is not False
        or resources.get("maximum_restarts") != 0
        or resources.get("task_count") != 2
        or resources.get("global_concurrency") != 2
    ):
        raise RuntimeError("phase35 resource policy is unsafe")
    if contract.get("device") != "cuda":
        raise RuntimeError("phase35 training device changed")
    if contract.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("sealed-test access is not forbidden")
    if contract.get("source_checkpoint_mutation") != "forbidden":
        raise RuntimeError("source checkpoint mutation is not forbidden")
    if contract.get("automatic_promotion") is not False:
        raise RuntimeError("automatic promotion is forbidden")

    authorization = contract.get("authorization", {})
    expected_authorization = {
        "basis": "explicit_user_operator_instruction_2026-09-04",
        **PHASE35_CAMPAIGN_PERMISSIONS,
    }
    if authorization != expected_authorization:
        raise RuntimeError("phase35 authorization matrix changed")
    if contract.get("submission_authorized") is not True:
        raise RuntimeError("phase35 submission is not authorized")
    if contract.get("campaign_submission_receipt") != CAMPAIGN_SUBMISSION_RECEIPT:
        raise RuntimeError("phase35 atomic campaign receipt path changed")
    expected_repair_lineage = {
        "repair_preregistration": {
            "path": REPAIR_PREREGISTRATION,
            "sha256": sha256(repo_path(REPAIR_PREREGISTRATION)),
        },
        "parent_study_id": PARENT_STUDY_ID,
        "parent_git_sha": repair_preregistration["parent_git_sha"],
        "parent_git_tag": repair_preregistration["parent_git_tag"],
        "clean_restart_required": True,
        "restart_source_checkpoint": repair_preregistration[
            "restart_source_checkpoint"
        ],
        "resume_from_failed_attempt_authorized": False,
        "reuse_failed_output_authorized": False,
        "excluded_completed_arm_role": "depth_balanced_fallback_frozen",
        "excluded_completed_arm_rerun_authorized": False,
    }
    if contract.get("repair_lineage") != expected_repair_lineage:
        raise RuntimeError("phase35 repair lineage changed")

    receipt_binding = contract.get("test_receipt", {})
    if receipt_binding.get("path") != PHASE35_TEST_RECEIPT:
        raise RuntimeError("phase35 test receipt path changed")
    receipt_path = safe_repo_output_path(
        PHASE35_TEST_RECEIPT,
        expected_relative=PHASE35_TEST_RECEIPT,
        require_file=True,
    )
    if sha256(receipt_path) != receipt_binding.get("sha256"):
        raise RuntimeError("phase35 test receipt hash changed")
    verify_test_receipt(
        receipt_path,
        expected_git_sha=expected_sha,
        expected_git_tag=expected_tag,
    )
    runtime_binding = preregistration.get("runtime_binding", {})
    gpu_environment = Path(str(contract.get("gpu_environment", "")))
    if (
        str(gpu_environment) != runtime_binding.get("gpu_environment")
        or not gpu_environment.is_absolute()
        or not (gpu_environment / "bin/python").is_file()
        or sha256(gpu_environment / "bin/python")
        != runtime_binding.get("python_sha256")
        or sha256(gpu_environment / "pyvenv.cfg")
        != runtime_binding.get("pyvenv_cfg_sha256")
        or sha256(repo_path(runtime_binding.get("runtime_contract", "")))
        != runtime_binding.get("runtime_contract_sha256")
        or sha256(repo_path(runtime_binding.get("requirements_lock", "")))
        != runtime_binding.get("requirements_lock_sha256")
    ):
        raise RuntimeError("phase35 exact GPU environment binding changed")
    exclusion_manifest = json.loads(
        repo_path(
            str(preregistration["validation_exclusion"]["manifest"])
        ).read_text(encoding="utf-8")
    )
    expected_hashed_paths = {
        *PHASE35_SOURCE_FILES,
        str(source["path"]),
        str(preregistration["baseline_reconstruction_checkpoint"]["path"]),
        str(data["selection_manifest"]),
        str(data["dataset_index"]),
        str(preregistration["validation_exclusion"]["manifest"]),
        str(preregistration["evaluation_cohort"]["manifest"]),
        str(preregistration["diagnostic_basis"]["full_decay_report"]["path"]),
        str(preregistration["diagnostic_basis"]["lineage_receipt"]["path"]),
        PHASE35_TEST_RECEIPT,
        str(repair_preregistration["parent_campaign_submission_receipt"]["path"]),
        *(
            str(binding["path"])
            for binding in repair_preregistration["step500_diagnostics"][
                "reports"
            ]
        ),
        *(
            str(binding[key]["path"])
            for binding in repair_preregistration["historical_attempts"]
            for key in ("contract", "attempt_receipt", "stderr")
        ),
        *(str(item["path"]) for item in exclusion_manifest["sources"]),
    }
    verify_hashed_inputs(
        list(contract.get("hashed_inputs", [])),
        expected_paths=expected_hashed_paths,
    )
    expected_output_relative = f"{OUTPUT_NAMESPACE}/{arm_role}"
    if contract.get("output_root") != expected_output_relative:
        raise RuntimeError("phase35 output root is outside its arm namespace")
    output_root = safe_repo_output_path(
        expected_output_relative,
        expected_relative=expected_output_relative,
    )

    if require_slurm:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            raise RuntimeError("phase35 execution requires Slurm")
        if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
            raise RuntimeError("phase35 task cannot execute after a restart")
        verify_live_slurm_record(
            slurm_job(job_id), contract=contract, contract_path=path
        )
        verify_campaign_submission_receipt(job_id=job_id, contract=contract)

    runtime = {
        "arm_role": arm_role,
        "contract_sha256": contract_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(source["sha256"]),
        "checkpoint_step": str(source["step"]),
        "selection_manifest": str(selection),
        "dataset_index": str(index),
        "validation_exclusion_manifest": str(
            repo_path(preregistration["validation_exclusion"]["manifest"])
        ),
        "evaluation_cohort_manifest": str(
            repo_path(preregistration["evaluation_cohort"]["manifest"])
        ),
        "evaluation_cohort_sha256": str(
            preregistration["evaluation_cohort"]["manifest_sha256"]
        ),
        "evaluation_event_uid_count": str(len(evaluation_cohort["event_uids"])),
        "gpu_environment": str(gpu_environment),
        "output_root": str(output_root),
        "expected_gres": str(resources["gres"]),
        "expected_git_sha": expected_sha,
        "max_steps": str(config["max_steps"]),
        "max_restarts": "0",
        "task_id": str(contract["task_id"]),
    }
    return contract, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, nargs="?")
    parser.add_argument("--shell-output", type=Path)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--expected-contract-sha256")
    args = parser.parse_args()
    if args.attempt_receipt is not None:
        if args.contract is not None or args.shell_output is not None:
            parser.error("attempt-receipt verification does not accept a contract")
        if not args.expected_contract_sha256:
            parser.error("--expected-contract-sha256 is required")
        verify_attempt_receipt(
            args.attempt_receipt,
            expected_contract_sha256=args.expected_contract_sha256,
        )
        print(json.dumps({"status": "completed"}, sort_keys=True))
        return 0
    if args.contract is None or args.shell_output is None:
        parser.error("contract and --shell-output are required")
    _, runtime = verify_contract(args.contract, require_slurm=True)
    args.shell_output.write_text(
        "\n".join(
            f"{key}={shlex.quote(value)}" for key, value in runtime.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
