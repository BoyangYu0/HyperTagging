#!/usr/bin/env python3
"""Write a hashed terminal receipt for a reconstruction Slurm attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_reconstruction_phase35_contract import (  # noqa: E402
    CONTRACT_VERSION as PHASE35_CONTRACT_VERSION,
    load_preregistration,
    verify_campaign_submission_receipt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--stage-log", type=Path, required=True)
    parser.add_argument("--wrapper-status", type=Path, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--batch-exit-status", type=int, required=True)
    parser.add_argument("--terminal-stage", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    wrapper = (
        json.loads(args.wrapper_status.read_text(encoding="utf-8"))
        if args.wrapper_status.is_file()
        else {}
    )
    run_root = (
        Path(args.run_root)
        if args.run_root
        else args.attempt_root / "__absent_run_root__"
    )
    result_path = run_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    full_decay_path = run_root / "full-decay-comparison.json"
    full_decay = (
        json.loads(full_decay_path.read_text(encoding="utf-8"))
        if full_decay_path.is_file()
        else None
    )
    telemetry_path = args.attempt_root / "gpu-telemetry.jsonl"
    telemetry_summary_path = args.attempt_root / "gpu-telemetry-summary.json"
    telemetry = (
        json.loads(telemetry_summary_path.read_text(encoding="utf-8"))
        if telemetry_summary_path.is_file()
        else {}
    )
    config = contract.get("config", {})
    # Keep terminal-evidence enforcement for both versions. A legacy contract
    # processed from this repair checkout intentionally fails closed at the
    # repair-only campaign verifier; re-finalization must use its pinned tag.
    phase35 = contract.get("contract_version") in {
        "hypertagging-reconstruction-phase35-improvement-contract-v1",
        PHASE35_CONTRACT_VERSION,
    }
    checkpoint_audits = (
        result.get("checkpoints", {}) if isinstance(result, dict) else {}
    )
    training_root = run_root / "training"
    phase35_checkpoint_paths = {
        "final": training_root / "checkpoint.pt",
        "best": training_root / "best.pt",
        "best_rollout_edge_f1": training_root / "best_rollout_edge_f1.pt",
    }
    expected_run_root = (
        Path.cwd()
        / str(contract.get("output_root", "__invalid_output_root__"))
        / str(os.environ.get("SLURM_JOB_ID", "__missing_job_id__"))
    )
    phase35_run_root_health = (
        not phase35
        or (
            bool(args.run_root)
            and run_root.is_dir()
            and run_root.resolve() == expected_run_root.resolve()
        )
    )
    campaign_receipt_path = Path(
        str(contract.get("campaign_submission_receipt", ""))
    )
    if not campaign_receipt_path.is_absolute():
        campaign_receipt_path = Path.cwd() / campaign_receipt_path
    phase35_campaign_receipt_health = not phase35
    campaign_receipt: dict[str, Any] = {}
    if phase35:
        try:
            campaign_receipt = verify_campaign_submission_receipt(
                job_id=str(os.environ.get("SLURM_JOB_ID", "")),
                contract=contract,
                allowed_statuses=("release_in_progress", "submitted"),
            )
            phase35_campaign_receipt_health = True
        except (OSError, KeyError, TypeError, ValueError, RuntimeError):
            phase35_campaign_receipt_health = False

    hashed_input_mismatches: list[str] = []
    if phase35:
        hashed_inputs = contract.get("hashed_inputs")
        if not isinstance(hashed_inputs, list) or not hashed_inputs:
            hashed_input_mismatches.append("missing_hashed_inputs")
        else:
            for item in hashed_inputs:
                if not isinstance(item, dict):
                    hashed_input_mismatches.append("malformed_hashed_input")
                    continue
                input_path = Path(str(item.get("path", "")))
                if not input_path.is_absolute():
                    input_path = Path.cwd() / input_path
                expected = str(item.get("sha256", ""))
                if (
                    not input_path.is_file()
                    or len(expected) != 64
                    or sha256(input_path) != expected
                ):
                    hashed_input_mismatches.append(str(item.get("path", "")))
    phase35_hashed_inputs_health = not phase35 or not hashed_input_mismatches
    source_checkpoint_path = Path(str(contract.get("checkpoint", "")))
    if not source_checkpoint_path.is_absolute():
        source_checkpoint_path = Path.cwd() / source_checkpoint_path
    phase35_source_health = (
        not phase35
        or (
            source_checkpoint_path.is_file()
            and sha256(source_checkpoint_path)
            == contract.get("checkpoint_sha256")
        )
    )

    def phase35_checkpoint_matches(name: str) -> bool:
        audit = checkpoint_audits.get(name)
        path = phase35_checkpoint_paths[name]
        if not isinstance(audit, dict) or not path.is_file():
            return False
        expected_hash = audit.get("sha256")
        return (
            audit.get("all_checkpoint_tensors_finite") is True
            and audit.get("all_model_tensors_finite") is True
            and isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and all(character in "0123456789abcdef" for character in expected_hash)
            and sha256(path) == expected_hash
            and int(audit.get("bytes", -1)) == path.stat().st_size
            and Path(str(audit.get("path", ""))).resolve() == path.resolve()
        )

    def audited_artifact_matches(
        audit: Any, expected_path: Path | None = None
    ) -> bool:
        if not isinstance(audit, dict):
            return False
        path = Path(str(audit.get("path", "")))
        if not path.is_file():
            return False
        if expected_path is not None and path.resolve() != expected_path.resolve():
            return False
        expected_hash = str(audit.get("sha256", ""))
        return (
            len(expected_hash) == 64
            and sha256(path) == expected_hash
            and path.stat().st_size == int(audit.get("bytes", -1))
        )

    phase35_identity_health = (
        not phase35
        or (
            isinstance(result, dict)
            and result.get("contract_sha256") == contract.get("contract_sha256")
            and result.get("study_id") == contract.get("study_id")
            and result.get("task_id") == contract.get("task_id")
            and result.get("arm_role") == contract.get("arm_role")
            and result.get("config") == config
            and result.get("data", {}).get("split_counts")
            == contract.get("data", {}).get("split_counts")
            and result.get("data", {}).get("sealed_test_role_access")
            == "forbidden"
            and result.get("post_training_policy", {}).get(
                "automatic_promotion"
            )
            is False
            and result.get("source_checkpoint", {}).get("sha256_before")
            == contract.get("checkpoint_sha256")
            and result.get("source_checkpoint", {}).get("sha256_after")
            == contract.get("checkpoint_sha256")
        )
    )
    phase35_checkpoint_health = (
        not phase35
        or all(
            phase35_checkpoint_matches(name)
            for name in ("final", "best", "best_rollout_edge_f1")
        )
    )
    candidate_steps = (500, 1000, 1500, 2000, 2188)
    phase35_full_decay_health = not phase35
    if phase35 and isinstance(full_decay, dict):
        preregistration = load_preregistration()
        baseline_binding = preregistration["baseline_reconstruction_checkpoint"]
        baseline_checkpoint = ROOT / str(baseline_binding["path"])
        report_labels = {
            "baseline_checkpoint_direct",
            "baseline_contracted_diagnostic",
            "candidate_checkpoint_direct",
            "candidate_contracted_diagnostic",
        }
        reports = full_decay.get("reports")
        logs = full_decay.get("logs")
        report_artifacts_health = (
            isinstance(reports, dict)
            and set(reports) == report_labels
            and isinstance(logs, dict)
            and set(logs) == report_labels
        )
        if report_artifacts_health:
            reports_root = run_root / "full-decay-reports"
            for label in sorted(report_labels):
                report_artifacts_health = (
                    report_artifacts_health
                    and audited_artifact_matches(
                        reports[label], reports_root / f"{label}.json"
                    )
                    and audited_artifact_matches(
                        logs[label], reports_root / f"{label}.log.json"
                    )
                )
        selected_step = full_decay.get("selected_checkpoint_step")
        expected_checkpoint_selection = {
            "checkpoint": "best.pt",
            "metric": "predicted_edge_f1",
            "cohort_role": "checkpoint_selection_validation",
            "evaluation_cohort_used_for_selection": False,
            "allowed_steps": list(candidate_steps),
        }
        expected_finetuned = bool(
            str(contract.get("arm_role", "")).endswith("encoder_adapt")
            and isinstance(selected_step, int)
            and selected_step > int(config["freeze_pretrained_encoder_steps"])
        )
        evaluation_manifest = ROOT / str(
            contract.get("evaluation_cohort", {}).get("manifest", "")
        )
        endpoint_comparison = full_decay.get(
            "checkpoint_direct_endpoint_comparison"
        )
        endpoint_comparison_health = (
            isinstance(endpoint_comparison, dict)
            and set(endpoint_comparison)
            == {
                "configured_root_completion",
                "full_decay_source_recall",
                "full_decay_lcag_pair_accuracy",
                "full_decay_mother_pid_coverage",
            }
        )
        phase35_full_decay_health = bool(
            full_decay.get("result_version")
            == "hypertagging-reconstruction-phase35-full-decay-comparison-v1"
            and full_decay.get("status") == "completed"
            and full_decay.get("study_id") == contract.get("study_id")
            and full_decay.get("task_id") == contract.get("task_id")
            and full_decay.get("arm_role") == contract.get("arm_role")
            and full_decay.get("contract_sha256")
            == contract.get("contract_sha256")
            and full_decay.get("sealed_test_role_access") == "forbidden"
            and full_decay.get("automatic_promotion") is False
            and full_decay.get("max_level") == 6
            and full_decay.get("comparison_role")
            == (
                "exploratory_validation_comparison_not_checkpoint_or_arm_"
                "selection_not_confirmation_not_promotion"
            )
            and full_decay.get("evaluation_cohort", {}).get("event_uid_count")
            == contract.get("evaluation_cohort", {}).get("event_uid_count")
            and full_decay.get("evaluation_cohort", {}).get("event_uids_sha256")
            == contract.get("evaluation_cohort", {}).get("event_uids_sha256")
            and audited_artifact_matches(
                full_decay.get("evaluation_cohort", {}).get("manifest"),
                evaluation_manifest,
            )
            and full_decay.get("checkpoint_selection")
            == expected_checkpoint_selection
            and selected_step in candidate_steps
            and selected_step == checkpoint_audits.get("best", {}).get("step")
            and audited_artifact_matches(
                full_decay.get("selected_checkpoint"), training_root / "best.pt"
            )
            and audited_artifact_matches(
                full_decay.get("baseline_checkpoint"), baseline_checkpoint
            )
            and full_decay.get("baseline_checkpoint_step")
            == baseline_binding.get("step")
            and full_decay.get("candidate_encoder_finetuned")
            is expected_finetuned
            and report_artifacts_health
            and endpoint_comparison_health
            and full_decay.get("interpretation_policy")
            == (
                "report_each_arm_independently_against_baseline_on_identical_"
                "events;do_not_rank_or_promote;follow_up_selection_requires_a_"
                "separately_preregistered_untouched_cohort_or_sealed_test_"
                "authorization"
            )
            and finite(full_decay)
        )
    phase35_wrapper_health = (
        not phase35
        or (
            wrapper.get("action") == "trainer_exit"
            and wrapper.get("wrapper_status") == 0
            and wrapper.get("trainer_status") == 0
            and wrapper.get("max_restarts") == 0
            and wrapper.get("restart_count") == 0
            and wrapper.get("termination_received") == 0
            and wrapper.get("usr1_received") == 0
        )
    )

    def telemetry_covered_training() -> bool:
        if not phase35:
            return True
        if not isinstance(result, dict):
            return False
        try:
            started = datetime.fromisoformat(str(telemetry["started_at"]))
            completed = datetime.fromisoformat(str(telemetry["completed_at"]))
            duration = (completed - started).total_seconds()
            interval = float(telemetry["interval_seconds"])
            training_seconds = float(result["elapsed_seconds"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            telemetry.get("telemetry_version")
            == "hypertagging-slurm-gpu-telemetry-v1"
            and telemetry.get("stop_reason") == "wrapper_authorized_signal"
            and duration >= max(training_seconds - max(2 * interval, 60.0), 0.0)
            and int(telemetry.get("sample_count", 0))
            >= max(int(duration / max(interval, 1.0)) - 2, 1)
        )

    healthy_result = (
        isinstance(result, dict)
        and result.get("status") in {"completed", "training_completed"}
        and result.get("optimizer_steps") == config.get("max_steps")
        and result.get("source_checkpoint", {}).get("unchanged") is True
        and result.get("data", {}).get("split_counts", {}).get("test", 1) == 0
        and result.get("first_20_optimizer_steps", {}).get("passed") is True
        and phase35_identity_health
        and phase35_checkpoint_health
        and phase35_full_decay_health
        and phase35_source_health
        and phase35_hashed_inputs_health
        and phase35_run_root_health
        and phase35_campaign_receipt_health
        and finite(result)
    )
    terminal_success = (
        args.batch_exit_status == 0
        and args.terminal_stage
        == ("full_decay_complete" if phase35 else "trainer_complete")
        and wrapper.get("action") == "trainer_exit"
        and wrapper.get("wrapper_status") == 0
        and phase35_wrapper_health
        and healthy_result
        and telemetry.get("status") == "completed"
        and int(telemetry.get("sample_count", 0)) > 0
        and telemetry_covered_training()
    )
    if contract.get("mode") == "calibration" and result is not None:
        terminal_success = terminal_success and float(result.get("elapsed_seconds", 1e99)) <= int(
            contract.get("max_wall_seconds", 900)
        )

    candidates = {
        "job_contract": args.contract,
        "stage_log": args.stage_log,
        "wrapper_status": args.wrapper_status,
        "allocation": args.attempt_root / "allocation.txt",
        "environment": args.attempt_root / "environment.txt",
        "nvidia_smi": args.attempt_root / "nvidia-smi.txt",
        "gpu_query": args.attempt_root / "gpu-query.csv",
        "gpu_preflight": args.attempt_root / "gpu-preflight.json",
        "gpu_telemetry": telemetry_path,
        "gpu_telemetry_summary": telemetry_summary_path,
    }
    if result_path.is_file():
        candidates["result"] = result_path
    if full_decay_path.is_file():
        candidates["full_decay_comparison"] = full_decay_path
    for name, path in (
        ("metrics", training_root / "metrics.jsonl"),
        ("checkpoint", training_root / "checkpoint.pt"),
        ("latest", training_root / "latest.pt"),
        ("best", training_root / "best.pt"),
        ("best_rollout_edge_f1", training_root / "best_rollout_edge_f1.pt"),
        ("best_teacher_forced", training_root / "best_teacher_forced.pt"),
        (
            "best_rollout_tree_validity",
            training_root / "best_rollout_tree_validity.pt",
        ),
        ("signal_checkpoint", training_root / "signal-checkpoint.pt"),
        ("split_manifest", training_root / "split_manifest.json"),
    ):
        candidates[name] = path
    artifacts = {name: artifact(path) for name, path in candidates.items() if path.is_file()}
    receipt: dict[str, Any] = {
        "receipt_version": (
            "hypertagging-reconstruction-phase35-attempt-v1"
            if phase35
            else "hypertagging-reconstruction-fullscale-attempt-v1"
        ),
        "status": "completed" if terminal_success else "failed_or_nonterminal",
        "terminal_stage": args.terminal_stage,
        "batch_exit_status": args.batch_exit_status,
        "wrapper": wrapper,
        "mode": contract.get("mode"),
        "experiment": contract.get("experiment"),
        "study_id": contract.get("study_id"),
        "task_id": contract.get("task_id"),
        "arm_role": contract.get("arm_role"),
        "contract_sha256": contract.get("contract_sha256"),
        "source_checkpoint": {
            "path": contract.get("checkpoint"),
            "step": contract.get("checkpoint_step"),
            "sha256": contract.get("checkpoint_sha256"),
            "unchanged": bool(result and result.get("source_checkpoint", {}).get("unchanged")),
            "live_sha256_at_finalization": (
                sha256(source_checkpoint_path)
                if source_checkpoint_path.is_file()
                else None
            ),
            "live_hash_matches_contract": phase35_source_health,
        },
        "precision": {
            "gres": (
                contract.get("resources", {}).get("gres")
                if phase35
                else contract.get("gres")
            ),
            "amp_dtype": config.get("amp_dtype"),
            "grad_scaler_enabled": config.get("grad_scaler_enabled"),
        },
        "first_20_optimizer_steps": (
            result.get("first_20_optimizer_steps") if isinstance(result, dict) else None
        ),
        "metrics": result.get("metrics") if isinstance(result, dict) else None,
        "full_decay_comparison": full_decay,
        "checkpoints": (
            result.get("checkpoints") if isinstance(result, dict) else None
        ),
        "data": result.get("data") if isinstance(result, dict) else None,
        "gpu_telemetry": telemetry,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "artifacts": artifacts,
        "terminal_hashed_inputs_audit": {
            "passed": phase35_hashed_inputs_health,
            "checked_count": len(contract.get("hashed_inputs", [])),
            "mismatches": hashed_input_mismatches,
        },
        "campaign_submission_receipt_audit": {
            "path": str(campaign_receipt_path),
            "passed": phase35_campaign_receipt_health,
            "status": campaign_receipt.get("status"),
        },
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.partial")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.receipt)
    return 0 if terminal_success or not phase35 else 1


if __name__ == "__main__":
    raise SystemExit(main())
