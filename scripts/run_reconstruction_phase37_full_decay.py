#!/usr/bin/env python3
"""Evaluate phase-37 checkpoint tracks and enforce preregistered hierarchy gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reconstruction_phase35_full_decay import (  # noqa: E402
    _artifact,
    _run_evaluator,
    _structural_guardrails,
    atomic_json,
)
from scripts.run_reconstruction_phase37 import (  # noqa: E402
    ALLOWED_SELECTION_STEPS,
    verify_contract,
)


TRACKS = {
    "primary_complete_target": "best.pt",
    "independent_complete_target": "best_rollout_complete_target_efficiency.pt",
    "independent_depth": "best_rollout_depth_fraction.pt",
    "independent_tree_validity": "best_rollout_tree_validity.pt",
}


def _metric(report: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = report
    for key in keys:
        value = value[key]
    if not isinstance(value, dict):
        raise RuntimeError(f"phase37 metric is malformed: {'.'.join(keys)}")
    return {
        "numerator": value["numerator"],
        "denominator": value["denominator"],
        "value": value["value"],
    }


def _training_metric(checkpoint: Path, name: str) -> float:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics", {})
    if name not in metrics:
        raise RuntimeError(f"selected checkpoint has no {name}")
    return float(metrics[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract, runtime = verify_contract(args.contract.resolve(strict=True))
    training_result = json.loads(
        args.training_result.resolve(strict=True).read_text(encoding="utf-8")
    )
    if (
        training_result.get("status") != "training_completed"
        or training_result.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise RuntimeError("phase37 training result is not evaluation-eligible")
    output_dir = args.output_dir.resolve()
    output = args.output.resolve()
    if output.exists() or output_dir.exists():
        raise RuntimeError("refusing to reuse phase37 full-decay outputs")
    output_dir.mkdir(parents=True)

    cohort = json.loads(Path(runtime["cohort"]).read_text(encoding="utf-8"))
    evaluation_manifest = output_dir / "evaluation-cohort.json"
    atomic_json(
        evaluation_manifest,
        {
            "manifest_version": "hypertagging-reconstruction-evaluation-cohort-v1",
            "role": "validation",
            "sealed_test_role_access": "forbidden",
            "event_uid_count": cohort["evaluation_event_uid_count"],
            "event_uids_sha256": cohort["evaluation_event_uids_sha256"],
            "event_uids": cohort["evaluation_event_uids"],
        },
    )
    python = Path(runtime["gpu_environment"]) / "bin/python"
    training_dir = args.training_dir.resolve(strict=True)
    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for label, filename in TRACKS.items():
        checkpoint = (training_dir / filename).resolve(strict=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = int(payload.get("step", -1))
        if step not in ALLOWED_SELECTION_STEPS:
            raise RuntimeError(f"phase37 {label} selected invalid step {step}")
        report_path = output_dir / f"{label}_direct.json"
        log_path = output_dir / f"{label}_direct.log.json"
        reports[label] = _run_evaluator(
            python=python,
            checkpoint=checkpoint,
            checkpoint_step=step,
            contract=contract,
            runtime=runtime,
            cohort_manifest=evaluation_manifest,
            topology_mode="checkpoint_direct",
            output=report_path,
            log=log_path,
            allow_finetuned_encoder=(
                step > int(contract["config"]["freeze_pretrained_encoder_steps"])
            ),
            object_threshold=float(contract["config"]["rollout_object_threshold"]),
            pointer_threshold=float(contract["config"]["rollout_pointer_threshold"]),
        )
        artifacts[label] = {
            "checkpoint": _artifact(checkpoint),
            "report": _artifact(report_path),
            "log": _artifact(log_path),
        }
        selected[label] = {
            "step": step,
            "micro_complete_target_efficiency": _training_metric(
                checkpoint, "micro_complete_target_efficiency"
            ),
            "predicted_depth_fraction": _training_metric(
                checkpoint, "predicted_depth_fraction"
            ),
            "predicted_tree_validity_rate": _training_metric(
                checkpoint, "predicted_tree_validity_rate"
            ),
        }

    primary = reports["primary_complete_target"]
    gates = dict(contract["post_training_gates"])
    endpoints = {
        "full_root_completion": _metric(
            primary,
            "summaries",
            "full",
            "inference",
            "configured_root_completion",
        ),
        "full_lcag": _metric(
            primary,
            "summaries",
            "full",
            "decay_metrics",
            "lcag_pair_accuracy",
        ),
        "exact_mother_coverage": _metric(
            primary,
            "summaries",
            "full",
            "decay_metrics",
            "mother_pid_coverage",
        ),
        "full_source_recall": _metric(
            primary,
            "summaries",
            "full",
            "decay_metrics",
            "source_recall",
        ),
        "full_source_precision": _metric(
            primary,
            "summaries",
            "full",
            "decay_metrics",
            "source_precision",
        ),
        "half_source_precision": _metric(
            primary,
            "summaries",
            "half",
            "decay_metrics",
            "source_precision",
        ),
        "half_perfect_lcag": _metric(
            primary,
            "summaries",
            "half",
            "decay_metrics",
            "perfect_lcag",
        ),
        "half_root_pid_accuracy": _metric(
            primary,
            "summaries",
            "half",
            "decay_metrics",
            "root_pid_accuracy",
        ),
    }
    primary_training = selected["primary_complete_target"]
    checks = {
        "nonzero_full_root_completion": endpoints["full_root_completion"][
            "numerator"
        ]
        >= gates["minimum_full_root_completion_numerator"],
        "nonzero_full_lcag": endpoints["full_lcag"]["numerator"]
        >= gates["minimum_full_lcag_numerator"],
        "nonzero_exact_mother_coverage": endpoints["exact_mother_coverage"][
            "numerator"
        ]
        >= gates["minimum_exact_mother_coverage_numerator"],
        "minimum_full_source_recall": endpoints["full_source_recall"]["value"]
        >= gates["minimum_full_source_recall"],
        "minimum_full_source_precision": endpoints["full_source_precision"]["value"]
        >= gates["minimum_full_source_precision"],
        "minimum_half_source_precision": endpoints["half_source_precision"]["value"]
        >= gates["minimum_half_source_precision"],
        "minimum_half_perfect_lcag": endpoints["half_perfect_lcag"]["numerator"]
        >= gates["minimum_half_perfect_lcag_numerator"],
        "minimum_half_root_pid_accuracy": endpoints["half_root_pid_accuracy"]["value"]
        >= gates["minimum_half_root_pid_accuracy"],
        "minimum_depth_fraction": primary_training["predicted_depth_fraction"]
        >= gates["minimum_predicted_depth_fraction"],
        "minimum_complete_target_efficiency": primary_training[
            "micro_complete_target_efficiency"
        ]
        >= gates["minimum_complete_target_efficiency"],
        "structural_guardrails": _structural_guardrails(primary),
    }
    all_passed = all(checks.values())
    payload = {
        "result_version": "hypertagging-reconstruction-phase37-full-decay-gates-v1",
        "status": "completed",
        "study_id": contract["study_id"],
        "task_id": contract["task_id"],
        "arm_role": contract["arm_role"],
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_selection_metric": "micro_complete_target_efficiency",
        "independent_checkpoint_tracks": [
            "micro_complete_target_efficiency",
            "predicted_depth_fraction",
            "predicted_tree_validity_rate",
        ],
        "evaluation_role": "validation",
        "sealed_test_role_access": "forbidden",
        "evaluation_cohort": _artifact(evaluation_manifest),
        "selected_checkpoints": selected,
        "artifacts": artifacts,
        "primary_full_decay_endpoints": endpoints,
        "preregistered_gate_thresholds": gates,
        "preregistered_gate_checks": checks,
        "all_preregistered_gates_passed": all_passed,
        "eligible_for_cross_arm_gate_review": all_passed,
        "longer_run_authorized": False,
        "promotion_authorized": False,
        "sealed_test_request_authorized": False,
        "authority_note": (
            "Passing one arm only permits a later cross-arm review; it does not "
            "itself authorize longer training, promotion, or sealed-test access."
        ),
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "all_preregistered_gates_passed": all_passed,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
