#!/usr/bin/env python3
"""Compare one preregistered phase-35 checkpoint with the phase-34 baseline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_reconstruction_phase35_contract import (  # noqa: E402
    load_evaluation_cohort,
    load_preregistration,
    load_validation_exclusions,
    verify_contract,
)


ALLOWED_SELECTION_STEPS = (500, 1000, 1500, 2000, 2188)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _under(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve()
    if resolved_root not in resolved.parents:
        raise RuntimeError(f"phase35 evaluation output escapes run root: {candidate}")
    return resolved


def _ratio(report: dict[str, Any], *path: str) -> float:
    value: Any = report
    for key in path:
        value = value[key]
    if isinstance(value, dict):
        value = value.get("value")
    return float(value) if value is not None else 0.0


COMPARISON_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "configured_root_completion",
        ("summaries", "full", "inference", "configured_root_completion"),
    ),
    (
        "full_decay_source_recall",
        ("summaries", "full", "decay_metrics", "source_recall"),
    ),
    (
        "full_decay_lcag_pair_accuracy",
        ("summaries", "full", "decay_metrics", "lcag_pair_accuracy"),
    ),
    (
        "full_decay_mother_pid_coverage",
        ("summaries", "full", "decay_metrics", "mother_pid_coverage"),
    ),
)


def _metric(report: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = report
    for key in path:
        value = value[key]
    if not isinstance(value, dict) or not {
        "numerator",
        "denominator",
        "value",
    }.issubset(value):
        raise RuntimeError(f"full-decay endpoint is malformed: {'.'.join(path)}")
    return {
        "numerator": value["numerator"],
        "denominator": value["denominator"],
        "value": value["value"],
    }


def _comparison_endpoints(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for name, path in COMPARISON_ENDPOINTS:
        baseline_metric = _metric(baseline, path)
        candidate_metric = _metric(candidate, path)
        if baseline_metric["value"] is None or candidate_metric["value"] is None:
            raise RuntimeError(f"full-decay endpoint is undefined: {name}")
        baseline_value = float(baseline_metric["value"])
        candidate_value = float(candidate_metric["value"])
        delta = candidate_value - baseline_value
        if not math.isfinite(delta):
            raise RuntimeError(f"full-decay endpoint delta is non-finite: {name}")
        endpoints[name] = {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
            "absolute_delta": delta,
        }
    return endpoints


def _structural_guardrails(report: dict[str, Any]) -> bool:
    for scope in ("full", "half"):
        inference = report["summaries"][scope]["inference"]
        if (
            _ratio(inference, "inference_structurally_valid") < 0.999
            or _ratio(inference, "p4_closure") < 1.0
            or _ratio(inference, "recursive_detector_sources_disjoint") < 1.0
            or _ratio(inference, "forest_root_sources_disjoint") < 1.0
        ):
            return False
    return True


def _run_evaluator(
    *,
    python: Path,
    checkpoint: Path,
    checkpoint_step: int,
    contract: dict[str, Any],
    runtime: dict[str, str],
    cohort_manifest: Path,
    topology_mode: str,
    output: Path,
    log: Path,
    allow_finetuned_encoder: bool,
    object_threshold: float = 0.5,
    pointer_threshold: float | None = None,
    threads: int = 4,
    deterministic_algorithms: bool = False,
    beam_width: int = 1,
    beam_max_events: int = 20,
    beam_max_proposals: int = 12,
    max_events: int = 100,
    scope: str = "both",
) -> dict[str, Any]:
    if threads <= 0:
        raise ValueError("full-decay evaluator threads must be positive")
    command = [
        str(python),
        "scripts/evaluate_full_decay.py",
        "--pretraining-checkpoint",
        runtime["checkpoint"],
        "--reconstruction-checkpoint",
        str(checkpoint),
        "--data",
        runtime["selection_manifest"],
        "--dataset-index",
        runtime["dataset_index"],
        "--split",
        "validation",
        "--event-uid-manifest",
        str(cohort_manifest),
        "--scope",
        scope,
        "--truth-topology-mode",
        topology_mode,
        "--max-events",
        str(max_events),
        "--max-level",
        "6",
        "--object-threshold",
        str(object_threshold),
        "--threads",
        str(threads),
        "--omit-trees",
        "--output",
        str(output),
    ]
    if allow_finetuned_encoder:
        command.append("--allow-finetuned-encoder")
    if deterministic_algorithms:
        command.append("--deterministic-algorithms")
    if pointer_threshold is not None:
        command.extend(("--pointer-threshold", str(pointer_threshold)))
    if beam_width > 1:
        command.extend(
            (
                "--beam-width",
                str(beam_width),
                "--beam-max-events",
                str(beam_max_events),
                "--beam-max-proposals",
                str(beam_max_proposals),
            )
        )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=7_200,
    )
    log.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"strict full-decay evaluator failed for step {checkpoint_step}: "
            f"{completed.stderr[-1000:]}"
        )
    report = json.loads(output.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_manifest.read_text(encoding="utf-8"))
    if (
        report.get("report_version")
        != "hypertagging-offline-full-decay-evaluation-v3"
        or report.get("device") != "cpu"
        or report.get("torch_num_threads") != threads
        or report.get("torch_deterministic_algorithms_enabled")
        is not deterministic_algorithms
        or report.get("configuration", {}).get("max_level") != 6
        or report.get("configuration", {}).get("truth_topology_mode")
        != topology_mode
        or report.get("configuration", {}).get("trees_included") is not False
        or report.get("context", {}).get("evaluation_split") != "validation"
        or report.get("context", {}).get("evaluation_event_selection")
        != "explicit_uid_cohort"
        or report.get("context", {}).get("evaluated_event_uids")
        != cohort["event_uids"]
        or report.get("context", {}).get("evaluation_uid_train_overlap") != []
        or report.get("checkpoint_pair", {}).get("reconstruction_sha256")
        != sha256(checkpoint)
        or report.get("checkpoint_pair", {}).get("reconstruction_step")
        != checkpoint_step
        or report.get("checkpoint_pair", {}).get("compatible") is not True
    ):
        raise RuntimeError(
            f"strict full-decay report contract failed for step {checkpoint_step}"
        )
    return report


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract, runtime = verify_contract(args.contract.resolve(strict=True))
    run_root = Path(runtime["output_root"]) / str(os.environ["SLURM_JOB_ID"])
    run_root = run_root.resolve(strict=True)
    training_result_path = _under(run_root, args.training_result)
    training_dir = _under(run_root, args.training_dir)
    output_dir = _under(run_root, args.output_dir)
    output = _under(run_root, args.output)
    if output.exists() or output_dir.exists():
        raise RuntimeError("refusing to reuse phase35 full-decay outputs")
    output_dir.mkdir(parents=True)
    training_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    if (
        training_result.get("status") not in {"completed", "training_completed"}
        or training_result.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise RuntimeError("phase35 training result is not eligible for evaluation")
    prereg = load_preregistration()
    exclusions = load_validation_exclusions(prereg)
    cohort = load_evaluation_cohort(prereg, exclusions=exclusions)
    cohort_manifest = Path(runtime["evaluation_cohort_manifest"])
    if sha256(cohort_manifest) != runtime["evaluation_cohort_sha256"]:
        raise RuntimeError("phase35 evaluation cohort changed before execution")
    python = Path(runtime["gpu_environment"]) / "bin/python"
    selected_checkpoint = (training_dir / "best.pt").resolve(strict=True)
    selected_payload = torch.load(
        selected_checkpoint, map_location="cpu", weights_only=False
    )
    selected_step = int(selected_payload.get("step", -1))
    selected_audit = training_result.get("checkpoints", {}).get("best", {})
    if (
        selected_step not in ALLOWED_SELECTION_STEPS
        or selected_audit.get("step") != selected_step
        or selected_audit.get("sha256") != sha256(selected_checkpoint)
        or Path(str(selected_audit.get("path", ""))).resolve()
        != selected_checkpoint
    ):
        raise RuntimeError("best.pt is not the preregistered training selection")

    baseline_binding = prereg["baseline_reconstruction_checkpoint"]
    baseline_checkpoint = (ROOT / baseline_binding["path"]).resolve(strict=True)
    baseline_step = int(baseline_binding["step"])
    if sha256(baseline_checkpoint) != baseline_binding["sha256"]:
        raise RuntimeError("phase34 reconstruction baseline changed")

    finetuned = (
        contract["arm_role"].endswith("encoder_adapt")
        and selected_step
        > int(contract["config"]["freeze_pretrained_encoder_steps"])
    )
    reports: dict[str, dict[str, Any]] = {}
    report_artifacts: dict[str, dict[str, Any]] = {}
    log_artifacts: dict[str, dict[str, Any]] = {}
    evaluations = (
        ("baseline_checkpoint_direct", baseline_checkpoint, baseline_step, False, "checkpoint_direct"),
        ("baseline_contracted_diagnostic", baseline_checkpoint, baseline_step, False, "contracted_diagnostic"),
        ("candidate_checkpoint_direct", selected_checkpoint, selected_step, finetuned, "checkpoint_direct"),
        ("candidate_contracted_diagnostic", selected_checkpoint, selected_step, finetuned, "contracted_diagnostic"),
    )
    for label, checkpoint, checkpoint_step, allow_finetuned, topology_mode in evaluations:
        report_path = output_dir / f"{label}.json"
        log_path = output_dir / f"{label}.log.json"
        reports[label] = _run_evaluator(
            python=python,
            checkpoint=checkpoint,
            checkpoint_step=checkpoint_step,
            contract=contract,
            runtime=runtime,
            cohort_manifest=cohort_manifest,
            topology_mode=topology_mode,
            output=report_path,
            log=log_path,
            allow_finetuned_encoder=allow_finetuned,
        )
        report_artifacts[label] = _artifact(report_path)
        log_artifacts[label] = _artifact(log_path)

    baseline_direct = reports["baseline_checkpoint_direct"]
    candidate_direct = reports["candidate_checkpoint_direct"]
    comparison = _comparison_endpoints(baseline_direct, candidate_direct)
    payload = {
        "result_version": "hypertagging-reconstruction-phase35-full-decay-comparison-v1",
        "status": "completed",
        "study_id": contract["study_id"],
        "task_id": contract["task_id"],
        "arm_role": contract["arm_role"],
        "contract_sha256": contract["contract_sha256"],
        "comparison_role": (
            "exploratory_validation_comparison_not_checkpoint_or_arm_selection_"
            "not_confirmation_not_promotion"
        ),
        "sealed_test_role_access": "forbidden",
        "automatic_promotion": False,
        "max_level": 6,
        "evaluation_cohort": {
            "manifest": _artifact(cohort_manifest),
            "event_uid_count": len(cohort["event_uids"]),
            "event_uids_sha256": cohort["event_uids_sha256"],
            "checkpoint_selection_overlap": 0,
            "historical_exclusion_overlap": 0,
        },
        "checkpoint_selection": {
            "checkpoint": "best.pt",
            "metric": "predicted_edge_f1",
            "cohort_role": "checkpoint_selection_validation",
            "evaluation_cohort_used_for_selection": False,
            "allowed_steps": list(ALLOWED_SELECTION_STEPS),
        },
        "baseline_checkpoint": _artifact(baseline_checkpoint),
        "baseline_checkpoint_step": baseline_step,
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint": _artifact(selected_checkpoint),
        "candidate_encoder_finetuned": finetuned,
        "reports": report_artifacts,
        "logs": log_artifacts,
        "baseline_checkpoint_direct_metrics": baseline_direct["summaries"],
        "candidate_checkpoint_direct_metrics": candidate_direct["summaries"],
        "checkpoint_direct_endpoint_comparison": comparison,
        "structural_guardrails": {
            "baseline_checkpoint_direct": _structural_guardrails(baseline_direct),
            "candidate_checkpoint_direct": _structural_guardrails(candidate_direct),
            "candidate_contracted_diagnostic": _structural_guardrails(
                reports["candidate_contracted_diagnostic"]
            ),
        },
        "interpretation_policy": (
            "report_each_arm_independently_against_baseline_on_identical_events;"
            "do_not_rank_or_promote;follow_up_selection_requires_a_separately_"
            "preregistered_untouched_cohort_or_sealed_test_authorization"
        ),
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "selected_checkpoint_step": selected_step,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
