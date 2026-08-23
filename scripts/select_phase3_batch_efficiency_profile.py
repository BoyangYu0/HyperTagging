#!/usr/bin/env python
"""Select one phase-3 continuation from the exact healthy calibration set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.training.phase3_parallel_study import (  # noqa: E402
    AGGREGATION_VERSION,
    REMAINING_PRESENTATIONS,
    assert_no_active_calibrations,
    canonical_hash,
    entry_by_id,
    file_sha256,
    load_healthy_receipt,
    load_study_plan,
    resolve_plan_path,
)


DEFAULT_PLAN = ROOT / "configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json"


def _load_aggregation(
    path: Path, plan: dict[str, Any], plan_path: Path
) -> dict[str, Any]:
    try:
        aggregate = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot load receipt aggregation") from error
    if not isinstance(aggregate, dict):
        raise RuntimeError("receipt aggregation must be an object")
    body = dict(aggregate)
    stored = body.pop("aggregation_sha256", None)
    if stored != canonical_hash(body):
        raise RuntimeError("receipt aggregation hash mismatch")
    if aggregate.get("artifact_version") != AGGREGATION_VERSION:
        raise RuntimeError("unsupported receipt aggregation version")
    plan_path = resolve_plan_path(aggregate.get("study_plan", ""), root=ROOT)
    expected_plan = plan_path.resolve()
    if plan_path != expected_plan or aggregate.get("study_plan_sha256") != file_sha256(plan_path):
        raise RuntimeError("receipt aggregation is bound to a different study plan")
    required = plan["required_receipt_policy"]["required_calibration_ids"]
    if aggregate.get("required_receipt_policy") != "exact_configured_set":
        raise RuntimeError("receipt aggregation does not use the exact configured set")
    if aggregate.get("required_calibration_ids") != required:
        raise RuntimeError("receipt aggregation receipt set differs from the configured set")
    if aggregate.get("all_required_terminal_healthy") is not True:
        raise RuntimeError("receipt aggregation is not terminal healthy")
    receipts = aggregate.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(required):
        raise RuntimeError("receipt aggregation is missing or duplicating configured receipts")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--receipt-aggregation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-production", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError("selection manifest exists; refusing to rewrite history")
    plan = load_study_plan(args.study_plan, root=ROOT)
    assert_no_active_calibrations(plan, root=ROOT)
    aggregate = _load_aggregation(args.receipt_aggregation, plan, args.study_plan)
    candidates: dict[str, dict[str, Any]] = {}
    receipt_hashes: set[str] = set()
    tuple_hashes: set[str] = set()
    for calibration_id in plan["required_receipt_policy"]["required_calibration_ids"]:
        entry = entry_by_id(plan, calibration_id)
        receipt_path = resolve_plan_path(entry["receipt_path"], root=ROOT)
        receipt, throughput, objective_ratio = load_healthy_receipt(
            receipt_path, plan, entry, root=ROOT
        )
        aggregate_row = aggregate["receipts"][calibration_id]
        if aggregate_row.get("receipt_sha256") != receipt["receipt_sha256"]:
            raise RuntimeError("receipt aggregation does not match the current receipt")
        if receipt["receipt_sha256"] in receipt_hashes or entry["tuple_sha256"] in tuple_hashes:
            raise RuntimeError("duplicate receipt or calibration contract hash")
        receipt_hashes.add(receipt["receipt_sha256"])
        tuple_hashes.add(entry["tuple_sha256"])
        queue_delay = float(receipt["queue_delay_seconds"])
        expected_completion = queue_delay + REMAINING_PRESENTATIONS / throughput
        if aggregate_row.get("expected_completion_seconds") != expected_completion:
            raise RuntimeError("receipt aggregation completion estimate is stale")
        candidates[calibration_id] = {
            "profile": entry["profile"],
            "exact_gres": entry["exact_gres"],
            "batch_size": entry["batch_size"],
            "precision_policy": entry["precision_policy"],
            "hypothesis_id": entry["hypothesis_id"],
            "tuple_sha256": entry["tuple_sha256"],
            "receipt": entry["receipt_path"],
            "receipt_sha256": receipt["receipt_sha256"],
            "queue_delay_seconds": queue_delay,
            "train_throughput_events_per_second": throughput,
            "objective_weighted_dominance_ratio": objective_ratio,
            "fixture_peak_memory_mib": receipt["fixture_probe"]["fixture_peak_memory_mib"],
            "expected_completion_seconds": expected_completion,
            "production_config": entry["production_config"],
        }
    if len(candidates) != 4:
        raise RuntimeError("selection did not validate all four configured calibrations")
    selected_id = min(
        candidates,
        key=lambda calibration_id: (
            candidates[calibration_id]["expected_completion_seconds"],
            candidates[calibration_id]["fixture_peak_memory_mib"],
            candidates[calibration_id]["profile"] != "h100nvl",
            calibration_id,
        ),
    )
    selected = candidates[selected_id]
    production_variant_id = (
        f"ht3-production-resume-{selected['profile']}-b{selected['batch_size']}-step54064-v1"
    )
    production_identity = canonical_hash(
        {
            "variant_id": production_variant_id,
            "selected_calibration_id": selected_id,
            "tuple_sha256": selected["tuple_sha256"],
            "parent_implementation_commit": plan["parent_implementation_commit"],
            "resume_presentations": REMAINING_PRESENTATIONS,
        }
    )
    production_contracts = (
        [
            {
                "production_variant_id": production_variant_id,
                "production_contract_identity_sha256": production_identity,
                "selected_calibration_id": selected_id,
                "scientifically_distinct": True,
                "duplicate_of": None,
            }
        ]
        if args.authorize_production
        else []
    )
    try:
        display_plan = str(args.study_plan.resolve().relative_to(ROOT))
        display_aggregation = str(args.receipt_aggregation.resolve().relative_to(ROOT))
    except ValueError:
        display_plan = str(args.study_plan.resolve())
        display_aggregation = str(args.receipt_aggregation.resolve())
    manifest: dict[str, Any] = {
        "artifact_version": "ht-pretraining-1m-phase3-parallel-selection-v1",
        "study_plan": display_plan,
        "study_plan_sha256": file_sha256(args.study_plan),
        "receipt_aggregation": display_aggregation,
        "receipt_aggregation_sha256": file_sha256(args.receipt_aggregation),
        "required_receipt_policy": "exact_configured_set",
        "calibration_ids": list(candidates),
        "calibration_active_count": 0,
        "calibration_receipts_terminal_healthy": True,
        "candidates": candidates,
        "selected_calibration_id": selected_id,
        "selected_profile": selected["profile"],
        "selected_batch_size": selected["batch_size"],
        "selection_metric": "queue_delay_seconds_plus_remaining_presentations_divided_by_train_throughput",
        "selection_basis": "finite_train_role_throughput_and_objective_safe_receipts_only",
        "remaining_presentations": REMAINING_PRESENTATIONS,
        "production_submission_authorized": bool(args.authorize_production),
        "production_resume_policy": "default_exactly_one",
        "production_variants": production_contracts,
        "production_variant_count": len(production_contracts),
        "duplicate_production_contracts_forbidden": True,
        "submission_performed": False,
        "job_count": 1,
        "one_viable_scientific_lineage": True,
        "created_at_unix": time.time(),
    }
    manifest["selection_sha256"] = canonical_hash(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
