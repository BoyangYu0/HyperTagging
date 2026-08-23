#!/usr/bin/env python
"""Collect the exact configured phase-3 calibration receipts.

Aggregation is fail-closed: missing, duplicate, nonhealthy, nonfinite, or
mis-bound receipts prevent an aggregate from being written.  It never calls
Slurm and never authorizes production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError("receipt aggregation exists; refusing to rewrite history")
    plan = load_study_plan(args.study_plan, root=ROOT)
    assert_no_active_calibrations(plan, root=ROOT)
    required_ids = plan["required_receipt_policy"]["required_calibration_ids"]
    configured_ids = [entry["calibration_id"] for entry in plan["calibration_matrix"]]
    if required_ids != configured_ids or len(set(required_ids)) != len(required_ids):
        raise RuntimeError("required receipt set is not exactly the configured matrix")
    receipts: dict[str, dict[str, object]] = {}
    for calibration_id in required_ids:
        entry = entry_by_id(plan, calibration_id)
        receipt_path = resolve_plan_path(entry["receipt_path"], root=ROOT)
        receipt, throughput, objective_ratio = load_healthy_receipt(
            receipt_path, plan, entry, root=ROOT
        )
        queue_delay = float(receipt["queue_delay_seconds"])
        expected_completion = queue_delay + REMAINING_PRESENTATIONS / throughput
        receipts[calibration_id] = {
            "calibration_id": calibration_id,
            "tuple_sha256": entry["tuple_sha256"],
            "receipt_path": entry["receipt_path"],
            "receipt_sha256": receipt["receipt_sha256"],
            "terminal_state": receipt["terminal_state"],
            "queue_delay_seconds": queue_delay,
            "train_throughput_events_per_second": throughput,
            "objective_weighted_dominance_ratio": objective_ratio,
            "expected_completion_seconds": expected_completion,
        }
    if set(receipts) != set(required_ids):
        raise RuntimeError("receipt aggregation omitted a configured calibration")
    try:
        display_plan = str(args.study_plan.resolve().relative_to(ROOT))
    except ValueError:
        display_plan = str(args.study_plan.resolve())
    aggregate: dict[str, object] = {
        "artifact_version": AGGREGATION_VERSION,
        "study_plan": display_plan,
        "study_plan_sha256": file_sha256(args.study_plan),
        "required_receipt_policy": "exact_configured_set",
        "required_calibration_ids": required_ids,
        "receipts": receipts,
        "all_required_terminal_healthy": True,
        "production_submission_authorized": False,
        "submission_performed": False,
        "created_at_unix": time.time(),
    }
    aggregate["aggregation_sha256"] = canonical_hash(aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    temporary.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
