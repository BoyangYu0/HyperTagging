#!/usr/bin/env python3
"""Fail-closed scientific planning gate; never reserve data or submit a job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(plan: dict, policy: dict, evidence: dict) -> dict:
    errors = []
    for field in policy["required_plan_fields"]:
        if not plan.get(field):
            errors.append(f"missing plan field: {field}")
    stage = plan.get("stage")
    if stage not in ("development", "primary"):
        errors.append("stage must be development or primary")
    if plan.get("sealed_test_access") is not False:
        errors.append("sealed test access must be explicitly false")
    family = plan.get("family")
    if family in policy["conditional_families"]:
        justification = plan.get("mechanism_evidence", {})
        if not justification.get("report_binding") or not justification.get("measured_effect"):
            errors.append(f"{family}: {policy['conditional_families'][family]}")
        if family == "candidate_reranking":
            gap = any(c["inclusive_pool"]["numerator"] > c["inclusive_top1"]["numerator"]
                      for c in evidence["conditions"].values())
            if not gap:
                errors.append("current evidence has no correct B pool for a reranking-only study")
    elif family not in policy["allowed_development_families"]:
        errors.append("study family is outside the structural policy")
    if plan.get("diagnostic_evidence") != policy["evidence"]:
        errors.append("plan must bind the current reviewed diagnostic evidence")
    if family == "capacity_pretraining_alignment":
        representation = policy["representation_study_policy"]
        expected = {(width, representation["fixed_hyperbolic_dimension"], objective)
                    for width in representation["initial_context_widths"]
                    for objective in representation["pretraining_objectives"]}
        arms = plan.get("arms", [])
        actual = [(arm.get("context_width"), arm.get("hyperbolic_dimension"),
                   arm.get("pretraining_objective")) for arm in arms]
        if len(actual) != len(expected) or any(actual.count(arm) != 1 for arm in expected):
            errors.append("representation study requires the complete width/objective factorial with fixed hyperbolic dimension")
        for control in representation["required_controls"]:
            if plan.get("representation_controls", {}).get(control) is not True:
                errors.append(f"missing representation control: {control}")
        for endpoint in representation["required_endpoints"]:
            if endpoint not in plan.get("evaluation_endpoints", []):
                errors.append(f"missing representation endpoint: {endpoint}")
    if stage == "primary":
        gates = plan.get("development_gate_results", {})
        minimum = policy["development_gates"]["tiny_training_membership_exact_fraction_min"]
        if gates.get("tiny_training_exact_fraction", 0) < minimum:
            errors.append("tiny training membership gate not met")
        if not gates.get("heldout_membership_gain_report"):
            errors.append("heldout membership gain evidence is required before primary scale-up")
        if not gates.get("heldout_background_comparison_report"):
            errors.append("heldout background comparison is required")
        coverage = plan.get("primary_evaluation", {})
        required = policy["primary_gates"]
        for category in required["categories"]:
            if coverage.get("reserved_events_by_category", {}).get(category, 0) < required["minimum_distinct_events_per_category_per_arm"]:
                errors.append(f"primary reservation shortage: {category}")
        for key in ("fresh_reserved_cohort", "separate_checkpoint_selection", "train_disjoint"):
            if coverage.get(key) is not True:
                errors.append(f"missing primary cohort invariant: {key}")
    return {"policy_version": policy["version"], "status": "PASS" if not errors else "FAIL",
            "errors": errors, "scope": "scientific_plan_only_not_runtime_admission_or_submission",
            "requires_existing_authenticated_cohort_and_scheduler_admission": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/reconstruction/next_study_policy.json")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    binding = policy["evidence"]
    path = ROOT / binding["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError("Policy evidence hash mismatch")
    result = validate(json.loads(args.plan.read_text()), policy, json.loads(path.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
