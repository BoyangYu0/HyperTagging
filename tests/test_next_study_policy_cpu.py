import copy
import json
from pathlib import Path

import pytest

from scripts.validate_next_reconstruction_study import validate

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    policy = json.loads((ROOT / "configs/reconstruction/next_study_policy.json").read_text())
    plan = json.loads((ROOT / "configs/reconstruction/structural_membership_development_plan.json").read_text())
    evidence = json.loads((ROOT / policy["evidence"]["path"]).read_text())
    return plan, policy, evidence


def test_development_plan_passes_without_claiming_training_admission():
    plan, policy, evidence = inputs()
    result = validate(plan, policy, evidence)
    assert result["status"] == "PASS"
    assert result["requires_existing_authenticated_cohort_and_scheduler_admission"]


def test_zero_pool_cannot_justify_reranking_only_study():
    plan, policy, evidence = inputs()
    plan["family"] = "candidate_reranking"
    result = validate(plan, policy, evidence)
    assert result["status"] == "FAIL"
    assert any("no correct B pool" in item for item in result["errors"])


def test_training_memorization_does_not_admit_primary_study():
    plan, policy, evidence = inputs()
    plan["stage"] = "primary"
    plan["development_gate_results"] = {"tiny_training_exact_fraction": 1.}
    result = validate(plan, policy, evidence)
    assert result["status"] == "FAIL"
    assert any("heldout membership" in item for item in result["errors"])


def test_sealed_test_and_stale_evidence_fail_closed():
    plan, policy, evidence = inputs()
    changed = copy.deepcopy(plan)
    changed["sealed_test_access"] = True
    changed["diagnostic_evidence"]["sha256"] = "stale"
    result = validate(changed, policy, evidence)
    assert result["status"] == "FAIL"
    assert len(result["errors"]) == 2


def test_capacity_pretraining_plan_passes():
    _, policy, evidence = inputs()
    plan = json.loads((ROOT / "configs/reconstruction/capacity_pretraining_development_plan.json").read_text())
    assert validate(plan, policy, evidence)["status"] == "PASS"


@pytest.mark.parametrize("violation", ["missing_arm", "duplicate_arm", "hyperbolic_width", "history", "bottleneck", "endpoint"])
def test_confounded_or_incomplete_representation_plan_fails(violation):
    _, policy, evidence = inputs()
    plan = json.loads((ROOT / "configs/reconstruction/capacity_pretraining_development_plan.json").read_text())
    if violation == "missing_arm":
        plan["arms"].pop()
    elif violation == "duplicate_arm":
        plan["arms"][-1] = plan["arms"][0]
    elif violation == "hyperbolic_width":
        plan["arms"][0]["hyperbolic_dimension"] = 64
    elif violation == "history":
        plan["representation_controls"]["comparable_pretraining_histories"] = False
    elif violation == "bottleneck":
        del plan["representation_controls"]["head_bottleneck_control"]
    else:
        plan["evaluation_endpoints"].remove("correct_deep_proposal_survival")
    assert validate(plan, policy, evidence)["status"] == "FAIL"
