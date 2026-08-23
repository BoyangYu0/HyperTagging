from __future__ import annotations

from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from hypertagging.training.phase3_parallel_study import (
    AGGREGATION_VERSION,
    OWNER,
    RECEIPT_VERSION,
    STUDY_VERSION,
    assert_no_active_calibrations,
    calibration_slot,
    canonical_hash,
    claim_production_contract,
    entry_by_id,
    load_study_plan,
    tuple_key,
    validate_study_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json"
IDS = [
    "ht3-cal-h100nvl-b32-20260823",
    "ht3-cal-h100nvl-b64-20260823",
    "ht3-cal-v100-b32-20260823",
    "ht3-cal-v100-b64-20260823",
]


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _temporary_plan(tmp_path: Path) -> tuple[dict, Path]:
    plan = json.loads(PLAN_PATH.read_text())
    plan = copy.deepcopy(plan)
    plan["coordination"]["root"] = str(tmp_path / "coordination")
    plan["coordination"]["production_contract_registry"] = str(
        tmp_path / "coordination" / "production-contracts.json"
    )
    for entry in plan["calibration_matrix"]:
        job_root = tmp_path / entry["calibration_id"]
        entry["output_root"] = str(job_root / "output")
        entry["attempt_root"] = str(job_root / "attempt")
        entry["checkpoint_copy_path"] = str(job_root / "attempt" / "checkpoint.copy.pt")
        entry["metrics_path"] = str(job_root / "attempt" / "metrics.jsonl")
        entry["receipt_path"] = str(job_root / "attempt" / "receipt.json")
        body = dict(entry)
        body.pop("tuple_sha256", None)
        entry["tuple_sha256"] = canonical_hash(body)
    body = dict(plan)
    body.pop("plan_sha256", None)
    plan["plan_sha256"] = canonical_hash(body)
    path = tmp_path / "parallel-study-plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    validate_study_plan(plan, root=ROOT)
    return plan, path


def test_parallel_matrix_has_max_four_distinct_immutable_slots():
    plan = load_study_plan(PLAN_PATH, root=ROOT)
    assert plan["max_concurrent_calibration_jobs"] == 4
    assert plan["required_receipt_policy"]["mode"] == "exact_configured_set"
    assert len(plan["calibration_matrix"]) == 4
    assert len({tuple_key(entry) for entry in plan["calibration_matrix"]}) == 4
    assert len({entry["tuple_sha256"] for entry in plan["calibration_matrix"]}) == 4
    assert plan["production_policy"]["default_resume_count"] == 1


def test_active_registry_admits_four_and_rejects_a_fifth(tmp_path):
    plan, _ = _temporary_plan(tmp_path)
    fifth = copy.deepcopy(plan["calibration_matrix"][0])
    fifth["calibration_id"] = "ht3-cal-extra-20260823"
    plan["calibration_matrix"].append(fifth)
    with ExitStack() as stack:
        for calibration_id in IDS:
            stack.enter_context(calibration_slot(plan, calibration_id, owner=OWNER, root=ROOT))
        with pytest.raises(RuntimeError, match="maximum four"):
            with calibration_slot(plan, fifth["calibration_id"], owner=OWNER, root=ROOT):
                pass


@pytest.mark.parametrize("mutation, message", [("id", "duplicate or missing"), ("root", "share or nest")])
def test_parallel_plan_rejects_duplicate_identity_or_shared_root(tmp_path, mutation, message):
    plan, _ = _temporary_plan(tmp_path)
    if mutation == "id":
        plan["calibration_matrix"][1]["calibration_id"] = plan["calibration_matrix"][0]["calibration_id"]
    else:
        plan["calibration_matrix"][1]["output_root"] = plan["calibration_matrix"][0]["output_root"]
        body = dict(plan["calibration_matrix"][1])
        body.pop("tuple_sha256", None)
        plan["calibration_matrix"][1]["tuple_sha256"] = canonical_hash(body)
    with pytest.raises(RuntimeError, match=message):
        validate_study_plan(plan, root=ROOT)


def test_duplicate_production_contract_identity_is_rejected(tmp_path):
    plan, _ = _temporary_plan(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    claim_production_contract(plan, identity="same-contract", output_path=first, root=ROOT)
    with pytest.raises(RuntimeError, match="duplicate production contract identity"):
        claim_production_contract(plan, identity="same-contract", output_path=second, root=ROOT)


def test_production_action_is_blocked_while_calibration_is_active(tmp_path):
    plan, _ = _temporary_plan(tmp_path)
    with calibration_slot(plan, IDS[0], owner=OWNER, root=ROOT):
        with pytest.raises(RuntimeError, match="while calibration jobs are active"):
            assert_no_active_calibrations(plan, root=ROOT)
        with pytest.raises(RuntimeError, match="while calibration jobs are active"):
            claim_production_contract(
                plan, identity="blocked-contract", output_path=tmp_path / "blocked.json", root=ROOT
            )


def _write_healthy_receipts(plan: dict, tmp_path: Path) -> None:
    for index, calibration_id in enumerate(IDS, start=1):
        entry = entry_by_id(plan, calibration_id)
        metrics_path = Path(entry["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "split": "train",
                    "loss": 1.0,
                    "raw_gradient_norm": 0.5,
                    "learning_rate": 0.0005,
                    "events_per_second": float(10 + index),
                    "objective_preflight_pass": True,
                    "objective_weighted_dominance_ratio": 1.0,
                }
            )
            + "\n"
        )
        receipt = {
            "artifact_version": RECEIPT_VERSION,
            "calibration_id": calibration_id,
            "tuple_sha256": entry["tuple_sha256"],
            "owner": OWNER,
            "hypothesis_id": entry["hypothesis_id"],
            "hypothesis": entry["hypothesis"],
            "profile": {
                "name": entry["profile"],
                "exact_gres": entry["exact_gres"],
                "batch_size": entry["batch_size"],
                "precision_policy": entry["precision_policy"],
            },
            "checkpoint_copy": {
                "source_path": entry["source_checkpoint_path"],
                "copy_path": entry["checkpoint_copy_path"],
                "source_sha256": entry["source_checkpoint_sha256"],
                "copy_sha256": entry["source_checkpoint_sha256"],
                "source_unchanged": True,
                "loadability": {
                    "loadable": True,
                    "finite_tensors": True,
                    "tensor_count": 1,
                    "tensor_numel": 1,
                },
            },
            "output_root": entry["output_root"],
            "attempt_root": entry["attempt_root"],
            "queue_delay_seconds": float(index),
            "fixture_probe": {
                "fixture_batches_per_second": 1.0,
                "fixture_peak_memory_mib": float(index),
            },
            "pilot": {
                "executed": True,
                "role": "train",
                "validation_access": "forbidden",
                "sealed_test_access": "forbidden",
                "stress_access": "forbidden",
                "metrics_path": entry["metrics_path"],
                "record_count": 1,
                "max_steps": 256,
                "max_seconds": 900,
                "expected_learning_rate": 0.0005,
            },
            "scientific_contract": {
                "resume_presentations": 865024,
                "remaining_presentations": 865024,
                "objective_dominance_limit": 20.0,
                "submission_performed": False,
                "production_submission_performed": False,
            },
            "production_allowed": False,
            "production_submission_performed": False,
            "calibration_complete": True,
            "terminal_state": "healthy",
        }
        receipt["receipt_sha256"] = canonical_hash(receipt)
        Path(entry["receipt_path"]).write_text(json.dumps(receipt, sort_keys=True) + "\n")


def test_selection_uses_all_receipts_and_defaults_to_one_production_resume(tmp_path):
    plan, plan_path = _temporary_plan(tmp_path)
    _write_healthy_receipts(plan, tmp_path)
    aggregator = _load_module("phase3_aggregator", "aggregate_phase3_batch_efficiency_receipts.py")
    selector = _load_module("phase3_selector_parallel", "select_phase3_batch_efficiency_profile.py")
    aggregate_path = tmp_path / "receipt-aggregation.json"
    assert aggregator.main(["--study-plan", str(plan_path), "--output", str(aggregate_path)]) == 0
    selection_path = tmp_path / "selection.json"
    assert selector.main(
        [
            "--study-plan", str(plan_path),
            "--receipt-aggregation", str(aggregate_path),
            "--output", str(selection_path),
            "--authorize-production",
        ]
    ) == 0
    selection = json.loads(selection_path.read_text())
    assert selection["production_resume_policy"] == "default_exactly_one"
    assert selection["production_variant_count"] == 1
    assert selection["job_count"] == 1
    assert selection["submission_performed"] is False
    assert len(selection["calibration_ids"]) == 4
    renderer = _load_module("phase3_renderer_parallel", "render_phase3_batch_efficiency_production_contract.py")
    verifier = _load_module("phase3_verifier_parallel", "verify_phase3_batch_efficiency_contract.py")
    contract_path = tmp_path / "production-contract.json"
    assert renderer.main(
        [
            "--selection", str(selection_path),
            "--expected-git-sha", "a" * 40,
            "--output", str(contract_path),
        ]
    ) == 0
    verifier.verify(contract_path)
