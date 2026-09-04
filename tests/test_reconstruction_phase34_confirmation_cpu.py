import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.evaluate_reconstruction_checkpoint_pair import (
    nested_equal,
    paired_bootstrap,
    select_confirmation_events,
)
from scripts.slurm import render_reconstruction_confirmation_job as render
from scripts.slurm import verify_reconstruction_confirmation_contract as verify
from hypertagging.training.checkpointing import load_training_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / render.PREREGISTRATION


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Event:
    event_uid: str


def test_confirmation_preregistration_binds_immutable_inputs() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    assert preregistration["study_classification"] == "validation_only_paired_confirmation"
    assert preregistration["analysis"]["automatic_promotion"] is False
    assert preregistration["analysis"]["training_performed"] is False
    assert preregistration["data_binding"]["sealed_test_role_access"] == "forbidden"
    for key in ("selection_manifest", "dataset_index"):
        path = ROOT / preregistration["data_binding"][key]
        assert digest(path) == preregistration["data_binding"][f"{key}_sha256"]
    for arm in preregistration["arms"]:
        assert digest(ROOT / arm["checkpoint"]) == arm["checkpoint_sha256"]
    prior = preregistration["prior_decision"]
    prior_payload = json.loads((ROOT / prior["path"]).read_text())
    assert digest(ROOT / prior["path"]) == prior["sha256"]
    assert prior_payload["decision"]["verdict"] == "NO_PROMOTION"
    assert prior_payload["decision"]["promotion_authorized"] is False
    assert prior_payload["decision"]["sealed_test_authorized"] is False


def test_bound_checkpoints_have_compatible_inference_contracts() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    payloads = [
        load_training_checkpoint(ROOT / arm["checkpoint"], map_location="cpu")
        for arm in preregistration["arms"]
    ]
    for key in ("architecture", "feature_contract", "normalizer_state"):
        assert nested_equal(payloads[0][key], payloads[1][key])
    for key in (
        "target_policy",
        "rollout_pid_kinematics_mode",
        "rollout_pid_temperature",
    ):
        assert payloads[0]["config"][key] == payloads[1]["config"][key]


def test_confirmation_cohorts_are_stable_disjoint_and_exclude_original() -> None:
    events = [Event(f"event-{index:05d}") for index in range(5000)]
    excluded = {f"event-{index:05d}" for index in range(250)}
    common = {
        "event_count": 1000,
        "seed": 20260904,
        "selection_manifest_hash": "a" * 64,
        "excluded_uids": excluded,
    }
    selected_a, uids_a, contract_a = select_confirmation_events(
        events, rank_offset=0, **common
    )
    selected_b, uids_b, contract_b = select_confirmation_events(
        events, rank_offset=1000, **common
    )
    repeated_a, repeated_uids_a, _ = select_confirmation_events(
        reversed(events), rank_offset=0, **common
    )
    assert len(selected_a) == len(selected_b) == 1000
    assert set(uids_a).isdisjoint(uids_b)
    assert set(uids_a).isdisjoint(excluded)
    assert set(uids_b).isdisjoint(excluded)
    assert uids_a == repeated_uids_a
    assert [event.event_uid for event in selected_a] == [event.event_uid for event in repeated_a]
    assert contract_a["rank_offset"] == 0
    assert contract_b["rank_offset"] == 1000


def test_paired_bootstrap_is_deterministic_and_applies_effect_gate() -> None:
    arguments = {
        "samples": 10000,
        "seed": 2026090401,
        "lower_index": 249,
        "upper_index": 9749,
        "minimum_effect_size": 0.01,
    }
    result = paired_bootstrap([0.02] * 1000, **arguments)
    assert result == paired_bootstrap([0.02] * 1000, **arguments)
    assert result["estimate"] == pytest.approx(0.02)
    assert result["confidence_interval"] == pytest.approx([0.02, 0.02])
    assert result["effect_size_gate_passed"] is True
    assert result["confidence_lower_bound_positive"] is True


def test_confirmation_workflow_is_fail_closed() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    assert verify.CONTRACT_VERSION.endswith("-v1")
    policy = preregistration["execution_policy"]
    assert policy == {
        "task_count": 2,
        "gres": "gpu:h100nvl:1",
        "cpus_per_task": 8,
        "memory": "32G",
        "time": "06:00:00",
        "requeue": False,
        "maximum_restarts": 0,
        "global_concurrency": 2,
    }
    wrapper = (ROOT / "scripts/slurm/run_reconstruction_confirmation.sbatch").read_text()
    assert "#SBATCH --no-requeue" in wrapper
    assert wrapper.index("trap finalize EXIT") < wrapper.index(
        "verify_reconstruction_confirmation_contract.py"
    )
    renderer = (ROOT / "scripts/slurm/render_reconstruction_confirmation_job.py").read_text()
    assert '"--no-requeue"' in renderer
    for denied in (
        '"training_authorized": False',
        '"checkpoint_mutation_authorized": False',
        '"sealed_test_access_authorized": False',
        '"promotion_authorized": False',
    ):
        assert denied in renderer
