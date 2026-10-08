import copy
import json
from pathlib import Path
import pytest
from hypertagging.training.capacity_development import ARMS, ARCHITECTURE
from scripts.run_phase74_development import validate_contract, verify_bindings
from scripts.slurm.submit_phase74_development import acquire_submission_lock
from scripts.prepare_phase74_development_data import sha, ranked


def contract():
    return {
        "arms": list(ARMS),
        "architecture": ARCHITECTURE,
        "stage": "development",
        "sealed_test_access": False,
        "automatic_successor": False,
        "history": "fresh_both_widths_shared_initialization_within_width",
        "head_interface": "lossless_zero_pad_to_common256_predicted_group_conditioning_reset_after_pretrain",
        "resources": {
            "cpus": 2,
            "memory_gib": 32,
            "hours": 6,
            "gpus": 0,
            "max_jobs": 4,
            "requeue": False,
        },
        "training_count": 1536,
        "heldout_count": 600,
        "settings": {
            "pretraining_updates": 1000,
            "downstream_updates": 1500,
            "tiny_updates": 1000,
            "batch_size": 8,
            "seed": 202610081,
            "checkpoint": "fixed_final",
            "threshold": 0.5,
            "pretraining_lr": 0.0003,
            "head_lr": 0.001,
            "encoder_lr": 0.00005,
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("arms", ["128-existing"] * 4),
        ("stage", "primary"),
        ("history", "historical128_vs_fresh256"),
        ("head_interface", "bottleneck64"),
        ("sealed_test_access", True),
        ("automatic_successor", True),
        ("heldout_count", 60),
        ("training_count", 70000),
    ],
)
def test_invalid_history_arms_controls_and_cohorts_rejected(field, value):
    c = copy.deepcopy(contract())
    c[field] = value
    with pytest.raises(ValueError):
        validate_contract(c)


def test_resource_geometry_or_budget_changes_rejected():
    validate_contract(contract())
    for key, field, value in [
        ("architecture", "hyper_dim", 64),
        ("settings", "downstream_updates", 1501),
        ("resources", "max_jobs", 5),
    ]:
        c = copy.deepcopy(contract())
        c[key][field] = value
        with pytest.raises(ValueError):
            validate_contract(c)


def test_source_and_input_changes_fail_closed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    data = tmp_path / "data.json"
    data.write_text("{}")
    c = {
        "source_hashes": {"source.py": sha(source)},
        "bindings": [{"path": str(data), "sha256": sha(data)}],
    }
    verify_bindings(c, tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source"):
        verify_bindings(c, tmp_path)
    source.write_text("original")
    data.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="input"):
        verify_bindings(c, tmp_path)


def test_submission_lock_is_atomic_and_preserved(tmp_path):
    path = tmp_path / "submission-lock.json"
    acquire_submission_lock(path, "abc")
    with pytest.raises(FileExistsError):
        acquire_submission_lock(path, "abc")
    assert json.loads(path.read_text())["contract_sha256"] == "abc"


def test_identity_ranking_is_order_independent_and_role_specific():
    ids = [str(i) for i in range(50)]
    assert ranked(ids, "heldout", 1) == ranked(reversed(ids), "heldout", 1)
    assert ranked(ids, "heldout", 1) != ranked(ids, "train", 1)


def test_downstream_not_teacher_state_and_four_arm_runner_is_separate():
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_phase74_development.py"
    ).read_text()
    assert "fresh_decoder()" in source
    assert (
        "tiny_model=copy.deepcopy(model)" in source
        or "tiny_model = copy.deepcopy(model)" in source
    )
    assert "run_phase73" not in source
