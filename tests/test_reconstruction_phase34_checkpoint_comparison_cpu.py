import json
from pathlib import Path

from scripts.slurm import render_reconstruction_fullscale_job as render
from scripts.slurm import verify_reconstruction_fullscale_contract as verify


ROOT = Path(__file__).resolve().parents[1]


def test_phase34_comparison_registry_matches_preregistration() -> None:
    preregistration = json.loads(
        (ROOT / render.CHECKPOINT_COMPARISON_PREREGISTRATION).read_text()
    )
    registered = {
        int(step): {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "role": entry["role"],
        }
        for step, entry in verify.CHECKPOINT_COMPARISON_SOURCES.items()
    }
    declared = {
        int(arm["step"]): {
            "path": arm["path"],
            "sha256": arm["sha256"],
            "role": arm["role"],
        }
        for arm in preregistration["checkpoint_arms"]
    }
    assert registered == declared
    assert {
        step: {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "role": entry["role"],
        }
        for step, entry in render.CHECKPOINT_COMPARISON_SOURCES.items()
    } == registered
    assert set(registered) == {54064, 81096, 108128}
    assert preregistration["study_id"] == verify.CHECKPOINT_COMPARISON_STUDY
    assert (
        preregistration["data_binding"]["selection_manifest"]
        == verify.CHECKPOINT_COMPARISON_SELECTION
    )
    assert (
        preregistration["data_binding"]["dataset_index"]
        == verify.CHECKPOINT_COMPARISON_INDEX
    )


def test_phase34_comparison_uses_identical_fullscale_configuration() -> None:
    config = render.build_config("production", "gpu:h100nvl:1", 64)
    preregistration = json.loads(
        (ROOT / render.CHECKPOINT_COMPARISON_PREREGISTRATION).read_text()
    )
    fixed = preregistration["fixed_training_contract"]
    for field in (
        "presentations_target",
        "batch_size",
        "max_steps",
        "seed",
        "learning_rate",
        "object_positive_weight",
        "pointer_positive_weight",
        "model_preset",
        "target_policy",
        "initial_state_policy",
        "scheduled_sampling_probability",
        "scheduled_sampling_schedule",
        "scheduled_sampling_duration_steps",
    ):
        assert config[field] == fixed[field]
    assert config["freeze_pretrained_encoder_steps"] == config["max_steps"]
    assert config["freeze_leaf_pid_head_steps"] == config["max_steps"]
    assert config["validation_enabled"] is True
