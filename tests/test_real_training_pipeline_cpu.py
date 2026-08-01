import pytest

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    train_level_reconstruction,
)


def test_real_parquet_train_transfer_validate_and_resume(tmp_path):
    data = write_notebook_fixture_v3(tmp_path / "tiny.parquet")
    pretrain = train_hyperbolic_pretraining(
        PretrainConfig(
            data=str(data),
            output_dir=str(tmp_path / "pretrain"),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=1,
            validation_batches=1,
            log_every=1,
        )
    )
    checkpoint = load_training_checkpoint(pretrain.checkpoint)
    assert checkpoint["encoder_state_dict"]
    assert checkpoint["normalizer_state"]
    for name in (
        "lca_relation_weight",
        "parent_ranking_weight",
        "exact_tree_distance_weight",
        "radius_depth_weight",
        "channel_weight",
        "variance_weight",
        "covariance_weight",
        "leaf_pid_weight",
        "corruption_class_weight",
        "candidate_correctness_weight",
        "hard_negative_weight",
        "radius_target_mode",
        "channel_pooling",
    ):
        assert name in checkpoint["config"]
    assert (tmp_path / "pretrain" / "best.pt").exists()
    assert (tmp_path / "pretrain" / "best_principal_topology.pt").exists()
    assert (tmp_path / "pretrain" / "best_parent_ranking.pt").exists()
    assert (tmp_path / "pretrain" / "best_tree_distance.pt").exists()
    assert (tmp_path / "pretrain" / "latest.pt").exists()
    assert checkpoint["training_state"]["best_metric"] == (
        "validation_full_training_objective"
    )
    assert pretrain.metrics["validation_batches"] == 1
    assert "validation_loss_total" in pretrain.metrics
    assert "validation_principal_loss" in pretrain.metrics
    assert "validation_full_training_objective" in pretrain.metrics
    assert "validation_loss_corruption_class" in pretrain.metrics
    assert "validation_loss_candidate_correctness" in pretrain.metrics
    assert "validation_loss_hard_negative" in pretrain.metrics
    assert "validation_relation_accuracy" in pretrain.metrics
    assert "validation_parent_ranking_accuracy" in pretrain.metrics
    reconstruction = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "reconstruction"),
            pretrained_encoder=str(pretrain.checkpoint),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=1,
            rollout_validate_every=1,
            checkpoint_every=1,
        )
    )
    assert reconstruction.transfer_report
    assert reconstruction.transfer_report.loaded_keys
    assert reconstruction.metrics["levels_trained"] >= 1
    reconstruction_payload = load_training_checkpoint(reconstruction.checkpoint)
    assert reconstruction_payload["training_state"]["best_metric"] == "validation_loss_total"
    assert reconstruction_payload["training_state"]["last_validation_step"] == 1
    assert (tmp_path / "reconstruction" / "best.pt").exists()
    assert (tmp_path / "reconstruction" / "best_teacher_forced.pt").exists()
    assert (tmp_path / "reconstruction" / "best_rollout_edge_f1.pt").exists()
    assert (tmp_path / "reconstruction" / "best_rollout_tree_validity.pt").exists()
    assert (tmp_path / "reconstruction" / "latest.pt").exists()
    assert (tmp_path / "reconstruction" / "checkpoint-step-1.pt").exists()
    selection = reconstruction_payload["training_state"][
        "checkpoint_selection_contract"
    ]
    assert selection["primary_metric"] == "validation_loss_total"
    assert "track_fit_policies" in reconstruction_payload["feature_contract"]
    assert reconstruction_payload["feature_contract"]["pid_reconstruction_mode"]
    assert reconstruction_payload["feature_contract"]["exclusive_resolver"]["production"] == "greedy"
    assert reconstruction_payload["validation_selection"]["event_uids"]
    resumed_without_new_steps = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "resumed_without_new_steps"),
            pretrained_encoder=str(pretrain.checkpoint),
            resume=str(reconstruction.checkpoint),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=1,
            rollout_validate_every=1,
            checkpoint_every=1,
        )
    )
    resumed_payload = load_training_checkpoint(resumed_without_new_steps.checkpoint)
    assert resumed_payload["training_state"] == reconstruction_payload["training_state"]
    assert resumed_payload["validation_selection"] == reconstruction_payload["validation_selection"]
    resumed = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "resumed"),
            pretrained_encoder=str(pretrain.checkpoint),
            resume=str(reconstruction.checkpoint),
            max_steps=2,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=1,
            rollout_validate_every=1,
            checkpoint_every=1,
        )
    )
    assert resumed.steps == 2
    with pytest.raises(ValueError, match="checkpoint-selection semantics differ"):
        train_level_reconstruction(
            ReconstructionConfig(
                data=str(data),
                output_dir=str(tmp_path / "resume_changed_selection"),
                pretrained_encoder=str(pretrain.checkpoint),
                resume=str(reconstruction.checkpoint),
                max_steps=1,
                batch_size=2,
                allow_legacy_conflated=True,
                validate_every=1,
                rollout_validate_every=2,
                checkpoint_every=1,
            )
        )
