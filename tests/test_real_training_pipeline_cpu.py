import pytest

import hypertagging.training.reconstruction_trainer as reconstruction_trainer
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.fixed_validation import excluded_event_uids_contract
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    train_level_reconstruction,
)


def test_reconstruction_threads_extended_rollout_and_final_validation_contracts(
    tmp_path, monkeypatch
):
    data = write_notebook_fixture_v3(tmp_path / "extended-tiny.parquet")
    rollout_configs = []
    validation_kwargs = []
    real_level_rollout = reconstruction_trainer.level_rollout
    real_validate = reconstruction_trainer.validate_reconstruction

    def recording_level_rollout(*args, **kwargs):
        rollout_configs.append(kwargs["config"])
        return real_level_rollout(*args, **kwargs)

    def recording_validate(*args, **kwargs):
        validation_kwargs.append(dict(kwargs))
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        reconstruction_trainer, "level_rollout", recording_level_rollout
    )
    monkeypatch.setattr(
        reconstruction_trainer, "validate_reconstruction", recording_validate
    )
    excluded_uids = ("fixture:validation:never-selected",)
    result = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "extended-reconstruction"),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
            validate_every=2,
            rollout_validate_every=2,
            checkpoint_every=2,
            max_validation_events=2,
            rollout_validation_events=0,
            scheduled_sampling_probability=1.0,
            scheduled_sampling_duration_steps=0,
            validation_excluded_event_uids=excluded_uids,
            rollout_continue_through_empty_levels=True,
        )
    )

    assert result.metrics["sampled_predicted_count"] > 0
    assert rollout_configs
    assert all(config.continue_through_empty_levels for config in rollout_configs)
    assert len(validation_kwargs) == 1
    assert validation_kwargs[0]["excluded_event_uids"] == excluded_uids
    assert validation_kwargs[0]["rollout_continue_through_empty_levels"] is True

    payload = load_training_checkpoint(result.checkpoint)
    selection = payload["training_state"]["checkpoint_selection_contract"]
    assert selection["version"] == "reconstruction-checkpoint-selection-v6"
    assert selection["rollout_configuration"]["policy_identity"][
        "empty_level_policy"
    ] == "continue_to_max_level"
    assert payload["data_order_contract"][
        "rollout_continue_through_empty_levels"
    ] is True
    exclusion_contract = excluded_event_uids_contract(excluded_uids)
    assert all(
        selection["validation_selection"][key] == value
        for key, value in exclusion_contract.items()
    )
    assert all(
        payload["validation_selection"][key] == value
        for key, value in exclusion_contract.items()
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
        "amp_dtype",
        "max_tangent_norm",
    ):
        assert name in checkpoint["config"]
    assert (tmp_path / "pretrain" / "best.pt").exists()
    assert (tmp_path / "pretrain" / "best_principal_topology.pt").exists()
    parent_checkpoint = tmp_path / "pretrain" / "best_parent_ranking.pt"
    if pretrain.metrics["validation_parent_ranking_accuracy_denominator"] > 0:
        assert parent_checkpoint.exists()
    else:
        assert not parent_checkpoint.exists()
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
    for name in (
        "raw_gradient_norm",
        "step_seconds",
        "data_wait_seconds",
        "events_per_second",
        "active_nodes_per_second",
        "process_peak_rss_bytes",
        "validation_seconds",
        "validation_event_views_per_second",
        "validation_model_forwards",
    ):
        assert pretrain.metrics[name] >= 0
    assert pretrain.metrics["amp_dtype"] == "float32"
    assert pretrain.metrics["validation_model_forwards"] == 6
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
    assert selection["thresholds"] == {
        "object_probability": 0.5,
        "daughter_pointer_probability": 0.5,
        "confidence": 0.0,
        "type_probability": None,
    }
    assert "track_fit_policies" in reconstruction_payload["feature_contract"]
    assert reconstruction_payload["feature_contract"]["pid_reconstruction_mode"]
    assert reconstruction_payload["feature_contract"]["exclusive_resolver"]["production"] == "greedy"
    assert reconstruction_payload["validation_selection"]["event_uids"]
    assert reconstruction_payload["validation_selection"]["rollout_was_run"]
    assert reconstruction_payload["validation_selection"]["rollout_event_uids"]
    assert reconstruction_payload["validation_selection"]["rollout_event_uids"] == (
        reconstruction_payload["validation_selection"]["event_uids"][
            : int(reconstruction.metrics["rollout_validation_events"])
        ]
    )
    exclusion_contract = excluded_event_uids_contract(())
    assert all(
        reconstruction_payload["validation_selection"][key] == value
        for key, value in exclusion_contract.items()
    )
    assert all(
        reconstruction_payload["data_order_contract"][key] == value
        for key, value in exclusion_contract.items()
    )
    assert reconstruction_payload["config"]["validation_excluded_event_uids"] == ()
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
