import pytest
import torch

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    train_level_reconstruction,
)
from hypertagging.training.slurm_signal import (
    PendingValidationInterrupted,
    SLURM_REQUEUE_EXIT_CODE,
    SafeBoundarySignalController,
)


def _requested_controller():
    controller = SafeBoundarySignalController()
    controller._handle(10, None)
    return controller


def test_signal_during_restartable_validation_interrupts_for_exact_replay():
    controller = SafeBoundarySignalController()
    with pytest.raises(PendingValidationInterrupted):
        with controller.restartable_validation():
            controller._handle(10, None)
    assert controller.requested is True


def test_pretrain_signal_checkpoint_resumes_exactly(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet", row_group_size=1)
    base = dict(
        data=str(shard),
        max_steps=2,
        checkpoint_every=10,
        batch_size=1,
        max_events=2,
        pilot_split_repair=True,
        shuffle_buffer_size=2,
        channel_memory_size=4,
    )
    full = train_hyperbolic_pretraining(
        PretrainConfig(output_dir=str(tmp_path / "full"), **base)
    )
    with pytest.raises(SystemExit) as stopped:
        train_hyperbolic_pretraining(
            PretrainConfig(output_dir=str(tmp_path / "interrupted"), **base),
            signal_controller=_requested_controller(),
        )
    assert stopped.value.code == SLURM_REQUEUE_EXIT_CODE
    signal_path = tmp_path / "interrupted" / "signal-checkpoint.pt"
    signal_payload = load_training_checkpoint(signal_path)
    assert signal_payload["step"] == 1
    assert signal_payload["training_state"]["termination_reason"] == (
        "sigusr1_safe_optimizer_boundary"
    )
    assert signal_payload["optimizer_state_dict"]
    assert signal_payload["scheduler_state_dict"]
    assert signal_payload["streaming_cursor"]["batch_index"] == 1
    assert signal_payload["training_state"]["curriculum_phase_cursor"][
        "completed_optimizer_steps"
    ] == 1
    resumed = train_hyperbolic_pretraining(
        PretrainConfig(
            output_dir=str(tmp_path / "resumed"), resume=str(signal_path), **base
        )
    )
    full_payload = load_training_checkpoint(full.checkpoint)
    resumed_payload = load_training_checkpoint(resumed.checkpoint)
    for key, value in full_payload["model_state_dict"].items():
        torch.testing.assert_close(value, resumed_payload["model_state_dict"][key])
    assert full_payload["streaming_cursor"] == resumed_payload["streaming_cursor"]


def test_reconstruction_signal_checkpoint_contains_safe_boundary_state(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "reconstruction.parquet", row_group_size=1)
    with pytest.raises(SystemExit) as stopped:
        train_level_reconstruction(
            ReconstructionConfig(
                data=str(shard),
                output_dir=str(tmp_path / "reconstruction"),
                max_steps=2,
                batch_size=1,
                max_events=2,
                pilot_split_repair=True,
                validate_every=100,
                rollout_validate_every=100,
            ),
            signal_controller=_requested_controller(),
        )
    assert stopped.value.code == SLURM_REQUEUE_EXIT_CODE
    payload = load_training_checkpoint(
        tmp_path / "reconstruction" / "signal-checkpoint.pt"
    )
    assert payload["step"] == payload["schedule_state"]["step"] == 1
    assert payload["training_state"]["termination_reason"] == (
        "sigusr1_safe_optimizer_boundary"
    )
    assert payload["random_states"]


def test_pretrain_pending_scheduled_validation_is_serialized_and_replayed(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "pending-pretrain.parquet", row_group_size=1)
    base = dict(
        data=str(shard),
        max_steps=1,
        checkpoint_every=10,
        validate_every=1,
        validation_batches=1,
        batch_size=1,
        max_events=2,
        pilot_split_repair=True,
        shuffle_buffer_size=2,
        channel_memory_size=4,
    )
    full = train_hyperbolic_pretraining(
        PretrainConfig(output_dir=str(tmp_path / "pending-full"), **base)
    )
    with pytest.raises(SystemExit) as stopped:
        train_hyperbolic_pretraining(
            PretrainConfig(output_dir=str(tmp_path / "pending-stop"), **base),
            signal_controller=_requested_controller(),
        )
    assert stopped.value.code == SLURM_REQUEUE_EXIT_CODE
    signal_path = tmp_path / "pending-stop" / "signal-checkpoint.pt"
    interrupted = load_training_checkpoint(signal_path)
    assert interrupted["training_state"]["pending_validation_step"] == 1
    assert interrupted["training_state"]["last_validation_step"] == 0
    resumed = train_hyperbolic_pretraining(
        PretrainConfig(
            output_dir=str(tmp_path / "pending-resumed"), resume=str(signal_path), **base
        )
    )
    full_payload = load_training_checkpoint(full.checkpoint)
    resumed_payload = load_training_checkpoint(resumed.checkpoint)
    assert resumed_payload["training_state"]["pending_validation_step"] is None
    assert resumed_payload["training_state"]["last_validation_step"] == 1
    assert full_payload["metrics"] == resumed_payload["metrics"]


def test_reconstruction_pending_validation_is_serialized_and_replayed(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "pending-recon.parquet", row_group_size=1)
    base = dict(
        data=str(shard),
        max_steps=1,
        checkpoint_every=10,
        validate_every=1,
        rollout_validate_every=100,
        batch_size=1,
        validation_batch_size=1,
        max_events=2,
        pilot_split_repair=True,
    )
    with pytest.raises(SystemExit) as stopped:
        train_level_reconstruction(
            ReconstructionConfig(output_dir=str(tmp_path / "recon-stop"), **base),
            signal_controller=_requested_controller(),
        )
    assert stopped.value.code == SLURM_REQUEUE_EXIT_CODE
    signal_path = tmp_path / "recon-stop" / "signal-checkpoint.pt"
    interrupted = load_training_checkpoint(signal_path)
    assert interrupted["training_state"]["pending_validation_step"] == 1
    resumed = train_level_reconstruction(
        ReconstructionConfig(
            output_dir=str(tmp_path / "recon-resumed"), resume=str(signal_path), **base
        )
    )
    resumed_payload = load_training_checkpoint(resumed.checkpoint)
    assert resumed_payload["training_state"]["pending_validation_step"] is None
    assert resumed_payload["training_state"]["last_validation_step"] == 1
