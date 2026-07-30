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
        )
    )
    checkpoint = load_training_checkpoint(pretrain.checkpoint)
    assert checkpoint["encoder_state_dict"]
    assert checkpoint["normalizer_state"]
    reconstruction = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "reconstruction"),
            pretrained_encoder=str(pretrain.checkpoint),
            max_steps=1,
            batch_size=2,
            allow_legacy_conflated=True,
        )
    )
    assert reconstruction.transfer_report
    assert reconstruction.transfer_report.loaded_keys
    assert reconstruction.metrics["levels_trained"] >= 1
    resumed = train_level_reconstruction(
        ReconstructionConfig(
            data=str(data),
            output_dir=str(tmp_path / "resumed"),
            pretrained_encoder=str(pretrain.checkpoint),
            resume=str(reconstruction.checkpoint),
            max_steps=2,
            batch_size=2,
            allow_legacy_conflated=True,
        )
    )
    assert resumed.steps == 2
