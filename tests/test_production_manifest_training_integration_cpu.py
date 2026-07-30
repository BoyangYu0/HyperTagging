import json

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.training.data_module import build_real_data_module, resolve_data_paths
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    train_level_reconstruction,
)


def test_output_file_manifest_reaches_streaming_pretraining(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "tiny-v4.parquet")
    manifest = tmp_path / "production.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": 0,
                "output_file": str(shard),
                "schema_version": "direct-mdst-tree-v4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert resolve_data_paths(manifest) == [shard.resolve()]
    module = build_real_data_module(
        manifest,
        max_events=4,
        pilot_split_repair=True,
        allow_legacy_conflated=False,
    )
    batch = next(module.batches("train", batch_size=2, shuffle=False))
    assert batch["node_mask"].any()
    result = train_hyperbolic_pretraining(
        PretrainConfig(
            data=str(manifest),
            output_dir=str(tmp_path / "training"),
            max_steps=1,
            batch_size=2,
            pilot_split_repair=True,
        )
    )
    assert result.steps == 1
    reconstruction = train_level_reconstruction(
        ReconstructionConfig(
            data=str(manifest),
            output_dir=str(tmp_path / "reconstruction"),
            pretrained_encoder=str(result.checkpoint),
            transfer_leaf_pid_head=True,
            max_steps=1,
            batch_size=2,
            pilot_split_repair=True,
        )
    )
    assert reconstruction.steps == 1
    assert reconstruction.metrics["validation_events"] >= 1
