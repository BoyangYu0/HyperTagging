from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.streaming import ParquetEventIterableDataset, StreamingCursor
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
import torch


def test_serialized_cursor_reproduces_single_worker_event_order(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet", row_group_size=1)
    dataset = ParquetEventIterableDataset([shard], seed=7)
    full = [event["event_uid"] for event in dataset]
    cursor = StreamingCursor(epoch=0, events_consumed=1)
    resumed = [
        event["event_uid"]
        for event in dataset.iter_from_cursor(StreamingCursor.from_state_dict(cursor.state_dict()))
    ]
    assert resumed == full[1:]


def test_real_streaming_trainer_resume_matches_uninterrupted(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "training.parquet", row_group_size=1)
    uninterrupted = train_hyperbolic_pretraining(
        PretrainConfig(
            data=str(shard),
            output_dir=str(tmp_path / "full"),
            max_steps=2,
            checkpoint_every=1,
            batch_size=1,
            max_events=2,
            pilot_split_repair=True,
            shuffle_buffer_size=2,
            channel_memory_size=4,
        )
    )
    resumed = train_hyperbolic_pretraining(
        PretrainConfig(
            data=str(shard),
            output_dir=str(tmp_path / "resumed"),
            max_steps=2,
            checkpoint_every=1,
            batch_size=1,
            max_events=2,
            pilot_split_repair=True,
            shuffle_buffer_size=2,
            channel_memory_size=4,
            resume=str(tmp_path / "full" / "checkpoint-step-1.pt"),
        )
    )
    full_state = load_training_checkpoint(uninterrupted.checkpoint)
    resumed_state = load_training_checkpoint(resumed.checkpoint)
    assert full_state["streaming_cursor"] == resumed_state["streaming_cursor"]
    for key, value in full_state["model_state_dict"].items():
        torch.testing.assert_close(value, resumed_state["model_state_dict"][key])
