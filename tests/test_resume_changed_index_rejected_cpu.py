import pytest
import torch

from hypertagging.training.checkpointing import restore_training_checkpoint, save_training_checkpoint


def test_changed_dataset_index_hash_rejected(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = save_training_checkpoint(
        tmp_path / "state.pt", model=model,
        data_order_contract={"batch_size": 2, "shuffle_buffer_size": 8, "dataset_index_hash": "a"},
    )
    with pytest.raises(ValueError, match="data-order"):
        restore_training_checkpoint(
            path, model=model,
            expected_data_order_contract={"batch_size": 2, "shuffle_buffer_size": 8, "dataset_index_hash": "b"},
        )

