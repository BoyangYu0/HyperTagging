import pytest
import torch

from hypertagging.training.checkpointing import restore_training_checkpoint, save_training_checkpoint


def _checkpoint(tmp_path, contract):
    model = torch.nn.Linear(2, 2)
    path = save_training_checkpoint(tmp_path / "state.pt", model=model, data_order_contract=contract)
    return path, model


def test_resume_accepts_identical_data_order_contract(tmp_path):
    contract = {"batch_size": 2, "shuffle_buffer_size": 8, "dataset_index_hash": "abc"}
    path, model = _checkpoint(tmp_path, contract)
    restore_training_checkpoint(path, model=model, expected_data_order_contract=contract)


@pytest.mark.parametrize("key,value", [
    ("batch_size", 3), ("shuffle_buffer_size", 9), ("dataset_index_hash", "changed")
])
def test_resume_rejects_changed_data_order_setting(tmp_path, key, value):
    contract = {"batch_size": 2, "shuffle_buffer_size": 8, "dataset_index_hash": "abc"}
    path, model = _checkpoint(tmp_path, contract)
    changed = {**contract, key: value}
    with pytest.raises(ValueError, match="data-order"):
        restore_training_checkpoint(path, model=model, expected_data_order_contract=changed)

