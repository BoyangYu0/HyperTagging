import pytest
import torch

from hypertagging.training.checkpointing import restore_training_checkpoint, save_training_checkpoint


def test_checkpoint_rejects_architecture_mismatch_before_loading(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = save_training_checkpoint(
        tmp_path / "state.pt", model=model, architecture={"d_model": 2, "n_heads": 1}
    )
    with pytest.raises(ValueError, match="architecture"):
        restore_training_checkpoint(
            path, model=model, expected_architecture={"d_model": 4, "n_heads": 1}
        )

