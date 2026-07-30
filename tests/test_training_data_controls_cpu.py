import pytest
import torch

from hypertagging.data.splitting import (
    MaskedFeatureNormalizer,
    SourceAwareSplitConfig,
    split_records,
    stable_split_name,
)
from hypertagging.training.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)


def test_source_file_grouping_prevents_split_leakage_and_is_stable():
    config = SourceAwareSplitConfig(group_by_source_file=True)
    first = {"event_uid": "1", "source_file": "same.root", "source_category": "charged"}
    second = {"event_uid": "2", "source_file": "same.root", "source_category": "charged"}
    assert stable_split_name(first, config) == stable_split_name(second, config)
    assert stable_split_name(first, config) == stable_split_name(first, config)


def test_duplicate_event_uid_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        split_records([{"event_uid": "x"}, {"event_uid": "x"}])


def test_normalization_uses_only_available_training_values():
    train_values = torch.tensor([[[1.0, 100.0], [3.0, -100.0]]])
    train_mask = torch.tensor([[[True, False], [True, False]]])
    normalizer = MaskedFeatureNormalizer().fit(train_values, train_mask)
    validation_values = torch.tensor([[[1000.0, float("nan")]]])
    validation_mask = torch.tensor([[[True, False]]])
    transformed = normalizer.transform(validation_values, validation_mask)

    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 0.0]))
    assert transformed[0, 0, 0] > 100
    assert transformed[0, 0, 1] == 0
    assert torch.isfinite(transformed).all()


def test_checkpoint_restores_model_optimizer_and_normalizer_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=17,
        epoch=3,
        normalizer_state={"common": {"mean": [1.0, 2.0]}},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    payload = restore_training_checkpoint(path, model=model, optimizer=optimizer)

    assert payload["step"] == 17
    assert payload["epoch"] == 3
    assert payload["normalizer_state"]["common"]["mean"] == [1.0, 2.0]
    for key, expected in original.items():
        torch.testing.assert_close(model.state_dict()[key], expected)
