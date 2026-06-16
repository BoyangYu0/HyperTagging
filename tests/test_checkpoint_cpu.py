import torch

from hypertagging.utils.checkpoint import (
    get_epoch,
    get_model_state_dict,
    load_checkpoint,
    load_model_state,
    save_checkpoint,
)
from hypertagging.utils.device import cpu_device, resolve_device
from hypertagging.utils.seeds import seed_everything


def test_load_checkpoint_defaults_to_cpu(tmp_path):
    path = tmp_path / "tiny_checkpoint.pt"
    torch.save(
        {"epoch": 3, "model_state_dict": {"weight": torch.ones(2)}},
        path,
    )

    checkpoint = load_checkpoint(path)

    assert get_epoch(checkpoint) == 3
    state_dict = get_model_state_dict(checkpoint)
    assert state_dict["weight"].device.type == "cpu"
    torch.testing.assert_close(state_dict["weight"], torch.ones(2))


def test_load_model_state_preserves_historical_model_state_dict_key(tmp_path):
    seed_everything(123)
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "linear_checkpoint.pt"
    save_checkpoint(path, source.state_dict(), epoch=7)

    target = torch.nn.Linear(2, 1)
    checkpoint, result = load_model_state(target, path, map_location="cpu")

    assert get_epoch(checkpoint) == 7
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_resolve_device_defaults_to_cpu():
    assert cpu_device().type == "cpu"
    assert resolve_device().type == "cpu"
    assert resolve_device("cpu").type == "cpu"
