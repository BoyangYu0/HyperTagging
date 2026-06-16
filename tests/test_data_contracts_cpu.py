import numpy as np
import pytest
import torch

from hypertagging.data.contracts import CONTRACTS, get_contract, validate_batch
from hypertagging.data.fixtures import all_tiny_batches, tiny_grafei_pair_batch


def test_all_tiny_fixtures_satisfy_registered_contracts():
    batches = all_tiny_batches()

    assert set(batches) == set(CONTRACTS)
    for name, batch in batches.items():
        validate_batch(name, batch)


def test_contract_validation_accepts_torch_tensors_on_cpu():
    batch = {
        key: torch.from_numpy(value)
        for key, value in tiny_grafei_pair_batch().items()
    }

    validate_batch("grafei_pairs", batch)
    for value in batch.values():
        assert value.device.type == "cpu"


def test_contract_validation_rejects_missing_required_key():
    batch = tiny_grafei_pair_batch()
    batch.pop("links")

    with pytest.raises(KeyError, match="links"):
        validate_batch("grafei_pairs", batch)


def test_contract_validation_rejects_wrong_dtype_family():
    batch = tiny_grafei_pair_batch()
    batch["pdg_x"] = batch["pdg_x"].astype(np.float32)

    with pytest.raises(TypeError, match="pdg_x"):
        validate_batch("grafei_pairs", batch)


def test_contract_validation_rejects_symbol_shape_mismatch():
    batch = tiny_grafei_pair_batch()
    batch["feature_y"] = batch["feature_y"][:, :2, :]

    with pytest.raises(ValueError, match="symbol 'P'"):
        validate_batch(get_contract("grafei_pairs"), batch)
