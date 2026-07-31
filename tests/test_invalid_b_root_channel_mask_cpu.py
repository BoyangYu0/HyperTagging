import torch

from hypertagging.training.pretrain_trainer import valid_b_root_channel_mask


def test_invalid_and_fallback_b_roots_are_masked_from_current_batch_channels():
    batch = {
        "node_mask": torch.ones((3, 2), dtype=torch.bool),
        "b_root_discovery_valid": torch.tensor([True, False, True]),
        "b_root_discovery_fallback": torch.tensor([False, False, True]),
    }
    assert valid_b_root_channel_mask(batch).tolist() == [True, False, False]


def test_invalid_candidate_corruption_is_not_a_positive_channel_example():
    batch = {
        "node_mask": torch.ones((2, 2), dtype=torch.bool),
        "b_root_discovery_valid": torch.ones(2, dtype=torch.bool),
        "b_root_discovery_fallback": torch.zeros(2, dtype=torch.bool),
    }
    corrupted = torch.tensor([[False, True], [False, False]])
    assert valid_b_root_channel_mask(
        batch,
        corrupted_node_mask=corrupted,
        corruption_objective="invalid_candidate",
    ).tolist() == [False, True]

