import pytest
import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.mother_pointer import MotherPointerDecoder


def test_level_conditioning_and_type_conditioned_pointer_gradients():
    torch.manual_seed(2)
    decoder = MotherPointerDecoder(hidden_dim=8, n_types=41, n_queries=3, max_cardinality=3)
    context = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    level1 = decoder(context, mask, target_level=1)
    level2 = decoder(context, mask, target_level=2)
    assert not torch.allclose(level1.pointer_logits, level2.pointer_logits)
    level1.pointer_logits.sum().backward()
    assert decoder.type_embedding.weight.grad is not None


def test_query_and_cardinality_overflow_raise():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[2])]
    )
    decoder = MotherPointerDecoder(hidden_dim=8, n_types=41, n_queries=1, max_cardinality=1)
    context = torch.randn(1, batch["node_mask"].shape[1], 8)
    output = decoder(context, batch["node_mask"], target_level=1)
    with pytest.raises((OverflowError, RuntimeError)):
        level_reconstruction_loss(
            output,
            batch,
            target_level=1,
            matching_production=False,
        )
