import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    RolloutConfig,
    resolve_exclusive_proposals,
)


def test_confidence_is_supervised_and_no_object_target_is_zero():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    model = LevelAutoregressiveReconstructor(n_features=12, n_types=41, hidden_dim=16, n_queries=3)
    output = model(batch, target_level=1)
    loss = level_reconstruction_loss(output.pointer, batch, target_level=1)
    assert "confidence" in loss.components
    assert loss.confidence_targets is not None
    assert (loss.confidence_targets == 0).sum() >= 2
    loss.components["confidence"].backward()
    assert model.decoder.confidence_head.weight.grad is not None


def test_recursive_exclusivity_ranking_is_deterministic():
    proposals = [
        CompositeProposal(1, 4, (0, 1), 0.8, 0.7),
        CompositeProposal(0, 4, (1, 2), 0.8, 0.9),
    ]
    sources = torch.eye(3, dtype=torch.bool)
    first = resolve_exclusive_proposals(proposals, recursive_leaf_source_mask=sources)
    second = resolve_exclusive_proposals(list(reversed(proposals)), recursive_leaf_source_mask=sources)
    assert first == second and first[0].query_id == 0
    assert RolloutConfig().use_learned_confidence is False


def test_confidence_target_is_independent_of_type_probability_magnitude():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=41, hidden_dim=16, n_queries=3
    )
    output = model(batch, target_level=1).pointer
    first = level_reconstruction_loss(output, batch, target_level=1)
    adjusted = type(output)(
        object_logits=output.object_logits,
        type_logits=output.type_logits * 10,
        pointer_logits=output.pointer_logits,
        cardinality_logits=output.cardinality_logits,
        confidence_logits=output.confidence_logits,
        expected_type_embedding=output.expected_type_embedding,
    )
    second = level_reconstruction_loss(adjusted, batch, target_level=1)
    torch.testing.assert_close(first.confidence_targets, second.confidence_targets)
