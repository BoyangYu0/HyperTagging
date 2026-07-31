import copy

import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import TARGET_COMPOSITE_METADATA_INDICES


def test_target_only_composite_metadata_cannot_change_model_outputs():
    torch.manual_seed(91)
    batch = collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in batch.items()}
    for index in TARGET_COMPOSITE_METADATA_INDICES:
        changed["composite_features"][..., index] = torch.randn_like(changed["composite_features"][..., index]) * 1e6
        changed["composite_availability"][..., index] = True
    for name in (
        "full_truth_daughter_count", "retained_daughter_count",
        "reconstructed_daughter_count", "partial_missing_daughters",
        "recursive_reconstructable_complete", "valid_reconstruction_target",
    ):
        changed[name] = ~batch[name] if batch[name].dtype == torch.bool else batch[name] + 999
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=len(PDG_TOKENS), hidden_dim=16,
        hyper_dim=4, n_queries=3, n_context_layers=1,
    ).eval()
    with torch.no_grad():
        left = model(batch, target_level=1)
        right = model(changed, target_level=1)
    for a, b in (
        (left.node_embeddings, right.node_embeddings),
        (left.hyperbolic_embeddings, right.hyperbolic_embeddings),
        (left.relation_bias, right.relation_bias),
        (left.pointer.pointer_logits, right.pointer.pointer_logits),
        (left.pointer.type_logits, right.pointer.type_logits),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

