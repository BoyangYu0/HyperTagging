import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.training.pretraining_curriculum import (
    PretrainingStage,
    build_curriculum_batch,
    relation_aware_hard_negative_pairs,
)


def test_corruption_codes_are_actual_and_hard_negatives_exclude_close_relatives():
    batch = collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[1])])
    view = build_curriculum_batch(
        batch, PretrainingStage.CORRUPTED_COMPOSITES, corruption_probability=1.0
    )
    assert torch.equal(view.corrupted_node_mask, view.corruption_code > 0)
    pairs = relation_aware_hard_negative_pairs(batch)
    for event, left, right in pairs.tolist():
        assert batch["parent_ids"][event, left] != right
        assert batch["parent_ids"][event, right] != left
        assert batch["parent_ids"][event, left] != batch["parent_ids"][event, right]
