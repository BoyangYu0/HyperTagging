import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.training.pretraining_curriculum import (
    PretrainingStage,
    build_curriculum_batch,
    curriculum_attention_mask,
)


def test_multilevel_attention_is_level_causal_and_stage_one_keeps_depth_targets():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    original_max = batch["full_event_max_level"].clone()
    stage1 = build_curriculum_batch(batch, PretrainingStage.FSP_ONLY)
    assert torch.equal(stage1.batch["full_event_max_level"], original_max)
    mask = curriculum_attention_mask(batch, PretrainingStage.TRUTH_GUIDED_MULTILEVEL)
    levels = batch["level_ids"][0]
    for query in range(levels.numel()):
        for key in range(levels.numel()):
            if levels[key] > levels[query] >= 0:
                assert not bool(mask[0, query, key])


def test_corruption_rebuilds_derived_inputs():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    corrupted = build_curriculum_batch(
        batch,
        PretrainingStage.CORRUPTED_COMPOSITES,
        corruption_probability=1.0,
    )
    changed = corrupted.corrupted_node_mask
    assert changed.any()
    adjacency = corrupted.batch["daughter_adjacency"].float()
    expected = torch.einsum("bmn,bnf->bmf", adjacency, corrupted.batch["p4"])
    torch.testing.assert_close(
        corrupted.batch["composite_features"][..., :4][changed],
        expected[changed],
    )
