import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.training.pretraining_curriculum import PretrainingStage, build_curriculum_batch


def test_all_curriculum_stages_are_finite_and_stage1_has_no_truth_composites():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )
    model = HeterogeneousNodeEncoder(d_model=16, hyper_dim=4)
    for stage in PretrainingStage:
        view = build_curriculum_batch(batch, stage, seed=3, corruption_probability=1.0)
        if stage is PretrainingStage.FSP_ONLY:
            assert torch.all(view.batch["level_ids"][view.batch["node_mask"]] == 0)
        if stage is PretrainingStage.CORRUPTED_COMPOSITES:
            assert view.corrupted_node_mask.any()
            assert view.hard_negative_pairs.shape[-1] == 3
        loss = model(view.batch).hyperbolic_embeddings.square().mean()
        loss.backward()
        assert torch.isfinite(loss)
