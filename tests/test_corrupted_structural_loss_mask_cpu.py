from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.training.pretraining_curriculum import PretrainingStage, build_curriculum_batch


def test_invalid_candidate_corruption_is_excluded_from_positive_structure():
    batch = collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    curriculum = build_curriculum_batch(
        batch, PretrainingStage.CORRUPTED_COMPOSITES,
        corruption_probability=1.0, corruption_objective="invalid_candidate",
    )
    assert not curriculum.structural_positive_mask[curriculum.corrupted_node_mask].any()
    denoising = build_curriculum_batch(
        batch, PretrainingStage.CORRUPTED_COMPOSITES,
        corruption_probability=1.0, corruption_objective="denoising",
    )
    assert denoising.structural_positive_mask[denoising.corrupted_node_mask].all()

