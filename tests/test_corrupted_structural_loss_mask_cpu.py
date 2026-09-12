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


def test_invalid_candidate_corruption_excludes_affected_ancestors():
    batch = collate_heterogeneous_events([
        heterogeneous_from_level_event(tiny_level_events()[1])
    ])
    curriculum = build_curriculum_batch(
        batch, PretrainingStage.CORRUPTED_COMPOSITES, seed=0,
        corruption_probability=0.5, corruption_objective="invalid_candidate",
    )
    # Node 4 drops one daughter. Node 5 retains its direct adjacency, but
    # includes node 4, so its reconstructed subtree is no longer the target.
    assert curriculum.corrupted_node_mask[0, 4]
    assert not curriculum.corrupted_node_mask[0, 5]
    assert not (
        curriculum.batch["recursive_leaf_source_mask"][0, 5]
        == batch["recursive_leaf_source_mask"][0, 5]
    ).all()
    assert not curriculum.structural_positive_mask[0, 5]
    assert curriculum.invalid_candidate_mask[0, 5]
    # Corruption-class labels still describe only the local edit operation.
    assert curriculum.corruption_code[0, 5] == 0
    for event, left, right in curriculum.hard_negative_pairs.tolist():
        assert not curriculum.invalid_candidate_mask[event, left]
        assert not curriculum.invalid_candidate_mask[event, right]

    denoising = build_curriculum_batch(
        batch, PretrainingStage.CORRUPTED_COMPOSITES, seed=0,
        corruption_probability=0.5, corruption_objective="denoising",
    )
    assert denoising.structural_positive_mask[0, 5]
