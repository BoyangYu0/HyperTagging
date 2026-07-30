from hypertagging.evaluation.hierarchical_metrics import align_subtrees_by_source


def test_wrong_type_aligns_structurally_but_reduces_type_accuracy():
    truth = [
        {"type": 5, "sources": {1, 2}, "depth": 1, "daughter_count": 2},
        {"type": 7, "sources": {3, 4}, "depth": 1, "daughter_count": 2},
    ]
    predicted = [
        {"type": 9, "sources": {1, 2}, "depth": 1, "daughter_count": 2},
        {"type": 7, "sources": {3, 4}, "depth": 1, "daughter_count": 2},
    ]
    result = align_subtrees_by_source(predicted, truth)
    assert len(result.matches) == 2
    assert result.mother_type_accuracy == 0.5
