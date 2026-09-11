import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/build_phase44_retained_metric_export.py'
spec = importlib.util.spec_from_file_location('phase44_retained_export', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_retained_labels_are_lossless_public_metric_names():
    source = {'summaries': {'full/proposal_beam/normalized_joint_log_probability': {'correct': 3, 'total': 7, 'unavailable': None}}}
    rows = list(module.numeric_rows(module.metric_names(source)))
    assert {row['metric']: row['value'] for row in rows} == {
        'summaries.full.proposal_beam.normalized_joint_log_probability.correct': 3,
        'summaries.full.proposal_beam.normalized_joint_log_probability.total': 7,
        'summaries.full.proposal_beam.normalized_joint_log_probability.unavailable': None,
    }
    with pytest.raises(ValueError, match='collide'):
        module.metric_names({'full/greedy': 1, 'full.greedy': 2})


def test_legacy_comparison_removes_only_additive_retained_fields():
    assert module.legacy_projection({'metrics': {'value': 0.5}, 'retained_tree_metrics': {'value': 1}, 'beam': [{'retained_tree_metrics_by_scope': {}, 'score': 2}]}) == {'metrics': {'value': 0.5}, 'beam': [{'score': 2}]}
