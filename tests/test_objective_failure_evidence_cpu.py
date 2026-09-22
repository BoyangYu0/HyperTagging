"""A rejected objective batch retains diagnostics without relaxing its guard."""
import json
import pytest
import torch
from hypertagging.training.pretrain_trainer import objective_preflight_report


def test_gradient_dominance_failure_is_saved_before_raising(tmp_path):
    output = tmp_path / 'objective-failure.json'
    with pytest.raises(RuntimeError, match='weighted_gradient_dominance:leaf_pid/lca=20.1546'):
        objective_preflight_report(
            {'lca':torch.tensor(1.),'leaf_pid':torch.tensor(1.)},
            {'lca':1.,'leaf_pid':.4},{'lca':10.,'leaf_pid':5.},
            {'gradient_norms':{'shared_encoder':{'lca':1.,'leaf_pid':50.3865}}},
            dominance_ratio=20.,action='fail',failure_output=output,completed_step=1642)
    record=json.loads(output.read_text())
    assert record['completed_optimizer_steps']==1642
    assert record['attempted_optimizer_step']==1643
    assert record['optimizer_step_executed'] is False
    assert record['report']['pass'] is False
    assert record['report']['weighted_dominance_ratio']==pytest.approx(20.1546)
    assert record['report']['dominance_threshold']==20.
    assert record['report']['objectives']['leaf_pid']['weighted_shared_encoder_gradient_norm']==pytest.approx(20.1546)


def test_success_does_not_create_failure_artifact(tmp_path):
    output=tmp_path/'failure.json'
    report=objective_preflight_report({'lca':torch.tensor(1.)},{'lca':1.},{'lca':2.},
        {'gradient_norms':{'shared_encoder':{'lca':1.}}},action='fail',failure_output=output,completed_step=0)
    assert report['pass'] and not output.exists()
