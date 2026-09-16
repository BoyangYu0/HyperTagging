"""Prospective corrected-source recovery-objective study isolation and cohort guards."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase52 import ARM_ROLES, validate_closeout_basis, validate_seed_contract, validation_exclusions
ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/'configs/reconstruction'/name).read_text())


def test_phase52_changes_only_recovery_objective_weight():
    p = load('ht_reconstruction_phase52_20260917.json')
    assert tuple(a['role'] for a in p['arms']) == ARM_ROLES
    assert set(p['capacity_admission']['reports_by_arm']) == set(ARM_ROLES)
    assert all(report['production_training_allowed'] and report['query_overflow_count'] == report['cardinality_overflow_count'] == 0 for report in p['capacity_admission']['reports_by_arm'].values())
    a,b = [{**copy.deepcopy(p['common_training_contract']), **copy.deepcopy(a['overrides'])} for a in p['arms']]
    assert a.pop('recovery_objective_weight') == 2.0
    assert b.pop('recovery_objective_weight') == 4.0
    assert a['freeze_leaf_pid_head_steps'] == b['freeze_leaf_pid_head_steps'] == 4376
    assert a == b and a['encoder_lr_multiplier'] == 0.05
    assert a['freeze_pretrained_encoder_steps'] == 2188
    assert {a['checkpoint_step'] for a in p['arms']} == {81096}
    assert len({a['checkpoint_sha256'] for a in p['arms']}) == 1
    previous = load('ht_reconstruction_phase51_20260916.json')
    assert p['post_training_gates'] == previous['post_training_gates']
    assert p['evaluation_contract'] == previous['evaluation_contract']
    assert p['common_training_contract']['max_steps'] == 4376
    assert p['data_binding']['train_events'] == 70000
    assert p['data_binding']['dataset_index'] == previous['data_binding']['dataset_index']
    index = json.loads((ROOT/p['data_binding']['dataset_index']).read_text())
    assert index['normalizer_scope'] == 'train'
    assert index['event_identity_validation']['sealed_test_opened'] is False
    assert not p['authority']['promotion_authorized']


def test_phase52_cohort_and_evidence_are_bound():
    p = load('ht_reconstruction_phase52_20260917.json')
    c = load('ht_reconstruction_phase52_validation_cohort_20260917.json')
    excluded = set(validation_exclusions(c))
    selected,evaluated = set(c['checkpoint_selection_event_uids']),set(c['event_uids'])
    assert len(selected) == 2000 and len(evaluated) == 100
    assert not selected & evaluated and not (selected | evaluated) & excluded
    for phase in range(36,52):
        old = json.loads((ROOT/c['source_bindings'][f'phase{phase}_cohort']['path']).read_text())
        assert set(old['checkpoint_selection_event_uids']) | set(old['event_uids']) <= excluded
    audit = json.loads((ROOT/c["source_bindings"]["independent_audit"]["path"]).read_text())
    assert set(audit["event_uids"]) <= excluded
    assert c['all_required_overlaps_zero'] and not any(c['overlap_audit'].values())
    validate_seed_contract(p['common_training_contract'],c['seed'])
    validate_closeout_basis(p)
    for key in ('phase51_closeout_basis','phase51_retained_metric_basis','untouched_validation_cohort'):
        record = p[key]
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest() == record['sha256']
    boundary = p['scientific_source_boundary']
    assert boundary['fresh_train_only_normalization_required'] and boundary['resume_forbidden']


@pytest.mark.parametrize('change',['training','replay','cohort','missing','bool'])
def test_phase52_rejects_stale_seeds(change):
    config = {'seed':20260919,'balanced_level_replay_contract':{'seed':20260919}}
    cohort_seed = 20260919
    if change == 'training': config['seed'] = 20260917
    if change == 'replay': config['balanced_level_replay_contract']['seed'] = 20260917
    if change == 'cohort': cohort_seed = 20260917
    if change == 'missing': config.pop('balanced_level_replay_contract')
    if change == 'bool': config['seed'] = True
    with pytest.raises(RuntimeError,match='seeds must agree'):
        validate_seed_contract(config,cohort_seed)


def test_phase52_rejects_missing_gradient_execution_contract():
    p = load('ht_reconstruction_phase52_20260917.json')
    p.pop('gradient_execution_contract')
    with pytest.raises(RuntimeError, match='gradient-safe PID'):
        validate_closeout_basis(p)
