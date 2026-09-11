"""Prospective Phase45 isolation, cohort integrity, and pre-allocation seed guards."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase45 import ARM_ROLES, validate_closeout_basis, validate_seed_contract, validation_exclusions
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'configs/reconstruction' / name).read_text())


def test_phase45_isolates_encoder_learning_rate_with_fixed_update_count():
    p = load('ht_reconstruction_phase45_20260911.json')
    previous = load('ht_reconstruction_phase44_20260911.json')
    assert tuple(a['role'] for a in p['arms']) == ARM_ROLES
    assert [a['checkpoint_step'] for a in p['arms']] == [81096, 81096]
    a, b = [dict(arm['overrides']) for arm in p['arms']]
    assert [a.pop('encoder_lr_multiplier'), b.pop('encoder_lr_multiplier')] == [0.05, 0.10]
    assert a == b and a['freeze_pretrained_encoder_steps'] == 2188
    assert len({arm['checkpoint_sha256'] for arm in p['arms']}) == 1
    a_common, b_common = copy.deepcopy(p['common_training_contract']), copy.deepcopy(previous['common_training_contract'])
    assert a_common.pop('seed') == 20260912
    assert b_common.pop('seed') == 20260911
    assert a_common['balanced_level_replay_contract'].pop('seed') == 20260912
    assert b_common['balanced_level_replay_contract'].pop('seed') == 20260910
    assert a_common == b_common
    assert p['data_binding'] == previous['data_binding']
    assert p['post_training_gates'] == previous['post_training_gates']
    evaluation = copy.deepcopy(p['evaluation_contract'])
    retained_contract = evaluation.pop('retained_tree_checks')
    assert retained_contract['required'] and retained_contract['all_returned_beam_candidates']
    assert evaluation == previous['evaluation_contract']
    assert p['common_training_contract']['max_steps'] == 4376
    assert p['data_binding']['train_events'] == 70000
    assert p['authority']['sealed_test_access_authorized'] is False
    assert p['authority']['promotion_authorized'] is False


def test_phase45_cohort_excludes_every_previous_cohort():
    c = load('ht_reconstruction_phase45_validation_cohort_20260911.json')
    excluded = set(validation_exclusions(c))
    selected, evaluation = c['checkpoint_selection_event_uids'], c['event_uids']
    assert len(selected) == len(set(selected)) == 2000
    assert len(evaluation) == len(set(evaluation)) == 100
    assert not set(selected) & (set(evaluation) | excluded)
    assert not set(evaluation) & excluded
    assert c['evaluation_event_uids_sha256'] == uid_sequence_sha256(evaluation)
    assert c['checkpoint_selection_event_uids_sha256'] == uid_sequence_sha256(selected)
    for phase in range(36, 45):
        old = json.loads((ROOT / c['source_bindings'][f'phase{phase}_cohort']['path']).read_text())
        assert set(old['checkpoint_selection_event_uids']) | set(old['event_uids']) <= excluded
    assert c['all_required_overlaps_zero'] and not any(c['overlap_audit'].values())


def test_phase45_rejects_tampered_exclusion_identity():
    c = load('ht_reconstruction_phase45_validation_cohort_20260911.json')
    c['validation_exclusion_event_uid_count'] += 1
    with pytest.raises(RuntimeError, match='exclusion identity'):
        validation_exclusions(c)


def test_phase45_binds_recovered_evidence_and_consistent_seeds():
    p = load('ht_reconstruction_phase45_20260911.json')
    c = load('ht_reconstruction_phase45_validation_cohort_20260911.json')
    validate_seed_contract(p['common_training_contract'], c['seed'])
    assert validate_closeout_basis(p) == p['phase44_closeout_basis']
    for record in (p['phase44_closeout_basis'], p['phase44_retained_metric_basis'], p['untouched_validation_cohort']):
        assert hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() == record['sha256']
    assert p['evaluation_contract']['retained_tree_checks']['all_returned_beam_candidates'] is True
    assert p['evaluation_contract']['retained_tree_checks']['required'] is True
    assert all(report['production_training_allowed'] and report['query_overflow_count'] == report['cardinality_overflow_count'] == 0 for report in p['capacity_admission']['reports_by_arm'].values())


@pytest.mark.parametrize('change', ['training', 'replay', 'cohort', 'missing', 'bool'])
def test_phase45_rejects_seed_inconsistency_before_allocation(change):
    config = {'seed': 20260912, 'balanced_level_replay_contract': {'seed': 20260912}}
    cohort_seed = 20260912
    if change == 'training': config['seed'] = 20260911
    if change == 'replay': config['balanced_level_replay_contract']['seed'] = 20260911
    if change == 'cohort': cohort_seed = 20260911
    if change == 'missing': config.pop('balanced_level_replay_contract')
    if change == 'bool': config['seed'] = True
    with pytest.raises(RuntimeError, match='seeds must agree'):
        validate_seed_contract(config, cohort_seed)


@pytest.mark.parametrize('field,value', [('selected_next_factor', 'replicate_encoder_freeze_steps_2188_vs_0'), ('classification', 'completed_controlled_study_no_promotion'), ('sealed_test_accessed', True)])
def test_phase45_rejects_changed_recovery_authority(field, value):
    p = load('ht_reconstruction_phase45_20260911.json')
    p['phase44_closeout_basis'][field] = value
    with pytest.raises(RuntimeError, match='closeout basis'):
        validate_closeout_basis(p)
