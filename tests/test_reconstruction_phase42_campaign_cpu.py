"""Ensure the next experiment isolates pretraining and preserves scientific gates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase42 import ARM_ROLES, validation_exclusions
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'configs/reconstruction' / name).read_text())


def test_phase42_changes_only_pretraining_at_fixed_compute_and_data():
    p = load('ht_reconstruction_phase42_20260909.json')
    assert tuple(a['role'] for a in p['arms']) == ARM_ROLES
    assert [a['checkpoint_step'] for a in p['arms']] == [81096, 108128]
    assert p['arms'][0]['overrides'] == p['arms'][1]['overrides']
    previous = load('ht_reconstruction_phase41_20260909.json')
    assert p['common_training_contract'] == previous['common_training_contract']
    assert p['data_binding'] == previous['data_binding']
    assert p['post_training_gates'] == previous['post_training_gates']
    assert p['common_training_contract']['max_steps'] == 4376
    assert p['data_binding']['train_events'] == 70000
    assert p['evaluation_contract']['beam_search']['evaluated_scopes'] == ['full', 'half']
    assert p['authority']['sealed_test_access_authorized'] is False
    assert p['authority']['promotion_authorized'] is False


def test_phase42_cohort_excludes_all_previous_tuning_and_evaluation():
    c = load('ht_reconstruction_phase42_validation_cohort_20260909.json')
    excluded = set(validation_exclusions(c))
    selected, evaluation = c['checkpoint_selection_event_uids'], c['event_uids']
    assert len(selected) == len(set(selected)) == 2000
    assert len(evaluation) == len(set(evaluation)) == 100
    assert not set(selected) & (set(evaluation) | excluded)
    assert not set(evaluation) & excluded
    assert c['evaluation_event_uids_sha256'] == uid_sequence_sha256(evaluation)
    assert c['checkpoint_selection_event_uids_sha256'] == uid_sequence_sha256(selected)
    old = load('ht_reconstruction_phase41_validation_cohort_20260909.json')
    assert set(old['checkpoint_selection_event_uids']) | set(old['event_uids']) <= excluded


def test_phase42_rejects_tampered_exclusion_identity():
    c = load('ht_reconstruction_phase42_validation_cohort_20260909.json')
    c['validation_exclusion_event_uid_count'] += 1
    with pytest.raises(RuntimeError, match='exclusion identity'):
        validation_exclusions(c)


def test_phase42_preregistration_binds_review_and_cohort():
    p = load('ht_reconstruction_phase42_20260909.json')
    for record in (p['phase41_closeout_basis'], p['untouched_validation_cohort']):
        assert hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() == record['sha256']
    assert all(report['production_training_allowed'] for report in p['capacity_admission']['reports_by_arm'].values())
    assert all(report['query_overflow_count'] == report['cardinality_overflow_count'] == 0 for report in p['capacity_admission']['reports_by_arm'].values())
