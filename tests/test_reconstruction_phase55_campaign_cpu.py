"""Matched parent-ranking pretraining study isolation and untouched-cohort guards."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.run_reconstruction_phase55 import ARM_ROLES, validate_closeout_basis, validate_seed_contract, validation_exclusions
ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/'configs/reconstruction'/name).read_text())


def test_phase55_changes_only_pretraining_parent_ranking_weight():
    p = load('ht_reconstruction_phase55_20260920.json')
    assert tuple(a['role'] for a in p['arms']) == ARM_ROLES
    assert set(p['capacity_admission']['reports_by_arm']) == set(ARM_ROLES)
    assert all(report['production_training_allowed'] and report['query_overflow_count'] == report['cardinality_overflow_count'] == 0 for report in p['capacity_admission']['reports_by_arm'].values())
    a,b = [{**copy.deepcopy(p['common_training_contract']), **copy.deepcopy(a['overrides'])} for a in p['arms']]
    assert a['recovery_objective_weight'] == b['recovery_objective_weight'] == 2.0
    assert [arm['pretraining_parent_ranking_weight'] for arm in p['arms']] == [1.0, 2.0]
    assert p['pretraining_refinement']['config']['max_steps'] == 2188
    assert p['pretraining_refinement']['config']['batch_size'] == 32
    assert a['freeze_leaf_pid_head_steps'] == b['freeze_leaf_pid_head_steps'] == 4376
    assert a == b and a['encoder_lr_multiplier'] == 0.05
    assert a['freeze_pretrained_encoder_steps'] == 2188
    assert {a['checkpoint_step'] for a in p['arms']} == {81096}
    assert len({a['checkpoint_sha256'] for a in p['arms']}) == 1
    previous = load('ht_reconstruction_phase54_20260918.json')
    assert p['post_training_gates'] == previous['post_training_gates']
    assert p['evaluation_contract'] == previous['evaluation_contract']
    assert p['common_training_contract']['max_steps'] == 4376
    assert p['data_binding']['train_events'] == 70000
    assert p['data_binding']['dataset_index'] == previous['data_binding']['dataset_index']
    index = json.loads((ROOT/p['data_binding']['dataset_index']).read_text())
    assert index['normalizer_scope'] == 'train'
    assert index['event_identity_validation']['sealed_test_opened'] is False
    assert not p['authority']['promotion_authorized']


def test_phase55_cohort_and_evidence_are_bound():
    p = load('ht_reconstruction_phase55_20260920.json')
    c = load('ht_reconstruction_phase55_validation_cohort_20260920.json')
    excluded = set(validation_exclusions(c))
    selected,evaluated = set(c['checkpoint_selection_event_uids']),set(c['event_uids'])
    assert len(selected) == 2000 and len(evaluated) == 100
    assert not selected & evaluated and not (selected | evaluated) & excluded
    for phase in range(36,55):
        old = json.loads((ROOT/c['source_bindings'][f'phase{phase}_cohort']['path']).read_text())
        assert set(old['checkpoint_selection_event_uids']) | set(old['event_uids']) <= excluded
    audit = json.loads((ROOT/c["source_bindings"]["independent_audit"]["path"]).read_text())
    assert set(audit["event_uids"]) <= excluded
    assert c['all_required_overlaps_zero'] and not any(c['overlap_audit'].values())
    validate_seed_contract(p['common_training_contract'],c['seed'])
    validate_closeout_basis(p)
    for key in ('phase54_closeout_basis','phase54_retained_metric_basis','untouched_validation_cohort'):
        record = p[key]
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest() == record['sha256']
    boundary = p['scientific_source_boundary']
    assert boundary['fresh_train_only_normalization_required'] and boundary['resume_forbidden']


@pytest.mark.parametrize('change',['training','replay','cohort','missing','bool'])
def test_phase55_rejects_stale_seeds(change):
    config = {'seed':20260922,'balanced_level_replay_contract':{'seed':20260922}}
    cohort_seed = 20260922
    if change == 'training': config['seed'] = 20260918
    if change == 'replay': config['balanced_level_replay_contract']['seed'] = 20260918
    if change == 'cohort': cohort_seed = 20260918
    if change == 'missing': config.pop('balanced_level_replay_contract')
    if change == 'bool': config['seed'] = True
    with pytest.raises(RuntimeError,match='seeds must agree'):
        validate_seed_contract(config,cohort_seed)


def test_phase55_rejects_missing_gradient_execution_contract():
    p = load('ht_reconstruction_phase55_20260920.json')
    p.pop('gradient_execution_contract')
    with pytest.raises(RuntimeError, match='gradient-safe PID'):
        validate_closeout_basis(p)


def test_phase55_pretraining_contract_rejects_budget_and_factor_drift():
    from scripts.run_phase55_pretraining import validate_pretraining_contract
    p = load('ht_reconstruction_phase55_20260920.json')
    for arm in p['arms']:
        c = {'arm_role':arm['role'], 'pretraining_refinement': {**copy.deepcopy(p['pretraining_refinement']), 'parent_ranking_weight':arm['pretraining_parent_ranking_weight']}}
        validate_pretraining_contract(c,p)
        c['pretraining_refinement']['parent_ranking_weight'] = 4.0
        with pytest.raises(RuntimeError): validate_pretraining_contract(c,p)
    c = {'arm_role':p['arms'][0]['role'], 'pretraining_refinement': {**copy.deepcopy(p['pretraining_refinement']), 'parent_ranking_weight':1.0}}
    c['pretraining_refinement']['config']['max_steps'] = 4376
    with pytest.raises(RuntimeError): validate_pretraining_contract(c,p)


def test_phase55_pretraining_validation_never_uses_strict_cohort(tmp_path):
    from scripts.run_phase55_pretraining import refinement_config
    p = load('ht_reconstruction_phase55_20260920.json')
    cohort = load('ht_reconstruction_phase55_validation_cohort_20260920.json')
    c = {'pretraining_refinement': {**copy.deepcopy(p['pretraining_refinement']), 'parent_ranking_weight':1.0}}
    runtime = {'selection_manifest':'selection', 'dataset_index':'index', 'checkpoint':'checkpoint'}
    config = refinement_config(c,runtime,cohort,tmp_path)
    assert config.validation_event_uids == tuple(cohort['checkpoint_selection_event_uids'][:1000])
    assert not set(config.validation_event_uids) & set(cohort['event_uids'])
    assert config.weights_initialization_checkpoint == 'checkpoint' and config.resume is None


def test_phase55_evaluation_uses_authenticated_refinement(tmp_path):
    from scripts.run_reconstruction_phase55_full_decay import refined_evaluation_runtime
    checkpoint = tmp_path / 'refined.pt'
    checkpoint.write_bytes(b'refined-checkpoint')
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runtime = {'checkpoint': '/original.pt', 'checkpoint_sha256': 'original', 'checkpoint_step': '81096'}
    result = {'pretraining_refinement': {'status': 'COMPLETED', 'step': 2188,
        'checkpoint_selection': 'fixed_final_step_2188', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': checksum, 'source_checkpoint_sha256': 'original', 'source_unchanged': True}}
    actual = refined_evaluation_runtime(result, runtime)
    assert actual['checkpoint'] == str(checkpoint) and actual['checkpoint_step'] == '2188'
    assert runtime['checkpoint'] == '/original.pt'
    for field,value in [('checkpoint_sha256','wrong'),('source_checkpoint_sha256','other'),('step',81096),('source_unchanged',False)]:
        bad = copy.deepcopy(result)
        bad['pretraining_refinement'][field] = value
        with pytest.raises(RuntimeError, match='lineage'):
            refined_evaluation_runtime(bad,runtime)
