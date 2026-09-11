"""The diagnostic recovery permits exactly one documented metadata deviation."""
import copy
import json
from pathlib import Path
import pytest
from scripts.recover_reconstruction_phase44 import reconciled_replay

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    config = json.loads((ROOT / 'configs/reconstruction/ht_reconstruction_phase44_20260911.json').read_text())['common_training_contract']
    observed = copy.deepcopy(config['balanced_level_replay_contract'])
    observed.pop('planned_schedule')
    observed['seed'] = config['seed']
    return config, observed


def test_documented_seed_only_recovery_preserves_budget_and_inputs():
    config, observed = inputs()
    original = copy.deepcopy(config)
    reconciled = reconciled_replay(config, observed)
    assert config == original
    assert reconciled['seed'] == 20260911
    assert reconciled['planned_schedule']['slot_count'] == 280064
    assert sum(reconciled['planned_schedule']['level_counts'].values()) == 280064


@pytest.mark.parametrize('change', ['observed_seed', 'unexpected_field', 'levels', 'budget', 'schedule', 'configured_seed'])
def test_recovery_rejects_any_other_deviation(change):
    config, observed = inputs()
    if change == 'observed_seed': observed['seed'] += 1
    if change == 'unexpected_field': observed['unrecognized'] = True
    if change == 'levels': observed['levels'] = [1, 2]
    if change == 'budget': config['replay_slot_budget'] -= 1
    if change == 'schedule': config['balanced_level_replay_contract']['planned_schedule']['slot_count'] -= 1
    if change == 'configured_seed': config['seed'] += 1
    with pytest.raises(ValueError):
        reconciled_replay(config, observed)


def test_recovery_closeout_requires_original_failed_receipt():
    import hashlib
    from scripts.build_reconstruction_phase44_closeout import verify_receipt
    receipt = {'status': 'failed', 'exit_status': 1, 'sealed_test_accessed': False}
    def seal(payload):
        return {**payload, 'receipt_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    verify_receipt(seal(receipt))
    with pytest.raises(ValueError):
        verify_receipt(seal({**receipt, 'status': 'completed', 'exit_status': 0}))
    bad = seal(receipt)
    bad['exit_status'] = 0
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_receipt(bad)


def test_successful_reconciliation_does_not_relabel_original_contract_as_matching():
    from scripts.recover_reconstruction_phase44 import label_reconciled_audit
    original = {'passed': True, 'preregistered_contract_match': True, 'slot_count': 280064}
    result = label_reconciled_audit(original)
    assert original['preregistered_contract_match'] is True
    assert result['preregistered_contract_match'] is False
    assert result['reconciled_contract_match'] is True
    assert result['passed'] and result['slot_count'] == 280064
    with pytest.raises(ValueError):
        label_reconciled_audit({**original, 'passed': False})


@pytest.mark.parametrize('change', [None, 'pointer', 'events', 'pretraining', 'freeze'])
def test_recovered_reports_keep_original_evaluation_settings(change):
    from scripts.build_reconstruction_phase44_closeout import verify_evaluation_settings
    contract = {'checkpoint_sha256': 'a' * 64, 'config': {'rollout_object_threshold': .6, 'rollout_pointer_threshold': .35, 'freeze_pretrained_encoder_steps': 2188}}
    report = {'configuration': {'object_threshold': .6, 'pointer_threshold': .35, 'max_level': 6, 'max_events': 100, 'target_policy': 'complete_only', 'forest_root_only_daughters': True},
              'checkpoint_pair': {'pretraining_sha256': 'a' * 64, 'pretraining_step': 81096, 'reconstruction_step': 4376, 'exact_frozen_encoder_required': False}}
    if change == 'pointer': report['configuration']['pointer_threshold'] = .5
    if change == 'events': report['configuration']['max_events'] = 20
    if change == 'pretraining': report['checkpoint_pair']['pretraining_sha256'] = 'b' * 64
    if change == 'freeze': report['checkpoint_pair']['exact_frozen_encoder_required'] = True
    if change is None:
        verify_evaluation_settings(report, contract, {'step': 4376}, 100)
    else:
        with pytest.raises(ValueError):
            verify_evaluation_settings(report, contract, {'step': 4376}, 100)
