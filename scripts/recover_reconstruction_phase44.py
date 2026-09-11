#!/usr/bin/env python3
"""Audit Phase44's stale replay seed and evaluate preserved checkpoints diagnostically."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def reconciled_replay(config, observed):
    """Permit only the documented stale seed; derive and audit the slot schedule."""
    expected = copy.deepcopy(config['balanced_level_replay_contract'])
    if config['seed'] != 20260911 or expected['seed'] != 20260910:
        raise ValueError('Not the documented Phase44 seed deviation')
    expected['seed'] = config['seed']
    schedule = expected.pop('planned_schedule')
    if observed != expected:
        raise ValueError('Replay differs beyond the documented seed deviation')
    levels = observed['levels']
    slots = config['batch_size'] * config['max_steps']
    computed = dict(start_slot=0, slot_count=slots, end_slot_exclusive=slots,
                    level_counts={str(k):slots//len(levels)+int(i<slots%len(levels)) for i,k in enumerate(levels)},
                    max_minus_min_level_count=int(bool(slots%len(levels))))
    if computed != schedule or slots != config['replay_slot_budget']:
        raise ValueError('Replay budget or planned schedule changed')
    return {**observed, 'planned_schedule':computed}


def label_reconciled_audit(audit):
    """Distinguish successful reconciliation from the failed original contract."""
    if not audit['passed'] or not audit['preregistered_contract_match']:
        raise ValueError('The reconciled audit itself did not pass')
    return {**audit, 'preregistered_contract_match': False,
            'reconciled_contract_match': True,
            'audit_basis': 'documented_seed_reconciliation_only'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--arm',choices=['late_adaptation_control','early_adaptation'],required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_root.resolve(strict=True)
    out=args.output_dir.resolve()
    if out.exists(): raise ValueError('Refusing existing recovery output')
    sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source))
    import torch
    from scripts.run_reconstruction_phase44 import verify_contract,validation_exclusions,CHECKPOINT_TRACKS
    from scripts.run_reconstruction_phase35 import finite_checkpoint,replay_slot_audit,_validation_selection_audit,first_twenty_gate,sha256,atomic_json
    from scripts import run_reconstruction_phase44_full_decay as evaluation
    contract_path=source/f'artifacts/slurm/reconstruction-phase44/contracts/phase44_{args.arm}_contract_20260911.json'
    contract,runtime=verify_contract(contract_path)
    run=source/f'artifacts/runs/ht-reconstruction-phase44-20260911/{args.arm}/{args.job}'
    receipt_path=source/f'artifacts/slurm/reconstruction-phase44/jobs/{args.job}/attempt-00/receipt.json'
    receipt=json.loads(receipt_path.read_text())
    canonical={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    assert hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==receipt['receipt_sha256']
    assert receipt['status']=='failed' and receipt['exit_status']==1 and not receipt['sealed_test_accessed']
    assert receipt['contract_sha256']==contract['contract_sha256'] and receipt['arm_role']==args.arm
    for binding in receipt['artifacts'].values():
        if binding: assert sha256(source/binding['path'])==binding['sha256']
    err=source/f'artifacts/slurm/phase44-{args.arm}-20260911-{args.job}.err'
    assert 'trainer balanced replay contract does not exactly match the preregistered real-data contract' in err.read_text()
    training=run/'training'
    checkpoints={k:finite_checkpoint(training/f) for k,f in CHECKPOINT_TRACKS.items()}
    assert checkpoints['final']['step']==4376
    final=torch.load(training/'checkpoint.pt',map_location='cpu',weights_only=False)
    config=contract['config']
    observed=final['data_order_contract']['balanced_level_replay']
    reconstructed=reconciled_replay(config,observed)
    corrected=copy.deepcopy(config);corrected['balanced_level_replay_contract']=reconstructed
    replay=replay_slot_audit(training/'metrics.jsonl',config=corrected,trainer_contract=reconstructed)
    replay=label_reconciled_audit(replay)
    cohort=json.loads(Path(runtime['cohort']).read_text())
    selection=_validation_selection_audit(training/'checkpoint.pt',exclusions=validation_exclusions(cohort),
        expected_event_uids=tuple(cohort['checkpoint_selection_event_uids']),
        expected_rollout_event_uids=tuple(cohort['checkpoint_selection_event_uids'][:config['rollout_validation_events']]))
    for name,file in CHECKPOINT_TRACKS.items():
        payload=torch.load(training/file,map_location='cpu',weights_only=False)
        assert payload['git_commit']==contract['expected_git_sha']
        assert payload['config']['seed']==config['seed']
        assert payload['config']['freeze_pretrained_encoder_steps']==config['freeze_pretrained_encoder_steps']
        assert payload['data_order_contract']['balanced_level_replay']==observed
        assert payload['step'] in {1000,2000,3000,4000,4376}
    assert sha256(Path(runtime['checkpoint']))==contract['checkpoint_sha256']
    out.mkdir(parents=True)
    result=dict(status='training_completed',study_id=contract['study_id'],arm_role=args.arm,contract_sha256=contract['contract_sha256'],
        optimizer_steps=4376,elapsed_seconds=None,checkpoints=checkpoints,balanced_level_replay=replay,validation_selection=selection,
        first_20_optimizer_steps=first_twenty_gate(training/'metrics.jsonl'),
        recovery_classification='POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION',original_job_status='FAILED',
        deviation=dict(field='balanced_level_replay_contract.seed',preregistered=20260910,configured_and_observed=20260911,
                       planned_schedule_reconstructed_from_budget=True),
        authority='User requested Phase44 review and complete metrics after both jobs failed; diagnostic evaluation only.',
        source_receipt_sha256=sha256(receipt_path),source_stderr_sha256=sha256(err),source_training_log_sha256=sha256(training/'metrics.jsonl'),
        sealed_test_accessed=False,promotion_authorized=False)
    atomic_json(out/'training-result.json',result)
    print(json.dumps({'arm':args.arm,'replay_audit':'PASS','classification':result['recovery_classification']}),flush=True)
    sys.argv=['run_reconstruction_phase44_full_decay.py','--contract',str(contract_path),'--training-result',str(out/'training-result.json'),
              '--training-dir',str(training),'--output-dir',str(out/'full-decay-reports'),'--output',str(out/'full-decay-gates.json')]
    return evaluation.main()


if __name__=='__main__':
    raise SystemExit(main())
