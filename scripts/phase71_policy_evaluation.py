"""Authenticate, reserve and execute the separately versioned Phase71 reserved policy study.

Raw UIDs, checkpoints and per-event reports belong in the supplied project-volume
output directory. This command never submits jobs or changes training contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    sha256, uid_sequence_sha256, uid_set_sha256, atomic_json,
)

CATEGORIES = ('charged', 'mixed', 'ccbar', 'uubar', 'ddbar', 'ssbar')
ARMS = ('aux_teacher_050', 'aux_teacher_100')
VERSION = 'phase71-policy-evaluation-v1-20261007'


def load(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(Path(path))}


def choose_cohort(category_by_uid, excluded, quota=2000):
    """Rank identities only; never condition on truth or reconstruction success."""
    pools = {cat: [] for cat in CATEGORIES}
    for uid, cat in category_by_uid.items():
        if uid not in excluded and cat in pools:
            pools[cat].append(uid)
    counts = {cat: len(rows) for cat, rows in pools.items()}
    if any(n < quota for n in counts.values()):
        return counts, None
    selected = {cat: sorted(rows, key=lambda uid: (
        hashlib.sha256(f'{VERSION}:{uid}'.encode()).hexdigest(), uid))[:quota]
        for cat, rows in pools.items()}
    return counts, selected


def manifest(uids):
    return {'manifest_version': 'hypertagging-reconstruction-evaluation-cohort-v1',
            'role': 'validation', 'sealed_test_role_access': 'forbidden',
            'event_uid_count': len(uids), 'event_uids': uids,
            'event_uids_sha256': uid_sequence_sha256(uids)}


def run_task(out, task_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Substantial evaluation requires a CPU Slurm allocation')
    plan = load(out/'plan.json'); task = plan['tasks'][task_id]
    subprocess.run(['git','diff','--exit-code',plan['source_head'],'--','src','scripts/evaluate_full_decay.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    for b in (task['cohort'], plan['selection'], plan['index'], plan['evaluator']):
        assert sha256(Path(b['path'])) == b['sha256']
    dest = out / 'tasks' / f'{task_id:03d}'
    dest.mkdir(parents=True, exist_ok=True)
    receipt_path = dest/'receipt.json'
    if receipt_path.exists():
        previous = load(receipt_path)
        if previous['returncode'] == 0 and sha256(dest/'report.json') == previous['report_sha256']:
            return
        raise RuntimeError('Prior failed task requires an explicit separate recovery attempt')
    cmd = [sys.executable, str(ROOT/'scripts/evaluate_full_decay.py')]
    for flag in ('pretraining-checkpoint','reconstruction-checkpoint'):
        b = plan['arms'][task['arm']][flag]
        assert sha256(Path(b['path'])) == b['sha256']
        cmd += ['--'+flag,b['path']]
    cmd += ['--data',plan['selection']['path'],'--dataset-index',plan['index']['path'],
            '--split','validation','--event-uid-manifest',task['cohort']['path'],
            '--scope','both','--truth-topology-mode','checkpoint_direct',
            '--max-events',str(task['count']),'--max-level','6','--object-threshold','0.6',
            '--pointer-threshold','0.35','--threads','1','--omit-trees',
            '--allow-finetuned-encoder','--diagnostic-external-independent-sample','--deterministic-algorithms','--output',str(dest/'report.json')]
    if task['view'] == 'beam_diagnostic':
        cmd += ['--beam-search','--beam-width','2','--beam-max-candidate-expansions-per-query','32',
                '--beam-max-proposals-per-level','8','--beam-max-candidates-per-query','2']
    receipt = {'task': task, 'command': cmd, 'job_id': os.environ['SLURM_JOB_ID'],
               'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'), 'status': 'RUNNING'}
    atomic_json(dest/'running.json',receipt)
    with (dest/'evaluation.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(returncode=result.returncode,status='COMPLETED' if result.returncode == 0 else 'FAILED')
    if result.returncode == 0:
        receipt['report_sha256'] = sha256(dest/'report.json')
    atomic_json(receipt_path,receipt)
    if result.returncode:
        raise SystemExit(result.returncode)


def prepare(out, source):
    """Authenticate the original reserved primary population; never draw replacements."""
    import pyarrow.parquet as pq
    from scripts.build_reconstruction_phase71_closeout import verify_receipt, ARMS as JOBS
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists(): raise FileExistsError(out/'plan.json')
    p=load(source/'configs/reconstruction/ht_reconstruction_phase71_20261006.json')
    inputs=[]
    def bound(b):
        path=source/b['path']; assert sha256(path)==b['sha256'], path
        inputs.append(binding(path)); return load(path)
    c=bound(p['validation_cohort']); reserved=bound(p['policy_evaluation_plan'])
    history=bound(c['source_bindings']['previous_validation_universe'])
    assert uid_set_sha256(history['event_uids'])==history['event_uids_sha256']==reserved['exclusion_uid_set_sha256']
    excluded=set(history['event_uids']); selection_uids=set(c['checkpoint_selection_event_uids'])
    groups=reserved['by_category']; primary=set(reserved['event_uids'])
    assert set(groups)==set(CATEGORIES) and all(len(x)==len(set(x))==2000 for x in groups.values())
    assert len(primary)==12000 and primary==set(u for rows in groups.values() for u in rows)
    assert len(selection_uids)==1000 and len(excluded)==85000
    assert not primary & (selection_uids|excluded) and not selection_uids & excluded
    registry=load(ROOT/'configs/reconstruction/supplementary_validation_reservations.json')
    for r in registry['reservations']:
        path=Path('/project/agkuhr/users/boyang/data/HyperTagging_artifacts')/r['cohort_manifest']['artifact_relative_path']
        assert sha256(path)==r['cohort_manifest']['sha256']
        u=set(load(path)['event_uids']); assert len(u)==r['reserved_count']
        if r['version'].startswith('phase71-'): assert u==primary|selection_uids
        else: assert u<=excluded
        inputs.append(binding(path))
    selection=source/p['data_binding']['selection_manifest']; index_path=source/p['data_binding']['dataset_index']
    assert sha256(selection)==p['data_binding']['selection_manifest_sha256']
    assert sha256(index_path)==p['data_binding']['dataset_index_sha256']
    m,index=load(selection),load(index_path)
    assert not m['selection_includes_test'] and index['normalizer_scope']=='train'
    parent=bound(p['parent_phase70']); old_index=load(source/parent['data_binding']['dataset_index'])
    assert index['normalizer_state']==old_index['normalizer_state']
    assert index['event_identity_validation']['status']=='passed'
    categories={}; train=set(); shards=[]; roles={}
    for e in m['entries']:
        role=e['split']; assert role in ('train','validation')
        assert roles.setdefault(e['source_file'],role)==role==index['source_groups'][e['source_file']]
        path=Path(m['data_root'])/e['path']; b=binding(path)
        assert b['sha256']==e['parquet_sha256_reference']; shards.append({**b,'split':role,'category':e['category']})
        t=pq.read_table(path,columns=['event_uid','source_category','source_file'],use_threads=False)
        assert t.num_rows==e['event_count']
        for row in t.to_pylist():
            u=row['event_uid']; assert u not in train and u not in categories
            assert row['source_category']==e['category'] and row['source_file']==e['source_file']
            if role=='train':train.add(u)
            else:categories[u]=row['source_category']
    assert len(train)==70000 and len(categories)==160000 and not train & (primary|selection_uids|excluded)
    assert primary|selection_uids|excluded<=set(categories)
    assert all(categories[u]==cat for cat,rows in groups.items() for u in rows)
    atomic_json(out/'cohort.json',{**manifest(reserved['event_uids']),'by_category':groups,'original_policy_plan':binding(source/p['policy_evaluation_plan']['path'])})
    coverage={'version':VERSION,'status':'AUTHENTICATED_RESERVED_NOT_PROCESSED','required_per_category':2000,'validation_total':160000,'train_events':70000,'prior_reserved':85000,'selection_events':1000,'training_overlap':0,'selection_overlap':0,'prior_reservation_overlap':0,'selection_uses_truth':False,'sealed_test_accessed':False,'source_bindings':inputs,'shards':shards,'original_reservation_preserved':True}
    atomic_json(out/'coverage-authentication.json',coverage)
    arms={}
    for arm,job in JOBS.items():
        run=source/f'artifacts/runs/ht-reconstruction-phase71-20261006/{arm}/{job}'
        native=source/f'artifacts/slurm/reconstruction-phase71/jobs/{job}/attempt-00/receipt.json'
        receipt=load(native);verify_receipt(receipt)
        contract=load(run/'provenance/submitted-contract.json')
        assert hashlib.sha256(json.dumps({k:v for k,v in contract.items() if k!='contract_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==contract['contract_sha256']==receipt['contract_sha256']
        assert contract['arm_role']==arm and contract['sealed_test_role_access']=='forbidden'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==contract['expected_git_sha']
        subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts'],cwd=source,check=True,stdout=subprocess.DEVNULL)
        result=load(run/'result.json'); cp=result['checkpoints']['best']
        checkpoints={'pretraining-checkpoint':binding(result['pretraining_refinement']['checkpoint']),'reconstruction-checkpoint':binding(cp['path'])}
        assert checkpoints['reconstruction-checkpoint']['sha256']==cp['sha256']
        old_report=run/'full-decay-reports/primary_complete_target_direct.json'
        assert load(old_report)['checkpoint_pair']['reconstruction_sha256']==cp['sha256']
        pair_path=out/f'{arm}-checkpoint-pair.json'
        cmd=[sys.executable,str(ROOT/'scripts/validate_reconstruction_checkpoint_pair.py')]
        for flag,b in checkpoints.items():cmd+=['--'+flag,b['path']]
        subprocess.run(cmd+['--allow-finetuned-encoder','--output',str(pair_path)],check=True,stdout=subprocess.DEVNULL)
        assert load(pair_path)['compatible']
        arms[arm]={**checkpoints,'native_receipt':binding(native),'historical_report':binding(old_report),'checkpoint_pair':binding(pair_path)}
    tasks=[]
    for cat in CATEGORIES:
        for part in range(4):
            rows=groups[cat][part*500:(part+1)*500]; path=out/'cohorts'/f'{cat}-{part}.json';atomic_json(path,manifest(rows))
            for arm in ARMS:tasks.append({'arm':arm,'category':cat,'view':'greedy','cohort':binding(path),'count':len(rows)})
    path=out/'cohorts/beam-diagnostic.json';atomic_json(path,manifest([u for cat in CATEGORIES for u in groups[cat][:10]]))
    for arm in ARMS:tasks.append({'arm':arm,'category':'stratified','view':'beam_diagnostic','cohort':binding(path),'count':60})
    atomic_json(out/'plan.json',{'version':VERSION,'cohort':binding(out/'cohort.json'),'coverage':binding(out/'coverage-authentication.json'),'arms':arms,'selection':binding(selection),'index':binding(index_path),'tasks':tasks,'evaluator':binding(ROOT/'scripts/evaluate_full_decay.py'),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--task-id',type=int);a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve(),a.source.resolve())
    else:run_task(a.output.resolve(),a.task_id)
