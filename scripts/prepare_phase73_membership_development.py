"""Authenticate a single development campaign; no reservation or scheduler submission."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.diagnose_search_survival import checked, sha, write
from scripts.validate_next_reconstruction_study import validate


def prepare(artifacts, diagnosis, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Real-data admission requires a bounded CPU allocation')
    import pyarrow.parquet as pq
    import torch
    if output.exists(): raise FileExistsError(output)
    policy_path = ROOT/'configs/reconstruction/next_study_policy.json'
    scientific_path = ROOT/'configs/reconstruction/phase73_membership_development_plan.json'
    policy = json.loads(policy_path.read_text())
    evidence_path = ROOT/policy['evidence']['path']
    assert sha(evidence_path) == policy['evidence']['sha256']
    result = validate(json.loads(scientific_path.read_text()), policy, json.loads(evidence_path.read_text()))
    assert result['status'] == 'PASS', result
    parent = json.loads((diagnosis/'plan.json').read_text())
    inputs = parent['inputs']
    paths = {k: checked(v) for k,v in inputs.items()}
    coverage_path = artifacts/'primary-v1/coverage-authentication.json'
    coverage = json.loads(coverage_path.read_text())
    for b in coverage['shards']: checked(b)
    selection, index = [json.loads(paths[k].read_text()) for k in ('selection','index')]
    assert not selection['selection_includes_test'] and index['normalizer_scope']=='train'
    primary_plan = json.loads((artifacts/'primary-v1/plan.json').read_text())
    assert primary_plan['selection']['sha256'] == inputs['selection']['sha256']
    assert primary_plan['index']['sha256'] == inputs['index']['sha256']
    old_train_path = diagnosis/'membership-pilot-v1/run/training-cohort.json'
    old_train = json.loads(old_train_path.read_text())
    train = sorted(old_train['event_uids'])
    dev = json.loads(paths['cohort'].read_text())['event_uids']
    assert len(train)==len(set(train))==384 and len(dev)==len(set(dev))==60
    assert not set(train)&set(dev)
    categories={}; roles={}
    for entry in selection['entries']:
        assert entry['split'] in ('train','validation')
        assert roles.setdefault(entry['source_file'],entry['split']) == entry['split'] == index['source_groups'][entry['source_file']]
        path=Path(selection['data_root'])/entry['path']
        for row in pq.read_table(path,columns=['event_uid','source_category'],use_threads=False).to_pylist():
            uid=row['event_uid']
            if uid in set(train)|set(dev):
                assert uid not in categories
                categories[uid]=(entry['split'],row['source_category'])
    assert all(categories[u][0]=='train' for u in train)
    assert all(categories[u][0]=='validation' for u in dev)
    from collections import Counter
    assert set(Counter(categories[u][1] for u in train).values())=={64}
    assert set(Counter(categories[u][1] for u in dev).values())=={10}
    native_source=Path(primary_plan['arms']['set_overlap_off']['native_receipt']['path']).parents[6]
    prereg=json.loads((native_source/'configs/reconstruction/ht_reconstruction_phase72_20261007.json').read_text())
    p72_cohort_path=checked({'path':str(native_source/prereg['validation_cohort']['path']),'sha256':prereg['validation_cohort']['sha256']})
    p72_cohort=json.loads(p72_cohort_path.read_text())
    primary=json.loads(checked(primary_plan['cohort']).read_text())['event_uids']
    assert not (set(train)|set(dev)) & (set(primary)|set(p72_cohort['checkpoint_selection_event_uids']))
    cp=torch.load(paths['reconstruction-checkpoint'],map_location='cpu',weights_only=False)
    assert cp['encoder_state_dict']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code',head,'--','src','scripts','configs/reconstruction'],cwd=ROOT,check=True)
    def binding(p): return {'path':str(p.resolve()),'sha256':sha(p)}
    source_hashes={str(p.relative_to(ROOT)):sha(p) for folder in ('src','scripts') for p in (ROOT/folder).rglob('*.py')}
    contract={'version':'phase73-membership-runtime-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'stage':'development','arms':['frozen','adapted'],'automatic_successor':False,'sealed_test_access':False,
        'source_sha':head,'source_root':str(ROOT),'source_hashes':source_hashes,'inputs':inputs,
        'scientific_plan':binding(scientific_path),'policy':binding(policy_path),'evidence':binding(evidence_path),
        'train_uids':train,'development_uids':dev,'data_shards':coverage['shards'],
        'exclusion_bindings':[binding(p72_cohort_path),primary_plan['cohort'],binding(old_train_path),binding(coverage_path)],
        'output_root':str(output/'runs'),'resources':{'cpus':2,'memory_gib':32,'hours':24,'gpus':0,'max_jobs':2,'requeue':False},
        'settings':{'seed':20261008,'head_width':128,'tiny_updates':1000,'tiny_batch_size':24,'pilot_updates':1000,'pilot_batch_size':16,'head_lr':.001,'encoder_lr':.00005,'checkpoint':'fixed_final','presence_threshold':.5},
        'normalizer':'inherited_train_fitted_unchanged','selection':'384_original_train_and60_reused_development_no_fresh_reservation',
        'phase72_aggregate':binding(artifacts/'primary-v1/aggregate.json')}
    output.mkdir(parents=True)
    write(output/'contract.json',contract)
    write(output/'admission.json',{'status':'PASS','plan_validation':result,'contract_sha256':sha(output/'contract.json'),
        'source_sha':head,'train_count':len(train),'development_count':len(dev),'phase72_overlap':0,'sealed_test_access':False,
        'fresh_primary_reserved':0,'scientific_promotion':False})
    print(json.dumps({'status':'PASS','contract':str(output/'contract.json'),'sha256':sha(output/'contract.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepare(a.artifacts.resolve(),a.diagnosis.resolve(),a.output.resolve())
