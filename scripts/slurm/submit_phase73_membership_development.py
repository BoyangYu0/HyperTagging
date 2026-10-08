"""Guarded, one-shot submission of the authorized two-arm CPU development campaign."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.diagnose_search_survival import sha, checked, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--submit',action='store_true');a=p.parse_args()
    path=a.contract.resolve();c=json.loads(path.read_text());base=path.parent
    receipt=json.loads((base/'admission.json').read_text())
    assert receipt['status']=='PASS' and receipt['contract_sha256']==sha(path)
    assert c['arms']==['frozen','adapted'] and c['stage']=='development'
    assert not c['automatic_successor'] and not c['sealed_test_access']
    assert c['resources']=={'cpus':2,'memory_gib':32,'hours':24,'gpus':0,'max_jobs':2,'requeue':False}
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==c['source_sha']
    for name,digest in c['source_hashes'].items(): assert sha(ROOT/name)==digest
    for binding in [*c['inputs'].values(),c['scientific_plan'],c['policy'],c['evidence'],c['phase72_aggregate'],c['gradient_preflight']]: checked(binding)
    queue=json.loads(subprocess.check_output(['squeue','--json','--user',__import__('getpass').getuser()],text=True))
    assert not any('phase73-membership-' in j['name'] for j in queue['jobs']), 'Existing campaign allocation'
    assert not (base/'submission-lock.json').exists(), 'Campaign already submitted or attempted'
    commands=[]
    for arm in c['arms']:
        commands.append(['sbatch','--parsable','--chdir='+str(ROOT),'--job-name=phase73-membership-'+arm+'-20261008',
            '--output='+str(base/(arm+'-%j.log')),str(ROOT/'scripts/slurm/run_phase73_membership_development.sbatch'),str(path),arm])
    if not a.submit:
        print(json.dumps({'status':'ELIGIBLE_NOT_SUBMITTED','commands':commands},indent=2));return
    write(base/'submission-lock.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'contract_sha256':sha(path),'pid':os.getpid()})
    for arm,cmd in zip(c['arms'],commands):
        job=subprocess.check_output(cmd,text=True).strip()
        assert job.split(';')[0].isdigit()
        write(base/(arm+'-submission.json'),{'arm':arm,'job_id':job,'command':cmd,'submitted_utc':datetime.now(timezone.utc).isoformat(),'source_sha':c['source_sha'],'contract_sha256':sha(path)})
        print(arm,job,flush=True)

if __name__=='__main__': main()
