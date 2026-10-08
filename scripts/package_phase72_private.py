"""Archive complete private Phase72 records with member integrity and cardinality.

Weights/data remain immutable external artifacts, hash-bound by lineage exports.
Never package CLI runner logs, credentials, live training or unrelated sessions.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def leaves(value, prefix=''):
    if isinstance(value,dict):
        for key,item in value.items():yield from leaves(item,prefix+'.'+str(key) if prefix else str(key))
    elif isinstance(value,list):
        for i,item in enumerate(value):yield from leaves(item,prefix+'.'+str(i))
    elif value is None or type(value) in (int,float,bool):yield prefix,value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.umask(0o077);a.output.mkdir(parents=True,exist_ok=False)
    aggregate=json.loads((a.artifacts/'primary-v1/aggregate.json').read_text())
    assert aggregate['status']=='COMPLETE_GREEDY_WITH_DIAGNOSTIC_BEAM'
    receipts=list((a.artifacts/'primary-v1/tasks').glob('*/receipt.json'));assert len(receipts)==50
    counts={};scalar_path=a.output/'supplemental-all-scalars.jsonl.gz'
    with gzip.open(scalar_path,'wt') as target:
        for path in sorted((a.artifacts/'primary-v1/tasks').glob('*/report.json')):
            name=path.parent.name;count=0
            for metric,value in leaves(json.loads(path.read_text())):
                target.write(json.dumps({'task':name,'metric':metric,'value':value},separators=(',',':'),allow_nan=False)+'\n');count+=1
            counts[name]={'scalar_records':count,'report_sha256':digest(path)}
    assert len(counts)==50
    members={}
    def collect(root,prefix):
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix not in ('.pt','.pth','.ckpt','.parquet','.pyc') and '__pycache__' not in path.parts:
                members[prefix+'/'+str(path.relative_to(root))]=path
    for folder in ('native','checkpoint-exhaustive','additive','primary-v1'):
        collect(a.artifacts/folder,folder)
    collect(a.native_source/'artifacts/runs/ht-reconstruction-phase72-20261007','native-original-runs')
    collect(a.native_source/'artifacts/slurm/reconstruction-phase72','native-original-contracts-receipts')
    members['review/primary-submission.json']=a.artifacts/'control/primary-submission.json'
    members['exports/'+scalar_path.name]=scalar_path
    manifest={'version':'phase72-private-evidence-bundle-v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'private':True,'checkpoint_weights_and_data':'External immutable hash-bound files; no payload mutation or republishing.',
              'scalar_cardinality':{'supplemental_total':sum(r['scalar_records'] for r in counts.values()),'supplemental_reports':counts,
                'native_records':sum(r['native_scalar_count'] for r in json.loads((a.artifacts/'native/export-manifest.json').read_text())['arms'].values()),'checkpoint_metadata_records':json.loads((a.artifacts/'checkpoint-exhaustive/checkpoint-export-manifest.json').read_text())['additional_native_scalar_count'],'additive_records':json.loads((a.artifacts/'additive/export-counts.json').read_text())['additive_native_scalar_records']},
              'files':[{'member':name,'sha256':digest(path),'bytes':path.stat().st_size} for name,path in sorted(members.items())]}
    manifest_path=a.output/'integrity-manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    archive=a.output/'phase72-private-evidence-v1.tar.gz'
    with archive.open('xb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for name,path in [*sorted(members.items()),('integrity-manifest.json',manifest_path)]:
            info=tar.gettarinfo(str(path),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o600
            with path.open('rb') as f:tar.addfile(info,f)
    # Independently stream every archived member to verify no missing/truncated content.
    expected={r['member']:r for r in manifest['files']};seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if member.name=='integrity-manifest.json':continue
            h=hashlib.sha256();size=0
            handle=tar.extractfile(member)
            for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk);size+=len(chunk)
            assert member.name not in seen and h.hexdigest()==expected[member.name]['sha256'] and size==expected[member.name]['bytes']
            seen.add(member.name)
    assert seen==set(expected)
    result={'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'member_count':len(seen)+1,
            'scalar_cardinality':manifest['scalar_cardinality'],'all_member_hashes_verified':True}
    (a.output/'bundle-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='scalar_cardinality'}))


if __name__=='__main__':main()
