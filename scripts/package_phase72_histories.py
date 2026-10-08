"""Losslessly compact every per-step native training scalar for public download."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def canonical(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pack(rows):
    columns=defaultdict(list)
    for r in rows:columns[r['metric']].append((r['record'],r['value']))
    result=[]
    for name,items in sorted(columns.items()):
        indices=[];values=[]
        for record,value in sorted(items,key=lambda item:item[0]):
            if indices and record==sum(indices[-1]):indices[-1][1]+=1
            else:indices.append([record,1])
            if values and type(values[-1][1]) is type(value) and values[-1][1]==value:values[-1][0]+=1
            else:values.append([1,value])
        result.append({'metric':name,'index_runs':indices,'value_runs':values})
    return result

def unpack(columns):
    rows=[]
    for c in columns:
        ids=[i for start,count in c['index_runs'] for i in range(start,start+count)]
        values=[v for count,v in c['value_runs'] for _ in range(count)]
        assert len(ids)==len(values)==len(set(ids))
        rows.extend({'record':i,'metric':c['metric'],'value':v} for i,v in zip(ids,values))
    return sorted(rows,key=lambda r:(r['record'],r['metric']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint-input',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);files=[]
    for arm in ('set_overlap_off','set_overlap_on'):
        source=a.input/(('all-checkpoint-native-scalars.jsonl.gz') if a.checkpoint_input else arm+'-training-history.json.gz')
        tracks=[]
        with gzip.open(source,'rt') as f:
            if a.checkpoint_input:
                native=[json.loads(line) for line in f]
                native=[r for r in native if r['arm']==arm]
                tracks=sorted({Path(r['checkpoint_track']).stem for r in native})
                rows=[{'arm':arm,'record':tracks.index(Path(r['checkpoint_track']).stem),'metric':r['metric'],'value':r['value']} for r in native]
            else:rows=json.load(f)['training']
        assert all(r['arm']==arm for r in rows)
        original=sorted([{k:r[k] for k in ('record','metric','value')} for r in rows],key=lambda r:(r['record'],r['metric']))
        columns=pack(rows);restored=unpack(columns)
        assert canonical(restored)==canonical(original)
        body={'record_semantics':'checkpoint_track_index' if a.checkpoint_input else 'native_log_record_index','checkpoint_tracks':tracks,'version':'phase72-training-scalar-rle-v1','arm':arm,'scalar_count':len(rows),'encoding':'per_metric_index_runs_and_typed_value_runs','columns':columns,
              'decoded_scalar_sha256':hashlib.sha256(canonical(original)).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        data=canonical(body);assert len(data)<=10*1024*1024
        filename='phase72-'+arm.replace('_','-')+('-checkpoint-scalars.json' if a.checkpoint_input else '-training-scalars.json')
        (a.output/(arm+'.json.gz')).write_bytes(gzip.compress(data,mtime=0))
        files.append({'source':arm+'.json.gz','filename':filename,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'scalar_records':len(rows),'decoded_scalar_sha256':body['decoded_scalar_sha256']})
    (a.output/'binding.json').write_bytes(canonical({'version':'phase72-training-scalar-binding-v1','files':files,'all_scalars_lossless':True}))
    print(json.dumps(files))
if __name__=='__main__':main()
