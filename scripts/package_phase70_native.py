"""Losslessly publish all native aggregate scalar records with compact metric names."""
from pathlib import Path
import argparse,gzip,hashlib,json

def encode(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 views_expected=('checkpoint_metrics','independent_complete_target_direct','independent_depth_direct','independent_tree_validity_direct','primary_complete_target_beam_direct','primary_complete_target_contracted_diagnostic','primary_complete_target_direct','primary_complete_target_repeat2_direct','training_history_summary')
 sources=[a.input/('public-'+view+'.json') for view in views_expected];assert all(s.is_file() for s in sources)
 names=[];lookup={};records=[];views=[];arms=['type_bias','no_type_bias'];identities=set();bindings=[];expected=[]
 for source in sources:
  doc=json.loads(source.read_text());view=doc['view'];views.append(view)
  bindings.append({'view':view,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scalar_records':len(doc['rows'])})
  for row in doc['rows']:
   expected.append((view,row['arm'],row['metric'],row['value']))
   name=row['metric'];identity=(view,row['arm'],name);assert identity not in identities;identities.add(identity)
   if name not in lookup:lookup[name]=len(names);names.append(name)
   records.append([len(views)-1,arms.index(row['arm']),lookup[name],row['value']])
 segments=list(dict.fromkeys(s for name in names for s in name.split('.')));indices={s:i for i,s in enumerate(segments)}
 value={'version':'phase70-native-aggregate-bundle-v1','views':views,'arms':arms,'metric_name_encoding':'dot_joined_segment_indices','metric_name_segments':segments,'metric_names':[[indices[s] for s in name.split('.')] for name in names],'columns':['view_index','arm_index','metric_index','value'],'records':records,'source_bindings':bindings}
 restored=[(views[v],arms[a],'.'.join(segments[x] for x in value['metric_names'][m]),n) for v,a,m,n in records]

 assert restored==expected
 data=encode(value);assert len(data)<10*1024*1024
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'native.json.gz').write_bytes(gzip.compress(data,mtime=0))
 binding={'version':'phase70-native-binding-v1','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'scalar_records':len(records),'views':views,'lossless_roundtrip_verified':True}
 (a.output/'binding.json').write_bytes(encode(binding));print(json.dumps(binding))
if __name__=='__main__':main()
