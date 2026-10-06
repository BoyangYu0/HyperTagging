"""Aggregate scientific audits without private identities or operational paths."""
from pathlib import Path
import argparse,collections,json

def load(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 close=load(a.native/'closeout.json');geometry=load(a.native/'geometry.json');lineage=load(a.native/'lineage.json');decoder=load(a.native/'decoder-independent-audit.json')
 result={'version':'phase70-additional-scientific-diagnostics-v1','native_coverage_classification':'HISTORICAL_DIAGNOSTIC_NOT_CATEGORY_COMPLIANT','geometry':geometry['models'],'geometry_sample_counts':{'train':len(geometry['train_calibration_event_uids']),'development':len(geometry['development_event_uids'])},'geometry_is_fixed_input_not_generated_state_census':True,'decoder':{arm:{**row,'independently_verified_tracks':{Path(name).stem:record for name,record in row['independently_verified_tracks'].items()}} for arm,row in decoder['arms'].items()},'paired_native_diagnostic':load(a.native/'paired.json'),'arms':{}}
 for arm,record in close['arms'].items():
  dst={k:record[k] for k in ('optimizer_steps','selected','gates','all_gates_passed','primary_repeat_identical','training_elapsed_seconds','complete_target_count','forest','beam')}
  dst['context_and_encoder_execution']=lineage['arms'][arm]
  dst['checkpoint_transfer']={track:{k:v for k,v in record.items() if k not in ('pretraining_checkpoint','reconstruction_checkpoint')} for track,record in load(a.native/(arm+'-checkpoint-lineage.json')).items()}
  dst['view_topology']={}
  for view,row in record['view_topology_audits'].items():
   counts=collections.Counter((x['scope'],x['search'],x['unit_semantics'],x['truth_mothers'],x['maximum_truth_depth']) for x in row['coherent_forests'])
   dst['view_topology'][view]={k:v for k,v in row.items() if k!='coherent_forests'}
   dst['view_topology'][view]['coherent_forest_shapes']=[{'scope':s,'search':mode,'unit_semantics':u,'mothers':m,'maximum_depth':d,'count':n} for (s,mode,u,m,d),n in sorted(counts.items())]
  result['arms'][arm]=dst
 if a.output.exists():raise FileExistsError(a.output)
 a.output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
if __name__=='__main__':main()
