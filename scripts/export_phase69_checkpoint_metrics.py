"""Audit every saved checkpoint track and export its native scalar metadata on CPU."""
import argparse, gzip, json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.export_phase69_native_metrics import leaves
from scripts.build_reconstruction_phase69_closeout import ARMS, digest
from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(2)
    inventory={}; public=[]; count=0
    destination=a.output/'all-checkpoint-native-scalars.jsonl.gz'
    if destination.exists(): raise FileExistsError(destination)
    with gzip.open(destination,'wt') as out:
        for arm,job in ARMS.items():
            run=a.source/f'artifacts/runs/ht-reconstruction-phase69-20261005/{arm}/{job}'
            result=json.loads((run/'result.json').read_text())
            initial=result['pretraining_refinement']['checkpoint']
            base=torch.load(initial,map_location='cpu',weights_only=False)
            inventory[arm]={}
            for path in sorted((run/'training').glob('*.pt')):
                cp=torch.load(path,map_location='cpu',weights_only=False)
                lineage=validate_checkpoint_pair(initial,path,require_exact_frozen_encoder=False).as_dict()
                for name,state in cp.items():
                    if name.endswith('state_dict') and isinstance(state,dict):
                        assert all(torch.isfinite(t).all() for t in state.values() if isinstance(t,torch.Tensor))
                # Every available JSON-like scalar in checkpoint metadata, excluding
                # numerical tensors, optimizer state, and random generator arrays.
                metadata={k:v for k,v in cp.items() if not k.endswith('state_dict') and k not in ('optimizer','optimizer_state','rng_state','rng_states','scaler','scheduler')}
                rows=list(leaves(metadata))
                for row in rows:
                    out.write(json.dumps({'arm':arm,'checkpoint_track':path.name,**row},separators=(',',':'))+'\n')
                count+=len(rows)
                metrics=list(leaves(cp.get('metrics',{})))
                public.extend({'arm':arm,'metric':path.stem+'.'+r['metric'],'value':r['value']} for r in metrics)
                pid_keys=[k for k in cp.get('model_state_dict',{}) if 'leaf_pid_head' in k]
                assert len(pid_keys)==2 and all(torch.equal(base['model_state_dict'][k],cp['model_state_dict'][k]) for k in pid_keys)
                inventory[arm][path.name]={'sha256':digest(path),'bytes':path.stat().st_size,'step':cp['step'],'native_scalar_count':len(rows),'metric_count':len(metrics),'lineage':lineage,'pid_keys':pid_keys,'strict_evaluation':'registered native views only; other saved tracks not strict-evaluated'}
                print(arm,path.name,cp['step'],len(rows),flush=True)
    (a.output/'all-checkpoint-lineage.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (a.output/'public-checkpoint_metrics.json').write_text(json.dumps({'version':'phase69-aggregate-metrics-v1','view':'checkpoint_metrics','rows':public},separators=(',',':'))+'\n')
    (a.output/'checkpoint-export-manifest.json').write_text(json.dumps({'additional_native_scalar_count':count,'additional_public_metric_count':len(public),'saved_checkpoint_count':sum(map(len,inventory.values())),'file':destination.name,'sha256':digest(destination)},indent=2)+'\n')
if __name__=='__main__':main()
