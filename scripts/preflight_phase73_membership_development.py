"""One authenticated training-event gradient preflight; no saved fitted model."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.diagnose_search_survival import checked, sha, write


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU allocation required')
    import torch
    from hypertagging.data.heterogeneous import collate_heterogeneous_events
    from hypertagging.evaluation.full_decay_metrics import _tree_view, _truth_b_roots
    from hypertagging.evaluation.trained_context import load_trained_evaluation_context
    from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
    from hypertagging.reconstruction.level_rollout import _constrained_rollout_model_batch
    from hypertagging.reconstruction.beam_search import _truth_free_model_view
    from hypertagging.models.direct_membership import DirectMembershipHead, direct_membership_loss
    p=argparse.ArgumentParser();p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((a.diagnosis/'plan.json').read_text());paths={k:checked(v) for k,v in plan['inputs'].items()}
    dev=json.loads(paths['cohort'].read_text())['event_uids']
    ctx=load_trained_evaluation_context(checkpoint=paths['reconstruction-checkpoint'],data=paths['selection'],dataset_index=paths['index'],split='validation',device='cpu',max_events=1,diagnostic_allow_external_independent_sample=True,event_selection='explicit_uids',explicit_event_uids=dev[:1])
    cohort=json.loads((a.diagnosis/'membership-pilot-v1/run/training-cohort.json').read_text())
    uid=cohort['by_category']['charged'][0]
    event=next(ctx.data_module.iter_events('train',event_uids=[uid]))
    truth=ctx.data_module.normalize_batch(collate_heterogeneous_events([event]));projection=project_schema_v4_fsps(truth)
    model_input=_truth_free_model_view(_constrained_rollout_model_batch(projection.batch,target_level=1,policy=ctx.constraint_policy))
    view=_tree_view(truth,0,truth=True);roots=_truth_b_roots(view);assert len(roots)==2
    keys=projection.evaluation_leaf_source_keys[0].tolist();target=torch.zeros((1,len(keys)),dtype=torch.long)
    for slot,root in enumerate(roots,1):
        members=view.source_set(root);assert members and members<=set(keys)
        for i,key in enumerate(keys):
            if key in members:target[0,i]=slot
    base=ctx.model.eval();results={};outputs=[]
    for arm in ('frozen','adapted'):
        model=copy.deepcopy(base)
        for par in model.parameters():par.requires_grad_(False)
        for par in model.encoder.parameters():par.requires_grad_(arm=='adapted')
        pid={k:v.clone() for k,v in model.leaf_pid_head.state_dict().items()}
        torch.manual_seed(20261008);head=DirectMembershipHead(128,128)
        features=model(model_input,target_level=1,pid_kinematics_mode_override='soft_expectation',pid_temperature_override=.5).node_embeddings
        assert features.shape[-1]==128
        logits,objects=head(features,model_input['node_mask']);outputs.append(logits.detach())
        loss=direct_membership_loss(logits,objects,target,model_input['node_mask']);loss.backward()
        gradients=[p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert bool(gradients)==(arm=='adapted')
        assert all(torch.isfinite(g).all() for g in gradients)
        assert not gradients or any(g.ne(0).any() for g in gradients)
        params=[p for p in [*head.parameters(),*model.encoder.parameters()] if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True)
        optimizer=torch.optim.AdamW(params,lr=.00005);optimizer.step()
        assert all(torch.equal(v,pid[k]) for k,v in model.leaf_pid_head.state_dict().items())
        results[arm]={'loss':float(loss.detach()),'gradient_norm':float(norm),'encoder_gradient_tensors':len(gradients),'pid_unchanged':True,'finite_parameters':all(bool(torch.isfinite(p).all()) for p in params)}
    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
    write(a.output,{'status':'PASS','scope':'one_original_training_identity_gradient_preflight_no_saved_model','identity_sha256':__import__('hashlib').sha256(uid.encode()).hexdigest(),'inputs':plan['inputs'],'script_sha256':sha(Path(__file__)),'initial_predictions_identical':True,'arms':results})
    print(json.dumps(results))
if __name__=='__main__':main()
