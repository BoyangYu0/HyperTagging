"""Prove the transfer intervention changes encoder parameters and preserves controls."""
import torch
import pytest
from scripts.phase68_encoder_initialization import derive, digest


def test_encoder_ablation_preserves_pid_normalization_and_sources(tmp_path):
    refined, initial, output = [tmp_path / name for name in ('refined.pt','initial.pt','output.pt')]
    encoder = {'weight':torch.tensor([2.]), 'klm_encoder.input_scales':torch.tensor([3.])}
    model = {'encoder.'+k:v for k,v in encoder.items()}
    model.update({'leaf_pid_head.weight':torch.tensor([4.]), 'runtime_feature_normalizer.common_mean':torch.tensor([5.])})
    torch.save({'encoder_state_dict':encoder,'model_state_dict':model,'step':2188,'epoch':1,'metrics':{'loss':1.},'normalizer_state':{'mean':[5.]},'optimizer_state_dict':{'old':True}},refined)
    torch.save({'model_state_dict':{'encoder.weight':torch.tensor([1.]),'encoder.klm_encoder.input_scales':torch.tensor([3.])}},initial)
    r=derive(refined,initial,output,refined_sha256=digest(refined),initialization_sha256=digest(initial))
    cp=torch.load(output,weights_only=False)
    assert cp['step']==0 and cp['metrics']=={} and 'optimizer_state_dict' not in cp
    assert cp['normalizer_state']=={'mean':[5.]}
    assert cp['encoder_state_dict']['weight'].item()==1
    assert cp['model_state_dict']['leaf_pid_head.weight'].item()==4
    assert cp['model_state_dict']['runtime_feature_normalizer.common_mean'].item()==5
    assert r['changed_encoder_keys']==['weight']
    with pytest.raises(FileExistsError):
        derive(refined,initial,output,refined_sha256=digest(refined),initialization_sha256=digest(initial))
    with pytest.raises(ValueError,match='hash'):
        derive(refined,initial,tmp_path/'bad.pt',refined_sha256='0'*64,initialization_sha256=digest(initial))


def test_arm_bound_evaluation_accepts_zero_step_transfer_and_rejects_wrong_lineage(tmp_path):
    from scripts.run_reconstruction_phase68_full_decay import refined_evaluation_runtime
    p=tmp_path/'checkpoint.pt';p.write_bytes(b'fixture')
    runtime={'checkpoint_sha256':digest(p),'checkpoint_step':'0'}
    ref={'checkpoint':str(p),'status':'COMPLETED','step':0,'checkpoint_selection':'arm_bound_parameter_transfer','source_checkpoint_sha256':digest(p),'source_unchanged':True,'checkpoint_sha256':digest(p)}
    assert refined_evaluation_runtime({'pretraining_refinement':ref},runtime)['checkpoint_step']=='0'
    ref['step']=2188
    with pytest.raises(RuntimeError):
        refined_evaluation_runtime({'pretraining_refinement':ref},runtime)
