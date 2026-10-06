"""Exercise both objective weights through the actual mixed-context loss graph."""
import copy
import torch
from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.reconstruction_trainer import ReconstructionConfig, _optimization_loss
from hypertagging.training.scheduled_sampling import TeacherForcingSchedule


def test_auxiliary_weight_changes_the_actual_gradient_without_changing_context():
    torch.manual_seed(20261007)
    batch=collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    base=LevelAutoregressiveReconstructor(n_features=batch['node_features'].shape[-1],n_types=len(PDG_TOKENS),hidden_dim=16,hyper_dim=4,n_queries=2)
    results=[]
    for weight in (0.5,1.0):
        model=copy.deepcopy(base);torch.manual_seed(20261007)
        config=ReconstructionConfig(data='unused',output_dir='unused',scheduled_sampling_probability=0.5,auxiliary_teacher_weight=weight,unrepresentable_target_policy='masked_representable_only',recovery_objective_weight=0.0)
        loss,leaf,_,_,metrics=_optimization_loss(model,batch,valid_levels=[1],config=config,schedule=TeacherForcingSchedule(start_probability=0.0,end_probability=0.0,duration_steps=2188),step=3000,use_scheduled_sampling=True,allowed_types_by_level={},constraint_policy=ReconstructionConstraintPolicy())
        assert metrics['sampled_predicted_count']==1 and metrics['auxiliary_teacher_loss']>0
        (loss+leaf).backward()
        gradient=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        assert torch.isfinite(gradient).all()
        results.append((loss.detach(),metrics,gradient))
    low,high=results
    assert low[1]['auxiliary_teacher_loss']==high[1]['auxiliary_teacher_loss']
    assert low[1]['model_forward_count']==high[1]['model_forward_count']
    assert torch.allclose(high[0]-low[0],torch.tensor(0.5*low[1]['auxiliary_teacher_loss']),rtol=1e-5)
    assert not torch.equal(low[2],high[2])
