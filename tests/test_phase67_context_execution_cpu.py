"""Exercise the proposed teacher-only regime through the real optimizer loss."""
import torch
from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.reconstruction_trainer import ReconstructionConfig, _optimization_loss
from hypertagging.training.scheduled_sampling import TeacherForcingSchedule


def test_teacher_only_has_finite_update_without_rollout_or_auxiliary_branch():
    torch.manual_seed(20261004)
    batch=collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    model=LevelAutoregressiveReconstructor(n_features=batch['node_features'].shape[-1],n_types=len(PDG_TOKENS),hidden_dim=16,hyper_dim=4,n_queries=2)
    config=ReconstructionConfig(data='unused',output_dir='unused',scheduled_sampling_probability=0.0,auxiliary_teacher_weight=0.5,unrepresentable_target_policy='masked_representable_only',recovery_objective_weight=0.0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    loss,leaf_loss,_,_,metrics=_optimization_loss(model,batch,valid_levels=[1],config=config,schedule=TeacherForcingSchedule(start_probability=1.0,end_probability=1.0,duration_steps=2188),step=3000,use_scheduled_sampling=True,allowed_types_by_level={},constraint_policy=ReconstructionConstraintPolicy())
    assert metrics['sampled_predicted_count']==metrics['rollout_call_count']==metrics['auxiliary_teacher_loss']==0
    assert metrics['sampled_teacher_count']==1
    total=loss+leaf_loss
    assert torch.isfinite(total)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert any(not torch.equal(before[k],v) for k,v in model.named_parameters())
