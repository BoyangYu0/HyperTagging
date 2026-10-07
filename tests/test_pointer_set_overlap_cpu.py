"""Mechanistic loss checks; these do not establish real-data efficiency."""
import torch
import pytest
from hypertagging.losses.level_reconstruction import soft_pointer_set_overlap_loss


def test_set_overlap_penalizes_both_missing_and_foreign_daughters():
    target=torch.tensor([1.,1.,0.,0.])
    correct=torch.tensor([6.,6.,-6.,-6.])
    missing=correct.clone();missing[1]=-6.
    foreign=correct.clone();foreign[2]=6.
    baseline=soft_pointer_set_overlap_loss(correct,target)
    assert baseline < soft_pointer_set_overlap_loss(missing,target)
    assert baseline < soft_pointer_set_overlap_loss(foreign,target)
    logits=torch.zeros(4,requires_grad=True)
    soft_pointer_set_overlap_loss(logits,target).backward()
    assert (logits.grad[:2]<0).all() and (logits.grad[2:]>0).all()


def test_set_overlap_permutation_and_empty_axis():
    x=torch.tensor([2.,-1.,3.],requires_grad=True);t=torch.tensor([1.,0.,1.]);order=torch.tensor([2,0,1])
    torch.testing.assert_close(soft_pointer_set_overlap_loss(x,t),soft_pointer_set_overlap_loss(x[order],t[order]))
    empty=torch.empty(0,requires_grad=True);loss=soft_pointer_set_overlap_loss(empty,empty);loss.backward();assert loss.item()==0
    with pytest.raises(ValueError):soft_pointer_set_overlap_loss(x,t[:2])


def test_fp32_extreme_logits_have_finite_loss_and_gradient():
    x=torch.tensor([100.,-100.,0.],dtype=torch.bfloat16,requires_grad=True)
    loss=soft_pointer_set_overlap_loss(x,torch.tensor([1.,1.,0.]));loss.backward()
    assert loss.dtype==torch.float32 and torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_default_loss_is_unchanged_and_matched_set_term_is_additive():
    from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
    from hypertagging.data.tiny_level_fixtures import tiny_level_events
    from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
    from hypertagging.losses.level_reconstruction import level_reconstruction_loss
    batch=collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=16,n_queries=3)
    output=model(batch,target_level=1).pointer
    default=level_reconstruction_loss(output,batch,target_level=1)
    off=level_reconstruction_loss(output,batch,target_level=1,weights={"pointer_set_overlap":0.0})
    on=level_reconstruction_loss(output,batch,target_level=1,weights={"pointer_set_overlap":1.0})
    assert torch.equal(default.total,off.total) and default.matches==on.matches
    assert "pointer_set_overlap" not in off.components
    torch.testing.assert_close(on.total-off.total,on.components["pointer_set_overlap"])
    assert on.components["pointer_set_overlap"]>0
    on.components["pointer_set_overlap"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.decoder.parameters())


def test_enabled_objective_is_resume_bound_without_changing_legacy_identity():
    from types import SimpleNamespace
    from hypertagging.training.reconstruction_trainer import ReconstructionConfig, _data_order_contract
    from dataclasses import replace
    dm=SimpleNamespace(dataset_index=None,split_manifest_hash="fixed")
    off=ReconstructionConfig(data="unused.parquet",output_dir="unused-output");on=replace(off,pointer_set_overlap_weight=1.)
    left=_data_order_contract(off,dm);right=_data_order_contract(on,dm)
    assert "matched_daughter_set_objective" not in left
    assert right.pop("matched_daughter_set_objective")=={"version":"soft-jaccard-v1","weight":1.}
    assert left==right


def test_dry_run_executes_positive_new_objective():
    from hypertagging.training.level_reconstruction_train import run_level_reconstruction_dry_run
    out=run_level_reconstruction_dry_run(max_steps=2,batch_size=2,pointer_set_overlap_weight=1.)
    assert 0 < out.component_losses["pointer_set_overlap"] < 1
