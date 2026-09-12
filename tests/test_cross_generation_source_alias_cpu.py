"""A detector alias cannot be consumed by two branches across generations."""

from dataclasses import replace

import pytest
import torch

from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    _batched_event_structural_validity,
    _constrained_rollout_model_batch,
    append_composite_proposals,
    batched_free_rollout,
    evaluation_reference_rollout,
)
from tests.test_full_depth_beam_cpu import _ScriptedModel, _fsp_batch, _query, _rollout


def _aliased_batch():
    batch = _fsp_batch()
    batch["recursive_leaf_source_mask"][0, 1] = batch["recursive_leaf_source_mask"][0, 0]
    sources = batch["recursive_leaf_source_mask"][0]
    batch["source_conflict_matrix"] = (
        (sources.float() @ sources.float().T > 0)
        & ~torch.eye(sources.shape[0], dtype=torch.bool)
    ).unsqueeze(0)
    return batch


@pytest.mark.parametrize("rollout", [evaluation_reference_rollout, batched_free_rollout])
def test_greedy_reserves_sources_consumed_in_an_earlier_generation(rollout):
    def program(level, _batch):
        return [_query((0, 2))] if level == 1 else [_query((1, 3))]

    result = rollout(_ScriptedModel(program), _aliased_batch(), config=_rollout(max_level=2))
    batch = result.batch
    composites = batch["node_mask"] & batch["daughter_adjacency"].any(dim=-1)
    assert composites.sum() == 1
    roots = composites & (batch["parent_ids"] < 0)
    usage = (batch["recursive_leaf_source_mask"] & roots[..., None]).sum(dim=1)
    assert not (usage > 1).any()


def test_committed_composite_remains_eligible_but_its_unused_alias_does_not():
    batch, _ = append_composite_proposals(
        _aliased_batch(), [CompositeProposal(0, 4, (0, 2), 0.9, 0.9)], target_level=1
    )
    policy = _rollout().constraint_policy
    model_batch = _constrained_rollout_model_batch(batch, target_level=2, policy=policy)
    valid = model_batch["pointer_validity_mask"][0]
    assert not valid[1]
    assert valid[3] and valid[-1]
    # The policy switch remains explicit for diagnostic callers.
    unconstrained = _constrained_rollout_model_batch(
        batch, target_level=2, policy=replace(policy, reject_recursive_source_conflicts=False)
    )
    assert unconstrained["pointer_validity_mask"][0, 1]


def test_structural_check_detects_two_committed_roots_sharing_a_resource():
    batch, _ = append_composite_proposals(
        _aliased_batch(), [CompositeProposal(0, 4, (0, 2), 0.9, 0.9)], target_level=1
    )
    batch, _ = append_composite_proposals(
        batch, [CompositeProposal(0, 4, (1, 3), 0.9, 0.9)], target_level=2
    )
    assert not _batched_event_structural_validity(
        batch, initial_event_nonempty=torch.tensor([True])
    )[0]
