from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.reconstruction_trainer import _with_allowed_types
import torch


def test_constraint_policy_round_trip_preserves_all_modes():
    policy = ReconstructionConstraintPolicy(
        allowed_mother_types_by_level=((1, (8, 9)),),
        empirical_type_prior_mode="soft",
        minimum_pointer_probability=0.37,
        daughter_cardinality_policy="threshold",
    )
    assert ReconstructionConstraintPolicy.from_dict(policy.to_dict()) == policy


def test_training_context_uses_the_serialized_policy_masks():
    policy = ReconstructionConstraintPolicy(
        allowed_mother_types_by_level=((1, (8, 9)),),
        empirical_type_prior_mode="hard",
        valid_leaf_node_kinds=(1,),
    )
    batch = {
        "node_mask": torch.tensor([[True, True]]),
        "level_ids": torch.tensor([[0, 0]]),
        "node_kind_ids": torch.tensor([[1, 2]]),
    }
    training = _with_allowed_types(batch, 1, {}, policy)
    assert torch.equal(training["pointer_validity_mask"], policy.pointer_validity_mask(batch, 1))
    assert training["allowed_type_mask"].nonzero().flatten().tolist() == [8, 9]
    assert batch["node_mask"].all(), "constraint evaluation must not mutate rollout state"
