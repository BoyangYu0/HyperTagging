"""Bounded CPU reproducer of scheduled-sampling/inference eligibility mismatch.

Run from the reviewed checkout with PYTHONPATH=src and the existing CPU env.
The target at level 2 needs leaf 0 and composite 4. Predicted composite 5 has
already consumed leaf 0, so that target cannot be constructed in this forest.
"""
import json

import torch

from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import _constrained_rollout_model_batch
from hypertagging.training.reconstruction_trainer import _with_allowed_types
from hypertagging.training.scheduled_sampling import aligned_level_targets

n = 6
truth = {
    "node_mask": torch.ones(1, n, dtype=torch.bool),
    "level_ids": torch.tensor([[0, 0, 0, 0, 1, 2]]),
    "valid_reconstruction_target": torch.tensor([[0, 0, 0, 0, 1, 1]], dtype=torch.bool),
    "recursive_reconstructable_complete": torch.ones(1, n, dtype=torch.bool),
    "daughter_adjacency": torch.zeros(1, n, n, dtype=torch.bool),
    "recursive_leaf_source_mask": torch.tensor(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
          [0, 1, 1, 0], [1, 1, 1, 0]]], dtype=torch.bool
    ),
    "pid_labels": torch.tensor([[11, 11, 11, 11, 4, 2]]),
}
truth["daughter_adjacency"][0, 4, [1, 2]] = True
truth["daughter_adjacency"][0, 5, [0, 4]] = True
state = {
    "node_mask": torch.ones(1, n, dtype=torch.bool),
    "level_ids": torch.tensor([[0, 0, 0, 0, 1, 1]]),
    "parent_ids": torch.tensor([[5, 4, 4, 5, -1, -1]]),
    "daughter_adjacency": torch.zeros(1, n, n, dtype=torch.bool),
    "recursive_leaf_source_mask": torch.tensor(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
          [0, 1, 1, 0], [1, 0, 0, 1]]], dtype=torch.bool
    ),
    "node_kind_ids": torch.tensor(
        [[NODE_KIND_TO_ID["track"]] * 4 + [NODE_KIND_TO_ID["composite"]] * 2]
    ),
    "p4": torch.zeros(1, n, 4),
    "charge": torch.zeros(1, n),
}
state["daughter_adjacency"][0, 4, [1, 2]] = True
state["daughter_adjacency"][0, 5, [0, 3]] = True
policy = ReconstructionConstraintPolicy()
aligned = aligned_level_targets(truth, state, target_level=2)
train = _with_allowed_types(state, 2, {}, policy)["pointer_validity_mask"]
infer = _constrained_rollout_model_batch(
    state, target_level=2, policy=policy
)["pointer_validity_mask"]
print(json.dumps({
    "truth_targets": aligned.truth_target_count,
    "reported_representable": aligned.representable_count,
    "training_target_mask": aligned.target_override[1][0].tolist(),
    "training_pointer_valid": train.tolist(),
    "inference_pointer_valid": infer.tolist(),
}, indent=2))
