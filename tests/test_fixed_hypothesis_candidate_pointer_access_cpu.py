import torch

from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID


def test_unknown_kind_fixed_hypothesis_candidate_remains_pointer_accessible():
    batch = {
        "node_mask": torch.tensor([[True, True]]),
        "level_ids": torch.tensor([[0, 0]]),
        "node_kind_ids": torch.tensor([[0, 0]]),
        "leaf_kinematics_mode_ids": torch.tensor([[
            LEAF_MODE_TO_ID["fixed_hypothesis_candidate"],
            LEAF_MODE_TO_ID["truth_topology_only"],
        ]]),
    }
    mask = ReconstructionConstraintPolicy().pointer_validity_mask(batch, 1)
    assert mask.tolist() == [[True, False]]

