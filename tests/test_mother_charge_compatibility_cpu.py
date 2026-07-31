from hypertagging.preprocessing.pid_filter import TOKENIZE_DICT
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import RolloutConfig, hard_decode_proposals
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from types import SimpleNamespace
import torch


def test_reduced_token_charge_lookup_is_explicit_for_neutral_and_charged_mothers():
    policy = ReconstructionConstraintPolicy()
    assert policy.expected_charge(TOKENIZE_DICT[521]) == 1.0
    assert policy.expected_charge(TOKENIZE_DICT[-521]) == -1.0
    assert policy.expected_charge(TOKENIZE_DICT[511]) == 0.0


def test_hard_rollout_rejects_charge_incompatible_mother():
    neutral = TOKENIZE_DICT[511]
    type_logits = torch.full((1, 1, 41), -10.0)
    type_logits[0, 0, neutral] = 10.0
    pointer = MotherPointerOutput(
        object_logits=torch.tensor([[10.0]]), type_logits=type_logits,
        pointer_logits=torch.tensor([[[10.0, 10.0]]]),
        cardinality_logits=torch.tensor([[[-10.0, -10.0, 10.0]]]),
        confidence_logits=torch.tensor([[10.0]]),
    )
    output = SimpleNamespace(
        pointer=pointer, context_mask=torch.tensor([[True, True]]), target_level=1
    )
    batch = {
        "node_mask": torch.tensor([[True, True]]),
        "level_ids": torch.tensor([[0, 0]]),
        "node_kind_ids": torch.tensor([[NODE_KIND_TO_ID["track"]] * 2]),
        "leaf_kinematics_mode_ids": torch.tensor(
            [[LEAF_MODE_TO_ID["fixed_hypothesis_candidate"]] * 2]
        ),
        "charge": torch.tensor([[1.0, 0.0]]),
    }
    policy = ReconstructionConstraintPolicy(mother_charge_compatibility="hard")
    proposals = hard_decode_proposals(
        output, batch,
        RolloutConfig(object_threshold=0.5, constraint_policy=policy),
    )
    assert proposals == []
