import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.training.pretrain_trainer import ContextualPretrainingModel
from hypertagging.training.pretraining_curriculum import PretrainingStage, build_curriculum_batch


def test_pretraining_builds_pid_conditioned_second_pass_without_truth_leaf_pid_input():
    batch = collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    leaves = batch["level_ids"] == 0
    batch["leaf_kinematics_mode_ids"][leaves] = LEAF_MODE_TO_ID["raw_track_predicted_pid"]
    batch["pid_labels"][leaves] = 0
    batch["charge"][0, leaves[0].nonzero().flatten()] = torch.tensor([1.0, -1.0])
    curriculum = build_curriculum_batch(batch, PretrainingStage.TRUTH_GUIDED_MULTILEVEL)
    model = ContextualPretrainingModel(d_model=16, hyper_dim=4)
    encoded, logits, second = model.encode_runtime(
        curriculum.batch, attention_mask=curriculum.batch["curriculum_attention_mask"]
    )
    assert second["current_pid_probabilities"].shape == (*batch["pid_labels"].shape, 41)
    assert encoded.hyperbolic_embeddings.shape[-1] == 4
    assert logits.shape[-1] == 41

