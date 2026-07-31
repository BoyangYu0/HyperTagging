import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.reconstruction.pid_state import (
    COMPOSITE_TYPE_SOURCE_TO_ID,
    rebuild_runtime_pid_state,
)


def test_teacher_and_predicted_composites_share_current_type_semantics():
    batch = collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    logits = torch.zeros((*batch["pid_labels"].shape, 41))
    composite = batch["level_ids"] > 0
    token = batch["pid_target_labels"].clone()
    batch["runtime_composite_type_source_ids"] = torch.where(
        composite,
        torch.full_like(token, COMPOSITE_TYPE_SOURCE_TO_ID["truth_teacher_forced"]),
        torch.full_like(token, COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"]),
    )
    teacher = rebuild_runtime_pid_state(batch, logits)
    predicted_batch = dict(batch)
    predicted_batch["runtime_composite_type_source_ids"] = torch.where(
        composite,
        torch.full_like(token, COMPOSITE_TYPE_SOURCE_TO_ID["predicted"]),
        batch["runtime_composite_type_source_ids"],
    )
    predicted_batch["current_pid_tokens"] = token
    predicted = rebuild_runtime_pid_state(predicted_batch, logits)
    torch.testing.assert_close(teacher.probabilities[composite], predicted.probabilities[composite])
    torch.testing.assert_close(
        teacher.daughter_input_histograms, predicted.daughter_input_histograms
    )

