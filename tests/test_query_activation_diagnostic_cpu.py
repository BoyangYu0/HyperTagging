from dataclasses import replace

import pytest
import torch

from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_from_level_event,
)
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.evaluation.query_activation import (
    gradient_reachability_diagnostics,
    inference_decision_diagnostics,
    matching_and_margin_diagnostics,
    per_query_probability_distributions,
    require_diagnostic_role,
    require_finite_json,
)
from hypertagging.losses.level_reconstruction import (
    focal_binary_cross_entropy_with_logits,
    level_reconstruction_loss,
)
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.models.mother_pointer import MotherPointerOutput
from hypertagging.reconstruction.level_rollout import RolloutConfig
from hypertagging.training.checkpoint_selection import rollout_checkpoint_eligibility


def _batch():
    return collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[1])]
    )


def test_read_only_diagnostic_probability_and_inference_counts_are_exact():
    torch.manual_seed(4)
    batch = _batch()
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=41,
        hidden_dim=16,
        hyper_dim=4,
        n_queries=3,
        n_context_layers=1,
        max_cardinality=3,
    ).eval()
    output = model(batch, target_level=1)
    pointer = MotherPointerOutput(
        object_logits=torch.tensor([[-2.0, -1.0, -0.5]]),
        type_logits=output.pointer.type_logits,
        pointer_logits=torch.full_like(output.pointer.pointer_logits, 2.0),
        cardinality_logits=output.pointer.cardinality_logits,
        confidence_logits=output.pointer.confidence_logits,
    )
    diagnostic_output = replace(output, pointer=pointer)
    distributions = per_query_probability_distributions(pointer.object_logits)
    assert len(distributions) == 3
    assert distributions[0]["null_probability"]["mean"] == pytest.approx(
        1.0 - torch.sigmoid(torch.tensor(-2.0)).item()
    )
    decision = inference_decision_diagnostics(
        diagnostic_output, batch, RolloutConfig(root_types=())
    )
    assert decision["object_active_query_count_before_decision"] == 0
    assert decision["active_query_count_after_unchanged_inference_decision"] == 0
    assert decision["predicted_node_count_before_pruning"] == 0
    expected_stop = float((1.0 - torch.sigmoid(pointer.object_logits)).prod())
    assert decision["derived_stop_probability"] == pytest.approx(expected_stop)
    assert not decision["depth_continue_head_present"]


def test_gradient_diagnostic_reaches_queries_objectness_and_pointer_heads():
    torch.manual_seed(5)
    batch = _batch()
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=41,
        hidden_dim=16,
        hyper_dim=4,
        n_queries=8,
        n_context_layers=1,
    )
    output = model(batch, target_level=1)
    loss = level_reconstruction_loss(
        output.pointer, batch, target_level=1, matching_production=False
    ).total
    report = gradient_reachability_diagnostics(model, loss, target_level=1)
    assert report["query_embeddings"]["gradient_reaches_parameters"]
    assert report["objectness_head"]["gradient_reaches_parameters"]
    assert report["pointer_heads"]["gradient_reaches_parameters"]
    assert report["depth_continue_head"] == {
        "present": False,
        "gradient_reaches_parameters": False,
        "reason": "architecture derives continuation from query decisions",
    }
    matching = matching_and_margin_diagnostics(
        output,
        batch,
        level_reconstruction_loss(
            output.pointer, batch, target_level=1, matching_production=False
        ).matches,
        target_level=1,
        target_policy="complete_only",
    )
    assert matching["matched_query_count"] == matching["truth_target_count"]
    assert matching["unmatched_query_count"] > 0
    assert matching["matched_cost"]["count"] == matching["matched_query_count"]


def test_diagnostic_is_finite_and_rejects_sealed_roles():
    require_diagnostic_role(
        "validation", split_counts={"train": 3, "validation": 2, "test": 0}
    )
    with pytest.raises(ValueError, match="train/validation"):
        require_diagnostic_role(
            "test", split_counts={"train": 3, "validation": 2, "test": 0}
        )
    with pytest.raises(ValueError, match="sealed-test"):
        require_diagnostic_role(
            "validation", split_counts={"train": 3, "validation": 2, "test": 1}
        )
    require_finite_json({"schema_version": "v1", "value": 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        require_finite_json({"value": float("nan")})


def test_query_activation_balance_does_not_change_primary_gate_semantics():
    logits_default = torch.zeros(32, requires_grad=True)
    logits_balanced = torch.zeros(32, requires_grad=True)
    targets = torch.zeros(32)
    targets[:2] = 1
    default = focal_binary_cross_entropy_with_logits(
        logits_default, targets, positive_weight=2.0, gamma=1.0
    )
    balanced = focal_binary_cross_entropy_with_logits(
        logits_balanced, targets, positive_weight=16.0, gamma=1.0
    )
    default.backward()
    balanced.backward()
    assert abs(float(logits_balanced.grad[0])) > abs(float(logits_default.grad[0]))

    gates = {
        "minimum_tree_validity": 0.999,
        "minimum_p4_closure": 1.0,
        "maximum_recursive_source_conflicts": 0,
        "required_denominators": [
            "rollout_validation_events",
            "predicted_edge_denominator",
            "predicted_p4_closure_denominator",
        ],
    }
    collapsed = {
        "rollout_validation_events": 256,
        "predicted_edge_denominator": 2774,
        "predicted_p4_closure_denominator": 0,
        "predicted_tree_validity_rate": 1.0,
        "predicted_p4_closure_rate": 1.0,
        "predicted_recursive_source_conflicts": 0,
    }
    result = rollout_checkpoint_eligibility(collapsed, gates)
    assert not result["eligible"]
    assert result["failures"] == [
        "nonzero_denominator:predicted_p4_closure_denominator"
    ]
