from __future__ import annotations

import torch

from hypertagging.evaluation.full_decay_runner import (
    inference_diagnostics,
    serialize_reconstructed_tree,
    summarize_inference_diagnostics,
)
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.reconstruction.hierarchical_inference import (
    FSPInputAudit,
    HierarchicalInferenceResult,
)
from hypertagging.reconstruction.level_rollout import BatchedRolloutResult


def _result() -> HierarchicalInferenceResult:
    p4 = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 2.0],
          [0.0, 0.0, 3.0, 3.0], [1.0, 2.0, 0.0, 3.0]]]
    )
    adjacency = torch.zeros(1, 4, 4, dtype=torch.bool)
    adjacency[0, 3, :2] = True
    sources = torch.zeros(1, 4, 3, dtype=torch.bool)
    sources[0, 0, 0] = True
    sources[0, 1, 1] = True
    sources[0, 2, 2] = True
    sources[0, 3, :2] = True
    batch = {
        "node_mask": torch.ones(1, 4, dtype=torch.bool),
        "daughter_adjacency": adjacency,
        "parent_ids": torch.tensor([[3, 3, -1, -1]]),
        "level_ids": torch.tensor([[0, 0, 0, 1]]),
        "p4": p4,
        "recursive_leaf_source_mask": sources,
        "evaluation_leaf_source_keys": torch.tensor([[100, 200, 300]]),
        "node_kind_ids": torch.tensor(
            [[NODE_KIND_TO_ID["track"]] * 3 + [NODE_KIND_TO_ID["composite"]]]
        ),
        "pid_labels": torch.tensor([[2, 2, 2, 21]]),
        "current_pid_tokens": torch.tensor([[8, 9, 2, 21]]),
        "charge": torch.zeros(1, 4),
        "node_ids": torch.arange(4).reshape(1, 4),
    }
    rollout = BatchedRolloutResult(
        batch=batch,
        levels_completed=torch.tensor([1]),
        stopped_event_mask=torch.tensor([True]),
        root_completed_mask=torch.tensor([False]),
        event_valid_mask=torch.tensor([True]),
        stop_code=torch.tensor([1]),
        accepted_query_masks=(torch.tensor([[True]]),),
        daughter_masks=(torch.tensor([[[True, True, False]]]),),
    )
    audit = FSPInputAudit(
        schema_version="direct-mdst-tree-v4",
        batch_size=1,
        source_node_width=4,
        projected_node_width=3,
        fsp_counts=(3,),
        discarded_active_node_counts=(1,),
        discarded_higher_level_node_counts=(1,),
        track_counts=(3,),
        ecl_cluster_counts=(0,),
        klm_cluster_counts=(0,),
        detector_source_counts=(3,),
        detector_source_conflict_pair_counts=(0,),
        original_fsp_positions=((0, 1, 2),),
        original_fsp_node_ids=((0, 1, 2),),
        original_fsp_reco_ids=((100, 200, 300),),
        original_fsp_source_node_ids=((0, 1, 2),),
        evaluation_fsp_source_keys=((100, 200, 300),),
        removed_truth_target_fields=("truth_pid_labels",),
    )
    forest = torch.tensor([[False, False, True, True]])
    b_roots = torch.tensor([[False, False, False, True]])
    return HierarchicalInferenceResult(
        scope="half",
        rollout=rollout,
        projected_fsp_batch=batch,
        input_audit=audit,
        forest_root_mask=forest,
        b_root_mask=b_roots,
        continuum_root_mask=torch.zeros_like(forest),
        evaluation_slice_root_mask=b_roots,
        evaluation_slice_multiplicity=torch.tensor([1]),
    )


def test_inference_diagnostics_cover_structure_sources_and_closure():
    row = inference_diagnostics(_result())
    assert row["stop_reason"] == "no_valid_new_mother"
    assert row["single_parent"]
    assert row["parent_adjacency_consistent"]
    assert row["acyclic_by_strict_level_order"]
    assert row["input_source_coverage"] == 1.0
    assert row["p4_closure_rate"] == 1.0
    assert row["leftover_input_fsp_count"] == 1


def test_tree_serialization_uses_canonical_fsp_keys():
    tree = serialize_reconstructed_tree(_result())
    assert tree["nodes"][0]["pid_token"] == 8
    mother = tree["nodes"][3]
    assert mother["daughter_positions"] == [0, 1]
    assert mother["fsp_source_keys"] == [100, 200]
    assert mother["is_b_root"]


def test_detector_conflicts_do_not_replace_unique_topology_keys():
    result = _result()
    result.batch["recursive_leaf_source_mask"][0, 1, 0] = True

    row = inference_diagnostics(result)
    tree = serialize_reconstructed_tree(result)

    assert row["input_source_coverage"] == 1.0
    assert not row["recursive_detector_sources_disjoint"]
    assert not row["inference_structurally_valid"]
    assert tree["nodes"][1]["fsp_source_keys"] == [200]
    assert tree["nodes"][1]["detector_resource_indices"] == [0, 1]


def test_disconnected_associated_fsp_is_diagnostic_not_structural_failure():
    result = _result()
    # The leftover root at position 2 shares a detector resource with leaf 0,
    # but no reconstructed mother combines the two objects.
    result.batch["recursive_leaf_source_mask"][0, 2, 0] = True

    row = inference_diagnostics(result)

    assert not row["forest_root_detector_resources_disjoint"]
    assert row["recursive_detector_sources_disjoint"]
    assert row["inference_structurally_valid"]


def test_inference_summary_keeps_event_denominators():
    row = inference_diagnostics(_result())
    summary = summarize_inference_diagnostics([row, row])
    assert summary["event_count"] == 2
    assert summary["single_parent"]["denominator"] == 2
    assert summary["p4_closure"]["denominator"] == 2


def test_p4_closure_is_undefined_without_reconstructed_mothers():
    result = _result()
    result.batch["daughter_adjacency"].zero_()
    result.batch["parent_ids"].fill_(-1)

    row = inference_diagnostics(result)
    summary = summarize_inference_diagnostics([row])

    assert row["p4_closure_denominator"] == 0
    assert row["p4_closure_rate"] is None
    assert not row["p4_closure_eligible"]
    assert summary["p4_closure"]["denominator"] == 0
    assert summary["p4_closure"]["value"] is None
    assert not summary["p4_closure"]["eligible"]
