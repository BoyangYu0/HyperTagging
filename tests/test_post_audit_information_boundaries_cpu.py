from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hypertagging.data.capacity import production_capacity_report
from hypertagging.data.dataset_index import build_dataset_index
from hypertagging.data.heterogeneous import (
    collate_heterogeneous_events,
    heterogeneous_event_from_record,
    heterogeneous_from_level_event,
)
from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    _upgrade_flat_batch,
    context_only_relation_summary,
)
from hypertagging.preprocessing.basf2_mdst import (
    Basf2PreprocessConfig,
    TRACK_FIT_POLICY_MAX_P_VALUE_V1,
    _DirectMdstCollector,
    _select_data_independent_track_fit,
    _track_fit_policy_diagnostics,
)
from hypertagging.preprocessing.mdst_tree_builder import FourVector, RecoRecord, build_truth_guided_tree
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, PidFilter
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import (
    KLM_FEATURE_NAMES,
    iter_event_records_v4,
)
from hypertagging.training.pretraining_curriculum import PretrainingStage, build_curriculum_batch


def _batch() -> dict[str, torch.Tensor]:
    return _upgrade_flat_batch(
        collate_level_events([tiny_level_events()[1]], max_query_slots=8).to_dict()
    )


def _model() -> LevelAutoregressiveReconstructor:
    torch.manual_seed(29)
    return LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=3,
        n_heads=2,
        n_context_layers=1,
        max_cardinality=6,
        type_conditioned_daughter_relation_bias=True,
        use_hyperbolic_relation_refinement=True,
    ).eval()


def _perturb_future(batch: dict[str, torch.Tensor], target_level: int) -> dict[str, torch.Tensor]:
    changed = {name: value.clone() for name, value in batch.items()}
    future = changed["node_mask"] & (changed["level_ids"] >= target_level)
    torch.manual_seed(100 + target_level)
    for name in (
        "p4", "common_features", "track_features", "cluster_features",
        "klm_features", "composite_features",
    ):
        changed[name][future] = torch.randn_like(changed[name][future]) * 7.0
    changed["charge"][future] = torch.randn_like(changed["charge"][future]) * 3.0
    changed["pid_labels"][future] = (
        changed["pid_labels"][future] + 7
    ) % len(PDG_TOKENS)
    changed["pid_target_labels"][future] = (
        changed["pid_target_labels"][future] + 11
    ) % len(PDG_TOKENS)
    if "truth_pid_labels" in changed:
        changed["truth_pid_labels"][future] = (
            changed["truth_pid_labels"][future] + 13
        ) % len(PDG_TOKENS)
    changed["node_kind_ids"][future] = NODE_KIND_TO_ID["klm_cluster"]
    changed["level_ids"][future] = target_level + 4
    changed["active"][future] ^= True
    changed["copied"][future] ^= True
    for name in (
        "common_availability", "track_availability", "cluster_availability",
        "klm_availability", "composite_availability",
    ):
        changed[name][future] ^= True
    changed["daughter_input_pid_histogram"][future] = torch.randn_like(
        changed["daughter_input_pid_histogram"][future]
    )
    changed["daughter_input_pid_histogram_available"][future] ^= True
    for name in ("source_node_ids", "reco_ids"):
        if name in changed:
            changed[name][future] += 10_000
    future_pairs = future[:, :, None] | future[:, None, :]
    changed["daughter_adjacency"][future_pairs] ^= True
    changed["ancestor_descendant_relation"][future_pairs] ^= True
    if "source_conflict_matrix" in changed:
        changed["source_conflict_matrix"][future_pairs] ^= True
    changed["parent_ids"][future] = -1
    if "recursive_leaf_source_mask" in changed:
        changed["recursive_leaf_source_mask"][future] ^= True
    for name in (
        "truth_pid_available", "valid_reconstruction_target",
        "complete_truth_decay", "complete_reconstructable_decay",
        "recursive_reconstructable_complete", "partial_missing_daughters",
    ):
        if name in changed:
            changed[name][future] ^= True
    return changed


def _assert_decoder_equal(left, right, node_count: int | None = None) -> None:
    for name in (
        "object_logits", "type_logits", "cardinality_logits", "confidence_logits"
    ):
        torch.testing.assert_close(getattr(left.pointer, name), getattr(right.pointer, name), atol=1e-6, rtol=1e-6)
    left_pointer = left.pointer.pointer_logits
    right_pointer = right.pointer.pointer_logits
    if node_count is not None:
        left_pointer = left_pointer[..., :node_count]
        right_pointer = right_pointer[..., :node_count]
    torch.testing.assert_close(left_pointer, right_pointer, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("target_level", [1, 2])
def test_every_target_level_decoder_output_is_future_invariant(target_level: int):
    batch = _batch()
    model = _model()
    baseline = model(batch, target_level=target_level)
    perturbed = model(_perturb_future(batch, target_level), target_level=target_level)
    _assert_decoder_equal(baseline, perturbed)


@pytest.mark.parametrize("target_level", [1, 2])
def test_future_node_removal_matches_padding_masking(target_level: int):
    batch = _batch()
    model = _model()
    full = model(batch, target_level=target_level)
    keep = (batch["node_mask"][0] & (batch["level_ids"][0] < target_level)).nonzero(as_tuple=False).flatten()
    node_count = batch["node_mask"].shape[1]
    reduced: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if value.ndim >= 3 and value.shape[1:3] == (node_count, node_count):
            reduced[name] = value[:, keep][:, :, keep]
        elif value.ndim >= 2 and value.shape[1] == node_count:
            reduced[name] = value[:, keep]
        else:
            reduced[name] = value
    reduced["parent_ids"] = torch.full_like(reduced["parent_ids"], -1)
    removed = model(reduced, target_level=target_level)
    _assert_decoder_equal(full, removed, node_count=keep.numel())


def test_context_only_relation_summary_changes_logits_and_has_gradients():
    batch = _batch()
    model = _model().train()
    enabled = model(batch, target_level=2)
    context = enabled.context_mask
    summary = context_only_relation_summary(enabled.relation_bias, context)
    assert torch.count_nonzero(summary[~context]) == 0
    # ModuleDict has no stable ``get`` API across supported torch versions.
    decoder = model.level_decoders["2"] if "2" in model.level_decoders else model.decoder
    decoder_arguments = dict(
        target_level=2,
        node_pid_probabilities=enabled.current_pid_probabilities,
        node_charge=batch["charge"],
        node_kind_ids=batch["node_kind_ids"],
        node_level_ids=batch["level_ids"],
    )
    with_summary = decoder(
        enabled.reconstruction_projection,
        context,
        node_relation_summary=summary,
        **decoder_arguments,
    )
    without_summary = decoder(
        enabled.reconstruction_projection,
        context,
        node_relation_summary=torch.zeros_like(summary),
        **decoder_arguments,
    )
    assert not torch.allclose(
        with_summary.pointer_logits, without_summary.pointer_logits
    )
    valid = context[:, None, :].expand_as(enabled.pointer.pointer_logits)
    enabled.pointer.pointer_logits[valid].sum().backward()
    gradients = [
        parameter.grad for parameter in model.encoder.physical_relation_bias.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_curriculum_structural_provenance_keeps_targets_out_by_default():
    batch = _batch()
    fsp = build_curriculum_batch(batch, PretrainingStage.FSP_ONLY)
    truth_default = build_curriculum_batch(batch, PretrainingStage.TRUTH_GUIDED_MULTILEVEL)
    truth_compat = build_curriculum_batch(
        batch,
        PretrainingStage.TRUTH_GUIDED_MULTILEVEL,
        truth_guided_structural_relation_inputs=True,
    )
    assert fsp.relation_input_policy == "inference_physical_relation_features"
    assert truth_default.relation_input_policy == "inference_physical_relation_features"
    assert not truth_default.batch["current_reconstructed_ancestor_descendant_relation"].any()
    assert truth_compat.relation_input_policy == "current_reconstructed_tree_state_features"
    assert truth_compat.batch["current_reconstructed_ancestor_descendant_relation"].any()

    encoder = HeterogeneousNodeEncoder(d_model=16, hyper_dim=4, n_heads=2, n_context_layers=1).eval()
    first = encoder(truth_default.batch, attention_mask=truth_default.batch["curriculum_attention_mask"])
    changed = {name: value.clone() for name, value in truth_default.batch.items()}
    changed["parent_ids"].fill_(-1)
    changed["ancestor_descendant_relation"] ^= True
    second = encoder(changed, attention_mask=changed["curriculum_attention_mask"])
    torch.testing.assert_close(first.node_embeddings, second.node_embeddings)
    torch.testing.assert_close(first.physical_relation_bias, second.physical_relation_bias)


def test_klm_features_are_masked_model_inputs_and_receive_gradients():
    event = heterogeneous_from_level_event(tiny_level_events()[0])
    kinds = event.node_kind_ids.clone()
    kinds[0] = NODE_KIND_TO_ID["klm_cluster"]
    values = event.klm_features.clone()
    availability = event.klm_availability.clone()
    values[0, KLM_FEATURE_NAMES.index("layers")] = 8.0
    availability[0, KLM_FEATURE_NAMES.index("layers")] = True
    batch = collate_heterogeneous_events([
        replace(event, node_kind_ids=kinds, klm_features=values, klm_availability=availability)
    ])
    encoder = HeterogeneousNodeEncoder(d_model=16, hyper_dim=4, n_heads=2, n_context_layers=1)
    output = encoder(batch)
    output.node_embeddings.square().sum().backward()
    assert encoder.klm_encoder.projection[0].weight.grad is not None
    changed = {name: value.clone() for name, value in batch.items()}
    changed["klm_features"][0, 0, KLM_FEATURE_NAMES.index("layers")] = 2.0
    assert not torch.allclose(output.adapter_embeddings, encoder(changed).adapter_embeddings)


def test_native_v4_klm_block_and_older_missing_block_are_both_readable(tmp_path: Path):
    path = write_notebook_fixture_v4(tmp_path / "fixture.parquet")
    record = next(iter_event_records_v4(path))
    old = heterogeneous_event_from_record(record)
    assert not old.klm_availability.any()
    record["nodes"][0]["klm_features"] = {"layers": 7.0}
    record["nodes"][0]["klm_availability"] = {"layers": True}
    upgraded = heterogeneous_event_from_record(record)
    position = KLM_FEATURE_NAMES.index("layers")
    assert (upgraded.klm_features[:, position] == 7.0).sum() == 1
    assert upgraded.klm_availability[:, position].sum() == 1


def test_ecl_klm_association_becomes_recursive_source_conflict():
    ecl_id = "ECLCluster:4"
    records = [
        RecoRecord(ecl_id, 22, 0.0, FourVector(0.1, 0.0, 0.0, 0.1), node_kind="ecl_cluster"),
        RecoRecord(
            "KLMCluster:9", 130, 0.0, FourVector(0.2, 0.0, 0.0, 0.3),
            node_kind="klm_cluster", associated_reco_id=ecl_id,
        ),
    ]
    tree = build_truth_guided_tree(
        event_id=1, mc_records=[], reco_records=records, pid_filter=PidFilter()
    )
    leaves = sorted(tree.nodes.values(), key=lambda node: node.reco_id or "")
    assert set(leaves[0].recursive_leaf_source_ids) & set(leaves[1].recursive_leaf_source_ids) == {ecl_id}


def test_track_fit_policy_is_versioned_and_unknown_values_fail(tmp_path: Path):
    config = Basf2PreprocessConfig(("input.root",), tmp_path / "out.parquet")
    assert config.track_fit_policy == TRACK_FIT_POLICY_MAX_P_VALUE_V1
    with pytest.raises(ValueError, match="unknown track_fit_policy"):
        Basf2PreprocessConfig(("input.root",), tmp_path / "out.parquet", track_fit_policy="truth_best")
    with pytest.raises(ValueError, match="unknown track fit policy"):
        _select_data_independent_track_fit(object(), pion_hypothesis="pion", policy="truth_best")

    class Writer:
        def __init__(self): self.metadata = {}
        def close(self): return tmp_path / "out.parquet"

    collector = _DirectMdstCollector(config)
    collector._v4_writer = Writer()
    collector._v4_pid_filter = PidFilter()
    collector.write_output()
    assert (
        collector._v4_writer.metadata["preprocessing_configuration"][
            "track_fit_policy"
        ]
        == TRACK_FIT_POLICY_MAX_P_VALUE_V1
    )


def test_track_fit_policy_diagnostic_records_reconstructed_momentum_delta():
    class Momentum:
        def __init__(self, xyz): self.xyz = xyz
        def X(self): return self.xyz[0]
        def Y(self): return self.xyz[1]
        def Z(self): return self.xyz[2]

    class Fit:
        def __init__(self, p_value, xyz): self.p_value, self.xyz = p_value, xyz
        def getPValue(self): return self.p_value
        def getMomentum(self): return Momentum(self.xyz)

    selected_fit = Fit(0.9, (1.0, 2.0, 3.0))
    pion_fit = Fit(0.2, (0.5, 1.5, 2.5))

    class Track:
        def getTrackFitResults(self): return [("kaon", selected_fit)]
        def getTrackFitResultWithClosestMass(self, _hypothesis): return pion_fit

    track = Track()
    selected = _select_data_independent_track_fit(track, pion_hypothesis="pion")
    report = _track_fit_policy_diagnostics(
        track, selected=selected, pion_hypothesis="pion"
    )
    assert report["policy"] == TRACK_FIT_POLICY_MAX_P_VALUE_V1
    assert report["pion_comparison_available"] is True
    assert report["delta_px"] == pytest.approx(0.5)
    assert report["delta_momentum_magnitude"] != 0.0


def test_capacity_report_includes_quantiles_and_representative_slices():
    index = {
        "target_policy": "complete_only",
        "mother_count_histograms_by_level": {"1": {"1": 2, "3": 1}},
        "daughter_cardinality_histograms_by_level": {"1": {"2": 2, "5": 1}},
        "capacity_slices_by_level": {
            "1": {"source_category": {"charged_B": {"event_count": 3, "maximum_mothers": 3, "maximum_daughter_cardinality": 5}}}
        },
        "channel_frequency_histogram": {"1": 2, "3": 1},
    }
    report = production_capacity_report(index, global_n_queries=4, global_max_cardinality=6)
    assert report["report_version"] == "hypertagging-capacity-report-v2"
    assert report["levels"][0]["mother_count_quantiles"]["p99"] == 3.0
    assert "source_category" in report["levels"][0]["representative_slices"]
    assert report["channel_frequency_histogram"] == {"1": 2, "3": 1}


def test_dataset_index_channel_frequency_slice_is_bounded_and_explicit(tmp_path: Path):
    import json

    fixture = write_notebook_fixture_v4(tmp_path / "capacity.parquet")
    index_path = build_dataset_index(
        [fixture], tmp_path / "index.json", target_policy="diagnostic_all"
    )
    index = json.loads(index_path.read_text())
    assert index["channel_frequency_slice_coverage"]["exact"] is True
    assert index["channel_frequency_slice_coverage"]["tracked_signatures"] > 0
    assert all(
        "channel_frequency" in dimensions
        for dimensions in index["capacity_slices_by_level"].values()
    )
