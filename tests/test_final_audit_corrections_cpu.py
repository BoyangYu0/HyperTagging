from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.data.heterogeneous import (
    MODEL_INPUT_SOURCE_TO_ID,
    TRUTH_SUPERVISION_SOURCE_TO_ID,
)
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    _upgrade_flat_batch,
)
from hypertagging.models.relation_attention import RelationAwareSetLayer
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import (
    BatchedRolloutState,
    CompositeProposal,
    append_composite_proposals,
    batched_level_step,
    batched_rollout_level_transition,
)
from hypertagging.training.model_config import ModelArchitecture, resolve_model_architecture
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    _architecture_contract,
)
from hypertagging.training.pretrain_trainer import ChannelMemoryBank
from hypertagging.training.pretrain_trainer import (
    PRINCIPAL_PRETRAINING_OBJECTIVES,
    PretrainConfig,
    _pretraining_weights,
    objective_gradient_diagnostics,
)


def _leaf_state(event_index: int = 0) -> dict[str, torch.Tensor]:
    batch = collate_level_events(
        [tiny_level_events()[event_index]], max_query_slots=4
    ).to_dict()
    upgraded = _upgrade_flat_batch(batch)
    keep = upgraded["node_mask"][0] & (upgraded["level_ids"][0] == 0)
    indices = keep.nonzero(as_tuple=False).flatten()
    count = upgraded["node_mask"].shape[1]
    result: dict[str, torch.Tensor] = {}
    for name, value in upgraded.items():
        if value.ndim >= 3 and value.shape[1:3] == (count, count):
            result[name] = value[:, indices][:, :, indices]
        elif value.ndim >= 2 and value.shape[1] == count:
            result[name] = value[:, indices]
        else:
            result[name] = value
    result["parent_ids"] = torch.full_like(result["parent_ids"], -1)
    result.setdefault("truth_pid_labels", result["pid_target_labels"].clone())
    result.setdefault("truth_pid_available", result["node_mask"].clone())
    return result


def _model(**overrides) -> LevelAutoregressiveReconstructor:
    values = dict(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
        n_heads=2,
        n_context_layers=1,
        max_cardinality=4,
        dropout=0.0,
    )
    values.update(overrides)
    return LevelAutoregressiveReconstructor(**values)


def test_attention_dropout_is_propagated_and_train_eval_behavior_is_deterministic():
    layer = RelationAwareSetLayer(8, n_heads=2, dropout=0.6)
    assert layer.attention.dropout == 0.6
    values = torch.randn(1, 4, 8)
    relation = torch.zeros(1, 4, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    layer.eval()
    first = layer(values, relation_bias=relation, attention_mask=mask[:, :, None] & mask[:, None], node_mask=mask)[0]
    second = layer(values, relation_bias=relation, attention_mask=mask[:, :, None] & mask[:, None], node_mask=mask)[0]
    torch.testing.assert_close(first, second)
    layer.train()
    torch.manual_seed(1)
    first = layer(values, relation_bias=relation, attention_mask=mask[:, :, None] & mask[:, None], node_mask=mask)[0]
    torch.manual_seed(2)
    second = layer(values, relation_bias=relation, attention_mask=mask[:, :, None] & mask[:, None], node_mask=mask)[0]
    assert not torch.allclose(first, second)


def test_physical_and_hyperbolic_attention_stages_are_exposed_separately():
    batch = _leaf_state()
    encoder = HeterogeneousNodeEncoder(
        d_model=16,
        hyper_dim=4,
        n_heads=2,
        n_context_layers=1,
        use_hyperbolic_refinement=True,
    ).eval()
    first = encoder(batch, return_attention=True)
    assert first.physical_attention_weights is not None
    assert first.hyperbolic_attention_weights is not None
    torch.testing.assert_close(first.final_contextual_embeddings, first.node_embeddings)

    changed_physical = {name: value.clone() for name, value in batch.items()}
    changed_physical["charge"] = changed_physical["charge"] + torch.tensor(
        [[2.0, -1.0]], dtype=changed_physical["charge"].dtype
    )
    second = encoder(changed_physical, return_attention=True)
    assert not torch.allclose(
        first.physical_attention_weights, second.physical_attention_weights
    )

    with torch.no_grad():
        encoder.hyper_projection.weight.add_(
            3.0 * torch.randn_like(encoder.hyper_projection.weight)
        )
    geometry_changed = encoder(batch, return_attention=True)
    torch.testing.assert_close(
        first.physical_attention_weights,
        geometry_changed.physical_attention_weights,
    )
    assert not torch.allclose(
        first.hyperbolic_attention_weights,
        geometry_changed.hyperbolic_attention_weights,
    )

    disabled = HeterogeneousNodeEncoder(
        d_model=16,
        hyper_dim=4,
        n_heads=2,
        n_context_layers=1,
        use_hyperbolic_refinement=False,
    ).eval()(batch, return_attention=True)
    assert disabled.hyperbolic_attention_weights is None


def test_both_attention_stages_respect_padding_and_stair_masks():
    batch = _leaf_state()
    padded = {name: value.clone() for name, value in batch.items()}
    padded["node_mask"][0, -1] = False
    padded["active"][0, -1] = False
    mask = padded["node_mask"][:, :, None] & padded["node_mask"][:, None, :]
    encoder = HeterogeneousNodeEncoder(
        d_model=16,
        hyper_dim=4,
        n_heads=2,
        n_context_layers=1,
        use_hyperbolic_refinement=True,
    ).eval()
    output = encoder(padded, attention_mask=mask, return_attention=True)
    assert output.physical_attention_weights is not None
    assert output.hyperbolic_attention_weights is not None
    for weights in (
        output.physical_attention_weights,
        output.hyperbolic_attention_weights,
    ):
        assert torch.count_nonzero(weights[..., -1]) == 0
        assert torch.count_nonzero(weights[..., -1, :]) == 0


def test_type_conditioned_query_node_bias_changes_logits_and_receives_gradients():
    torch.manual_seed(4)
    model = _model(type_conditioned_daughter_relation_bias=True)
    batch = _leaf_state()
    enabled = model(batch, target_level=1)
    assert enabled.pointer.query_node_compatibility_bias is not None
    decoder = model.decoder
    decoder.type_conditioned_daughter_relation_bias = False
    disabled = model(batch, target_level=1)
    decoder.type_conditioned_daughter_relation_bias = True
    assert not torch.allclose(
        enabled.pointer.pointer_logits, disabled.pointer.pointer_logits
    )
    enabled.pointer.pointer_logits[enabled.context_mask[:, None].expand_as(enabled.pointer.pointer_logits)].sum().backward()
    assert decoder.type_relation_table is not None
    assert decoder.type_relation_table.grad is not None
    assert decoder.type_relation_table.grad.abs().sum() > 0


def test_first_level_yaml_is_consumed_and_architecture_round_trips():
    from scripts.train_level_reconstruction import parse_args

    path = Path("configs/ablations/first_level_type_relation_bias.yaml")
    parsed = parse_args(["--config", str(path), "--dry-run", "--tiny"])
    assert parsed.ablation == "first_level_type_relation_bias"
    assert parsed.type_conditioned_daughter_relation_bias is True
    architecture = resolve_model_architecture(
        "tiny_cpu", type_conditioned_daughter_relation_bias=True
    )
    assert ModelArchitecture.from_dict(architecture.to_dict()) == architecture
    contract = _architecture_contract(
        ReconstructionConfig(
            data="unused",
            output_dir="unused",
            ablation="first_level_type_relation_bias",
        )
    )
    assert contract["type_conditioned_daughter_relation_bias"] is True


def test_deferred_first_level_fields_have_no_runnable_config_and_unknown_keys_fail(tmp_path):
    from scripts.train_level_reconstruction import parse_args

    assert not Path("configs/ablations/first_level_whole_set_scorer.yaml").exists()
    assert not Path("configs/ablations/first_level_iterative_pointer.yaml").exists()
    invalid = tmp_path / "ignored.yaml"
    invalid.write_text("whole_set_compatibility_scorer: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown YAML configuration key"):
        parse_args(["--config", str(invalid), "--dry-run", "--tiny"])


def test_bounded_tangent_level_embedding_is_distinct_ordered_and_differentiable():
    batch = _leaf_state()
    torch.manual_seed(9)
    euclidean = _model(hyperbolic_level_encoding="learned_euclidean")
    torch.manual_seed(9)
    tangent = _model(
        hyperbolic_level_encoding="bounded_tangent_level_embedding"
    )
    euclidean_output = euclidean(batch, target_level=1)
    tangent_output = tangent(batch, target_level=1)
    assert not torch.allclose(
        euclidean_output.node_embeddings, tangent_output.node_embeddings
    )
    tangent_output.node_embeddings.square().sum().backward()
    encoder = tangent.encoder
    assert isinstance(encoder, HeterogeneousNodeEncoder)
    assert encoder.tangent_level_directions.grad is not None
    assert encoder.tangent_level_gaps.grad is not None
    gaps = torch.nn.functional.softplus(encoder.tangent_level_gaps.detach())
    radii = torch.flip(torch.cumsum(torch.flip(gaps, (0,)), 0), (0,))
    assert torch.all(radii[:-1] > radii[1:])
    assert radii[0] > radii[-1]  # leaves remain outside higher/root levels


def test_bounded_tangent_active_outputs_are_padding_invariant():
    event = tiny_level_events()[0]
    single = _upgrade_flat_batch(
        collate_level_events([event], max_query_slots=4).to_dict()
    )
    padded = _upgrade_flat_batch(
        collate_level_events([event, tiny_level_events()[1]], max_query_slots=4).to_dict()
    )
    model = _model(hyperbolic_level_encoding="bounded_tangent_level_embedding")
    model.eval()
    single_output = model(single, target_level=1)
    padded_output = model(padded, target_level=1)
    active = single["node_mask"][0]
    torch.testing.assert_close(
        single_output.node_embeddings[0, active],
        padded_output.node_embeddings[0, : active.numel()][active],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("clone_truth_first", [False, True])
def test_truth_only_perturbation_cannot_change_inference(clone_truth_first: bool):
    model = _model(type_conditioned_daughter_relation_bias=True)
    model.eval()
    batch = _leaf_state()
    if clone_truth_first:
        batch["daughter_truth_pid_histogram"] = batch[
            "daughter_input_pid_histogram"
        ].clone()
    original = model(batch, target_level=1)
    changed = {name: value.clone() for name, value in batch.items()}
    for name in (
        "pid_target_labels",
        "truth_pid_labels",
        "daughter_truth_pid_histogram",
        "full_truth_daughter_count",
        "retained_truth_daughter_count_expected",
        "truth_root_distance",
    ):
        if name in changed:
            changed[name] = torch.remainder(changed[name] + 7, len(PDG_TOKENS))
    for name in (
        "truth_pid_available",
        "daughter_truth_pid_histogram_available",
        "complete_truth_decay",
    ):
        if name in changed:
            changed[name] = ~changed[name]
    for name in (
        "b_channel_count_arrays",
        "b_depth_pid_count_arrays",
        "b_branch_multiplicity_summaries",
        "b_intermediate_count_arrays",
    ):
        if name in changed:
            changed[name] = torch.randn_like(changed[name].float())
    perturbed = model(changed, target_level=1)
    for left, right in (
        (original.node_embeddings, perturbed.node_embeddings),
        (original.relation_bias, perturbed.relation_bias),
        (original.pointer.pointer_logits, perturbed.pointer.pointer_logits),
        (original.pointer.type_logits, perturbed.pointer.type_logits),
        (original.leaf_pid_logits, perturbed.leaf_pid_logits),
    ):
        torch.testing.assert_close(left, right)
    base_loss = torch.nn.functional.cross_entropy(
        original.leaf_pid_logits[batch["node_mask"]],
        batch["truth_pid_labels"][batch["node_mask"]],
    )
    changed_loss = torch.nn.functional.cross_entropy(
        perturbed.leaf_pid_logits[changed["node_mask"]],
        changed["truth_pid_labels"][changed["node_mask"]],
    )
    assert not torch.allclose(base_loss, changed_loss)


def test_identical_links_have_identical_next_pass_composite_embeddings():
    model = _model()
    state = _leaf_state()
    proposal = CompositeProposal(0, 4, tuple(range(state["node_mask"].shape[1])), 1.0, 1.0)
    truth_state, _ = append_composite_proposals(state, [proposal], target_level=1)
    predicted_state, _ = append_composite_proposals(state, [proposal], target_level=1)
    truth_output = model(truth_state, target_level=2)
    predicted_output = model(predicted_state, target_level=2)
    torch.testing.assert_close(truth_state["composite_features"], predicted_state["composite_features"])
    torch.testing.assert_close(truth_output.node_embeddings, predicted_output.node_embeddings)


def test_predicted_runtime_nodes_have_no_truth_provenance_or_targets():
    state = _leaf_state()
    proposal = CompositeProposal(
        0, 4, tuple(range(state["node_mask"].shape[1])), 1.0, 1.0
    )
    predicted, _ = append_composite_proposals(state, [proposal], target_level=1)
    position = predicted["node_mask"].shape[1] - 1
    assert predicted["model_input_source_ids"][0, position] == MODEL_INPUT_SOURCE_TO_ID[
        "runtime_reconstructed"
    ]
    assert predicted["truth_supervision_source_ids"][0, position] == (
        TRUTH_SUPERVISION_SOURCE_TO_ID["unavailable"]
    )
    assert not predicted["truth_pid_available"][0, position]
    assert predicted["pid_target_labels"][0, position] == 0
    assert predicted["truth_pid_labels"][0, position] == 0
    assert predicted["full_truth_daughter_count"][0, position] == -1
    assert predicted["retained_truth_daughter_count_expected"][0, position] == -1
    assert not predicted["valid_reconstruction_target"][0, position]
    assert predicted["runtime_structurally_valid"][0, position]


def test_predicted_runtime_truth_fields_cannot_change_subsequent_outputs():
    model = _model(type_conditioned_daughter_relation_bias=True).eval()
    state = _leaf_state()
    proposal = CompositeProposal(
        0, 4, tuple(range(state["node_mask"].shape[1])), 1.0, 1.0
    )
    predicted, _ = append_composite_proposals(state, [proposal], target_level=1)
    changed = {name: value.clone() for name, value in predicted.items()}
    position = changed["node_mask"].shape[1] - 1
    for name in (
        "pid_target_labels",
        "truth_pid_labels",
        "full_truth_daughter_count",
        "retained_truth_daughter_count_expected",
        "truth_root_distance",
        "full_event_max_level",
    ):
        changed[name][0, position] = 37
    changed["daughter_truth_pid_histogram"][0, position] = torch.randn_like(
        changed["daughter_truth_pid_histogram"][0, position]
    )
    changed["truth_pid_available"][0, position] = True
    changed["daughter_truth_pid_histogram_available"][0, position] = True
    changed["complete_truth_decay"][0, position] = True
    original_output = model(predicted, target_level=2)
    changed_output = model(changed, target_level=2)
    for left, right in (
        (original_output.node_embeddings, changed_output.node_embeddings),
        (original_output.relation_bias, changed_output.relation_bias),
        (original_output.pointer.pointer_logits, changed_output.pointer.pointer_logits),
        (original_output.pointer.type_logits, changed_output.pointer.type_logits),
        (original_output.leaf_pid_logits, changed_output.leaf_pid_logits),
    ):
        torch.testing.assert_close(left, right)


def test_free_rollout_never_marks_predicted_nodes_truth_available(monkeypatch):
    import importlib

    rollout_module = importlib.import_module(
        "hypertagging.reconstruction.level_rollout"
    )
    state = _leaf_state()
    proposal = CompositeProposal(
        0, 4, tuple(range(state["node_mask"].shape[1])), 1.0, 1.0
    )
    monkeypatch.setattr(
        rollout_module,
        "hard_decode_proposals",
        lambda output, batch, config: [proposal],
    )
    result = rollout_module.level_rollout(
        _model(),
        state,
        mode="predicted",
        config=rollout_module.RolloutConfig(
            max_level=1, root_types=(), exclusive_final=False
        ),
    )
    predicted = result.batch["model_input_source_ids"] == MODEL_INPUT_SOURCE_TO_ID[
        "runtime_reconstructed"
    ]
    assert predicted.any()
    assert not result.batch["truth_pid_available"][predicted].any()
    assert torch.all(
        result.batch["truth_supervision_source_ids"][predicted]
        == TRUTH_SUPERVISION_SOURCE_TO_ID["unavailable"]
    )


def test_batched_level_step_matches_reference_for_two_events():
    model = _model()
    first = _leaf_state(0)
    second = _leaf_state(0)
    assert first["node_mask"].shape == second["node_mask"].shape
    batched = {
        name: torch.cat([value, second[name]], dim=0)
        if value.ndim > 0 and value.shape[0] == 1 and name in second
        else value
        for name, value in first.items()
    }
    output = model(batched, target_level=1)
    daughter_mask = torch.zeros_like(output.pointer.pointer_logits, dtype=torch.bool)
    daughter_mask[:, 0] = batched["node_mask"]
    accepted = torch.zeros_like(output.pointer.object_logits, dtype=torch.bool)
    accepted[:, 0] = True
    mother_types = torch.zeros_like(output.pointer.object_logits, dtype=torch.long)
    mother_types[:, 0] = 4
    stepped = batched_level_step(
        batched,
        output,
        daughter_mask=daughter_mask,
        accepted_query_mask=accepted,
        mother_types=mother_types,
        target_level=1,
    )
    stepped_next = model(stepped, target_level=2)
    for index, state in enumerate((first, second)):
        proposal = CompositeProposal(
            0, 4, tuple(range(state["node_mask"].shape[1])), 1.0,
                float(torch.sigmoid(output.pointer.confidence_logits[index, 0]).detach()),
        )
        reference, _ = append_composite_proposals(state, [proposal], target_level=1)
        new_position = state["node_mask"].shape[1]
        torch.testing.assert_close(stepped["p4"][index, new_position], reference["p4"][0, -1])
        torch.testing.assert_close(
            stepped["composite_features"][index, new_position],
            reference["composite_features"][0, -1],
        )
        assert torch.equal(
            stepped["daughter_adjacency"][index, new_position, :new_position],
            reference["daughter_adjacency"][0, -1, :-1],
        )
        reference_next = model(reference, target_level=2)
        reference_active = reference["node_mask"][0]
        torch.testing.assert_close(
            stepped_next.node_embeddings[index, : reference_active.numel()][reference_active],
            reference_next.node_embeddings[0, reference_active],
            atol=1e-6,
            rtol=1e-6,
        )


def test_batched_rollout_transition_stops_events_independently():
    first = _leaf_state(0)
    batched = {
        name: torch.cat([value, value.clone()], dim=0)
        if value.ndim > 0 and value.shape[0] == 1
        else value
        for name, value in first.items()
    }
    model = _model()
    output = model(batched, target_level=1)
    daughters = torch.zeros_like(output.pointer.pointer_logits, dtype=torch.bool)
    daughters[0, 0] = batched["node_mask"][0]
    accepted = torch.zeros_like(output.pointer.object_logits, dtype=torch.bool)
    accepted[0, 0] = True
    control = BatchedRolloutState(
        batch=batched,
        active_event_mask=torch.ones(2, dtype=torch.bool),
        stopped_event_mask=torch.zeros(2, dtype=torch.bool),
        levels_completed=torch.zeros(2, dtype=torch.long),
        stop_code=torch.zeros(2, dtype=torch.long),
    )
    transitioned = batched_rollout_level_transition(
        control,
        output,
        daughter_mask=daughters,
        accepted_query_mask=accepted,
        target_level=1,
    )
    assert transitioned.active_event_mask.tolist() == [True, False]
    assert transitioned.stopped_event_mask.tolist() == [False, True]
    assert transitioned.stop_code.tolist() == [0, 1]
    old_count = batched["node_mask"].shape[1]
    assert transitioned.batch["node_mask"][0, old_count:].any()
    assert not transitioned.batch["node_mask"][1, old_count:].any()


def test_channel_memory_ring_buffer_wraps_and_serializes_cursor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = ChannelMemoryBank(3, 2).to(device)
    bank.enqueue(
        torch.arange(10, device=device, dtype=torch.float32).reshape(1, 5, 2),
        torch.ones(1, 5, dtype=torch.bool, device=device),
        torch.arange(1, 6, device=device).reshape(1, 5),
        torch.arange(11, 16, device=device).reshape(1, 5),
    )
    embeddings, full_ids, reco_ids = bank.contents()
    assert embeddings.shape == (3, 2)
    assert full_ids.cpu().tolist() == [3, 4, 5]
    assert reco_ids.cpu().tolist() == [13, 14, 15]
    restored = ChannelMemoryBank(3, 2).to(device)
    restored.load_state_dict(bank.state_dict())
    assert torch.equal(restored.cursor, bank.cursor)
    assert torch.equal(restored.count, bank.count)
    torch.testing.assert_close(restored.contents()[0], embeddings)


def test_objective_gradient_diagnostics_reports_all_objectives_and_groups():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    objectives = {
        name: (index + 1) * parameter.square().sum()
        for index, name in enumerate(PRINCIPAL_PRETRAINING_OBJECTIVES)
    }
    report = objective_gradient_diagnostics(
        objectives,
        {name: (parameter,) for name in (
            "shared_encoder",
            "tree_projection",
            "hyperbolic_projection",
            "reconstruction_projection",
            "channel_projection",
        )},
    )
    assert report["objective_order"] == list(PRINCIPAL_PRETRAINING_OBJECTIVES)
    assert set(report["gradient_norms"]) == {
        "shared_encoder",
        "tree_projection",
        "hyperbolic_projection",
        "reconstruction_projection",
        "channel_projection",
    }
    assert not report["zero_gradient_objectives"]
    assert report["gradient_cosines"]["shared_encoder"]["lca"]["leaf_pid"] == pytest.approx(1.0)


def test_pretraining_objective_weights_are_independent_and_yaml_consumed():
    from scripts.train_hyperbolic_pretrain import parse_args

    args = parse_args(["--config", "configs/hyperbolic_pretrain.yaml", "--dry-run"])
    names = (
        "lca_relation_weight",
        "parent_ranking_weight",
        "exact_tree_distance_weight",
        "radius_depth_weight",
        "channel_weight",
        "variance_weight",
        "covariance_weight",
        "leaf_pid_weight",
        "corruption_class_weight",
        "candidate_correctness_weight",
        "hard_negative_weight",
    )
    assert all(hasattr(args, name) for name in names)
    config = PretrainConfig(
        data="unused",
        output_dir="unused",
        lca_relation_weight=1.1,
        parent_ranking_weight=2.2,
        exact_tree_distance_weight=3.3,
    )
    weights = _pretraining_weights(config)
    assert weights["lca"] == pytest.approx(1.1)
    assert weights["parent"] == pytest.approx(2.2)
    assert weights["tree_distance"] == pytest.approx(3.3)


def test_active_contract_docs_and_generators_have_no_known_stale_versions():
    active_paths = [
        Path("docs/hyperbolic_level_autoregressive_reconstruction.md"),
        Path("docs/heterogeneous_node_encoding.md"),
        Path("scripts/create_leaf_input_pid_notebook.py"),
        Path("scripts/create_dataset_inspection_notebook.py"),
        Path("scripts/create_hyperbolic_inspection_notebook.py"),
        Path("scripts/create_reconstruction_inspection_notebook.py"),
        Path("scripts/create_preprocessing_qa_notebook.py"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in active_paths)
    for stale in (
        "physical-relations-overlap-aware-v2",
        "notebook_fixture_v3",
        "schema-v3 validation",
        "schema-v1/v2/v3 dataset",
    ):
        assert stale not in text
    assert "physical-relations-overlap-aware-v3" in text
    assert "precontext_daughter_pool" in text


def test_schema_v4_writer_avoids_post_basf2_python_string_apis():
    source = Path("src/hypertagging/preprocessing/schema_v4.py").read_text(
        encoding="utf-8"
    )
    assert ".removeprefix(" not in source
    assert ".removesuffix(" not in source
