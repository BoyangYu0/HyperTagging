from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.models.level_autoregressive import (
    LevelAutoregressiveReconstructor,
    _upgrade_flat_batch,
)
from hypertagging.models.relation_attention import RelationAwareSetLayer
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    append_composite_proposals,
    batched_level_step,
)
from hypertagging.training.model_config import ModelArchitecture, resolve_model_architecture
from hypertagging.training.reconstruction_trainer import (
    ReconstructionConfig,
    _architecture_contract,
)
from hypertagging.training.pretrain_trainer import ChannelMemoryBank
from hypertagging.training.pretrain_trainer import (
    PRINCIPAL_PRETRAINING_OBJECTIVES,
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
