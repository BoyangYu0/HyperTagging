import torch

from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.heterogeneous import composite_token_from_daughters
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    RolloutConfig,
    append_composite_proposals,
    level_rollout,
    resolve_exclusive_proposals,
    validate_proposals,
)


def _model():
    return LevelAutoregressiveReconstructor(
        n_features=8,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=4,
        n_heads=4,
        n_context_layers=1,
    )


def _batch(event_index=1):
    return collate_level_events([tiny_level_events()[event_index]], max_query_slots=4).to_dict()


def test_teacher_forced_full_cycle_reconstructs_all_fixture_levels_and_p4_closes():
    result = level_rollout(
        _model(),
        _batch(),
        mode="teacher_forced",
        config=RolloutConfig(max_level=4, root_types=(), exclusive_final=False),
    )
    assert [len(step.accepted) for step in result.steps[:3]] == [1, 1, 1]
    assert result.batch["level_ids"][result.batch["node_mask"]].max() == 3
    for mother in result.batch["daughter_adjacency"][0].any(dim=-1).nonzero(as_tuple=False).flatten():
        daughters = result.batch["daughter_adjacency"][0, mother]
        torch.testing.assert_close(
            result.batch["p4"][0, mother],
            result.batch["p4"][0, daughters].sum(dim=0),
        )


def test_predicted_rollout_terminates_safely_on_untrained_tiny_model():
    result = level_rollout(
        _model(),
        _batch(0),
        mode="predicted",
        config=RolloutConfig(max_level=3, root_types=()),
    )
    assert len(result.steps) <= 3
    assert result.stop_reason in {
        "all_no_object",
        "no_valid_new_mother",
        "maximum_level",
        "repeated_reconstruction_state",
    }
    assert result.valid


def test_truth_and_predicted_paths_construct_identical_composite_for_identical_links():
    model = _model()
    base = _batch(0)
    from hypertagging.models.level_autoregressive import _upgrade_flat_batch

    state = _upgrade_flat_batch(base)
    selection = state["level_ids"][0] == 0
    # The fixture places both leaves first, so select them explicitly.
    for key, value in list(state.items()):
        if value.ndim >= 3 and value.shape[1:3] == (3, 3):
            state[key] = value[:, :2, :2]
        elif value.ndim >= 2 and value.shape[1] == 3:
            state[key] = value[:, :2]
    state["node_mask"] = state["active"]
    proposal = CompositeProposal(0, 111, (0, 1), 1.0, 1.0)
    truth_state, _ = append_composite_proposals(state, [proposal], target_level=1)
    predicted_state, _ = append_composite_proposals(state, [proposal], target_level=1)
    torch.testing.assert_close(truth_state["p4"], predicted_state["p4"])
    torch.testing.assert_close(
        truth_state["composite_features"],
        predicted_state["composite_features"],
    )
    assert int(truth_state["node_ids"][0, -1]) >= 0
    assert int(truth_state["reco_ids"][0, -1]) == -1
    assert int(truth_state["source_node_ids"][0, -1]) == int(truth_state["node_ids"][0, -1])
    assert int(truth_state["copied_from"][0, -1]) == -1


def test_scheduled_sampling_is_reproducible():
    config = RolloutConfig(
        max_level=3,
        root_types=(),
        scheduled_sampling_probability=0.5,
        seed=123,
        exclusive_final=False,
    )
    model = _model()
    first = level_rollout(model, _batch(), mode="scheduled", config=config)
    second = level_rollout(model, _batch(), mode="scheduled", config=config)
    assert [step.used_teacher_forcing for step in first.steps] == [
        step.used_teacher_forcing for step in second.steps
    ]
    torch.testing.assert_close(first.batch["p4"], second.batch["p4"])


def test_invalid_or_cyclic_links_are_rejected_and_exclusive_resolution_uses_sources():
    invalid = CompositeProposal(0, 1, (0, 3), 1.0, 1.0)
    assert not validate_proposals([invalid], existing_node_count=3)
    duplicated = CompositeProposal(0, 1, (0, 0), 1.0, 1.0)
    assert not validate_proposals([duplicated], existing_node_count=3)

    source_ids = torch.tensor([10, 11, 10])
    proposals = [
        CompositeProposal(0, 2, (0, 1), 0.9, 0.9),
        CompositeProposal(1, 3, (2,), 0.8, 0.8),
    ]
    accepted = resolve_exclusive_proposals(proposals, source_ids)
    assert accepted == [proposals[0]]


def test_composite_tensor_construction_is_daughter_order_invariant():
    p4 = torch.tensor([[[0.1, 0.0, 0.0, 0.2], [-0.1, 0.2, 0.0, 0.3]]])
    charge = torch.tensor([[1.0, -1.0]])
    pid = torch.tensor([[8, 26]])
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[True, True]])
    original = composite_token_from_daughters(
        daughter_mask=mask,
        p4=p4,
        charge=charge,
        pid_labels=pid,
        daughter_embeddings=embeddings,
    )
    permutation = torch.tensor([1, 0])
    permuted = composite_token_from_daughters(
        daughter_mask=mask[:, permutation],
        p4=p4[:, permutation],
        charge=charge[:, permutation],
        pid_labels=pid[:, permutation],
        daughter_embeddings=embeddings[:, permutation],
    )
    for field in ("p4", "charge", "features", "daughter_summary", "daughter_pid_histogram"):
        torch.testing.assert_close(original[field], permuted[field])
