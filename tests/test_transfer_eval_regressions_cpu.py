import torch

from hypertagging.losses.level_reconstruction import level_reconstruction_loss
from hypertagging.models.mother_pointer import MotherPointerDecoder, MotherPointerOutput
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import _select_nodes


def test_leaf_selection_preserves_pid_vocabulary_when_node_count_matches_it():
    node_count = len(PDG_TOKENS)
    selection = torch.arange(node_count) < 39
    active = torch.ones((1, node_count), dtype=torch.bool)
    histogram = torch.arange(
        node_count * node_count, dtype=torch.float32
    ).reshape(1, node_count, node_count)
    batch = {
        "active": active,
        "node_mask": active,
        "node_ids": torch.arange(node_count).reshape(1, node_count),
        "parent_ids": torch.full((1, node_count), -1, dtype=torch.long),
        "common_features": torch.zeros(1, node_count, 12),
        "daughter_input_pid_histogram": histogram,
        "daughter_truth_pid_histogram": histogram.clone(),
        "daughter_adjacency": torch.eye(node_count, dtype=torch.bool).unsqueeze(0),
    }

    selected = _select_nodes(batch, selection)

    assert selected["daughter_input_pid_histogram"].shape == (1, 39, node_count)
    assert selected["daughter_truth_pid_histogram"].shape == (1, 39, node_count)
    assert selected["daughter_adjacency"].shape == (1, 39, 39)
    torch.testing.assert_close(
        selected["daughter_input_pid_histogram"], histogram[:, :39]
    )


def _pointer_output(type_count: int) -> MotherPointerOutput:
    return MotherPointerOutput(
        object_logits=torch.zeros(1, 2),
        type_logits=torch.zeros(1, 2, type_count),
        pointer_logits=torch.zeros(1, 2, 3),
        cardinality_logits=torch.zeros(1, 2, 3),
        confidence_logits=torch.zeros(1, 2),
    )


def _loss_batch(target_token: int) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.zeros(1, 3, 2),
        "node_mask": torch.ones(1, 3, dtype=torch.bool),
        "level_ids": torch.tensor([[0, 0, 1]]),
        "daughter_adjacency": torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [1, 1, 0]]], dtype=torch.bool
        ),
        "pid_labels": torch.tensor([[2, 2, target_token]]),
        "pid_target_labels": torch.tensor([[2, 2, target_token]]),
        "p4": torch.zeros(1, 3, 4),
        "charge": torch.zeros(1, 3),
        "valid_reconstruction_target": torch.tensor([[0, 0, 1]], dtype=torch.bool),
        "recursive_reconstructable_complete": torch.ones(1, 3, dtype=torch.bool),
    }


def test_complete_composite_kl_and_electron_targets_are_allowed_and_loss_fails_closed():
    policy = ReconstructionConstraintPolicy(empirical_type_prior_mode="off")
    allowed, _bias = policy.type_constraints(1, device=torch.device("cpu"))
    assert allowed[3]  # K_L0 production composite target
    assert allowed[24]  # electron production composite target
    decoded = MotherPointerDecoder(hidden_dim=8, n_types=len(PDG_TOKENS), n_queries=2)(
        torch.zeros(1, 2, 8),
        torch.ones(1, 2, dtype=torch.bool),
        target_level=1,
        allowed_type_mask=allowed,
    )
    assert (decoded.type_logits[..., (3, 24)] > -1e3).all()
    assert (decoded.type_logits[..., 2] == -1e4).all()

    for target_token in (3, 24):
        batch = _loss_batch(target_token)
        batch["allowed_type_mask"] = allowed
        loss = level_reconstruction_loss(
            _pointer_output(len(PDG_TOKENS)),
            batch,
            target_level=1,
            constraint_policy=policy,
            matching_production=False,
        )
        assert torch.isfinite(loss.components["type"])
        assert float(loss.components["type"]) < 10.0

    rejected = _loss_batch(2)  # gamma remains outside the mother ontology
    rejected["allowed_type_mask"] = allowed
    try:
        level_reconstruction_loss(
            _pointer_output(len(PDG_TOKENS)),
            rejected,
            target_level=1,
            constraint_policy=policy,
            matching_production=False,
        )
    except ValueError as error:
        assert "excluded by the decoder mask" in str(error)
    else:
        raise AssertionError("decoder-mask target mismatch did not fail closed")


if __name__ == "__main__":
    test_leaf_selection_preserves_pid_vocabulary_when_node_count_matches_it()
    test_complete_composite_kl_and_electron_targets_are_allowed_and_loss_fails_closed()
