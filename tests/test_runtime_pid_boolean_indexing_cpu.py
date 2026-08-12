import pytest
import torch

from hypertagging.preprocessing.pid_filter import PDG_TOKENS, TOKENIZE_DICT
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.reconstruction.kinematics import (
    PARTICLE_CHARGES,
    hard_reconstructed_p4_from_leaf_pid,
    hard_track_p4_from_pid_token,
    soft_reconstructed_p4_from_leaf_pid,
    soft_track_p4_from_pid_logits,
)
from hypertagging.reconstruction.pid_state import rebuild_runtime_pid_state


_RAW_POSITIONS = ((0, 0), (0, 4), (1, 1), (1, 3))
_PREFERRED_PDGS = (11, 321, 2212, 13)


def _multi_event_raw_track_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch_size, node_count = 2, 5
    p3 = torch.arange(
        1,
        batch_size * node_count * 3 + 1,
        dtype=torch.float32,
    ).reshape(batch_size, node_count, 3) / 10
    stored_energy = torch.arange(10, 20, dtype=torch.float32).reshape(
        batch_size, node_count, 1
    )
    p4 = torch.cat((p3, stored_energy), dim=-1)
    raw = torch.zeros((batch_size, node_count), dtype=torch.bool)
    charge = torch.zeros((batch_size, node_count), dtype=torch.float32)
    logits = torch.full(
        (batch_size, node_count, len(PDG_TOKENS)),
        -2.0,
        dtype=torch.float32,
    )
    for (batch_index, node_index), pdg in zip(
        _RAW_POSITIONS, _PREFERRED_PDGS, strict=True
    ):
        raw[batch_index, node_index] = True
        charge[batch_index, node_index] = PARTICLE_CHARGES[pdg]
        logits[batch_index, node_index, TOKENIZE_DICT[pdg]] = 3.0

    modes = torch.full(
        (batch_size, node_count),
        LEAF_MODE_TO_ID["ecl_cluster"],
        dtype=torch.long,
    )
    modes[raw] = LEAF_MODE_TO_ID["raw_track_predicted_pid"]
    batch = {
        "p4": p4,
        "charge": charge,
        "node_mask": torch.ones_like(raw),
        "leaf_kinematics_mode_ids": modes,
        "pid_labels": torch.zeros_like(modes),
        "level_ids": torch.zeros_like(modes),
        "daughter_adjacency": torch.zeros(
            (batch_size, node_count, node_count), dtype=torch.bool
        ),
        "node_kind_ids": torch.full_like(modes, NODE_KIND_TO_ID["ecl_cluster"]),
    }
    batch["node_kind_ids"][raw] = NODE_KIND_TO_ID["track"]
    return batch, logits.requires_grad_()


def _expected_soft_p4(
    batch: dict[str, torch.Tensor], logits: torch.Tensor
) -> torch.Tensor:
    expected = batch["p4"].clone()
    raw = batch["leaf_kinematics_mode_ids"] == LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    input_p3 = batch["p4"][..., :3]
    for sign in (-1, 1):
        selected = raw & (
            (batch["charge"] < 0) if sign < 0 else (batch["charge"] > 0)
        )
        allowed = tuple(
            TOKENIZE_DICT[pdg]
            for pdg, particle_charge in PARTICLE_CHARGES.items()
            if particle_charge == sign
        )
        expected[selected] = soft_track_p4_from_pid_logits(
            input_p3[selected], logits[selected], allowed_tokens=allowed
        )
    return expected


def _expected_hard_p4(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    expected = batch["p4"].clone()
    for position, pdg in zip(_RAW_POSITIONS, _PREFERRED_PDGS, strict=True):
        expected[position] = hard_track_p4_from_pid_token(
            batch["p4"][position][..., :3], TOKENIZE_DICT[pdg]
        )
    return expected


@pytest.mark.parametrize(
    ("mode", "expected_kind", "has_logit_gradient"),
    [
        ("soft_expectation", "soft", True),
        ("hard", "hard", False),
        ("straight_through_hard", "hard", True),
    ],
)
def test_runtime_pid_p4_boolean_selection_is_batch_shape_safe(
    mode: str, expected_kind: str, has_logit_gradient: bool
):
    batch, logits = _multi_event_raw_track_batch()
    assert batch["p4"].shape == (2, 5, 4)
    assert "pid_target_labels" not in batch and "truth_pid_labels" not in batch

    runtime = rebuild_runtime_pid_state(batch, logits, mode=mode)
    raw = batch["leaf_kinematics_mode_ids"] == LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    expected = (
        _expected_soft_p4(batch, logits.detach())
        if expected_kind == "soft"
        else _expected_hard_p4(batch)
    )

    torch.testing.assert_close(runtime.p4, expected)
    torch.testing.assert_close(runtime.p4[..., :3], batch["p4"][..., :3])
    torch.testing.assert_close(runtime.p4[~raw], batch["p4"][~raw])
    if has_logit_gradient:
        runtime.p4[..., 3][raw].sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[raw].abs().sum() > 0
        assert torch.count_nonzero(logits.grad[~raw]) == 0
    else:
        assert not runtime.p4.requires_grad
        assert logits.grad is None


@pytest.mark.parametrize("hard", [False, True])
def test_legacy_p4_rebuild_boolean_selection_is_batch_shape_safe(hard: bool):
    batch, logits = _multi_event_raw_track_batch()
    actual = (
        hard_reconstructed_p4_from_leaf_pid(batch, logits)
        if hard
        else soft_reconstructed_p4_from_leaf_pid(batch, logits)
    )
    expected = (
        _expected_hard_p4(batch)
        if hard
        else _expected_soft_p4(batch, logits.detach())
    )
    torch.testing.assert_close(actual, expected)
