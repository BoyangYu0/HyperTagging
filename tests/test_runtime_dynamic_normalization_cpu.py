import pytest
import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.models.level_autoregressive import _upgrade_flat_batch
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import CONTINUOUS_COMMON_INDICES
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.reconstruction.level_rollout import (
    CompositeProposal,
    _select_nodes,
    append_composite_proposals,
)


def test_runtime_p4_is_normalized_for_pass_b_and_remains_differentiable():
    batch = collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[0])])
    normalizer = RuntimeFeatureNormalizer(
        common_mean=torch.arange(12, dtype=torch.float32),
        common_std=torch.full((12,), 2.0),
        composite_mean=torch.zeros(13),
        composite_std=torch.full((13,), 2.0),
    )
    model = LevelAutoregressiveReconstructor(
        n_features=12, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4
    )
    model.set_runtime_feature_normalizer(normalizer)
    output = model(batch, target_level=1)
    assert output.second_pass_common_features is not None
    expected = (output.current_p4[..., :4] - torch.arange(4)) / 2
    torch.testing.assert_close(output.second_pass_common_features[..., :4], expected)
    categorical = set(range(12)) - set(CONTINUOUS_COMMON_INDICES)
    for index in categorical:
        assert not output.second_pass_common_availability[..., index].any()
    output.pointer.pointer_logits.square().mean().backward()
    assert torch.isfinite(model.leaf_pid_head.weight.grad).all()


def test_categorical_token_numbers_do_not_enter_continuous_geometry():
    normalizer = RuntimeFeatureNormalizer.identity(12, 13)
    common = torch.randn(1, 3, 12)
    availability = torch.ones_like(common, dtype=torch.bool)
    composite = torch.zeros(1, 3, 13)
    composite_mask = torch.zeros_like(composite, dtype=torch.bool)
    changed = common.clone()
    reduced_pid_index = feature_spec_v4()["common"].index("reduced_pid")
    changed[..., reduced_pid_index] = torch.tensor([1.0, 17.0, 40.0])
    first = normalizer.normalize_runtime(
        common, availability, composite, composite_mask
    )
    second = normalizer.normalize_runtime(
        changed, availability, composite, composite_mask
    )
    torch.testing.assert_close(first[0], second[0])
    assert not first[1][..., reduced_pid_index].any()


def _runtime_confidence_normalizer() -> RuntimeFeatureNormalizer:
    composite_count = torch.ones(13)
    composite_count[6:8] = 0
    composite_std = torch.ones(13)
    composite_std[6:8] = 1e-6
    return RuntimeFeatureNormalizer(
        common_mean=torch.zeros(12),
        common_std=torch.ones(12),
        composite_mean=torch.zeros(13),
        composite_std=composite_std,
        common_count=torch.ones(12),
        composite_count=composite_count,
    )


def test_unobserved_runtime_confidence_uses_identity_scale_and_fails_closed():
    normalizer = _runtime_confidence_normalizer()
    common = torch.zeros((1, 1, 12))
    common_available = torch.ones_like(common, dtype=torch.bool)
    composite = torch.zeros((1, 1, 13))
    composite[..., 6:8] = 0.51
    composite_available = torch.zeros_like(composite, dtype=torch.bool)
    composite_available[..., 6:8] = True

    normalized = normalizer.normalize_runtime(
        common,
        common_available,
        composite,
        composite_available,
    )[2]
    torch.testing.assert_close(normalized[..., 6:8], composite[..., 6:8])
    assert torch.isfinite(normalized[..., 6:8].half()).all()

    composite[..., 6] = float("nan")
    with pytest.raises(
        ValueError, match="available runtime composite features must be finite"
    ):
        normalizer.normalize_runtime(
            common,
            common_available,
            composite,
            composite_available,
        )


def test_scheduled_sampling_runtime_append_keeps_allowed_pid_logits_finite_in_fp16():
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    batch = _upgrade_flat_batch(batch)
    state = _select_nodes(batch, batch["level_ids"][0] == 0)
    state["node_kind_ids"][:] = NODE_KIND_TO_ID["track"]
    state["leaf_kinematics_mode_ids"][:] = LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    state["charge"][0] = torch.tensor([-1.0, 1.0])
    state["pid_labels"][:] = 0
    state, _ = append_composite_proposals(
        state,
        [CompositeProposal(0, 4, (0, 1), 0.51, 0.51)],
        target_level=1,
    )
    model = LevelAutoregressiveReconstructor(
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=4,
        n_heads=4,
        n_context_layers=1,
    ).eval()
    model.set_runtime_feature_normalizer(_runtime_confidence_normalizer())

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.float16):
        output = model(state, target_level=2)

    raw = state["leaf_kinematics_mode_ids"] == LEAF_MODE_TO_ID[
        "raw_track_predicted_pid"
    ]
    assert output.leaf_pid_logits is not None
    assert torch.isfinite(output.leaf_pid_logits[raw]).all()
