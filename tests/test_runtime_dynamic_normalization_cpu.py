import torch

from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.preprocessing.schema_v4 import CONTINUOUS_COMMON_INDICES
from hypertagging.preprocessing.schema_v4 import feature_spec_v4


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
