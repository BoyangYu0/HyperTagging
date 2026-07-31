import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import (
    RolloutConfig,
    cached_context_for_level,
    level_rollout,
)


def _model():
    torch.manual_seed(4)
    return LevelAutoregressiveReconstructor(
        n_features=8, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4,
        n_queries=4, n_heads=4, n_context_layers=1,
    )


def test_one_rollout_caches_each_prior_level_context():
    batch = collate_level_events([tiny_level_events()[1]], max_query_slots=4).to_dict()
    result = level_rollout(
        _model(), batch, mode="teacher_forced",
        config=RolloutConfig(max_level=3, root_types=(), exclusive_final=False),
    )
    assert [level for level, _ in result.cached_states] == [0, 1, 2, 3]
    assert int(cached_context_for_level(result, 3)["level_ids"].max()) == 2

