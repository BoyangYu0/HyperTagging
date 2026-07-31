import torch

from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import RolloutConfig, level_rollout


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = LevelAutoregressiveReconstructor(
            n_features=8, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4,
            n_queries=4, n_heads=4, n_context_layers=1,
        )
        self.forward_count = 0

    def forward(self, *args, **kwargs):
        self.forward_count += 1
        return self.base(*args, **kwargs)


def test_deterministic_cached_microbenchmark_uses_fewer_forwards():
    batch = collate_level_events([tiny_level_events()[1]], max_query_slots=4).to_dict()
    cached = _CountingModel()
    level_rollout(
        cached, batch, mode="teacher_forced",
        config=RolloutConfig(max_level=2, root_types=(), exclusive_final=False),
    )
    uncached = _CountingModel()
    uncached.base.load_state_dict(cached.base.state_dict())
    # Historical path independently rebuilt contexts for target levels 2 and 3.
    for prior_level in (1, 2):
        level_rollout(
            uncached, batch, mode="teacher_forced",
            config=RolloutConfig(
                max_level=prior_level, root_types=(), exclusive_final=False
            ),
        )
    assert cached.forward_count == 2
    assert uncached.forward_count == 3

