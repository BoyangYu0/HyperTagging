import pytest
import torch

import hypertagging.training.reconstruction_trainer as reconstruction_trainer
from hypertagging.data.heterogeneous import heterogeneous_from_level_event
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.level_rollout import rollout_policy_identity
from hypertagging.training.reconstruction_trainer import validate_reconstruction


class _DataModule:
    def __init__(self):
        self.events = [heterogeneous_from_level_event(tiny_level_events()[0]) for _ in range(3)]
        self.split_counts = {"validation": 3}
        self.allowed_types_by_level = {}

    def iter_events(self, *_args, **_kwargs):
        return iter(self.events)

    def normalize_batch(self, batch):
        return batch


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = LevelAutoregressiveReconstructor(
            n_features=12, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4,
            n_queries=4, n_heads=4, n_context_layers=1,
        )
        self.batch_sizes = []

    def forward(self, batch, *, target_level, **kwargs):
        self.batch_sizes.append(batch["node_mask"].shape[0])
        return self.base(batch, target_level=target_level, **kwargs)


def test_validation_batch_size_controls_next_level_forward_batching():
    model = _RecordingModel()
    result = validate_reconstruction(
        model, _DataModule(), device=torch.device("cpu"),
        max_validation_events=3, rollout_validation_events=0,
        validation_batch_size=2,
    )
    assert 2 in model.batch_sizes
    assert result["validation_batch_size"] == 2
    assert result["validation_events"] == 3
    assert result["rollout_empty_level_policy"] == "stop_on_first_empty"
    assert result["rollout_policy_version"] == "level-rollout-source-isolation-v3"
    assert len(result["rollout_policy_sha256"]) == 64


def test_lightweight_data_module_still_fails_closed_in_scientific_mode():
    with pytest.raises(
        ValueError,
        match="scientific fixed validation requires a training-selection manifest",
    ):
        validate_reconstruction(
            _RecordingModel(),
            _DataModule(),
            device=torch.device("cpu"),
            max_validation_events=3,
            rollout_validation_events=0,
            validation_batch_size=2,
            scientific_mode=True,
        )


def test_validation_threads_continue_through_empty_levels_to_every_rollout(
    monkeypatch,
):
    observed = []
    real_level_rollout = reconstruction_trainer.level_rollout

    def recording_level_rollout(*args, **kwargs):
        observed.append((kwargs["mode"], kwargs["config"]))
        return real_level_rollout(*args, **kwargs)

    monkeypatch.setattr(
        reconstruction_trainer, "level_rollout", recording_level_rollout
    )
    result = validate_reconstruction(
        _RecordingModel(),
        _DataModule(),
        device=torch.device("cpu"),
        max_validation_events=1,
        rollout_validation_events=1,
        validation_batch_size=1,
        rollout_continue_through_empty_levels=True,
    )

    assert [mode for mode, _config in observed] == [
        "teacher_forced",
        "predicted",
        "predicted",
        "scheduled",
    ]
    assert all(config.continue_through_empty_levels for _mode, config in observed)
    policy = rollout_policy_identity(continue_through_empty_levels=True)
    assert result["rollout_empty_level_policy"] == "continue_to_max_level"
    assert result["rollout_policy_sha256"] == policy["sha256"]
