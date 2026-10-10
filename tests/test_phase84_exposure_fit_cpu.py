"""Bounded real-Adam tests of uninterrupted exposure snapshot callbacks."""

from collections import Counter
import copy
import hashlib
import json
import random

import pytest
import torch
from torch import nn

from scripts.run_phase74_development import fit


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.encoder.d_model = 2
        self.pid = nn.Linear(2, 2)
        self.runtime_feature_normalizer = None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, h, sources):
        out = self.head(self.dropout(h))
        return {"value": out, "states": out}


@pytest.fixture
def bounded_fit(monkeypatch, tmp_path):
    from hypertagging.training import capacity_development as capacity
    from hypertagging.models import assembly_development as assembly

    monkeypatch.setattr(
        capacity, "detector_features", lambda model, x: (model.encoder(x), None, None)
    )
    monkeypatch.setattr(
        capacity, "membership_loss", lambda out, y: (out["value"] - y).square().mean()
    )
    monkeypatch.setattr(
        assembly,
        "assembly_relation_loss",
        lambda state, _: (state.square().mean() * 0.1, [1, 0, 0]),
    )
    rows = [
        {
            "uid": f"fixture-{i}",
            "detector": torch.tensor([[i / 3, 1.0]]),
            "targets": torch.tensor([[i / 7]]),
            "sources": None,
            "supervision": None,
        }
        for i in range(5)
    ]

    def run(name, callback=None, updates=6):
        torch.manual_seed(84)
        model, decoder = Model(), Decoder()
        output = tmp_path / name
        output.mkdir()
        compute = Counter()
        result = fit(
            model,
            decoder,
            rows,
            stage="downstream",
            objective="existing",
            settings={
                "downstream_updates": updates,
                "batch_size": 2,
                "seed": 19,
                "encoder_lr": 0.01,
                "head_lr": 0.02,
            },
            output=output,
            compute=compute,
            cache_binding={"fixture": True},
            source_sha="fixture",
            step_callback=callback,
        )
        checkpoint = torch.load(output / "downstream-final.pt", weights_only=False)
        logs = [
            json.loads(line)
            for line in (output / "downstream-metrics.jsonl").read_text().splitlines()
        ]
        return result, checkpoint, compute, logs

    return run


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_passive_snapshots_preserve_optimizer_rng_and_final_trajectory(bounded_fit):
    base = bounded_fit("base")
    observed = []
    object_ids = []

    def snapshot(**kw):
        object_ids.append((id(kw["optimizer"]), id(kw["rng"])))
        observed.append(
            {
                "step": kw["step"],
                "optimizer": copy.deepcopy(kw["optimizer"].state_dict()),
                "sampling_rng": kw["rng"].getstate(),
                "torch_rng": torch.get_rng_state().clone(),
                "sequence": kw["sequence_sha256"],
                "loss": kw["last_loss"],
                "presentations": kw["compute"]["event_presentations"],
            }
        )
        return False

    with_callback = bounded_fit("callback", snapshot)
    assert len(set(object_ids)) == 1  # no resets or independent trajectory segments
    assert [x["step"] for x in observed] == list(range(1, 7))
    assert with_callback[0]["stopped_early"] is False
    assert "stopped_early" not in base[0] and "stopped_early" not in base[1]
    for key, value in base[1].items():
        equal_tree(value, with_callback[1][key])
    assert base[2] == with_callback[2]
    rng, order = random.Random(19), hashlib.sha256()
    for step, obs in enumerate(observed, 1):
        for _ in range(2):
            order.update(f"fixture-{rng.randrange(5)}\n".encode())
        assert obs["sampling_rng"] == rng.getstate()
        assert obs["sequence"] == order.hexdigest()
        assert obs["presentations"] == 2 * step
        assert obs["loss"] == base[3][step - 1]["loss"]
        assert all(int(v["step"]) == step for v in obs["optimizer"]["state"].values())
    equal_tree(observed[-1]["optimizer"], base[1]["optimizer_state_dict"])
    assert torch.equal(observed[-1]["torch_rng"], base[1]["torch_rng_state"])


def test_early_stop_saves_actual_completed_state_not_requested_budget(bounded_fit):
    result, checkpoint, compute, logs = bounded_fit(
        "stop", lambda **kw: kw["step"] == 3
    )
    reference = bounded_fit("three", updates=3)
    assert result["updates"] == checkpoint["step"] == len(logs) == 3
    assert result["presentations"] == compute["event_presentations"] == 6
    assert result["stopped_early"] is checkpoint["stopped_early"] is True
    assert result["requested_updates"] == checkpoint["requested_updates"] == 6
    for key, value in reference[1].items():
        if key != "settings":
            equal_tree(value, checkpoint[key])
    assert checkpoint["resume_authorized"] is False


def test_callback_return_requires_explicit_bool(bounded_fit):
    with pytest.raises(TypeError, match="return bool"):
        bounded_fit("bad-return", lambda **kw: None)
    with pytest.raises(TypeError, match="callable"):
        bounded_fit("bad-callable", 1)


def test_final_step_stop_is_completed_not_early(bounded_fit):
    result, checkpoint, _, _ = bounded_fit("final-stop", lambda **kw: kw["step"] == 6)
    assert result["updates"] == 6
    assert result["stopped_early"] is checkpoint["stopped_early"] is False
