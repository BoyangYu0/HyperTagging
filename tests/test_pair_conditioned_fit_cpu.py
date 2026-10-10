"""Bounded fit integration; synthetic losses do not establish scientific benefit."""

from collections import Counter
import copy
import json

import pytest
import torch
from torch import nn

from scripts.run_phase74_development import fit


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.encoder.d_model = 2
        self.runtime_feature_normalizer = None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)
        self.edge = nn.Linear(2, 3)
        self.forward_calls = 0

    def forward(self, h, sources):
        # Truth cannot enter this signature. Source masks alone form pairs.
        assert sources.dtype == torch.bool
        self.forward_calls += 1
        pairs = torch.triu_indices(len(h), len(h), 1)
        return dict(
            value=self.head(h),
            states=h,
            edge_logits=self.edge((h[pairs[0]] - h[pairs[1]]).abs()),
            edge_pairs=pairs,
        )


@pytest.fixture
def run_fit(monkeypatch, tmp_path):
    from hypertagging.training import capacity_development as capacity
    from hypertagging.models import assembly_development as assembly

    monkeypatch.setattr(
        capacity, "detector_features", lambda m, x: (m.encoder(x), None, None)
    )
    monkeypatch.setattr(
        capacity, "membership_loss", lambda result, y: result["value"].square().mean()
    )
    monkeypatch.setattr(
        assembly,
        "assembly_relation_loss",
        lambda states, _: (states.square().mean() * 0.1, [1, 0, 0]),
    )

    def run(name, targets=None, **kwargs):
        torch.manual_seed(85)
        model, decoder = Model(), Decoder()
        before = copy.deepcopy((model.state_dict(), decoder.state_dict()))
        rows = [
            dict(
                uid="fixture",
                detector=torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0], [2.0, 4.0]]
                ),
                targets=torch.tensor([1, 1, 2, 0, -1]) if targets is None else targets,
                sources=torch.eye(5, dtype=torch.bool),
                supervision=None,
            )
        ]
        output = tmp_path / name
        output.mkdir()
        compute = Counter()
        stage = kwargs.pop("stage", "downstream")
        history = fit(
            model,
            decoder,
            rows,
            stage=stage,
            objective="existing",
            settings={
                stage + "_updates": 2,
                "batch_size": 1,
                "seed": 19,
                "encoder_lr": 0.01,
                "head_lr": 0.01,
            },
            output=output,
            compute=compute,
            cache_binding={"fixture": True},
            source_sha="fixture",
            **kwargs,
        )
        checkpoint = torch.load(output / (stage + "-final.pt"), weights_only=False)
        logs = [
            json.loads(line)
            for line in (output / (stage + "-metrics.jsonl")).read_text().splitlines()
        ]
        return history, checkpoint, logs, compute, before, model, decoder

    return run


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_default_and_explicit_false_preserve_history_optimizer_rng_and_loss(run_fit):
    a, b = run_fit("old"), run_fit("false", semantic_pair_supervision=False)
    equal(a[1], b[1])
    assert "semantic_pair_supervision" not in a[0]
    assert "semantic_pair_supervision" not in a[1]
    assert not any(k.startswith("semantic_pair") for k in a[3])
    for left, right in zip(a[2], b[2]):
        equal(
            {k: v for k, v in left.items() if k != "elapsed_seconds"},
            {k: v for k, v in right.items() if k != "elapsed_seconds"},
        )


def test_semantic_support_loss_coefficient_and_real_updates(run_fit):
    h, cp, logs, counts, before, model, decoder = run_fit(
        "semantic", semantic_pair_supervision=True
    )
    assert h["semantic_pair_supervision"] is cp["semantic_pair_supervision"] is True
    assert h["semantic_pair_coefficient"] == cp["semantic_pair_coefficient"] == 1.0
    assert (
        counts["semantic_pair_edge_scores"]
        == counts["semantic_pair_eligible_pairs"]
        == 20
    )
    assert counts["semantic_pair_unknown_pairs"] == 8
    assert counts["semantic_pair_same_b"] == 2
    assert counts["semantic_pair_cross_b"] == 4
    assert counts["semantic_pair_background"] == 6
    assert counts["semantic_pair_source_excluded_pairs"] == 0
    for row in logs:
        assert row["loss"] == pytest.approx(sum(row["components"].values()))
        assert row["components"]["semantic_pair"] > 0
    assert decoder.forward_calls == 2
    assert not torch.equal(before[1]["edge.weight"], decoder.edge.weight)
    assert not torch.equal(before[0]["encoder.weight"], model.encoder.weight)
    assert decoder.edge.weight.grad.abs().sum() > 0


def test_unknown_targets_supply_no_semantic_gradient(run_fit):
    result = run_fit(
        "unknown", targets=torch.full((5,), -1), semantic_pair_supervision=True
    )
    assert result[3]["semantic_pair_unknown_pairs"] == 20
    assert all(row["components"]["semantic_pair"] == 0 for row in result[2])
    assert result[6].edge.weight.grad.count_nonzero() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semantic_pair_supervision": 1},
        {"semantic_pair_supervision": True, "stage": "pretraining"},
        {"semantic_pair_supervision": True, "stage": "tiny"},
        {"semantic_pair_supervision": True, "gradient_rule": "joint_diagnostic"},
        {
            "semantic_pair_supervision": True,
            "gradient_rule": "project_conflicting_relation",
        },
        {"semantic_pair_supervision": True, "pair_supervision": True},
    ],
)
def test_incompatible_histories_rejected_before_fitting(run_fit, kwargs):
    with pytest.raises(ValueError, match="Semantic pair supervision"):
        run_fit("rejected", **kwargs)


def connection_contract(tmp_path, monkeypatch):
    from scripts import phase85_pair_connection as runner
    from scripts.prepare_phase74_development_data import binding

    receipts = []
    for index in range(2):
        path = tmp_path / f"review-{index}.json"
        path.write_text(
            json.dumps(
                dict(
                    status="PASS",
                    distinct_fixture=index,
                    pool=384,
                    review_scheduler="COMPLETED|0:0",
                    source_sha=runner.REPLICATION_SOURCE_SHA if index else "original",
                    bindings=[{"sha256": runner.REPLICATION_CONTRACT_SHA256}]
                    if index
                    else [],
                    gate=dict(
                        absolute_trainable=True,
                        exposure_gain=True,
                        fixed6000_eligible=True,
                    ),
                )
            )
        )
        receipts.append(binding(path))
    monkeypatch.setattr(runner, "ORIGINAL_REVIEW_SHA256", receipts[0]["sha256"])
    return dict(
        kind="phase85_pair_connection",
        arms=list(runner.ARMS),
        pool_size=384,
        settings=runner.study_settings(),
        milestones=list(runner.MILESTONES),
        addition_seed=runner.ADDITION_SEED,
        resources=runner.RESOURCES,
        semantic_pair_coefficient=1.0,
        heldout_events=0,
        sealed_test_access=False,
        automatic_successor=False,
        maximum_scientific_jobs=2,
        exposure_prerequisites=receipts,
    )


def test_connection_runner_accepts_two_bound_reviews_and_rejects_changed_content(
    tmp_path,
    monkeypatch,
):
    from scripts import phase85_pair_connection as runner
    from pathlib import Path

    contract = connection_contract(tmp_path, monkeypatch)
    runner.validate(contract)
    Path(contract["exposure_prerequisites"][1]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        runner.validate(contract)


@pytest.mark.parametrize(
    "key,value",
    [
        ("pool_size", 1536),
        ("semantic_pair_coefficient", 0.5),
        ("heldout_events", 600),
        ("maximum_scientific_jobs", 3),
        ("automatic_successor", True),
        ("resources", dict(cpus=4, memory_gib=32, hours=8, gpus=0, requeue=False)),
    ],
)
def test_connection_runner_rejects_unregistered_contract_changes(
    tmp_path, monkeypatch, key, value
):
    from scripts import phase85_pair_connection as runner

    contract = connection_contract(tmp_path, monkeypatch)
    contract[key] = value
    with pytest.raises(ValueError, match="controls changed"):
        runner.validate(contract)


def test_connection_runner_rejects_repeated_or_failed_review(tmp_path, monkeypatch):
    from scripts import phase85_pair_connection as runner
    from scripts.prepare_phase74_development_data import binding
    from pathlib import Path

    contract = connection_contract(tmp_path, monkeypatch)
    contract["exposure_prerequisites"][1] = contract["exposure_prerequisites"][0]
    with pytest.raises(ValueError, match="Exact reserved sampling-seed replication"):
        runner.validate(contract)
    contract = connection_contract(tmp_path, monkeypatch)
    path = Path(contract["exposure_prerequisites"][1]["path"])
    review = json.loads(path.read_text())
    review["gate"]["absolute_trainable"] = False
    path.write_text(json.dumps(review))
    contract["exposure_prerequisites"][1] = binding(path)
    with pytest.raises(ValueError, match="prerequisite absent"):
        runner.validate(contract)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pool", 96),
        ("review_scheduler", "RUNNING|0:0"),
        ("source_sha", "wrong-source"),
        ("bindings", [{"sha256": "wrong-contract"}]),
    ],
)
def test_connection_runner_requires_exact_replication_identity(
    tmp_path, monkeypatch, field, value
):
    from scripts import phase85_pair_connection as runner
    from scripts.prepare_phase74_development_data import binding
    from pathlib import Path

    contract = connection_contract(tmp_path, monkeypatch)
    path = Path(contract["exposure_prerequisites"][1]["path"])
    review = json.loads(path.read_text())
    review[field] = value
    path.write_text(json.dumps(review))
    contract["exposure_prerequisites"][1] = binding(path)
    with pytest.raises(ValueError):
        runner.validate(contract)


def test_connection_runner_rejects_distinct_fake_original(tmp_path, monkeypatch):
    from scripts import phase85_pair_connection as runner
    from scripts.prepare_phase74_development_data import binding
    from pathlib import Path

    contract = connection_contract(tmp_path, monkeypatch)
    path = Path(contract["exposure_prerequisites"][0]["path"])
    review = json.loads(path.read_text())
    review["distinct_fixture"] = "altered-original"
    path.write_text(json.dumps(review))
    contract["exposure_prerequisites"][0] = binding(path)
    with pytest.raises(ValueError, match="Exact original384"):
        runner.validate(contract)


def test_smoke_four_update_sampler_actually_presents_worst_index_zero():
    from scripts.phase84_exposure import exposure
    from scripts.phase85_pair_connection import study_settings
    import hashlib
    import random

    rows = [dict(uid=f"fixture-{i}", category="charged") for i in range(8)]
    seed = study_settings()["seed"]
    assert seed == 202610081
    measured = exposure(rows, 4, seed)
    rng = random.Random(seed)
    indices = [rng.randrange(8) for _ in range(32)]
    assert 0 in indices
    assert measured["per_uid"]["fixture-0"] == indices.count(0)
    expected = hashlib.sha256(
        "".join(f"fixture-{i}\n" for i in indices).encode()
    ).hexdigest()
    assert measured["sequence_sha256"] == expected
