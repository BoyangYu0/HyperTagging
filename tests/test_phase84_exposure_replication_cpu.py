"""Bounded replication contract tests, no model fitting or real data."""

import copy
import json

import pytest

from scripts import phase84_exposure as study


@pytest.fixture
def reference(tmp_path, monkeypatch):
    original = dict(
        kind="phase84_exposure",
        pool_size=384,
        source_sha=study.REPLICATION_SOURCE,
        parent="authenticated-train-parent",
        pretrain="immutable-pretrain",
        settings={**study.settings(), "downstream_updates": 6000},
    )
    startup = dict(
        contract=None,
        source_sha=study.REPLICATION_SOURCE,
        pool=384,
        initial_digest=study.REPLICATION_INITIAL_DIGEST,
        initial_checkpoint={"path": "pretraining-final.pt", "sha256": "pretrained"},
        uid_order_sha256="same-384-order",
    )
    review = dict(
        status="PASS",
        source_sha=study.REPLICATION_SOURCE,
        pool=384,
        review_scheduler="COMPLETED|0:0",
        gate=dict(absolute_trainable=True, exposure_gain=True, fixed6000_eligible=True),
    )

    def write_refs(original=original, startup=startup, review=review):
        bindings = {}
        for name, value in [
            ("reference_contract", original),
            ("reference_startup", startup),
            ("reference_terminal_review", review),
        ]:
            if name == "reference_startup":
                value = {**value, "contract": bindings["reference_contract"]}
            path = tmp_path / (name + ".json")
            path.write_text(json.dumps(value))
            bindings[name] = study.binding(path)
        monkeypatch.setattr(
            study,
            "REPLICATION_REFERENCE_HASHES",
            {k: v["sha256"] for k, v in bindings.items()},
        )
        return bindings

    refs = write_refs()
    c = dict(
        kind="phase84_exposure",
        pool_size=384,
        milestones=list(study.MILESTONES),
        settings={**study.settings(), "downstream_updates": 6000, "seed": 202610082},
        resources=dict(cpus=2, memory_gib=32, hours=8, gpus=0, requeue=False),
        heldout_events=0,
        automatic_successor=False,
        sealed_test_access=False,
        parent=original["parent"],
        pretrain=original["pretrain"],
        bindings=list(refs.values()),
        replication=dict(
            kind="training_order",
            reference_pool=384,
            reference_sampler_seed=202610081,
            new_sampler_seed=202610082,
            reference_source_sha=study.REPLICATION_SOURCE,
            expected_initial_digest=study.REPLICATION_INITIAL_DIGEST,
            **refs,
        ),
    )
    return c, original, startup, review, write_refs


def test_valid_replication_has_same_initializer_and_fresh_optimizer(reference):
    c, _, startup, _, _ = reference
    study.validate(c)
    info = study.validate_replication_initial(
        c,
        startup["initial_digest"],
        startup["initial_checkpoint"],
        startup["uid_order_sha256"],
    )
    assert info["fresh_optimizer"] is True
    assert info["new_sampler_seed"] == 202610082


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_sampler_seed", 202610083),
        ("reference_pool", 96),
        ("reference_source_sha", "other"),
        ("expected_initial_digest", "changed"),
        ("kind", "checkpoint_resume"),
    ],
)
def test_replication_controls_rejected(reference, key, value):
    c = copy.deepcopy(reference[0])
    c["replication"][key] = value
    with pytest.raises(ValueError):
        study.validate(c)


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", 4),
        ("downstream_updates", 6001),
        ("encoder_lr", 0.01),
        ("seed", 202610081),
    ],
)
def test_settings_cannot_change_except_sampler(reference, key, value):
    c = copy.deepcopy(reference[0])
    c["settings"][key] = value
    with pytest.raises(ValueError):
        study.validate(c)


def test_unbound_and_changed_reference_rejected(reference):
    c = copy.deepcopy(reference[0])
    c["bindings"].pop()
    with pytest.raises(ValueError, match="Unbound"):
        study.validate(c)
    c = copy.deepcopy(reference[0])
    c["replication"]["reference_startup"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="Unbound"):
        study.validate(c)


@pytest.mark.parametrize("pool", [96, 1536])
def test_replication_forbids_other_pool(reference, pool):
    c = copy.deepcopy(reference[0])
    c["pool_size"] = pool
    with pytest.raises(ValueError, match="Replication controls"):
        study.validate(c)


def test_replication_rejects_undeclared_fields(reference):
    c = copy.deepcopy(reference[0])
    c["replication"]["resume_checkpoint"] = "original-final.pt"
    with pytest.raises(ValueError, match="fields changed"):
        study.validate(c)


@pytest.mark.parametrize("which", ["history", "initializer", "gate", "final"])
def test_authenticated_reference_must_support_replication(reference, which):
    c, original, startup, review, rewrite = copy.deepcopy(reference[:4]) + (
        reference[4],
    )
    if which == "history":
        original["settings"]["pretraining_updates"] = 1
    elif which == "initializer":
        startup["initial_digest"] = "changed"
    elif which == "gate":
        review["gate"]["exposure_gain"] = False
    else:
        review["gate"]["fixed6000_eligible"] = False
    refs = rewrite(original, startup, review)
    c["replication"].update(refs)
    c["bindings"] = list(refs.values())
    with pytest.raises(ValueError, match="history/gates"):
        study.validate(c)


@pytest.mark.parametrize("which", ["digest", "checkpoint", "order"])
def test_actual_startup_mismatch_fails_before_fit(reference, which):
    c, _, startup, _, _ = reference
    values = [
        startup["initial_digest"],
        startup["initial_checkpoint"],
        startup["uid_order_sha256"],
    ]
    values[["digest", "checkpoint", "order"].index(which)] = "changed"
    with pytest.raises(ValueError, match="initialization/checkpoint/pool"):
        study.validate_replication_initial(c, *values)


def test_training_order_is_independent_but_budget_and_pool_fixed():
    rows = [dict(uid=str(i), category=study.CATEGORIES[i % 6]) for i in range(384)]
    a = study.exposure(rows, 6000, 202610081)
    b = study.exposure(rows, 6000, 202610082)
    assert a["presentations"] == b["presentations"] == 48000
    assert a["unique_sampled"] == b["unique_sampled"] == 384
    assert a["sequence_sha256"] != b["sequence_sha256"]
    assert b == study.exposure(rows, 6000, 202610082)


def test_default_remains_original_and_unregistered_seed_fails():
    c = dict(
        kind="phase84_exposure",
        pool_size=384,
        milestones=list(study.MILESTONES),
        settings={**study.settings(), "downstream_updates": 6000},
        resources=dict(cpus=2, memory_gib=32, hours=8, gpus=0, requeue=False),
        heldout_events=0,
        automatic_successor=False,
        sealed_test_access=False,
    )
    study.validate(c)
    c["settings"]["seed"] = 202610082
    with pytest.raises(ValueError):
        study.validate(c)
