"""Bounded runner contracts and collision-level metric accounting, no real fits."""

from copy import deepcopy
import hashlib

import pytest
import torch

from scripts.run_phase84_information import (
    ARMS,
    aggregate,
    classify,
    paired_bootstrap,
    validate_contract,
    verify_bindings,
)


def contract():
    return {
        "stage": "training_role_information_diagnostic",
        "seed": 8409,
        "updates": 2048,
        "batch_size": 256,
        "probe_parameters": 24835,
        "training_events": 1536,
        "validation_events": 0,
        "scientific_model_updates": 0,
        "automatic_successor": False,
        "arms": list(ARMS),
        "resources": {
            "cpus": 2,
            "memory_gib": 16,
            "hours": 1,
            "gpus": 0,
            "requeue": False,
        },
        "smoke": False,
        "checkpoint_selection": "fixed_phase84_full1536_step6000",
        "cache": {"path": "/private/train.pt", "sha256": "a" * 64},
        "checkpoint": {"path": "/private/final.pt", "sha256": "b" * 64},
        "bindings": [
            {"path": "/private/train.pt", "sha256": "a" * 64},
            {"path": "/private/final.pt", "sha256": "b" * 64},
        ],
    }


def test_exact_contract_and_declared_checkpoint_alternative():
    c = contract()
    validate_contract(c, False)
    c["checkpoint_selection"] = "fixed_historical_phase76_native"
    c["smoke"] = True
    validate_contract(c, True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("stage", "validation"),
        ("seed", 8410),
        ("updates", 2049),
        ("batch_size", 257),
        ("probe_parameters", 16513),
        ("training_events", 600),
        ("validation_events", 1),
        ("scientific_model_updates", 1),
        ("automatic_successor", True),
        ("arms", list(reversed(ARMS))),
        ("checkpoint_selection", "best_assessment_AUC"),
        ("smoke", True),
    ],
)
def test_unregistered_changes_rejected(key, value):
    c = contract()
    c[key] = value
    with pytest.raises(ValueError):
        validate_contract(c, False)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cpus", 3),
        ("memory_gib", 33),
        ("hours", 9),
        ("hours", 0),
        ("gpus", 1),
        ("requeue", True),
    ],
)
def test_resource_limits_not_advisory(key, value):
    c = contract()
    c["resources"][key] = value
    with pytest.raises(ValueError):
        validate_contract(c, False)


@pytest.mark.parametrize("key", ["cache", "checkpoint"])
@pytest.mark.parametrize("change", ["path", "hash", "missing"])
def test_cache_checkpoint_binding_required(key, change):
    c = contract()
    if change == "path":
        c[key]["path"] += ".replacement"
    elif change == "hash":
        c[key]["sha256"] = "f" * 64
    else:
        c["bindings"] = [b for b in c["bindings"] if b["path"] != c[key]["path"]]
    with pytest.raises(ValueError):
        validate_contract(c, False)


def test_source_and_input_hashes_are_verified_not_only_declared(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("frozen source")
    data = tmp_path / "cache.bin"
    data.write_bytes(b"frozen train")
    c = {
        "source_hashes": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "bindings": [
            {"path": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        ],
    }
    verify_bindings(c, tmp_path)
    source.write_text("modified source")
    with pytest.raises(ValueError, match="Frozen source changed"):
        verify_bindings(c, tmp_path)
    source.write_text("frozen source")
    data.write_bytes(b"other cohort")
    with pytest.raises(ValueError, match="Immutable input changed"):
        verify_bindings(c, tmp_path)


def fixture():
    target = torch.tensor([1, 1, 2, 0, -1])
    ij = torch.triu_indices(5, 5, 1)
    # First six known pairs: same=1, cross=2, background=3. Unknown=4.
    scores = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.7, 0.1, 0.2],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
        ]
    )
    return target, ij, scores


def test_counts_unknown_background_and_permutation_invariance():
    target, ij, scores = fixture()
    m = classify(scores, target, ij)
    assert m["known_pairs"] == 6 and m["unknown_pairs"] == 4
    assert m["confusion"] == [[1, 0, 0], [0, 2, 0], [1, 0, 2]]
    assert m["background_false_B_pairs"] == 1 and m["background_pairs"] == 3
    assert m["conditional_same_cross"] == {"positive": 1, "negative": 2, "auc": 1.0}
    swapped = torch.where(target > 0, 3 - target, target)
    assert classify(scores, swapped, ij) == m
    assert classify(scores, target, ij.flip(0)) == m
    permutation = torch.tensor([3, 1, 4, 0, 2])
    inverse = torch.argsort(permutation)
    assert classify(scores, target[permutation], inverse[ij]) == m
    changed = scores.clone()
    unknown = (target[ij] < 0).any(0)
    changed[unknown] = torch.tensor([0.99, 0.005, 0.005])
    assert classify(changed, target, ij) == m


def event(category, metric, alternative=None):
    return {
        "category": category,
        "arms": {
            "detector_true": deepcopy(metric),
            "embedding_true": deepcopy(metric if alternative is None else alternative),
        },
    }


def test_aggregate_counts_before_division_and_missing_classes():
    m = {
        "confusion": [[1, 0, 0], [0, 2, 0], [1, 0, 2]],
        "unknown_pairs": 4,
        "conditional_same_cross": {"auc": 1.0},
    }
    n = {
        "confusion": [[0, 9, 0], [0, 0, 0], [0, 0, 0]],
        "unknown_pairs": 1,
        "conditional_same_cross": {"auc": None},
    }
    a = aggregate([event("charged", m), event("mixed", n)], "detector_true")
    assert a["class_support"] == [10, 2, 3]
    assert a["class_recall"] == [0.1, 1.0, 2 / 3]
    assert a["conditional_AUC_available_events"] == 1
    assert a["conditional_AUC_unavailable_events"] == 1
    assert a["unknown_pairs"] == 5
    assert (
        aggregate([event("charged", n)], "detector_true")["balanced_class_recall"]
        is None
    )


def test_whole_collision_bootstrap_not_pair_sample_bootstrap():
    target, ij, scores = fixture()
    good = classify(scores, target, ij)
    bad = classify(scores[:, [1, 0, 2]], target, ij)
    rows = [
        event("charged", good, bad),
        event("charged", bad, good),
        event("mixed", good, good),
        event("mixed", good, bad),
    ]
    original = paired_bootstrap(rows, "detector_true", "embedding_true", 64)
    multiplied = deepcopy(rows)
    for row in multiplied:
        for m in row["arms"].values():
            m["confusion"] = [[100 * v for v in line] for line in m["confusion"]]
            m["unknown_pairs"] *= 100
    expanded = paired_bootstrap(multiplied, "detector_true", "embedding_true", 64)
    # Multiplying each collision's pair count cannot increase bootstrap precision.
    assert expanded == original
    assert original["conditional_AUC_event_mean"]["available_replicates"] == 64
    assert "whole collision" in original["conditional_AUC_event_mean"]["unit"]
    assert "TRAIN-role" in original["conditional_AUC_event_mean"]["unit"]


def test_absent_support_remains_unavailable_and_zero_effect_is_descriptive():
    target, ij, scores = fixture()
    absent = classify(scores, torch.full_like(target, -1), ij)
    r = paired_bootstrap(
        [event("charged", absent)], "detector_true", "embedding_true", 32
    )
    for metric in r.values():
        assert metric["difference"] is None
        assert metric["interval95"] is None
        assert metric["available_replicates"] == 0
    known = classify(scores, target, ij)
    same = paired_bootstrap(
        [event("charged", known)], "detector_true", "embedding_true", 32
    )
    for metric in same.values():
        assert metric["difference"] == 0
        assert metric["interval95"] == [0, 0]
        # The output is explicitly adaptive training-role, not a population proof.
        assert "adaptive TRAIN-role assessment" in metric["unit"]


@pytest.mark.parametrize("wrong", ["root", "sha"])
def test_frozen_source_identity_rejected_before_loading_data(
    tmp_path, monkeypatch, wrong
):
    import json
    import scripts.run_phase84_information as runner

    c = contract()
    c["source_root"] = (
        str(runner.ROOT) if wrong != "root" else str(tmp_path / "other-tree")
    )
    c["source_sha"] = "a" * 40 if wrong != "sha" else "b" * 40
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(c))
    monkeypatch.setattr(runner, "guarded", lambda: None)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "a" * 40 + "\n"
    )
    monkeypatch.setattr(
        runner.torch,
        "load",
        lambda *a, **kw: pytest.fail("Input accessed before source guard"),
    )
    with pytest.raises(ValueError, match="Frozen source binding mismatch"):
        runner.main(path)


def test_checkpoint_history_rule_not_only_name(tmp_path):
    from scripts.run_phase84_information import validate_checkpoint

    c = contract()
    with pytest.raises(ValueError):
        validate_checkpoint(c, {"step": 1500, "settings": {"downstream_updates": 6000}})
    parent = tmp_path / "exposure.json"
    parent.write_text('{"kind":"phase84_exposure","pool_size":96}')
    b = {"path": str(parent), "sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}
    c["bindings"].append(b)
    ck = {"step": 6000, "settings": {"downstream_updates": 6000}, "contract": b}
    with pytest.raises(ValueError):
        validate_checkpoint(c, ck)
    parent.write_text('{"kind":"phase84_exposure","pool_size":1536}')
    b["sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
    validate_checkpoint(c, ck)
    c["checkpoint_selection"] = "fixed_historical_phase76_native"
    with pytest.raises(ValueError):
        validate_checkpoint(c, {"step": 1500})
