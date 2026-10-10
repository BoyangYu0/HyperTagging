"""Information-probe leakage, symmetry, normalization and matched-fit guards."""

import pytest
import torch
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
from scripts.phase84_information_probe import (
    BLOCKS,
    KINDS,
    MODES,
    detector_nodes,
    fit_normalizer,
    normalize,
    pair_inputs,
    pair_labels,
    shuffled_targets,
    new_probe,
    fit_probe,
    score_pairs,
)


def detector():
    d = {
        "node_mask": torch.ones(1, 3, dtype=torch.bool),
        "node_kind_ids": torch.tensor([KINDS]),
        "leaf_kinematics_mode_ids": torch.tensor(
            [[MODES[0], LEAF_MODE_TO_ID["ecl_cluster"], LEAF_MODE_TO_ID["klm_cluster"]]]
        ),
        "pid_labels": torch.tensor([[0, 2, 3]]),
    }
    for block, width in BLOCKS:
        x = torch.arange(3 * width).reshape(1, 3, width).float()
        a = torch.ones_like(x, dtype=torch.bool)
        if block == "common":
            x[:, :, 10] = 0
            a[:, :, [6, 7, 8, 9]] = False
        else:
            a[:] = False
            a[:, ("track", "cluster", "klm").index(block), :] = True
        d[block + "_features"], d[block + "_availability"] = x, a
    return d


def test_lossless_detector_allowlist_masks_and_flags():
    d = detector()
    x, a, continuous = detector_nodes(d)
    assert x.shape == a.shape == (3, 128)
    assert int(continuous.sum()) == 41
    assert torch.equal(x[:, 41:82], a[:, :41].float())
    d["truth_pid_labels"] = torch.full((1, 3), 999)
    d["b_side"] = torch.tensor([[2, 1, 1]])
    d["parent_ids"] = torch.tensor([[999, 888, 777]])
    assert torch.equal(detector_nodes(d)[0], x)
    for block, _ in BLOCKS:
        d[block + "_features"][~d[block + "_availability"]] = float("nan")
    assert torch.equal(detector_nodes(d)[0], x)


@pytest.mark.parametrize("track_mode", MODES)
def test_native_cluster_modes_lossless_categorical_roundtrip(track_mode):
    d = detector()
    d["leaf_kinematics_mode_ids"][0, 0] = track_mode
    d["pid_labels"][0, 0] = 0 if track_mode == MODES[0] else 2
    x, observed, continuous = detector_nodes(d)
    assert x.shape == observed.shape == (3, 128)
    assert int(continuous.sum()) == 41
    kinds = torch.tensor(KINDS)[x[:, 82:85].argmax(-1)]
    track_flags = x[:, 85:87]
    assert not track_flags[1:].any()
    modes = torch.where(
        kinds == KINDS[0],
        torch.tensor(MODES)[track_flags.argmax(-1)],
        torch.where(
            kinds == KINDS[1],
            LEAF_MODE_TO_ID["ecl_cluster"],
            LEAF_MODE_TO_ID["klm_cluster"],
        ),
    )
    assert torch.equal(kinds, d["node_kind_ids"][0])
    assert torch.equal(modes, d["leaf_kinematics_mode_ids"][0])
    assert torch.equal(x[:, 87:].argmax(-1), d["pid_labels"][0])
    assert sum(p.numel() for p in new_probe(8409).parameters()) == 24835


@pytest.mark.parametrize("node", range(3))
@pytest.mark.parametrize("mode", list(LEAF_MODE_TO_ID.values()))
def test_exact_native_mode_kind_pairs(node, mode):
    d = detector()
    d["leaf_kinematics_mode_ids"][0, node] = mode
    allowed = (MODES, (LEAF_MODE_TO_ID["ecl_cluster"],), (LEAF_MODE_TO_ID["klm_cluster"],))
    if mode in allowed[node]:
        detector_nodes(d)
    else:
        with pytest.raises(ValueError, match="kind/kinematics mode pairing"):
            detector_nodes(d)


@pytest.mark.parametrize("kind", ["unknown", "other", "composite"])
def test_non_detector_kinds_rejected(kind):
    d = detector()
    d["node_kind_ids"][0, 0] = NODE_KIND_TO_ID[kind]
    with pytest.raises(ValueError, match="Unsupported reconstructed category"):
        detector_nodes(d)


@pytest.mark.parametrize(
    "change", ["categorical", "composite", "mode", "rawpid", "kindmask", "nonfinite"]
)
def test_reject_nonprojected_or_invalid_inputs(change):
    d = detector()
    if change == "categorical":
        d["common_availability"][0, 0, 6] = True
    elif change == "composite":
        d["common_features"][0, 0, 10] = 2
    elif change == "mode":
        d["leaf_kinematics_mode_ids"][0, 0] = 3
    elif change == "rawpid":
        d["pid_labels"][0, 0] = 2
    elif change == "kindmask":
        d["track_availability"][0, 2, 0] = True
    else:
        d["track_features"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        detector_nodes(d)


def records():
    x, a, c = detector_nodes(detector())
    return [
        {
            "uid": role,
            "source_role": "train",
            "partition": role,
            "detector": x.clone(),
            "detector_observed": a.clone(),
            "detector_continuous": c.clone(),
        }
        for role in ("fit", "assessment")
    ]


def test_normalization_fit_only_missing_and_flags():
    r = records()
    s = fit_normalizer(r, "detector")
    r[1]["detector"] *= 100000
    again = fit_normalizer(r, "detector")
    assert torch.equal(s["mean"], again["mean"])
    assert torch.equal(s["scale"], again["scale"])
    norm = normalize(r[0]["detector"], r[0]["detector_observed"], s)
    assert torch.equal(norm[:, 41:], r[0]["detector"][:, 41:])
    assert not norm[~r[0]["detector_observed"]].any()
    r[1]["source_role"] = "validation"
    with pytest.raises(ValueError):
        fit_normalizer(r, "detector")


def test_pair_symmetry_permutation_and_targets():
    h = torch.randn(4, 128)
    ij = torch.tensor([[0, 1, 2], [1, 2, 3]])
    x = pair_inputs(h, ij)
    assert torch.equal(x, pair_inputs(h, ij.flip(0)))
    p = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(p)
    assert torch.allclose(x, pair_inputs(h[p], inverse[ij]), atol=1e-6)
    target = torch.tensor([1, 1, 2, 0])
    assert pair_labels(target, ij).tolist() == [0, 1, 2]
    assert torch.equal(
        pair_labels(target, ij),
        pair_labels(torch.where(target > 0, 3 - target, target), ij),
    )
    target[2] = -1
    assert pair_labels(target, ij).tolist() == [0, -1, -1]
    shuffled = shuffled_targets(target, "event", 9)
    assert torch.equal(shuffled == -1, target == -1)
    assert sorted(shuffled.tolist()) == sorted(target.tolist())
    assert torch.equal(shuffled, shuffled_targets(target, "event", 9))


def test_finite_matched_fit_gradient_and_no_assessment_access():
    torch.manual_seed(7)
    events = [
        {
            "uid": str(i),
            "source_role": "train",
            "partition": "fit",
            "x": torch.randn(12, 384),
            "y": torch.arange(12) % 3,
        }
        for i in range(2)
    ]
    m, report = fit_probe(events, seed=8, updates=2, batch_size=6)
    n, second = fit_probe(events, seed=8, updates=2, batch_size=6)
    assert report == second
    assert report["parameters"] == 24835
    assert report["pair_presentations"] == 12
    assert any(
        not torch.equal(a, b) for a, b in zip(m.parameters(), new_probe(8).parameters())
    )
    scores = score_pairs(m, events[0]["x"])
    assert scores.shape == (12, 3) and not scores.requires_grad
    assert torch.equal(scores, score_pairs(n, events[0]["x"]))
    events[0]["partition"] = "assessment"
    with pytest.raises(ValueError):
        fit_probe(events, seed=8, updates=2, batch_size=6)
    with pytest.raises(ValueError):
        fit_probe(events, seed=8, updates=4097, batch_size=6)


def test_source_alias_exclusion_and_missing_support():
    from scripts.phase84_information_probe import source_pairs

    sources = torch.tensor([[True, False], [True, False], [False, True]])
    ij, counts = source_pairs(sources)
    assert ij.tolist() == [[0, 1], [2, 2]]
    assert counts == {
        "possible_pairs": 3,
        "source_alias_excluded": 1,
        "eligible_pairs": 2,
    }
    with pytest.raises(ValueError):
        source_pairs(torch.zeros(2, 2, dtype=torch.bool))
