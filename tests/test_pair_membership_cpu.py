import pytest
import torch

from hypertagging.training.pair_membership import (
    pair_membership_loss,
    proposal_refinement_pair_loss,
)


def fixture():
    return (
        torch.tensor(
            [[0.3, 1.0, -0.5], [0.1, 1.4, -0.2], [0.2, -0.4, 1.2], [1.3, 0.2, -0.1]],
            requires_grad=True,
        ),
        torch.tensor([1, 1, 2, 0]),
        torch.eye(4, dtype=torch.bool),
    )


def test_pair_probabilities_match_enumeration_and_class_average():
    x, y, s = fixture()
    loss, c = pair_membership_loss(x, y, s)
    p = x.softmax(-1)
    expected = []
    for category in range(3):
        terms = []
        for i in range(4):
            for j in range(i + 1, 4):
                label = 2 if min(y[i], y[j]) == 0 else int(y[i] != y[j])
                if label == category:
                    probability = sum(
                        p[i, a] * p[j, b]
                        for a in range(3)
                        for b in range(3)
                        if (2 if min(a, b) == 0 else int(a != b)) == label
                    )
                    terms.append(-probability.log())
        expected.append(torch.stack(terms).mean())
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    assert [c[k] for k in ("same_b", "cross_b", "background")] == [1, 2, 3]
    loss.backward()
    assert torch.isfinite(x.grad).all() and (x.grad.abs().sum(1) > 0).all()


def test_slot_target_and_node_permutations_are_invariant():
    x, y, s = fixture()
    base, _ = pair_membership_loss(x, y, s)
    for xx, yy, ss in [
        (x[:, [0, 2, 1]], y, s),
        (x, torch.where(y == 0, y, 3 - y), s),
        (x.flip(0), y.flip(0), s.flip(0)),
    ]:
        value, _ = pair_membership_loss(xx, yy, ss)
        torch.testing.assert_close(value, base)


def test_unknown_and_source_overlap_are_excluded_not_background():
    x, y, s = fixture()
    y[3] = -1
    s[1] = s[0]
    loss, c = pair_membership_loss(x, y, s)
    assert c == {
        "same_b": 0,
        "cross_b": 2,
        "background": 0,
        "unknown_pairs": 3,
        "shared_source_pairs": 1,
    }
    loss.backward()
    assert x.grad[3].abs().sum() == 0


def test_empty_support_and_extreme_logits_finite():
    x, y, s = fixture()
    y.fill_(-1)
    loss, _ = pair_membership_loss(x, y, s)
    loss.backward()
    assert loss == 0 and x.grad.abs().sum() == 0
    x, y, s = fixture()
    x = (x.detach() * 10000).requires_grad_()
    loss, _ = pair_membership_loss(x, y, s)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_both_head_branches_receive_gradient():
    x, y, s = fixture()
    z = x.detach().clone().requires_grad_()
    loss, _ = proposal_refinement_pair_loss(
        {"proposal_logits": x[None], "logits": z[None]}, y, s
    )
    loss.backward()
    assert x.grad.abs().sum() > 0 and z.grad.abs().sum() > 0


def test_bad_contracts_fail():
    x, y, s = fixture()
    for xx, yy, ss in [
        (x[:, :2], y, s),
        (x, y[:-1], s),
        (x, y.float(), s),
        (x, torch.tensor([1, 2, 3, 0]), s),
        (x, y, torch.zeros(4, 4, dtype=torch.bool)),
    ]:
        with pytest.raises(ValueError):
            pair_membership_loss(xx, yy, ss)


def test_study_contract_rejects_wrong_arms_history_resources_and_validation():
    import copy
    from scripts.phase81_pair_supervision import (
        ARMS,
        RESOURCES,
        INITIAL_SHA256,
        settings,
        validate,
    )

    c = dict(
        arms=list(ARMS),
        settings=settings(),
        resources=RESOURCES,
        initial_checkpoint={"sha256": INITIAL_SHA256},
        stage="development_training_screen",
        heldout_count=0,
        training_count=1536,
        automatic_successor=False,
        sealed_test_access=False,
        arm_pair_supervision={"native": False, "pair_supervised": True},
    )
    validate(c)
    for key, value in [
        ("arms", ["native", "native"]),
        ("heldout_count", 600),
        ("training_count", 384),
        ("automatic_successor", True),
        ("initial_checkpoint", {"sha256": "wrong"}),
        ("settings", {**settings(), "downstream_updates": 1501}),
        ("resources", {**RESOURCES, "gpus": 1}),
        ("arm_pair_supervision", {"native": True, "pair_supervised": True}),
    ]:
        bad = copy.deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            validate(bad)


def test_control_does_not_call_pair_objective_and_same_init_replays(
    tmp_path, monkeypatch
):
    from collections import Counter
    from scripts.run_phase74_development import fit

    from scripts.phase81_pair_supervision import settings
    import inspect

    # Fail-closed before fitting for incompatible controls; baseline default is false.
    assert inspect.signature(fit).parameters["pair_supervision"].default is False
    with pytest.raises(ValueError, match="downstream joint"):
        fit(
            None,
            None,
            [],
            stage="pretraining",
            objective="existing",
            settings=settings(),
            output=tmp_path,
            compute=Counter(),
            cache_binding={},
            source_sha="test",
            pair_supervision=True,
        )
    with pytest.raises(ValueError, match="downstream joint"):
        fit(
            None,
            None,
            [],
            stage="tiny",
            objective="existing",
            settings=settings(),
            output=tmp_path,
            compute=Counter(),
            cache_binding={},
            source_sha="test",
            gradient_rule="project_conflicting_relation",
            pair_supervision=True,
        )


def test_training_manifest_membership_does_not_reorder_historical_cache():
    from scripts.phase81_pair_supervision import authenticate_training_rows

    cats = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
    rows = [{"uid": str(i), "category": cats[i // 256]} for i in range(1536)]
    original = [dict(r) for r in rows]
    manifest = {"event_uids": [r["uid"] for r in reversed(rows)]}
    digest = authenticate_training_rows(rows, manifest)
    assert rows == original
    assert digest != authenticate_training_rows(list(reversed(rows)), manifest)
    for bad in (
        rows[:-1],
        rows[:-1] + [rows[0]],
        [{**r, "category": "charged"} for r in rows],
    ):
        with pytest.raises(ValueError):
            authenticate_training_rows(bad, manifest)


def test_native_source_masks_ignore_column_naming_and_count_aliases():
    x, y, s = fixture()
    value, counts = pair_membership_loss(x, y, s)
    other, other_counts = pair_membership_loss(x, y, s[:, [2, 0, 3, 1]])
    torch.testing.assert_close(value, other)
    assert counts == other_counts
    s[1] = s[0]
    _, counts = pair_membership_loss(x, y, s)
    assert counts["shared_source_pairs"] == 1 and counts["same_b"] == 0
    for bad in (s.float(), s[:, 0], [[0], [1], [2], [3]]):
        with pytest.raises(ValueError):
            pair_membership_loss(x, y, bad)
