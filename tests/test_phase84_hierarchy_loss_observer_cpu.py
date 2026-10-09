"""Observer regression: exact native values, graph allocation and no fitting."""

import copy
from types import SimpleNamespace

import pytest
import torch

from hypertagging.losses.level_reconstruction import LevelLossOutput
from scripts.phase84_hierarchy_loss_observer import (
    capture_native_losses,
    observe_optimization_loss,
)


def fixture_trainer():
    trainer = SimpleNamespace()

    def loss(pointer, batch, *, target_level, **kwargs):
        p = pointer
        zero = p * 0
        components = {
            "object": (p - target_level).square(),
            "type": (p + target_level).square(),
            "pointer": p.square(),
            "zero_a": zero,
            "zero_b": zero,
        }
        total = (
            components["object"]
            + 2 * components["type"]
            + 0.5 * components["pointer"]
            + 3 * zero
            + 4 * zero
        )
        return LevelLossOutput(total, components, [[(0, 0)]])

    def mean(losses, weights):
        return sum(x * w for x, w in zip(losses, weights)) / sum(weights)

    def optimization(model, batch, **kwargs):
        a = trainer.level_reconstruction_loss(
            model.p, batch, target_level=1, target_override=None
        )
        b = trainer.level_reconstruction_loss(
            model.p, batch, target_level=2, target_override=([torch.tensor(1)],)
        )
        auxiliary = trainer.level_reconstruction_loss(
            model.p, batch, target_level=2, target_override=None
        )
        primary = trainer._normalized_weighted_mean(
            [a.total + 0.3 * model.p.square(), b.total + model.p * 0], [1, 3]
        )
        aux = trainer._normalized_weighted_mean([auxiliary.total], [3])
        with torch.no_grad():
            trainer.level_reconstruction_loss(model.p, batch, target_level=1)
        return primary + 0.5 * aux, 0.1 * model.p.square(), {}, [], {}

    trainer.level_reconstruction_loss = loss
    trainer._normalized_weighted_mean = mean
    trainer._optimization_loss = optimization
    return trainer


def model():
    m = torch.nn.Module()
    m.p = torch.nn.Parameter(torch.tensor(0.4))
    return m


def test_exact_native_allocation_auxiliary_recovery_and_leaf():
    trainer = fixture_trainer()
    m = model()
    before = m.p.detach().clone()
    rng = torch.get_rng_state().clone()
    original_loss, original_mean = (
        trainer.level_reconstruction_loss,
        trainer._normalized_weighted_mean,
    )
    baseline = trainer._optimization_loss(m, {})
    expected = torch.autograd.grad(baseline[0] + baseline[1], m.p)[0]
    result, report, capture = observe_optimization_loss(
        m,
        {},
        native_kwargs={},
        parameter_groups={"all": list(m.named_parameters())},
        trainer=trainer,
    )
    torch.testing.assert_close(result[0] + result[1], baseline[0] + baseline[1])
    assert (
        trainer.level_reconstruction_loss is original_loss
        and trainer._normalized_weighted_mean is original_mean
    )
    assert [c["effective_native_coefficient"] for c in report["calls"]] == [
        0.25,
        0.75,
        0.5,
        None,
    ]
    assert [c["role"] for c in report["calls"]] == [
        "primary_derived_normalized_loss",
        "primary_derived_normalized_loss",
        "auxiliary_direct_normalized_loss",
        "detached_readout",
    ]
    assert report["normalizations"][0]["weights"] == [1, 3]
    assert report["calls"][0]["components"][-1]["names"] == ["zero_a", "zero_b"]
    assert report["calls"][0]["components"][-1]["native_component_coefficient"] == 7
    assert report["unallocated_recovery_or_other_residual"]["value"] == pytest.approx(
        0.3 * float(m.p**2) / 4, abs=2e-6
    )
    assert report["replay"]["gradient_sum"] == "PASS"
    assert (
        m.p.grad is None
        and torch.equal(before, m.p)
        and torch.equal(rng, torch.get_rng_state())
    )
    # Graph survives observation for the caller's own backward/reference check.
    torch.testing.assert_close(
        torch.autograd.grad(result[0] + result[1], m.p)[0], expected
    )
    assert capture["calls"][0]["pointer_output"] is m.p
    assert capture["calls"][0]["posthoc"]["pointer_output"] is not m.p
    assert capture["calls"][0]["posthoc"]["pointer_output"].requires_grad is False


def test_failure_restores_wrappers_and_return_identity():
    trainer = fixture_trainer()
    orig_loss, orig_mean = (
        trainer.level_reconstruction_loss,
        trainer._normalized_weighted_mean,
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        with capture_native_losses(trainer) as (calls, norms):
            returned = trainer.level_reconstruction_loss(
                torch.tensor(1.0, requires_grad=True), {}, target_level=1
            )
            assert returned is calls[0]["output"]
            raise RuntimeError("deliberate")
    assert (
        trainer.level_reconstruction_loss is orig_loss
        and trainer._normalized_weighted_mean is orig_mean
    )


def test_parameter_groups_reject_duplicates_and_frozen():
    trainer = fixture_trainer()
    m = model()
    with pytest.raises(ValueError, match="Overlapping"):
        observe_optimization_loss(
            m,
            {},
            native_kwargs={},
            parameter_groups={
                "a": list(m.named_parameters()),
                "b": list(m.named_parameters()),
            },
            trainer=trainer,
        )
    frozen = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
    with pytest.raises(ValueError, match="frozen"):
        observe_optimization_loss(
            m,
            {},
            native_kwargs={},
            parameter_groups={"a": [("frozen", frozen)]},
            trainer=trainer,
        )


def test_real_tiny_native_mixed_context_objective_preserved():
    from hypertagging.data.heterogeneous import (
        collate_heterogeneous_events,
        heterogeneous_from_level_event,
    )
    from hypertagging.data.tiny_level_fixtures import tiny_level_events
    from hypertagging.models.level_autoregressive import (
        LevelAutoregressiveReconstructor,
    )
    from hypertagging.preprocessing.pid_filter import PDG_TOKENS
    from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
    from hypertagging.training import reconstruction_trainer as trainer
    from hypertagging.training.scheduled_sampling import TeacherForcingSchedule

    torch.manual_seed(84)
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    base = LevelAutoregressiveReconstructor(
        n_features=batch["node_features"].shape[-1],
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
    )
    observed = copy.deepcopy(base)
    config = trainer.ReconstructionConfig(
        data="unused",
        output_dir="unused",
        auxiliary_teacher_weight=0.5,
        pointer_set_overlap_weight=1.0,
        unrepresentable_target_policy="masked_representable_only",
        recovery_objective_weight=0.0,
    )
    kwargs = dict(
        valid_levels=[1],
        config=config,
        schedule=TeacherForcingSchedule(
            start_probability=0.0, end_probability=0.0, duration_steps=2188
        ),
        step=3000,
        use_scheduled_sampling=True,
        allowed_types_by_level={},
        constraint_policy=ReconstructionConstraintPolicy(),
    )
    torch.manual_seed(85)
    native = trainer._optimization_loss(base, batch, **kwargs)
    native_grad = torch.autograd.grad(
        native[0] + native[1], tuple(base.parameters()), allow_unused=True
    )
    torch.manual_seed(85)
    result, report, capture = observe_optimization_loss(
        observed,
        batch,
        native_kwargs=kwargs,
        parameter_groups={"native_all": list(observed.named_parameters())},
    )
    torch.testing.assert_close(
        result[0] + result[1], native[0] + native[1], rtol=0, atol=0
    )
    assert result[-1] == native[-1]
    actual = torch.autograd.grad(
        result[0] + result[1], tuple(observed.parameters()), allow_unused=True
    )
    for a, b in zip(actual, native_grad):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(p.grad is None for p in observed.parameters())
    assert report["replay"]["gradient_sum"] == "PASS"
    assert any(c["role"] == "primary_derived_normalized_loss" for c in report["calls"])
    assert any(c["role"] == "auxiliary_direct_normalized_loss" for c in report["calls"])
    assert any(c["role"] == "detached_readout" for c in report["calls"])


def test_matched_logit_reference_supports_local_gradients_without_encoder_backward():
    from hypertagging.losses.level_reconstruction import LevelLossOutput
    from types import SimpleNamespace

    type_logits = torch.tensor(
        [[[0.2, 1.0, -0.5], [0.7, -0.3, 0.2]]], requires_grad=True
    )
    pointer_logits = torch.tensor([[[0.3, -0.2], [0.4, 0.5]]], requires_grad=True)
    output = SimpleNamespace(type_logits=type_logits, pointer_logits=pointer_logits)
    trainer = fixture_trainer()

    def local_loss(pointer, batch, **kwargs):
        type_loss = torch.nn.functional.cross_entropy(
            pointer.type_logits[0, :1], torch.tensor([1])
        )
        pointer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pointer.pointer_logits[0, 0], torch.tensor([1.0, 0.0])
        )
        return LevelLossOutput(
            type_loss + pointer_loss,
            {"type": type_loss, "pointer": pointer_loss},
            [[(0, 0)]],
        )

    trainer.level_reconstruction_loss = local_loss
    with capture_native_losses(trainer, capture_tensors=False) as (calls, _):
        loss = trainer.level_reconstruction_loss(output, {}, target_level=2)
    assert calls[0]["pointer_output"] is output
    assert calls[0]["output"] is loss
    grad_type, grad_pointer = torch.autograd.grad(
        loss.total,
        (
            calls[0]["pointer_output"].type_logits,
            calls[0]["pointer_output"].pointer_logits,
        ),
    )
    assert grad_type[0, 0].abs().sum() > 0 and grad_type[0, 1].abs().sum() == 0
    assert grad_pointer[0, 0].abs().sum() > 0 and grad_pointer[0, 1].abs().sum() == 0
    assert type_logits.grad is pointer_logits.grad is None


def matched_call(order=(0, 1), unknown=False, source=True, grad_enabled=True):
    from types import SimpleNamespace
    from scripts.phase84_hierarchy_loss_observer import matched_query_logit_diagnostics

    logits = torch.tensor([[[0.1, 1.1, -0.4], [0.3, -0.2, 0.8]]], requires_grad=True)[
        :, list(order)
    ]
    pointers = torch.tensor(
        [[[1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]], requires_grad=True
    )[:, list(order)]
    types = torch.tensor([-1 if unknown else 1, 2])
    masks = torch.tensor([[True, True, False, False], [False, False, True, True]])
    matches = [(q, target) for q, target in enumerate(order)]
    type_loss = torch.stack(
        [
            torch.nn.functional.cross_entropy(
                logits[0, q][None], types[t].clamp_min(0)[None]
            )
            for q, t in matches
        ]
    ).mean()
    from hypertagging.losses.level_reconstruction import (
        focal_binary_cross_entropy_with_logits,
    )

    pointer_loss = torch.stack(
        [
            focal_binary_cross_entropy_with_logits(
                pointers[0, q], masks[t].float(), positive_weight=4.0, gamma=1.0
            )
            for q, t in matches
        ]
    ).mean()
    loss = LevelLossOutput(
        type_loss + pointer_loss,
        {"type": type_loss, "pointer": pointer_loss},
        [matches],
    )
    batch = {
        "node_mask": torch.ones(1, 4, dtype=torch.bool),
        "level_ids": torch.zeros(1, 4, dtype=torch.long),
    }
    if source:
        batch["recursive_leaf_source_mask"] = torch.tensor(
            [[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.bool
        )
    call = {
        "pointer_output": SimpleNamespace(type_logits=logits, pointer_logits=pointers),
        "output": loss,
        "loss_batch": batch,
        "controls": {},
        "target_level": 1,
        "target_override_values": ([types], [masks], [], []),
        "grad_enabled": grad_enabled,
    }
    return matched_query_logit_diagnostics(call, 0.25)


def test_matched_query_logit_pid_size_cells_are_query_permutation_invariant():
    first, second = matched_call(), matched_call((1, 0))
    a = {r["pid_token"]: r for r in first["rows"]}
    b = {r["pid_token"]: r for r in second["rows"]}
    assert a.keys() == b.keys()
    assert a[1]["daughter_count"] == 2 and a[1]["recursive_source_count"] == 1
    assert a[2]["daughter_count"] == a[2]["recursive_source_count"] == 2
    for token in a:
        assert a[token]["local_type_ce"] == b[token]["local_type_ce"]
        assert (
            a[token]["local_pointer_focal_loss"] == b[token]["local_pointer_focal_loss"]
        )
        assert a[token]["gradients"] == b[token]["gradients"]
        assert a[token]["gradients"]["type"]["l2"] > 0


def test_matched_query_unknown_pid_and_missing_sources_remain_unavailable():
    report = matched_call(unknown=True, source=False)
    assert report["rows"][0]["pid_status"] == "UNAVAILABLE_INVALID_TARGET_TOKEN"
    assert (
        report["rows"][0]["pid_token"]
        is report["rows"][0]["signed_pdg"]
        is report["rows"][0]["local_type_ce"]
        is None
    )
    assert all(
        r["recursive_source_count"] is None
        and r["source_status"] == "UNAVAILABLE_SOURCE_MASK"
        for r in report["rows"]
    )


def test_matched_query_detached_readout_gradient_is_unavailable():
    report = matched_call(grad_enabled=False)
    for row in report["rows"]:
        for item in row["gradients"].values():
            assert item["status"] == "UNAVAILABLE_DETACHED_OR_UNUSED_GRAPH"
            assert item["l1"] is item["l2"] is None
