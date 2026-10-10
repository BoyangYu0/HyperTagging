"""Bounded native hierarchy diagnostic orchestration and failure contracts."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_phase84_hierarchy_applicability as runner


def contract():
    return {
        "kind": "phase84_native_hierarchy_applicability",
        "panel_events": 96,
        "gradient_events": 24,
        "parameter_updates": 0,
        "resources": dict(cpus=2, memory_gib=32, hours=4, gpus=0, requeue=False),
        "validation_events": 0,
        "sealed_test_access": False,
        "automatic_successor": False,
        "smoke": True,
        "runtime_guard_seconds": 10800,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("panel_events", 24),
        ("gradient_events", 96),
        ("parameter_updates", 1),
        ("validation_events", 1),
        ("sealed_test_access", True),
        ("automatic_successor", True),
        ("smoke", 1),
        ("runtime_guard_seconds", 14400),
        ("resources", dict(cpus=2, memory_gib=32, hours=8, gpus=0, requeue=False)),
    ],
)
def test_contract_rejects_scope_resource_and_role_changes(key, value):
    candidate = contract()
    runner.validate_contract(candidate)
    candidate[key] = value
    with pytest.raises(ValueError):
        runner.validate_contract(candidate)


def native_document():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/reconstruction/ht_reconstruction_phase72_20261007.json"
    )
    study = json.loads(path.read_text())
    return {
        **study["common_training_contract"],
        "data": "fixture-unused",
        "output_dir": "fixture-unused",
        "pointer_set_overlap_weight": 0.0,
    }


def test_native_config_retains_objective_and_rejects_other_contract():
    original = native_document()
    cfg = runner.native_config(original)
    assert cfg.device == "cpu" and cfg.mixed_precision is False
    assert cfg.auxiliary_teacher_weight == original["auxiliary_teacher_weight"]
    assert dict(cfg.level_loss_weights) == {1: 1, 2: 1, 3: 1.25, 4: 1.5, 5: 2, 6: 3}
    for key, value in [
        ("unknown_knob", 1),
        ("level_sampling_mode", "all_levels"),
        ("target_policy", "reconstructable_partial"),
        ("pointer_set_overlap_weight", 1),
        ("recovery_objective_weight", 1),
    ]:
        changed = copy.deepcopy(original)
        changed[key] = value
        with pytest.raises(ValueError):
            runner.native_config(changed)


def tiny_context():
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

    torch.manual_seed(84)
    batch = collate_heterogeneous_events(
        [heterogeneous_from_level_event(tiny_level_events()[0])]
    )
    model = LevelAutoregressiveReconstructor(
        n_features=batch["node_features"].shape[-1],
        n_types=len(PDG_TOKENS),
        hidden_dim=16,
        hyper_dim=4,
        n_queries=2,
    ).eval()
    context = SimpleNamespace(
        model=model,
        checkpoint={"step": 4376, "confidence_head_trained": True},
        allowed_types_by_level={},
        constraint_policy=ReconstructionConstraintPolicy(),
    )
    return context, batch, runner.native_config(native_document())


def test_freeze_boundary_is_last_zero_based_optimizer_step():
    context, _, cfg = tiny_context()
    cfg = replace(
        cfg, freeze_pretrained_encoder_steps=4376, freeze_leaf_pid_head_steps=4376
    )
    groups = runner.trainable_groups(context.model, cfg, 4376)
    assert "encoder" not in groups
    assert all(not p.requires_grad for p in context.model.encoder.parameters())
    assert all(not p.requires_grad for p in context.model.leaf_pid_head.parameters())
    groups = runner.trainable_groups(context.model, cfg, 4377)
    assert "encoder" in groups
    assert all(p.requires_grad for p in context.model.parameters())


def test_gradient_panel_is_fixed_balanced_and_order_invariant():
    cats = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
    events = [
        SimpleNamespace(source_category=c, event_uid=f"{c}-{i}")
        for c in cats
        for i in range(16)
    ]
    chosen = runner.gradient_panel(events)
    assert len(chosen) == 24 and chosen == runner.gradient_panel(list(reversed(events)))
    assert all(
        sum(e.event_uid in chosen for e in events if e.source_category == c) == 4
        for c in cats
    )
    with pytest.raises(ValueError):
        runner.gradient_panel(events[:-1])


def test_repeated_views_preserve_inputs_and_mark_explicit_heights():
    _, batch, _ = tiny_context()
    before = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    repeated = runner.repeated_level_batch(batch, [1, 3, 6])
    assert repeated["selected_target_levels"].tolist() == [1, 3, 6]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key])
            if value.ndim and value.shape[0] == 1:
                assert repeated[key].shape[0] == 3
                assert all(torch.equal(repeated[key][i], value[0]) for i in range(3))


@pytest.mark.parametrize("label", ["teacher", "generated"])
@pytest.mark.parametrize("full_gradients", [False, True])
def test_real_tiny_native_objective_serialization_and_replay(
    tmp_path, label, full_gradients
):
    context, batch, cfg = tiny_context()
    inventory = runner.target_inventory(batch)
    heights = sorted({x["height"] for x in inventory if x["policy"] == "complete_only"})
    assert heights == [1]
    repeated = runner.repeated_level_batch(batch, heights)
    groups = runner.trainable_groups(context.model, cfg, 4376)
    before = runner.model_digest(context.model)
    report = runner.objective_audit(
        context, repeated, cfg, groups, tmp_path, label, full_gradients, replay=True
    )
    assert (
        report["real_native_observer_replay"] == "EXACT_SCALAR_AND_PARAMETER_GRADIENTS"
    )
    assert json.loads((tmp_path / f"{label}-objective.json").read_text()) == report
    assert runner.model_digest(context.model) == before
    assert all(p.grad is None for p in context.model.parameters())
    assert bool(report["generated_traces_before_join"]) == (label == "generated")
    if label == "generated":
        assert all(
            not json.loads(Path(b["path"]).read_text())["truth_used_for_generation"]
            for b in report["generated_traces_before_join"]
        )
    if full_gradients:
        assert report["replay"]["gradient_sum"] == "PASS"
    else:
        assert report["parameter_gradients"] == "UNAVAILABLE_BY_DESIGN_24_EVENT_SUBSET"
    assert all(
        row["scope"].startswith("Matched-query local")
        for row in report["matched_query_logit_diagnostics"]
    )


def test_rollout_configuration_rejects_untrained_confidence():
    context, _, cfg = tiny_context()
    context.checkpoint["confidence_head_trained"] = False
    with pytest.raises(ValueError, match="confidence"):
        runner.native_rollout_config(
            context, replace(cfg, rollout_use_learned_confidence=True)
        )


def historical_payload():
    return {
        "step": 5,
        "streaming_cursor": {"events_consumed": 20, "batch_index": 5},
        "data_order_contract": {
            "balanced_level_replay": {
                "level_schedule": "global_slot_round_robin",
                "levels": [1, 2, 3, 4, 5, 6],
                "eligible_pool_counts_by_level": {str(h): 2 for h in range(1, 7)},
            }
        },
    }


def test_historical_exposure_verified_cursor_counts_round_robin_remainder():
    result = runner.historical_exposure(
        historical_payload(), SimpleNamespace(batch_size=4)
    )
    assert result["status"] == "VERIFIED_CHECKPOINT_CURSOR_AND_ROUND_ROBIN_SLOTS"
    assert result["presentations"] == 20
    assert result["checkpoint_updates"] == 5
    assert result["slot_counts_by_height"] == {
        "1": 4,
        "2": 4,
        "3": 3,
        "4": 3,
        "5": 3,
        "6": 3,
    }
    assert sum(result["slot_counts_by_height"].values()) == result["presentations"]
    assert result["mean_presentations_per_eligible_event_by_height"]["1"] == 2
    assert result["mean_presentations_per_eligible_event_by_height"]["6"] == 1.5
    assert "per-PID gradient histories are not reconstructed" in result["unavailable"]


@pytest.mark.parametrize("field,value", [("events_consumed", 19), ("batch_index", 4)])
def test_historical_exposure_rejects_cursor_mismatch(field, value):
    payload = historical_payload()
    payload["streaming_cursor"][field] = value
    with pytest.raises(ValueError, match="cursor/update mismatch"):
        runner.historical_exposure(payload, SimpleNamespace(batch_size=4))


@pytest.mark.parametrize("field", ["streaming_cursor", "data_order_contract"])
def test_historical_exposure_missing_metadata_is_unavailable(field):
    payload = historical_payload()
    payload.pop(field)
    assert runner.historical_exposure(payload, SimpleNamespace(batch_size=4)) == {
        "status": "UNAVAILABLE_CHECKPOINT_SAMPLER_OR_CURSOR"
    }


@pytest.mark.parametrize(
    "field,value", [("level_schedule", "random"), ("levels", [1, 2])]
)
def test_historical_exposure_rejects_unsupported_height_schedule(field, value):
    payload = historical_payload()
    payload["data_order_contract"]["balanced_level_replay"][field] = value
    with pytest.raises(ValueError, match="historical height sampling"):
        runner.historical_exposure(payload, SimpleNamespace(batch_size=4))


@pytest.mark.parametrize("last_has_no_eligible_height", [False, True])
def test_runner_mocked_authenticated_loader_smoke_terminal_and_accounting(
    tmp_path, monkeypatch, last_has_no_eligible_height
):
    """Synthetic loader/strict-evaluator seam; native objective remains real."""
    from scripts import phase84_hierarchy_train_context as loader
    from scripts import phase84_hierarchy_trace as trace
    from hypertagging.reconstruction import hierarchical_inference as inference
    from hypertagging.evaluation import tag_efficiency as tagging
    from hypertagging.evaluation import full_decay_runner as full_decay

    context, batch, cfg = tiny_context()
    cats = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
    context.events = [
        SimpleNamespace(source_category=c, event_uid=f"fixture-{c}-{i}")
        for c in cats
        for i in range(16)
    ]
    context.config = native_document()
    context.metadata = {"scope": "synthetic mocked loader; not real-data admission"}

    def event_batch(index):
        copied = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        if (
            last_has_no_eligible_height
            and context.events[index].event_uid == "fixture-ccbar-0"
        ):
            copied["valid_reconstruction_target"].zero_()
        return copied

    context.collated_event_batch = event_batch
    monkeypatch.setattr(
        loader, "load_hierarchy_train_context", lambda **kwargs: context
    )
    monkeypatch.setattr(
        trace,
        "capture_native_trace",
        lambda *args, **kwargs: (
            None,
            None,
            {"truth_used_for_generation": False, "scope": "synthetic trace seam"},
        ),
    )
    monkeypatch.setattr(
        trace,
        "evaluate_native_trace",
        lambda *args, **kwargs: {"scope": "synthetic join seam"},
    )
    monkeypatch.setattr(
        inference,
        "reconstruct_full_tree_from_fsps",
        lambda *args, **kwargs: SimpleNamespace(
            rollout=SimpleNamespace(batch={}, event_valid_mask=torch.tensor([True]))
        ),
    )
    monkeypatch.setattr(
        tagging,
        "evaluate_tag_efficiency_event",
        lambda *args, **kwargs: {"category": kwargs["source_category"]},
    )
    monkeypatch.setattr(
        tagging,
        "summarize_tag_efficiency_events",
        lambda tags: {"synthetic_processed": len(tags)},
    )
    validity_scope = "Synthetic validity seam; not physical validation"
    monkeypatch.setattr(
        full_decay,
        "inference_diagnostics",
        lambda strict: {"scope": validity_scope},
    )
    monkeypatch.setattr(
        full_decay,
        "summarize_inference_diagnostics",
        lambda rows: {"scope": validity_scope, "synthetic_processed": len(rows)},
    )
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *args, **kwargs: "fixture-sha\n"
    )
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args: None)
    c = contract()
    c.update(
        source_root=str(runner.ROOT),
        source_sha="fixture-sha",
        source_hashes={},
        bindings=[],
        native_inputs={},
        output=str(tmp_path / "run"),
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(c))
    runner.run(path)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["processed"] == 2 and summary["gradient_events"] == 2
    assert summary["actual_gradient_objective_events"] == (
        2 - int(last_has_no_eligible_height)
    )
    assert summary["selected_gradient_no_eligible_height"] == int(
        last_has_no_eligible_height
    )
    assert summary["native_validity_closure"] == {
        "scope": validity_scope,
        "synthetic_processed": 2,
    }
    assert summary["events"][-1]["gradient_parameter_attribution_available"] is (
        not last_has_no_eligible_height
    )
    assert summary["parameter_updates"] == summary["validation_events"] == 0
    assert summary["per_category"] == {"charged": 1, "ccbar": 1}
    assert summary["model_before_sha256"] == summary["model_after_sha256"]
    assert summary["compute"]["model_forward_calls"] > 0
    assert summary["compute"]["encoder_event_views"] > 0
    assert summary["compute"]["encoder_node_squared_proxy"] > 0
    assert (
        not context.model._forward_pre_hooks
        and not context.model.encoder._forward_pre_hooks
    )
    terminal = json.loads((tmp_path / "run/terminal.json").read_text())
    assert terminal["status"] == "COMPLETED"
    assert all(runner.binding(item["path"]) == item for item in terminal["bindings"])


def test_full_native_requires_bound_smoke_and_finite_forecast(tmp_path):
    c = contract()
    c["smoke"] = False
    with pytest.raises(ValueError, match="real-data admission"):
        runner.validate_contract(c)
    c.update(source_sha="fixture", native_inputs={"immutable": "fixture"})
    summary = tmp_path / "smoke-summary.json"
    summary.write_text(
        json.dumps(
            dict(
                status="COMPLETED",
                smoke=True,
                source_sha="fixture",
                processed=2,
                parameter_updates=0,
            )
        )
    )
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps(dict(status="COMPLETED", bindings=[runner.binding(summary)]))
    )
    a = dict(
        status="PASS",
        source_sha="fixture",
        native_inputs=c["native_inputs"],
        forecast_wall_seconds=30,
        smoke_summary=runner.binding(summary),
        smoke_terminal=runner.binding(terminal),
    )
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps(a))
    c["admission"] = runner.binding(admission)
    runner.validate_contract(c)
    runner.verify_admission(c)
    for key, bad in [
        ("source_sha", "other"),
        ("native_inputs", {}),
        ("forecast_wall_seconds", 10800),
        ("forecast_wall_seconds", float("nan")),
    ]:
        changed = {**a, key: bad}
        admission.write_text(json.dumps(changed))
        c["admission"] = runner.binding(admission)
        with pytest.raises(ValueError):
            runner.verify_admission(c)
    admission.write_text(json.dumps(a))
    c["admission"] = runner.binding(admission)
    summary.write_text("{}")
    with pytest.raises(ValueError, match="smoke receipt changed"):
        runner.verify_admission(c)
