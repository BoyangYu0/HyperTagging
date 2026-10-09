"""Finite native-hierarchy TRAIN diagnostic; no optimizer or automatic submission."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def binding(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_contract(c):
    if c["kind"] != "phase84_native_hierarchy_applicability" or c["panel_events"] != 96:
        raise ValueError("Native diagnostic shape changed")
    if c["gradient_events"] != 24 or c["parameter_updates"] != 0:
        raise ValueError("Native diagnostic optimization scope changed")
    if c["resources"] != dict(cpus=2, memory_gib=32, hours=4, gpus=0, requeue=False):
        raise ValueError("Native diagnostic resource envelope changed")
    if c["validation_events"] or c["sealed_test_access"] or c["automatic_successor"]:
        raise ValueError("Native diagnostic role/authority changed")
    if type(c["smoke"]) is not bool:
        raise ValueError("Explicit smoke flag required")
    if not c["smoke"] and not isinstance(c.get("admission"), dict):
        raise ValueError("Full native diagnostic requires real-data admission")
    if c["runtime_guard_seconds"] != 10800:
        raise ValueError("Native diagnostic finite guard changed")


def verify_admission(c):
    if c["smoke"]:
        return
    item = c["admission"]
    if binding(item["path"])["sha256"] != item["sha256"]:
        raise ValueError("Native admission changed")
    a = json.loads(Path(item["path"]).read_text())
    if (
        a.get("status") != "PASS"
        or a.get("source_sha") != c["source_sha"]
        or a.get("native_inputs") != c["native_inputs"]
    ):
        raise ValueError("Native runtime admission does not bind this source/data")
    forecast = a.get("forecast_wall_seconds", float("inf"))
    if not math.isfinite(forecast) or not 0 < forecast < c["runtime_guard_seconds"]:
        raise ValueError("Native full diagnostic exceeds finite forecast")
    for key in ("smoke_summary", "smoke_terminal"):
        item = a[key]
        if binding(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Native smoke receipt changed")
    summary = json.loads(Path(a["smoke_summary"]["path"]).read_text())
    if (
        summary["status"] != "COMPLETED"
        or summary["smoke"] is not True
        or summary["source_sha"] != c["source_sha"]
        or summary["processed"] != 2
        or summary["parameter_updates"] != 0
    ):
        raise ValueError("Native smoke did not complete required admission")
    terminal = json.loads(Path(a["smoke_terminal"]["path"]).read_text())
    if terminal["status"] != "COMPLETED":
        raise ValueError("Native smoke terminal failed")
    for item in terminal["bindings"]:
        if binding(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Native smoke artifact changed")


def native_config(document):
    from hypertagging.training.reconstruction_trainer import ReconstructionConfig

    allowed = {f.name for f in fields(ReconstructionConfig)}
    metadata = {
        "balanced_level_replay_contract",
        "four_train_size_equivalent_slot_target",
        "optimizer_steps_per_train_size_equivalent",
        "replay_slot_budget",
        "replay_slot_counts_by_level",
        "scheduled_sampling_zero_based_step_range",
        "train_split_event_count",
        "training_budget_semantics",
    }
    if set(document) - allowed - metadata:
        raise ValueError(
            "Unknown native config fields: " + str(set(document) - allowed - metadata)
        )
    cfg = ReconstructionConfig(**{k: v for k, v in document.items() if k in allowed})
    if (
        cfg.level_sampling_mode != "balanced_level_replay"
        or cfg.target_policy != "complete_only"
    ):
        raise ValueError("Unexpected native checkpoint objective contract")
    if cfg.unrepresentable_target_policy != "masked_representable_only":
        raise ValueError("Native context missing-target policy changed")
    if list(map(tuple, cfg.level_loss_weights)) != [
        (1, 1),
        (2, 1),
        (3, 1.25),
        (4, 1.5),
        (5, 2),
        (6, 3),
    ]:
        raise ValueError("Native global level weights changed")
    if cfg.pointer_set_overlap_weight != 0 or cfg.recovery_objective_weight != 0:
        raise ValueError("Expected Phase72 native control")
    # CPU FP32 sensitivities, explicitly not a historical mixed-precision replay.
    return replace(cfg, device="cpu", mixed_precision=False)


def trainable_groups(model, cfg, checkpoint_step):
    last_optimization_step = checkpoint_step - 1
    for p in model.parameters():
        p.requires_grad_(True)
    if last_optimization_step < cfg.freeze_pretrained_encoder_steps:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    if last_optimization_step < cfg.freeze_leaf_pid_head_steps:
        for p in model.leaf_pid_head.parameters():
            p.requires_grad_(False)
    groups = {"encoder": [], "other_native_trainable": []}
    encoder = {id(p) for p in model.encoder.parameters()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[
                "encoder" if id(parameter) in encoder else "other_native_trainable"
            ].append((name, parameter))
    return {name: params for name, params in groups.items() if params}


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_panel(events):
    chosen = set()
    for category in ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"):
        rows = [e for e in events if e.source_category == category]
        if len(rows) != 16:
            raise ValueError("Native category balance changed")
        rows.sort(
            key=lambda e: hashlib.sha256(
                ("phase84-native-gradient:" + e.event_uid).encode()
            ).hexdigest()
        )
        chosen.update(e.event_uid for e in rows[:4])
    return chosen


def native_rollout_config(context, cfg):
    from hypertagging.reconstruction.level_rollout import RolloutConfig

    trained = bool(context.checkpoint.get("confidence_head_trained", False))
    if cfg.rollout_use_learned_confidence and not trained:
        raise ValueError("Native requested confidence is not trained")
    return RolloutConfig(
        max_level=6,
        root_types=tuple(cfg.rollout_root_types or ()),
        exclusive_final=bool(cfg.rollout_exclusive_final),
        use_learned_confidence=bool(cfg.rollout_use_learned_confidence),
        confidence_trained=trained,
        constraint_policy=context.constraint_policy,
        seed=cfg.seed + context.checkpoint["step"] - 1,
        rollout_pid_kinematics_mode=cfg.rollout_pid_kinematics_mode,
        rollout_pid_temperature=cfg.rollout_pid_temperature,
        object_threshold=cfg.rollout_object_threshold,
        pointer_threshold=cfg.rollout_pointer_threshold,
        continue_through_empty_levels=cfg.rollout_continue_through_empty_levels,
    )


def target_inventory(batch):
    """Truth-only support inventory; never model inputs or generation controls."""
    from hypertagging.losses.level_reconstruction import targets_for_level
    from hypertagging.preprocessing.pid_filter import PDG_TOKENS

    result = []
    for policy in ("complete_only", "reconstructable_partial"):
        for height in range(1, 7):
            types, masks, _, _ = targets_for_level(batch, height, target_policy=policy)
            context_positions = (
                (batch["node_mask"][0] & (batch["level_ids"][0] < height))
                .nonzero()
                .flatten()
            )
            for pid, daughters in zip(types[0].tolist(), masks[0]):
                positions = context_positions[daughters]
                sources = batch["recursive_leaf_source_mask"][0, positions].any(0)
                result.append(
                    dict(
                        policy=policy,
                        height=height,
                        pid_token=pid,
                        signed_pdg=int(PDG_TOKENS[pid]),
                        daughters=int(daughters.sum()),
                        source_size=int(sources.sum()),
                    )
                )
    return result


def repeated_level_batch(batch, heights):
    import torch

    result = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1:
            result[name] = value.repeat((len(heights),) + (1,) * (value.ndim - 1))
        else:
            result[name] = value
    result["selected_target_levels"] = torch.tensor(heights, dtype=torch.long)
    return result


@contextmanager
def record_training_rollouts(trainer, output, label):
    from scripts.phase84_hierarchy_trace import serialize_native_rollout

    original = trainer.level_rollout
    calls = []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        path = output / f"{label}-training-rollout-{len(calls)}.json"
        # Called before the native trainer performs aligned supervision joins.
        write(path, serialize_native_rollout(result))
        calls.append(binding(path))
        return result

    trainer.level_rollout = wrapped
    try:
        yield calls
    finally:
        trainer.level_rollout = original


def scalar_report(result, calls, normalizations):
    from scripts.phase84_hierarchy_loss_observer import scalar_derivative

    total = result[0] + result[1]
    report = {
        "native_total": float(total.detach()),
        "reconstruction_loss": float(result[0].detach()),
        "leaf_pid_loss": float(result[1].detach()),
        "parameter_gradients": "UNAVAILABLE_BY_DESIGN_24_EVENT_SUBSET",
        "calls": [],
    }
    for index, call in enumerate(calls):
        loss = call["output"]
        coefficient = scalar_derivative(total, loss.total)
        report["calls"].append(
            {
                "call_index": index,
                "height": call["target_level"],
                "target_override": call["target_override"],
                "effective_native_coefficient": coefficient,
                "unweighted_total": float(loss.total.detach()),
                "components": [
                    {"names": [name], "unweighted_value": float(value.detach())}
                    for name, value in loss.components.items()
                ],
            }
        )
    report["normalizations"] = [
        {"weights": list(n["weights"]), "denominator": sum(n["weights"])}
        for n in normalizations
    ]
    return report


def objective_audit(
    context, batch, cfg, groups, output, label, gradients_enabled, replay=False
):
    import torch
    from hypertagging.training import reconstruction_trainer as trainer
    from hypertagging.training.scheduled_sampling import TeacherForcingSchedule
    from scripts.phase84_hierarchy_loss_observer import (
        observe_optimization_loss,
        capture_native_losses,
        matched_query_logit_diagnostics,
    )

    kwargs = dict(
        valid_levels=sorted(set(batch["selected_target_levels"].tolist())),
        config=cfg,
        schedule=TeacherForcingSchedule(
            kind="constant",
            start_probability=float(label == "teacher"),
            end_probability=float(label == "teacher"),
            duration_steps=0,
        ),
        step=context.checkpoint["step"] - 1,
        use_scheduled_sampling=True,
        allowed_types_by_level=context.allowed_types_by_level,
        constraint_policy=context.constraint_policy,
    )
    objective_started = time.monotonic()
    with record_training_rollouts(trainer, output, label) as traces:
        if gradients_enabled:
            result, report, capture = observe_optimization_loss(
                context.model,
                batch,
                native_kwargs=kwargs,
                parameter_groups=groups,
                capture_tensors=True,
            )
        else:
            with capture_native_losses(trainer, capture_tensors=True) as (
                calls,
                normalizations,
            ):
                result = trainer._optimization_loss(context.model, batch, **kwargs)
            capture = {"calls": calls, "normalizations": normalizations}
            report = scalar_report(result, calls, normalizations)
            report["timing"] = {
                "forward_wall_seconds": time.monotonic() - objective_started,
                "parameter_attribution_wall_seconds": 0.0,
            }
    if not torch.isfinite(result[0] + result[1]):
        raise FloatingPointError("Nonfinite native audit objective")
    report["native_context_metrics"] = result[4]
    report["generated_traces_before_join"] = traces
    query_started = time.monotonic()
    report["matched_query_logit_diagnostics"] = [
        matched_query_logit_diagnostics(
            call, rr["effective_native_coefficient"], context.constraint_policy
        )
        for call, rr in zip(capture["calls"], report["calls"])
    ]
    report["timing"]["query_attribution_wall_seconds"] = (
        time.monotonic() - query_started
    )
    if replay:
        replay_started = time.monotonic()
        params = [p for g in groups.values() for _, p in g]
        baseline = trainer._optimization_loss(context.model, batch, **kwargs)
        torch.testing.assert_close(
            result[0] + result[1], baseline[0] + baseline[1], rtol=0, atol=0
        )
        ga = torch.autograd.grad(result[0] + result[1], params, allow_unused=True)
        gb = torch.autograd.grad(baseline[0] + baseline[1], params, allow_unused=True)
        for a, b in zip(ga, gb):
            if a is None or b is None:
                if not (a is None and b is None):
                    raise ValueError("Native observer changed gradient support")
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        report["real_native_observer_replay"] = "EXACT_SCALAR_AND_PARAMETER_GRADIENTS"
        report["timing"]["replay_wall_seconds"] = time.monotonic() - replay_started
        del baseline, ga, gb
    write(output / f"{label}-objective.json", report)
    del result, capture
    gc.collect()
    return report


def historical_exposure(payload, cfg):
    """Authenticated sampler slots, not inferred representable-target gradients."""
    replay = payload.get("data_order_contract", {}).get("balanced_level_replay")
    cursor = payload.get("streaming_cursor", {})
    if not replay or "events_consumed" not in cursor:
        return {"status": "UNAVAILABLE_CHECKPOINT_SAMPLER_OR_CURSOR"}
    slots = int(payload["step"]) * cfg.batch_size
    if int(cursor["events_consumed"]) != slots or int(
        cursor.get("batch_index", -1)
    ) != int(payload["step"]):
        raise ValueError("Native historical cursor/update mismatch")
    if replay.get("level_schedule") != "global_slot_round_robin" or replay.get(
        "levels"
    ) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Unexpected native historical height sampling")
    quotient, remainder = divmod(slots, 6)
    counts = {str(h): quotient + int(h <= remainder) for h in range(1, 7)}
    pools = replay["eligible_pool_counts_by_level"]
    return {
        "status": "VERIFIED_CHECKPOINT_CURSOR_AND_ROUND_ROBIN_SLOTS",
        "checkpoint_updates": payload["step"],
        "presentations": slots,
        "slot_counts_by_height": counts,
        "eligible_pool_counts_by_height": pools,
        "mean_presentations_per_eligible_event_by_height": {
            h: counts[h] / pools[h] for h in counts
        },
        "unavailable": "Historical teacher/generated decisions, representable-target and per-PID gradient histories are not reconstructed from slot counts.",
    }


def run(path):
    c = json.loads(path.read_text())
    validate_contract(c)
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Bounded two-CPU Slurm allocation required")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        raise RuntimeError("GPU/requeue not admitted")
    if (
        str(ROOT) != c["source_root"]
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Native diagnostic source revision changed")
    for relative, digest in c["source_hashes"].items():
        if binding(ROOT / relative)["sha256"] != digest:
            raise ValueError("Frozen native source changed: " + relative)
    for item in c["bindings"]:
        if binding(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Native diagnostic binding changed")
    verify_admission(c)
    out = Path(c["output"])
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    import torch
    from scripts.phase84_hierarchy_train_context import load_hierarchy_train_context
    from scripts.phase84_hierarchy_trace import (
        capture_native_trace,
        evaluate_native_trace,
    )
    from hypertagging.reconstruction.hierarchical_inference import (
        HierarchicalInferenceConfig,
        reconstruct_full_tree_from_fsps,
    )
    from hypertagging.evaluation.tag_efficiency import (
        evaluate_tag_efficiency_event,
        summarize_tag_efficiency_events,
    )

    from hypertagging.evaluation.full_decay_runner import (
        inference_diagnostics,
        summarize_inference_diagnostics,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    context = load_hierarchy_train_context(**c["native_inputs"], repo_root=ROOT)
    if len(context.events) != 96:
        raise ValueError("Native panel coverage changed")
    write(out / "authentication.json", context.metadata)
    cfg = native_config(context.config)
    if context.checkpoint["step"] != 4376:
        raise ValueError("Native fixed checkpoint changed")
    groups = trainable_groups(context.model, cfg, context.checkpoint["step"])
    before = model_digest(context.model)
    work = Counter()

    def count_model(_module, args):
        work["model_forward_calls"] += 1
        work["model_event_views"] += args[0]["node_mask"].shape[0]

    def count_encoder(_module, args):
        counts = args[0]["node_mask"].sum(-1)
        work["encoder_forward_calls"] += 1
        work["encoder_event_views"] += len(counts)
        work["encoder_node_squared_proxy"] += int(counts.square().sum())

    model_hook = context.model.register_forward_pre_hook(count_model)
    encoder_hook = context.model.encoder.register_forward_pre_hook(count_encoder)
    chosen = gradient_panel(context.events)
    events = list(enumerate(context.events))
    if c["smoke"]:
        events = [
            next(x for x in events if x[1].source_category == category)
            for category in ("charged", "ccbar")
        ]
    rollout_config = native_rollout_config(context, cfg)
    profile = []
    for event_index, event in enumerate(context.events):
        native_batch = context.collated_event_batch(event_index)
        heights = sorted(
            {
                r["height"]
                for r in target_inventory(native_batch)
                if r["policy"] == cfg.target_policy
            }
        )
        nodes = int(native_batch["node_mask"].sum())
        profile.append(
            dict(
                uid=event.event_uid,
                category=event.source_category,
                nodes=nodes,
                objective_heights=heights,
                node_squared_proxy=nodes**2,
                objective_view_node_squared_proxy=nodes**2 * len(heights),
                parameter_gradient_selected=event.event_uid in chosen,
            )
        )
    del native_batch
    write(out / "full-panel-profile.json", profile)
    load_wall = time.monotonic() - started
    rows, tags, validity_rows = [], [], []
    for index, event in events:
        if (
            time.monotonic() - started > c["runtime_guard_seconds"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            raise RuntimeError("Native diagnostic finite resource guard")
        event_start = time.monotonic()
        folder = out / ("event-" + str(index))
        folder.mkdir()
        truth = context.collated_event_batch(index)
        projection, rollout, trace = capture_native_trace(
            context.model, truth, rollout_config
        )
        write(folder / "model-only-trace.json", trace)
        # First use of target labels is after detached native generation is saved.
        joined = evaluate_native_trace(
            truth,
            projection,
            trace,
            target_policy=cfg.target_policy,
            minimum_daughters=context.constraint_policy.minimum_daughters,
        )
        write(folder / "trace-evaluation.json", joined)
        strict = reconstruct_full_tree_from_fsps(
            context.model,
            truth,
            config=HierarchicalInferenceConfig(
                scope="full", max_level=6, rollout_config=rollout_config
            ),
        )
        # Shared evaluator attaches source identities outside inference; one greedy candidate only.
        tag = evaluate_tag_efficiency_event(
            truth,
            [strict.rollout.batch],
            source_category=event.source_category,
            event=event,
        )
        tags.append(tag)
        validity = inference_diagnostics(strict)
        validity_rows.append(validity)
        write(folder / "greedy-validity-closure.json", validity)
        write(folder / "greedy-tag-efficiency.json", tag)
        generation_wall = time.monotonic() - event_start
        inventory = target_inventory(truth)
        heights = sorted(
            {r["height"] for r in inventory if r["policy"] == cfg.target_policy}
        )
        gradient_enabled = c["smoke"] or event.event_uid in chosen
        reports = {}
        if heights:
            batch = repeated_level_batch(truth, heights)
            for label in ("teacher", "generated"):
                reports[label] = objective_audit(
                    context,
                    batch,
                    cfg,
                    groups,
                    folder,
                    label,
                    gradient_enabled,
                    replay=c["smoke"],
                )
            del batch
        row = dict(
            uid=event.event_uid,
            category=event.source_category,
            objective_heights=heights,
            target_inventory=inventory,
            gradient_selected=gradient_enabled,
            gradient_parameter_attribution_available=bool(gradient_enabled and heights),
            unavailable_context_reason=None
            if heights
            else "NO_ELIGIBLE_NATIVE_COMPLETE_TARGET_HEIGHT",
            context_metrics={
                k: v["native_context_metrics"] for k, v in reports.items()
            },
            generation_wall_seconds=generation_wall,
            objective_timing={k: v["timing"] for k, v in reports.items()},
            wall_seconds=time.monotonic() - event_start,
            native_nodes=int(truth["node_mask"].sum()),
            strict_valid=bool(strict.rollout.event_valid_mask[0]),
            trace=binding(folder / "model-only-trace.json"),
            evaluation=binding(folder / "trace-evaluation.json"),
        )
        rows.append(row)
        write(folder / "event.json", row)
        print(
            json.dumps(
                {
                    "uid": event.event_uid,
                    "completed": len(rows),
                    "planned": len(events),
                    "wall_seconds": row["wall_seconds"],
                }
            ),
            flush=True,
        )
        del truth, projection, rollout, trace, strict, reports
        gc.collect()
    model_hook.remove()
    encoder_hook.remove()
    if (
        time.monotonic() - started > c["runtime_guard_seconds"]
        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
    ):
        raise RuntimeError("Native diagnostic final resource guard")
    after = model_digest(context.model)
    if before != after or any(p.grad is not None for p in context.model.parameters()):
        raise ValueError("Diagnostic changed native model or parameter gradients")
    summary = dict(
        status="COMPLETED",
        smoke=c["smoke"],
        source_sha=c["source_sha"],
        contract=binding(path),
        events=rows,
        processed=len(rows),
        per_category=dict(Counter(r["category"] for r in rows)),
        parameter_updates=0,
        validation_events=0,
        historical_exposure=historical_exposure(context.checkpoint, cfg),
        full_panel_profile=binding(out / "full-panel-profile.json"),
        gradient_events=sum(r["gradient_selected"] for r in rows),
        actual_gradient_objective_events=sum(
            r["gradient_parameter_attribution_available"] for r in rows
        ),
        selected_gradient_no_eligible_height=sum(
            r["gradient_selected"] and not r["objective_heights"] for r in rows
        ),
        tag_efficiency=summarize_tag_efficiency_events(tags),
        native_validity_closure=summarize_inference_diagnostics(validity_rows),
        model_before_sha256=before,
        model_after_sha256=after,
        compute=dict(
            **work,
            workload_note="Measured forward proxies are not FLOPs; gradient calls add unrepresented compute",
            load_wall_seconds=load_wall,
            wall_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            parameters=sum(p.numel() for p in context.model.parameters()),
            native_trainable_parameters=sum(
                p.numel() for g in groups.values() for _, p in g
            ),
        ),
        limitations=[
            "TRAIN-only pairedcontextaudit,notindependentphysicsconfirmation",
            "Forcedteacher/generatedcontexts and repeatedeligiblelevelviews do not replay historicalsampling/exposure",
            "Audit uses eval mode and CPU FP32; scalar/gradient observer replay is against the same native CPU objective, not historic stochastic mixed-precision gradients",
            "Native unbatched training-context trace and strict batched greedy metrics are separate paths,notassumedequivalent",
            "Gradientnorms are local sensitivities,notadditiveallocationfractions orproofunderweighting",
            "No broader beam pool is generated; primarycoverage not met; retainedtruth is notFEI-equivalentefficiency",
        ],
    )
    write(out / "summary.json", summary)
    write(
        out / "terminal.json",
        {
            "status": "COMPLETED",
            "bindings": [binding(p) for p in sorted(out.rglob("*")) if p.is_file()],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    run(parser.parse_args().contract)
