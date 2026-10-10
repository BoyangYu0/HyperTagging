"""Authenticated, finite TRAIN-only structural audit; no fitting or submission."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
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
from scripts.run_phase84_hierarchy_applicability import (  # noqa: E402
    binding,
    write,
    native_config,
    native_rollout_config,
    model_digest,
    target_inventory,
)


def checked(item):
    if binding(item["path"])["sha256"] != item["sha256"]:
        raise ValueError("Frozen binding changed: " + item["path"])
    return Path(item["path"])


def validate_contract(c):
    if c["kind"] != "phase86_structural_first_failure" or c["panel_events"] != 96:
        raise ValueError("Structural diagnostic shape changed")
    if c["parameter_updates"] != 0 or c["validation_events"] != 0:
        raise ValueError("TRAIN-only observation contract changed")
    if c["sealed_test_access"] is not False or c["automatic_successor"] is not False:
        raise ValueError("Diagnostic authority changed")
    if c["resources"] != dict(cpus=2, memory_gib=32, hours=4, gpus=0, requeue=False):
        raise ValueError("Resource envelope changed")
    if type(c["smoke"]) is not bool or c["runtime_guard_seconds"] != 10800:
        raise ValueError("Finite scope changed")
    for key in ("plan_binding", "partition_binding"):
        checked(c[key])
    if not c["smoke"]:
        a = json.loads(checked(c["admission"]).read_text())
        if (
            a.get("status") != "PASS"
            or a.get("source_sha") != c["source_sha"]
            or a.get("native_inputs") != c["native_inputs"]
        ):
            raise ValueError("Runtime admission source/data mismatch")
        forecast = a.get("forecast_wall_seconds", float("inf"))
        if not math.isfinite(forecast) or not 0 < forecast < c["runtime_guard_seconds"]:
            raise ValueError("Runtime forecast not admitted")
        s = json.loads(checked(a["smoke_summary"]).read_text())
        t = json.loads(checked(a["smoke_terminal"]).read_text())
        if (
            s["status"] != "COMPLETED"
            or s["smoke"] is not True
            or s["source_sha"] != c["source_sha"]
            or s["processed"] != 2
            or s["parameter_updates"] != 0
            or s["model_before_sha256"] != s["model_after_sha256"]
            or t["status"] != "COMPLETED"
        ):
            raise ValueError("Smoke admission incomplete")
        for item in t["bindings"]:
            checked(item)


def legacy_trace(trace):
    """Remove observational additions only; compare historical semantics exactly."""
    trace = json.loads(json.dumps(trace))
    if trace.get("config") is not None:
        trace["config"].pop("capture_decode_trace", None)
    for step in trace["steps"]:
        step.pop("decode_trace", None)
    return trace


def partition_map(document):
    if document.get("role") != "train":
        raise ValueError("TRAIN partition role required")
    partitions = {name: document[name] for name in ("fit", "assessment")}
    result = {}
    for name, values in partitions.items():
        if isinstance(values, dict):
            values = values["event_uids"]
        for uid in values:
            if uid in result:
                raise ValueError("Overlapping TRAIN partitions")
            result[uid] = name
    if set(result.values()) != {"fit", "assessment"}:
        raise ValueError("Frozen fit/assessment partitions required")
    return result


def run(path):
    c = json.loads(path.read_text())
    validate_contract(c)
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Two-CPU Slurm allocation required")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        or os.environ.get("SLURM_JOB_GPUS", "")
        or os.environ.get("SLURM_STEP_GPUS", "")
    ):
        raise RuntimeError("GPU/requeue not admitted")
    if (
        str(ROOT) != c["source_root"]
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Source revision changed")
    for relative, digest in c["source_hashes"].items():
        if binding(ROOT / relative)["sha256"] != digest:
            raise ValueError("Frozen source changed: " + relative)
    for item in c["bindings"]:
        checked(item)
    partitions = partition_map(json.loads(checked(c["partition_binding"]).read_text()))
    out = Path(c["output"])
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()

    def guard():
        if (
            time.monotonic() - started > c["runtime_guard_seconds"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            raise RuntimeError("Finite runtime/RSS guard exceeded")

    import torch
    from scripts.phase84_hierarchy_train_context import load_hierarchy_train_context
    from scripts.phase84_hierarchy_trace import (
        capture_native_trace,
        evaluate_native_trace,
    )
    from scripts.phase86_first_failure import evaluate_first_failure
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
        serialize_reconstructed_tree,
    )

    from hypertagging.evaluation.full_decay_metrics import (
        evaluate_full_decay,
        evaluate_half_decays,
        evaluate_retained_decays,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    context = load_hierarchy_train_context(**c["native_inputs"], repo_root=ROOT)
    if len(context.events) != 96 or set(partitions) != {
        e.event_uid for e in context.events
    }:
        raise ValueError("Frozen TRAIN panel/partition mismatch")
    if Counter(e.source_category for e in context.events) != Counter(
        dict.fromkeys(("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"), 16)
    ):
        raise ValueError("TRAIN category balance changed")
    write(out / "authentication.json", context.metadata)
    cfg = native_config(context.config)
    if context.checkpoint["step"] != 4376:
        raise ValueError("Native checkpoint changed")
    context.model.eval()
    before = model_digest(context.model)
    config = native_rollout_config(context, cfg)
    traced_config = replace(config, capture_decode_trace=True)
    work = Counter()

    def count_model(_module, args):
        work["model_forward_calls"] += 1
        work["model_event_views"] += args[0]["node_mask"].shape[0]

    def count_encoder(_module, args):
        counts = args[0]["node_mask"].sum(-1)
        work["encoder_forward_calls"] += 1
        work["encoder_node_squared_proxy"] += int(counts.square().sum())

    hooks = [
        context.model.register_forward_pre_hook(count_model),
        context.model.encoder.register_forward_pre_hook(count_encoder),
    ]
    profile = []
    for index, event in enumerate(context.events):
        guard()
        batch = context.collated_event_batch(index)
        nodes = int(batch["node_mask"].sum())
        profile.append(
            dict(
                uid=event.event_uid,
                category=event.source_category,
                partition=partitions[event.event_uid],
                nodes=nodes,
                node_squared_proxy=nodes**2,
            )
        )
    del batch
    write(out / "full-panel-profile.json", profile)
    load_wall = time.monotonic() - started
    events = list(enumerate(context.events))
    if c["smoke"]:
        events = [
            next(x for x in events if x[1].source_category == category)
            for category in ("charged", "ccbar")
        ]
    rows, tags, validity_rows = [], [], []
    eligible_total = 0
    for index, event in events:
        guard()
        event_start = time.monotonic()
        folder = out / f"event-{index}"
        folder.mkdir()
        truth = context.collated_event_batch(index)
        projection, rollout, trace = capture_native_trace(
            context.model, truth, traced_config
        )
        # Save detached model-only state BEFORE either posthoc truth join.
        write(folder / "model-only-trace.json", trace)
        parity = "NOT_REPEATED_OUTSIDE_SMOKE"
        if c["smoke"]:
            baseline_projection, baseline_rollout, baseline_trace = (
                capture_native_trace(context.model, truth, config)
            )
            if legacy_trace(trace) != legacy_trace(baseline_trace):
                raise ValueError("Observation changed native rollout predictions")
            parity = "EXACT_DISABLED_TRACE_PARITY"
            del baseline_projection, baseline_rollout, baseline_trace
        historical = "NOT_REQUESTED"
        if c.get("historical_output"):
            old = (
                Path(c["historical_output"])
                / f"event-{index}"
                / "model-only-trace.json"
            )
            if legacy_trace(trace) != legacy_trace(json.loads(old.read_text())):
                raise ValueError("Historical native trace parity failed")
            historical = "EXACT_LEGACY_TRACE_PARITY"
        joined = evaluate_native_trace(
            truth,
            projection,
            trace,
            target_policy=cfg.target_policy,
            minimum_daughters=context.constraint_policy.minimum_daughters,
        )
        write(folder / "trace-evaluation.json", joined)
        first_failure = evaluate_first_failure(
            truth,
            projection,
            trace,
            target_policy=cfg.target_policy,
            minimum_daughters=context.constraint_policy.minimum_daughters,
        )
        write(folder / "first-failure.json", first_failure)
        partial_failure = evaluate_first_failure(
            truth,
            projection,
            trace,
            target_policy="reconstructable_partial",
            minimum_daughters=context.constraint_policy.minimum_daughters,
        )
        write(folder / "partial-target-first-failure.json", partial_failure)
        eligible = sum(bool(t["policy_eligible"]) for t in joined["targets"])
        if first_failure["policy_eligible_targets"] != eligible:
            raise ValueError("First-failure join dropped eligible targets")
        eligible_total += eligible
        strict = reconstruct_full_tree_from_fsps(
            context.model,
            truth,
            config=HierarchicalInferenceConfig(
                scope="full", max_level=6, rollout_config=config
            ),
        )
        tag = evaluate_tag_efficiency_event(
            truth,
            [strict.rollout.batch],
            source_category=event.source_category,
            event=event,
        )
        # Same strict greedy prediction, full shared posthoc metrics; no new inference.
        metric_options = dict(
            target_policy=cfg.target_policy,
            minimum_daughters=context.constraint_policy.minimum_daughters,
        )
        write(
            folder / "greedy-full-decay.json",
            evaluate_full_decay(
                strict.rollout.batch, truth, **metric_options
            ).as_dict(),
        )
        write(
            folder / "greedy-half-components.json",
            evaluate_half_decays(
                strict.rollout.batch,
                truth,
                source_category=event.source_category,
                **metric_options,
            ).as_dict(),
        )
        write(
            folder / "greedy-retained-forest.json",
            evaluate_retained_decays(
                strict.rollout.batch,
                truth,
                source_category=event.source_category,
                **metric_options,
            ).as_dict(),
        )
        write(folder / "greedy-tree.json", serialize_reconstructed_tree(strict))
        validity = inference_diagnostics(strict)
        tags.append(tag)
        validity_rows.append(validity)
        write(folder / "greedy-tag-efficiency.json", tag)
        write(folder / "greedy-validity-closure.json", validity)
        row = dict(
            uid=event.event_uid,
            category=event.source_category,
            partition=partitions[event.event_uid],
            eligible_mothers=eligible,
            target_inventory=target_inventory(truth),
            native_nodes=int(truth["node_mask"].sum()),
            legacy_first_error_counts=joined["first_error_counts"],
            first_failure_counts=first_failure["first_failure_counts"],
            partial_eligible_mothers=partial_failure["policy_eligible_targets"],
            partial_first_failure_counts=partial_failure["first_failure_counts"],
            trace_disabled_parity=parity,
            historical_trace_parity=historical,
            generation_wall_seconds=time.monotonic() - event_start,
            wall_seconds=time.monotonic() - event_start,
            trace=binding(folder / "model-only-trace.json"),
            evaluation=binding(folder / "trace-evaluation.json"),
            first_failure=binding(folder / "first-failure.json"),
            partial_first_failure=binding(folder / "partial-target-first-failure.json"),
        )
        rows.append(row)
        write(folder / "event.json", row)
        print(
            json.dumps(
                dict(
                    completed=len(rows),
                    planned=len(events),
                    wall_seconds=row["wall_seconds"],
                )
            ),
            flush=True,
        )
        del (
            truth,
            projection,
            rollout,
            trace,
            strict,
            joined,
            first_failure,
            partial_failure,
        )
        gc.collect()
    for hook in hooks:
        hook.remove()
    guard()
    after = model_digest(context.model)
    if before != after or any(p.grad is not None for p in context.model.parameters()):
        raise ValueError("Diagnostic changed model or accumulated gradients")
    if not c["smoke"] and eligible_total != 455:
        raise ValueError("Historical eligible denominator changed")
    grouped = {}
    for dimension in ("category", "partition"):
        grouped[dimension] = {}
        for value in sorted({r[dimension] for r in rows}):
            selected = [r for r in rows if r[dimension] == value]
            causes = Counter()
            partial_causes = Counter()
            inventory_counts = Counter()
            for row in selected:
                causes.update(row["first_failure_counts"])
                partial_causes.update(row["partial_first_failure_counts"])
                for target in row["target_inventory"]:
                    key = "|".join(
                        f"{k}={target[k]}"
                        for k in (
                            "policy",
                            "height",
                            "signed_pdg",
                            "daughters",
                            "source_size",
                        )
                    )
                    inventory_counts[key] += 1
            grouped[dimension][value] = dict(
                processed=len(selected),
                eligible_mothers=sum(r["eligible_mothers"] for r in selected),
                first_failure_counts=dict(causes),
                partial_eligible_mothers=sum(
                    r["partial_eligible_mothers"] for r in selected
                ),
                partial_first_failure_counts=dict(partial_causes),
                target_inventory=dict(inventory_counts),
            )
    summary = dict(
        status="COMPLETED",
        smoke=c["smoke"],
        source_sha=c["source_sha"],
        contract=binding(path),
        events=rows,
        processed=len(rows),
        eligible_mothers=eligible_total,
        partial_eligible_mothers=sum(r["partial_eligible_mothers"] for r in rows),
        partial_target_policy="reconstructable_partial",
        per_category=dict(Counter(r["category"] for r in rows)),
        per_partition=dict(Counter(r["partition"] for r in rows)),
        grouped_diagnostics=grouped,
        parameter_updates=0,
        validation_events=0,
        objectives_or_backward_calls=0,
        full_panel_profile=binding(out / "full-panel-profile.json"),
        authentication=binding(out / "authentication.json"),
        tag_efficiency=summarize_tag_efficiency_events(tags),
        native_validity_closure=summarize_inference_diagnostics(validity_rows),
        model_before_sha256=before,
        model_after_sha256=after,
        compute=dict(
            **work,
            load_wall_seconds=load_wall,
            wall_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            parameters=sum(p.numel() for p in context.model.parameters()),
        ),
        limitations=[
            "TRAIN-only diagnostic; previously inspected panel, not independent confirmation.",
            "Generated native states only; no teacher-forced objective or optimization.",
            "Greedy unbatched native trace and strict batched evaluator are separate paths.",
            "No beam pool or complete evaluation; retained truth is not physical FEI efficiency.",
            "Source coverage is necessary but does not establish legal reachability.",
            "Any posthoc injected group or repair is an oracle opportunity, not deployable gain.",
        ],
    )
    write(out / "summary.json", summary)
    write(
        out / "terminal.json",
        dict(
            status="COMPLETED",
            bindings=[binding(p) for p in sorted(out.rglob("*")) if p.is_file()],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    run(parser.parse_args().contract)
