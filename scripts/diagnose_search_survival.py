#!/usr/bin/env python3
"""Freeze and run a bounded CPU search diagnosis on an existing development cohort.

Preparation never submits jobs. Run only in the supplied CPU Slurm wrapper.
Every result is exploratory reuse of the Phase71 beam cohort, not new primary
coverage. Existing checkpoints, cohorts and study contracts are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "frozen-search-survival-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked(binding):
    path = Path(binding["path"])
    if sha(path) != binding["sha256"]:
        raise ValueError(f"Input hash mismatch: {path}")
    return path


def prepare(parent_path, output):
    parent = json.loads(parent_path.read_text())
    arm = "aux_teacher_050"
    beam_task = next(t for t in parent["tasks"] if t["arm"] == arm and t["view"] == "beam_diagnostic")
    inputs = {k: parent[k] for k in ("selection", "index")}
    inputs.update({k: parent["arms"][arm][k] for k in ("pretraining-checkpoint", "reconstruction-checkpoint")})
    inputs["cohort"] = beam_task["cohort"]
    for binding in inputs.values():
        checked(binding)
    cohort = json.loads(checked(inputs["cohort"]).read_text())
    if cohort["role"] != "validation" or len(set(cohort["event_uids"])) != 60:
        raise ValueError("Expected existing 60-event validation beam diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "scripts").mkdir()
    shutil.copy2(__file__, snapshot / "scripts" / Path(__file__).name)
    source_hashes = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*.py"))}
    base = {"beam_width": 2, "max_candidates_per_query": 2, "max_proposals_per_level": 8,
            "max_candidate_expansions_per_query": 32}
    broad = {"max_candidates_per_query": 8, "max_proposals_per_level": 32,
             "max_daughter_options": 8, "max_cardinality_options": 4,
             "max_candidate_expansions_per_query": 256}
    conditions = [
        {"name": "reference", "search": base},
        {"name": "width_only", "search": {**base, "beam_width": 8}},
        {"name": "proposal_only", "search": {**base, **broad}},
        {"name": "width_and_proposal", "search": {**base, **broad, "beam_width": 8}},
        {"name": "threshold_support", "search": {**base, **broad, "beam_width": 8},
         "object_threshold": .2, "pointer_threshold": .1},
    ]
    write(output / "plan.json", {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_plan": {"path": str(parent_path.resolve()), "sha256": sha(parent_path)},
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_root": str(snapshot), "source_hashes": source_hashes, "inputs": inputs,
        "arm": arm, "conditions": conditions, "scope": "full", "max_level": 6,
        "cohort_role": "reused_phase71_diagnostic_not_independent_primary", "event_count": 60,
        "sealed_test_access": False, "training_updates": 0,
        "resource_ceiling": {"tasks": 5, "concurrent_tasks": 3, "cpu_per_task": 1,
                             "memory_gib_per_task": 8, "walltime_hours_per_task": 2, "gpus": 0},
        "limitations": ["one_frozen_checkpoint", "small_exploratory_cohort", "full_scope_only",
                        "threshold_support_changes_two_gates_jointly", "no_physical_FEI_claim"]})
    print(output / "plan.json")


def run(plan_path, task_id):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("Requires a fresh CPU Slurm allocation")
    plan = json.loads(plan_path.read_text())
    for relative, digest in plan["source_hashes"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Frozen source changed: {relative}")
    paths = {k: checked(b) for k, b in plan["inputs"].items()}
    condition = plan["conditions"][task_id]
    out = plan_path.parent / condition["name"]
    out.mkdir(exist_ok=False)
    write(out / "started.json", {"plan_sha256": sha(plan_path), "condition": condition,
                               "job_id": os.environ["SLURM_JOB_ID"], "status": "RUNNING"})
    started = time.monotonic()
    try:
        import torch
        from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair
        from hypertagging.evaluation.trained_context import load_trained_evaluation_context
        from hypertagging.evaluation.search_survival import analyze_search_trace, summarize_search_survival
        from hypertagging.evaluation.tag_efficiency import evaluate_tag_efficiency_event, summarize_tag_efficiency_events
        from hypertagging.reconstruction.beam_search import BeamSearchConfig
        from hypertagging.reconstruction.hierarchical_inference import (
            HierarchicalInferenceConfig, project_schema_v4_fsps, reconstruct_beam_from_fsps,
            reconstruct_full_tree_from_fsps,
        )
        from hypertagging.reconstruction.level_rollout import RolloutConfig
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        validate_checkpoint_pair(paths["pretraining-checkpoint"], paths["reconstruction-checkpoint"],
                                           require_exact_frozen_encoder=False)
        uids = json.loads(paths["cohort"].read_text())["event_uids"]
        context = load_trained_evaluation_context(
            checkpoint=paths["reconstruction-checkpoint"], data=paths["selection"],
            dataset_index=paths["index"], split="validation", max_events=len(uids), device="cpu",
            diagnostic_allow_external_independent_sample=True, event_selection="explicit_uids",
            explicit_event_uids=uids)
        policy, cfg = context.constraint_policy, context.checkpoint["config"]
        confidence_trained = bool(context.checkpoint.get("confidence_head_trained", False))
        learned = context.checkpoint.get("training_state", {}).get("checkpoint_selection_contract", {}).get(
            "rollout_configuration", {}).get("learned_confidence", confidence_trained)
        rollout = RolloutConfig(max_level=6, object_threshold=condition.get("object_threshold", .6),
            pointer_threshold=condition.get("pointer_threshold", .35), min_daughters=policy.minimum_daughters,
            use_learned_confidence=bool(learned), confidence_trained=confidence_trained,
            cardinality_insufficient_policy=policy.cardinality_insufficient_policy, constraint_policy=policy,
            rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode,
            rollout_pid_temperature=float(cfg.get("rollout_pid_temperature", .5)))
        inference = HierarchicalInferenceConfig(scope="full", rollout_config=rollout, max_level=6)
        search = BeamSearchConfig(**condition["search"])
        survival_rows, tag_rows, greedy_rows, diagnostics = [], [], [], Counter()
        categories = Counter()
        with (out / "events.jsonl").open("x") as records, gzip.open(out / "traces.jsonl.gz", "xt") as traces:
            for index, event in enumerate(context.events):
                truth = context.collated_event_batch(index)
                projection = project_schema_v4_fsps(truth)
                trace = []
                result = reconstruct_beam_from_fsps(context.model, truth, config=inference,
                                                     beam_config=search, trace=trace)
                # Truth is first joined AFTER all generation, ranking and pruning.
                survival = analyze_search_trace(truth, projection, trace, target_policy=cfg.get("target_policy", "complete_only"),
                    minimum_daughters=policy.minimum_daughters, object_threshold=rollout.object_threshold,
                    pointer_threshold=rollout.pointer_threshold)
                tags = evaluate_tag_efficiency_event(truth, [c.rollout.batch for c in result.candidates],
                    source_category=event.source_category, event=event, oracle_ks=(1, search.beam_width))
                survival_rows.append(survival)
                tag_rows.append(tags)
                if task_id == 0:
                    greedy = reconstruct_full_tree_from_fsps(context.model, truth, config=inference)
                    greedy_rows.append(evaluate_tag_efficiency_event(truth, [greedy.rollout.batch],
                        source_category=event.source_category, event=event))
                categories[event.source_category] += 1
                for key, value in result.diagnostics.items():
                    if type(value) is int:
                        if key.startswith("max_"):
                            diagnostics[key] = max(diagnostics[key], value)
                        else:
                            diagnostics[key] += value
                records.write(json.dumps({"event_uid": event.event_uid, "source_category": event.source_category,
                    "survival": survival, "tag_efficiency": tags}, allow_nan=False) + "\n")
                records.flush()
                traces.write(json.dumps({"event_uid": event.event_uid, "trace": trace}, allow_nan=False) + "\n")
                print(f"{condition['name']} {index + 1}/{len(uids)} {event.source_category}", flush=True)
        summary = {"version": VERSION, "condition": condition, "configuration": asdict(search),
            "processed_events": len(context.events), "categories": dict(categories), "failures": 0,
            "cohort_role": plan["cohort_role"], "survival": summarize_search_survival(survival_rows),
            "tag_efficiency": summarize_tag_efficiency_events(tag_rows), "search_counters": dict(diagnostics),
            "context_metadata": context.report_metadata, "elapsed_seconds": time.monotonic() - started,
            "plan_sha256": sha(plan_path), "truth_used_for_generation": False}
        if greedy_rows:
            summary["greedy_tag_efficiency"] = summarize_tag_efficiency_events(greedy_rows)
        write(out / "summary.json", summary)
        write(out / "receipt.json", {"status": "COMPLETED", "summary_sha256": sha(out / "summary.json"),
            "events_sha256": sha(out / "events.jsonl"), "trace_sha256": sha(out / "traces.jsonl.gz"),
            "plan_sha256": sha(plan_path), "job_id": os.environ["SLURM_JOB_ID"]})
    except BaseException as error:
        write(out / "failure.json", {"status": "FAILED", "type": type(error).__name__, "message": str(error),
              "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parent_plan.resolve(), args.output.resolve())
    else:
        run(args.plan.resolve(), args.task_id)
