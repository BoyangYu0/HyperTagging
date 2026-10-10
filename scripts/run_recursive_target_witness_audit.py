"""Execute the immutable cr001 trace audit inside its admitted CPU allocation."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(contract_path):
    c = json.loads(contract_path.read_text())
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
    ):
        raise RuntimeError("Two CPU Slurm allocation required")
    if (
        any(
            os.environ.get(k, "")
            for k in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
        )
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        raise RuntimeError("GPU/requeue forbidden")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Frozen HEAD changed")
    for item in c["bindings"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Binding mismatch: " + item["path"])
    for rel, h in c["source_hashes"].items():
        if digest(ROOT / rel) != h:
            raise ValueError("Source mismatch: " + rel)
    plan = json.loads(Path(c["plan"]).read_text())
    if plan["version"] != "continuous-recursive-contract-audit-v2":
        raise ValueError("Wrong protocol")
    if c["smoke"] is not True:
        admission = json.loads(Path(c["runtime_admission"]).read_text())
        if admission["status"] != "PASS" or admission["source_sha"] != c["source_sha"]:
            raise ValueError("Full runtime not admitted")
    out = Path(c["output"])
    out.mkdir(exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()

    def guard():
        if (
            time.monotonic() - start > c["watchdog_seconds"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            raise RuntimeError("Resource guard exceeded")

    import torch
    from scripts.phase84_hierarchy_train_context import load_hierarchy_train_context
    from scripts.phase84_hierarchy_trace import evaluate_native_trace
    from scripts.run_phase84_hierarchy_applicability import target_inventory
    from scripts.recursive_target_witness_audit import audit_event
    from hypertagging.reconstruction.hierarchical_inference import (
        project_schema_v4_fsps,
    )

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Existing authenticated loader restores weights but never invokes a forward.
    context = load_hierarchy_train_context(
        **plan["source_contract"]["native_inputs"], repo_root=ROOT
    )

    def forbidden(*args, **kwargs):
        raise RuntimeError("Model forward prohibited")

    context.model.register_forward_pre_hook(forbidden)
    context.model.encoder.register_forward_pre_hook(forbidden)
    load_seconds = time.monotonic() - start
    write(out / "authentication.json", context.metadata)
    uids = json.loads(
        Path(
            plan["source_contract"]["native_inputs"]["uid_manifest"]["path"]
        ).read_text()
    )["event_uids"]
    if [e.event_uid for e in context.events] != uids or len(uids) != 96:
        raise ValueError("Panel identity/order mismatch")
    parts = json.loads(
        Path(
            plan["data_and_compute_control"]["partitions"]["binding"]["path"]
        ).read_text()
    )
    from scripts.run_phase86_structural_diagnostic import partition_map

    partitions = partition_map(parts)
    prior_contract = json.loads(
        Path(plan["source_contract"]["original_execution_contract"]["path"]).read_text()
    )
    prior = Path(prior_contract["output"])
    full_profile = [
        dict(uid=e.event_uid, category=e.source_category, nodes=len(e.node_ids))
        for e in context.events
    ]
    write(out / "full-panel-metadata-profile.json", full_profile)
    indices = list(range(96))
    if c["smoke"]:
        indices = [
            next(i for i, e in enumerate(context.events) if e.source_category == cat)
            for cat in ("charged", "ccbar")
        ]
    if indices != c["indices"]:
        raise ValueError("Frozen event indices differ")
    rows = []
    total_attempts = 0
    inherited_bindings = []
    partial_total = 0
    for index in indices:
        guard()
        event_start = time.monotonic()
        event = context.events[index]
        folder = prior / f"event-{index}"
        oldrow = json.loads((folder / "event.json").read_text())
        if (
            oldrow["uid"] != event.event_uid
            or oldrow["category"] != event.source_category
            or oldrow["partition"] != partitions[event.event_uid]
        ):
            raise ValueError("Event binding mismatch")
        for artifact in sorted(folder.glob("*.json")):
            inherited_bindings.append(
                dict(
                    path=str(artifact),
                    sha256=digest(artifact),
                    scope="unchanged_inherited_same_checkpoint_TRAIN",
                )
            )
        partial_total += oldrow["partial_eligible_mothers"]
        trace = json.loads((folder / "model-only-trace.json").read_text())
        historical = json.loads((folder / "trace-evaluation.json").read_text())
        truth = context.collated_event_batch(index)
        projection = project_schema_v4_fsps(truth)
        replay = evaluate_native_trace(
            truth,
            projection,
            trace,
            target_policy="complete_only",
            minimum_daughters=context.constraint_policy.minimum_daughters,
        )
        if replay != historical:
            raise ValueError("Historical target/source counters mismatch")
        inventory = target_inventory(truth)
        if inventory != oldrow["target_inventory"]:
            raise ValueError("Supervision inventory mismatch")
        result = audit_event(
            truth,
            projection,
            trace,
            historical,
            context.constraint_policy,
            context.checkpoint["architecture"],
        )
        result.update(
            uid=event.event_uid,
            category=event.source_category,
            partition=partitions[event.event_uid],
            historical_target_inventory=inventory,
            wall_seconds=time.monotonic() - event_start,
        )
        result["detector_measurement_availability"] = {
            k: v[0].tolist()
            for k, v in truth.items()
            if "availability" in k
            or k in ("truth_pid_available", "leaf_kinematics_mode_ids")
        }
        result["inherited_first_failure"] = json.loads(
            (folder / "first-failure.json").read_text()
        )
        for target in result["targets"]:
            target["emission_timing"] = {}
            for stage in ("raw_hard_decoded", "local_retained", "accepted"):
                for definition in ("source", "topology", "recursive_pid"):
                    rounds = [
                        r["round"]
                        for r in target["rounds"]
                        if any(e[definition] is True for e in r["emissions"][stage])
                    ]
                    target["emission_timing"][stage + "_" + definition] = dict(
                        first=min(rounds) if rounds else None,
                        last=max(rounds) if rounds else None,
                    )
            target["historical_repair_exact_legality"] = (
                "unavailable_counterfactual_state_not_executed"
            )
        total_attempts += result["attempted_tuples"]
        if total_attempts > 1021440:
            raise RuntimeError("Tuple budget exceeded")
        write(out / f"event-{index}.json", result)
        rows.append(result)
        print(json.dumps(dict(processed=len(rows), planned=len(indices))), flush=True)
    counts = Counter()
    categories = {}
    strata = {}
    for event in rows:
        cat = categories.setdefault(
            event["category"], Counter(processed=0, failed=0, unavailable=0)
        )
        cat["processed"] += 1
        for t in event["targets"]:
            eligible = bool(t["historical"]["policy_eligible"])
            counts["all_retained_mothers"] += 1
            counts["eligible" if eligible else "outside_policy"] += 1
            cat["all_retained_mothers"] += 1
            cat["eligible"] += eligible
            if eligible:
                counts[t["contract_status"]] += 1
                for k, v in t["available_any_round"].items():
                    counts["available_" + k] += v
            for key in ("partition", "category"):
                cell = strata.setdefault(key + "=" + event[key], Counter())
                cell["all"] += 1
                cell["eligible"] += eligible
            for key, val in [
                ("height", t["intrinsic_height"]),
                ("pid", t["historical"]["pid_token"]),
                ("arity", t["historical"]["daughter_count"]),
                ("source_size", t["historical"]["source_size"]),
                ("population", t["historical"]["population"]),
            ]:
                cell = strata.setdefault(str(key) + "=" + str(val), Counter())
                cell["all"] += 1
                cell["eligible"] += eligible
    if not c["smoke"] and (
        len(rows) != 96
        or counts["all_retained_mothers"] != 665
        or counts["eligible"] != 455
        or counts["outside_policy"] != 210
    ):
        raise ValueError("Frozen denominator mismatch")
    write(out / "inherited-metric-bindings.json", inherited_bindings)
    summary = dict(
        status="COMPLETED",
        source_sha=c["source_sha"],
        smoke=c["smoke"],
        processed=len(rows),
        counts=counts,
        categories=categories,
        strata=strata,
        attempted_tuples=total_attempts,
        wall_seconds=time.monotonic() - start,
        process_cpu_seconds=time.process_time() - cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        load_seconds=load_seconds,
        max_event_seconds=max(r["wall_seconds"] for r in rows),
        full_panel_max_nodes=max(r["nodes"] for r in full_profile),
        training_updates=0,
        training_presentations=0,
        model_forward_calls=0,
        encoder_forward_calls=0,
        backward_calls=0,
        new_collision_identities=0,
        validation_events=0,
        test_events=0,
        scientific_gate="not_applicable",
        complete_post_study_evaluation=False,
    )
    if not c["smoke"]:
        if partial_total != 455:
            raise ValueError("Partial target denominator mismatch")
        import importlib.util

        prior_root = prior.parent.parent
        script = prior_root / "preparation/reproduce-deep-root-height-review.py"
        spec = importlib.util.spec_from_file_location("inherited_height_review", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inherited = module.reproduce(json.loads((prior / "summary.json").read_text()))
        expected = json.loads(
            (prior_root / "reviews/deep-root-height-review-v2.json").read_text()
        )
        for key in inherited:
            if inherited[key] != expected[key]:
                raise ValueError("Inherited height counter mismatch:" + key)
        if (
            inherited["counts"]["generation_members"] != 196
            or inherited["counts"]["repair_members"] != 124
        ):
            raise ValueError("Historical opportunity reconciliation mismatch")
        by_target = {
            (e["uid"], t["node_position"]): t for e in rows for t in e["targets"]
        }
        cross = []
        for old in inherited["targets"]:
            new = by_target[(old["uid"], old["node_position"])]
            cross.append(
                dict(
                    inherited=old,
                    exact_observed=new["available_any_round"],
                    contract_status=new["contract_status"],
                    repair_counterfactual="unavailable",
                )
            )
        write(out / "inherited-height-reconciliation.json", inherited)
        write(out / "deep-opportunity-cross-tabs.json", cross)
        import numpy as np

        rng = np.random.default_rng(202610100101)
        names = ("source", "height", "topology", "recursive_pid")
        vectors = []
        for event in rows:
            eligible = [
                t for t in event["targets"] if t["historical"]["policy_eligible"]
            ]
            vectors.append(
                [len(eligible)]
                + [sum(t["available_any_round"][k] for t in eligible) for k in names]
            )
        vectors = np.asarray(vectors, dtype=np.int64)
        draws = []
        for _ in range(2000):
            indices = np.concatenate(
                [
                    rng.choice(
                        [i for i, e in enumerate(rows) if e["category"] == cat],
                        16,
                        replace=True,
                    )
                    for cat in ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
                ]
            )
            total = vectors[indices].sum(0)
            draws.append((total[1:] / total[0]).tolist() if total[0] else [None] * 4)
        totals = vectors.sum(0)
        paired = {}
        for i in range(3):
            valid = [d[i + 1] - d[i] for d in draws if d[0] is not None]
            paired[names[i + 1] + " minus " + names[i]] = dict(
                observed=float((totals[i + 2] - totals[i + 1]) / totals[0]),
                interval95=np.percentile(valid, [2.5, 97.5]).tolist(),
                valid_draws=len(valid),
                unavailable_draws=2000 - len(valid),
            )
        write(
            out / "paired-bootstrap.json",
            dict(
                seed=202610100101,
                replicates=2000,
                paired=paired,
                draws=draws,
                interpretation="Adaptive TRAIN panel resampling sensitivity only. PID counts are verified lower bounds; unavailable PID is not measured failure. Training-seed uncertainty unavailable.",
            ),
        )
    write(out / "summary.json", summary)
    guard()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path, required=True)
    run(p.parse_args().contract)
