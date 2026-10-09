"""Read-only detector-generated proposal traces, joined to truth after generation."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, read, write  # noqa: E402


def detached_trace(result):
    """Accept model output only; no truth or event row can affect this trace."""
    return {
        "stages": {
            name: {
                "probabilities": result[logits][0]
                .detach()
                .float()
                .softmax(-1)
                .cpu()
                .tolist(),
                "presence": result[objects][0]
                .detach()
                .float()
                .sigmoid()
                .cpu()
                .tolist(),
            }
            for name, logits, objects in [
                ("proposal", "proposal_logits", "proposal_objects"),
                ("refinement", "logits", "objects"),
            ]
        },
        "states": [
            {
                "groups": [sorted(g) for g in s["groups"]],
                "pairs": [list(p) for p in s["pairs"]],
                "logits": s["logits"].detach().float().cpu().tolist(),
            }
            for s in result["states"]
        ],
    }


def stage_errors(stage, targets):
    """Truth join for unordered diagnostics only, never returned to generation."""
    p = stage["probabilities"]
    assigned = [max(range(3), key=lambda k: p[i][k]) for i in range(len(p))]
    groups = [{i for i, k in enumerate(assigned) if k == j} for j in (1, 2)]
    valid = [i for i, k in enumerate(targets) if k >= 0]
    costs = [
        sum(
            assigned[i] != (3 - targets[i] if swap and targets[i] > 0 else targets[i])
            for i in valid
        )
        for swap in (False, True)
    ]
    swap = costs[1] < costs[0]
    mapped = [3 - k if swap and k > 0 else k for k in targets]
    c = Counter(
        events=1,
        nodes=len(p),
        valid_nodes=len(valid),
        source_classification_errors=min(costs),
        predicted_unassigned=sum(k == 0 for k in assigned),
        raw_nonempty_groups=sum(bool(g) for g in groups),
        present_groups=sum(x >= 0.5 for x in stage["presence"]),
    )
    trials = []
    for k in (1, 2):
        target = {i for i, t in enumerate(mapped) if t == k}
        if not target:
            continue
        pred = groups[k - 1]
        missing = target - pred
        extra = pred - target
        exact = int(target in groups)
        best = max(
            (len(target & g) / len(target | g) if target | g else 0 for g in groups),
            default=0,
        )
        t = {
            "target_id": 3 - k if swap else k,
            "size": len(target),
            "raw_exact": exact,
            "missing": len(missing),
            "extra": len(extra),
            "intersection": len(target & pred),
            "union": len(target | pred),
            "best_iou": best,
            "missing_to_unassigned": sum(assigned[i] == 0 for i in missing),
            "missing_to_other_B": sum(assigned[i] > 0 for i in missing),
            "extra_unassigned_truth": sum(mapped[i] == 0 for i in extra),
            "extra_other_B_truth": sum(mapped[i] > 0 for i in extra),
            "exact_hidden_by_presence": int(
                exact
                and all(
                    stage["presence"][j] < 0.5
                    for j, g in enumerate(groups)
                    if g == target
                )
            ),
        }
        trials.append(t)
        c.update(
            {key: value for key, value in t.items() if key not in ("size", "target_id")}
        )
        c["trials"] += 1
    return dict(c), sorted(trials, key=lambda t: t["target_id"]), groups


def joined_trace(trace, row):
    targets = row["targets"].tolist()
    sides = [set(g) for g in row["supervision"]["b_groups"]]
    stages = {k: stage_errors(v, targets) for k, v in trace["stages"].items()}
    merge = Counter()
    from hypertagging.models.assembly_development import relation_targets

    for state_index, state in enumerate(trace["states"]):
        labels = relation_targets(
            tuple(frozenset(g) for g in state["groups"]),
            state["pairs"],
            row["supervision"],
        )
        prefix = "detector" if state_index == 0 else "generated"
        merge[prefix + "_relation_total_pairs"] += len(labels)
        merge[prefix + "_relation_ignored_pairs"] += sum(label < 0 for label in labels)
    proposal = trace["stages"]["proposal"]["probabilities"]
    assignment = [max(range(3), key=lambda k: p[k]) for p in proposal]
    for prev, nxt in zip(trace["states"], trace["states"][1:]):
        previous = {frozenset(g) for g in prev["groups"]}
        for raw in nxt["groups"]:
            g = set(raw)
            if frozenset(g) in previous:
                continue
            merge["merged_groups"] += 1
            clean = any(g <= b for b in sides)
            merge["clean_within_B_merges"] += int(clean)
            merge["cross_B_merges"] += int(sum(bool(g & b) for b in sides) > 1)
            merge["contains_unassigned_merges"] += int(any(targets[i] == 0 for i in g))
            eligible = (
                len({assignment[i] for i in g}) == 1 and assignment[next(iter(g))] > 0
            )
            merge["same_predicted_B_merges"] += int(eligible)
            merge["same_predicted_B_clean_merges"] += int(eligible and clean)
            merge["same_predicted_B_cross_B_merges"] += int(
                eligible and sum(bool(g & b) for b in sides) > 1
            )
    return stages, dict(merge)


def main(parent, out, replay=None):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Guarded two-CPU diagnostic required")
    import torch
    from scripts.run_phase74_development import verify_bindings
    from hypertagging.training.capacity_development import (
        fresh_model,
        fresh_decoder,
        detector_features,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    c = json.loads((parent / "successor-campaign-v1/contract.json").read_text())
    verify_bindings(c, Path(c["source_root"]))
    cache = torch.load(c["cache"]["path"], weights_only=False, map_location="cpu")
    result = {
        "version": "phase76-prestudy-diagnostic-v2",
        "scope": "fixed_preregistered_phase76_endpoint"
        if c["version"].startswith("phase76")
        else "train_and_previously_inspected_phase75_development_exploratory_only",
        "input_contract": binding(parent / "successor-campaign-v1/contract.json"),
        "arms": {},
    }
    start = time.monotonic()
    for arm in c["arms"]:
        run = Path(c["output_root"]) / arm
        term = json.loads((run / "terminal.json").read_text())
        summary = read(term["summary"])
        records = {}
        for fit in ("downstream", "tiny"):
            cb = next(
                b
                for b in term["checkpoints"]
                if Path(b["path"]).name == fit + "-final.pt"
            )
            assert binding(cb["path"]) == cb
            ck = torch.load(cb["path"], weights_only=False, map_location="cpu")
            model = fresh_model(128, cache["runtime_normalizer"])
            decoder = fresh_decoder()
            decoder.partial_state_conditioning = ck["settings"].get(
                "partial_state_conditioning", True
            )
            model.load_state_dict(ck["model_state_dict"])
            decoder.load_state_dict(ck["decoder_state_dict"])
            model.eval()
            decoder.eval()
            tiny = sum(
                (
                    [r for r in cache["train"] if r["category"] == cat][:8]
                    for cat in ("charged", "mixed", "ccbar")
                ),
                [],
            )
            roles = (
                {"tiny": tiny}
                if fit == "tiny"
                else {
                    "train": cache["train"],
                    (
                        "heldout"
                        if c["version"].startswith("phase76")
                        else "inspected_development"
                    ): cache["development"],
                }
            )
            for role, rows in roles.items():
                total = defaultdict(Counter)
                cats = defaultdict(lambda: defaultdict(Counter))
                sizes = defaultdict(lambda: defaultdict(Counter))
                transitions = Counter()
                merges = Counter()
                category_merges = defaultdict(Counter)
                event_records = []
                trace_path = (replay or out) / f"{arm}-{role}-detached-traces.jsonl.gz"
                with (
                    gzip.open(trace_path, "rt" if replay else "xt") as log,
                    torch.inference_mode(),
                ):
                    for row in rows:
                        if replay:
                            item = json.loads(next(log))
                            assert item["uid"] == row["uid"]
                            trace = item["trace"]
                        else:
                            h, _, _ = detector_features(model, row["detector"])
                            prediction = decoder(h, row["sources"])
                            trace = detached_trace(prediction)
                            log.write(
                                json.dumps({"uid": row["uid"], "trace": trace}) + "\n"
                            )
                        stages, merge = joined_trace(trace, row)
                        merges.update(merge)
                        category_merges[row["category"]].update(merge)
                        for stage, (counts, trials, groups) in stages.items():
                            total[stage].update(counts)
                            cats[row["category"]][stage].update(counts)
                            for t in trials:
                                size = t["size"]
                                bucket = (
                                    "1-4"
                                    if size <= 4
                                    else "5-8"
                                    if size <= 8
                                    else "9-16"
                                    if size <= 16
                                    else "17+"
                                )
                                sizes[bucket][stage].update(
                                    {
                                        k: v
                                        for k, v in t.items()
                                        if k not in ("size", "target_id")
                                    }
                                )
                                sizes[bucket][stage]["trials"] += 1
                        pc, pt, _ = stages["proposal"]
                        rc, rt, raw = stages["refinement"]
                        transitions["refinement_fewer_node_errors"] += int(
                            rc["source_classification_errors"]
                            < pc["source_classification_errors"]
                        )
                        transitions["refinement_more_node_errors"] += int(
                            rc["source_classification_errors"]
                            > pc["source_classification_errors"]
                        )
                        transitions["proposal_exact_refinement_not"] += sum(
                            int(a["raw_exact"] and not b["raw_exact"])
                            for a, b in zip(pt, rt)
                        )
                        transitions["refinement_exact_proposal_not"] += sum(
                            int(b["raw_exact"] and not a["raw_exact"])
                            for a, b in zip(pt, rt)
                        )
                        original_role = (
                            "heldout" if role == "inspected_development" else role
                        )
                        old = summary[original_role]["events"][len(event_records)]
                        assert old["uid"] == row["uid"] and old["raw_groups"] == [
                            sorted(g) for g in raw
                        ]
                        event_records.append(
                            {
                                "uid": row["uid"],
                                "category": row["category"],
                                "stages": {
                                    k: {"counts": v[0], "trials": v[1]}
                                    for k, v in stages.items()
                                },
                                "merge": merge,
                                "original_acceptance_counts": old["counts"],
                            }
                        )
                records[role] = {
                    "events": len(rows),
                    "stages": dict(total),
                    "by_category": dict(cats),
                    "by_size": dict(sizes),
                    "transitions": dict(transitions),
                    "merges": dict(merges),
                    "by_category_merges": dict(category_merges),
                    "checkpoint": cb,
                    "trace": binding(trace_path),
                }
                write(out / f"{arm}-{role}-joined-private.json", event_records)
                print(arm, role, records[role]["stages"], flush=True)
        result["arms"][arm] = records
    result["wall_seconds"] = time.monotonic() - start
    write(out / "diagnostic-summary.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replay", type=Path)
    a = p.parse_args()
    main(a.parent, a.output, a.replay)
