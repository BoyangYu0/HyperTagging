"""Training-only proposal discrimination and supervision audit, no inference changes."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from hypertagging.models.direct_membership import direct_membership_loss  # noqa: E402
from scripts.prepare_phase74_development_data import binding, read, write  # noqa: E402


def loss_parts(logits, objects, targets):
    """Native one-event total-loss permutation; unknown targets retain zero support."""
    valid = targets >= 0
    p = logits.softmax(-1)[:, 1:].T
    alternatives = []
    for swap in (False, True):
        labels = torch.where(targets > 0, 3 - targets, targets) if swap else targets
        masks = torch.stack((labels == 1, labels == 2)) & valid[None]
        present = masks.any(-1)
        ce_nodes = F.cross_entropy(logits, labels.clamp_min(0), reduction="none")
        ce = (ce_nodes * valid).sum() / valid.sum().clamp_min(1)
        intersection = (p * masks).sum(-1)
        union = ((p + masks - p * masks) * valid[None]).sum(-1)
        overlap = (
            (1 - (intersection + 1e-6) / (union + 1e-6)) * present
        ).sum() / present.sum().clamp_min(1)
        presence = F.binary_cross_entropy_with_logits(objects, present.float())
        available = valid.any().float()
        alternatives.append(
            {
                "ce": ce * available,
                "overlap": overlap * available,
                "presence": presence * available,
                "labels": labels,
                "ce_nodes": ce_nodes * valid,
            }
        )
    totals = torch.stack([a["ce"] + a["overlap"] + a["presence"] for a in alternatives])
    selected = int(totals.detach().argmin())
    return alternatives[selected], selected, totals


def quantiles(values):
    if not values:
        return {"count": 0, "mean": None, "quantiles": None}
    v = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(v.mean()),
        "quantiles": dict(
            zip(
                ["min", "p10", "p25", "p50", "p75", "p90", "max"],
                v.quantile(
                    torch.tensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=v.dtype)
                ).tolist(),
            )
        ),
    }


def auc(values, labels):
    """Exact pooled rank AUC with tied scores; descriptive nodes, not iid trials."""
    pairs = sorted(zip(values, labels))
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return {"positive": n1, "negative": n0, "auc": None}
    wins = 0.0
    neg = 0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        pos = sum(y for _, y in pairs[i:j])
        n = j - i - pos
        wins += pos * (neg + n / 2)
        neg += n
        i = j
    return {"positive": n1, "negative": n0, "auc": wins / (n0 * n1)}


def hard_mapping(probabilities, targets):
    pred = probabilities.argmax(-1)
    costs = [
        int(
            (
                (
                    pred
                    != (
                        torch.where(targets > 0, 3 - targets, targets)
                        if swap
                        else targets
                    )
                )
                & (targets >= 0)
            ).sum()
        )
        for swap in (False, True)
    ]
    swap = costs[1] < costs[0]
    if costs[0] == costs[1]:
        # Canonical target-set tie break is evaluation-only and invariant to
        # renaming the two truth slots. Historical hard-error totals are unchanged.
        a = tuple(torch.where(targets == 1)[0].tolist())
        b = tuple(torch.where(targets == 2)[0].tolist())
        swap = b < a
    return torch.where(targets > 0, 3 - targets, targets) if swap else targets, swap


def describe(logits, objects, targets):
    """Evaluation-only label joins; all alternative decisions are diagnostic only."""
    p = logits.softmax(-1)
    pred = p.argmax(-1)
    labels, swap = hard_mapping(p, targets)
    valid = labels >= 0
    is_b = labels > 0
    is_bg = labels == 0
    bslot = p[:, 1:].argmax(-1) + 1
    aggregate = torch.where(p[:, 1:].sum(-1) > 0.5, bslot, torch.zeros_like(bslot))
    # Explicitly truth-assisted ceilings; never returned to inference.
    oracle_foreground = torch.where(is_b, bslot, torch.zeros_like(bslot))
    oracle_slot = torch.where((pred > 0) & is_b, labels, pred)
    counts = Counter(
        events=1,
        nodes=len(p),
        valid_nodes=int(valid.sum()),
        unknown_nodes=int((~valid).sum()),
        B_nodes=int(is_b.sum()),
        background_nodes=int(is_bg.sum()),
        B_to_unassigned=int((is_b & (pred == 0)).sum()),
        B_to_other_B=int((is_b & (pred > 0) & (pred != labels)).sum()),
        B_correct=int((is_b & (pred == labels)).sum()),
        background_to_B=int((is_bg & (pred > 0)).sum()),
        background_correct=int((is_bg & (pred == 0)).sum()),
        unknown_to_B=int(((~valid) & (pred > 0)).sum()),
        conditional_slot_correct=int((is_b & (bslot == labels)).sum()),
        aggregate_B_to_unassigned=int((is_b & (aggregate == 0)).sum()),
        aggregate_B_to_other_B=int(
            (is_b & (aggregate > 0) & (aggregate != labels)).sum()
        ),
        aggregate_background_to_B=int((is_bg & (aggregate > 0)).sum()),
        B_unassigned_but_aggregate_gt_half=int(
            (is_b & (pred == 0) & (p[:, 1:].sum(-1) > 0.5)).sum()
        ),
    )
    counts["available_B_targets"] = sum(bool((targets == k).any()) for k in (1, 2))
    for name, assignment in [
        ("native", pred),
        ("aggregate_half", aggregate),
        ("oracle_foreground", oracle_foreground),
        ("oracle_slot", oracle_slot),
    ]:
        groups = [set(torch.where(assignment == k)[0].tolist()) for k in (1, 2)]
        success = sum(
            set(torch.where(labels == k)[0].tolist()) in groups
            for k in (1, 2)
            if bool((labels == k).any())
        )
        counts[name + "_raw_exact"] = success
        counts[name + "_event_any"] = int(success > 0)
        counts[name + "_event_both"] = int(success == 2)
        counts[name + "_nonempty_groups"] = sum(bool(g) for g in groups)
    counts["B_events"] = int(bool(is_b.any()))
    counts["continuum_events"] = int(not bool(is_b.any()))
    measures = defaultdict(list)
    binary = logits[:, 1:].logsumexp(-1) - logits[:, 0]
    conditional = logits[:, 1:].softmax(-1)
    for i in range(len(p)):
        kind = "unknown" if labels[i] < 0 else "B" if labels[i] > 0 else "background"
        outcome = (
            "unassigned"
            if pred[i] == 0
            else "correct_B"
            if pred[i] == labels[i]
            else "other_B"
            if labels[i] > 0
            else "B"
        )
        for key, mask in [(kind, True), (kind + "_" + outcome, True)]:
            if mask:
                measures[key + "/binary_log_odds"].append(float(binary[i]))
                measures[key + "/winning_B_minus_background"].append(
                    float(logits[i, 1:].max() - logits[i, 0])
                )
                measures[key + "/absolute_slot_logit_gap"].append(
                    float((logits[i, 1] - logits[i, 2]).abs())
                )
                if labels[i] > 0:
                    measures[key + "/correct_slot_conditional_probability"].append(
                        float(conditional[i, int(labels[i]) - 1])
                    )
    presence = torch.sigmoid(objects)
    measures["presence/B_event" if is_b.any() else "presence/continuum"] = (
        presence.tolist()
    )
    sizes = {}
    for k in (1, 2):
        m = labels == k
        n = int(m.sum())
        if not n:
            continue
        bucket = "1-4" if n <= 4 else "5-8" if n <= 8 else "9-16" if n <= 16 else "17+"
        c = sizes.setdefault(bucket, Counter())
        c.update(
            trials=1,
            B_nodes=n,
            B_to_unassigned=int((m & (pred == 0)).sum()),
            B_to_other_B=int((m & (pred > 0) & (pred != k)).sum()),
            B_correct=int((m & (pred == k)).sum()),
            conditional_slot_correct=int((m & (bslot == k)).sum()),
        )
    return (
        dict(counts),
        dict(measures),
        sizes,
        {"scores": binary[valid].tolist(), "labels": is_b[valid].int().tolist()},
        swap,
    )


def logit_loss_audit(logits, objects, targets):
    z = logits.detach().clone().requires_grad_(True)
    o = objects.detach().clone().requires_grad_(True)
    parts, swap, totals = loss_parts(z, o, targets)
    native = direct_membership_loss(
        z[None],
        o[None],
        targets[None],
        torch.ones_like(targets[None], dtype=torch.bool),
    )
    assert torch.allclose(native, totals.min(), atol=1e-6, rtol=1e-6)
    labels = parts["labels"]
    valid = labels >= 0
    result = {
        "total": float(native.detach()),
        "ce": float(parts["ce"].detach()),
        "overlap": float(parts["overlap"].detach()),
        "presence": float(parts["presence"].detach()),
        "selected_swap": swap,
        "alternative_gap": float((totals[0] - totals[1]).detach().abs()),
        "valid_nodes": int(valid.sum()),
        "unknown_nodes": int((~valid).sum()),
        "classes": {},
    }
    gradients = {
        name: torch.autograd.grad(parts[name], z, retain_graph=True)[0]
        for name in ["ce", "overlap"]
    }
    for name, m in [
        ("B", labels > 0),
        ("background", labels == 0),
        ("unknown", labels < 0),
    ]:
        r = {
            "nodes": int(m.sum()),
            "ce_numerator": float(parts["ce_nodes"][m].detach().sum()),
            "ce_event_contribution": float(
                parts["ce_nodes"][m].detach().sum() / valid.sum().clamp_min(1)
            ),
        }
        for component, g in gradients.items():
            r[component + "_background_logit_gradient_sum"] = float(g[m, 0].sum())
            r[component + "_gradient_squared_norm"] = float(g[m].square().sum())
        result["classes"][name] = r
    return result


def main(parent, out):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ):
        raise RuntimeError("Fresh guarded2CPU/noGPU diagnostic required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from scripts.run_phase74_development import verify_bindings
    from hypertagging.training.capacity_development import (
        fresh_model,
        fresh_decoder,
        detector_features,
    )

    c = json.loads((parent / "successor-campaign-v1/contract.json").read_text())
    verify_bindings(c, Path(c["source_root"]))
    cache = torch.load(c["cache"]["path"], weights_only=False, map_location="cpu")
    train = cache["train"]
    assert len(train) == 1536
    # Validation rows are never iterated, evaluated, joined or used to select a hypothesis.
    del cache["development"]
    out.mkdir(exist_ok=False)
    gradient_uids = set()
    for cat in sorted({r["category"] for r in train}):
        gradient_uids.update(
            r["uid"]
            for r in sorted(
                [r for r in train if r["category"] == cat],
                key=lambda r: hashlib.sha256(
                    ("phase77-gradient:" + r["uid"]).encode()
                ).hexdigest(),
            )[:16]
        )
    write(
        out / "gradient-selection-private.json",
        {
            "uids": sorted(gradient_uids),
            "rule": "16/category SHA256(phase77-gradient:uid), training only",
        },
    )
    summary = {
        "version": "phase77-training-only-diagnostic-v1",
        "stage": "training_only_diagnostic",
        "validation_events_evaluated": 0,
        "new_training_jobs": 0,
        "contract": binding(parent / "successor-campaign-v1/contract.json"),
        "populations": {},
    }
    started = time.monotonic()
    cpu = time.process_time()
    term = json.loads(
        (
            parent / "successor-campaign-v1/runs/partial_context_on/terminal.json"
        ).read_text()
    )
    old_summary = read(term["summary"])
    trace_summary = json.loads(
        (parent / "endpoint-diagnostic-v1/diagnostic-summary.json").read_text()
    )
    for role, fit in [("train", "downstream"), ("tiny", "tiny")]:
        cp = next(
            b for b in term["checkpoints"] if Path(b["path"]).name == fit + "-final.pt"
        )
        assert binding(cp["path"]) == cp
        ck = torch.load(cp["path"], weights_only=False, map_location="cpu")
        model = fresh_model(128, cache["runtime_normalizer"])
        decoder = fresh_decoder()
        model.load_state_dict(ck["model_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])
        model.eval()
        decoder.eval()
        for q in model.parameters():
            q.requires_grad_(False)
        for q in model.encoder.parameters():
            q.requires_grad_(True)
        rows = (
            train
            if role == "train"
            else sum(
                (
                    [r for r in train if r["category"] == cat][:8]
                    for cat in ["charged", "mixed", "ccbar"]
                ),
                [],
            )
        )
        ref = trace_summary["arms"]["partial_context_on"][role]["trace"]
        assert binding(ref["path"]) == ref
        totals = defaultdict(Counter)
        cats = defaultdict(lambda: defaultdict(Counter))
        sizes = defaultdict(lambda: defaultdict(Counter))
        measures = defaultdict(list)
        category_measures = defaultdict(lambda: defaultdict(list))
        scores = defaultdict(list)
        labels_auc = defaultdict(list)
        loss_aggregate = defaultdict(Counter)
        loss_category = defaultdict(lambda: defaultdict(Counter))
        gradients = []
        replay_max = 0.0
        pairs = 0
        with (
            gzip.open(ref["path"], "rt") as old,
            gzip.open(out / f"{role}-model-only.jsonl.gz", "xt") as traces,
            gzip.open(out / f"{role}-joined-private.jsonl.gz", "xt") as joined,
        ):
            for index, row in enumerate(rows):
                with torch.no_grad():
                    h, _, _ = detector_features(model, row["detector"])
                    result = decoder(h, row["sources"])
                detached = {
                    stage: {
                        "logits": result[lk][0].detach().tolist(),
                        "presence_logits": result[ok][0].detach().tolist(),
                    }
                    for stage, lk, ok in [
                        ("proposal", "proposal_logits", "proposal_objects"),
                        ("refinement", "logits", "objects"),
                    ]
                }
                traces.write(
                    json.dumps({"uid": row["uid"], "outputs": detached}) + "\n"
                )
                previous = json.loads(next(old))
                assert previous["uid"] == row["uid"]
                record = {"uid": row["uid"], "category": row["category"], "stages": {}}
                for stage, v in detached.items():
                    z = torch.tensor(v["logits"])
                    o = torch.tensor(v["presence_logits"])
                    target = row["targets"]
                    delta = float(
                        (
                            z.softmax(-1)
                            - torch.tensor(
                                previous["trace"]["stages"][stage]["probabilities"]
                            )
                        )
                        .abs()
                        .max()
                    )
                    replay_max = max(replay_max, delta)
                    assert delta < 1e-7
                    counts, vals, sz, au, hard_swap = describe(z, o, target)
                    counts["nominal_B_trials"] = (
                        2 if row["category"] in ("charged", "mixed") else 0
                    )
                    counts["unavailable_B_trials"] = (
                        counts["nominal_B_trials"] - counts["available_B_targets"]
                    )
                    audit = logit_loss_audit(z, o, target)
                    audit["matching_disagrees_with_hard_error"] = int(
                        bool(audit["selected_swap"]) != hard_swap
                    )
                    totals[stage].update(counts)
                    cats[row["category"]][stage].update(counts)
                    for b, k in sz.items():
                        sizes[b][stage].update(k)
                    for k, vv in vals.items():
                        measures[stage + "/" + k].extend(vv)
                        category_measures[row["category"]][stage + "/" + k].extend(vv)
                    scores[stage].extend(au["scores"])
                    labels_auc[stage].extend(au["labels"])
                    flat = {
                        k: v for k, v in audit.items() if isinstance(v, (int, float))
                    }
                    for cl, vv in audit["classes"].items():
                        flat.update({cl + "/" + k: v for k, v in vv.items()})
                    loss_aggregate[stage].update(flat)
                    loss_category[row["category"]][stage].update(flat)
                    record["stages"][stage] = {"counts": counts, "loss_audit": audit}
                raw = [
                    sorted(torch.where(result["logits"][0].argmax(-1) == k)[0].tolist())
                    for k in (1, 2)
                ]
                assert (
                    old_summary[role]["events"][index]["uid"] == row["uid"]
                    and old_summary[role]["events"][index]["raw_groups"] == raw
                )
                joined.write(json.dumps(record) + "\n")
                pairs += len(row["targets"]) ** 2
                if role == "train" and row["uid"] in gradient_uids:
                    h, _, _ = detector_features(model, row["detector"])
                    rr = decoder(h, row["sources"])
                    mask = torch.ones_like(row["targets"][None], dtype=torch.bool)
                    prop = (
                        direct_membership_loss(
                            rr["proposal_logits"],
                            rr["proposal_objects"],
                            row["targets"][None],
                            mask,
                        )
                        / 2
                    )
                    ref_loss = (
                        direct_membership_loss(
                            rr["logits"], rr["objects"], row["targets"][None], mask
                        )
                        / 2
                    )
                    parameters = tuple(decoder.proposal.parameters())
                    gp = torch.autograd.grad(
                        prop, parameters, retain_graph=True, allow_unused=True
                    )
                    gr = torch.autograd.grad(ref_loss, parameters, allow_unused=True)
                    a = torch.cat(
                        [
                            torch.zeros_like(p).flatten() if g is None else g.flatten()
                            for p, g in zip(parameters, gp)
                        ]
                    )
                    b = torch.cat(
                        [
                            torch.zeros_like(p).flatten() if g is None else g.flatten()
                            for p, g in zip(parameters, gr)
                        ]
                    )
                    gradients.append(
                        {
                            "uid": row["uid"],
                            "category": row["category"],
                            "proposal_gradient_norm": float(a.norm()),
                            "refinement_gradient_norm": float(b.norm()),
                            "dot": float(a @ b),
                            "cosine": float(
                                (a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)
                            ),
                        }
                    )
                if index % 256 == 0:
                    print(role, index, flush=True)
            assert not old.readline()
        if role == "train":
            assert totals["proposal"]["B_to_unassigned"] == 4172
            assert totals["proposal"]["B_to_other_B"] == 1828
        summary["populations"][role] = {
            "events": len(rows),
            "checkpoint": cp,
            "reproduced_trace": ref,
            "max_probability_difference": replay_max,
            "stages": dict(totals),
            "by_category": dict(cats),
            "by_size": dict(sizes),
            "margins": {k: quantiles(v) for k, v in measures.items()},
            "category_margins": {
                c: {k: quantiles(v) for k, v in vv.items()}
                for c, vv in category_measures.items()
            },
            "foreground_rank_auc": {k: auc(scores[k], labels_auc[k]) for k in scores},
            "loss_component_sums": dict(loss_aggregate),
            "loss_component_sums_by_category": dict(loss_category),
            "gradient_probes": len(gradients),
            "gradient_by_category": {
                cat: {
                    key: quantiles([g[key] for g in gradients if g["category"] == cat])
                    for key in [
                        "proposal_gradient_norm",
                        "refinement_gradient_norm",
                        "dot",
                        "cosine",
                    ]
                }
                for cat in sorted({g["category"] for g in gradients})
            },
            "event_node_squared_proxy": pairs,
            "detached_trace": binding(out / f"{role}-model-only.jsonl.gz"),
            "joined_private": binding(out / f"{role}-joined-private.jsonl.gz"),
        }
        write(out / f"{role}-gradient-private.json", gradients)
    summary["compute"] = {
        "wall_seconds": time.monotonic() - started,
        "process_cpu_seconds": time.process_time() - cpu,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "forward_events": 1560,
        "additional_gradient_events": 96,
        "new_fit_presentations": 0,
        "parameters": {
            "encoder": sum(p.numel() for p in model.encoder.parameters()),
            "decoder": sum(p.numel() for p in decoder.parameters()),
        },
    }
    write(out / "diagnostic-summary.json", summary)
    print(json.dumps(summary["compute"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.parent, a.output)
