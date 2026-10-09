"""Bounded frozen-feature, training-role partition diagnosis; no deployed changes."""

from __future__ import annotations
import argparse
from collections import defaultdict
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from scripts.diagnose_phase77_proposals import auc, quantiles, loss_parts  # noqa: E402
from scripts.prepare_phase74_development_data import binding, write  # noqa: E402


def split_rows(rows):
    roles = {}
    for cat in sorted({r["category"] for r in rows}):
        ordered = sorted(
            [r for r in rows if r["category"] == cat],
            key=lambda r: hashlib.sha256(
                ("phase78-probe:" + r["uid"]).encode()
            ).hexdigest(),
        )
        if len(ordered) != 256:
            raise ValueError("Authenticated category capacity changed")
        roles.update(
            {
                r["uid"]: "fit" if i < 192 else "assessment"
                for i, r in enumerate(ordered)
            }
        )
    if len(roles) != len(rows):
        raise ValueError("Duplicate training identities")
    return roles


def source_pairs(sources):
    ij = torch.triu_indices(len(sources), len(sources), offset=1)
    shared = (sources[ij[0]] & sources[ij[1]]).any(-1)
    return ij[:, ~shared], int(shared.sum())


def pair_features(h, ij):
    a, b = h[ij[0]], h[ij[1]]
    return torch.cat(((a - b).abs(), a * b), -1)


def pair_targets(targets, ij):
    a, b = targets[ij[0]], targets[ij[1]]
    known = (a >= 0) & (b >= 0)
    both = (a > 0) & (b > 0)
    return (a == b).long(), both, known & ~both, ~known


def conditional_counts(z, target):
    b = target > 0
    pred = z[:, 1:].argmax(-1) + 1
    counts = [
        int(((pred == (3 - target if swap else target)) & b).sum())
        for swap in (False, True)
    ]
    sizes = [int((target == k).sum()) for k in (1, 2)]
    return {
        "B_nodes": int(b.sum()),
        "conditional_optimal_correct": max(counts),
        "majority_slot_null_correct": max(sizes),
        "matching_correct_count_gap": abs(counts[0] - counts[1]),
        "tie": int(counts[0] == counts[1]),
        "has_B": int(bool(b.any())),
    }


def head_capture(head, features, mask):
    """Read-only exact native head intermediates; no targets accepted."""
    h = head.nodes(features)
    pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    q = head.queries[None] + pooled[:, None]
    a, _ = head.attention(q, h, h, key_padding_mask=~mask, need_weights=False)
    q = head.norm(q + a)
    q = head.norm(q + head.refine(q))
    z = torch.cat(
        (head.background(h), torch.einsum("bnd,bkd->bnk", h, q) / h.shape[-1] ** 0.5),
        -1,
    )
    return h[0].detach(), q[0].detach(), z[0].detach()


def fit_probe(records, shuffled=False):
    generator = torch.Generator().manual_seed(7809)
    eligible = []
    for r in records:
        if r["partition"] != "fit":
            continue
        y = r["target"].clone()
        if shuffled:
            positions = torch.where(y > 0)[0]
            seed = int(
                hashlib.sha256(("7810:" + r["uid"]).encode()).hexdigest()[:8], 16
            )
            order = torch.randperm(
                len(positions), generator=torch.Generator().manual_seed(seed)
            )
            y[positions] = y[positions[order]]
        labels, both, _, _ = pair_targets(y, r["ij"])
        classes = [torch.where(both & (labels == k))[0] for k in (0, 1)]
        if all(len(x) for x in classes):
            eligible.append((r, classes))
    if not eligible:
        raise ValueError("No fitting pair support")
    # Normalization fits only original fitting-event B-pair features; independent of labels.
    xx = torch.cat(
        [pair_features(r["encoder"], r["ij"][:, torch.cat(c)]) for r, c in eligible]
    )
    mean, scale = xx.mean(0), xx.std(0).clamp_min(1e-5)
    del xx
    probe = torch.nn.Linear(len(mean), 1)
    torch.nn.init.zeros_(probe.weight)
    torch.nn.init.zeros_(probe.bias)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    curve, used = [], set()
    for step in range(512):
        xs, ys = [], []
        for k in range(256):
            r, classes = eligible[
                int(torch.randint(len(eligible), (), generator=generator))
            ]
            label = k % 2
            candidates = classes[label]
            pos = candidates[
                int(torch.randint(len(candidates), (), generator=generator))
            ]
            xs.append(pair_features(r["encoder"], r["ij"][:, pos : pos + 1])[0])
            ys.append(label)
            used.add(r["uid"])
        x = (torch.stack(xs) - mean) / scale
        loss = F.binary_cross_entropy_with_logits(
            probe(x).flatten(), torch.tensor(ys, dtype=torch.float32)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    return {
        "state": probe.state_dict(),
        "mean": mean,
        "scale": scale,
        "curve": curve,
        "unique_fit_events": len(used),
        "eligible_fit_events": len(eligible),
        "pair_presentations": 512 * 256,
        "parameters": sum(p.numel() for p in probe.parameters()),
    }


def probe_score(fit, x):
    return (
        F.linear(
            (x - fit["mean"]) / fit["scale"],
            fit["state"]["weight"],
            fit["state"]["bias"],
        )
        .flatten()
        .sigmoid()
    )


def score_record(r, probes):
    ij, target = r["ij"], r["target"]
    labels, both, bg, unknown = pair_targets(target, ij)
    scores = {}
    for stage in ("proposal", "refinement"):
        p = r[stage]["logits"][:, 1:].softmax(-1)
        scores[stage + "_conditional_same"] = (p[ij[0]] * p[ij[1]]).sum(-1)
        h = F.normalize(r[stage]["nodes"], dim=-1)
        scores[stage + "_cosine"] = (h[ij[0]] * h[ij[1]]).sum(-1)
    h = F.normalize(r["encoder"], dim=-1)
    scores["encoder_cosine"] = (h[ij[0]] * h[ij[1]]).sum(-1)
    x = pair_features(r["encoder"], ij)
    for name, fit in probes.items():
        scores[name] = probe_score(fit, x)
    value = {
        "category": r["category"],
        "partition": r["partition"],
        "B_sizes": [int((target == k).sum()) for k in (1, 2)],
        "nodes": len(target),
        "unknown_nodes": int((target < 0).sum()),
        "pairs": len(labels),
        "excluded_shared_source_pairs": r["excluded_shared_source_pairs"],
        "unknown_pairs": int(unknown.sum()),
        "B_pair_support": int(both.sum()),
        "background_pair_support": int(bg.sum()),
        "scores": {},
        "stages": {},
    }
    for name, s in scores.items():
        value["scores"][name] = {
            "B_pair_auc": auc(s[both].tolist(), labels[both].tolist()),
            "background": quantiles(s[bg].tolist()),
            "background_score_ge_half": int((s[bg] >= 0.5).sum()),
            "same_B": quantiles(s[both & (labels == 1)].tolist()),
            "cross_B": quantiles(s[both & (labels == 0)].tolist()),
        }
    for stage in ("proposal", "refinement"):
        z, q = r[stage]["logits"], r[stage]["queries"]
        counts = conditional_counts(z, target)
        _, swap, totals = loss_parts(z, r[stage]["objects"], target)
        counts.update(
            native_permutation=int(swap),
            native_matching_loss_gap=float((totals[0] - totals[1]).abs()),
            query_cosine=float(F.cosine_similarity(q[:1], q[1:])[0]),
            query_difference_norm=float((q[0] - q[1]).norm()),
            query_norm_mean=float(q.norm(dim=-1).mean()),
            slot_logit_gap=quantiles((z[:, 1] - z[:, 2]).abs().tolist()),
        )
        value["stages"][stage] = counts
    # Raw score arrays only private; pooled summaries retain support and counts publicly.
    return value, {
        k: (s[both].tolist(), labels[both].tolist()) for k, s in scores.items()
    }


def aggregate(records):
    result = {}
    for key in (
        "all",
        "charged",
        "mixed",
        "ccbar",
        "uubar",
        "ddbar",
        "ssbar",
        "size1-4",
        "size5-8",
        "size9-16",
        "size17+",
    ):

        def keep(r):
            if key == "all":
                return True
            if key.startswith("size"):
                lo, hi = {
                    "size1-4": (1, 4),
                    "size5-8": (5, 8),
                    "size9-16": (9, 16),
                    "size17+": (17, 999),
                }[key]
                return any(lo <= n <= hi for n in r["B_sizes"])
            return r["category"] == key

        rr = [r for r in records if keep(r)]
        result[key] = {
            "events": len(rr),
            "size_condition": "event has at least one target in range; strata overlap"
            if key.startswith("size")
            else None,
            "nodes": sum(r["nodes"] for r in rr),
            "B_pair_support": sum(r["B_pair_support"] for r in rr),
            "background_pair_support": sum(r["background_pair_support"] for r in rr),
            "unknown_pairs": sum(r["unknown_pairs"] for r in rr),
            "excluded_shared_source_pairs": sum(
                r["excluded_shared_source_pairs"] for r in rr
            ),
            "scores": {},
            "stages": {},
        }
        for name in records[0]["scores"]:
            vals = [
                r["scores"][name]["B_pair_auc"]["auc"]
                for r in rr
                if r["scores"][name]["B_pair_auc"]["auc"] is not None
            ]
            result[key]["scores"][name] = {
                "event_auc": quantiles(vals),
                "unavailable_events": len(rr) - len(vals),
                "background_score_ge_half": sum(
                    r["scores"][name]["background_score_ge_half"] for r in rr
                ),
                "same_pairs": sum(
                    r["scores"][name]["B_pair_auc"]["positive"] for r in rr
                ),
                "cross_pairs": sum(
                    r["scores"][name]["B_pair_auc"]["negative"] for r in rr
                ),
            }
        for stage in ("proposal", "refinement"):
            result[key]["stages"][stage] = {
                k: sum(r["stages"][stage][k] for r in rr)
                for k in (
                    "B_nodes",
                    "conditional_optimal_correct",
                    "majority_slot_null_correct",
                    "tie",
                    "has_B",
                )
            }
            result[key]["stages"][stage].update(
                {
                    k: quantiles([r["stages"][stage][k] for r in rr])
                    for k in (
                        "query_cosine",
                        "query_difference_norm",
                        "query_norm_mean",
                        "matching_correct_count_gap",
                        "native_matching_loss_gap",
                    )
                }
            )
    return result


def main(parent, previous, output):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ):
        raise RuntimeError("Guarded fresh2CPU/noGPU allocation required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from scripts.run_phase74_development import verify_bindings
    from hypertagging.training.capacity_development import (
        fresh_model,
        fresh_decoder,
        detector_features,
    )

    start, cpu = time.monotonic(), time.process_time()
    c = json.loads((parent / "successor-campaign-v1/contract.json").read_text())
    verify_bindings(c, Path(c["source_root"]))
    cache = torch.load(c["cache"]["path"], weights_only=False, map_location="cpu")
    del cache["development"]
    rows = cache["train"]
    roles = split_rows(rows)
    output.mkdir(exist_ok=False)
    write(output / "probe-partition-private.json", roles)
    term = json.loads(
        (
            parent / "successor-campaign-v1/runs/partial_context_on/terminal.json"
        ).read_text()
    )
    summary = {
        "stage": "training_only_diagnostic",
        "validation_events_evaluated": 0,
        "scientific_training_jobs": 0,
        "probe_fits": 2,
        "populations": {},
        "contract": binding(parent / "successor-campaign-v1/contract.json"),
    }
    all_records = {}
    node_proxy = 0
    for role, fitname in [("train", "downstream"), ("tiny", "tiny")]:
        cp = next(
            b
            for b in term["checkpoints"]
            if Path(b["path"]).name == fitname + "-final.pt"
        )
        assert binding(cp["path"]) == cp
        ck = torch.load(cp["path"], weights_only=False, map_location="cpu")
        model = fresh_model(128, cache["runtime_normalizer"])
        decoder = fresh_decoder()
        model.load_state_dict(ck["model_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])
        model.eval()
        decoder.eval()
        selected = (
            rows
            if role == "train"
            else sum(
                (
                    [r for r in rows if r["category"] == cat][:8]
                    for cat in ("charged", "mixed", "ccbar")
                ),
                [],
            )
        )
        records = []
        delta = 0.0
        with gzip.open(
            previous / "diagnostic-v1" / f"{role}-model-only.jsonl.gz", "rt"
        ) as ref:
            for row in selected:
                captured = {}

                def hook(stage):
                    def capture(module, args, outputs):
                        n, q, z = head_capture(module, *args)
                        assert torch.equal(z, outputs[0][0])
                        captured[stage] = {
                            "nodes": n,
                            "queries": q,
                            "logits": z,
                            "objects": outputs[1][0].detach(),
                        }

                    return capture

                handles = [
                    getattr(decoder, s).register_forward_hook(hook(s))
                    for s in ("proposal", "refinement")
                ]
                with torch.no_grad():
                    h, _, _ = detector_features(model, row["detector"])
                    decoder(h, row["sources"])
                for handle in handles:
                    handle.remove()
                old = json.loads(next(ref))
                assert old["uid"] == row["uid"]
                for s in captured:
                    delta = max(
                        delta,
                        float(
                            (
                                captured[s]["logits"]
                                - torch.tensor(old["outputs"][s]["logits"])
                            )
                            .abs()
                            .max()
                        ),
                    )
                assert delta == 0
                ij, excluded_pairs = source_pairs(row["sources"])
                # Detached truth-free state serialized before target join.
                records.append(
                    {
                        "uid": row["uid"],
                        "encoder": h.detach(),
                        "ij": ij,
                        "excluded_shared_source_pairs": excluded_pairs,
                        **captured,
                    }
                )
                node_proxy += len(h) ** 2
            assert not ref.readline()
        torch.save(records, output / f"{role}-model-only.pt")
        for r, row in zip(records, selected):
            r.update(
                target=row["targets"].clone(),
                category=row["category"],
                partition=roles[row["uid"]],
            )
        all_records[role] = records
        summary["populations"][role] = {
            "events": len(records),
            "checkpoint": cp,
            "max_logit_difference": delta,
            "trace": binding(output / f"{role}-model-only.pt"),
        }
    probes = {
        name: fit_probe(all_records["train"], shuffled=name == "shuffled_probe")
        for name in ("pair_probe", "shuffled_probe")
    }
    torch.save(probes, output / "probe-final.pt")
    summary["probes"] = {
        k: {x: y for x, y in v.items() if x not in ("state", "mean", "scale")}
        for k, v in probes.items()
    }
    for role, records in all_records.items():
        values = []
        pooled = defaultdict(lambda: [[], []])
        for r in records:
            value, scores = score_record(r, probes)
            values.append(value)
            for name, (s, labels) in scores.items():
                key = r["partition"] + "/" + name
                pooled[key][0].extend(s)
                pooled[key][1].extend(labels)
        write(
            output / f"{role}-event-metrics-private.json",
            [{"uid": r["uid"], **v} for r, v in zip(records, values)],
        )
        summary["populations"][role].update(
            aggregate=aggregate(values),
            partitions={
                part: aggregate([v for v in values if v["partition"] == part])
                for part in ("fit", "assessment")
                if any(v["partition"] == part for v in values)
            },
            pooled_pair_auc={k: auc(*v) for k, v in pooled.items()},
        )
        if role == "train":
            rng = np.random.default_rng(7811)
            percat = []
            for cat in ("charged", "mixed"):
                rr = [
                    v
                    for v in values
                    if v["partition"] == "assessment"
                    and v["category"] == cat
                    and v["scores"]["pair_probe"]["B_pair_auc"]["auc"] is not None
                ]
                percat.append(
                    np.array(
                        [
                            v["scores"]["pair_probe"]["B_pair_auc"]["auc"]
                            - v["scores"]["proposal_conditional_same"]["B_pair_auc"][
                                "auc"
                            ]
                            for v in rr
                        ]
                    )
                )
            draws = [
                float(
                    np.concatenate(
                        [a[rng.integers(len(a), size=len(a))] for a in percat]
                    ).mean()
                )
                for _ in range(2000)
            ]
            summary["paired_probe_minus_native_event_auc"] = {
                "events": sum(map(len, percat)),
                "difference": float(np.concatenate(percat).mean()),
                "interval95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "scope": "training-role probe-assessment, correlated pairs aggregated per collision; no fresh confirmation",
            }
    summary["compute"] = {
        "wall_seconds": time.monotonic() - start,
        "process_cpu_seconds": time.process_time() - cpu,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "forward_events": 1560,
        "distinct_training_identities": 1536,
        "node_squared_proxy": node_proxy,
        "probe_updates": 1024,
        "probe_pair_presentations": 262144,
        "scientific_model_updates": 0,
        "parameters": {
            "encoder": sum(p.numel() for p in model.encoder.parameters()),
            "decoder": sum(p.numel() for p in decoder.parameters()),
        },
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary["compute"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.parent, a.previous, a.output)
