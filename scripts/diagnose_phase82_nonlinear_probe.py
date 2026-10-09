"""Bounded nonlinear readability probe on authenticated frozen training features."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np  # noqa:E402
import torch  # noqa:E402
from torch.nn import functional as F  # noqa:E402
from scripts.prepare_phase74_development_data import binding, write  # noqa:E402
from scripts.diagnose_phase78_partition import (  # noqa: E402
    pair_features,
    pair_targets,
    source_pairs,
    score_record,
    aggregate,
)  # noqa:E402
from scripts.diagnose_phase77_proposals import auc, quantiles  # noqa:E402
from scripts.run_phase74_development import save  # noqa:E402

UPDATES = 2048
BATCH = 256
SEED = 8209


def validate_contract(c):
    if (
        c["stage"] != "training_role_diagnostic"
        or c["updates"] != UPDATES
        or c["batch_size"] != BATCH
        or c["probe_parameters"] != 16513
    ):
        raise ValueError("Diagnostic budget/capacity changed")
    if c["arms"] != ["nonlinear_probe", "shuffled_nonlinear"] or c["resources"] != {
        "cpus": 2,
        "memory_gib": 16,
        "hours": 1,
        "gpus": 0,
        "requeue": False,
    }:
        raise ValueError("Arms/resources changed")
    if (
        c["scientific_model_updates"] != 0
        or c["validation_events"] != 0
        or c["automatic_successor"]
    ):
        raise ValueError("Diagnostic authority changed")


def new_probe():
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)
    )


def prepare_fit(records, reference, shuffled=False):
    xs, ys, weights, classes, ids = [], [], [], [], []
    offset = 0
    for r in records:
        if r["partition"] != "fit":
            continue
        target = r["target"].clone()
        if target.dtype != torch.long or not ((target >= -1) & (target <= 2)).all():
            raise ValueError("Target encoding changed")
        if shuffled:
            positions = torch.where(target > 0)[0]
            seed = int(
                hashlib.sha256(("7810:" + r["uid"]).encode()).hexdigest()[:8], 16
            )
            order = torch.randperm(
                len(positions), generator=torch.Generator().manual_seed(seed)
            )
            target[positions] = target[positions[order]]
        labels, both, _, _ = pair_targets(target, r["ij"])
        cc = [torch.where(both & (labels == k))[0] for k in (0, 1)]
        if not all(len(v) for v in cc):
            continue
        idx = torch.cat(cc)
        x = pair_features(r["encoder"], r["ij"][:, idx]).detach()
        xs.append((x - reference["mean"]) / reference["scale"])
        ys.append(labels[idx].float())
        weights.append(torch.cat([torch.full((len(v),), 0.5 / len(v)) for v in cc]))
        classes.append(
            [
                list(range(offset, offset + len(cc[0]))),
                list(range(offset + len(cc[0]), offset + len(idx))),
            ]
        )
        offset += len(idx)
        ids.append(r["uid"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate fitting roles")
    return torch.cat(xs), torch.cat(ys), torch.cat(weights) / len(ids), classes, ids


def risk(model, x, y, w, gradient=False):
    model.zero_grad(set_to_none=True)
    total = 0.0
    with torch.set_grad_enabled(gradient):
        for start in range(0, len(x), 4096):
            loss = (
                F.binary_cross_entropy_with_logits(
                    model(x[start : start + 4096]).flatten(),
                    y[start : start + 4096],
                    reduction="none",
                )
                * w[start : start + 4096]
            ).sum()
            if gradient:
                loss.backward()
            total += float(loss.detach())
    norm = (
        sum(
            float(p.grad.square().sum())
            for p in model.parameters()
            if p.grad is not None
        )
        ** 0.5
        if gradient
        else None
    )
    return {"risk": total, "gradient_l2": norm}


def fit(records, reference, shuffled=False, *, updates=UPDATES, batch=BATCH):
    x, y, w, classes, ids = prepare_fit(records, reference, shuffled)
    if (
        x.shape[1] != 256
        or not torch.isfinite(x).all()
        or abs(float(w.sum()) - 1) > 1e-5
    ):
        raise ValueError("Feature/weight contract")
    model = new_probe()
    initial = hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in model.parameters())
    ).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(SEED)
    curve = []
    risks = [{"step": 0, **risk(model, x, y, w)}]
    used = set()
    pairs = set()
    order = hashlib.sha256()
    start = time.monotonic()
    for step in range(updates):
        events = torch.randint(len(ids), (batch,), generator=generator).tolist()
        u = torch.rand(batch, generator=generator).tolist()
        idx = []
        for k, (event, v) in enumerate(zip(events, u)):
            label = k % 2
            available = classes[event][label]
            pos = available[int(v * len(available))]
            idx.append(pos)
            used.add(ids[event])
            pairs.add(pos)
            order.update((ids[event] + ":" + str(label) + "\n").encode())
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x[idx]).flatten(), y[idx])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0, error_if_nonfinite=True
        )
        if not torch.isfinite(loss) or (
            step == 0
            and not all(
                p.grad is not None and float(p.grad.abs().sum()) > 0
                for p in model.parameters()
            )
        ):
            raise ValueError("Nonfinite or absent probe gradient")
        opt.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite probe state")
        curve.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
            }
        )
        if (step + 1) % 256 == 0 or step + 1 == updates:
            risks.append({"step": step + 1, **risk(model, x, y, w)})
        if step == 31 and (time.monotonic() - start) / 32 * updates * 3 > 3000:
            raise ValueError("Bounded diagnostic runtime forecast failed")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 12 * 1024**2:
            raise ValueError("Diagnostic memory admission failed")
    final = risk(model, x, y, w, gradient=True)
    return (
        model,
        {
            "state": model.state_dict(),
            "mean": reference["mean"],
            "scale": reference["scale"],
        },
        {
            "updates": updates,
            "batch_size": batch,
            "optimizer_pair_presentations": updates * batch,
            "full_risk_pair_presentations": len(x) * (len(risks) + 1),
            "eligible_pairs": len(x),
            "unique_sampled_pairs": len(pairs),
            "fitting_events": len(ids),
            "unique_fitting_events": len(used),
            "fitting_uids": ids,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_state_sha256": initial,
            "event_class_order_sha256": order.hexdigest(),
            "curve": curve,
            "full_fitting_risks": risks,
            "final_risk_gradient": final,
            "convergence": "Fixed finite nonconvex fit; no stationarity or global-optimum assertion",
            "wall_seconds": time.monotonic() - start,
        },
    )


def score_features(model, h, ij, reference):
    with torch.inference_mode():
        x = (pair_features(h, ij) - reference["mean"]) / reference["scale"]
        return model(x).flatten().sigmoid().detach()


def main(phase81, phase78, phase79, output):
    from scripts.phase76_development import guarded

    guarded()
    from scripts.run_phase74_development import verify_bindings

    contract = json.loads((output.parent / "diagnostic-contract.json").read_text())
    validate_contract(contract)
    if (
        str(ROOT) != contract["source_root"]
        or str(output.resolve()) != contract["output_root"]
    ):
        raise ValueError("Frozen source/output path changed")
    for name, root in [
        ("phase81", phase81),
        ("phase78", phase78),
        ("phase79", phase79),
    ]:
        if str(root.resolve()) != contract["input_roots"][name]:
            raise ValueError("Unbound diagnostic input root")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != contract["source_sha"]
    ):
        raise ValueError("Source commit changed")
    verify_bindings(contract, ROOT)
    start = time.monotonic()
    cpu = time.process_time()
    c = json.loads((phase81 / "campaign-v1/contract.json").read_text())
    if binding(c["cache"]["path"]) != c["cache"]:
        raise ValueError("Train-only cache changed")
    review = json.loads((phase81 / "training-screen-review.json").read_text())
    if not review["training_screen_gates"]["control_exact_replay"]:
        raise ValueError("Native reference mismatch")
    for root, name in [
        (phase78, "diagnostic-terminal-v2.json"),
        (phase79, "terminal-review.json"),
    ]:
        terminal = json.loads((root / name).read_text())
        for b in terminal.get("artifacts", terminal.get("bindings", [])):
            if binding(b["path"]) != b:
                raise ValueError("Frozen evidence changed")
    cache = torch.load(c["cache"]["path"], map_location="cpu", weights_only=False)
    if set(cache) != {"train", "runtime_normalizer"}:
        raise ValueError("Validation in diagnostic cache")
    rows = {r["uid"]: r for r in cache["train"]}
    roles = json.loads(
        (phase78 / "diagnostic-v2/probe-partition-private.json").read_text()
    )
    traces = torch.load(
        phase78 / "diagnostic-v2/train-model-only.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        set(rows) != set(roles)
        or set(rows) != {r["uid"] for r in traces}
        or len(traces) != 1536
    ):
        raise ValueError("Cohort mismatch")
    for r in traces:
        row = rows[r["uid"]]
        ij, excluded = source_pairs(row["sources"])
        if (
            not torch.equal(ij, r["ij"])
            or excluded != r["excluded_shared_source_pairs"]
        ):
            raise ValueError("Source-pair binding changed")
        r.update(
            target=row["targets"], category=row["category"], partition=roles[r["uid"]]
        )
    old = torch.load(
        phase78 / "diagnostic-v2/probe-final.pt", map_location="cpu", weights_only=False
    )
    linear = torch.load(
        phase79 / "diagnostic-v1/probe-final.pt", map_location="cpu", weights_only=False
    )
    reference = old["pair_probe"]
    output.mkdir(exist_ok=False)
    models = {}
    states = {}
    fits = {}
    for name, shuffled in [("nonlinear_probe", False), ("shuffled_nonlinear", True)]:
        models[name], states[name], fits[name] = fit(traces, reference, shuffled)
        print(
            name,
            {
                k: v
                for k, v in fits[name].items()
                if k not in ("curve", "fitting_uids", "full_fitting_risks")
            },
            flush=True,
        )
    if (
        len({v["initial_state_sha256"] for v in fits.values()}) != 1
        or len({v["event_class_order_sha256"] for v in fits.values()}) != 1
    ):
        raise ValueError("Unmatched fit controls")
    save(output / "probe-final.pt", states)
    # Complete detector-feature-only probe scoring before truth joins below.
    scored = [
        {
            "uid": r["uid"],
            "scores": {
                name: score_features(model, r["encoder"], r["ij"], reference)
                for name, model in models.items()
            },
        }
        for r in traces
    ]
    save(output / "probe-model-only-scores.pt", scored)
    metrics = []
    for r, pred in zip(traces, scored):
        assert r["uid"] == pred["uid"]
        value, _ = score_record(r, {"optimized_linear": linear["pair_probe"]})
        labels, both, bg, _ = pair_targets(r["target"], r["ij"])
        for name, s in pred["scores"].items():
            value["scores"][name] = {
                "B_pair_auc": auc(s[both].tolist(), labels[both].tolist()),
                "background": quantiles(s[bg].tolist()),
                "background_score_ge_half": int((s[bg] >= 0.5).sum()),
                "same_B": quantiles(s[both & (labels == 1)].tolist()),
                "cross_B": quantiles(s[both & (labels == 0)].tolist()),
            }
        metrics.append({"uid": r["uid"], **value})
    contrasts = {}
    rng = np.random.default_rng(SEED)
    for other in [
        "proposal_conditional_same",
        "optimized_linear",
        "shuffled_nonlinear",
    ]:
        strata = []
        for cat in ["charged", "mixed"]:
            values = []
            for r in metrics:
                if r["partition"] != "assessment" or r["category"] != cat:
                    continue
                a = r["scores"]["nonlinear_probe"]["B_pair_auc"]["auc"]
                b = r["scores"][other]["B_pair_auc"]["auc"]
                if a is not None and b is not None:
                    values.append(a - b)
            strata.append(np.array(values))
        if any(not len(v) for v in strata):
            raise ValueError("Assessment support missing")
        draws = [
            float(
                np.concatenate(
                    [v[rng.integers(len(v), size=len(v))] for v in strata]
                ).mean()
            )
            for _ in range(2000)
        ]
        contrasts[other] = {
            "paired_events": sum(map(len, strata)),
            "difference": float(np.concatenate(strata).mean()),
            "interval95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "scope": "training-role event-disjoint probe assessment, one seed; not independent physics confirmation",
        }
    write(output / "event-metrics-private.json", metrics)
    result = {
        "stage": "training_role_nonlinear_probe",
        "scientific_model_updates": 0,
        "encoder_forwards": 0,
        "validation_events": 0,
        "probe_fits": 2,
        "fits": fits,
        "paired_contrasts": contrasts,
        "partitions": {
            p: aggregate([r for r in metrics if r["partition"] == p])
            for p in ["fit", "assessment"]
        },
        "readability_gate": all(v["interval95"][0] > 0 for v in contrasts.values()),
        "background_interpretation": "Same/cross-B probe is not a foreground detector. Background-containing score support is diagnostic, never deployable acceptance.",
        "physical_or_exact_membership_improvement": "UNMEASURED; probe does not generate sets or trees",
        "compute": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "model_trace_identities": 1536,
            "score_pair_presentations": 2 * sum(r["ij"].shape[1] for r in traces),
        },
    }
    write(output / "summary.json", result)
    write(
        output / "terminal.json",
        {
            "status": "COMPLETED",
            "bindings": [binding(p) for p in sorted(output.iterdir()) if p.is_file()],
        },
    )
    print("READABILITY_GATE", result["readability_gate"], contrasts, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["phase81", "phase78", "phase79", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    main(a.phase81, a.phase78, a.phase79, a.output)
