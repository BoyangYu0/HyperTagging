"""Exact-risk optimization audit of frozen Phase78 pair probes, training role only."""

from __future__ import annotations
import argparse
from collections import defaultdict
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
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402
from scripts.diagnose_phase78_partition import (  # noqa: E402
    pair_features,
    pair_targets,
    score_record,
    aggregate,
)
from scripts.diagnose_phase77_proposals import auc  # noqa: E402
from scripts.prepare_phase74_development_data import binding, write  # noqa: E402


def risk_gradient(theta, x, y, w):
    z = x @ theta[:-1] + theta[-1]
    residual = w * (expit(z) - y)
    loss = float(w @ (np.logaddexp(0, z) - y * z))
    gradient = np.r_[x.T @ residual, residual.sum()]
    return loss, gradient


def prepare(records, reference, shuffled=False):
    xx, yy, ww, identities = [], [], [], []
    for r in records:
        if r["partition"] != "fit":
            continue
        target = r["target"].clone()
        if shuffled:
            ids = torch.where(target > 0)[0]
            seed = int(
                hashlib.sha256(("7810:" + r["uid"]).encode()).hexdigest()[:8], 16
            )
            order = torch.randperm(
                len(ids), generator=torch.Generator().manual_seed(seed)
            )
            target[ids] = target[ids[order]]
        labels, both, _, _ = pair_targets(target, r["ij"])
        classes = [torch.where(both & (labels == k))[0] for k in (0, 1)]
        if not all(len(c) for c in classes):
            continue
        indices = torch.cat(classes)
        x = pair_features(r["encoder"], r["ij"][:, indices])
        # Preserve the prior FP32 feature transform, then evaluate convex risk in float64.
        xx.append(((x - reference["mean"]) / reference["scale"]).double().numpy())
        yy.append(labels[indices].double().numpy())
        ww.append(
            np.r_[
                np.full(len(classes[0]), 0.5 / len(classes[0])),
                np.full(len(classes[1]), 0.5 / len(classes[1])),
            ]
        )
        identities.append(r["uid"])
    if not identities:
        raise ValueError("No eligible fitting events")
    return (
        np.concatenate(xx),
        np.concatenate(yy),
        np.concatenate(ww) / len(identities),
        identities,
    )


def fit(records, reference, shuffled=False):
    x, y, w, ids = prepare(records, reference, shuffled)
    initial = np.zeros(x.shape[1] + 1)
    old = np.r_[
        reference["state"]["weight"].numpy().ravel(),
        reference["state"]["bias"].numpy().ravel(),
    ].astype("float64")
    old_loss, old_grad = risk_gradient(old, x, y, w)
    evaluations = []

    def fun(t):
        if len(evaluations) >= 300:
            raise RuntimeError("Preregistered function-evaluation cap reached")
        loss, grad = risk_gradient(t, x, y, w)
        evaluations.append({"risk": loss, "gradient_inf": float(np.abs(grad).max())})
        return loss, grad

    result = minimize(
        fun,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": 200,
            "maxfun": 280,
            "maxls": 20,
            "gtol": 1e-7,
            "ftol": 1e-12,
        },
    )
    final_loss, gradient = risk_gradient(result.x, x, y, w)
    assert np.isfinite(result.x).all() and len(evaluations) <= 300
    final = {
        "state": {
            "weight": torch.tensor(result.x[:-1][None], dtype=torch.float32),
            "bias": torch.tensor(result.x[-1:], dtype=torch.float32),
        },
        "mean": reference["mean"],
        "scale": reference["scale"],
    }
    final32 = np.r_[
        final["state"]["weight"].numpy().ravel(), final["state"]["bias"].numpy().ravel()
    ].astype("float64")
    loss32, grad32 = risk_gradient(final32, x, y, w)
    z = x @ result.x[:-1] + result.x[-1]
    prob = expit(z)
    design = np.c_[x, np.ones(len(x))]
    hessian = design.T @ (design * (w * prob * (1 - prob))[:, None])
    eig = np.linalg.eigvalsh(hessian)
    return final, {
        "optimizer": "L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": len(evaluations),
        "optimizer_pair_presentations": len(x) * len(evaluations),
        "actual_pair_presentations_including_risk_and_hessian": len(x) * (len(evaluations) + 4),
        "risk_comparison_pair_passes": 3,
        "hessian_pair_passes": 1,
        "fitting_events": len(ids),
        "unique_pairs": len(x),
        "reference_Adam_risk": old_loss,
        "reference_Adam_gradient_inf": float(np.abs(old_grad).max()),
        "final_risk": final_loss,
        "final_gradient_inf": float(np.abs(gradient).max()),
        "serialized_fp32_risk": loss32,
        "serialized_fp32_gradient_inf": float(np.abs(grad32).max()),
        "stationarity_1e7_pass": bool(np.abs(gradient).max() <= 1e-7),
        "hessian_min_eigenvalue": float(eig.min()),
        "hessian_max_eigenvalue": float(eig.max()),
        "hessian_rank_at_1e10": int((eig > 1e-10).sum()),
        "evaluations": evaluations,
    }


def main(parent, phase78, output):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ):
        raise RuntimeError("Fresh guarded2CPU/noGPU diagnostic allocation required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start, cpu = time.monotonic(), time.process_time()
    from scripts.run_phase74_development import verify_bindings

    c = json.loads((parent / "successor-campaign-v1/contract.json").read_text())
    verify_bindings(c, Path(c["source_root"]))
    terminal = json.loads((phase78 / "diagnostic-terminal-v2.json").read_text())
    for b in terminal["artifacts"]:
        assert binding(b["path"]) == b
    cache = torch.load(c["cache"]["path"], map_location="cpu", weights_only=False)
    del cache["development"]
    train = {r["uid"]: r for r in cache["train"]}
    roles = json.loads(
        (phase78 / "diagnostic-v2/probe-partition-private.json").read_text()
    )
    traces = torch.load(
        phase78 / "diagnostic-v2/train-model-only.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert (
        set(train) == set(roles) == {r["uid"] for r in traces} and len(traces) == 1536
    )
    for r in traces:
        r.update(
            target=train[r["uid"]]["targets"],
            category=train[r["uid"]]["category"],
            partition=roles[r["uid"]],
        )
    old = torch.load(
        phase78 / "diagnostic-v2/probe-final.pt", map_location="cpu", weights_only=False
    )
    output.mkdir(exist_ok=False)
    probes = {}
    fits = {}
    for name, shuffled in [("pair_probe", False), ("shuffled_probe", True)]:
        probes[name], fits[name] = fit(traces, old[name], shuffled)
        print(
            name,
            json.dumps({k: v for k, v in fits[name].items() if k != "evaluations"}),
            flush=True,
        )
    torch.save(probes, output / "probe-final.pt")
    metrics = []
    pooled = defaultdict(lambda: [[], []])
    for r in traces:
        value, scores = score_record(r, probes)
        metrics.append(value)
        for name, (s, y) in scores.items():
            key = r["partition"] + "/" + name
            pooled[key][0].extend(s)
            pooled[key][1].extend(y)
    write(
        output / "event-metrics-private.json",
        [{"uid": r["uid"], **v} for r, v in zip(traces, metrics)],
    )
    previous = json.loads(
        (phase78 / "diagnostic-v2/train-event-metrics-private.json").read_text()
    )
    rng = np.random.default_rng(7911)
    contrasts = {}
    for label, other in [
        ("solver_minus_native", "proposal_conditional_same"),
        ("solver_minus_Adam", None),
        ("solver_minus_null", "shuffled_probe"),
    ]:
        bycat = []
        for cat in ("charged", "mixed"):
            v = []
            for r, old_r in zip(metrics, previous):
                if r["partition"] != "assessment" or r["category"] != cat:
                    continue
                first = r["scores"]["pair_probe"]["B_pair_auc"]["auc"]
                second = (old_r if other is None else r)["scores"][
                    "pair_probe" if other is None else other
                ]["B_pair_auc"]["auc"]
                if first is not None and second is not None:
                    v.append(first - second)
            bycat.append(np.array(v))
        draws = [
            np.concatenate([a[rng.integers(len(a), size=len(a))] for a in bycat]).mean()
            for _ in range(2000)
        ]
        contrasts[label] = {
            "events": sum(map(len, bycat)),
            "difference": float(np.concatenate(bycat).mean()),
            "interval95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    summary = {
        "stage": "training_only_probe_optimization",
        "scientific_training_jobs": 0,
        "validation_events_evaluated": 0,
        "source_binding": binding(phase78 / "diagnostic-terminal-v2.json"),
        "fits": fits,
        "aggregate": aggregate(metrics),
        "partitions": {
            p: aggregate([r for r in metrics if r["partition"] == p])
            for p in ("fit", "assessment")
        },
        "pooled_pair_auc": {k: auc(*v) for k, v in pooled.items()},
        "paired_exploratory_contrasts": contrasts,
        "compute": {
            "wall_seconds": time.monotonic() - start,
            "process_cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "encoder_forward_events": 0,
            "scientific_model_updates": 0,
            "unique_training_trace_identities": 1536,
        },
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary["compute"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("parent", "phase78", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    main(a.parent, a.phase78, a.output)
