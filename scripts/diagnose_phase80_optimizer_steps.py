"""Prespecified one-step native optimizer counterfactuals; training-role audit only."""

from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch  # noqa: E402
from scripts.prepare_phase74_development_data import binding, write  # noqa: E402
from scripts.run_phase74_development import evaluate, verify_bindings  # noqa: E402
from scripts.diagnose_phase77_proposals import describe  # noqa: E402
from hypertagging.training.capacity_development import (  # noqa: E402
    fresh_model,
    fresh_decoder,
    detector_features,
    membership_loss,
)
from hypertagging.models.assembly_development import assembly_relation_loss  # noqa: E402


def interpolate(parameters, initial, delta, scale):
    if scale not in (1.0, 0.5):
        raise ValueError("Only prespecified native/half step")
    with torch.no_grad():
        for p, a, d in zip(parameters, initial, delta):
            p.copy_(a + scale * d)


def detached_result(result):
    return {k: v.detach().clone() for k, v in result.items() if k != "states"} | {
        "states": [
            {
                "groups": s["groups"],
                "pairs": s["pairs"],
                "logits": s["logits"].detach().clone(),
            }
            for s in result["states"]
        ]
    }


def trace_json(result):
    return {k: v.tolist() for k, v in result.items() if k != "states"} | {
        "states": [
            {
                "groups": [sorted(g) for g in s["groups"]],
                "pairs": s["pairs"],
                "logits": s["logits"].tolist(),
            }
            for s in result["states"]
        ]
    }


def audit(model, decoder, rows, path):
    outputs = []
    with gzip.open(path, "xt") as stream:

        def capture(module, args, result):
            detached = detached_result(result)
            stream.write(
                json.dumps(
                    {"uid": rows[len(outputs)]["uid"], "model": trace_json(detached)}
                )
                + "\n"
            )
            outputs.append(detached)

        handle = decoder.register_forward_hook(capture)
        try:
            native = evaluate(model, decoder, rows)
        finally:
            handle.remove()
    assert len(outputs) == len(rows)
    events = []
    for row, result in zip(rows, outputs):
        with torch.no_grad():
            member = membership_loss(result, row["targets"])
            relation, _ = assembly_relation_loss(result["states"], row["supervision"])
        stages = {}
        for name, lk, ok in [
            ("proposal", "proposal_logits", "proposal_objects"),
            ("refinement", "logits", "objects"),
        ]:
            counts, _, sizes, _, _ = describe(
                result[lk][0], result[ok][0], row["targets"]
            )
            stages[name] = {
                "counts": dict(counts),
                "sizes": {k: dict(v) for k, v in sizes.items()},
            }
        events.append(
            {
                "uid": row["uid"],
                "category": row["category"],
                "member": float(member),
                "relation": float(relation),
                "total": float(member + relation),
                "stages": stages,
            }
        )
    return {
        "native": native,
        "events": events,
        "mean_risk": {
            k: sum(r[k] for r in events) / len(events)
            for k in ("member", "relation", "total")
        },
        "trace": binding(path),
    }


def summarize_delta(reference, current, excluded):
    pairs = [
        (a, b)
        for a, b in zip(reference["events"], current["events"])
        if a["uid"] not in excluded
    ]
    assert all(a["uid"] == b["uid"] for a, b in pairs)
    values = {}
    for stage in ("proposal", "refinement"):
        c = Counter()
        base = Counter()
        for a, b in pairs:
            aa = a["stages"][stage]["counts"]
            bb = b["stages"][stage]["counts"]
            for k in (
                "B_nodes",
                "B_correct",
                "B_to_unassigned",
                "B_to_other_B",
                "background_nodes",
                "background_to_B",
                "native_raw_exact",
                "B_events",
            ):
                base[k] += aa[k]
                c[k] += bb[k] - aa[k]
        values[stage] = {"baseline": dict(base), "change": dict(c)}
    return {
        "events": len(pairs),
        "risk_change": {
            k: sum(b[k] - a[k] for a, b in pairs) / len(pairs)
            for k in ("member", "relation", "total")
        },
        "stages": values,
    }


def main(parent, output):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ):
        raise RuntimeError("Fresh2CPU/noGPU guarded allocation required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start, cpu = time.monotonic(), time.process_time()
    contract = json.loads((parent / "successor-campaign-v1/contract.json").read_text())
    verify_bindings(contract, Path(contract["source_root"]))
    cache = torch.load(
        contract["cache"]["path"], map_location="cpu", weights_only=False
    )
    del cache["development"]
    train = cache["train"]
    assert len(train) == 1536 and len({r["uid"] for r in train}) == 1536
    terminal = json.loads(
        (
            parent / "successor-campaign-v1/runs/partial_context_on/terminal.json"
        ).read_text()
    )
    cp = next(
        b
        for b in terminal["checkpoints"]
        if Path(b["path"]).name == "downstream-final.pt"
    )
    assert binding(cp["path"]) == cp
    ck = torch.load(cp["path"], map_location="cpu", weights_only=False)
    rows = sum(
        (
            sorted(
                [r for r in train if r["category"] == cat],
                key=lambda r: hashlib.sha256(
                    ("phase77-gradient:" + r["uid"]).encode()
                ).hexdigest(),
            )[:16]
            for cat in sorted({r["category"] for r in train})
        ),
        [],
    )
    assert len(rows) == 96 and len({r["uid"] for r in rows}) == 96
    rng = random.Random()
    rng.setstate(ck["sampling_rng_state"])
    batches = [[train[rng.randrange(len(train))] for _ in range(8)] for _ in range(12)]
    output.mkdir(exist_ok=False)
    write(
        output / "selection-private.json",
        {
            "audit": [r["uid"] for r in rows],
            "batches": [[r["uid"] for r in b] for b in batches],
            "checkpoint": cp,
        },
    )
    model = fresh_model(128, cache["runtime_normalizer"])
    decoder = fresh_decoder()
    model.load_state_dict(ck["model_state_dict"])
    decoder.load_state_dict(ck["decoder_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    parameters = list(model.encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ck["settings"]["encoder_lr"]},
            {"params": decoder.parameters(), "lr": ck["settings"]["head_lr"]},
        ],
        weight_decay=1e-4,
    )
    initial = [p.detach().clone() for p in parameters]
    baseline = audit(model, decoder, rows, output / "baseline-model-only.jsonl.gz")
    write(output / "baseline.json", baseline)
    results = []
    train_pair_proxy = 0
    for step, batch in enumerate(batches):
        model.load_state_dict(ck["model_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])
        optimizer.load_state_dict(copy.deepcopy(ck["optimizer_state_dict"]))
        model.eval()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        for row in batch:
            h, _, _ = detector_features(model, row["detector"])
            rr = decoder(h, row["sources"])
            member = membership_loss(rr, row["targets"])
            relation, _ = assembly_relation_loss(rr["states"], row["supervision"])
            ((member + relation) / 8).backward()
            train_pair_proxy += len(row["targets"]) ** 2
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        delta = [p.detach().clone() - a for p, a in zip(parameters, initial)]
        norm_delta = float(torch.sqrt(sum(d.square().sum() for d in delta)))
        if not norm_delta > 0:
            raise RuntimeError("No actual optimizer update")
        for key, value in model.state_dict().items():
            if not key.startswith("encoder.") and not torch.equal(
                value, ck["model_state_dict"][key]
            ):
                raise RuntimeError("Frozen PID/model state changed")
        record = {
            "batch_index": step,
            "gradient_norm_before_clip": float(norm),
            "parameter_delta_norm": norm_delta,
            "minibatch_audit_overlap": len(
                set(r["uid"] for r in batch) & set(r["uid"] for r in rows)
            ),
            "conditions": {},
        }
        for scale in (1.0, 0.5):
            interpolate(parameters, initial, delta, scale)
            current = audit(
                model,
                decoder,
                rows,
                output / f"batch-{step:02d}-scale-{scale}-model-only.jsonl.gz",
            )
            write(output / f"batch-{step:02d}-scale-{scale}.json", current)
            record["conditions"][str(scale)] = {
                "mean_risk": current["mean_risk"],
                "all_audit": summarize_delta(baseline, current, set()),
                "non_minibatch_audit": summarize_delta(
                    baseline, current, {r["uid"] for r in batch}
                ),
                "native_counts": current["native"]["counts"],
            }
        results.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        "stage": "training_only_optimizer_counterfactual",
        "checkpoint": cp,
        "baseline_risk": baseline["mean_risk"],
        "baseline_native_counts": baseline["native"]["counts"],
        "results": results,
        "compute": {
            "wall_seconds": time.monotonic() - start,
            "process_cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "independent_optimizer_steps": 12,
            "optimizer_event_presentations": 96,
            "unique_optimizer_identities": len({r["uid"] for b in batches for r in b}),
            "audit_event_views": 2400,
            "unique_audit_identities": 96,
            "training_node_squared_proxy": train_pair_proxy,
            "evaluation_node_squared_proxy": 25
            * sum(len(r["targets"]) ** 2 for r in rows),
            "scientific_campaign_jobs": 0,
            "validation_events": 0,
        },
        "interpretation": "Counterfactual one-step fits on training data, reset to same checkpoint. Repeated audit/gradient states correlated; no independent confirmation or selected model.",
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary["compute"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.parent, a.output)
