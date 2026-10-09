"""Matched one-step current-gradient versus stored-first-moment mechanism audit."""

from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch  # noqa:E402
from scripts.prepare_phase74_development_data import binding, write  # noqa:E402
from scripts.phase76_development import guarded  # noqa:E402
from scripts.run_phase74_development import verify_bindings  # noqa:E402
from scripts.diagnose_phase80_optimizer_steps import audit, summarize_delta, interpolate  # noqa:E402
from scripts.diagnose_phase78_partition import conditional_counts  # noqa:E402
from hypertagging.training.capacity_development import (  # noqa: E402
    fresh_model,
    fresh_decoder,
    detector_features,
    membership_loss,
)  # noqa:E402
from hypertagging.models.assembly_development import assembly_relation_loss  # noqa:E402


def use_current_gradient(optimizer):
    for group in optimizer.param_groups:
        if tuple(group["betas"]) != (0.9, 0.999):
            raise ValueError("Unexpected native optimizer history")
        group["betas"] = (0.0, 0.999)


def text_digest(path):
    with gzip.open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def add_conditional(result, path, rows):
    with gzip.open(path, "rt") as f:
        seen = 0
        for event, row, line in zip(result["events"], rows, f):
            seen += 1
            record = json.loads(line)
            assert event["uid"] == row["uid"] == record["uid"]
            for name, key in [
                ("proposal", "proposal_logits"),
                ("refinement", "logits"),
            ]:
                event["stages"][name]["conditional"] = conditional_counts(
                    torch.tensor(record["model"][key][0]), row["targets"]
                )
        if seen != len(rows) or f.readline():
            raise ValueError("Trace row coverage mismatch")
    return result


def delta_summary(base, current, excluded):
    result = summarize_delta(base, current, excluded)
    for stage in ["proposal", "refinement"]:
        paired = [
            (a, b)
            for a, b in zip(base["events"], current["events"])
            if a["uid"] not in excluded
        ]
        for key in [
            "B_nodes",
            "conditional_optimal_correct",
            "majority_slot_null_correct",
        ]:
            result["stages"][stage].setdefault("conditional_baseline", {})[key] = sum(
                a["stages"][stage]["conditional"][key] for a, b in paired
            )
            result["stages"][stage].setdefault("conditional_change", {})[key] = sum(
                b["stages"][stage]["conditional"][key]
                - a["stages"][stage]["conditional"][key]
                for a, b in paired
            )
    return result


def qualifies(result):
    d = result["non_minibatch_audit"]
    stages = d["stages"]
    return (
        d["risk_change"]["member"] < 0
        and d["risk_change"]["total"] < 0
        and all(
            stages[s]["change"]["B_correct"] > 0
            and stages[s]["change"]["background_to_B"] <= 0
            and stages[s]["change"]["native_raw_exact"] >= 0
            and stages[s]["conditional_change"]["conditional_optimal_correct"] >= 0
            for s in stages
        )
        and any(
            stages[s]["conditional_change"]["conditional_optimal_correct"] > 0
            for s in stages
        )
        and result["accepted_continuum_change"] <= 0
    )


def validate_contract(c):
    if (
        c["stage"] != "training_only_history_diagnostic"
        or c["batches"] != 12
        or c["batch_size"] != 8
        or c["audit_events"] != 96
        or c["gradient_presentations"] != 96
        or c["optimizer_steps"] != 24
    ):
        raise ValueError("Diagnostic finite budget changed")
    if (
        c["resources"]
        != {"cpus": 2, "memory_gib": 16, "hours": 1, "gpus": 0, "requeue": False}
        or c["validation_events"]
        or c["automatic_successor"]
    ):
        raise ValueError("Resources or authority changed")


def main(phase81, phase80, output):
    guarded()
    c = json.loads((output.parent / "diagnostic-contract.json").read_text())
    validate_contract(c)
    if (
        str(ROOT) != c["source_root"]
        or str(output.resolve()) != c["output_root"]
        or str(phase81.resolve()) != c["phase81"]
        or str(phase80.resolve()) != c["phase80"]
    ):
        raise ValueError("Bound roots changed")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Source commit changed")
    verify_bindings(c, ROOT)
    start = time.monotonic()
    cpu = time.process_time()
    cache = torch.load(c["cache"]["path"], map_location="cpu", weights_only=False)
    if set(cache) != {"train", "runtime_normalizer"}:
        raise ValueError("Validation in diagnostic cache")
    train = cache["train"]
    rowmap = {r["uid"]: r for r in train}
    old = json.loads((phase80 / "diagnostic-v1/summary.json").read_text())
    cp = old["checkpoint"]
    assert binding(cp["path"]) == cp
    ck = torch.load(cp["path"], map_location="cpu", weights_only=False)
    selection = json.loads(
        (phase80 / "diagnostic-v1/selection-private.json").read_text()
    )
    rows = [rowmap[u] for u in selection["audit"]]
    assert len(rows) == 96
    rng = random.Random()
    rng.setstate(ck["sampling_rng_state"])
    batches = [[train[rng.randrange(len(train))] for _ in range(8)] for _ in range(12)]
    assert [[r["uid"] for r in b] for b in batches] == selection["batches"]
    output.mkdir(exist_ok=False)
    write(output / "selection-private.json", selection)
    model = fresh_model(128, cache["runtime_normalizer"])
    decoder = fresh_decoder()
    model.load_state_dict(ck["model_state_dict"])
    decoder.load_state_dict(ck["decoder_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    parameters = list(model.encoder.parameters()) + list(decoder.parameters())
    initial = [p.detach().clone() for p in parameters]
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ck["settings"]["encoder_lr"]},
            {"params": decoder.parameters(), "lr": ck["settings"]["head_lr"]},
        ],
        weight_decay=1e-4,
    )
    path = output / "baseline-model-only.jsonl.gz"
    baseline = audit(model, decoder, rows, path)
    if text_digest(path) != text_digest(
        phase80 / "diagnostic-v1/baseline-model-only.jsonl.gz"
    ):
        raise ValueError("Native baseline trace changed")
    add_conditional(baseline, path, rows)
    write(output / "baseline.json", baseline)
    results = []
    pair_proxy = 0
    for index, batch in enumerate(batches):
        model.load_state_dict(ck["model_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])
        opt.load_state_dict(copy.deepcopy(ck["optimizer_state_dict"]))
        model.eval()
        decoder.train()
        opt.zero_grad(set_to_none=True)
        for row in batch:
            h, _, _ = detector_features(model, row["detector"])
            pred = decoder(h, row["sources"])
            member = membership_loss(pred, row["targets"])
            relation, _ = assembly_relation_loss(pred["states"], row["supervision"])
            ((member + relation) / 8).backward()
            pair_proxy += len(row["targets"]) ** 2
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        grads = [None if p.grad is None else p.grad.clone() for p in parameters]
        opt.step()
        native_delta = [p.detach().clone() - v for p, v in zip(parameters, initial)]
        native_norm = float(torch.sqrt(sum(v.square().sum() for v in native_delta)))
        reference = old["results"][index]
        if (
            abs(float(norm) - reference["gradient_norm_before_clip"]) > 1e-6
            or abs(native_norm - reference["parameter_delta_norm"]) > 1e-7
        ):
            raise ValueError("Native update reference changed")
        second = {
            i: (opt.state[p]["exp_avg_sq"].clone(), opt.state[p]["step"].clone())
            for i, p in enumerate(parameters)
            if "exp_avg_sq" in opt.state[p]
        }
        # Match Phase80 FP32 delta reconstruction in both conditions.
        interpolate(parameters, initial, native_delta, 1.0)
        if index == 0:
            p = output / "native-replay-model-only.jsonl.gz"
            native = audit(model, decoder, rows, p)
            if text_digest(p) != text_digest(
                phase80 / "diagnostic-v1/batch-00-scale-1.0-model-only.jsonl.gz"
            ):
                raise ValueError("Native updated trace changed")
            write(output / "native-replay.json", native)
        model.load_state_dict(ck["model_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])
        opt.load_state_dict(copy.deepcopy(ck["optimizer_state_dict"]))
        for p, g in zip(parameters, grads):
            p.grad = None if g is None else g.clone()
        use_current_gradient(opt)
        opt.step()
        for i, p in enumerate(parameters):
            if i in second and (
                not torch.equal(opt.state[p]["exp_avg_sq"], second[i][0])
                or not torch.equal(opt.state[p]["step"], second[i][1])
            ):
                raise ValueError("Second moments/counters diverged")
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite state")
        current_delta = [p.detach().clone() - v for p, v in zip(parameters, initial)]
        current_norm = float(torch.sqrt(sum(v.square().sum() for v in current_delta)))
        cosine = float(
            sum(a.mul(b).sum() for a, b in zip(native_delta, current_delta))
            / (native_norm * current_norm)
        )
        for key, v in model.state_dict().items():
            if not key.startswith("encoder.") and not torch.equal(
                v, ck["model_state_dict"][key]
            ):
                raise ValueError("Frozen PID/model state changed")
        interpolate(parameters, initial, current_delta, 1.0)
        p = output / f"batch-{index:02d}-current-model-only.jsonl.gz"
        current = add_conditional(audit(model, decoder, rows, p), p, rows)
        write(output / f"batch-{index:02d}-current.json", current)
        excluded = {r["uid"] for r in batch}
        continuum_change = sum(
            b["counts"]["continuum_accepted_events"]
            - a["counts"]["continuum_accepted_events"]
            for a, b in zip(baseline["native"]["events"], current["native"]["events"])
            if a["uid"] not in excluded
        )
        record = {
            "batch_index": index,
            "gradient_norm_before_clip": float(norm),
            "native_parameter_delta_norm": native_norm,
            "current_parameter_delta_norm": current_norm,
            "delta_cosine": cosine,
            "second_moments_and_counters_equal": True,
            "all_audit": delta_summary(baseline, current, set()),
            "non_minibatch_audit": delta_summary(
                baseline, current, {r["uid"] for r in batch}
            ),
            "accepted_continuum_change": continuum_change,
            "native_counts": current["native"]["counts"],
            "phase80_native_reference": reference["conditions"]["1.0"],
        }
        record["qualifying_structural_step"] = qualifies(record)
        results.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        "stage": c["stage"],
        "checkpoint": cp,
        "source_sha": c["source_sha"],
        "baseline_counts": baseline["native"]["counts"],
        "results": results,
        "qualifying_steps": sum(r["qualifying_structural_step"] for r in results),
        "mechanism_support_gate": sum(r["qualifying_structural_step"] for r in results)
        >= 9,
        "interpretation": "Single-step counterfactual fits reset to same checkpoint; shared/overlapping training audits correlated. No independent confirmation or selected model. Native first moment0.9 versus current-gradient beta1=0, same second moment/LR/clipping.",
        "compute": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "optimizer_steps": 24,
            "gradient_event_presentations": 96,
            "optimizer_condition_event_presentations": 192,
            "gradient_encoder_passes": 192,
            "evaluation_encoder_passes": 2688,
            "unique_optimizer_identities": len({r["uid"] for b in batches for r in b}),
            "new_audit_event_views": 1344,
            "unique_audit_events": 96,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "training_node_squared_proxy": pair_proxy,
            "evaluation_node_squared_proxy": 14
            * sum(len(r["targets"]) ** 2 for r in rows),
            "scientific_campaign_jobs": 0,
            "validation_events": 0,
        },
    }
    write(output / "summary.json", summary)
    write(
        output / "terminal.json",
        {
            "status": "COMPLETED",
            "bindings": [binding(p) for p in sorted(output.iterdir()) if p.is_file()],
        },
    )
    print(
        "MECHANISM_SUPPORT",
        summary["qualifying_steps"],
        summary["mechanism_support_gate"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["phase81", "phase80", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    main(a.phase81, a.phase80, a.output)
