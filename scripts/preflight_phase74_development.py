"""Real-data gradients, immutable fresh initializations and bounded CPU timing."""

from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, sha, write  # noqa: E402
from scripts.run_phase74_development import save, fit, evaluate  # noqa: E402


def preflight(output, destination):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Guarded two-CPU allocation required")
    import torch
    from hypertagging.training.capacity_development import (
        fresh_model,
        fresh_decoder,
        baseline_loss,
        assembly_loss,
        detector_features,
        membership_loss,
        ARCHITECTURE,
    )
    from hypertagging.models.assembly_development import assembly_relation_loss

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(exist_ok=False)
    admission = json.loads((output / "cache-admission.json").read_text())
    cache_binding = admission["cache"]
    if sha(cache_binding["path"]) != cache_binding["sha256"]:
        raise ValueError("Cache changed")
    cache = torch.load(cache_binding["path"], map_location="cpu", weights_only=False)
    # Predeclared first two original-train identities/category, no held-out fitting.
    rows = []
    for cat in ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"):
        rows.extend([r for r in cache["train"] if r["category"] == cat][:2])
    results = {}
    initial_predictions = {}
    initializations = {}
    for width in (128, 256):
        model = fresh_model(width, cache["runtime_normalizer"])
        decoder = fresh_decoder()
        initializations[str(width)] = destination / f"initial-{width}.pt"
        save(
            initializations[str(width)],
            {
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "width": width,
                "architecture": ARCHITECTURE,
                "fresh_history_updates": 0,
                "cache": cache_binding,
            },
        )
        for objective in ("existing", "assembly"):
            start = time.monotonic()
            m = copy.deepcopy(model)
            d = copy.deepcopy(decoder)
            with torch.no_grad():
                initial_h, _, _ = detector_features(m, rows[0]["detector"])
                initial_logits = d(initial_h, rows[0]["sources"])["logits"].clone()
            if width in initial_predictions:
                torch.testing.assert_close(
                    initial_logits, initial_predictions[width], rtol=0, atol=0
                )
            else:
                initial_predictions[width] = initial_logits
            params = list(m.parameters()) + list(d.parameters())
            optimizer = torch.optim.AdamW(params, lr=0.0003)
            phase_results = []
            for step in (0, 250, 500, 800):
                optimizer.zero_grad(set_to_none=True)
                total = 0
                for row in rows[:4]:
                    baseline, parts = baseline_loss(m, row["full"], step)
                    if objective == "assembly":
                        extra, _, _ = assembly_loss(m, d, row)
                        loss = baseline + extra
                    else:
                        loss = baseline
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite native phase loss")
                    (loss / 4).backward()
                    total += float(loss.detach()) / 4
                norm = torch.nn.utils.clip_grad_norm_(
                    params, 5.0, error_if_nonfinite=True
                )
                if not any(
                    p.grad is not None and bool(p.grad.ne(0).any())
                    for p in m.encoder.parameters()
                ):
                    raise RuntimeError("No real encoder gradient")
                optimizer.step()
                phase_results.append(
                    {
                        "phase_step": step,
                        "mean_loss": total,
                        "gradient_norm": float(norm),
                    }
                )
            encoder_grads = {}
            relation_counts = [0, 0, 0]
            component_norms = {"membership": 0.0, "within_b_relation": 0.0}
            for row in rows:
                m.zero_grad(set_to_none=True)
                d.zero_grad(set_to_none=True)
                _, components, counts = assembly_loss(m, d, row)
                for i, n in enumerate(counts):
                    relation_counts[i] += n
                for name, loss in components.items():
                    grads = torch.autograd.grad(
                        loss,
                        tuple(m.encoder.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    value = (
                        sum(
                            float(g.detach().float().square().sum())
                            for g in grads
                            if g is not None
                        )
                        ** 0.5
                    )
                    component_norms[name] += value
            if (
                relation_counts[0] == 0
                or relation_counts[1] == 0
                or relation_counts[2] == 0
                or min(component_norms.values()) <= 0
            ):
                raise RuntimeError(
                    "Missing real within-B parent/sibling/other support or objective gradient: "
                    + str((relation_counts, component_norms))
                )
            # Shared downstream: reset head, train both encoder and common decoder.
            d = fresh_decoder()
            for parameter in m.parameters():
                parameter.requires_grad_(False)
            for parameter in m.encoder.parameters():
                parameter.requires_grad_(True)
            pid_before = {k: v.clone() for k, v in m.leaf_pid_head.state_dict().items()}
            m.zero_grad(set_to_none=True)
            row = rows[0]
            h, _, _ = detector_features(m, row["detector"])
            result = d(h, row["sources"])
            downstream = membership_loss(result, row["targets"])
            relation, _ = assembly_relation_loss(result["states"], row["supervision"])
            (downstream + relation).backward()
            for name, parameters in [
                ("encoder", m.encoder.parameters()),
                ("membership_conditioner", d.condition.parameters()),
                ("proposal", d.proposal.parameters()),
                ("relations", d.relations.parameters()),
            ]:
                gradients = [p.grad for p in parameters if p.grad is not None]
                encoder_grads[name] = (
                    sum(float(g.float().square().sum()) for g in gradients) ** 0.5
                )
                if (
                    not gradients
                    or not all(torch.isfinite(g).all() for g in gradients)
                    or encoder_grads[name] <= 0
                ):
                    raise RuntimeError("Downstream branch gradient failure: " + name)
            downstream_optimizer = torch.optim.AdamW(
                list(m.encoder.parameters()) + list(d.parameters()), lr=0.00005
            )
            torch.nn.utils.clip_grad_norm_(
                list(m.encoder.parameters()) + list(d.parameters()),
                5.0,
                error_if_nonfinite=True,
            )
            downstream_optimizer.step()
            if any(
                not torch.equal(v, pid_before[k])
                for k, v in m.leaf_pid_head.state_dict().items()
            ):
                raise RuntimeError("Downstream frozen PID weights changed")
            benchmark = {
                "pretraining_max_event_seconds": 0.0,
                "downstream_max_event_seconds": 0.0,
            }
            worst_rows = [
                max(cache["train"], key=lambda r: r["full"]["node_mask"].shape[1]),
                max(cache["train"], key=lambda r: len(r["targets"])),
            ]
            for worst in worst_rows:
                for parameter in m.parameters():
                    parameter.requires_grad_(True)
                m.zero_grad(set_to_none=True)
                d.zero_grad(set_to_none=True)
                tick = time.monotonic()
                native, _ = baseline_loss(m, worst["full"], 800)
                if objective == "assembly":
                    extra, _, _ = assembly_loss(m, d, worst)
                    native = native + extra
                else:
                    with torch.no_grad():
                        assembly_loss(m, d, worst)
                native.backward()
                benchmark["pretraining_max_event_seconds"] = max(
                    benchmark["pretraining_max_event_seconds"], time.monotonic() - tick
                )
                for parameter in m.parameters():
                    parameter.requires_grad_(False)
                for parameter in m.encoder.parameters():
                    parameter.requires_grad_(True)
                m.zero_grad(set_to_none=True)
                d.zero_grad(set_to_none=True)
                tick = time.monotonic()
                value, _, _ = assembly_loss(m, d, worst)
                value.backward()
                benchmark["downstream_max_event_seconds"] = max(
                    benchmark["downstream_max_event_seconds"], time.monotonic() - tick
                )
            benchmark["conservative_compute_seconds"] = 1.5 * (
                8000 * benchmark["pretraining_max_event_seconds"]
                + 22160 * benchmark["downstream_max_event_seconds"]
            )
            if benchmark["conservative_compute_seconds"] > 8 * 3600:
                raise RuntimeError(
                    "Measured CPU bound exceeds requested8hours: " + str(benchmark)
                )
            if not all(torch.isfinite(p).all() for p in params):
                raise FloatingPointError("Nonfinite state")
            # Exercise production fit/checkpoint/evaluation paths on train-only rows.
            from collections import Counter

            runtime_dir = destination / f"runtime-{width}-{objective}"
            runtime_dir.mkdir()
            runtime_settings = {
                "pretraining_updates": 2,
                "tiny_updates": 2,
                "downstream_updates": 2,
                "batch_size": 2,
                "seed": 202610081,
                "pretraining_lr": 0.0003,
                "encoder_lr": 0.00005,
                "head_lr": 0.001,
            }
            runtime_model = copy.deepcopy(model)
            runtime_decoder = fresh_decoder()
            runtime_histories = {}
            for stage in ("pretraining", "tiny", "downstream"):
                runtime_histories[stage] = fit(
                    runtime_model,
                    runtime_decoder,
                    rows[:4],
                    stage=stage,
                    objective=objective,
                    settings=runtime_settings,
                    output=runtime_dir,
                    compute=Counter(),
                    cache_binding=cache_binding,
                    source_sha="preflight-only",
                )
                checkpoint = torch.load(
                    runtime_dir / (stage + "-final.pt"), weights_only=False
                )
                if checkpoint["step"] != 2:
                    raise RuntimeError("Runtime checkpoint step mismatch")
            runtime_evaluation = evaluate(runtime_model, runtime_decoder, rows[:4])
            write(runtime_dir / "evaluation.json", runtime_evaluation)
            results[f"{width}-{objective}"] = {
                "status": "PASS",
                "runtime_fit_checkpoint_evaluation": runtime_histories,
                "bounded_cpu_timing": benchmark,
                "pid_weights_unchanged_downstream": True,
                "baseline_phases": phase_results,
                "encoder_and_head_gradient_norms": encoder_grads,
                "assembly_component_encoder_gradient_norms": component_norms,
                "within_b_support": relation_counts,
                "elapsed_seconds": time.monotonic() - start,
                "model_parameters": sum(p.numel() for p in m.parameters()),
                "encoder_parameters": sum(p.numel() for p in m.encoder.parameters()),
                "decoder_parameters": sum(p.numel() for p in d.parameters()),
            }
            print(width, objective, results[f"{width}-{objective}"], flush=True)
    write(
        destination / "admission.json",
        {
            "status": "PASS",
            "arms": results,
            "initializations": {k: binding(v) for k, v in initializations.items()},
            "cache": cache_binding,
            "training_smoke_event_count": len(rows),
            "heldout_smoke_events": 0,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "architecture": ARCHITECTURE,
            "preflight_source_hashes": {
                str(p.relative_to(ROOT)): sha(p)
                for folder in ("src", "scripts")
                for p in (ROOT / folder).rglob("*.py")
            },
            "scope": "real_training_data_gradients_and_native_objective_phases_not_physics_success",
            "initial_predictions_shared_within_width": True,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    preflight(a.output.resolve(), a.destination.resolve())
