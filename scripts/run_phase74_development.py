"""Exactly one admitted Phase74 fresh-history factorial arm; CPU development only."""

from __future__ import annotations
import argparse
from collections import Counter
import copy
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
from scripts.prepare_phase74_development_data import binding, sha, write  # noqa: E402


def save(path, value):
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def validate_contract(c):
    from hypertagging.training.capacity_development import ARMS, ARCHITECTURE

    expected = {
        "pretraining_updates": 1000,
        "downstream_updates": 1500,
        "tiny_updates": 1000,
        "batch_size": 8,
        "seed": 202610081,
        "checkpoint": "fixed_final",
        "threshold": 0.5,
        "pretraining_lr": 0.0003,
        "head_lr": 0.001,
        "encoder_lr": 0.00005,
    }
    if (
        c["arms"] != list(ARMS)
        or c["architecture"] != ARCHITECTURE
        or c["settings"] != expected
    ):
        raise ValueError("Wrong arms, geometry, history or finite budgets")
    if (
        c["stage"] != "development"
        or c["sealed_test_access"]
        or c["automatic_successor"]
    ):
        raise ValueError("Authority exceeded")
    if (
        c["history"] != "fresh_both_widths_shared_initialization_within_width"
        or c["head_interface"]
        != "lossless_zero_pad_to_common256_predicted_group_conditioning_reset_after_pretrain"
    ):
        raise ValueError("Incompatible pretraining history/head interface")
    if c["resources"] != {
        "cpus": 2,
        "memory_gib": 32,
        "hours": 8,
        "gpus": 0,
        "max_jobs": 4,
        "requeue": False,
    }:
        raise ValueError("Resource contract changed")
    if c["training_count"] != 1536 or c["heldout_count"] != 600:
        raise ValueError("Cohort contract changed")


def verify_bindings(c, root):
    for relative, digest in c["source_hashes"].items():
        if sha(root / relative) != digest:
            raise ValueError("Frozen source changed: " + relative)
    for item in c["bindings"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Immutable input changed: " + item["path"])


def evaluate(model, decoder, rows):
    import torch
    from hypertagging.training.capacity_development import detector_features
    from hypertagging.models.assembly_development import relation_targets

    totals = Counter()
    event_rows = []
    model.eval()
    decoder.eval()
    with torch.inference_mode():
        for row in rows:
            counts = Counter(
                {
                    "events": 1,
                    "raw_exact_memberships": 0,
                    "accepted_exact_memberships": 0,
                    "nominal_b_trials": 0,
                    "unavailable_membership_trials": 0,
                    "accepted_source_conflicts": 0,
                    "continuum_events": 0,
                    "continuum_accepted_events": 0,
                }
            )
            h, _, _ = detector_features(model, row["detector"])
            result = decoder(h, row["sources"])
            probabilities = result["logits"][0].softmax(-1)
            assignments = probabilities.argmax(-1)
            raw = [
                set((assignments == slot).nonzero().flatten().tolist())
                for slot in (1, 2)
            ]
            accepted = []
            used = set()
            for slot in sorted(range(2), key=lambda i: -float(result["objects"][0, i])):
                if float(result["objects"][0, slot].sigmoid()) < 0.5:
                    continue
                members = []
                local = set()
                for i in sorted(
                    raw[slot], key=lambda j: -float(probabilities[j, slot + 1])
                ):
                    sources = set(row["sources"][i].nonzero().flatten().tolist())
                    if not sources or sources & (local | used):
                        counts["rejected_source_nodes"] += 1
                        continue
                    members.append(i)
                    local.update(sources)
                charge = float(row["charge"][members].sum()) if members else 0.0
                if len(members) < 2:
                    counts["rejected_cardinality_groups"] += 1
                elif min(abs(charge - q) for q in (-1, 0, 1)) > 1e-5:
                    counts["rejected_charge_groups"] += 1
                else:
                    accepted.append(set(members))
                    used.update(local)
            seen = set()
            for group in accepted:
                for i in group:
                    source = set(row["sources"][i].nonzero().flatten().tolist())
                    counts["accepted_source_conflicts"] += len(source & seen)
                    seen.update(source)
            if counts["accepted_source_conflicts"]:
                raise RuntimeError("Accepted source conflict")
            counts["accepted_groups"] = len(accepted)
            if row["category"] in ("charged", "mixed"):
                counts["nominal_b_trials"] = 2
                if bool((row["targets"] < 0).all()):
                    counts["unavailable_membership_trials"] = 2
                else:
                    for slot in (1, 2):
                        target = set(
                            (row["targets"] == slot).nonzero().flatten().tolist()
                        )
                        counts["raw_exact_memberships"] += int(target in raw)
                        counts["accepted_exact_memberships"] += int(target in accepted)
            else:
                counts["continuum_events"] = 1
                counts["continuum_accepted_events"] = int(bool(accepted))
            for state_index, state in enumerate(result["states"]):
                labels = relation_targets(
                    state["groups"], state["pairs"], row["supervision"]
                )
                predictions = state["logits"].argmax(-1).tolist() if labels else []
                for label, prediction in zip(labels, predictions):
                    if label >= 0:
                        prefix = (
                            "detector_relation"
                            if state_index == 0
                            else "generated_state_relation"
                        )
                        counts[prefix + "_trials"] += 1
                        counts[prefix + "_correct"] += int(label == prediction)
                        counts[prefix + f"_class{label}_trials"] += 1
                        counts[prefix + f"_class{label}_correct"] += int(
                            label == prediction
                        )
            # Explicitly diagnostic source-set survival, not physical tree reachability.
            deep = {
                group
                for i, group in enumerate(row["supervision"]["node_sets"])
                if group and int(row["full"]["level_ids"][0, i]) >= 2
            }
            proposed = {
                state["groups"][i] | state["groups"][j]
                for state in result["states"]
                for i, j in state["pairs"]
            }
            retained = set(result["states"][-1]["groups"])
            counts["latent_deep_source_sets_trials"] = len(deep)
            counts["latent_deep_source_sets_proposed"] = len(deep & proposed)
            counts["latent_deep_source_sets_retained"] = len(deep & retained)
            totals.update(counts)
            event_rows.append(
                {
                    "uid": row["uid"],
                    "category": row["category"],
                    "counts": dict(counts),
                    "raw_groups": [sorted(g) for g in raw],
                    "accepted_groups": [sorted(g) for g in accepted],
                }
            )
    return {
        "counts": dict(totals),
        "events": event_rows,
        "metric_definition": "unordered_flat_FSP_membership_plus_latent_source_set_proposal_diagnostics_not_physical_B_hierarchy",
        "inclusive_top1_membership": {
            "numerator": totals["accepted_exact_memberships"],
            "denominator": totals["nominal_b_trials"],
            "unavailable": totals["unavailable_membership_trials"],
        },
        "unavailable": {
            key: {"numerator": None, "denominator": None, "reason": reason}
            for key, reason in {
                "inclusive_B_beam_pool": "No competing physical beam hypotheses are generated",
                "exact_B_top1": "No intermediate particle types or physical tree are decoded",
                "exact_B_pool": "No physical hierarchy/beam pool exists",
                "physical_deep_proposal_survival": "Only latent membership source-set merges; report those separately",
                "daughter_sum_p4_closure": "No persistent physical mother is constructed; no physical closure claim",
            }.items()
        },
    }


def fit(
    model,
    decoder,
    rows,
    *,
    stage,
    objective,
    settings,
    output,
    compute,
    cache_binding,
    source_sha,
):
    import torch
    from hypertagging.training.capacity_development import (
        baseline_loss,
        assembly_loss,
        detector_features,
        membership_loss,
        ARCHITECTURE,
    )
    from hypertagging.models.assembly_development import assembly_relation_loss

    updates = settings[stage + "_updates"]
    model.eval()
    decoder.train()
    for p in model.parameters():
        p.requires_grad_(stage == "pretraining")
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    if stage == "pretraining":
        parameters = list(model.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=settings["pretraining_lr"], weight_decay=1e-4
        )
    else:
        parameters = list(model.encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": settings["encoder_lr"]},
                {"params": decoder.parameters(), "lr": settings["head_lr"]},
            ],
            weight_decay=1e-4,
        )
    rng = random.Random(settings["seed"])
    sequence = hashlib.sha256()
    relation_support = [0, 0, 0]
    last_loss = None
    started = time.monotonic()
    with (output / (stage + "-metrics.jsonl")).open("x") as log:
        for step in range(updates):
            optimizer.zero_grad(set_to_none=True)
            summed = 0.0
            components = Counter()
            chosen = [
                rows[rng.randrange(len(rows))] for _ in range(settings["batch_size"])
            ]
            for row in chosen:
                sequence.update((row["uid"] + "\n").encode())
                if stage == "pretraining":
                    base, parts = baseline_loss(model, row["full"], step, updates)
                    if objective == "assembly":
                        auxiliary, extra, support = assembly_loss(model, decoder, row)
                        loss = base + auxiliary
                    else:
                        # Same detector/generated-state input views are presented,
                        # with no extra objective gradients in the existing arm.
                        with torch.no_grad():
                            auxiliary, extra, support = assembly_loss(
                                model, decoder, row
                            )
                        loss = base
                    components.update(
                        {key: float(value.detach()) for key, value in parts.items()}
                    )
                    components.update(
                        {
                            "assembly_" + key: float(value.detach())
                            for key, value in extra.items()
                        }
                    )
                    compute["baseline_full_node_pairs"] += (
                        row["full"]["node_mask"].shape[1] ** 2
                    )
                    compute["baseline_full_event_views"] += 1
                    compute["pretraining_detector_event_views"] += 1
                else:
                    h, _, _ = detector_features(model, row["detector"])
                    result = decoder(h, row["sources"])
                    member = membership_loss(result, row["targets"])
                    relation, support = assembly_relation_loss(
                        result["states"], row["supervision"]
                    )
                    loss = member + relation
                    components.update(
                        membership=float(member.detach()),
                        within_b_relation=float(relation.detach()),
                    )
                for i, n in enumerate(support):
                    relation_support[i] += n
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                (loss / settings["batch_size"]).backward()
                summed += float(loss.detach())
                compute["detector_node_pairs"] += len(row["targets"]) ** 2
                compute["encoder_passes"] += 4 if stage == "pretraining" else 2
                compute["event_presentations"] += 1
                compute[stage + "_presentations"] += 1
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 5.0, error_if_nonfinite=True
            )
            if step == 0 and not any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.encoder.parameters()
            ):
                raise RuntimeError("Encoder adaptation absent")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("Nonfinite parameters")
            last_loss = summed / settings["batch_size"]
            log.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": last_loss,
                        "components": {
                            k: v / settings["batch_size"] for k, v in components.items()
                        },
                        "gradient_norm": float(norm),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                + "\n"
            )
            if step % 25 == 0 or step + 1 == updates:
                log.flush()
                print(stage, step + 1, last_loss, flush=True)
    save(
        output / (stage + "-final.pt"),
        {
            "model_state_dict": model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": None,
            "torch_rng_state": torch.get_rng_state(),
            "sampling_rng_state": rng.getstate(),
            "step": updates,
            "source_sha": source_sha,
            "data_cache": cache_binding,
            "settings": settings,
            "stage": stage,
            "data_order_sha256": sequence.hexdigest(),
            "normalizer": model.runtime_feature_normalizer,
            "architecture": {"context_width": model.encoder.d_model, **ARCHITECTURE},
            "feature_normalization_pid_and_cohort_contract": cache_binding,
            "resume_authorized": False,
            "pid_weights_frozen_downstream": stage != "pretraining",
        },
    )
    return {
        "updates": updates,
        "presentations": updates * settings["batch_size"],
        "data_order_sha256": sequence.hexdigest(),
        "relation_support_by_class": relation_support,
        "final_loss": last_loss,
        "elapsed_seconds": time.monotonic() - started,
    }


def run(contract_path, arm):
    import torch
    from hypertagging.training.capacity_development import fresh_model, fresh_decoder

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    c = json.loads(contract_path.read_text())
    validate_contract(c)
    verify_bindings(c, ROOT)
    admission = json.loads((contract_path.parent / "admission.json").read_text())
    if admission["status"] != "PASS" or admission["contract_sha256"] != sha(
        contract_path
    ):
        raise ValueError("Runtime admission receipt mismatch")
    import subprocess

    if (
        str(ROOT) != c["source_root"]
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Runtime frozen source revision/root mismatch")
    if (
        arm not in c["arms"]
        or not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
    ):
        raise RuntimeError("Arm/allocation mismatch")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("Requeue forbidden")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CPU allocation cannot use a GPU")
    out = Path(c["output_root"]) / arm
    out.mkdir(exist_ok=False)
    width, objective = arm.split("-")
    width = int(width)
    cache = torch.load(c["cache"]["path"], map_location="cpu", weights_only=False)
    model = fresh_model(width, cache["runtime_normalizer"])
    decoder = fresh_decoder()
    initial = torch.load(
        c["initializations"][str(width)]["path"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(initial["model"])
    decoder.load_state_dict(initial["decoder"])
    write(
        out / "startup.json",
        {
            "status": "ADMITTED_STARTED",
            "job_id": os.environ["SLURM_JOB_ID"],
            "arm": arm,
            "source_sha": c["source_sha"],
            "contract": binding(contract_path),
            "initialization": c["initializations"][str(width)],
            "cache": c["cache"],
            "resources": c["resources"],
            "settings": c["settings"],
        },
    )
    started = time.monotonic()
    compute = Counter()
    compute["encoder_parameters"] = sum(p.numel() for p in model.encoder.parameters())
    compute["pretraining_model_parameters"] = sum(p.numel() for p in model.parameters())
    compute["common_decoder_parameters"] = sum(p.numel() for p in decoder.parameters())
    pretrain = fit(
        model,
        decoder,
        cache["train"],
        stage="pretraining",
        objective=objective,
        settings=c["settings"],
        output=out,
        compute=compute,
        cache_binding=c["cache"],
        source_sha=c["source_sha"],
    )
    tiny = []
    for category in ("charged", "mixed", "ccbar"):
        tiny.extend([r for r in cache["train"] if r["category"] == category][:8])
    # Independent tiny branch cannot alter the main fit/history.
    tiny_model = copy.deepcopy(model)
    tiny_decoder = fresh_decoder()
    tiny_history = fit(
        tiny_model,
        tiny_decoder,
        tiny,
        stage="tiny",
        objective=objective,
        settings=c["settings"],
        output=out,
        compute=compute,
        cache_binding=c["cache"],
        source_sha=c["source_sha"],
    )
    tiny_result = evaluate(tiny_model, tiny_decoder, tiny)
    del tiny_model, tiny_decoder
    decoder = fresh_decoder()
    main = fit(
        model,
        decoder,
        cache["train"],
        stage="downstream",
        objective=objective,
        settings=c["settings"],
        output=out,
        compute=compute,
        cache_binding=c["cache"],
        source_sha=c["source_sha"],
    )
    train_result = evaluate(model, decoder, cache["train"])
    heldout = evaluate(model, decoder, cache["development"])
    compute["evaluation_event_views"] = (
        len(tiny) + len(cache["train"]) + len(cache["development"])
    )
    compute["encoder_passes"] += 2 * compute["evaluation_event_views"]
    compute["evaluation_node_pairs"] = sum(
        len(r["targets"]) ** 2 for r in tiny + cache["train"] + cache["development"]
    )
    compute.update(
        wall_seconds=time.monotonic() - started,
        cpu_seconds=time.process_time(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    result = {
        "arm": arm,
        "stage": "development",
        "source_sha": c["source_sha"],
        "contract_sha256": sha(contract_path),
        "pretraining": pretrain,
        "tiny_history": tiny_history,
        "downstream": main,
        "tiny": tiny_result,
        "train": train_result,
        "heldout": heldout,
        "compute": dict(compute),
        "flops": "not_measured_node_pair_workload_is_only_a_proxy_equal_updates_not_equal_FLOPs",
        "automatic_successor": False,
        "primary_eligible": False,
        "one_seed_exploratory": True,
    }
    write(out / "summary.json", result)
    write(
        out / "terminal.json",
        {
            "status": "COMPLETED",
            "job_id": os.environ["SLURM_JOB_ID"],
            "summary": binding(out / "summary.json"),
            "checkpoints": [
                binding(out / (stage + "-final.pt"))
                for stage in ("pretraining", "tiny", "downstream")
            ],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--arm", required=True)
    a = p.parse_args()
    run(a.contract.resolve(), a.arm)
