#!/usr/bin/env python3
"""One two-arm CPU development campaign: frozen versus adapted membership encoder."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.diagnose_search_survival import checked, sha, write  # noqa: E402


def run(plan_path, out, arm):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The real-data pilot requires a CPU Slurm allocation")
    import pyarrow.parquet as pq
    import torch
    from hypertagging.data.heterogeneous import collate_heterogeneous_events
    from hypertagging.evaluation.full_decay_metrics import _tree_view, _truth_b_roots
    from hypertagging.evaluation.trained_context import load_trained_evaluation_context
    from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair
    from hypertagging.models.direct_membership import DirectMembershipHead, direct_membership_loss
    from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
    from hypertagging.reconstruction.level_rollout import _constrained_rollout_model_batch
    from hypertagging.reconstruction.beam_search import _truth_free_model_view

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261008)
    random.seed(20261008)
    plan = json.loads(plan_path.read_text())
    expected_settings = {"seed":20261008,"head_width":128,"tiny_updates":1000,"tiny_batch_size":24,"pilot_updates":1000,"pilot_batch_size":16,"head_lr":.001,"encoder_lr":.00005,"checkpoint":"fixed_final","presence_threshold":.5}
    if plan["settings"] != expected_settings:
        raise ValueError("Runtime settings differ from admitted fixed budget")
    if arm not in ("frozen", "adapted") or plan["stage"] != "development":
        raise ValueError("Unknown development arm/stage")
    if plan["automatic_successor"] or plan["sealed_test_access"]:
        raise ValueError("Scientific authority exceeded")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise ValueError("Restart not authorized")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 2:
        raise ValueError("Expected bounded two-CPU allocation")
    if out != Path(plan["output_root"]) / arm:
        raise ValueError("Output differs from admitted contract")
    from scripts.validate_next_reconstruction_study import validate
    scientific = json.loads(checked(plan["scientific_plan"]).read_text())
    policy = json.loads(checked(plan["policy"]).read_text())
    evidence = json.loads(checked(plan["evidence"]).read_text())
    if validate(scientific, policy, evidence)["status"] != "PASS":
        raise ValueError("Scientific planning gate failed")
    checked(plan["gradient_preflight"])
    checked(plan["phase72_aggregate"])
    for item in plan["data_shards"]:
        checked(item)
    for item in plan["exclusion_bindings"]:
        checked(item)
    for relative, digest in plan["source_hashes"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Frozen source changed: {relative}")
    paths = {k: checked(v) for k, v in plan["inputs"].items()}
    validate_checkpoint_pair(paths["pretraining-checkpoint"], paths["reconstruction-checkpoint"],
                            require_exact_frozen_encoder=False)
    uids = json.loads(paths["cohort"].read_text())["event_uids"]
    context = load_trained_evaluation_context(checkpoint=paths["reconstruction-checkpoint"],
        data=paths["selection"], dataset_index=paths["index"], split="validation", device="cpu",
        max_events=len(uids), diagnostic_allow_external_independent_sample=True,
        event_selection="explicit_uids", explicit_event_uids=uids)
    selection = json.loads(paths["selection"].read_text())
    categories = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
    by_category = {cat: [] for cat in categories}
    for entry in selection["entries"]:
        if entry["split"] == "train" and entry["category"] in by_category:
            table = pq.read_table(Path(selection["data_root"]) / entry["path"], columns=["event_uid"], use_threads=False)
            by_category[entry["category"]].extend(table.column("event_uid").to_pylist())
    selected = {cat: sorted(ids, key=lambda uid: hashlib.sha256(f"membership-pilot-v1:{uid}".encode()).digest())[:64]
                for cat, ids in by_category.items()}
    if any(len(ids) != 64 for ids in selected.values()):
        raise ValueError("Insufficient training category coverage")
    train_uids = {uid for ids in selected.values() for uid in ids}
    if sorted(train_uids) != plan["train_uids"] or uids != plan["development_uids"]:
        raise ValueError("Admitted identities differ")
    if len(train_uids) != 384 or train_uids & set(uids):
        raise ValueError("Training/development identity violation")
    out.mkdir(exist_ok=False)
    write(out / "training-cohort.json", {"event_uids": sorted(train_uids), "by_category": selected,
          "role": "train", "selection_uses_truth": False})
    write(out / "contract.json", {"version": "phase73-direct-membership-development-v1",
        "plan_sha256": sha(plan_path), "seed": 20261008, "encoder_frozen": arm == "frozen", "arm": arm, "head_width": 128,
        "train_event_count": 384, "dev_event_count": 60, "updates": 1000, "batch_size": 16,
        "learning_rate": .001, "threshold": .5, "checkpoint_selection": "fixed_final",
        "tiny_memorization_updates": 1000, "gpus": 0, "hierarchy_prediction": False,
        "primary_study": False, "job_id": os.environ["SLURM_JOB_ID"], "source_sha": plan["source_sha"]})
    write(out / "startup.json", {"status": "ADMITTED", "arm": arm, "job_id": os.environ["SLURM_JOB_ID"], "contract_sha256": sha(plan_path)})
    started = time.monotonic()
    context.model.eval()
    for parameter in context.model.parameters():
        parameter.requires_grad_(False)
    initial_encoder = {k: v.detach().clone() for k, v in context.model.encoder.state_dict().items()}
    initial_pid = {k: v.detach().clone() for k, v in context.model.leaf_pid_head.state_dict().items()}
    compute = {"model_event_forwards": 0, "model_gradient_event_forwards": 0,
               "event_node_pairs": 0, "encoder_parameters": sum(p.numel() for p in context.model.encoder.parameters())}

    def save_checkpoint(value, destination):
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if destination.exists(): raise FileExistsError(destination)
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, destination)

    def dynamic_features(row):
        compute["model_event_forwards"] += 1
        compute["model_gradient_event_forwards"] += int(torch.is_grad_enabled() and arm == "adapted")
        compute["event_node_pairs"] += len(row["targets"]) ** 2
        return context.model(row["model_input"], target_level=1,
            pid_kinematics_mode_override="soft_expectation", pid_temperature_override=.5).node_embeddings[0]


    def encode(event):
        truth = context.data_module.normalize_batch(collate_heterogeneous_events([event]))
        projection = project_schema_v4_fsps(truth)
        model_input = _constrained_rollout_model_batch(projection.batch, target_level=1, policy=context.constraint_policy)
        with torch.inference_mode():
            output = context.model(_truth_free_model_view(model_input), target_level=1,
                                   pid_kinematics_mode_override="soft_expectation", pid_temperature_override=.5)
        compute["model_event_forwards"] += 1
        compute["event_node_pairs"] += output.node_embeddings.shape[1] ** 2
        # Clone outside inference_mode so cached features can feed a trained head.
        features = output.node_embeddings[0].clone()
        view = _tree_view(truth, 0, truth=True)
        roots = _truth_b_roots(view) if event.source_category in ("charged", "mixed") else []
        source_keys = projection.evaluation_leaf_source_keys[0].tolist()
        labels = torch.zeros(len(source_keys), dtype=torch.long)
        if event.source_category in ("charged", "mixed") and len(roots) != 2:
            labels.fill_(-1)
        else:
            for slot, root in enumerate(roots, 1):
                group = view.source_set(root)
                if not group or not group <= set(source_keys):
                    labels.fill_(-1)
                    break
                for position, key in enumerate(source_keys):
                    if key in group:
                        if labels[position] != 0:
                            raise ValueError("Overlapping truth groups")
                        labels[position] = slot
        return {"uid": event.event_uid, "category": event.source_category, "features": features,
                "model_input": _truth_free_model_view(model_input), "targets": labels, "charge": projection.batch["charge"][0].clone(),
                "sources": projection.batch["recursive_leaf_source_mask"][0].clone()}

    train = []
    for event in context.data_module.iter_events("train", event_uids=train_uids):
        train.append(encode(event))
        if len(train) % 64 == 0:
            print(f"encoded train {len(train)}/384", flush=True)
    if {row["uid"] for row in train} != train_uids:
        raise ValueError("Training selection incomplete")
    train.sort(key=lambda row: row["uid"])
    dev = [encode(event) for event in context.events]
    save_checkpoint({"train": train, "development": dev}, out / "features.pt")

    def batch(rows):
        width = max(len(row["targets"]) for row in rows)
        x = torch.zeros(len(rows), width, rows[0]["features"].shape[-1])
        labels = torch.full((len(rows), width), -1, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.bool)
        for index, row in enumerate(rows):
            n = len(row["targets"])
            x[index, :n], labels[index, :n], mask[index, :n] = row["features"], row["targets"], True
        return x, labels, mask

    def evaluate(model, rows, dynamic=False):
        totals = Counter()
        event_rows = []
        iou_sum = 0.
        with torch.inference_mode():
            for row in rows:
                before = totals.copy()
                x, target, mask = batch([{**row, "features": dynamic_features(row)} if dynamic else row])
                logits, objects = model(x, mask)
                probability = logits[0].softmax(-1)
                assignment = probability.argmax(-1)
                raw = [set((assignment == slot).nonzero().flatten().tolist()) for slot in (1, 2)]
                candidates, used = [], set()
                for slot in sorted(range(2), key=lambda s: -float(objects[0, s])):
                    if float(objects[0, slot].sigmoid()) < .5:
                        continue
                    chosen, local_used = [], set()
                    for position in sorted(raw[slot], key=lambda p: -float(probability[p, slot + 1])):
                        sources = set(row["sources"][position].nonzero().flatten().tolist())
                        if sources & (used | local_used):
                            totals["rejected_source_conflict_nodes"] += 1
                        if sources and not sources & (used | local_used):
                            chosen.append(position)
                            local_used.update(sources)
                    charge = float(row["charge"][chosen].sum()) if chosen else 0.
                    totals["rejected_cardinality_groups"] += int(len(chosen) < 2)
                    totals["rejected_charge_groups"] += int(len(chosen) >= 2 and min(abs(charge - q) for q in (-1,0,1)) > 1e-5)
                    if len(chosen) >= 2 and min(abs(charge - q) for q in (-1, 0, 1)) <= 1e-5:
                        totals["accepted_groups"] += 1
                        candidates.append(set(chosen))
                        used.update(local_used)
                accepted_sources = set()
                for candidate in candidates:
                    candidate_sources = set()
                    for position in candidate:
                        sources = set(row["sources"][position].nonzero().flatten().tolist())
                        totals["accepted_source_conflicts"] += len(sources & (accepted_sources | candidate_sources))
                        candidate_sources.update(sources)
                    accepted_sources.update(candidate_sources)
                if totals["accepted_source_conflicts"]:
                    raise ValueError("Accepted membership source conflict")
                totals["events"] += 1
                if row["category"] not in ("charged", "mixed"):
                    totals["continuum_events"] += 1
                    totals["continuum_accepted_events"] += bool(candidates)
                    event_rows.append({"uid": row["uid"], "category": row["category"], "counts": dict(totals-before)})
                    continue
                totals["nominal_b_trials"] += 2
                if not bool((target >= 0).any()):
                    totals["unknown_membership_trials"] += 2
                    event_rows.append({"uid": row["uid"], "category": row["category"], "counts": dict(totals-before)})
                    continue
                for slot in (1, 2):
                    truth = set((target[0] == slot).nonzero().flatten().tolist())
                    totals["raw_exact_memberships"] += truth in raw
                    totals["accepted_exact_memberships"] += truth in candidates
                    best = max(raw, key=lambda g: len(g & truth) / max(len(g | truth), 1))
                    iou_sum += len(best & truth) / max(len(best | truth), 1)
                    totals["membership_iou_trials"] += 1
                    totals["best_group_intersection"] += len(best & truth)
                    totals["best_group_truth_sources"] += len(truth)
                    totals["best_group_predicted_sources"] += len(best)
                event_rows.append({"uid": row["uid"], "category": row["category"], "counts": dict(totals-before)})
        return {**dict(totals), "event_rows": event_rows, "mean_best_group_iou": iou_sum / max(totals["membership_iou_trials"], 1),
                "accepted_source_conflicts": totals["accepted_source_conflicts"], "definition": "flat_exclusive_FSP_groups_no_internal_hierarchy_no_FEI_equivalence"}

    def fit(rows, batch_size, name):
        torch.manual_seed(20261008)
        rng = random.Random(20261008)
        model = DirectMembershipHead(rows[0]["features"].shape[-1], hidden_dim=128)
        dynamic = name == "pilot_384"
        for parameter in context.model.encoder.parameters():
            parameter.requires_grad_(dynamic and arm == "adapted")
        groups = [{"params": list(model.parameters()), "lr": .001}]
        if dynamic and arm == "adapted":
            groups.append({"params": list(context.model.encoder.parameters()), "lr": .00005})
        optimizer = torch.optim.AdamW(groups)
        parameters = [p for group in groups for p in group["params"]]
        log = (out / (name + "-metrics.jsonl")).open("x")
        history = []
        for step in range(1000):
            chosen = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
            if dynamic:
                chosen = [{**row, "features": dynamic_features(row)} for row in chosen]
            features, targets, mask = batch(chosen)
            logits, objects = model(features, mask)
            loss = direct_membership_loss(logits, objects, targets, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite pilot loss")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            if dynamic and arm == "adapted" and step == 0:
                if not any(p.grad is not None and bool(p.grad.ne(0).any()) for p in context.model.encoder.parameters()):
                    raise RuntimeError("Encoder adaptation did not execute")
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Nonfinite fitted state")
            log.write(json.dumps({"step": step+1, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
                                  "elapsed_seconds": time.monotonic()-started}) + "\n")
            if step % 100 == 0: log.flush()
            if step % 100 == 0 or step == 999:
                history.append({"step": step + 1, "loss": float(loss.detach())})
                print(name, history[-1], flush=True)
        log.close()
        model.eval()
        save_checkpoint({"state_dict": model.state_dict(), "input_dim": rows[0]["features"].shape[-1],
                    "hidden_dim": 128, "encoder_state_dict": context.model.encoder.state_dict(),
                    "model_state_dict": context.model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None, "scaler_state_dict": None, "step": 1000,
                    "torch_rng_state": torch.get_rng_state(), "sampling_rng_state": rng.getstate(),
                    "source_checkpoint_sha256": plan["inputs"]["reconstruction-checkpoint"]["sha256"],
                    "normalizer_state": context.checkpoint["normalizer_state"],
                    "architecture_and_pid_contract": context.checkpoint["config"],
                    "fit_settings": plan["settings"], "fitting_event_uids": [r["uid"] for r in rows],
                    "development_event_uids": plan["development_uids"], "source_sha": plan["source_sha"],
                    "encoder_adaptation_executed": dynamic and arm == "adapted", "resume_authorized": False,
                    "contract_sha256": sha(out / "contract.json")}, out / f"{name}.pt")
        return {"history": history, "train": evaluate(model, rows, dynamic), "development": evaluate(model, dev, dynamic)}

    tiny = []
    for category in categories[:3]:
        tiny.extend([row for row in train if row["category"] == category][:8])
    results = {"tiny_memorization": fit(tiny, 24, "tiny_memorization"),
               "pilot_384": fit(train, 16, "pilot_384")}
    changed = [k for k, v in context.model.encoder.state_dict().items() if not torch.equal(v, initial_encoder[k])]
    assert bool(changed) == (arm == "adapted")
    assert all(torch.equal(v, initial_pid[k]) for k, v in context.model.leaf_pid_head.state_dict().items())
    import resource
    compute.update(encoder_passes=2*compute["model_event_forwards"], elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time(),
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   head_parameters=sum(p.numel() for p in DirectMembershipHead(128,128).parameters()),
                   flops="unavailable; event forwards and node-pair counts are workload proxies, not FLOPs")
    write(out / "compute.json", compute)
    write(out / "summary.json", {"version": "phase73-direct-membership-development-v1", "results": results,
          "elapsed_seconds": time.monotonic() - started, "train_category_counts": dict(Counter(r["category"] for r in train)),
          "source_sha256": plan["source_hashes"], "encoder_changed": bool(changed), "changed_encoder_keys": changed, "pid_head_unchanged": True, "compute": compute,
          "limitations": ["small_training_sample", "frozen_vs_adapted_encoder", "one_seed", "development_cohort_reused",
                          "different_flat_group_target_not_exact_tree", "no_heldout_model_superiority_claim"]})
    write(out / "receipt.json", {"status": "COMPLETED", "job_id": os.environ["SLURM_JOB_ID"],
          "summary_sha256": sha(out / "summary.json"), "features_sha256": sha(out / "features.pt"), "checkpoints": {name: sha(out / name) for name in ("tiny_memorization.pt", "pilot_384.pt")}, "source_sha": plan["source_sha"], "runtime_contract_sha256": sha(plan_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["frozen", "adapted"], required=True)
    args = parser.parse_args()
    run(args.plan.resolve(), args.output.resolve(), args.arm)
