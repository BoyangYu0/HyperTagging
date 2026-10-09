"""Guarded fixed-budget detector-versus-context readability diagnostic.

No production changes, validation evaluation, study selection, or submission.
All fitted probes are exploratory TRAIN-role diagnostics, not reconstruction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
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
from scripts.phase84_information_probe import (  # noqa:E402
    detector_nodes,
    fit_normalizer,
    normalize,
    pair_inputs,
    pair_labels,
    shuffled_targets,
    source_pairs,
    new_probe,
    fit_probe,
    score_pairs,
)
from scripts.diagnose_phase78_partition import split_rows  # noqa:E402
from scripts.diagnose_phase77_proposals import auc  # noqa:E402
from scripts.prepare_phase74_development_data import binding, write  # noqa:E402
from scripts.run_phase74_development import verify_bindings, save  # noqa:E402
from scripts.phase76_development import guarded  # noqa:E402
from hypertagging.training.capacity_development import fresh_model, detector_features  # noqa:E402

SEED = 8409
UPDATES = 2048
BATCH = 256
ARMS = ("detector_true", "embedding_true", "detector_shuffled", "embedding_shuffled")


def validate_contract(c, smoke):
    if c["stage"] != "training_role_information_diagnostic":
        raise ValueError("Wrong diagnostic stage")
    expected = {
        "seed": SEED,
        "updates": UPDATES,
        "batch_size": BATCH,
        "probe_parameters": 24835,
        "training_events": 1536,
        "validation_events": 0,
        "scientific_model_updates": 0,
        "automatic_successor": False,
    }
    if any(c.get(k) != v for k, v in expected.items()) or c["arms"] != list(ARMS):
        raise ValueError("Diagnostic budget or authority changed")
    r = c["resources"]
    if (
        r["cpus"] != 2
        or r["memory_gib"] != 16
        or r["hours"] != 1
        or r["gpus"] != 0
        or r["requeue"]
    ):
        raise ValueError("Resource envelope changed")
    if bool(c["smoke"]) != smoke:
        raise ValueError("Smoke/scientific contract mismatch")
    if c["checkpoint_selection"] not in (
        "fixed_phase84_full1536_step6000",
        "fixed_historical_phase76_native",
    ):
        raise ValueError("Unregistered checkpoint selection")
    bound = {b["path"]: b["sha256"] for b in c["bindings"]}
    for key in ("cache", "checkpoint"):
        if bound.get(c[key]["path"]) != c[key]["sha256"]:
            raise ValueError("Unbound cache/checkpoint")


def validate_checkpoint(c, ck):
    if c["checkpoint_selection"] == "fixed_phase84_full1536_step6000":
        if (
            ck.get("step") != 6000
            or ck.get("settings", {}).get("downstream_updates") != 6000
        ):
            raise ValueError("Checkpoint is not fixed6000")
        parent = ck.get("contract")
        if not parent or parent not in c["bindings"]:
            raise ValueError("Missing frozen exposure contract")
        if binding(Path(parent["path"])) != parent:
            raise ValueError("Changed parent checkpoint contract")
        record = json.loads(Path(parent["path"]).read_text())
        if record.get("kind") != "phase84_exposure" or record.get("pool_size") != 1536:
            raise ValueError("Wrong checkpoint training pool")
    elif (
        c["checkpoint"]["sha256"]
        != "ea86f04cbbdf72216f8dbd93a0833a6fe9105beec519310214a7e52c4a43e603"
        or ck.get("step") != 1500
    ):
        raise ValueError("Historical native checkpoint changed")


def classify(probabilities, target, ij):
    """Evaluation-only join after detached probabilities have been serialized."""
    labels = pair_labels(target, ij)
    known = labels >= 0
    pred = probabilities.argmax(-1)
    confusion = torch.bincount(3 * labels[known] + pred[known], minlength=9).reshape(
        3, 3
    )
    conditional = known & (labels < 2)
    conditional_probability = probabilities[:, 0] / probabilities[:, :2].sum(
        -1
    ).clamp_min(1e-30)
    ranked = auc(
        conditional_probability[conditional].tolist(),
        (labels[conditional] == 0).long().tolist(),
    )
    background = labels == 2
    return {
        "confusion": confusion.tolist(),
        "known_pairs": int(known.sum()),
        "unknown_pairs": int((~known).sum()),
        "conditional_same_cross": ranked,
        "background_false_B_pairs": int((background & (pred != 2)).sum()),
        "background_pairs": int(background.sum()),
        "class_recall": [
            None
            if not int(confusion[k].sum())
            else float(confusion[k, k] / confusion[k].sum())
            for k in range(3)
        ],
    }


def aggregate(records, arm):
    confusion = np.zeros((3, 3), dtype=np.int64)
    values = []
    unknown = 0
    for r in records:
        m = r["arms"][arm]
        confusion += np.asarray(m["confusion"], dtype=np.int64)
        unknown += m["unknown_pairs"]
        if m["conditional_same_cross"]["auc"] is not None:
            values.append(m["conditional_same_cross"]["auc"])
    support = confusion.sum(1)
    recalls = [
        None if not support[k] else float(confusion[k, k] / support[k])
        for k in range(3)
    ]
    return {
        "events": len(records),
        "confusion": confusion.tolist(),
        "class_support": support.tolist(),
        "class_recall": recalls,
        "balanced_class_recall": None
        if any(v is None for v in recalls)
        else float(np.mean(recalls)),
        "conditional_AUC_event_mean": None if not values else float(np.mean(values)),
        "conditional_AUC_available_events": len(values),
        "conditional_AUC_unavailable_events": len(records) - len(values),
        "known_pairs": int(support.sum()),
        "unknown_pairs": unknown,
        "background_false_B_pairs": int(confusion[2, :2].sum()),
        "background_pairs": int(support[2]),
    }


def paired_bootstrap(records, a, b, repetitions=2000):
    """Stratified whole-event bootstrap, never independent pair resampling."""
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
    generator = np.random.default_rng(SEED)
    base_a, base_b = aggregate(records, a), aggregate(records, b)
    keys = ("conditional_AUC_event_mean", "balanced_class_recall")
    draws = {key: [] for key in keys}
    for _ in range(repetitions):
        sample = []
        for category in sorted(by_category):
            values = by_category[category]
            sample.extend(
                values[i] for i in generator.integers(len(values), size=len(values))
            )
        av, bv = aggregate(sample, a), aggregate(sample, b)
        for key in keys:
            if av[key] is not None and bv[key] is not None:
                draws[key].append(av[key] - bv[key])
    return {
        key: {
            "difference": None
            if base_a[key] is None or base_b[key] is None
            else base_a[key] - base_b[key],
            "interval95": None
            if not draws[key]
            else np.quantile(draws[key], [0.025, 0.975]).tolist(),
            "available_replicates": len(draws[key]),
            "requested_replicates": repetitions,
            "unit": "whole collision, stratified category; adaptive TRAIN-role assessment",
        }
        for key in keys
    }


def strata(records):
    result = {}
    for key in ("partition", "category", "fsp_size", "B_size_signature"):
        grouped = defaultdict(list)
        for r in records:
            grouped[str(r[key])].append(r)
        result[key] = {
            name: {arm: aggregate(rows, arm) for arm in ARMS}
            for name, rows in sorted(grouped.items())
        }
    return result


def digest_state(model):
    return hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()


def main(contract_path, smoke=False):
    guarded()
    c = json.loads(contract_path.read_text())
    validate_contract(c, smoke)
    if (
        str(ROOT) != c["source_root"]
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Frozen source binding mismatch")
    verify_bindings(c, ROOT)
    output = Path(c["output"]).resolve()
    if (
        output.exists()
        or output == contract_path.resolve()
        or output in [Path(c[k]["path"]).resolve() for k in ("cache", "checkpoint")]
    ):
        raise ValueError("Output must be new and separate")
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    cache = torch.load(c["cache"]["path"], map_location="cpu", weights_only=False)
    if set(cache) != {"train", "runtime_normalizer"}:
        raise ValueError("Non-training cache contents")
    rows = cache["train"]
    roles = split_rows(rows)
    if len(rows) != 1536 or len(roles) != 1536:
        raise ValueError("Authenticated training capacity mismatch")
    from scripts.phase84_train_isolation import authenticate_train_isolation

    write(
        output / "isolation.json",
        authenticate_train_isolation([r["uid"] for r in rows], ROOT),
    )
    full_pair_upper = sum(
        len(r["sources"]) * (len(r["sources"]) - 1) // 2 for r in rows
    )
    memory_bound = 4 * 1024**3 + 3 * full_pair_upper * 384 * 4
    if memory_bound > 12 * 1024**3:
        raise ValueError("Full pair-feature analytic memory admission failed")
    write(
        output / "analytic-memory-admission.json",
        {
            "status": "PASS",
            "upper_pairs": full_pair_upper,
            "bound_bytes": memory_bound,
            "limit_bytes": 12 * 1024**3,
            "formula": "4GiBbase +3copies*allpossiblepairs*384float32",
        },
    )
    if smoke:
        chosen = set()
        for category in sorted({r["category"] for r in rows}):
            for role, count in (("fit", 2), ("assessment", 1)):
                eligible = sorted(
                    r["uid"]
                    for r in rows
                    if r["category"] == category and roles[r["uid"]] == role
                )
                chosen.update(eligible[:count])
        rows = [r for r in rows if r["uid"] in chosen]
    model = fresh_model(128, cache["runtime_normalizer"])
    ck = torch.load(c["checkpoint"]["path"], map_location="cpu", weights_only=False)
    validate_checkpoint(c, ck)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    del ck
    detached = []
    with torch.inference_mode():
        for row in rows:
            d, observed, continuous = detector_nodes(row["detector"])
            h, _, _ = detector_features(model, row["detector"])
            if h.shape != d.shape or h.shape[1] != 128 or len(row["sources"]) != len(h):
                raise ValueError("Detector/embedding/source alignment")
            ij, support = source_pairs(row["sources"])
            detached.append(
                {
                    "uid": row["uid"],
                    "detector": d.detach(),
                    "detector_observed": observed,
                    "detector_continuous": continuous,
                    "embedding": h.detach(),
                    "embedding_observed": torch.ones_like(h, dtype=torch.bool),
                    "embedding_continuous": torch.ones(128, dtype=torch.bool),
                    "sources": row["sources"].clone(),
                    "ij": ij,
                    "source_support": support,
                }
            )
    save(output / "model-only-inputs.pt", detached)
    input_receipt = binding(output / "model-only-inputs.pt")
    del model
    by_uid = {r["uid"]: r for r in rows}
    # Targets are first attached only after the detector-only extraction is saved.
    records = [
        dict(
            r,
            target=by_uid[r["uid"]]["targets"],
            category=by_uid[r["uid"]]["category"],
            source_role="train",
            partition=roles[r["uid"]],
        )
        for r in detached
    ]
    normalizers = {
        feature: fit_normalizer(records, feature)
        for feature in ("detector", "embedding")
    }
    save(output / "normalizers.pt", normalizers)
    models, fits, score_files = {}, {}, {}
    exclusions = {}
    for arm in ARMS:
        feature, label_condition = arm.split("_")
        fitting = []
        unsupported = []
        for r in records:
            if r["partition"] != "fit":
                continue
            target = (
                r["target"]
                if label_condition == "true"
                else shuffled_targets(r["target"], r["uid"], SEED)
            )
            y = pair_labels(target, r["ij"])
            if not (y >= 0).any():
                unsupported.append(
                    {"uid": r["uid"], "reason": "no known source-disjoint pairs"}
                )
                continue
            x = pair_inputs(
                normalize(r[feature], r[feature + "_observed"], normalizers[feature]),
                r["ij"],
            )
            fitting.append(
                {
                    "uid": r["uid"],
                    "source_role": "train",
                    "partition": "fit",
                    "x": x,
                    "y": y,
                }
            )
        initial = new_probe(SEED)
        initial_hash = digest_state(initial)
        initial.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(
            initial(fitting[0]["x"]), fitting[0]["y"], ignore_index=-1
        ).backward()
        gradients = {
            name: float(p.grad.norm()) for name, p in initial.named_parameters()
        }
        if not all(np.isfinite(v) and v > 0 for v in gradients.values()):
            raise ValueError("Real-data probe gradient admission failed")
        fit_start, fit_cpu = time.monotonic(), time.process_time()
        fitted, report = fit_probe(
            fitting, seed=SEED, updates=2 if smoke else UPDATES, batch_size=BATCH
        )
        report.update(
            initial_sha256=initial_hash,
            real_data_parameter_gradient_norms=gradients,
            diagnostic_gradient_pair_presentations=len(fitting[0]["y"]),
            wall_seconds=time.monotonic() - fit_start,
            cpu_seconds=time.process_time() - fit_cpu,
            eligible_fit_events=len(fitting),
            fitting_pair_support=sum(int((r["y"] >= 0).sum()) for r in fitting),
            final_state_sha256=digest_state(fitted),
        )
        fits[arm] = report
        models[arm] = fitted.state_dict()
        exclusions[arm] = unsupported
        del fitting, initial
        gc.collect()
        # Score model-visible inputs only; serialize all scores before outcome joins.
        scores = []
        for r in records:
            x = pair_inputs(
                normalize(r[feature], r[feature + "_observed"], normalizers[feature]),
                r["ij"],
            )
            scores.append(
                {
                    "uid": r["uid"],
                    "ij": r["ij"],
                    "probabilities": score_pairs(fitted, x),
                }
            )
        path = output / (arm + "-model-only-scores.pt")
        save(path, scores)
        score_files[arm] = binding(path)
        del scores, fitted
        print("finished", arm, report, flush=True)
    if len({v["initial_sha256"] for v in fits.values()}) != 1:
        raise ValueError("Probe initialization changed across arms")
    if (
        fits["detector_true"]["order_sha256"] != fits["embedding_true"]["order_sha256"]
        or fits["detector_shuffled"]["order_sha256"]
        != fits["embedding_shuffled"]["order_sha256"]
    ):
        raise ValueError("Feature-arm sampling not matched")
    save(output / "probe-final.pt", models)
    scored = {
        arm: {
            r["uid"]: r
            for r in torch.load(b["path"], map_location="cpu", weights_only=False)
        }
        for arm, b in score_files.items()
    }
    events = []
    for r in records:
        target = r["target"]
        event = {
            "uid": r["uid"],
            "partition": r["partition"],
            "category": r["category"],
            "fsp_size": len(target),
            "B_size_signature": sorted(int((target == k).sum()) for k in (1, 2)),
            "unknown_nodes": int((target < 0).sum()),
            "source_support": r["source_support"],
            "arms": {},
        }
        for arm in ARMS:
            s = scored[arm][r["uid"]]
            if not torch.equal(s["ij"], r["ij"]):
                raise ValueError("Saved pair alignment changed")
            event["arms"][arm] = classify(s["probabilities"], target, r["ij"])
        events.append(event)
    assessment = [r for r in events if r["partition"] == "assessment"]
    paired = {
        f"{a}-minus-{b}": paired_bootstrap(assessment, a, b, 20 if smoke else 2000)
        for a, b in [
            ("detector_true", "embedding_true"),
            ("detector_true", "detector_shuffled"),
            ("embedding_true", "embedding_shuffled"),
        ]
    }
    aggregates = {arm: aggregate(assessment, arm) for arm in ARMS}

    def lower_positive(comparison, metric):
        ci = paired[comparison][metric]["interval95"]
        return ci is not None and ci[0] > 0

    readable = all(
        lower_positive(comparison, metric)
        for comparison in (
            "detector_true-minus-embedding_true",
            "detector_true-minus-detector_shuffled",
        )
        for metric in ("conditional_AUC_event_mean", "balanced_class_recall")
    )
    bg_d, bg_e = aggregates["detector_true"], aggregates["embedding_true"]
    background_ok = (
        bg_d["background_pairs"] == bg_e["background_pairs"]
        and bg_d["background_false_B_pairs"] <= bg_e["background_false_B_pairs"]
    )
    summary = {
        "stage": c["stage"],
        "smoke": smoke,
        "checkpoint_selection": c["checkpoint_selection"],
        "checkpoint": c["checkpoint"],
        "cache": c["cache"],
        "source_sha": c["source_sha"],
        "model_only_inputs": input_receipt,
        "score_files": score_files,
        "training_role_events": len(records),
        "fitting_events": sum(r["partition"] == "fit" for r in records),
        "assessment_events": len(assessment),
        "validation_events": 0,
        "scientific_model_updates": 0,
        "fits": fits,
        "excluded_fit_events": exclusions,
        "assessment": aggregates,
        "paired": paired,
        "strata": strata(events),
        "gate": {
            "stronger_detector_information_supported": bool(
                readable and background_ok and not smoke
            ),
            "background_nonworsening": bool(background_ok),
            "scope": "Exploratory finite probe readability, not deployable membership or independent confirmation",
        },
        "compute": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "detector_runtime_calls": len(records),
            "encoder_forwards": 2 * len(records),
            "encoder_node_square_proxy": 2
            * sum(len(r["target"]) ** 2 for r in records),
            "unique_training_identities": len(records),
            "source_disjoint_pairs": sum(
                r["source_support"]["eligible_pairs"] for r in records
            ),
            "source_alias_excluded_pairs": sum(
                r["source_support"]["source_alias_excluded"] for r in records
            ),
            "probe_pair_presentations": sum(
                r["pair_presentations"] for r in fits.values()
            ),
            "gradient_admission_pair_presentations": sum(
                r["diagnostic_gradient_pair_presentations"] for r in fits.values()
            ),
            "score_pair_presentations": len(ARMS)
            * sum(r["source_support"]["eligible_pairs"] for r in records),
            "analytic_memory_bound_bytes": memory_bound,
            "probe_parameters_each": 24835,
            "equal_FLOPs_claim": False,
        },
        "limitations": [
            "Zero bootstrap intervals with sparse/all-zero observations do not establish equivalence.",
            "All events are TRAIN role; repeated exploratory assessment is not independent confirmation.",
            "Embeddings already include learned whole-event context/PID processing; equal head capacity does not match upstream processing.",
            "Detector availability flags and categorical one-hot fields are retained; only constant projected daughter count is omitted.",
            "Finite nonconvex fit has no stationarity/global-optimum claim. No heldout tuning or refit.",
            "Pair probabilities are diagnostic, not legal B groups. Exact physical trees/beam/p4/FEI efficiency unavailable.",
            "Within-event shuffled controls preserve known-node class counts and unknown positions; evaluation joins original truth.",
            "Event-level bootstrap accounts pair correlation; inference beyond reused training cohort is unsupported.",
        ],
    }
    write(output / "event-metrics-private.json", events)
    write(output / "summary.json", summary)
    verify_bindings(c, ROOT)
    write(
        output / "terminal.json",
        {
            "status": "COMPLETED",
            "source_sha": c["source_sha"],
            "artifacts": [binding(p) for p in sorted(output.iterdir()) if p.is_file()],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.contract, args.smoke)
