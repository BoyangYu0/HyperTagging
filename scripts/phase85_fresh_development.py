"""Fixed-checkpoint fresh DEVELOPMENT evaluation; never optimize or select models."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, read, sha, write  # noqa:E402

ARMS = ("connection_off", "connection_on")
CATEGORIES = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
TRAINING_SOURCE = "8b9be2b318d2ebc749f431343113b50971b101d6"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(item):
    require(sha(item["path"]) == item["sha256"], "Changed binding")
    return Path(item["path"])


def validate_request(request, mode):
    require(mode in ("smoke", "evaluate"), "Unknown evaluation mode")
    require(
        request.get("stage") == "development"
        and request.get("arms") == list(ARMS)
        and request.get("training_source_sha") == TRAINING_SOURCE
        and request.get("checkpoint_step") == 6000
        and request.get("threshold") == 0.5
        and request.get("optimizer_updates") == 0
        and request.get("sealed_test_access") is False
        and request.get("maximum_events") == 600
        and request.get("training_pool_size") == 384
        and request.get("bootstrap_seed") == 20261010087
        and request.get("bootstrap_replicates") == 2000
        and bool(request.get("plan")),
        "Evaluation scientific controls changed",
    )
    require(set(request["checkpoints"]) == set(ARMS), "Missing checkpoint arm")
    require(
        len({b["sha256"] for b in request["checkpoints"].values()}) == 2,
        "Same checkpoint used for both arms",
    )
    if mode == "evaluate":
        require(
            bool(request.get("data_admission")) and bool(request.get("admission")),
            "Fresh evaluation requires cohort and runtime admission",
        )
    else:
        require(not request.get("data_admission"), "Smoke must not read validation")


def claim_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def helper(item, name):
    path = checked(item)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_checkpoint(cp, arm, training_contract_binding, training_contract):
    require(
        cp["source_sha"] == TRAINING_SOURCE
        and cp["step"] == 6000
        and cp["contract"] == training_contract_binding
        and cp["settings"] == training_contract["settings"]
        and cp["architecture"] == dict(context=128, hyperbolic=32, head=256, depth=4)
        and cp["connection_enabled"] == (arm == "connection_on")
        and cp["semantic_pair_supervision"] is True
        and cp["resume_authorized"] is False,
        "Final checkpoint lineage/architecture/objective mismatch",
    )


def encode_event(event, ctx):
    """Canonical Phase74 encoding, with targets outside the detector-only view."""
    from hypertagging.data.heterogeneous import collate_heterogeneous_events

    full = ctx.data_module.normalize_batch(collate_heterogeneous_events([event]))
    return encode_normalized(
        full, event.event_uid, event.source_category, ctx.constraint_policy
    )


def encode_normalized(full, uid, category, policy):
    """Canonical projection of already train-normalized input, also used in smoke."""
    import torch
    from hypertagging.evaluation.full_decay_metrics import _tree_view, _truth_b_roots
    from hypertagging.reconstruction.hierarchical_inference import (
        project_schema_v4_fsps,
    )
    from hypertagging.reconstruction.level_rollout import (
        _constrained_rollout_model_batch,
    )
    from hypertagging.reconstruction.beam_search import _truth_free_model_view

    projection = project_schema_v4_fsps(full)
    detector = _truth_free_model_view(
        _constrained_rollout_model_batch(
            projection.batch, target_level=1, policy=policy
        )
    )
    keys = projection.evaluation_leaf_source_keys[0].tolist()
    view = _tree_view(full, 0, truth=True)
    mapping = {key: i for i, key in enumerate(keys)}
    node_sets = []
    for node in range(len(view.active)):
        sources = view.source_set(node) if bool(view.active[node]) else frozenset()
        node_sets.append(
            frozenset(mapping[k] for k in sources)
            if sources <= mapping.keys()
            else frozenset()
        )
    roots = _truth_b_roots(view) if category in CATEGORIES[:2] else []
    groups = [node_sets[r] for r in roots]
    targets = torch.zeros(len(keys), dtype=torch.long)
    if category in CATEGORIES[:2] and (len(groups) != 2 or any(not g for g in groups)):
        targets.fill_(-1)
        groups = []
    else:
        for slot, group in enumerate(groups, 1):
            for pos in group:
                require(targets[pos] == 0, "Overlapping truth B groups")
                targets[pos] = slot
    require(1 <= len(keys) <= 256, "FSP capacity exceeded; no replacement/truncation")
    return dict(
        uid=uid,
        category=category,
        full=full,
        detector=detector,
        sources=projection.batch["recursive_leaf_source_mask"][0].clone(),
        charge=projection.batch["charge"][0].clone(),
        targets=targets,
        supervision=dict(
            node_sets=node_sets,
            parents=full["parent_ids"][0].tolist(),
            b_groups=groups,
            deep_sets=[
                node_sets[i]
                for i in range(len(node_sets))
                if int(full["level_ids"][0, i]) >= 2 and node_sets[i]
            ],
        ),
    )


def check_rows(rows, *, expected_uids, per_category=None):
    ids = [r["uid"] for r in rows]
    require(
        len(ids) == len(set(ids)) and set(ids) == set(expected_uids),
        "Cached UID mismatch",
    )
    require(all(1 <= len(r["targets"]) <= 256 for r in rows), "FSP capacity exceeded")
    if per_category is not None:
        require(
            Counter(r["category"] for r in rows)
            == Counter({c: per_category for c in CATEGORIES}),
            "Category coverage mismatch",
        )


def fresh_rows(request, original_cache, legacy, output, started):
    from scripts.phase85_fresh_development_data import revalidate
    from hypertagging.evaluation.trained_context import load_trained_evaluation_context
    import torch

    current = revalidate(request["data_admission"])
    admission = read(request["data_admission"])
    candidate = read(admission["development"])
    checked(request["normalizer_checkpoint"])
    ctx = load_trained_evaluation_context(
        checkpoint=request["normalizer_checkpoint"]["path"],
        data=admission["selection"]["path"],
        dataset_index=admission["index"]["path"],
        split="validation",
        device="cpu",
        max_events=600,
        diagnostic_allow_external_independent_sample=True,
        event_selection="explicit_uids",
        explicit_event_uids=candidate["event_uids"],
    )
    legacy.equal_tree(
        ctx.checkpoint["normalizer_state"],
        original_cache["normalizer_state"],
        "static normalizer",
    )
    legacy.equal_tree(
        ctx.checkpoint["feature_contract"],
        original_cache["source_feature_contract"],
        "feature contract",
    )
    legacy.equal_tree(
        ctx.model.runtime_feature_normalizer.state_dict(),
        original_cache["runtime_normalizer"].state_dict(),
        "runtime normalizer",
    )
    rows = []
    for event in ctx.events:
        resource_guard(started)
        rows.append(encode_event(event, ctx))
    rows.sort(key=lambda r: r["uid"])
    resource_guard(started)
    check_rows(rows, expected_uids=candidate["event_uids"], per_category=100)
    require(
        not set(candidate["event_uids"]) & {r["uid"] for r in original_cache["train"]},
        "Fresh rows overlap TRAIN",
    )
    cache_path = output / "fresh-evaluation-cache.pt"
    with cache_path.open("xb") as stream:
        torch.save(
            dict(
                development=rows,
                runtime_normalizer=original_cache["runtime_normalizer"],
                normalizer_state=original_cache["normalizer_state"],
                source_feature_contract=original_cache["source_feature_contract"],
            ),
            stream,
        )
    write(
        output / "fresh-cache-admission.json",
        dict(
            status="PASS",
            cache=binding(cache_path),
            identity_admission=request["data_admission"],
            current_isolation=current,
            count=len(rows),
            categories=dict(Counter(r["category"] for r in rows)),
            train_refitted=False,
            training_rows_reselected=False,
            unavailable_events=sum(bool((r["targets"] < 0).all()) for r in rows),
        ),
    )
    return rows


def restore(request, original_cache, arm, legacy):
    import torch
    from scripts.phase85_pair_connection import initialize

    training = read(request["training_contract"])
    model, decoder, _, _ = initialize(Path(training["pretrain"]), original_cache, arm)
    cp = torch.load(
        checked(request["checkpoints"][arm]), map_location="cpu", weights_only=False
    )
    validate_checkpoint(cp, arm, request["training_contract"], training)
    legacy.equal_tree(
        cp["normalizer"].state_dict(),
        original_cache["runtime_normalizer"].state_dict(),
        "checkpoint normalizer",
    )
    model.load_state_dict(cp["model_state_dict"], strict=True)
    decoder.load_state_dict(cp["decoder_state_dict"], strict=True)
    legacy.equal_tree(
        model.runtime_feature_normalizer.state_dict(),
        original_cache["runtime_normalizer"].state_dict(),
        "loaded normalizer",
    )
    model.requires_grad_(False)
    decoder.requires_grad_(False)
    model.eval()
    decoder.eval()
    return model, decoder


def summarize_arm(ev, rows, metric, legacy):
    from scripts.review_phase76_terminal import continuum_membership

    ids = [r["uid"] for r in rows]
    require(
        [r["uid"] for r in ev["native"]["events"]]
        == ids
        == [r["uid"] for r in ev["events"]],
        "Evaluation alignment mismatch",
    )
    joined = {
        u: dict(native=a, diagnostic=b)
        for u, a, b in zip(ids, ev["native"]["events"], ev["events"])
    }
    aggregate = legacy.aggregate(
        joined, {r["uid"]: r for r in rows}, continuum_membership
    )
    require(aggregate["counts"] == ev["native"]["counts"], "Aggregation counts differ")
    edges = []
    with gzip.open(ev["trace"]["path"], "rt") as stream:
        for row in rows:
            traced = json.loads(next(stream))
            require(traced["uid"] == row["uid"], "Trace UID mismatch")
            edges.append(
                dict(
                    uid=row["uid"],
                    category=row["category"],
                    fsp_cardinality=len(row["targets"]),
                    **metric.semantic_join(
                        traced["model"],
                        row["targets"].tolist(),
                        row["sources"].tolist(),
                    ),
                )
            )
        require(next(stream, None) is None, "Unexpected trace row")
    return dict(aggregate=aggregate, semantic_edges=edges)


def fresh_source_errors(left, right, correction):
    """Correct Phase80 counts, fresh preregistered bootstrap seed; no zero defaults."""
    import numpy as np

    require(
        [(r["uid"], r["category"]) for r in left]
        == [(r["uid"], r["category"]) for r in right],
        "Paired source UID mismatch",
    )
    selected = [i for i, r in enumerate(left) if r["category"] in CATEGORIES[:2]]
    cats = np.array([left[i]["category"] for i in selected])
    strata = [np.flatnonzero(cats == c) for c in sorted(set(cats))]
    out = {}
    for stage in ("proposal", "refinement"):
        rows = [
            [arm[i]["stages"][stage]["counts"] for i in selected]
            for arm in (left, right)
        ]
        for arm in rows:
            for counts in arm:
                correction.validate_counts(counts)
        for a, b in zip(*rows):
            require(
                all(
                    a[k] == b[k]
                    for k in ("B_nodes", "background_nodes", "unknown_nodes", "nodes")
                ),
                "Source support mismatch",
            )

        def values(key):
            return np.array([[r[key] for r in arm] for arm in rows], dtype=float)

        correct = values("B_correct")
        support = values("B_nodes")
        metrics = {
            "source_recall": (correct, support),
            "source_precision": (
                correct,
                correct + values("B_to_other_B") + values("background_to_B"),
            ),
            "missing_to_unassigned_fraction": (values("B_to_unassigned"), support),
            "missing_to_other_B_fraction": (values("B_to_other_B"), support),
        }
        out[stage] = {}
        for name, (num, den) in metrics.items():
            result = dict(
                numerators=num.sum(1).astype(int).tolist(),
                denominators=den.sum(1).astype(int).tolist(),
                unknown_nodes=values("unknown_nodes").sum(1).astype(int).tolist(),
                unknown_to_B=values("unknown_to_B").sum(1).astype(int).tolist(),
                paired_collisions=len(selected),
                seed=20261010087,
                resamples=2000,
            )
            if (den.sum(1) == 0).any():
                result.update(
                    status="UNAVAILABLE_ZERO_SUPPORT", difference=None, interval95=None
                )
            else:
                rng = np.random.default_rng(20261010087)
                draws = []
                for _ in range(2000):
                    ix = np.concatenate(
                        [rng.choice(s, len(s), replace=True) for s in strata]
                    )
                    ds = den[:, ix].sum(1)
                    if (ds > 0).all():
                        draws.append(float(np.diff(num[:, ix].sum(1) / ds)[0]))
                result.update(
                    status="AVAILABLE",
                    difference=float(np.diff(num.sum(1) / den.sum(1))[0]),
                    interval95=np.quantile(draws, [0.025, 0.975]).tolist()
                    if draws
                    else None,
                    available_resamples=len(draws),
                    limitation="Known-label precision; unknown assigned nodes unavailable. Whole-event fresh conditional bootstrap; sparse intervals do not establish equivalence.",
                )
            out[stage][name] = result
    return out


def fresh_gate(paired, reports):
    counts = [reports[a]["aggregate"]["counts"] for a in ARMS]
    coverage = all(
        c["events"] == 600
        and c["nominal_b_trials"] == 400
        and c["unavailable_membership_trials"] == 0
        for c in counts
    )
    category_net = all(
        reports[ARMS[1]]["aggregate"]["by_category"][cat][key]
        >= reports[ARMS[0]]["aggregate"]["by_category"][cat][key]
        for cat in CATEGORIES[:2]
        for key in ("raw_exact_memberships", "accepted_exact_memberships")
    )
    exact = all(
        paired["paired_effects"][k]["status"] == "AVAILABLE"
        and paired["paired_effects"][k]["difference"] > 0
        and paired["paired_effects"][k]["interval95"][0] > 0
        for k in ("raw_exact_memberships", "accepted_exact_memberships")
    )
    controls = [
        paired["paired_effects"]["continuum_accepted_events"],
        paired["b_event_background_fraction"],
        *paired["continuum_by_category"].values(),
    ]
    background = all(
        x["status"] == "AVAILABLE" and x["interval95"][1] <= 0.05 for x in controls
    )
    checks = dict(
        full_coverage=coverage,
        category_net_nonnegative=category_net,
        exact_paired_gain=exact,
        same_target_collision_gain=len(paired["gaining_same_target_collisions"]) >= 4,
        background=background,
        source_validity=all(c["accepted_source_conflicts"] == 0 for c in counts),
    )
    return dict(passed=all(checks.values()), checks=checks, primary_readiness=False)


def admission_controls(request):
    return {
        k: request[k]
        for k in (
            "source_sha",
            "source_hashes",
            "training_contract",
            "checkpoints",
            "original_cache",
            "normalizer_checkpoint",
            "metric_helper",
            "legacy_helper",
            "source_error_helper",
            "plan",
            "threshold",
            "bootstrap_seed",
            "bootstrap_replicates",
        )
    }


def resource_guard(started):
    require(
        time.monotonic() - started < 5400
        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 26 * 1024**2,
        "Evaluation resource stop",
    )


@contextmanager
def observed_audit(model, decoder, started):
    """Observed forwards and per-event deadline guard; never alter outputs."""
    counts = Counter()

    def watch(module, args):
        resource_guard(started)
        counts["decoder_calls"] += 1

    def encoder(module, args):
        resource_guard(started)
        counts["encoder_calls"] += 1

    handles = [
        decoder.register_forward_pre_hook(watch),
        model.encoder.register_forward_pre_hook(encoder),
    ]
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def evaluate(request, mode, output):
    from scripts.phase84_exposure import load_cache, nested
    from scripts.diagnose_phase80_optimizer_steps import audit

    start, cpu = time.monotonic(), time.process_time()
    training = read(request["training_contract"])
    require(training["source_sha"] == TRAINING_SOURCE, "Wrong training source")
    original, cache_admission = load_cache(Path(training["parent"]))
    require(
        cache_admission["cache"] == request["original_cache"], "Original cache changed"
    )
    checked(request["original_cache"])
    checked(request["plan"])
    metric = helper(request["metric_helper"], "fresh_bound_metrics")
    legacy = helper(request["legacy_helper"], "fresh_bound_legacy")
    correction = helper(request["source_error_helper"], "fresh_bound_source_errors")
    native_ratio = metric.paired_ratio

    def fresh_ratio(*args, **kwargs):
        value = native_ratio(*args, **kwargs, seed=20261010087)
        value["limitation"] = (
            "Fresh fixed-model assessment; whole-event correlation retained. Sparse/all-zero empirical bootstrap does not bound unseen risks or establish equivalence; no training-seed uncertainty."
        )
        return value

    metric.paired_ratio = fresh_ratio
    selected = nested(original["train"], 384)
    require(
        [r["uid"] for r in selected] == read(training["admission"])["selection"],
        "Original384 TRAIN selection changed",
    )
    if mode == "smoke":
        worst = max(original["train"], key=lambda r: len(r["targets"]))
        rows = [worst] + [r for r in selected if r["uid"] != worst["uid"]][:7]
        check_rows(rows, expected_uids=[r["uid"] for r in rows])
        from hypertagging.reconstruction.constraints import (
            ReconstructionConstraintPolicy,
        )

        policy = ReconstructionConstraintPolicy.from_dict(
            original["source_feature_contract"]["reconstruction_constraint_policy"]
        )
        for row in rows:
            projected = encode_normalized(
                row["full"], row["uid"], row["category"], policy
            )
            for key in ("detector", "sources", "charge", "targets", "supervision"):
                legacy.equal_tree(
                    projected[key], row[key], "canonical TRAIN replay:" + key
                )
    else:
        admitted = read(request["admission"])
        require(
            admitted["status"] == "PASS_TRAIN_ONLY_EVALUATION_SMOKE"
            and admitted["checkpoints"] == request["checkpoints"]
            and admitted["source_sha"] == request["source_sha"]
            and admitted["original_cache"] == request["original_cache"]
            and admitted["controls"] == admission_controls(request),
            "Stale evaluation admission",
        )
        resource_guard(start)
        rows = fresh_rows(request, original, legacy, output, start)
        resource_guard(start)
    reports, evaluations, timings, observed = {}, {}, {}, {}
    for arm in ARMS:
        model, decoder = restore(request, original, arm, legacy)
        then = time.monotonic()
        with observed_audit(model, decoder, start) as measured:
            ev = audit(model, decoder, rows, output / f"{arm}-model-only.jsonl.gz")
        observed[arm] = dict(
            measured,
            full_model_and_decoder_parameters=sum(
                p.numel() for module in (model, decoder) for p in module.parameters()
            ),
            node_squared_proxy=sum(len(r["targets"]) ** 2 for r in rows),
            processed_unique_identities=len({r["uid"] for r in rows}),
            optimizer_updates=0,
        )
        write(output / f"{arm}-audit.json", ev)
        reports[arm] = summarize_arm(ev, rows, metric, legacy)
        evaluations[arm] = ev
        timings[arm] = time.monotonic() - then
        require(
            all(math.isfinite(x) for x in ev["mean_risk"].values()),
            "Nonfinite evaluation",
        )
        require(
            time.monotonic() - start < 5400
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 26 * 1024**2,
            "Evaluation resource stop",
        )
        if mode == "smoke":
            with observed_audit(model, decoder, start) as repeat_counts:
                repeated = audit(
                    model,
                    decoder,
                    rows[:1],
                    output / f"{arm}-repeat-model-only.jsonl.gz",
                )
            observed[arm]["additional_smoke_repeat_calls"] = dict(repeat_counts)
            observed[arm]["primary_counts_exclude_repeat"] = True
            with (
                gzip.open(ev["trace"]["path"], "rt") as first,
                gzip.open(repeated["trace"]["path"], "rt") as second,
            ):
                require(
                    json.loads(next(first)) == json.loads(next(second)),
                    "Deterministic final-checkpoint repeat mismatch",
                )
            require(
                all(
                    p.grad is None and not p.requires_grad
                    for module in (model, decoder)
                    for p in module.parameters()
                ),
                "Evaluation gradients enabled",
            )
        del model, decoder
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(
        rss < (24 if mode == "smoke" else 26) * 1024**2,
        "Evaluation memory admission exceeded",
    )
    if mode == "smoke":
        forecast = 2 * sum(timings.values()) / len(rows) * 600 + 1200
        require(forecast < 5400, "Evaluation runtime forecast exceeds cap")
        write(
            output / "admission.json",
            dict(
                status="PASS_TRAIN_ONLY_EVALUATION_SMOKE",
                source_sha=request["source_sha"],
                checkpoints=request["checkpoints"],
                original_cache=request["original_cache"],
                controls=admission_controls(request),
                request=request["request_binding"],
                job_id=os.environ["SLURM_JOB_ID"],
                rows=len(rows),
                heldout_events=0,
                optimizer_updates=0,
                forecast_seconds=forecast,
                peak_rss_kib=rss,
                arm_seconds=timings,
                observed_compute=observed,
            ),
        )
    else:
        # The legacy candidate method supplies paired statistics only; TRAIN gate labels
        # are removed, and fresh acceptance is determined by separately frozen plan.
        paired = metric.paired_gate(
            reports[ARMS[0]]["aggregate"]["events"],
            reports[ARMS[1]]["aggregate"]["events"],
            False,
            legacy,
        )
        paired = {
            k: v
            for k, v in paired.items()
            if k not in ["exact_gate", "continuum_gate", "fixed6000_eligible"]
        }
        diagnostics = [evaluations[a]["events"] for a in ARMS]
        b_rows = [
            [e for e in d if e["category"] in CATEGORIES[:2]] for d in diagnostics
        ]
        require(
            [e["uid"] for e in b_rows[0]] == [e["uid"] for e in b_rows[1]],
            "Paired background UID mismatch",
        )
        background = [[e["stages"]["refinement"]["counts"] for e in r] for r in b_rows]
        require(
            all(
                x["background_nodes"] == y["background_nodes"]
                for x, y in zip(*background)
            ),
            "Background support mismatch",
        )
        paired["b_event_background_fraction"] = metric.paired_ratio(
            [e["background_to_B"] for e in background[0]],
            [e["background_to_B"] for e in background[1]],
            [e["background_nodes"] for e in background[0]],
            [e["category"] for e in b_rows[0]],
        )
        paired["source_errors"] = fresh_source_errors(
            *diagnostics, correction=correction
        )
        paired["fresh_development_gate"] = fresh_gate(paired, reports)
        write(
            output / "summary.json",
            dict(
                status="COMPLETED_FIXED_FRESH_DEVELOPMENT",
                arms=reports,
                paired=paired,
                stage="development",
                heldout_events=600,
                optimizer_updates=0,
                source_sha=request["source_sha"],
                training_source_sha=TRAINING_SOURCE,
                checkpoints=request["checkpoints"],
                identity_admission=request["data_admission"],
                primary_coverage_met=False,
                physical_tree_beam_p4="UNAVAILABLE_FLAT_OPTIONAL_B",
                interpretation="Fresh fixed-checkpoint paired assessment, not complete primary. Fixed preregistered gate; no threshold tuning. Previous TRAIN exploration is not independent confirmation.",
                compute=dict(
                    wall_seconds=time.monotonic() - start,
                    cpu_seconds=time.process_time() - cpu,
                    peak_rss_kib=rss,
                    arm_seconds=timings,
                    observed_compute=observed,
                    event_model_views=2 * len(rows),
                    equal_FLOPs_claim=False,
                ),
            ),
        )
    write(
        output / "terminal.json",
        dict(
            status="COMPLETED",
            mode=mode,
            bindings=[binding(f) for f in sorted(output.iterdir()) if f.is_file()],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "evaluate"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scripts.phase76_development import guarded
    from scripts.run_phase74_development import verify_bindings

    guarded()
    request = json.loads(args.request.read_text())
    validate_request(request, args.mode)
    require(
        request["source_root"] == str(ROOT)
        and subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        == request["source_sha"],
        "Evaluator source mismatch",
    )
    verify_bindings(request, ROOT)
    request["request_binding"] = binding(args.request)
    require(
        str(args.output.resolve()) == request["outputs"][args.mode],
        "Unbound output path",
    )
    claim_output(args.output)
    evaluate(request, args.mode, args.output.resolve())


if __name__ == "__main__":
    main()
