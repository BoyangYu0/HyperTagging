"""Sequential, training-only exposure trajectories with immutable milestone audits."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, write, read  # noqa:E402
from scripts.phase76_development import guarded, initial, settings  # noqa:E402
from scripts.phase81_pair_supervision import (  # noqa:E402
    authenticate_training_rows,
    initial_state_digest,
)  # noqa:E402
from scripts.run_phase74_development import fit, save, verify_bindings  # noqa:E402

CATEGORIES = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
MILESTONES = (375, 1500, 3000, 6000)
REPLICATION_SOURCE = "1b4fe6b3a54c45981d87af5f464b69a9c63bb430"
REPLICATION_INITIAL_DIGEST = (
    "3e8b9ee3bd144d1d836dcb1372dcfe1b012d867af456e1fd4df643c869db0c01"
)
REPLICATION_REFERENCE_HASHES = {
    "reference_contract": "a31ade1e9ccdf827901376ac2cda6d6f12b79d882ead04156c8fb60bfd7034bd",
    "reference_startup": "3f43edd66f6f62f25ab051390d796fd7115467b2f6ea23f896dfe998b11c37b5",
    "reference_terminal_review": "10396dad02625441645f417b584c12b47712486975f74252b44e36a90d57163e",
}


def validate_replication(c):
    """One explicit training-order replication of immutable original384 evidence."""
    r = c["replication"]
    controls = dict(
        kind="training_order",
        reference_pool=384,
        reference_sampler_seed=202610081,
        new_sampler_seed=202610082,
        reference_source_sha=REPLICATION_SOURCE,
        expected_initial_digest=REPLICATION_INITIAL_DIGEST,
    )
    if not isinstance(r, dict) or set(r) != set(controls) | set(
        REPLICATION_REFERENCE_HASHES
    ):
        raise ValueError("Replication contract fields changed")
    if any(r[k] != v for k, v in controls.items()) or c["pool_size"] != 384:
        raise ValueError("Replication controls changed")
    documents = {}
    for name, digest in REPLICATION_REFERENCE_HASHES.items():
        item = r[name]
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or item.get("sha256") != digest
            or item not in c.get("bindings", [])
            or binding(Path(item["path"])) != item
        ):
            raise ValueError("Unbound or changed replication reference: " + name)
        documents[name] = read(item)
    original = documents["reference_contract"]
    startup = documents["reference_startup"]
    review = documents["reference_terminal_review"]
    if (
        original.get("replication") is not None
        or original.get("source_sha") != REPLICATION_SOURCE
        or original.get("pool_size") != 384
        or original.get("settings") != {**settings(), "downstream_updates": 6000}
        or any(c.get(k) != original.get(k) for k in ("parent", "pretrain"))
        or startup.get("contract") != r["reference_contract"]
        or startup.get("source_sha") != REPLICATION_SOURCE
        or startup.get("pool") != 384
        or startup.get("initial_digest") != REPLICATION_INITIAL_DIGEST
        or review.get("status") != "PASS"
        or review.get("source_sha") != REPLICATION_SOURCE
        or review.get("pool") != 384
        or review.get("review_scheduler") != "COMPLETED|0:0"
        or any(
            review.get("gate", {}).get(k) is not True
            for k in ("absolute_trainable", "exposure_gain", "fixed6000_eligible")
        )
    ):
        raise ValueError("Replication reference history/gates changed")
    return startup


def validate_replication_initial(c, digest, checkpoint, uid_order_digest):
    """Checked before fit creates its fresh optimizer; never resume original weights."""
    startup = validate_replication(c)
    if (
        digest != REPLICATION_INITIAL_DIGEST
        or checkpoint != startup["initial_checkpoint"]
        or uid_order_digest != startup["uid_order_sha256"]
    ):
        raise ValueError("Replication initialization/checkpoint/pool order changed")
    return {
        **c["replication"],
        "initial_digest_verified": digest,
        "fresh_optimizer": True,
        "rng_scope": "Only Python training-row sampler seed changes; initializer and Torch/generated-state RNG rules unchanged.",
    }


def nested(rows, size):
    if size not in (96, 384, 1536) or len(rows) != 1536:
        raise ValueError("Unsupported pool")
    if len({r["uid"] for r in rows}) != 1536 or Counter(
        r["category"] for r in rows
    ) != dict.fromkeys(CATEGORIES, 256):
        raise ValueError("Duplicate or wrong category support")
    keep = set()
    for cat in CATEGORIES:
        rr = sorted(
            (r for r in rows if r["category"] == cat),
            key=lambda r: hashlib.sha256(
                ("phase84-nested-20261009:" + r["uid"]).encode()
            ).hexdigest(),
        )
        keep.update(r["uid"] for r in rr[: size // 6])
    return [r for r in rows if r["uid"] in keep]


def exposure(rows, steps, seed):
    rng = random.Random(seed)
    counts = Counter()
    digest = hashlib.sha256()
    for _ in range(steps * 8):
        uid = rows[rng.randrange(len(rows))]["uid"]
        counts[uid] += 1
        digest.update((uid + "\n").encode())
    return {
        "presentations": steps * 8,
        "unique_sampled": len(counts),
        "eligible": len(rows),
        "per_uid": {r["uid"]: counts[r["uid"]] for r in rows},
        "per_category": dict(
            Counter(
                {
                    c: sum(counts[r["uid"]] for r in rows if r["category"] == c)
                    for c in CATEGORIES
                }
            )
        ),
        "sequence_sha256": digest.hexdigest(),
    }


def plateau(curves):
    # No futility stop on zero exact alone. Require three risk plateaus AND
    # >=95% raw fit success; otherwise complete the registered6000 updates.
    if len(curves) < 3:
        return False
    a, b, c = curves[-3:]
    risks = [x["mean_risk"]["member"] for x in (a, b, c)]
    counts = [x["native"]["counts"] for x in (a, b, c)]
    return (
        all(
            x["raw_exact_memberships"] / max(1, x["nominal_b_trials"]) >= 0.95
            for x in counts
        )
        and all(
            abs(y - x) <= 0.001 * max(abs(x), 1e-8) for x, y in zip(risks, risks[1:])
        )
        and len({x["raw_exact_memberships"] for x in counts}) == 1
    )


def validate(c):
    if (
        c["kind"] != "phase84_exposure"
        or c["pool_size"] not in (96, 384, 1536)
        or c["milestones"] != list(MILESTONES)
    ):
        raise ValueError("Study shape changed")
    expected_settings = {**settings(), "downstream_updates": 6000}
    if "replication" in c:
        validate_replication(c)
        expected_settings["seed"] = 202610082
    if c["settings"] != expected_settings or c["resources"] != {
        "cpus": 2,
        "memory_gib": 32,
        "hours": 8,
        "gpus": 0,
        "requeue": False,
    }:
        raise ValueError("Settings/resources changed")
    if c["heldout_events"] or c["automatic_successor"] or c["sealed_test_access"]:
        raise ValueError("Data/authority changed")


def load_cache(parent):
    import torch

    a = json.loads((parent / "train-admission.json").read_text())
    verify_bindings(
        {
            "source_hashes": {},
            "bindings": [
                binding(parent / "train-admission.json"),
                a["cache"],
                *a["bindings"],
            ],
        },
        ROOT,
    )
    c = torch.load(a["cache"]["path"], map_location="cpu", weights_only=False)
    if set(c) != {"train", "runtime_normalizer"}:
        raise ValueError("Nontraining cache")
    authenticate_training_rows(c["train"], read(a["training"]))
    from scripts.phase84_train_isolation import authenticate_train_isolation

    a["current_isolation"] = authenticate_train_isolation(
        [r["uid"] for r in c["train"]], ROOT
    )
    return c, a


def strata(rows):
    return [
        {
            "uid": r["uid"],
            "category": r["category"],
            "fsp_count": len(r["targets"]),
            "b_sizes": [int((r["targets"] == i).sum()) for i in (1, 2)],
            "unknown": int((r["targets"] < 0).sum()),
            "nodes_with_source_support": int(r["sources"].any(-1).sum()),
            "distinct_detector_sources": int(r["sources"].any(0).sum()),
        }
        for r in rows
    ]


def smoke(root, parent, pretrain):
    guarded()
    from scripts.diagnose_phase80_optimizer_steps import audit

    cache, a = load_cache(parent)
    out = root / "admission-v1"
    out.mkdir(exist_ok=False)
    memberships = {
        str(n): [r["uid"] for r in nested(cache["train"], n)] for n in (96, 384, 1536)
    }
    write(
        out / "selection-private.json",
        {
            "algorithm": "phase84-nested-20261009 SHA256 per category; stable original loader order",
            "memberships": memberships,
            "strata": strata(cache["train"]),
            "role": "train",
        },
    )
    worst = max(cache["train"], key=lambda r: len(r["targets"]))
    timings = {}
    for size in (96, 1536):
        model, decoder, cp = initial(pretrain, cache)
        run = out / str(size)
        run.mkdir()
        rows = nested(cache["train"], size)
        probe = [worst] + [rows[i] for i in range(7)]
        cfg = {**settings(), "downstream_updates": 4}
        compute = Counter()
        start = time.monotonic()
        gradient_receipt = {}

        def check_gradients(**kw):
            if kw["step"] == 1:
                for name, module in [
                    ("encoder", model.encoder),
                    ("proposal", decoder.proposal),
                    ("refinement", decoder.refinement),
                    ("condition", decoder.condition),
                ]:
                    gradient_receipt[name] = sum(
                        float(p.grad.abs().sum())
                        for p in module.parameters()
                        if p.grad is not None
                    )
                if not all(v > 0 for v in gradient_receipt.values()):
                    raise ValueError("Missing real gradient branch")
            return False

        result = fit(
            model,
            decoder,
            probe,
            stage="downstream",
            objective="existing",
            settings=cfg,
            output=run,
            compute=compute,
            cache_binding=a["cache"],
            source_sha="SMOKE_ONLY",
            step_callback=check_gradients,
        )
        elapsed = time.monotonic() - start
        # Size-mixed measured fit and historical native timing; conservative2x.
        forecast = elapsed / 4 * 6000 * 2 + 1200
        t = time.monotonic()
        audit(model, decoder, rows[:8], run / "model-only.jsonl.gz")
        eval_per = (time.monotonic() - t) / 8
        forecast += eval_per * size * 4 * 2
        timings[str(size)] = {
            "fit_seconds": elapsed,
            "fit_updates": 4,
            "evaluation_seconds_per_event": eval_per,
            "forecast_seconds": forecast,
            "history": result,
            "gradient_checks": gradient_receipt,
        }
        if (
            not math.isfinite(forecast)
            or forecast > 7.0 * 3600
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 24 * 1024**2
        ):
            raise ValueError("Resource admission failed")
    write(
        out / "admission.json",
        {
            "status": "PASS",
            "job_id": os.environ["SLURM_JOB_ID"],
            "cache": a["cache"],
            "training_admission": binding(parent / "train-admission.json"),
            "isolation": a["current_isolation"],
            "initial_checkpoint": cp,
            "selection": binding(out / "selection-private.json"),
            "timings": timings,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "real_gradient_checks": "native fit finite/nonzero encoder plus joint head updates",
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )


def run(path):
    guarded()
    import torch
    from scripts.diagnose_phase80_optimizer_steps import audit

    c = json.loads(path.read_text())
    validate(c)
    verify_bindings(c, ROOT)
    if (
        str(ROOT) != c["source_root"]
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Source root/revision mismatch")
    admission = read(c["admission"])
    assert admission["status"] == "PASS"
    cache, a = load_cache(Path(c["parent"]))
    rows = nested(cache["train"], c["pool_size"])
    selection = read(admission["selection"])
    if [r["uid"] for r in rows] != selection["memberships"][str(c["pool_size"])]:
        raise ValueError("Selection changed")
    out = Path(c["output"])
    out.mkdir(exist_ok=False)
    write(out / "isolation.json", a["current_isolation"])
    model, decoder, cp = initial(Path(c["pretrain"]), cache)
    replication_metadata = {}
    if "replication" in c:
        replication_metadata["replication"] = validate_replication_initial(
            c,
            initial_state_digest(model, decoder),
            cp,
            hashlib.sha256("\n".join(r["uid"] for r in rows).encode()).hexdigest(),
        )
        replication_metadata["replication"].update(
            initial_torch_rng_sha256=hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            initial_sampler_rng_sha256=hashlib.sha256(
                repr(random.Random(c["settings"]["seed"]).getstate()).encode()
            ).hexdigest(),
        )
    write(
        out / "startup.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "contract": binding(path),
            "source_sha": c["source_sha"],
            "initial_checkpoint": cp,
            "initial_digest": initial_state_digest(model, decoder),
            "pool": c["pool_size"],
            "uid_order_sha256": hashlib.sha256(
                "\n".join(r["uid"] for r in rows).encode()
            ).hexdigest(),
            **replication_metadata,
        },
    )
    start = time.monotonic()
    cpu = time.process_time()
    compute = Counter()
    curves = []
    reason = "fixed6000"
    snapshots = []

    def callback(**kw):
        nonlocal reason
        step = kw["step"]
        elapsed = time.monotonic() - start
        if (
            elapsed > 7.25 * 3600
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            reason = "runtime_or_memory_guard"
            return True
        if step not in MILESTONES:
            return False
        rngstate = torch.get_rng_state()
        mode = (model.training, decoder.training)
        state = {
            "model_state_dict": model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": kw["optimizer"].state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": None,
            "torch_rng_state": rngstate,
            "sampling_rng_state": kw["rng"].getstate(),
            "step": step,
            "source_sha": c["source_sha"],
            "contract": binding(path),
            "data_order_sha256": kw["sequence_sha256"],
            "normalizer": model.runtime_feature_normalizer,
            "settings": c["settings"],
            "architecture": {"context": 128, "hyperbolic": 32, "head": 256, "depth": 4},
            "resume_authorized": False,
        }
        save(out / f"step-{step}.pt", state)
        result = audit(model, decoder, rows, out / f"step-{step}-model-only.jsonl.gz")
        if not all(math.isfinite(v) for v in result["mean_risk"].values()):
            raise FloatingPointError("Nonfinite audit risk")
        write(out / f"step-{step}-evaluation.json", result)
        sample = exposure(rows, step, c["settings"]["seed"])
        assert sample["sequence_sha256"] == kw["sequence_sha256"]
        write(out / f"step-{step}-exposure-private.json", sample)
        curves.append(result)
        snapshots.append(
            {
                "step": step,
                "counts": result["native"]["counts"],
                "mean_risk": result["mean_risk"],
                "checkpoint": binding(out / f"step-{step}.pt"),
                "evaluation": binding(out / f"step-{step}-evaluation.json"),
            }
        )
        compute["evaluation_event_views"] += len(rows)
        compute["evaluation_encoder_passes"] += 2 * len(rows)
        compute["evaluation_node_squared_proxy"] += sum(
            len(r["targets"]) ** 2 for r in rows
        )
        torch.set_rng_state(rngstate)
        model.train(mode[0])
        decoder.train(mode[1])
        if (
            time.monotonic() - start > 7.25 * 3600
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            reason = "runtime_or_memory_guard"
            return True
        if step < 6000 and plateau(curves):
            reason = "saturated95percent_plateau"
            return True
        return False

    history = fit(
        model,
        decoder,
        rows,
        stage="downstream",
        objective="existing",
        settings=c["settings"],
        output=out,
        compute=compute,
        cache_binding=a["cache"],
        source_sha=c["source_sha"],
        step_callback=callback,
    )
    write(
        out / "summary.json",
        {
            "status": "COMPLETED",
            **replication_metadata,
            "stop_reason": reason,
            "candidate_eligible_for_preregistered_replication": reason == "fixed6000"
            and history["updates"] == 6000,
            "source_sha": c["source_sha"],
            "history": history,
            "pool_size": len(rows),
            "snapshots": snapshots,
            "compute": {
                **compute,
                "wall_seconds": time.monotonic() - start,
                "cpu_seconds": time.process_time() - cpu,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "parameters": sum(p.numel() for p in model.encoder.parameters())
                + sum(p.numel() for p in decoder.parameters()),
            },
            "validation_events": 0,
            "physical_tree_beam_p4_metrics": "UNAVAILABLE_FLAT_OPTIONAL_B",
            "final_candidate_rule": "Only fixed6000 checkpoint eligible; plateau/runtime stop diagnostic only. No heldout selection.",
        },
    )
    write(
        out / "terminal.json",
        {
            "status": "COMPLETED",
            "bindings": [binding(p) for p in sorted(out.iterdir()) if p.is_file()],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["smoke", "run"])
    p.add_argument("--root", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--pretrain", type=Path)
    p.add_argument("--contract", type=Path)
    a = p.parse_args()
    if a.action == "smoke":
        smoke(a.root, a.parent, a.pretrain)
    else:
        run(a.contract)
