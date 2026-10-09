"""Guarded matched pair-supervision training screen; no automatic heldout use."""

from __future__ import annotations
import argparse
from collections import Counter
import copy
import getpass
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, sha, read, write  # noqa:E402
from scripts.phase76_development import guarded, initial, settings, INITIAL_SHA256  # noqa:E402
from scripts.run_phase74_development import fit, save, verify_bindings  # noqa:E402
from scripts.slurm.submit_phase74_development import acquire_submission_lock  # noqa:E402

ARMS = ("native", "pair_supervised")
RESOURCES = dict(cpus=2, memory_gib=32, hours=8, gpus=0, max_jobs=2, requeue=False)
PLAN = "configs/reconstruction/phase81_pair_supervision_plan.json"


def source_hashes():
    files = subprocess.check_output(
        ["git", "ls-files", "src", "scripts", "configs", "uv.lock", "AGENTS.md"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return {p: sha(ROOT / p) for p in files}


def validate(c):
    if (
        c["arms"] != list(ARMS)
        or c["settings"] != settings()
        or c["resources"] != RESOURCES
    ):
        raise ValueError("Arms/settings/resources changed")
    if c["initial_checkpoint"]["sha256"] != INITIAL_SHA256:
        raise ValueError("Incompatible history")
    if (
        c["stage"] != "development_training_screen"
        or c["heldout_count"] != 0
        or c["training_count"] != 1536
    ):
        raise ValueError("Training-only contract changed")
    if (
        c["automatic_successor"]
        or c["sealed_test_access"]
        or c["arm_pair_supervision"] != {"native": False, "pair_supervised": True}
    ):
        raise ValueError("Scientific contrast/authority changed")


def prepare(root, parent):
    guarded()
    import torch

    old = Path(parent)
    ca = json.loads((old / "cache-admission.json").read_text())
    data = read(ca["data_admission"])
    if (
        data["status"] != "PASS"
        or ca["status"] != "PASS"
        or data["original_corpus_count"] != 70000
        or ca["capacity_dropped_events"]
    ):
        raise ValueError("Parent admission failed")
    training = read(data["training"])
    if training["role"] != "train" or len(training["event_uids"]) != 1536:
        raise ValueError("Not authenticated train role")
    # Reauthenticate the original index/selection and train shards; no new cohort.
    bindings = [
        binding(old / "cache-admission.json"),
        ca["data_admission"],
        data["training"],
        ca["cache"],
    ]
    bindings += list(data["inputs"].values()) + [
        s for s in data["shards"] if s["role"] == "train"
    ]
    verify_bindings({"source_hashes": {}, "bindings": bindings}, ROOT)
    cached = torch.load(ca["cache"]["path"], map_location="cpu", weights_only=False)
    rows = cached["train"]
    if [r["uid"] for r in rows] != training["event_uids"] or len(
        set(r["uid"] for r in rows)
    ) != 1536:
        raise ValueError("Cache/cohort/order mismatch")
    if Counter(r["category"] for r in rows) != {
        k: 256 for k in ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
    }:
        raise ValueError("Category coverage changed")
    # Historical serialized cache is read only to extract train data; development
    # rows are never iterated, scored, selected, or copied into the new cache.
    cache = {"train": rows, "runtime_normalizer": cached["runtime_normalizer"]}
    save(root / "train-only-cache.pt", cache)
    write(
        root / "train-admission.json",
        {
            "status": "PASS",
            "cache": binding(root / "train-only-cache.pt"),
            "bindings": bindings,
            "training": data["training"],
            "training_count": 1536,
            "heldout_count": 0,
            "max_fsp": max(len(r["targets"]) for r in rows),
            "normalization": ca["normalization"],
            "prior_cache_contains_historical_development_but_not_used": True,
        },
    )


def smoke(root, parent):
    guarded()
    import torch
    from hypertagging.training.capacity_development import detector_features
    from hypertagging.training.pair_membership import proposal_refinement_pair_loss

    admission = json.loads((root / "train-admission.json").read_text())
    verify_bindings({"source_hashes": {}, "bindings": [admission["cache"]]}, ROOT)
    cache = torch.load(
        admission["cache"]["path"], map_location="cpu", weights_only=False
    )
    out = root / "smoke-v1"
    out.mkdir(exist_ok=False)
    worst = max(cache["train"], key=lambda r: len(r["targets"]))
    results = {}
    for arm in ARMS:
        model, decoder, checkpoint = initial(parent, cache)
        b = next(
            r
            for r in cache["train"]
            if all((r["targets"] == i).any() for i in range(3))
        )
        h, _, _ = detector_features(model, b["detector"])
        prediction = decoder(h, b["sources"])
        pair, support = proposal_refinement_pair_loss(
            prediction, b["targets"], b["sources"]
        )
        pair.backward()
        gradients = {
            name: sum(
                float(p.grad.abs().sum()) for p in parameters if p.grad is not None
            )
            for name, parameters in [
                ("encoder", model.encoder.parameters()),
                ("proposal", decoder.proposal.parameters()),
                ("refinement", decoder.refinement.parameters()),
            ]
        }
        gradients["predicted_context"] = float(
            decoder.condition[0].weight.grad[:, 256:512].abs().sum()
        )
        if not all(v > 0 for v in gradients.values()) or not all(
            support[k] > 0 for k in ("same_b", "cross_b", "background")
        ):
            raise ValueError("Pair branches/target support absent")
        # Reset exactly after diagnostic gradient; no smoke parameter transfer.
        model, decoder, checkpoint = initial(parent, cache)
        cfg = {
            **settings(),
            "tiny_updates": 2,
            "downstream_updates": 2,
            "batch_size": 2,
        }
        run = out / arm
        run.mkdir()
        compute = Counter()
        start = time.monotonic()
        for stage in ("tiny", "downstream"):
            fit(
                model,
                decoder,
                [worst],
                stage=stage,
                objective="existing",
                settings=cfg,
                output=run,
                compute=compute,
                cache_binding=admission["cache"],
                source_sha="SMOKE_ONLY",
                pair_supervision=arm == "pair_supervised",
            )
        elapsed = time.monotonic() - start
        forecast = elapsed / 8 * 20000 * 1.5
        if (
            forecast > 8 * 3600
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 24 * 1024**2
        ):
            raise ValueError("Scientific resource forecast exceeds guard")
        results[arm] = {
            "gradient_checks": gradients,
            "pair_support": support,
            "compute": dict(compute),
            "seconds": elapsed,
            "guarded_seconds_forecast": forecast,
        }
    write(
        out / "admission.json",
        {
            "status": "PASS",
            "cache": admission["cache"],
            "initial_checkpoint": checkpoint,
            "source_hashes": source_hashes(),
            "results": results,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "job_id": os.environ["SLURM_JOB_ID"],
        },
    )


def freeze(root, parent):
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Source must be clean")
    sm = json.loads((root / "smoke-v1/admission.json").read_text())
    data = json.loads((root / "train-admission.json").read_text())
    if sm["status"] != "PASS" or sm["source_hashes"] != source_hashes():
        raise ValueError("Stale smoke")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_next_reconstruction_study.py"),
            "--plan",
            str(ROOT / PLAN),
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source = root / "source"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), head], cwd=ROOT, check=True
    )
    campaign = root / "campaign-v1"
    campaign.mkdir()
    (campaign / "runs").mkdir()
    c = {
        "version": "phase81-pair-training-screen-v1",
        "stage": "development_training_screen",
        "source_sha": head,
        "source_root": str(source),
        "source_hashes": source_hashes(),
        "arms": list(ARMS),
        "arm_pair_supervision": {"native": False, "pair_supervised": True},
        "settings": settings(),
        "resources": RESOURCES,
        "training_count": 1536,
        "heldout_count": 0,
        "automatic_successor": False,
        "sealed_test_access": False,
        "parent": str(parent),
        "cache": sm["cache"],
        "initial_checkpoint": sm["initial_checkpoint"],
        "output_root": str(campaign / "runs"),
        "bindings": [
            binding(root / "smoke-v1/admission.json"),
            binding(root / "train-admission.json"),
            binding(root / "preregistered-plan.json"),
            binding(source / PLAN),
            sm["cache"],
            sm["initial_checkpoint"],
            *data["bindings"],
        ],
    }
    validate(c)
    verify_bindings(c, source)
    write(campaign / "contract.json", c)
    write(
        campaign / "admission.json",
        {
            "status": "PASS",
            "contract": binding(campaign / "contract.json"),
            "training_only": True,
            "heldout_evaluation_authorized_by_this_contract": False,
        },
    )


def submit(path):
    c = json.loads(path.read_text())
    validate(c)
    verify_bindings(c, Path(c["source_root"]))
    a = json.loads((path.parent / "admission.json").read_text())
    if a["status"] != "PASS" or a["contract"] != binding(path):
        raise ValueError("Admission mismatch")
    queue = subprocess.check_output(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%j|%T"],
        text=True,
    )
    if queue.strip():
        raise ValueError("Queue must be empty before this study")
    if list(path.parent.glob("*-submission.json")):
        raise ValueError("Prior submission")
    acquire_submission_lock(path.parent / "submission-lock.json", sha(path))
    for arm in ARMS:
        cmd = [
            "sbatch",
            "--parsable",
            "--export=NIL",
            "--chdir=" + c["source_root"],
            "--job-name=phase81-train-" + arm,
            "--output=" + str(path.parent / (arm + "-%j.log")),
            str(
                Path(c["source_root"])
                / "scripts/slurm/run_phase81_pair_supervision.sbatch"
            ),
            str(path),
            arm,
        ]
        job = subprocess.check_output(cmd, text=True).strip()
        assert job.split(";")[0].isdigit()
        write(
            path.parent / (arm + "-submission.json"),
            {
                "job_id": job,
                "arm": arm,
                "contract": binding(path),
                "source_sha": c["source_sha"],
                "command": cmd,
            },
        )
        print(arm, job, flush=True)


def run(path, arm):
    guarded()
    import torch
    from hypertagging.training.capacity_development import fresh_decoder
    from scripts.diagnose_phase80_optimizer_steps import audit

    c = json.loads(path.read_text())
    validate(c)
    verify_bindings(c, ROOT)
    if str(ROOT) != c["source_root"] or arm not in ARMS:
        raise ValueError("Source/arm mismatch")
    a = json.loads((path.parent / "admission.json").read_text())
    if a["status"] != "PASS" or a["contract"] != binding(path):
        raise ValueError("Admission mismatch")
    out = Path(c["output_root"]) / arm
    out.mkdir(exist_ok=False)
    cache = torch.load(c["cache"]["path"], map_location="cpu", weights_only=False)
    if set(cache) != {"train", "runtime_normalizer"}:
        raise ValueError("Validation data in training screen")
    model, decoder, cp = initial(Path(c["parent"]), cache)
    assert cp == c["initial_checkpoint"]
    write(
        out / "startup.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "source_sha": c["source_sha"],
            "contract": binding(path),
            "arm": arm,
            "initial_checkpoint": cp,
        },
    )
    tiny = [
        r
        for cat in ("charged", "mixed", "ccbar")
        for r in [r for r in cache["train"] if r["category"] == cat][:8]
    ]
    compute = Counter()
    start = time.monotonic()
    cpu = time.process_time()
    histories = {}
    results = {}
    for stage, rows, m, d in [
        ("tiny", tiny, copy.deepcopy(model), fresh_decoder()),
        ("downstream", cache["train"], model, decoder),
    ]:
        histories[stage] = fit(
            m,
            d,
            rows,
            stage=stage,
            objective="existing",
            settings=c["settings"],
            output=out,
            compute=compute,
            cache_binding=c["cache"],
            source_sha=c["source_sha"],
            pair_supervision=c["arm_pair_supervision"][arm],
        )
        m.eval()
        d.eval()
        result = audit(m, d, rows, out / (stage + "-model-only.jsonl.gz"))
        write(out / (stage + "-evaluation.json"), result)
        results[stage] = result["native"]
        rng = random.Random(c["settings"]["seed"])
        sampled = [
            rows[rng.randrange(len(rows))]["uid"]
            for _ in range(
                c["settings"][stage + "_updates"] * c["settings"]["batch_size"]
            )
        ]
        histories[stage]["unique_sampled_identities"] = len(set(sampled))
        compute["evaluation_event_views"] += len(rows)
        compute["evaluation_node_squared_proxy"] += sum(
            len(r["targets"]) ** 2 for r in rows
        )
    compute.update(
        wall_seconds=time.monotonic() - start,
        cpu_seconds=time.process_time() - cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        decoder_parameters=sum(p.numel() for p in decoder.parameters()),
    )
    write(
        out / "summary.json",
        {
            "arm": arm,
            "stage": c["stage"],
            "source_sha": c["source_sha"],
            "histories": histories,
            "tiny": results["tiny"],
            "train": results["downstream"],
            "heldout": {
                "status": "NOT_DESIGNATED_OR_EVALUATED",
                "reason": "Training-stage screen before fresh validation",
            },
            "compute": dict(compute),
            "primary_eligible": False,
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
    p.add_argument("action", choices=["prepare", "smoke", "freeze", "submit", "run"])
    p.add_argument("--root", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--contract", type=Path)
    p.add_argument("--arm")
    a = p.parse_args()
    if a.action in ("prepare", "smoke", "freeze"):
        globals()[a.action](a.root.resolve(), a.parent.resolve())
    elif a.action == "submit":
        submit(a.contract.resolve())
    else:
        run(a.contract.resolve(), a.arm)
