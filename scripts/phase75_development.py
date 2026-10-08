"""Guarded one-shot two-arm gradient-routing development, no automatic successor."""

from __future__ import annotations
import argparse
from collections import Counter
import copy
import getpass
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, sha, write, read  # noqa: E402
from scripts.run_phase74_development import fit, evaluate, verify_bindings  # noqa: E402
from scripts.slurm.submit_phase74_development import acquire_submission_lock  # noqa: E402

ARMS = ("joint", "project_conflicting_relation")


def guarded():
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 2
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        or os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    ):
        raise RuntimeError("Fresh guarded two-CPU allocation required")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def initial(parent, cache):
    import torch
    from hypertagging.training.capacity_development import fresh_model, fresh_decoder

    model = fresh_model(128, cache["runtime_normalizer"])
    checkpoint = parent / "campaign-v1/runs/128-existing/pretraining-final.pt"
    terminal = json.loads((checkpoint.parent / "terminal.json").read_text())
    b = next(x for x in terminal["checkpoints"] if x["path"] == str(checkpoint))
    if sha(checkpoint) != b["sha256"]:
        raise ValueError("Initial checkpoint changed")
    model.load_state_dict(
        torch.load(checkpoint, weights_only=False, map_location="cpu")[
            "model_state_dict"
        ]
    )
    return model, fresh_decoder(), b


def settings():
    return {
        "pretraining_updates": 0,
        "tiny_updates": 1000,
        "downstream_updates": 1500,
        "batch_size": 8,
        "seed": 202610081,
        "checkpoint": "fixed_final",
        "threshold": 0.5,
        "pretraining_lr": 0.0003,
        "encoder_lr": 0.00005,
        "head_lr": 0.001,
    }


def validate(c):
    if (
        c["arms"] != list(ARMS)
        or c["settings"] != settings()
        or c["stage"] != "development"
        or c["automatic_successor"]
        or c["sealed_test_access"]
    ):
        raise ValueError("Study controls changed")
    if c["resources"] != {
        "cpus": 2,
        "memory_gib": 32,
        "hours": 8,
        "gpus": 0,
        "max_jobs": 2,
        "requeue": False,
    }:
        raise ValueError("Resource bounds changed")
    if (
        c["training_count"] != 1536
        or c["heldout_count"] != 600
        or c["context_width"] != 128
    ):
        raise ValueError("Data/capacity contract changed")


def smoke(root, parent):
    guarded()
    import torch

    cache_binding = json.loads((root / "cache-admission.json").read_text())["cache"]
    cache = torch.load(cache_binding["path"], weights_only=False, map_location="cpu")
    old = torch.load(parent / "data-cache.pt", weights_only=False, map_location="cpu")
    assert [r["uid"] for r in cache["train"]] == [r["uid"] for r in old["train"]]
    assert not {r["uid"] for r in cache["development"]} & {
        r["uid"] for r in old["development"]
    }
    out = root / "successor-smoke-v2"
    out.mkdir(exist_ok=False)
    results = {}
    for arm in ARMS:
        model, decoder, checkpoint = initial(parent, cache)
        run = out / arm
        run.mkdir()
        cfg = settings()
        cfg.update(tiny_updates=2, downstream_updates=2, batch_size=2)
        compute = Counter()
        worst = max(cache["train"], key=lambda r: len(r["targets"]))
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
                cache_binding=cache_binding,
                source_sha="smoke_only",
                gradient_rule="joint_diagnostic" if arm == "joint" else arm,
            )
        seconds = time.monotonic() - start
        forecast = seconds / 8 * 20000 * 1.5
        if forecast > 8 * 3600:
            raise RuntimeError("Measured worst-event CPU bound exceeds8h")
        evaluation = evaluate(model, decoder, [worst])
        write(run / "evaluation.json", evaluation)
        assert compute["gradient_probe_events"] == 8
        results[arm] = {
            "status": "PASS",
            "compute": dict(compute),
            "worst_event_seconds": seconds / 8,
            "conservative_fit_seconds": forecast,
        }
    write(
        out / "admission.json",
        {
            "status": "PASS",
            "arms": results,
            "cache": cache_binding,
            "initial_checkpoint": checkpoint,
            "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_hashes": {
                str(p.relative_to(ROOT)): sha(p)
                for folder in ("src", "scripts")
                for p in (ROOT / folder).rglob("*.py")
            },
        },
    )


def freeze(root, parent):
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Clean source required")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    sm = read(binding(root / "successor-smoke-v2/admission.json"))
    for p, digest in sm["source_hashes"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("Source changed after smoke:" + p)
    data = json.loads((root / "data-admission.json").read_text())
    dev = read(data["development"])
    train = read(data["training"])
    excluded = read(data["exclusion_union"])
    cache = json.loads((root / "cache-admission.json").read_text())
    assert (
        data["status"] == "PASS_FULL_IDENTITY_ADMISSION_NOT_MODEL_ADMISSION"
        and not cache["capacity_dropped_events"]
        and cache["status"] == "PASS"
    )
    assert not set(dev["event_uids"]) & (
        set(train["event_uids"]) | set(excluded["event_uids"])
    )
    registry = json.loads(
        (
            ROOT / "configs/reconstruction/supplementary_validation_reservations.json"
        ).read_text()
    )
    records = [
        r
        for r in registry["reservations"]
        if r["version"] == "phase75-development-designation-v1"
    ]
    assert (
        len(records) == 1
        and records[0]["cohort_manifest"]["sha256"] == data["development"]["sha256"]
    )
    assert (
        sha(root / "reservation-registry-before.json")
        == excluded["registry_before"]["sha256"]
    )
    from scripts.validate_next_reconstruction_study import validate as planning

    policy = json.loads(
        (ROOT / "configs/reconstruction/next_study_policy.json").read_text()
    )
    plan = json.loads(
        (
            ROOT / "configs/reconstruction/phase75_gradient_development_plan.json"
        ).read_text()
    )
    evidence = read(
        {
            "path": str(ROOT / policy["evidence"]["path"]),
            "sha256": policy["evidence"]["sha256"],
        }
    )
    assert planning(plan, policy, evidence)["status"] == "PASS"
    source = root / "successor-source"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), head], check=True, cwd=ROOT
    )
    base = root / "successor-campaign-v1"
    base.mkdir(exist_ok=False)
    (base / "runs").mkdir()
    files = subprocess.check_output(
        ["git", "ls-files", "src", "scripts", "configs", "uv.lock", "AGENTS.md"],
        cwd=source,
        text=True,
    ).splitlines()
    c = {
        "version": "phase75-gradient-projection-v1",
        "stage": "development",
        "source_sha": head,
        "source_root": str(source),
        "source_hashes": {p: sha(source / p) for p in files},
        "arms": list(ARMS),
        "settings": settings(),
        "resources": {
            "cpus": 2,
            "memory_gib": 32,
            "hours": 8,
            "gpus": 0,
            "max_jobs": 2,
            "requeue": False,
        },
        "training_count": 1536,
        "heldout_count": 600,
        "context_width": 128,
        "cache": sm["cache"],
        "initial_checkpoint": sm["initial_checkpoint"],
        "parent": str(parent),
        "output_root": str(base / "runs"),
        "sealed_test_access": False,
        "automatic_successor": False,
        "bindings": [
            binding(root / "data-admission.json"),
            binding(root / "cache-admission.json"),
            binding(root / "successor-smoke-v2/admission.json"),
            binding(root / "reservation-registry-before.json"),
            data["development"],
            data["training"],
            data["exclusion_union"],
            data["design"],
            sm["cache"],
            sm["initial_checkpoint"],
            *excluded["bindings"],
            *data["shards"],
            *data["inputs"].values(),
            binding(
                source / "configs/reconstruction/phase75_gradient_development_plan.json"
            ),
            binding(
                source
                / "configs/reconstruction/supplementary_validation_reservations.json"
            ),
        ],
    }
    validate(c)
    verify_bindings(c, source)
    write(base / "contract.json", c)
    write(
        base / "admission.json",
        {
            "status": "PASS",
            "contract_sha256": sha(base / "contract.json"),
            "real_smoke": binding(root / "successor-smoke-v2/admission.json"),
            "full_identity_admission": binding(root / "data-admission.json"),
        },
    )


def submit(path):
    c = json.loads(path.read_text())
    validate(c)
    verify_bindings(c, Path(c["source_root"]))
    admission = json.loads((path.parent / "admission.json").read_text())
    assert admission["status"] == "PASS" and admission["contract_sha256"] == sha(path)
    queue = subprocess.check_output(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%j|%T"],
        text=True,
    )
    if "phase75-train" in queue:
        raise RuntimeError("Concurrent successor")
    if list(path.parent.glob("*-submission.json")):
        raise RuntimeError("Prior submission receipt")
    acquire_submission_lock(path.parent / "submission-lock.json", sha(path))
    for arm in ARMS:
        cmd = [
            "sbatch",
            "--parsable",
            "--export=NIL",
            "--chdir=" + c["source_root"],
            "--job-name=phase75-train-" + arm,
            "--output=" + str(path.parent / (arm + "-%j.log")),
            str(
                Path(c["source_root"]) / "scripts/slurm/run_phase75_development.sbatch"
            ),
            str(path),
            arm,
        ]
        job = subprocess.check_output(cmd, text=True).strip()
        assert job.split(";")[0].isdigit()
        write(
            path.parent / (arm + "-submission.json"),
            {
                "arm": arm,
                "job_id": job,
                "source_sha": c["source_sha"],
                "contract_sha256": sha(path),
                "command": cmd,
            },
        )
        print(arm, job, flush=True)


def run(path, arm):
    guarded()
    import torch
    from hypertagging.training.capacity_development import fresh_decoder

    c = json.loads(path.read_text())
    validate(c)
    verify_bindings(c, ROOT)
    assert str(ROOT) == c["source_root"]
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        == c["source_sha"]
    )
    assert arm in ARMS
    a = json.loads((path.parent / "admission.json").read_text())
    assert a["status"] == "PASS" and a["contract_sha256"] == sha(path)
    out = Path(c["output_root"]) / arm
    out.mkdir(exist_ok=False)
    cache = torch.load(c["cache"]["path"], weights_only=False, map_location="cpu")
    model, decoder, initial_binding = initial(Path(c["parent"]), cache)
    assert initial_binding == c["initial_checkpoint"]
    write(
        out / "startup.json",
        {
            "arm": arm,
            "job_id": os.environ["SLURM_JOB_ID"],
            "source_sha": c["source_sha"],
            "contract": binding(path),
            "initial_checkpoint": initial_binding,
            "cache": c["cache"],
            "settings": c["settings"],
        },
    )
    compute = Counter()
    start = time.monotonic()
    tiny = []
    for cat in ("charged", "mixed", "ccbar"):
        tiny.extend([r for r in cache["train"] if r["category"] == cat][:8])
    tiny_model = copy.deepcopy(model)
    tiny_decoder = fresh_decoder()
    rule = "joint_diagnostic" if arm == "joint" else arm
    histories = {}
    histories["tiny"] = fit(
        tiny_model,
        tiny_decoder,
        tiny,
        stage="tiny",
        objective="existing",
        settings=c["settings"],
        output=out,
        compute=compute,
        cache_binding=c["cache"],
        source_sha=c["source_sha"],
        gradient_rule=rule,
    )
    tiny_result = evaluate(tiny_model, tiny_decoder, tiny)
    del tiny_model, tiny_decoder
    histories["downstream"] = fit(
        model,
        decoder,
        cache["train"],
        stage="downstream",
        objective="existing",
        settings=c["settings"],
        output=out,
        compute=compute,
        cache_binding=c["cache"],
        source_sha=c["source_sha"],
        gradient_rule=rule,
    )
    result = {
        "arm": arm,
        "stage": "development",
        "source_sha": c["source_sha"],
        "contract_sha256": sha(path),
        "histories": histories,
        "tiny": tiny_result,
        "train": evaluate(model, decoder, cache["train"]),
        "heldout": evaluate(model, decoder, cache["development"]),
        "automatic_successor": False,
        "primary_eligible": False,
    }
    compute.update(
        wall_seconds=time.monotonic() - start,
        cpu_seconds=time.process_time(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        decoder_parameters=sum(p.numel() for p in decoder.parameters()),
        evaluation_event_views=2160,
    )
    result["compute"] = dict(compute)
    write(out / "summary.json", result)
    write(
        out / "terminal.json",
        {
            "status": "COMPLETED",
            "summary": binding(out / "summary.json"),
            "checkpoints": [
                binding(out / f"{s}-final.pt") for s in ("tiny", "downstream")
            ],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["smoke", "freeze", "submit", "run"])
    p.add_argument("--root", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--contract", type=Path)
    p.add_argument("--arm")
    a = p.parse_args()
    if a.action == "smoke":
        smoke(a.root.resolve(), a.parent.resolve())
    elif a.action == "freeze":
        freeze(a.root.resolve(), a.parent.resolve())
    elif a.action == "submit":
        submit(a.contract.resolve())
    else:
        run(a.contract.resolve(), a.arm)
