"""Bind tested immutable source, data and initial states; no scheduler submission."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, read, sha, write  # noqa: E402
from scripts.run_phase74_development import validate_contract, verify_bindings  # noqa: E402
from scripts.validate_next_reconstruction_study import validate  # noqa: E402


def freeze(output, preflight):
    from hypertagging.training.capacity_development import ARMS, ARCHITECTURE

    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Commit source and audit lineage before freeze")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    smoke = json.loads((preflight / "admission.json").read_text())
    if smoke["status"] != "PASS" or set(smoke["arms"]) != set(ARMS):
        raise ValueError("Real-data preflight incomplete")
    for path, digest in smoke["preflight_source_hashes"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Source changed after real preflight: " + path)
    data = json.loads((output / "data-admission.json").read_text())
    cache = json.loads((output / "cache-admission.json").read_text())
    if (
        data["status"] != "PASS_FULL_IDENTITY_ADMISSION_NOT_MODEL_ADMISSION"
        or cache["status"] != "PASS"
        or cache["capacity_dropped_events"]
    ):
        raise ValueError("Data/schema/capacity admission failed")
    dev = read(data["development"])
    train = read(data["training"])
    excluded = read(data["exclusion_union"])
    for item in excluded["bindings"]:
        read(item)
    before_path = output / "reservation-registry-before.json"
    if sha(before_path) != excluded["registry_before"]["sha256"]:
        raise ValueError("Historical registry snapshot mismatch")
    if set(dev["event_uids"]) & (
        set(train["event_uids"]) | set(excluded["event_uids"])
    ):
        raise ValueError("Fresh development cohort leakage")
    registry = json.loads(
        (
            ROOT / "configs/reconstruction/supplementary_validation_reservations.json"
        ).read_text()
    )
    before = json.loads(before_path.read_text())
    if (
        registry["reservations"][: len(before["reservations"])]
        != before["reservations"]
    ):
        raise ValueError("Historical reservations changed")
    designated = [
        r
        for r in registry["reservations"]
        if r.get("version") == "phase74-development-designation-v1"
    ]
    if (
        len(designated) != 1
        or designated[0]["cohort_manifest"]["sha256"] != data["development"]["sha256"]
        or designated[0]["stage"] != "development"
    ):
        raise ValueError("Shared registry designation missing or changed")
    policy = json.loads(
        (ROOT / "configs/reconstruction/next_study_policy.json").read_text()
    )
    scientific = ROOT / "configs/reconstruction/phase74_capacity_development_plan.json"
    plan = json.loads(scientific.read_text())
    evidence_path = ROOT / policy["evidence"]["path"]
    if sha(evidence_path) != policy["evidence"]["sha256"]:
        raise ValueError("Policy evidence changed")
    result = validate(plan, policy, json.loads(evidence_path.read_text()))
    if result["status"] != "PASS":
        raise ValueError(result)
    source = output / "training-source"
    if source.exists():
        raise FileExistsError(source)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), head], cwd=ROOT, check=True
    )
    contract_root = output / "campaign-v1"
    contract_root.mkdir(exist_ok=False)
    file_names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "scripts",
            "configs",
            "AGENTS.md",
            "uv.lock",
            "environment/gpu/requirements-cu126.lock",
        ],
        cwd=source,
        text=True,
    ).splitlines()
    source_hashes = {name: sha(source / name) for name in file_names}
    inputs = [
        data["training"],
        data["development"],
        data["exclusion_union"],
        data["design"],
        cache["cache"],
        binding(output / "data-admission.json"),
        binding(output / "cache-admission.json"),
        binding(preflight / "admission.json"),
        binding(before_path),
        *excluded["bindings"],
        *smoke["initializations"].values(),
        *data["shards"],
        data["inputs"]["selection"],
        data["inputs"]["index"],
        data["inputs"]["reconstruction-checkpoint"],
        binding(source / scientific.relative_to(ROOT)),
        binding(source / "configs/reconstruction/next_study_policy.json"),
        binding(source / policy["evidence"]["path"]),
        binding(
            source / "configs/reconstruction/supplementary_validation_reservations.json"
        ),
    ]
    c = {
        "version": "phase74-development-runtime-v1",
        "source_sha": head,
        "source_root": str(source),
        "source_hashes": source_hashes,
        "stage": "development",
        "arms": list(ARMS),
        "architecture": ARCHITECTURE,
        "bindings": inputs,
        "cache": cache["cache"],
        "initializations": smoke["initializations"],
        "history": "fresh_both_widths_shared_initialization_within_width",
        "head_interface": "lossless_zero_pad_to_common256_predicted_group_conditioning_reset_after_pretrain",
        "training_count": 1536,
        "heldout_count": 600,
        "sealed_test_access": False,
        "automatic_successor": False,
        "output_root": str(contract_root / "runs"),
        "settings": {
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
        },
        "resources": {
            "cpus": 2,
            "memory_gib": 32,
            "hours": 8,
            "gpus": 0,
            "max_jobs": 4,
            "requeue": False,
        },
        "admission": {"path": str(contract_root / "admission.json")},
    }
    # Admission is one-way bound to the complete contract; avoid circular hashes.
    c.pop("admission")
    validate_contract(c)
    verify_bindings(c, source)
    write(contract_root / "contract.json", c)
    write(
        contract_root / "admission.json",
        {
            "status": "PASS",
            "source_sha": head,
            "contract_sha256": sha(contract_root / "contract.json"),
            "planning": result,
            "data": binding(output / "data-admission.json"),
            "real_gradient_preflight": binding(preflight / "admission.json"),
            "full_cohort_isolation": True,
            "development_only": True,
        },
    )
    (contract_root / "runs").mkdir()
    print(
        json.dumps(
            {
                "status": "FROZEN_ADMITTED_NOT_SUBMITTED",
                "contract": str(contract_root / "contract.json"),
                "source_sha": head,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    a = p.parse_args()
    freeze(a.output.resolve(), a.preflight.resolve())
