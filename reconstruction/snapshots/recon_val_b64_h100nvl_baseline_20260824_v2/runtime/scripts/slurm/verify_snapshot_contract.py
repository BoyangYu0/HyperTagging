#!/usr/bin/env python3
"""Fail-closed verifier for the committed reconstruction replacement snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_POINTER = "9fc76128e930cab3119354b1ab41b5654ff688d2af1898d231ea6de29e76d213"
EXPECTED_CHECKPOINT = "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
EXPECTED_BASE_COMMIT = "f06268c148b5ed5f07a3ed484fac7f142ecda80b"
EXPECTED_BASE_TAG = "ht-reconstruction-validation-followup-20260824-final-gres"
EXPECTED_MANIFEST_JSON = "641d89bf6edafb044b90e98185d2171c568e1b9eb45c76aee3281716e22d5280"
EXPECTED_MANIFEST_MD = "77a93ed587db1cf840b0d55b4ab79bb23cacc155e37ed455f734eb076b7463c7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any], field: str) -> str:
    without = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(without, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=root, capture_output=True, text=True, check=True, timeout=30
    )
    return result.stdout.strip()


def inside(base: Path, value: str) -> Path:
    candidate = (Path(value) if Path(value).is_absolute() else base / value).resolve(strict=True)
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise RuntimeError(f"path escapes bound root: {value}") from error
    return candidate


def verify_manifest(snapshot: Path, contract_hash: str) -> dict[str, Any]:
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "reconstruction-snapshot-manifest-v1":
        raise RuntimeError("unsupported snapshot manifest")
    if manifest.get("contract_sha256") != contract_hash:
        raise RuntimeError("snapshot manifest is not bound to the submitted contract")
    listed: set[str] = set()
    for item in manifest.get("files", []):
        relative = str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        if not relative or not HEX64.fullmatch(expected):
            raise RuntimeError("malformed snapshot file hash")
        path = inside(snapshot, relative)
        if relative in {"job_contract.json", "snapshot_manifest.json"}:
            raise RuntimeError("self-referential snapshot file listed")
        if sha256(path) != expected:
            raise RuntimeError(f"snapshot file changed: {relative}")
        listed.add(relative)
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.name not in {"job_contract.json", "snapshot_manifest.json"}
    }
    if listed != actual:
        missing = sorted(actual - listed)
        extra = sorted(listed - actual)
        raise RuntimeError(f"snapshot file inventory mismatch: missing={missing[:4]} extra={extra[:4]}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--shell-output", type=Path, required=True)
    parser.add_argument("--allow-static", action="store_true")
    parser.add_argument("--precommit", action="store_true", help="allow the final tag to be absent before the one snapshot commit")
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    snapshot = contract_path.parent
    if contract_path.name != "job_contract.json":
        raise RuntimeError("replacement contract must be snapshot/job_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    actual_contract_hash = canonical_hash(contract, "contract_sha256")
    if contract.get("contract_sha256") != actual_contract_hash:
        raise RuntimeError("replacement contract canonical hash mismatch")
    verify_manifest(snapshot, actual_contract_hash)
    if contract.get("contract_version") != "hypertagging-reconstruction-snapshot-v1":
        raise RuntimeError("unsupported replacement contract")
    if contract.get("snapshot_root") != str(snapshot):
        raise RuntimeError("contract snapshot root is not absolute and exact")
    if contract.get("final_commit_binding_mode") != "exact_tag_resolution":
        raise RuntimeError("replacement commit binding mode changed")
    if contract.get("snapshot_git_tag") != "ht-reconstruction-validation-followup-20260824-v2-immutable":
        raise RuntimeError("unexpected replacement snapshot tag")
    head = git(repo, "rev-parse", "HEAD")
    tag_ref = f"refs/tags/{contract['snapshot_git_tag']}^{{commit}}"
    try:
        tagged_head = git(repo, "rev-parse", tag_ref)
    except subprocess.CalledProcessError:
        if not args.precommit:
            raise RuntimeError("final immutable snapshot tag is absent")
        tagged_head = None
    if tagged_head is not None and tagged_head != head:
        raise RuntimeError("checked out HEAD is not the exact tagged replacement snapshot")
    if git(repo, "rev-list", "-n", "1", EXPECTED_BASE_TAG) != EXPECTED_BASE_COMMIT:
        raise RuntimeError("base implementation tag/commit binding changed")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean")
    if contract.get("source_implementation_commit") != EXPECTED_BASE_COMMIT or contract.get("source_implementation_tag") != EXPECTED_BASE_TAG:
        raise RuntimeError("source implementation commit/tag binding changed")
    if contract.get("gres") != "gpu:h100nvl:1":
        raise RuntimeError("exact H100-NVL GRES binding changed")
    config = contract.get("config", {})
    expected_config = {
        "batch_size": 64, "max_steps": 1094, "presentations_target": 70016,
        "learning_rate": 0.001, "lr_schedule_total_steps": 1094,
        "object_positive_weight": 12.0, "pointer_positive_weight": 16.0,
        "amp_dtype": "bfloat16", "grad_scaler_enabled": False,
        "mixed_precision": True, "validation_enabled": True,
        "transfer_leaf_pid_head": True, "freeze_pretrained_encoder_steps": 1094,
        "freeze_leaf_pid_head_steps": 1094, "validation_access_policy": "validation_only",
        "train_validation_fallback": False, "best_metric": "predicted_edge_f1",
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("replacement scientific config binding changed")
    resources = contract.get("resources", {})
    if resources != {"cpus": 8, "memory": "64G", "time": "06:00:00", "max_restarts": 0}:
        raise RuntimeError("replacement resource/restart binding changed")
    if contract.get("training_role") != "train" or contract.get("evaluation_role") != "validation":
        raise RuntimeError("role binding changed")
    if contract.get("sealed_test_role_access") != "forbidden" or contract.get("stress_role_access") != "forbidden" or contract.get("restricted_raw_source_payload_access") != "forbidden":
        raise RuntimeError("restricted-role policy changed")
    if contract.get("pretraining_success_claimed") is not False or contract.get("production_resume_authorized") is not False:
        raise RuntimeError("pretraining/reconstruction authorization boundary changed")
    pointer_path = inside(repo, str(contract["pointer_path"]))
    if sha256(pointer_path) != EXPECTED_POINTER:
        raise RuntimeError("live pretraining pointer hash changed")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("scientific_gate_passed") is not False or pointer.get("production_readiness_passed") is not False or pointer.get("production_resume_authorized") is not False:
        raise RuntimeError("pointer incorrectly authorizes production/pretraining")
    checkpoint = inside(repo, str(contract["checkpoint_path"]))
    if sha256(checkpoint) != EXPECTED_CHECKPOINT or contract.get("checkpoint_step") != 54064:
        raise RuntimeError("source checkpoint binding changed")
    for item in contract.get("external_hashes", []):
        path = inside(repo, str(item["path"]))
        expected = str(item["sha256"])
        if not HEX64.fullmatch(expected) or sha256(path) != expected:
            raise RuntimeError(f"external immutable input changed: {item['path']}")
    if contract.get("shared_manifest_json_sha256") != EXPECTED_MANIFEST_JSON or contract.get("shared_manifest_md_sha256") != EXPECTED_MANIFEST_MD:
        raise RuntimeError("shared CUDA manifest binding changed")
    selection = inside(snapshot, str(contract["selection_manifest_snapshot"]))
    index = inside(snapshot, str(contract["dataset_index_snapshot"]))
    output_root = Path(str(contract["output_root"]))
    if not output_root.is_absolute():
        output_root = (repo / output_root).resolve()
    expected_parent = (repo / "artifacts/runs/ht-reconstruction-transfer-fullscale-20260824").resolve()
    if expected_parent not in output_root.parents:
        raise RuntimeError("replacement output root is outside the authorized study root")
    if not (Path(str(contract["gpu_environment"])) / "bin/python").is_file():
        raise RuntimeError("uv GPU environment is unavailable")
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("zero-restart policy violated")
    if not args.allow_static and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("replacement training must run inside Slurm")
    runtime = {
        "contract_sha256": actual_contract_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "checkpoint_step": "54064",
        "selection_manifest": str(selection),
        "dataset_index": str(index),
        "gpu_environment": str(contract["gpu_environment"]),
        "output_root": str(output_root),
        "expected_gres": "gpu:h100nvl:1",
        "max_steps": "1094",
        "max_restarts": "0",
        "snapshot_root": str(snapshot),
        "snapshot_git_tag": str(contract["snapshot_git_tag"]),
    }
    args.shell_output.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in runtime.items()) + "\n", encoding="utf-8")
    print(json.dumps(runtime, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
