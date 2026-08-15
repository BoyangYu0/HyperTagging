#!/usr/bin/env python3
"""Verify and expose a source-bound read-only pretraining validation contract."""

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


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "hypertagging-pretraining-validation-job-v1"
ALLOWED_GRES = {"gpu:h100nvl:1", "gpu:v100:1"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_root(value: str, *, required_suffix: str | None = None) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("contract path escapes the repository") from error
    if required_suffix is not None and resolved.suffix != required_suffix:
        raise RuntimeError(f"contract path must end in {required_suffix}")
    return resolved


def verify_contract_hash(contract: dict[str, Any]) -> None:
    stored = str(contract.get("contract_sha256", ""))
    canonical = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError("evaluation job contract hash mismatch")


def verify_hashed_inputs(inputs: list[dict[str, str]]) -> None:
    if not inputs:
        raise RuntimeError("evaluation job contract has no hashed inputs")
    for item in inputs:
        path = _inside_root(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not HEX64.fullmatch(expected) or _sha256(path) != expected:
            raise RuntimeError(f"hashed evaluation input changed: {path}")


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def validated_runtime_values(contract: dict[str, Any]) -> dict[str, str]:
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("unsupported pretraining validation contract")
    if contract.get("mode") != "read_only_pretraining_validation":
        raise RuntimeError("evaluation contract is not read-only validation")
    if contract.get("evaluation_role") != "validation":
        raise RuntimeError("evaluation contract must bind the validation role")
    if contract.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("evaluation contract must forbid sealed-test access")
    if contract.get("validation_events") != 2000:
        raise RuntimeError("evaluation contract must bind exactly 2,000 events")
    if contract.get("optimizer_steps") != 0:
        raise RuntimeError("evaluation contract must forbid optimizer steps")
    if contract.get("checkpoint_step") != 3282:
        raise RuntimeError("evaluation contract must bind best checkpoint step 3282")
    gres = str(contract.get("gres", ""))
    if gres not in ALLOWED_GRES:
        raise RuntimeError("evaluation contract has an unsupported exact GRES")
    checkpoint_sha = str(contract.get("checkpoint_sha256", ""))
    source_sha = str(contract.get("expected_git_sha", ""))
    if not HEX64.fullmatch(checkpoint_sha) or not HEX40.fullmatch(source_sha):
        raise RuntimeError("evaluation contract has an invalid SHA256/Git SHA")
    checkpoint = _inside_root(
        str(contract.get("checkpoint", "")), required_suffix=".pt"
    )
    data = _inside_root(
        str(contract.get("selection_manifest", "")), required_suffix=".json"
    )
    dataset_index = _inside_root(
        str(contract.get("dataset_index", "")), required_suffix=".json"
    )
    if _sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("evaluation checkpoint SHA256 mismatch")
    output_base = Path(str(contract.get("evaluation_output_base", "")))
    output_resolved = (ROOT / output_base).resolve()
    expected_parent = (ROOT / "artifacts" / "evaluations").resolve()
    if expected_parent not in output_resolved.parents:
        raise RuntimeError(
            "evaluation output base must remain under artifacts/evaluations"
        )
    gpu_environment = Path(str(contract.get("gpu_environment", "")))
    if (
        not gpu_environment.is_absolute()
        or not (gpu_environment / "bin/python").is_file()
    ):
        raise RuntimeError("evaluation GPU environment is unavailable")
    return {
        "expected_gres": gres,
        "gpu_environment": str(gpu_environment),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": "3282",
        "expected_git_sha": source_sha,
        "selection_manifest": str(data),
        "dataset_index": str(dataset_index),
        "evaluation_output_base": str(output_resolved),
    }


def verify_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    verify_contract_hash(contract)
    verify_hashed_inputs(list(contract.get("hashed_inputs", [])))
    runtime = validated_runtime_values(contract)
    if _git_output("rev-parse", "HEAD") != runtime["expected_git_sha"]:
        raise RuntimeError("evaluation source Git SHA mismatch")
    tag = str(contract.get("expected_git_tag", ""))
    if (
        not tag
        or _git_output("rev-list", "-n", "1", tag) != runtime["expected_git_sha"]
    ):
        raise RuntimeError("evaluation source tag mismatch")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            "tracked worktree changes invalidate evaluation source binding"
        )
    if contract.get("submission_authorized") is not True:
        raise RuntimeError("evaluation contract is not authorized for submission")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("read-only evaluation jobs may not restart or requeue")
    return contract, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--shell-output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    contract, runtime = verify_contract(contract_path)
    lines = [f"{key}={shlex.quote(value)}" for key, value in runtime.items()]
    args.shell_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "contract_verified": True,
                "contract_sha256": contract["contract_sha256"],
                "checkpoint_sha256": contract["checkpoint_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
