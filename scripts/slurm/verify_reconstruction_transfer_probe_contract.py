#!/usr/bin/env python3
"""Verify a frozen-transfer downstream reconstruction probe contract."""

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
CONTRACT_VERSION = "hypertagging-reconstruction-transfer-probe-v1"
HEADWARMUP_200_CONTRACT_VERSION = (
    "hypertagging-reconstruction-transfer-probe-v2-headwarmup-200"
)
QUERY_ACTIVATION_CONTRACT_VERSION = (
    "hypertagging-reconstruction-transfer-probe-v3-query-activation-balance"
)
ALLOWED_GRES = {"gpu:h100nvl:1", "gpu:v100:1"}
ALLOWED_STEPS = {2188, 3282, 4376}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_PROBE = {
    "max_steps": 100,
    "batch_size": 4,
    "seed": 20260815,
    "max_validation_events": 512,
    "rollout_validation_events": 256,
    "validation_batch_size": 4,
    "validate_every": 50,
    "rollout_validate_every": 100,
    "freeze_pretrained_encoder_steps": 100,
    "freeze_leaf_pid_head_steps": 100,
    "transfer_leaf_pid_head": True,
    "model_preset": "small_candidate",
    "max_cardinality": 16,
    "max_cardinality_by_level": [[1, 16], [2, 16], [3, 16]],
    "target_policy": "complete_only",
    "initial_state_policy": "upsilon4s",
    "best_metric": "predicted_edge_f1",
    "best_mode": "max",
}
HEADWARMUP_200_REQUIRED_PROBE = {
    **REQUIRED_PROBE,
    "max_steps": 200,
    "freeze_pretrained_encoder_steps": 200,
    "freeze_leaf_pid_head_steps": 200,
}
QUERY_ACTIVATION_REQUIRED_PROBE = {
    **REQUIRED_PROBE,
    "object_positive_weight": 16.0,
    "pointer_positive_weight": 16.0,
}
REQUIRED_PROBES_BY_CONTRACT_VERSION = {
    CONTRACT_VERSION: REQUIRED_PROBE,
    HEADWARMUP_200_CONTRACT_VERSION: HEADWARMUP_200_REQUIRED_PROBE,
    QUERY_ACTIVATION_CONTRACT_VERSION: QUERY_ACTIVATION_REQUIRED_PROBE,
}
ALLOWED_STEPS_BY_CONTRACT_VERSION = {
    CONTRACT_VERSION: ALLOWED_STEPS,
    HEADWARMUP_200_CONTRACT_VERSION: {3282},
    QUERY_ACTIVATION_CONTRACT_VERSION: {3282},
}


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
        raise RuntimeError("transfer-probe path escapes the repository") from error
    if required_suffix is not None and resolved.suffix != required_suffix:
        raise RuntimeError(f"transfer-probe path must end in {required_suffix}")
    return resolved


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def verify_contract_hash(contract: dict[str, Any]) -> None:
    stored = str(contract.get("contract_sha256", ""))
    canonical = {key: value for key, value in contract.items() if key != "contract_sha256"}
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError("transfer-probe contract hash mismatch")


def verify_hashed_inputs(inputs: list[dict[str, str]]) -> None:
    if not inputs:
        raise RuntimeError("transfer-probe contract has no hashed inputs")
    for item in inputs:
        path = _inside_root(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not HEX64.fullmatch(expected) or _sha256(path) != expected:
            raise RuntimeError(f"hashed transfer-probe input changed: {path}")


def required_probe_for_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the exact probe profile selected by the contract."""

    version = str(contract.get("contract_version", ""))
    required = REQUIRED_PROBES_BY_CONTRACT_VERSION.get(version)
    if required is None:
        raise RuntimeError("unsupported transfer-probe contract")
    probe = dict(contract.get("probe", {}))
    if probe != required:
        raise RuntimeError(
            "transfer-probe configuration differs from the registered contract profile"
        )
    if int(contract.get("optimizer_steps", -1)) != required["max_steps"]:
        raise RuntimeError("transfer-probe optimizer-step count is inconsistent")
    step = int(contract.get("checkpoint_step", -1))
    if step not in ALLOWED_STEPS_BY_CONTRACT_VERSION[version]:
        raise RuntimeError(
            "transfer-probe step is not approved for the selected contract profile"
        )
    return dict(required)


def verify_contract(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    verify_contract_hash(contract)
    verify_hashed_inputs(list(contract.get("hashed_inputs", [])))
    required_probe = required_probe_for_contract(contract)
    if contract.get("mode") != "frozen_pretrained_reconstruction_transfer_probe":
        raise RuntimeError("contract is not a frozen transfer probe")
    if contract.get("training_role") != "train" or contract.get("evaluation_role") != "validation":
        raise RuntimeError("transfer probe must train on train and evaluate validation")
    if contract.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("transfer probe must forbid sealed-test access")
    step = int(contract["checkpoint_step"])
    gres = str(contract.get("gres", ""))
    if gres not in ALLOWED_GRES:
        raise RuntimeError("transfer probe has unsupported exact GRES")
    checkpoint_sha = str(contract.get("checkpoint_sha256", ""))
    source_sha = str(contract.get("expected_git_sha", ""))
    if not HEX64.fullmatch(checkpoint_sha) or not HEX40.fullmatch(source_sha):
        raise RuntimeError("transfer probe has an invalid digest")
    checkpoint = _inside_root(str(contract.get("checkpoint", "")), required_suffix=".pt")
    if checkpoint.name != f"checkpoint-step-{step}.pt" or _sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("transfer-probe checkpoint binding failed")
    selection = _inside_root(str(contract.get("selection_manifest", "")), required_suffix=".json")
    dataset_index = _inside_root(str(contract.get("dataset_index", "")), required_suffix=".json")
    output_base = (ROOT / str(contract.get("study_output_base", ""))).resolve()
    expected_parent = (ROOT / "artifacts" / "studies" / "reconstruction-transfer-probe").resolve()
    if expected_parent not in output_base.parents:
        raise RuntimeError("transfer-probe output must remain under its study root")
    gpu_environment = Path(str(contract.get("gpu_environment", "")))
    if not gpu_environment.is_absolute() or not (gpu_environment / "bin/python").is_file():
        raise RuntimeError("transfer-probe GPU environment is unavailable")
    if _git_output("rev-parse", "HEAD") != source_sha:
        raise RuntimeError("transfer-probe source Git SHA mismatch")
    tag = str(contract.get("expected_git_tag", ""))
    if not tag or _git_output("rev-list", "-n", "1", tag) != source_sha:
        raise RuntimeError("transfer-probe source tag mismatch")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree changes invalidate transfer-probe source binding")
    if contract.get("submission_authorized") is not True:
        raise RuntimeError("transfer-probe contract is not authorized")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("transfer probe must run inside Slurm")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("transfer-probe jobs may not restart or requeue")
    runtime = {
        "expected_gres": gres,
        "gpu_environment": str(gpu_environment),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": str(step),
        "expected_git_sha": source_sha,
        "selection_manifest": str(selection),
        "dataset_index": str(dataset_index),
        "study_output_base": str(output_base),
        "optimizer_steps": str(required_probe["max_steps"]),
    }
    return contract, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--shell-output", type=Path, required=True)
    args = parser.parse_args()
    contract, runtime = verify_contract(args.contract.resolve(strict=True))
    args.shell_output.write_text(
        "\n".join(f"{key}={shlex.quote(value)}" for key, value in runtime.items()) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "contract_verified": True,
                "contract_sha256": contract["contract_sha256"],
                "checkpoint_step": contract["checkpoint_step"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
