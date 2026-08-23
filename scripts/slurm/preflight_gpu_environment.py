#!/usr/bin/env python
"""Verify the frozen GPU runtime and exact Slurm GPU allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.utils.gpu_safety import (  # noqa: E402
    ALLOWED_SLURM_GRES,
    assert_scientific_slurm_gpu_allowed,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_lock_hashes(root: Path) -> None:
    receipt = root / "environment/gpu/environment-lock.sha256"
    for line in receipt.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if sha256(path) != expected:
            raise RuntimeError(f"GPU environment lock hash mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gres", choices=ALLOWED_SLURM_GRES)
    parser.add_argument(
        "--execution-contract",
        type=Path,
        help="bind the preflight attestation to a recovery execution contract",
    )
    parser.add_argument(
        "--authorization-artifact",
        type=Path,
        help="bind the preflight attestation to a phase-3 authorization artifact",
    )
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    if (args.execution_contract is None) != (args.authorization_artifact is None):
        raise RuntimeError(
            "execution contract and authorization artifact must be supplied together"
        )
    contract_path = ROOT / "environment/gpu/runtime-contract.json"
    verify_lock_hashes(ROOT)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if args.lock_only:
        print(json.dumps({"lock_verified": True, "contract_sha256": sha256(contract_path)}))
        return 0
    if f"{sys.version_info.major}.{sys.version_info.minor}" != contract["expected_python"]:
        raise RuntimeError("GPU environment Python version mismatch")
    for distribution, expected in contract["versions"].items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"GPU environment version mismatch for {distribution}: {actual} != {expected}"
            )
    for module in (
        "yaml", "numpy", "scipy", "pyarrow", "pandas", "awkward", "uproot",
        "pdg", "onnx", "torch",
    ):
        importlib.import_module(module)
    import torch

    if torch.version.cuda != contract["cuda_build"] or not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA build/runtime is unavailable or mismatched")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    assert_scientific_slurm_gpu_allowed(
        args.expected_gres,
        gpu_name=torch.cuda.get_device_name(0),
    )
    payload = {
        "preflight_version": "ht-pretraining-1m-phase3-in-allocation-preflight-v1",
        "status": "passed",
        "fresh_in_allocation": True,
        "gpu_pilot_completed": False,
        "never_cpu": True,
        "expected_gres": args.expected_gres,
        "observed_gres": args.expected_gres,
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_gpus_on_node": os.environ["SLURM_GPUS_ON_NODE"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if args.execution_contract is not None:
        contract_payload = json.loads(
            args.execution_contract.read_text(encoding="utf-8")
        )
        authorization_payload = json.loads(
            args.authorization_artifact.read_text(encoding="utf-8")
        )
        payload.update(
            {
                "execution_contract_file_sha256": sha256(args.execution_contract),
                "authorization_artifact_file_sha256": sha256(
                    args.authorization_artifact
                ),
                "authorization_artifact_canonical_sha256": authorization_payload.get(
                    "artifact_sha256"
                ),
                "execution_contract_canonical_sha256": contract_payload.get(
                    "contract_sha256"
                ),
            }
        )
    payload["preflight_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
