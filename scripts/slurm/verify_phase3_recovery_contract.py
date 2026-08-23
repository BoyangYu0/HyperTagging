#!/usr/bin/env python
"""Verify the exact production-1M phase-3 recovery contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_job_contract import verify_contract  # noqa: E402
from scripts.slurm.phase3_execution_authorization_v1 import (  # noqa: E402
    AUTHORIZATION_ARTIFACT,
    verify_authorization_artifact,
)

EXPECTED_CHECKPOINT = (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
)
EXPECTED_EXPERIMENT = "ht-pretrain-1m-phase3-recovery-20260823"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract",
        type=Path,
        default=ROOT
        / "artifacts/slurm/ht-pretrain-1m-phase3-recovery-20260823.operator-authorized.job-contract.json",
        nargs="?",
    )
    args = parser.parse_args()
    contract, _, contract_hash = verify_contract(args.contract)
    authorization_path = ROOT / AUTHORIZATION_ARTIFACT
    verify_authorization_artifact(
        authorization_path,
        contract_path=args.contract,
    )
    lineage = contract.get("recovery_lineage")
    if not isinstance(lineage, dict):
        raise RuntimeError("recovery lineage is missing")
    if contract.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError("recovery output experiment is not immutable and distinct")
    if contract.get("resume_checkpoint") != EXPECTED_CHECKPOINT:
        raise RuntimeError("recovery checkpoint path is not the exact step-54064 source")
    if contract.get("resume_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("recovery checkpoint hash is not the exact source hash")
    if lineage.get("historical_job_id") != "15933802":
        raise RuntimeError("historical failed job binding is missing")
    if lineage.get("replacement_attempt_root_must_not_be") != lineage.get(
        "historical_attempt_root"
    ):
        raise RuntimeError("replacement attempt root is not protected from historical attempt-00")
    if contract.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("sealed-test role isolation is not preserved")
    print(
        json.dumps(
            {
                "authorization_artifact": str(authorization_path),
                "contract": str(args.contract),
                "contract_sha256": contract_hash,
                "fresh_in_allocation_preflight_required": True,
                "submission_performed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
