#!/usr/bin/env python
"""Verify a post-calibration phase-3 contract; never submit or inspect payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TOTAL_PRESENTATIONS = 1_730_048
CHECKPOINT_SHA256 = "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
PARENT_AUTHORIZATION = "artifacts/codex/ht_pretraining_1m_phase3_execution_authorization_20260823.json"
PARENT_AUTHORIZATION_SHA256 = "1af20420655a95aa7ce0a3d1ad4a6e357c7fe45510c3f8bafaf80ad3fdbb7991"
PARENT_AUTHORIZATION_CANONICAL_SHA256 = "c952524ce32b1c504cc6210cc8bc540bb6180a928c145fe29109b89b3fe3b5e3"
MILESTONES = [13_516, 27_032, 40_548, 54_064, 67_580, 81_096, 94_612, 108_128]


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    stored = payload.get("contract_sha256")
    body = dict(payload)
    body.pop("contract_sha256", None)
    if stored != _hash(body):
        raise RuntimeError("production contract hash mismatch")
    if payload.get("job_count") != 1 or payload.get("submission_performed") is not False:
        raise RuntimeError("contract does not preserve exactly one unsubmitted job")
    if payload.get("contract_version") != "hypertagging-slurm-one-gpu-contract-v2":
        raise RuntimeError("contract is not compatible with the one-GPU wrapper")
    if payload.get("batch_efficiency_production") is not True:
        raise RuntimeError("batch-efficiency production marker is missing")
    if payload.get("mode") != "scientific" or payload.get("export_policy") != "NIL":
        raise RuntimeError("production contract runtime mode/export policy is not exact")
    if payload.get("production_submission_authorized") is not True:
        raise RuntimeError("contract lacks post-calibration production authorization")
    if payload.get("operator_authorization_parent") is not True:
        raise RuntimeError("parent operator authorization is not preserved")
    if payload.get("parent_operator_authorization_artifact") != PARENT_AUTHORIZATION:
        raise RuntimeError("parent operator authorization artifact binding changed")
    if payload.get("parent_operator_authorization_sha256") != PARENT_AUTHORIZATION_SHA256:
        raise RuntimeError("parent operator authorization file hash changed")
    if payload.get("parent_operator_authorization_canonical_sha256") != PARENT_AUTHORIZATION_CANONICAL_SHA256:
        raise RuntimeError("parent operator authorization canonical hash changed")
    if payload.get("scientific_slurm_submission_allowed") is not False:
        raise RuntimeError("structural scientific provenance gate was weakened")
    if payload.get("total_presentations") != TOTAL_PRESENTATIONS:
        raise RuntimeError("total presentation contract changed")
    if payload.get("remaining_presentations") != 865_024:
        raise RuntimeError("remaining presentation contract changed")
    if payload.get("resume_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("resume checkpoint binding changed")
    if payload.get("gres") not in {"gpu:h100nvl:1", "gpu:v100:1"}:
        raise RuntimeError("contract contains unsupported or generic GRES")
    if not payload.get("expected_git_sha") or not payload.get("expected_git_tag"):
        raise RuntimeError("contract lacks immutable source commit/tag binding")
    if payload.get("dataset_index") != "artifacts/experiment_readiness/production_1m_20260812/train_865k/train_865k.complete_only.index.json":
        raise RuntimeError("dataset index binding changed")
    if payload.get("selection_manifest") != "configs/training_selection/production_1m_20260812/train_865k.json":
        raise RuntimeError("train-role selection binding changed")
    if not payload.get("calibration_selection_evidence"):
        raise RuntimeError("calibration selection evidence is missing")
    if payload.get("validation_milestones_virtual_steps") != MILESTONES:
        raise RuntimeError("validation/checkpoint milestone mapping changed")
    if payload.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("sealed-test isolation was weakened")
    if payload.get("stress_payload_access") != "forbidden":
        raise RuntimeError("stress isolation was weakened")
    if payload.get("exactly_one_submission_command_required") is not True:
        raise RuntimeError("exactly-one submission gate is missing")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args(argv)
    payload = verify(args.contract)
    print(json.dumps({"contract": str(args.contract), "verified": True, "selected_profile": payload["selected_profile"], "submission_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
