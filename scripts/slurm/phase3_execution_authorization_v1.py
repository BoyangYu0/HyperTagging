#!/usr/bin/env python
"""Versioned, fail-closed authorization checks for the phase-3 recovery.

This module is intentionally stdlib-only.  It verifies the operator
authorization record and, when requested, the fresh in-allocation preflight
attestation.  It does not submit or mutate Slurm jobs and it never opens a
training, sealed-test, or stress payload.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

AUTHORIZATION_VERSION = "ht-pretraining-1m-phase3-execution-authorization-v1"
PREFLIGHT_VERSION = "ht-pretraining-1m-phase3-in-allocation-preflight-v1"
AUTHORIZATION_ARTIFACT = (
    "artifacts/codex/ht_pretraining_1m_phase3_execution_authorization_20260823.json"
)
BASE_CONTRACT = (
    "artifacts/slurm/ht-pretrain-1m-phase3-recovery-20260823.operator-authorized.job-contract.json"
)
CHECKPOINT = (
    "artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/"
    "15933802/checkpoint-step-54064.pt"
)
CONFIG = "configs/slurm/pretrain_1m_phase3_recovery_20260823.yaml"
OLD_REPORT_MD = "artifacts/codex/ht_pretraining_1m_phase3_recovery_20260823.md"
OLD_REPORT_JSON = "artifacts/codex/ht_pretraining_1m_phase3_recovery_20260823.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d"
)
EXPECTED_CONFIG_SHA256 = (
    "eced70932466ea07783122f7d2bce7fb344c87f45e72e57622e6363df3a2ad3f"
)
EXPECTED_CONTRACT_CANONICAL_SHA256 = (
    "20805cd37f914ea9ffb85789a200188bf23b1f6ee23e38067e5512f16393ac94"
)
EXPECTED_CONTRACT_FILE_SHA256 = (
    "2dec2fc5c793230d9decde5f41a6b9e2c83cdc6b6237c1cd5a9145cd1f46857c"
)
EXPECTED_HISTORICAL_COMMIT = "93b71c5d7c1bc20181640aafb4e918abb9267362"
EXPECTED_IMPLEMENTATION_COMMIT = "88b4fcdbd8bec2c1cd772c3e45742aa39ff077b7"
EXPECTED_ARTIFACT_PARENT_COMMIT = "fcf19adf16b1492fc1c0478fc79fe358e13d1809"
EXPECTED_MISSING_SOURCE_COMMIT = "f4e54df23b5c60115e475c5d68df4651899d678e"
EXPECTED_MISSING_SOURCE_TREE = "b6e3a4118b960e3a4676a61af9601438d56cef96"
EXPECTED_EXPERIMENT = "ht-pretrain-1m-phase3-recovery-20260823"
EXPECTED_GRES = "gpu:h100nvl:1"
ALLOWED_GRES = ("gpu:h100nvl:1", "gpu:v100:1")
EXPECTED_PHASE_STEPS = [27032, 27032, 27032, 27032]
EXPECTED_MILESTONES = [
    {"kind": kind, "step": step}
    for step in (13516, 27032, 40548, 54064, 67580, 81096, 94612, 108128)
    for kind in ("validation", "checkpoint")
]
EXPECTED_PROVENANCE_BLOCKER = (
    "The production source object f4e54df23b5c60115e475c5d68df4651899d678e "
    "remains unavailable locally; independently recover it and verify tree "
    "b6e3a4118b960e3a4676a61af9601438d56cef96."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(payload: dict[str, Any], *, digest_field: str) -> str:
    canonical = {key: value for key, value in payload.items() if key != digest_field}
    return sha256_bytes(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _repo_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"authorization path must remain repository-relative: {value}")
    return root / path


def _verify_file_binding(
    item: dict[str, Any], *, root: Path, path_key: str = "path", hash_key: str = "sha256"
) -> Path:
    path_value = item.get(path_key)
    expected = item.get(hash_key)
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise RuntimeError("authorization file binding is incomplete")
    path = _repo_path(path_value, root=root)
    if not path.is_file():
        raise RuntimeError(f"authorization-bound file is missing: {path_value}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"authorization-bound file hash changed: {path_value}")
    return path


def _verify_tag(tag: str, commit: str, *, root: Path) -> None:
    result = subprocess.run(
        ("git", "rev-list", "-n", "1", tag),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != commit:
        raise RuntimeError(f"preserved tag does not identify its immutable commit: {tag}")


def _verify_iso_timestamp(value: object) -> None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("authorization timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise RuntimeError("authorization timestamp must include an offset")


def _verify_contract_digest(contract: dict[str, Any]) -> str:
    stored = str(contract.get("contract_sha256", ""))
    actual = canonical_sha256(contract, digest_field="contract_sha256")
    if stored != actual:
        raise RuntimeError("recovery contract hash mismatch")
    return stored


def _verify_exact_contract(contract: dict[str, Any]) -> None:
    contract_digest = _verify_contract_digest(contract)
    if contract_digest != EXPECTED_CONTRACT_CANONICAL_SHA256 and not contract.get(
        "authorization_parent_contract"
    ):
        raise RuntimeError("recovery contract canonical hash differs from preserved parent")
    if contract.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError("authorization is bound to a different recovery experiment")
    if contract.get("submission_authorized") is not True:
        raise RuntimeError("corrected recovery contract is not submission-authorized")
    if contract.get("submission_performed") is not False:
        raise RuntimeError("recovery contract records a performed submission")
    if contract.get("sealed_test_role_access") != "forbidden":
        raise RuntimeError("recovery contract does not forbid sealed-test access")
    if contract.get("initialization_policy") != "exact_resume_from_checkpoint":
        raise RuntimeError("recovery contract is not exact-resume")
    if contract.get("gres") != EXPECTED_GRES:
        raise RuntimeError("recovery contract does not bind the authorized H100 NVL")
    if contract.get("resume_checkpoint") != CHECKPOINT:
        raise RuntimeError("recovery contract checkpoint path changed")
    if contract.get("resume_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("recovery contract checkpoint hash changed")
    if contract.get("expected_curriculum_phase_steps") != EXPECTED_PHASE_STEPS:
        raise RuntimeError("recovery contract phase schedule changed")
    if contract.get("expected_validation_checkpoint_milestones") != EXPECTED_MILESTONES:
        raise RuntimeError("recovery contract milestone set changed")
    preflight = contract.get("fullscale_gpu_preflight", {})
    if preflight.get("submission_blocked_until_pass") is not True:
        raise RuntimeError("recovery contract does not block until GPU preflight")
    if preflight.get("required_gres") != EXPECTED_GRES:
        raise RuntimeError("recovery contract preflight GRES changed")
    provenance = contract.get("provenance_validation", {})
    if provenance.get("scientific_slurm_submission_allowed") is not False:
        raise RuntimeError("authorization changed the structural provenance gate")
    if provenance.get("expected_missing_source_commit") != EXPECTED_MISSING_SOURCE_COMMIT:
        raise RuntimeError("authorization changed the missing provenance commit")
    if provenance.get("expected_missing_source_tree") != EXPECTED_MISSING_SOURCE_TREE:
        raise RuntimeError("authorization changed the missing provenance tree")
    lineage = contract.get("recovery_lineage", {})
    if lineage.get("historical_job_id") != "15933802":
        raise RuntimeError("recovery contract lost failed job 15933802")
    if lineage.get("source_checkpoint_step") != 54064:
        raise RuntimeError("recovery contract lost checkpoint step 54064")
    if lineage.get("replacement_attempt_root_must_not_be") != (
        "artifacts/slurm/jobs/15933802/attempt-00"
    ):
        raise RuntimeError("recovery output isolation binding changed")
    output = contract.get("output_contract", {})
    if output.get("no_silent_overwrite") is not True:
        raise RuntimeError("recovery output contract permits silent overwrite")


def verify_preflight(
    preflight_path: Path,
    *,
    authorization_path: Path,
    contract_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    payload = _load_json(preflight_path)
    if payload.get("preflight_version") != PREFLIGHT_VERSION:
        raise RuntimeError("unsupported phase-3 in-allocation preflight artifact")
    if payload.get("preflight_sha256") != canonical_sha256(
        payload, digest_field="preflight_sha256"
    ):
        raise RuntimeError("in-allocation preflight hash mismatch")
    if payload.get("status") != "passed":
        raise RuntimeError("fresh in-allocation preflight did not pass")
    if payload.get("fresh_in_allocation") is not True:
        raise RuntimeError("preflight is not marked fresh and in allocation")
    if payload.get("gpu_pilot_completed") is not False:
        raise RuntimeError("preflight incorrectly claims a GPU pilot completed")
    if payload.get("never_cpu") is not True:
        raise RuntimeError("preflight does not enforce GPU-only execution")
    if payload.get("expected_gres") != EXPECTED_GRES or payload.get("observed_gres") != EXPECTED_GRES:
        raise RuntimeError("preflight GRES does not match the authorized contract")
    if payload.get("slurm_gpus_on_node") != "1":
        raise RuntimeError("preflight does not prove one allocated GPU")
    if not re.fullmatch(r"[0-9]+", str(payload.get("slurm_job_id", ""))):
        raise RuntimeError("preflight does not identify a Slurm allocation")
    if not re.fullmatch(r"[^,\s]+", str(payload.get("cuda_visible_devices", ""))):
        raise RuntimeError("preflight does not prove one visible CUDA device")
    _verify_iso_timestamp(payload.get("timestamp"))
    if payload.get("execution_contract_file_sha256") != file_sha256(contract_path):
        raise RuntimeError("preflight is bound to a different execution contract")
    if payload.get("authorization_artifact_file_sha256") != file_sha256(authorization_path):
        raise RuntimeError("preflight is bound to a different authorization artifact")
    authorization = _load_json(authorization_path)
    if payload.get("authorization_artifact_canonical_sha256") != authorization.get(
        "artifact_sha256"
    ):
        raise RuntimeError("preflight authorization canonical binding changed")
    return payload


def verify_authorization_artifact(
    authorization_path: Path,
    *,
    contract_path: Path | None = None,
    root: Path = ROOT,
    require_fresh_preflight: bool = False,
    preflight_path: Path | None = None,
) -> dict[str, Any]:
    authorization_path = authorization_path.resolve()
    contract_path = (contract_path or (root / BASE_CONTRACT)).resolve()
    payload = _load_json(authorization_path)
    if payload.get("artifact_version") != AUTHORIZATION_VERSION:
        raise RuntimeError("unsupported phase-3 execution authorization artifact")
    if payload.get("artifact_sha256") != canonical_sha256(
        payload, digest_field="artifact_sha256"
    ):
        raise RuntimeError("execution authorization artifact hash mismatch")
    if payload.get("authorization_basis") != "explicit_user_operator_instruction":
        raise RuntimeError("authorization basis is not the explicit operator instruction")
    if payload.get("submission_authorized") is not True:
        raise RuntimeError("execution authorization is not true")
    if payload.get("submission_performed") is not False:
        raise RuntimeError("execution authorization claims a performed submission")
    if payload.get("gpu_pilot_completed") is not False:
        raise RuntimeError("operator authorization incorrectly implies pilot completion")
    if payload.get("fresh_in_allocation_preflight_required") is not True:
        raise RuntimeError("fresh in-allocation preflight is not mandatory")
    if payload.get("execution_role_required") != "gpt-5.3-codex-spark medium":
        raise RuntimeError("execution role requirement changed")
    if payload.get("sealed_test_or_stress_payload_accessed") is not False:
        raise RuntimeError("authorization artifact does not prove sealed/stress isolation")
    if payload.get("structural_scientific_slurm_submission_allowed") is not False:
        raise RuntimeError("authorization artifact changes the structural provenance gate")
    _verify_iso_timestamp(payload.get("timestamp"))

    parents = payload.get("immutable_parent_hashes")
    if not isinstance(parents, dict):
        raise RuntimeError("immutable parent hashes are missing")
    for key, expected in {
        "historical_failed_commit": EXPECTED_HISTORICAL_COMMIT,
        "implementation_commit": EXPECTED_IMPLEMENTATION_COMMIT,
        "artifact_parent_commit": EXPECTED_ARTIFACT_PARENT_COMMIT,
    }.items():
        if parents.get(key) != expected:
            raise RuntimeError(f"immutable parent commit changed: {key}")
    for tag, commit in {
        "ht-pretraining-production-1m-h100-operator-authorized-20260821": EXPECTED_HISTORICAL_COMMIT,
        "ht-pretraining-1m-phase3-recovery-implementation-v2-20260823": EXPECTED_IMPLEMENTATION_COMMIT,
        "ht-pretraining-1m-phase3-recovery-20260823-final-v2": EXPECTED_ARTIFACT_PARENT_COMMIT,
    }.items():
        if parents.get("preserved_tags", {}).get(tag) != commit:
            raise RuntimeError(f"preserved tag binding changed: {tag}")
        _verify_tag(tag, commit, root=root)

    bound_files = payload.get("immutable_parent_files")
    if not isinstance(bound_files, list):
        raise RuntimeError("immutable parent file hashes are missing")
    for item in bound_files:
        if not isinstance(item, dict):
            raise RuntimeError("invalid immutable parent file binding")
        _verify_file_binding(item, root=root)
    expected_file_hashes = {
        BASE_CONTRACT: EXPECTED_CONTRACT_FILE_SHA256,
        CHECKPOINT: EXPECTED_CHECKPOINT_SHA256,
        CONFIG: EXPECTED_CONFIG_SHA256,
    }
    for item in bound_files:
        if item.get("path") in expected_file_hashes and item.get("sha256") != expected_file_hashes[item["path"]]:
            raise RuntimeError(f"exact parent hash changed: {item['path']}")
    contract = _load_json(contract_path)
    _verify_exact_contract(contract)
    contract_file_hash = file_sha256(contract_path)
    base_contract_path = (root / BASE_CONTRACT).resolve()
    if contract_path == base_contract_path:
        if contract_file_hash != EXPECTED_CONTRACT_FILE_SHA256:
            raise RuntimeError("preserved recovery contract file hash changed")
    else:
        if contract.get("authorization_parent_contract") != BASE_CONTRACT:
            raise RuntimeError("execution contract does not identify the preserved parent")
        if contract.get("authorization_parent_contract_file_sha256") != EXPECTED_CONTRACT_FILE_SHA256:
            raise RuntimeError("execution contract parent contract hash changed")
        if contract.get("execution_authorization_artifact") != str(
            authorization_path.relative_to(root)
        ):
            raise RuntimeError("execution contract is not bound to the authorization artifact")
        if contract.get("authorization_artifact_sha256") != file_sha256(authorization_path):
            raise RuntimeError("execution contract authorization hash changed")
        if contract.get("fresh_in_allocation_preflight_required") is not True:
            raise RuntimeError("execution contract does not require fresh preflight")
        if contract.get("gpu_pilot_completed") is not False:
            raise RuntimeError("execution contract implies pilot completion")

    if require_fresh_preflight:
        if preflight_path is None:
            raise RuntimeError("fresh in-allocation preflight is required before launch")
        verify_preflight(
            preflight_path.resolve(),
            authorization_path=authorization_path,
            contract_path=contract_path,
            root=root,
        )
    return payload
