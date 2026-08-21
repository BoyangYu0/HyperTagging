#!/usr/bin/env python
"""Fail-closed prologue verification for a rendered one-GPU job contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _load_gpu_safety_module() -> ModuleType:
    """Load the pure-stdlib safety module without importing package __init__."""

    module_name = "_hypertagging_slurm_gpu_safety"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = ROOT / "src/hypertagging/utils/gpu_safety.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load standalone GPU safety verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_local_microtest_completion_receipt(*args, **kwargs):
    """Lazily dispatch receipt validation for the scientific-only path."""

    return _load_gpu_safety_module().load_local_microtest_completion_receipt(
        *args, **kwargs
    )

CONTRACT_VERSION = "hypertagging-slurm-one-gpu-contract-v2"
OPERATOR_AUTHORIZATION_DATE = "2026-08-21"
OPERATOR_AUTHORIZATION_SOURCE = "interactive_user_instruction"
OPERATOR_AUTHORIZATION_SCOPE = (
    "exactly_one_production_1m_pretraining_job_on_gpu:h100nvl:1"
)
EXPECTED_MISSING_PROVENANCE_COMMIT = (
    "f4e54df23b5c60115e475c5d68df4651899d678e"
)
EXPECTED_MISSING_PROVENANCE_TREE = "b6e3a4118b960e3a4676a61af9601438d56cef96"
RUNTIME_FIELDS = (
    "gpu_environment",
    "gres",
    "train_config",
    "experiment",
    "seed",
    "max_restarts",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_rendered_contract_hash(contract: dict[str, object]) -> str:
    payload = dict(contract)
    stored = str(payload.pop("contract_sha256", ""))
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual:
        raise RuntimeError("rendered job contract hash mismatch")
    return stored


def verify_hashed_inputs(
    hashed_inputs: list[dict[str, str]], *, root: Path = ROOT
) -> None:
    for item in hashed_inputs:
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"hashed job input changed: {item['path']}")


def _load_hashed_manifest(path: Path, *, expected_version: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.get("manifest_hash", ""))
    canonical = {key: value for key, value in payload.items() if key != "manifest_hash"}
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored != actual or payload.get("manifest_version") != expected_version:
        raise RuntimeError(f"invalid hashed readiness manifest: {path}")
    return payload


def _safe_absolute_path(value: object, *, field: str) -> str:
    path = Path(str(value))
    if not path.is_absolute() or "\n" in str(path) or "\r" in str(path):
        raise RuntimeError(f"contract {field} must be a newline-free absolute path")
    return str(path)


def _safe_relative_path(value: object, *, field: str) -> str:
    path = Path(str(value))
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\n" in str(path)
        or "\r" in str(path)
    ):
        raise RuntimeError(f"contract {field} must remain inside the repository")
    return str(path)


def validated_runtime_values(contract: dict[str, object]) -> dict[str, str]:
    missing = [field for field in RUNTIME_FIELDS if field not in contract]
    if missing:
        raise RuntimeError(f"rendered job contract lacks runtime fields: {missing}")
    gpu_environment = _safe_absolute_path(
        contract["gpu_environment"], field="gpu_environment"
    )
    train_config = _safe_relative_path(contract["train_config"], field="train_config")
    if contract["gres"] not in {
        "gpu:h200nvl:1",
        "gpu:h100nvl:1",
        "gpu:v100:1",
    }:
        raise RuntimeError("contract contains unsupported or generic GRES")
    experiment = str(contract["experiment"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", experiment) is None:
        raise RuntimeError("contract experiment is not a safe job/run identifier")
    seed = int(contract["seed"])
    max_restarts = int(contract["max_restarts"])
    if seed < 0 or not 0 <= max_restarts <= 10:
        raise RuntimeError("contract seed/restart bounds are invalid")
    resume_value = contract.get("resume_checkpoint")
    resume_checkpoint = ""
    if resume_value is not None:
        resume_checkpoint = _safe_relative_path(
            resume_value, field="resume_checkpoint"
        )
        resume_path = ROOT / resume_checkpoint
        if resume_path.suffix != ".pt" or not resume_path.is_file():
            raise RuntimeError("contract resume checkpoint is not an existing .pt file")
    return {
        "gpu_environment": gpu_environment,
        "gres": str(contract["gres"]),
        "train_config": train_config,
        "experiment": experiment,
        "seed": str(seed),
        "max_restarts": str(max_restarts),
        "resume_checkpoint": resume_checkpoint,
    }


def verify_contract(
    contract_path: Path,
) -> tuple[dict[str, object], dict[str, str], str]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("unsupported rendered job contract")
    stored = verify_rendered_contract_hash(contract)
    if contract.get("export_policy") != "NIL":
        raise RuntimeError("rendered job contract requires exact NIL export policy")
    if contract.get("submission_authorized", True) is False and contract.get(
        "verification_scope"
    ) != "blocked_no_submit":
        raise RuntimeError("unauthorized contract lacks blocked no-submit scope")
    runtime = validated_runtime_values(contract)
    if _git("rev-parse", "HEAD") != contract["expected_git_sha"]:
        raise RuntimeError("job source Git SHA differs from rendered contract")
    if _git("status", "--porcelain"):
        raise RuntimeError("scientific/diagnostic Slurm jobs require a clean worktree")
    expected_tag = contract.get("expected_git_tag")
    if (
        expected_tag
        and _git("rev-list", "-n", "1", str(expected_tag))
        != contract["expected_git_sha"]
    ):
        raise RuntimeError(
            "immutable experiment tag does not identify expected Git SHA"
        )
    verify_hashed_inputs(contract["hashed_inputs"])
    if contract.get("mode") == "scientific":
        admission_path = contract.get("local_admission_receipt")
        completion_path = contract.get("local_completion_receipt")
        if not admission_path or not completion_path:
            raise RuntimeError(
                "scientific contract lacks admission/completion microtest evidence"
            )
        admission_path = _safe_absolute_path(
            admission_path, field="local_admission_receipt"
        )
        completion_path = _safe_absolute_path(
            completion_path, field="local_completion_receipt"
        )
        load_local_microtest_completion_receipt(
            completion_path,
            admission_path=admission_path,
        )
    if contract.get("fullscale"):
        if contract.get("mode") != "scientific":
            raise RuntimeError("full-scale contract must be scientific")
        if contract.get("gres") != "gpu:h100nvl:1":
            raise RuntimeError("full-scale contract must target exactly H100 NVL")
        if contract.get("initialization_policy") != "from_scratch":
            raise RuntimeError("full-scale scientific pretraining must start from scratch")
        if contract.get("partition_max_time") != "2-00:00:00":
            raise RuntimeError("full-scale contract must bind the two-day partition limit")
        if contract.get("resource_contract") != {
            "cpus_per_task": 8,
            "gres": "gpu:h100nvl:1",
            "memory": "64G",
            "partition": "inter",
            "requested_time": "2-00:00:00",
        }:
            raise RuntimeError("full-scale resource contract is not exact")
        if contract.get("output_contract") != {
            "attempt_root_template": "artifacts/slurm/jobs/{slurm_job_id}/attempt-{restart_count:02d}",
            "contract_copy": "provenance/job-contract.json",
            "no_silent_overwrite": True,
            "required_attempt_receipt": "receipt.json",
            "required_checkpoint": "checkpoint.pt",
            "required_metrics": "metrics.jsonl",
            "required_signal_checkpoint": "signal-checkpoint.pt",
            "run_root_template": "artifacts/runs/{experiment}/{seed}/{slurm_job_id}",
        }:
            raise RuntimeError("full-scale output and receipt contract is not exact")
        resume_policy = contract.get("checkpoint_resume_policy", {})
        for key in (
            "checkpoint_at_optimizer_boundary",
            "pending_validation_serialized",
            "requeue_uses_signal_checkpoint",
            "no_silent_restart",
            "no_double_counting",
        ):
            if resume_policy.get(key) is not True:
                raise RuntimeError(f"full-scale resume policy lacks {key}")
        if contract.get("submission_performed") is not False:
            raise RuntimeError("full-scale handoff must not record a performed submission")
        provenance = contract.get("provenance_status", {})
        if provenance.get("scientific_slurm_submission_allowed") is not False:
            raise RuntimeError("full-scale contract lacks the blocked provenance status")
        if not any(
            "f4e54df23b5c60115e475c5d68df4651899d678e" in str(blocker)
            and "b6e3a4118b960e3a4676a61af9601438d56cef96" in str(blocker)
            for blocker in provenance.get("blockers", [])
        ):
            raise RuntimeError("full-scale contract lacks the exact provenance blocker")
        override = contract.get("stage_gate_override", {})
        if override.get("status") != "operator_directed_fullscale_advancement":
            raise RuntimeError("full-scale contract lacks the stage-gate override record")
        if override.get("technical_and_scientific_gates_preserved") is not True:
            raise RuntimeError("stage-gate override weakens required safety gates")
        provenance_validation = contract.get("provenance_validation")
        if contract.get("submission_authorized") is True:
            if not isinstance(provenance_validation, dict):
                raise RuntimeError(
                    "authorized full-scale contract lacks structural provenance status"
                )
            if provenance_validation.get("status") != "valid":
                raise RuntimeError("full-scale contract lacks valid structural provenance status")
            if provenance_validation.get("scientific_slurm_submission_allowed") is not False:
                raise RuntimeError("full-scale contract changes the structural provenance gate")
            if provenance_validation.get("expected_missing_source_commit") != EXPECTED_MISSING_PROVENANCE_COMMIT:
                raise RuntimeError("full-scale contract changes the missing provenance object")
            if provenance_validation.get("expected_missing_source_tree") != EXPECTED_MISSING_PROVENANCE_TREE:
                raise RuntimeError("full-scale contract changes the missing provenance tree")
            if provenance_validation.get("execution_authorization_does_not_modify_validator") is not True:
                raise RuntimeError("full-scale contract does not separate execution authorization from validation")
        if contract.get("submission_authorized") is False:
            if contract.get("verification_scope") != "blocked_no_submit":
                raise RuntimeError("blocked full-scale contract lacks blocked scope")
        else:
            if not contract.get("expected_git_tag"):
                raise RuntimeError("authorized full-scale contract requires an immutable Git tag")
            if contract.get("verification_scope") != "operator_authorized_execution_with_provenance_exception":
                raise RuntimeError("authorized full-scale contract lacks the operator exception scope")
            execution_authorization = contract.get("execution_authorization", {})
            if execution_authorization != {
                "basis": "operator_provenance_exception",
                "execution_authorized": True,
            }:
                raise RuntimeError("execution authorization is not bound to the operator exception")
            operator_exception = contract.get("operator_provenance_exception")
            if not isinstance(operator_exception, dict):
                raise RuntimeError("authorized full-scale contract lacks operator exception")
            if operator_exception.get("status") != "explicit_operator_authorized_exception":
                raise RuntimeError("operator provenance exception status is not exact")
            if operator_exception.get("authorization_date") != OPERATOR_AUTHORIZATION_DATE:
                raise RuntimeError("operator provenance exception date is not exact")
            if operator_exception.get("source") != OPERATOR_AUTHORIZATION_SOURCE:
                raise RuntimeError("operator provenance exception source is not exact")
            if operator_exception.get("scope") != OPERATOR_AUTHORIZATION_SCOPE:
                raise RuntimeError("operator provenance exception scope is not exact")
            if operator_exception.get("job_count") != 1:
                raise RuntimeError("operator provenance exception must bind exactly one job")
            if operator_exception.get("gres") != "gpu:h100nvl:1":
                raise RuntimeError("operator provenance exception GRES is not exact")
            if operator_exception.get("execution_authorized") is not True:
                raise RuntimeError("operator provenance exception does not authorize execution")
            if operator_exception.get("limitation") not in provenance.get("blockers", []):
                raise RuntimeError("operator exception does not retain the provenance limitation verbatim")
            structural = operator_exception.get("structural_provenance_validation", {})
            if structural != {
                "missing_source_commit": EXPECTED_MISSING_PROVENANCE_COMMIT,
                "missing_source_tree": EXPECTED_MISSING_PROVENANCE_TREE,
                "scientific_slurm_submission_allowed": False,
                "status": "valid",
                "validator_unchanged": True,
            }:
                raise RuntimeError("operator exception changes structural provenance evidence")
    return contract, runtime, stored


def write_shell_runtime(values: dict[str, str], destination: Path) -> None:
    names = {
        "gpu_environment": "gpu_environment",
        "gres": "expected_gres",
        "train_config": "train_config",
        "experiment": "experiment",
        "seed": "seed",
        "max_restarts": "max_restarts",
        "resume_checkpoint": "resume_checkpoint",
    }
    lines = [
        f"readonly {names[key]}={shlex.quote(value)}" for key, value in values.items()
    ]
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--shell-output", type=Path)
    args = parser.parse_args()
    contract, runtime, stored = verify_contract(args.contract.resolve())
    index_path = ROOT / contract["dataset_index"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selection = json.loads(
        (ROOT / contract["selection_manifest"]).read_text(encoding="utf-8")
    )
    if index.get("selection_contract", {}).get(
        "selection_manifest_hash"
    ) != selection.get("manifest_hash"):
        raise RuntimeError("dataset index is not bound to the selected manifest hash")
    identity = index.get("event_identity_validation", {})
    if (
        identity.get("status") != "passed"
        or identity.get("sealed_test_opened") is not False
        or identity.get("task_binding")
        != "selection_to_sidecar_to_completion_marker_validated"
    ):
        raise RuntimeError(
            "dataset UID/source gate is absent or sealed test was opened"
        )
    if index.get("selection_contract", {}).get("included_splits") != [
        "train",
        "validation",
    ]:
        raise RuntimeError("job index must be restricted to train and validation roles")
    if contract.get("fullscale"):
        if selection.get("selection_name") != "train_865k":
            raise RuntimeError("full-scale contract is not bound to train_865k")
        if selection.get("selection_includes_test") is not False:
            raise RuntimeError("full-scale selection includes sealed test")
        if selection.get("split_counts") != {
            "test": 0,
            "train": 865000,
            "validation": 50000,
        }:
            raise RuntimeError("full-scale selection counts are not exact")
        index_counts = index.get("split_counts", {})
        if (
            index_counts.get("train") != 865000
            or index_counts.get("validation") != 50000
            or index_counts.get("test", 0) != 0
            or set(index_counts) - {"train", "validation", "test"}
        ):
            raise RuntimeError("full-scale index counts are not exact")
        if contract.get("expected_optimizer_steps") != 108128:
            raise RuntimeError("full-scale optimizer-step contract is not exact")
        if contract.get("expected_presentations") != 1730048:
            raise RuntimeError("full-scale presentation contract is not exact")
        if contract.get("expected_validation_events") != 5000:
            raise RuntimeError("full-scale validation cohort is not exact")
    readiness = _load_hashed_manifest(
        ROOT
        / "configs/training_selection/production_1m_20260812/training_readiness.json",
        expected_version="hypertagging-training-readiness-v1",
    )
    if readiness.get("index_gate", {}).get("status") != "complete":
        raise RuntimeError(
            "tracked readiness manifest does not complete the index gate"
        )
    if readiness.get("capacity_gate", {}).get("status") != "complete":
        raise RuntimeError(
            "tracked readiness manifest does not complete the capacity gate"
        )
    provenance = _load_hashed_manifest(
        ROOT
        / "configs/training_selection/production_1m_20260812/provenance_status.json",
        expected_version="hypertagging-training-provenance-status-v1",
    )
    if not provenance.get("cpu_implementation_allowed", False):
        raise RuntimeError("training provenance status blocks all integration work")
    if contract["mode"] == "scientific" and not provenance.get(
        "scientific_slurm_submission_allowed", False
    ):
        user_authorization = contract.get("user_submission_authorization", {})
        if contract.get("submission_authorized", True) and not (
            isinstance(user_authorization, dict)
            and user_authorization.get("authorized") is True
            and contract.get("verification_scope") in {
                "user_authorized_execution_with_recorded_provenance_limitations",
                "operator_authorized_execution_with_provenance_exception",
            }
            and contract.get("scientific_submission_blockers")
        ):
            raise RuntimeError("scientific render is blocked by provenance status")
        if not contract.get("scientific_submission_blockers"):
            raise RuntimeError("blocked no-submit contract does not record blockers")
    if args.shell_output is not None and not contract.get(
        "submission_authorized", True
    ):
        raise RuntimeError(
            "blocked no-submit contract is verified but not authorized for execution"
        )
    if args.shell_output is not None:
        write_shell_runtime(runtime, args.shell_output)
    print(
        json.dumps(
            {
                "contract_verified": True,
                "contract_sha256": stored,
                "execution_authorized": contract.get("submission_authorized", True),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
