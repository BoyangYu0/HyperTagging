"""Independent, fail-closed verifier for joint terminal evidence v13.

The normative JSON spec is the only source of schema/key truth.  This module
does not import or alias v10/v11/v12 verifier semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

SPEC_FILE = "artifacts/codex/joint_optimization_20260824/joint_terminal_evidence_verifier_v13_spec_20260827.json"
SPEC_SHA256 = "63c230481a07408b9be68192e1ea3c2989f027906bd05d64a8abcd82c0ea3583"
SCHEMA_COUNT = 38
ORACLE_COUNT = 194
FALSE_FLAGS = {
    "implementation_complete": False,
    "execution_complete": False,
    "feasibility": False,
    "submission_authorized": False,
    "operator_submission_authorized": False,
    "action_authorized": False,
}
AUTH_KEYS = ("submission_authorized", "execution_authorized", "scheduler_authorized", "payload_access_authorized", "scientific_execution_authorized", "root_final_go")
DENIAL_KEYS = ("sealed_test_used", "stress_used", "restricted_raw_used", "restricted_source_used", "train_loss_used")
EXTENDED_AUTH_KEYS = AUTH_KEYS + ("feasibility_execution_authorized", "recovery_authorized", "promotion_authorized")
_HEX = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(ValueError):
    """A fail-closed evidence error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise VerificationError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def parse_json_bytes(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid UTF-8/JSON") from exc


def jcs(value: Any) -> bytes:
    """Canonical JSON subset used by the contracts (RFC8785-compatible data)."""
    # Normative records contain only strings, booleans, integers, arrays and
    # objects. Rejecting floats avoids Python's non-JCS number spellings.
    def check(v: Any) -> None:
        if isinstance(v, float):
            if not math.isfinite(v):
                raise VerificationError("non-finite JCS number")
            raise VerificationError("floating values are not accepted in exact records")
        if isinstance(v, str) and "\x00" in v:
            raise VerificationError("NUL in JCS string")
        if isinstance(v, Mapping):
            for k, x in v.items():
                if not isinstance(k, str):
                    raise VerificationError("non-string JCS key")
                check(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                check(x)
    check(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def digest_projection(value: Mapping[str, Any], keys: Iterable[str]) -> str:
    names = list(keys)
    return sha256_bytes(jcs({k: value[k] for k in names}))


def exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{label} is not an object")
    wanted = list(expected)
    if list(value.keys()) != wanted:
        raise VerificationError(f"{label} keys/order differ from v13 contract")


def require_hex(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise VerificationError(f"{label} is not lowercase sha256")


def validate_artifact_ref(ref: Mapping[str, Any], trusted_roots: Iterable[str] = ()) -> None:
    """Validate an ArtifactRef.v5 before any content consumer opens it."""
    exact_keys(ref, ("path", "byte_length", "sha256", "media_type", "schema", "version"), "ArtifactRef.v5")
    path = ref["path"]
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path or "/../" in (path + "/"):
        raise VerificationError("ArtifactRef path is not an absolute normalized path")
    if isinstance(ref["byte_length"], bool) or not isinstance(ref["byte_length"], int) or ref["byte_length"] <= 0:
        raise VerificationError("ArtifactRef byte_length is not positive uint")
    require_hex(ref["sha256"], "ArtifactRef sha256")
    if isinstance(ref["version"], bool) or not isinstance(ref["version"], int) or ref["version"] <= 0:
        raise VerificationError("ArtifactRef version is not positive uint")
    roots = [Path(r).resolve() for r in trusted_roots]
    target = Path(path)
    if roots and not any(target.is_relative_to(root) for root in roots):
        raise VerificationError("ArtifactRef escapes trusted root")
    if not target.exists():
        raise VerificationError("ArtifactRef target is absent")
    st = target.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise VerificationError("ArtifactRef is not a regular nonsymlink file")
    if st.st_size != ref["byte_length"] or sha256_bytes(target.read_bytes()) != ref["sha256"]:
        raise VerificationError("ArtifactRef bytes do not match bound length/hash")


def load_spec(repo: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(repo) / SPEC_FILE
    raw = path.read_bytes()
    if sha256_bytes(raw) != SPEC_SHA256:
        raise VerificationError("v13 normative spec hash mismatch")
    spec = parse_json_bytes(raw)
    if not isinstance(spec, dict):
        raise VerificationError("spec root is not object")
    validate_schema_registry(spec)
    if len(spec.get("test_oracle", [])) != ORACLE_COUNT:
        raise VerificationError("v13 oracle must contain exactly 194 tests")
    return spec


def validate_spec_bytes(raw: bytes) -> dict[str, Any]:
    if sha256_bytes(raw) != SPEC_SHA256:
        raise VerificationError("normative v13 spec hash mismatch")
    spec = parse_json_bytes(raw)
    if not isinstance(spec, dict):
        raise VerificationError("spec root is not object")
    validate_schema_registry(spec)
    return spec


def validate_schema_registry(spec: Mapping[str, Any]) -> None:
    schemas = spec.get("schemas")
    if not isinstance(schemas, Mapping) or len(schemas) != SCHEMA_COUNT:
        raise VerificationError("schema registry is not exactly the 38-schema v13 registry")
    required = {
        "ArtifactRef.v5", "ProducerBinding.v5", "CommonReceipt.v5", "RawCommandReceipt.v5",
        "SchedulerEvidence.v5", "ManifestEntry.v5", "SourceManifest.v5", "ArtifactManifest.v5",
        "RuntimeReceipt.v5", "TelemetryReceipt.v5", "ControllerReceipt.v5", "CheckpointReceipt.v5",
        "TerminalResultReceipt.v5", "EventScoreReceipt.v5", "TeacherEventSufficientStatistics.v11",
        "TeacherStatisticsReceipt.v11", "RolloutEventSufficientStatistics.v11", "RolloutStatisticsReceipt.v11",
        "ABBAEvaluationReceipt.v11", "SchedulerNativeReceipt.v11", "ControllerNativeReceipt.v11",
        "RuntimeNativeReceipt.v11", "TelemetryNativeReceipt.v11", "CheckpointNativeReceipt.v11",
        "ValidationNativeReceipt.v11", "MidpointNativeReceipt.v11", "ResourceNativeReceipt.v11",
        "TerminalNode.v5", "PairNode.v5", "LocatorNode.v5", "PilotNode.v5", "LadderNode.v5",
        "PointerNode.v5", "HpoNode.v5", "FinalNode.v5", "NormalizedTensorManifest.v12",
        "BatchPlanReceipt.v12", "StateIntegrityReceipt.v12",
    }
    if set(schemas) != required:
        raise VerificationError("v13 schema names differ from normative registry")


def validate_authorization(value: Mapping[str, Any], *, allow_operator: bool = True) -> None:
    auth = value.get("authorization")
    if not isinstance(auth, Mapping):
        raise VerificationError("authorization is not an object")
    if tuple(auth.keys()) not in (AUTH_KEYS, EXTENDED_AUTH_KEYS):
        raise VerificationError("authorization keys differ from v13 authorization_false")
    for key in auth:
        if auth.get(key) is not False:
            raise VerificationError(f"authorization flag {key} must remain false")
    if not allow_operator and "operator_submission_authorized" in auth:
        raise VerificationError("operator authorization is forbidden here")


def validate_usage_denials(value: Mapping[str, Any]) -> None:
    denials = value.get("usage_denials")
    if not isinstance(denials, Mapping) or list(denials.keys()) != list(DENIAL_KEYS) or any(v is not False for v in denials.values()):
        raise VerificationError("usage_denials must be the exact all-false v13 object")


def validate_common_receipt(spec: Mapping[str, Any], receipt: Mapping[str, Any], schema_name: str) -> None:
    """Validate one of the v13 typed receipt envelopes without broadening keys."""
    schema = spec["schemas"][schema_name]
    exact = schema.get("exact_keys")
    if exact is None:
        raise VerificationError(f"{schema_name} has no exact key declaration")
    exact_keys(receipt, exact, schema_name)
    if "authorization" in receipt:
        validate_authorization(receipt)
    if "usage_denials" in receipt:
        validate_usage_denials(receipt)


def validate_all_schema_objects(spec: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]) -> None:
    """Apply exact-key validation to all 38 schemas supplied by a producer."""
    if set(objects) != set(spec["schemas"]):
        raise VerificationError("producer did not provide exactly the 38 v13 schema objects")
    for name, obj in objects.items():
        schema = spec["schemas"][name]
        keys = schema.get("exact_keys") or schema.get("common_plus_exact_keys")
        if keys is None:
            raise VerificationError(f"schema {name} has no exact key list")
        if "common_plus_exact_keys" in schema:
            keys = list(spec["schemas"]["CommonReceipt.v5"]["exact_keys"]) + list(keys)
        exact_keys(obj, keys, name)
        if "authorization" in obj:
            validate_authorization(obj)
        if "usage_denials" in obj:
            validate_usage_denials(obj)


def validate_normalized_manifest(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    schema = spec["schemas"]["NormalizedTensorManifest.v12"]
    exact_keys(manifest, schema["exact_keys"], "normalized tensor manifest")
    for key, expected in schema["required"].items():
        if manifest.get(key) != expected:
            raise VerificationError(f"normalized manifest required value mismatch: {key}")
    universe = schema["compiled_tensor_key_universe"]
    groups = schema["compiled_tensor_dtype_groups"]
    if manifest["tensor_key_universe"] != universe:
        raise VerificationError("compiled tensor universe mismatch")
    if set().union(*map(set, groups.values())) != set(universe):
        raise VerificationError("dtype groups are not exhaustive")
    if sum(map(len, groups.values())) != len(universe):
        raise VerificationError("dtype groups overlap")
    if manifest["record_count"] != len(manifest["records"]):
        raise VerificationError("record count mismatch")
    if manifest["record_count"] != 118500 or len(manifest["batch_index"]) != 1500:
        raise VerificationError("compiled record/batch count mismatch")
    rec_keys = schema["record_exact_keys"]
    rec_proj = schema["record_digest_projection_exact_keys"]
    expected_records: list[tuple[str, int, int]] = []
    expected_records += [("teacher", n, i) for n in range(500) for i in range(79)]
    expected_records += [("rollout", n, i) for n in range(1000) for i in range(79)]
    if len(manifest["records"]) != len(expected_records):
        raise VerificationError("record universe length mismatch")
    for rec, (phase, ordinal, tensor_index) in zip(manifest["records"], expected_records):
        exact_keys(rec, rec_keys, "normalized record")
        if (rec["phase"], rec["ordinal"], rec["tensor_index"]) != (phase, ordinal, tensor_index):
            raise VerificationError("record order mismatch")
        if rec["batch_size"] != (4 if phase == "teacher" else 1):
            raise VerificationError("record batch size mismatch")
        if rec["tensor_key"] != universe[tensor_index]:
            raise VerificationError("record tensor key/index mismatch")
        expected_dtype = next(name for name, vals in groups.items() if rec["tensor_key"] in vals)
        if rec["dtype"] != expected_dtype:
            raise VerificationError("record dtype partition mismatch")
        if not isinstance(rec["tensor_file_ref"], Mapping):
            raise VerificationError("record tensor_file_ref is not ArtifactRef.v5")
        validate_artifact_ref(rec["tensor_file_ref"])
        require_hex(rec["tensor_sha256"], "tensor_sha256")
        require_hex(rec["record_sha256"], "record_sha256")
        if rec["record_sha256"] != digest_projection(rec, rec_proj):
            raise VerificationError("record digest mismatch")
    batch_keys = schema["batch_index_exact_keys"]
    batch_proj = schema["batch_index_digest_projection_exact_keys"]
    expected_batches = [("teacher", n) for n in range(500)] + [("rollout", n) for n in range(1000)]
    for batch, (phase, ordinal) in zip(manifest["batch_index"], expected_batches):
        exact_keys(batch, batch_keys, "batch index")
        if (batch["phase"], batch["ordinal"]) != (phase, ordinal):
            raise VerificationError("batch index order mismatch")
        if batch["batch_size"] != (4 if phase == "teacher" else 1):
            raise VerificationError("batch index size mismatch")
        expected = [r["record_sha256"] for r in manifest["records"] if r["phase"] == phase and r["ordinal"] == ordinal]
        if batch["tensor_record_sha256s"] != expected:
            raise VerificationError("batch index record projection mismatch")
        require_hex(batch["batch_sha256"], "batch_sha256")
        if batch["batch_sha256"] != digest_projection(batch, batch_proj):
            raise VerificationError("batch index digest mismatch")
    validate_authorization(manifest)
    validate_usage_denials(manifest)


def validate_batch_plan(spec: Mapping[str, Any], plan: Mapping[str, Any], normalized: Mapping[str, Any]) -> None:
    schema = spec["schemas"]["BatchPlanReceipt.v12"]
    exact_keys(plan, schema["exact_keys"], "batch plan")
    for key, expected in schema["required"].items():
        if plan.get(key) != expected:
            raise VerificationError(f"batch plan required value mismatch: {key}")
    if len(plan["batches"]) != 1500:
        raise VerificationError("batch plan count mismatch")
    batch_schema = schema["batch_exact_keys"]
    for i, batch in enumerate(plan["batches"]):
        exact_keys(batch, batch_schema, "batch plan batch")
        source = normalized["batch_index"][i]
        for key in ("phase", "ordinal", "batch_size", "event_uids"):
            if batch[key] != source[key]:
                raise VerificationError(f"batch plan mapping mismatch: {key}")
        if batch["normalized_batch_sha256"] != source["batch_sha256"]:
            raise VerificationError("batch plan normalized digest mismatch")
    validate_authorization(plan)
    validate_usage_denials(plan)


RUNTIME_STATE_KEYS = ("checkpoint", "model_state", "event_manifest", "normalized_tensor_manifest", "batch_plan", "runtime_manifest")
RUNTIME_STATE_FIELDS = {
    "checkpoint": "checkpoint_file_sha256", "model_state": "model_state_sha256",
    "event_manifest": "event_manifest_sha256", "normalized_tensor_manifest": "normalized_tensor_manifest_sha256",
    "batch_plan": "batch_plan_sha256", "runtime_manifest": "runtime_manifest_sha256",
}


def validate_state_integrity(spec: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    schema = spec["schemas"]["StateIntegrityReceipt.v12"]
    exact_keys(receipt, schema["exact_keys"], "state integrity receipt")
    if receipt["phase"] != "terminal":
        raise VerificationError("state integrity receipt must be terminal")
    targets = receipt["targets"]
    if not isinstance(targets, Mapping) or list(targets.keys()) != list(RUNTIME_STATE_KEYS):
        raise VerificationError("state integrity targets differ from runtime mapping")
    entry_keys = schema["entry_exact_keys"]
    for name in RUNTIME_STATE_KEYS:
        entry = targets[name]
        exact_keys(entry, entry_keys, f"state target {name}")
        require_hex(entry["sha256"], f"state target {name} sha256")
        if not entry["absolute_path"].startswith("/"):
            raise VerificationError("state target path must be absolute")
    validate_authorization(receipt)
    validate_usage_denials(receipt)


def validate_abba(spec: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    schema = spec["schemas"]["ABBAEvaluationReceipt.v11"]
    exact_keys(receipt, schema["exact_keys"], "ABBA receipt")
    if receipt["block_order"] != ["q32_A", "relbias_A", "relbias_B", "q32_B"]:
        raise VerificationError("ABBA block order mismatch")
    identity = receipt["repeated_arm_identity"]
    if not isinstance(identity, Mapping) or any(v is not True for v in identity.values()):
        raise VerificationError("ABBA repeated-arm identity is not proven")
    blocks = receipt["blocks"]
    if len(blocks) != 4:
        raise VerificationError("ABBA must contain four blocks")
    block_schema = schema["block_exact_keys"]
    content_schema = schema["scientific_content_exact_keys"]
    state_mapping = RUNTIME_STATE_FIELDS
    for block in blocks:
        exact_keys(block, block_schema, "ABBA block")
        exact_keys(block["scientific_content"], content_schema, "ABBA scientific content")
        if block["block_id"] not in ("q32_A", "relbias_A", "relbias_B", "q32_B"):
            raise VerificationError("ABBA block id is outside the fixed universe")
        expected_arm = "q32" if block["block_id"].startswith("q32") else "relbias"
        if block["arm"] != expected_arm or block["relation_bias_enabled"] is not (expected_arm == "relbias"):
            raise VerificationError("ABBA arm/relation-bias projection mismatch")
        for ref_name in ("source_contract_ref", "teacher_statistics_ref", "rollout_statistics_ref", "normalized_tensor_manifest_ref", "batch_plan_ref", "before_state_ref", "after_state_ref"):
            if not isinstance(block[ref_name], Mapping):
                raise VerificationError(f"ABBA {ref_name} is not an ArtifactRef.v5")
            validate_artifact_ref(block[ref_name])
        if block["block_ordinal"] != ("q32_A", "relbias_A", "relbias_B", "q32_B").index(block["block_id"]) + 1:
            raise VerificationError("ABBA block ordinal mismatch")
        exact_keys(block["timing"], schema["timing_exact_keys"], "ABBA timing")
        exact_keys(block["process_memory"], schema["process_memory_exact_keys"], "ABBA process memory")
        exact_keys(block["resource"], schema["resource_exact_keys"], "ABBA resource")
        if block["fresh_execve"] is not True or block["fresh_cuda_context"] is not True:
            raise VerificationError("ABBA block is not a fresh execution/context")
        ops = block["training_operations"]
        exact_keys(ops, schema["training_operations_exact_keys"], "ABBA training operations")
        if any(ops.get(k) not in (0, False) for k in ("backward_calls", "optimizer_constructed", "optimizer_steps", "scheduler_steps", "parameter_updates", "checkpoint_writes")):
            raise VerificationError("ABBA training operation is nonzero")
        for field in state_mapping.values():
            require_hex(block["scientific_content"].get(field), f"ABBA scientific content {field}")
        require_hex(block["scientific_content"]["source_contract_sha256"], "ABBA source contract")
        require_hex(block["block_digest"], "ABBA block_digest")
        require_hex(block["scientific_content_digest"], "ABBA scientific_content_digest")
    validate_authorization(receipt)
    validate_usage_denials(receipt)


def validate_native_receipt(schema_name: str, spec: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    schema = spec["schemas"][schema_name]
    exact_keys(receipt, schema["exact_keys"], schema_name)
    validate_authorization(receipt)
    validate_usage_denials(receipt)


def validate_chain(repo: str | os.PathLike[str], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the v13 envelope; all flags are deliberately non-authorizing."""
    spec = load_spec(repo)
    if evidence.get("spec_sha256") != SPEC_SHA256:
        raise VerificationError("evidence is not bound to v13 spec")
    validate_authorization(evidence, allow_operator=False)
    validate_usage_denials(evidence)
    if "normalized_tensor_manifest" in evidence:
        validate_normalized_manifest(spec, evidence["normalized_tensor_manifest"])
    if "batch_plan" in evidence and "normalized_tensor_manifest" in evidence:
        validate_batch_plan(spec, evidence["batch_plan"], evidence["normalized_tensor_manifest"])
    if "state_integrity" in evidence:
        validate_state_integrity(spec, evidence["state_integrity"])
    if "abba" in evidence:
        validate_abba(spec, evidence["abba"])
    return {"implementation_complete": False, "execution_complete": False, "feasibility": False, "submission_authorized": False}


__all__ = [
    "SPEC_FILE", "SPEC_SHA256", "VerificationError", "jcs", "load_spec", "validate_spec_bytes",
    "validate_schema_registry", "validate_normalized_manifest", "validate_batch_plan",
    "validate_state_integrity", "validate_abba", "validate_native_receipt", "validate_chain",
]
