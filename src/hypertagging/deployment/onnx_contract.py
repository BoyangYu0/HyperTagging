"""Versioned and hash-checked ONNX deployment-bundle contract.

The basf2 runtime intentionally consumes JSON and ONNX only.  It never loads a
PyTorch checkpoint, pickle, training configuration object, or project virtual
environment.  This module is standard-library only so the same validation can
run in both the training and basf2 Python environments.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


BUNDLE_FORMAT_VERSION = "hypertagging-onnx-bundle-v1"
MODEL_FAMILY = "level-autoregressive-full-decay"
MAX_EXACT_PROPOSALS_PER_HYPOTHESIS = 12

# Fixed-width, batch-size-one deployment interface.  A bundle chooses its
# max_nodes/max_sources capacities; hypotheses are compacted and then padded.
MODEL_INPUT_NAMES: tuple[str, ...] = (
    "common_features",
    "common_availability",
    "track_features",
    "track_availability",
    "cluster_features",
    "cluster_availability",
    "klm_features",
    "klm_availability",
    "composite_features",
    "composite_availability",
    "daughter_input_pid_histogram",
    "daughter_input_pid_histogram_available",
    "node_kind_ids",
    "leaf_kinematics_mode_ids",
    "pid_labels",
    "level_ids",
    "p4",
    "charge",
    "parent_ids",
    "daughter_adjacency",
    "node_mask",
    "active",
    "copied",
    "node_ids",
    "reco_ids",
    "source_node_ids",
    "recursive_leaf_source_mask",
    "copied_from",
    "runtime_composite_type_source_ids",
    "allowed_type_mask",
    "type_logit_bias",
    "pointer_validity_mask",
)

MODEL_OUTPUT_NAMES: tuple[str, ...] = (
    "object_logits",
    "type_logits",
    "pointer_logits",
    "cardinality_logits",
    "confidence_logits",
    "leaf_pid_logits",
    "current_p4",
)

FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "common": (
        "px", "py", "pz", "energy", "mass", "charge", "reduced_pid",
        "level", "active", "copied", "n_daughters", "candidate_confidence",
    ),
    "track": (
        "fit_p_value", "d0", "z0", "phi0", "omega", "tan_lambda",
        "pid_log_likelihood_electron", "pid_log_likelihood_muon",
        "pid_log_likelihood_pion", "pid_log_likelihood_kaon",
        "pid_log_likelihood_proton", "energy_hypothesis_electron",
        "energy_hypothesis_muon", "energy_hypothesis_pion",
        "energy_hypothesis_kaon", "energy_hypothesis_proton",
    ),
    "cluster": (
        "cluster_energy", "theta", "phi", "time", "e9_over_e21",
        "n_crystals", "min_track_distance", "photon_hypothesis",
        "track_matched",
    ),
    "klm": (
        "energy", "momentum_magnitude", "x", "y", "z", "time", "layers",
        "innermost_layer", "associated_ecl_cluster",
    ),
    "composite": (
        "daughter_sum_px", "daughter_sum_py", "daughter_sum_pz",
        "daughter_sum_energy", "summed_charge", "daughter_count",
        "pointer_confidence_mean", "pointer_confidence_min",
        "copied_daughter_fraction", "full_truth_daughter_count",
        "retained_daughter_count", "reconstructed_daughter_count",
        "partial_missing_daughters",
    ),
}


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value with deterministic serialization."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def load_bundle_manifest(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a deployment manifest.

    File hashes are checked by the runtime after resolving paths relative to
    the manifest.  Keeping structural and filesystem checks separate lets the
    exporter validate a manifest before atomically publishing its directory.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ONNX bundle manifest must be a JSON object")
    _require_equal("format_version", payload.get("format_version"), BUNDLE_FORMAT_VERSION)
    _require_equal("model_family", payload.get("model_family"), MODEL_FAMILY)
    contract = _mapping(payload.get("contract"), "contract")
    for positive in ("max_nodes", "max_sources", "n_queries", "max_cardinality"):
        value = contract.get(positive)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"contract.{positive} must be a positive integer")
    levels = contract.get("levels")
    if not isinstance(levels, list) or not levels or any(
        not isinstance(level, int) or isinstance(level, bool) or level <= 0
        for level in levels
    ):
        raise ValueError("contract.levels must be a non-empty positive-integer list")
    if levels != list(range(1, max(levels) + 1)):
        raise ValueError("contract.levels must be contiguous and start at one")
    pid_tokens = contract.get("pid_tokens")
    if not isinstance(pid_tokens, list) or not pid_tokens or any(
        not isinstance(token, int) or isinstance(token, bool) for token in pid_tokens
    ):
        raise ValueError("contract.pid_tokens must be a non-empty integer list")
    if pid_tokens[0] != 0 or len(set(pid_tokens)) != len(pid_tokens):
        raise ValueError("contract.pid_tokens must start with unknown PDG 0 and be unique")
    feature_names = _mapping(contract.get("feature_names"), "contract.feature_names")
    for block, expected in FEATURE_NAMES.items():
        _require_equal(
            f"contract.feature_names.{block}", feature_names.get(block), list(expected)
        )
    _require_equal(
        "contract.model_input_names",
        contract.get("model_input_names"),
        list(MODEL_INPUT_NAMES),
    )
    _require_equal(
        "contract.model_output_names",
        contract.get("model_output_names"),
        list(MODEL_OUTPUT_NAMES),
    )
    models = _mapping(payload.get("models"), "models")
    expected_model_keys = {str(level) for level in levels}
    if set(models) != expected_model_keys:
        raise ValueError(
            "models must contain exactly one ONNX entry per configured level: "
            f"{sorted(models)} != {sorted(expected_model_keys)}"
        )
    for level, model in models.items():
        entry = _mapping(model, f"models.{level}")
        filename = entry.get("file")
        digest = entry.get("sha256")
        inputs = entry.get("runtime_inputs")
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError(f"models.{level}.file must be a non-empty relative path")
        if Path(filename).name != filename or ".." in Path(filename).parts:
            raise ValueError(f"models.{level}.file must stay inside the bundle directory")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"models.{level}.sha256 must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"models.{level}.sha256 is not hexadecimal") from exc
        if (
            not isinstance(inputs, list)
            or any(not isinstance(name, str) for name in inputs)
            or len(inputs) != len(set(inputs))
            or not set(inputs).issubset(MODEL_INPUT_NAMES)
        ):
            raise ValueError(f"models.{level}.runtime_inputs violate the input contract")
        canonical_inputs = [name for name in MODEL_INPUT_NAMES if name in inputs]
        if inputs != canonical_inputs:
            raise ValueError(f"models.{level}.runtime_inputs are not in canonical order")
    policy = _mapping(payload.get("reconstruction_policy"), "reconstruction_policy")
    if policy.get("beam_score") != "sum_confidence":
        raise ValueError("only the versioned sum_confidence beam score is supported")
    for name in ("beam_width", "max_proposals_per_hypothesis", "minimum_daughters"):
        value = policy.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"reconstruction_policy.{name} must be positive")
    if (
        policy["max_proposals_per_hypothesis"]
        > MAX_EXACT_PROPOSALS_PER_HYPOTHESIS
    ):
        raise ValueError(
            "reconstruction_policy.max_proposals_per_hypothesis exceeds the "
            f"exact-search limit of {MAX_EXACT_PROPOSALS_PER_HYPOTHESIS}"
        )
    if policy["minimum_daughters"] < 2:
        raise ValueError("reconstructed mothers require at least two daughters")
    if policy["minimum_daughters"] > contract["max_cardinality"]:
        raise ValueError("minimum_daughters exceeds the model cardinality capacity")
    for name in ("object_threshold", "pointer_threshold", "confidence_threshold"):
        value = policy.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"reconstruction_policy.{name} must be in [0, 1]")
    optional_type_threshold = policy.get("type_probability_threshold")
    if optional_type_threshold is not None and (
        not isinstance(optional_type_threshold, (int, float))
        or isinstance(optional_type_threshold, bool)
        or not math.isfinite(float(optional_type_threshold))
        or not 0.0 <= float(optional_type_threshold) <= 1.0
    ):
        raise ValueError(
            "reconstruction_policy.type_probability_threshold must be null or in [0, 1]"
        )
    n_types = len(pid_tokens)
    for name, allow_empty in (
        ("root_tokens", False),
        ("static_allowed_mother_tokens", False),
    ):
        _validate_index_vector(
            policy.get(name),
            name=f"reconstruction_policy.{name}",
            upper_bound=n_types,
            allow_empty=allow_empty,
        )
    if 0 in policy["static_allowed_mother_tokens"]:
        raise ValueError("unknown PID token cannot be an allowed mother")
    if not set(policy["root_tokens"]).issubset(policy["static_allowed_mother_tokens"]):
        raise ValueError("root tokens must be allowed mother tokens")
    observed_by_level = policy.get("allowed_mother_types_by_level")
    if not isinstance(observed_by_level, list):
        raise ValueError("allowed_mother_types_by_level must be a list")
    observed_levels: list[int] = []
    for item in observed_by_level:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("allowed_mother_types_by_level entries must be [level, tokens]")
        level, tokens = item
        if not isinstance(level, int) or isinstance(level, bool) or level not in levels:
            raise ValueError("allowed_mother_types_by_level contains an unknown level")
        _validate_index_vector(
            tokens,
            name=f"allowed_mother_types_by_level[{level}]",
            upper_bound=n_types,
            allow_empty=False,
        )
        if not set(tokens).issubset(policy["static_allowed_mother_tokens"]):
            raise ValueError("empirical mother types exceed the static ontology")
        observed_levels.append(level)
    if observed_levels != sorted(set(observed_levels)):
        raise ValueError("allowed_mother_types_by_level must be sorted and unique")
    token_charge = policy.get("token_charge")
    if (
        not isinstance(token_charge, list)
        or len(token_charge) != n_types
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in token_charge
        )
    ):
        raise ValueError("reconstruction_policy.token_charge has the wrong contract")
    for name in ("valid_leaf_node_kinds", "valid_composite_node_kinds"):
        _validate_index_vector(
            policy.get(name),
            name=f"reconstruction_policy.{name}",
            upper_bound=6,
            allow_empty=False,
        )
    for name in (
        "allow_fixed_hypothesis_unknown_kind",
        "use_learned_confidence",
        "root_requires_all_sources",
        "reject_recursive_source_conflicts",
        "require_lower_level_context",
    ):
        if not isinstance(policy.get(name), bool):
            raise ValueError(f"reconstruction_policy.{name} must be boolean")
    if not policy["reject_recursive_source_conflicts"]:
        raise ValueError("the deployment runtime requires recursive source rejection")
    if not policy["require_lower_level_context"]:
        raise ValueError("the deployment runtime requires lower-level context")
    _require_member(
        "reconstruction_policy.daughter_cardinality_policy",
        policy.get("daughter_cardinality_policy"),
        {"predicted", "threshold"},
    )
    _require_member(
        "reconstruction_policy.cardinality_insufficient_policy",
        policy.get("cardinality_insufficient_policy"),
        {"invalid", "reduce"},
    )
    _require_member(
        "reconstruction_policy.empirical_type_prior_mode",
        policy.get("empirical_type_prior_mode"),
        {"hard", "soft", "off"},
    )
    _require_member(
        "reconstruction_policy.initial_state_policy",
        policy.get("initial_state_policy"),
        {"unknown", "upsilon4s"},
    )
    _require_member(
        "reconstruction_policy.mother_charge_compatibility",
        policy.get("mother_charge_compatibility"),
        {"off", "soft", "hard", "soft_train_hard_rollout"},
    )
    normalization = _mapping(payload.get("normalization"), "normalization")
    for block in ("common", "track", "cluster", "composite"):
        values = _mapping(normalization.get(block), f"normalization.{block}")
        width = len(FEATURE_NAMES[block])
        for key in ("mean", "standard_deviation"):
            vector = values.get(key)
            if not isinstance(vector, list) or len(vector) != width:
                raise ValueError(
                    f"normalization.{block}.{key} must have width {width}"
                )
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise ValueError(f"normalization.{block}.{key} must be numeric")
        if any(float(value) <= 0.0 for value in values["standard_deviation"]):
            raise ValueError(f"normalization.{block}.standard_deviation must be positive")
    declared_hash = payload.get("manifest_payload_sha256")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ValueError("manifest_payload_sha256 is missing or malformed")
    unhashed = dict(payload)
    unhashed.pop("manifest_payload_sha256", None)
    actual_hash = canonical_json_sha256(unhashed)
    if actual_hash != declared_hash:
        raise ValueError(
            "manifest payload hash mismatch: "
            f"declared {declared_hash}, computed {actual_hash}"
        )
    return payload


def with_manifest_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a manifest copy with its canonical payload hash attached."""

    result = dict(payload)
    result.pop("manifest_payload_sha256", None)
    result["manifest_payload_sha256"] = canonical_json_sha256(result)
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")


def _require_member(name: str, actual: object, expected: set[str]) -> None:
    if not isinstance(actual, str) or actual not in expected:
        raise ValueError(f"{name} must be one of {sorted(expected)}")


def _validate_index_vector(
    value: object,
    *,
    name: str,
    upper_bound: int,
    allow_empty: bool,
) -> None:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or item >= upper_bound
            for item in value
        )
        or value != sorted(set(value))
    ):
        raise ValueError(f"{name} must be a sorted unique token-index list")


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "FEATURE_NAMES",
    "MAX_EXACT_PROPOSALS_PER_HYPOTHESIS",
    "MODEL_FAMILY",
    "MODEL_INPUT_NAMES",
    "MODEL_OUTPUT_NAMES",
    "canonical_json_sha256",
    "file_sha256",
    "load_bundle_manifest",
    "with_manifest_hash",
]
