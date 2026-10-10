"""Strict native hierarchy TRAIN-only audit loading, without model execution.

This is deliberately separate from the held-out evaluation loader. It preserves
its model/data contracts but permits only authenticated, explicitly frozen TRAIN
UIDs. Raw tree targets remain supervision/evaluation; callers must physically
project FSPs and keep source-column rebasing outside inference model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from hypertagging.data.heterogeneous import collate_heterogeneous_events
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.splitting import SourceAwareSplitConfig
from hypertagging.evaluation.checkpoint_pair import validate_checkpoint_pair
from hypertagging.models.ablation import ALL_ABLATIONS, build_ablation_model
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.data_module import (
    build_real_data_module,
    preflight_dataset_index_data_binding,
)
from hypertagging.training.model_config import ModelArchitecture
from scripts.phase84_train_isolation import authenticate_train_isolation


def checked(binding):
    if not {"path", "sha256"} <= set(binding):
        raise ValueError("Missing immutable path/hash")
    path = Path(binding["path"]).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != binding["sha256"]:
        raise ValueError("Immutable native input changed: " + str(path))
    return path


def train_uids(document):
    values = document.get("event_uids")
    if document.get("role") != "train" or not isinstance(values, list) or not values:
        raise ValueError("Explicit nonempty TRAIN UID manifest required")
    if any(not isinstance(uid, str) or not uid for uid in values) or len(
        set(values)
    ) != len(values):
        raise ValueError("Invalid/duplicate TRAIN UIDs")
    return tuple(values)


def train_paths(selection):
    if (
        selection.get("selection_includes_test") is not False
        or selection.get("normalizer_scope") != "train_split_only"
    ):
        raise ValueError("Sealed-test or normalization scope changed")
    base = Path(selection["data_root"]).resolve()
    paths = []
    for row in selection["entries"]:
        if row["split"] != "train":
            continue
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Training shard escapes source root")
        paths.append((base / relative).resolve())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Empty/duplicate TRAIN paths")
    return tuple(paths)


def validate_payload(payload):
    current = feature_spec_v4()
    feature = payload.get("feature_contract", {})
    for name, observed, expected in [
        (
            "feature_spec_hash",
            payload.get("feature_specification", {}).get("feature_spec_hash"),
            current["feature_spec_hash"],
        ),
        (
            "model_feature_contract_hash",
            feature.get("model_feature_contract_hash"),
            current["model_feature_contract_hash"],
        ),
        (
            "PID vocabulary",
            payload.get("pid_vocabulary_version"),
            PID_VOCABULARY_VERSION,
        ),
    ]:
        if observed != expected:
            raise ValueError("Native checkpoint " + name + " mismatch")
    if not payload.get("architecture") or not feature.get(
        "reconstruction_constraint_policy"
    ):
        raise ValueError("Missing native architecture/constraint contract")
    if payload.get("legacy_conflated_fraction", 0.0) or not payload.get(
        "data_compatible_performance", False
    ):
        raise ValueError("Checkpoint is not strict data-compatible")
    if {"track", "cluster", "common", "composite"} - set(
        payload.get("normalizer_state", {})
    ):
        raise ValueError("Missing train-fitted normalizers")
    pid_mode = feature.get("pid_reconstruction_mode")
    config = payload.get("config", {})
    if not isinstance(pid_mode, str) or not pid_mode:
        raise ValueError("Missing authoritative PID mode")
    if (
        config.get("pid_kinematics_mode") is not None
        and config["pid_kinematics_mode"] != pid_mode
    ):
        raise ValueError("Conflicting native PID mode")
    if config.get("max_events") is not None:
        raise ValueError("Prefix-trained checkpoint not admitted for this native audit")
    if config.get("target_policy") not in ("complete_only", "reconstructable_partial"):
        raise ValueError("Missing/native target policy mismatch")
    return pid_mode


def validate_module(payload, module):
    schemas = set(module.source_schema_versions)
    checkpoint_schema = payload.get("preprocessing_schema_version")
    if checkpoint_schema != "mixed" and schemas != {checkpoint_schema}:
        raise ValueError("Native preprocessing schema mismatch")
    index_hash = (module.dataset_index or {}).get("index_hash")
    if (
        not index_hash
        or payload.get("data_order_contract", {}).get("dataset_index_hash")
        != index_hash
    ):
        raise ValueError("Native training index mismatch")
    if payload.get("split_manifest_hash") != module.split_manifest_hash:
        raise ValueError("Native source-role split mismatch")


def select_events(module, uids, selected_paths):
    """Restrict I/O to authenticated TRAIN shards after full-index validation."""
    allowed = set(Path(p).resolve() for p in module.input_paths)
    if not set(selected_paths) <= allowed:
        raise ValueError("TRAIN paths absent from authenticated native module")
    # Retain original index, normalization, source-role overrides and split hashes.
    # Only physical read paths are restricted, so validation rows are never read.
    train_module = replace(module, input_paths=selected_paths)
    events = tuple(train_module.iter_events("train", shuffle=False, event_uids=uids))
    by_uid = {event.event_uid: event for event in events}
    if len(events) != len(by_uid) or set(by_uid) != set(uids):
        raise ValueError("Native TRAIN UID coverage/uniqueness mismatch")
    return train_module, tuple(by_uid[uid] for uid in uids)


def restore_model(payload, module, pid_mode):
    """Mirror native evaluator construction; load state without any forward."""
    architecture = ModelArchitecture.from_dict(payload["architecture"])
    config = payload["config"]
    ablation = str(config.get("ablation", "full_revised"))
    if ablation not in ALL_ABLATIONS:
        raise ValueError("Unknown native architecture ablation")
    model = build_ablation_model(
        ablation,
        n_features=12,
        n_types=len(PDG_TOKENS),
        hidden_dim=architecture.d_model,
        hyper_dim=architecture.hyper_dim,
        n_queries=architecture.n_queries,
        max_cardinality=architecture.max_cardinality,
        n_heads=architecture.n_heads,
        n_context_layers=architecture.n_context_layers,
        curvature=architecture.curvature,
        ffn_dim=architecture.ffn_dim,
        dropout=architecture.dropout,
        n_queries_by_level=architecture.n_queries_by_level,
        max_cardinality_by_level=architecture.max_cardinality_by_level,
        hyper_projection_init_scale=architecture.hyper_projection_init_scale,
        tangent_scale_mode=architecture.tangent_scale_mode,
        hyperbolic_level_encoding=architecture.hyperbolic_level_encoding,
        type_conditioned_daughter_relation_bias=architecture.type_conditioned_daughter_relation_bias,
        pid_kinematics_mode=pid_mode,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_runtime_feature_normalizer(
        RuntimeFeatureNormalizer(
            common_mean=module.normalizers["common"].mean,
            common_std=module.normalizers["common"].std,
            composite_mean=module.normalizers["composite"].mean,
            composite_std=module.normalizers["composite"].std,
            common_count=module.normalizers["common"].count,
            composite_count=module.normalizers["composite"].count,
        )
    )
    model.cpu().eval()
    if model.pid_kinematics_mode != pid_mode:
        raise ValueError("Restored native PID mode changed")
    return model


@dataclass(frozen=True)
class HierarchyTrainContext:
    model: object
    data_module: object
    events: tuple
    constraint_policy: ReconstructionConstraintPolicy
    checkpoint: dict
    config: dict
    normalizers: dict
    allowed_types_by_level: dict
    pid_kinematics_mode: str
    rollout_pid_kinematics_mode: str
    metadata: dict

    def collated_event_batch(self, index):
        """Native static normalization only; original tree/source axes retained."""
        return self.data_module.normalize_batch(
            collate_heterogeneous_events([self.events[index]])
        )


def load_hierarchy_train_context(
    *,
    checkpoint,
    pretraining_checkpoint,
    selection,
    dataset_index,
    uid_manifest,
    repo_root,
):
    """Bindings are path+SHA256 dicts. Caller owns guarded CPU/runtime admission.

    No model forward, optimization, outcome-based selection, or model promotion.
    Full historical/current exclusion union is mandatory, never caller-only lists.
    """
    bindings = {
        "checkpoint": checkpoint,
        "pretraining_checkpoint": pretraining_checkpoint,
        "selection": selection,
        "dataset_index": dataset_index,
        "uid_manifest": uid_manifest,
    }
    paths = {name: checked(item) for name, item in bindings.items()}
    uids = train_uids(json.loads(paths["uid_manifest"].read_text()))
    isolation = authenticate_train_isolation(uids, Path(repo_root))
    selected_paths = train_paths(json.loads(paths["selection"].read_text()))
    lineage = validate_checkpoint_pair(
        paths["pretraining_checkpoint"],
        paths["checkpoint"],
        require_exact_frozen_encoder=False,
    ).as_dict()
    if not lineage["compatible"]:
        raise ValueError("Native pretraining/reconstruction lineage incompatible")
    payload = load_training_checkpoint(paths["checkpoint"], map_location="cpu")
    pid_mode = validate_payload(payload)
    config = payload["config"]
    index = preflight_dataset_index_data_binding(
        paths["selection"],
        paths["dataset_index"],
        required_splits=("train",),
        target_policy=config["target_policy"],
        scientific_mode=True,
    )
    module = build_real_data_module(
        paths["selection"],
        dataset_index=paths["dataset_index"],
        max_events=None,
        seed=int(config.get("seed", 20260730)),
        split_config=SourceAwareSplitConfig(**index["split_config"]),
        normalization_state=payload["normalizer_state"],
        target_policy=config["target_policy"],
        required_splits=("train",),
        allow_legacy_conflated=False,
        scientific_mode=True,
    )
    validate_module(payload, module)
    module, events = select_events(module, uids, selected_paths)
    policy = ReconstructionConstraintPolicy.from_dict(
        payload["feature_contract"]["reconstruction_constraint_policy"]
    )
    model = restore_model(payload, module, pid_mode)
    metadata = {
        "stage": "training_only_native_hierarchy_diagnostic",
        "bindings": bindings,
        "lineage": lineage,
        "isolation": isolation,
        "train_count": len(events),
        "ordered_training_uids": list(uids),
        "training_uid_set_sha256": hashlib.sha256(
            "\n".join(sorted(uids)).encode()
        ).hexdigest(),
        "read_paths": [str(p) for p in selected_paths],
        "validation_rows_read": 0,
        "sealed_test_access": False,
        "normalization_refit": False,
        "checkpoint_step": payload.get("step"),
        "architecture": payload["architecture"],
        "native_target_policy": config["target_policy"],
        "generation_height_semantics": "Preserved original native level_ids; not B-root depth/mass/cardinality",
        "source_axis_semantics": "Raw full-tree recursive source columns retained. Strict FSP projection may compact them; evaluation-only truth alignment must use identical kept columns.",
        "model_forward_count": 0,
        "optimizer_steps": 0,
    }
    return HierarchyTrainContext(
        model,
        module,
        events,
        policy,
        payload,
        config,
        module.normalizers,
        module.allowed_types_by_level,
        pid_mode,
        str(
            config.get("rollout_pid_kinematics_mode", "soft_decision_hard_construction")
        ),
        metadata,
    )
