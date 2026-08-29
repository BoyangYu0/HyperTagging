"""Contract-checked preparation for trained held-out reconstruction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Sequence

import torch

from hypertagging.data.heterogeneous import (
    HeterogeneousEvent,
    collate_heterogeneous_events,
)
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.splitting import SourceAwareSplitConfig, stable_split_name
from hypertagging.models.ablation import ALL_ABLATIONS, build_ablation_model
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.data_module import (
    RealDataModule,
    _require_source_role_manifest_binding as _shared_source_role_preflight,
    build_real_data_module,
    preflight_dataset_index_data_binding,
)
from hypertagging.training.model_config import ModelArchitecture


@dataclass(frozen=True)
class TrainedEvaluationContext:
    model: LevelAutoregressiveReconstructor
    data_module: RealDataModule
    events: tuple[HeterogeneousEvent, ...]
    constraint_policy: ReconstructionConstraintPolicy
    checkpoint: dict[str, Any]
    split: str
    pid_kinematics_mode: str
    rollout_pid_kinematics_mode: str
    report_metadata: dict[str, Any]

    def collated_batch(self, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Apply the exact static and runtime normalization used in training."""

        batch = self.data_module.normalize_batch(
            collate_heterogeneous_events(self.events)
        )
        return {name: value.to(device) for name, value in batch.items()}

    def collated_event_batch(
        self, index: int, *, device: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        batch = self.data_module.normalize_batch(
            collate_heterogeneous_events([self.events[index]])
        )
        return {name: value.to(device) for name, value in batch.items()}


def load_trained_evaluation_context(
    *,
    checkpoint: str | Path,
    data: str | Path | Sequence[str | Path],
    dataset_index: str | Path,
    split: str = "test",
    max_events: int = 100,
    device: str | torch.device = "cpu",
    diagnostic_allow_external_independent_sample: bool = False,
    source_categories: Sequence[str] | None = None,
    event_selection: str = "auto",
) -> TrainedEvaluationContext:
    """Load a trained model and a deterministic held-out event selection.

    The diagnostic external-sample override relaxes only training split/hash
    identity. It does not relax schema, feature, PID, architecture,
    normalization, legacy-state, or reconstruction-policy checks.
    """

    if split not in {"validation", "test"}:
        raise ValueError("trained evaluation split must be validation or test")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    if event_selection not in {
        "auto",
        "checkpoint_rollout",
        "checkpoint_validation",
        "stream",
    }:
        raise ValueError(f"unknown evaluation event selection: {event_selection}")
    requested_categories = {
        str(category).strip() for category in (source_categories or ())
    }
    if "" in requested_categories:
        raise ValueError("source_categories must not contain empty values")
    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    current_spec = feature_spec_v4()
    checkpoint_spec = payload.get("feature_specification", {})
    _require_equal(
        "feature_spec_hash",
        checkpoint_spec.get("feature_spec_hash"),
        current_spec["feature_spec_hash"],
    )
    contract = payload.get("feature_contract", {})
    _require_equal(
        "model_feature_contract_hash",
        contract.get("model_feature_contract_hash"),
        current_spec["model_feature_contract_hash"],
    )
    _require_equal(
        "PID vocabulary version",
        payload.get("pid_vocabulary_version"),
        PID_VOCABULARY_VERSION,
    )
    if not payload.get("architecture"):
        raise ValueError("checkpoint has no serialized architecture contract")
    if not contract.get("reconstruction_constraint_policy"):
        raise ValueError("checkpoint has no reconstruction constraint policy")
    if payload.get("legacy_conflated_fraction", 0.0) and not diagnostic_allow_external_independent_sample:
        raise ValueError("legacy-conflated checkpoint is not data-compatible")
    if not payload.get("data_compatible_performance", False):
        raise ValueError("checkpoint is not marked data-compatible")
    normalizer_state = payload.get("normalizer_state", {})
    missing_normalizers = {"track", "cluster", "common", "composite"} - set(normalizer_state)
    if missing_normalizers:
        raise ValueError(
            f"checkpoint is missing normalizer blocks: {sorted(missing_normalizers)}"
        )

    training_config = payload.get("config", {})
    contract_pid_mode = contract.get("pid_reconstruction_mode")
    if not isinstance(contract_pid_mode, str) or not contract_pid_mode:
        raise ValueError("checkpoint has no authoritative PID reconstruction mode")
    configured_pid_mode = training_config.get("pid_kinematics_mode")
    if (
        configured_pid_mode is not None
        and str(configured_pid_mode) != contract_pid_mode
    ):
        raise ValueError(
            "checkpoint PID mode conflicts with authoritative feature contract"
        )
    pid_mode = contract_pid_mode
    target_policy = str(training_config.get("target_policy", "complete_only"))
    index_payload = _preflight_evaluation_data_binding(data, dataset_index)
    split_config = SourceAwareSplitConfig(**index_payload["split_config"])
    data_module = build_real_data_module(
        data,
        dataset_index=dataset_index,
        max_events=training_config.get("max_events"),
        seed=int(training_config.get("seed", 20260730)),
        split_config=split_config,
        normalization_state=normalizer_state,
        target_policy=target_policy,
        required_splits=(split,),
        allow_legacy_conflated=False,
    )
    schemas = set(data_module.source_schema_versions)
    checkpoint_schema = str(payload.get("preprocessing_schema_version", ""))
    if checkpoint_schema != "mixed" and schemas != {checkpoint_schema}:
        raise ValueError(
            "evaluation preprocessing schema differs from the checkpoint: "
            f"{sorted(schemas)} != {checkpoint_schema!r}"
        )
    index_hash = str((data_module.dataset_index or {}).get("index_hash", ""))
    if not index_hash:
        raise ValueError("trained evaluation requires a validated dataset index")
    split_hash_matches = payload.get("split_manifest_hash") == data_module.split_manifest_hash
    training_index_hash = str(payload.get("data_order_contract", {}).get("dataset_index_hash", ""))
    index_hash_matches = training_index_hash == index_hash
    if not diagnostic_allow_external_independent_sample and (
        not split_hash_matches or not index_hash_matches
    ):
        raise ValueError(
            "evaluation dataset/split contract differs from training; use "
            "diagnostic_allow_external_independent_sample=True only for a verified "
            "independent external sample"
        )

    cohort_uids, resolved_event_selection = _resolve_checkpoint_event_selection(
        payload,
        data_module=data_module,
        split=split,
        requested=event_selection,
    )
    events = _select_evaluation_events(
        data_module,
        split=split,
        max_events=max_events,
        requested_categories=requested_categories,
        cohort_uids=cohort_uids,
    )
    if not events:
        category_suffix = (
            f" for source categories {sorted(requested_categories)}"
            if requested_categories
            else ""
        )
        raise ValueError(f"held-out {split} selection is empty{category_suffix}")
    evaluated_uids = [event.event_uid for event in events]
    if len(evaluated_uids) != len(set(evaluated_uids)):
        raise ValueError("evaluation event UIDs are not unique")
    assigned_splits = {
        event.event_uid: _evaluation_split_assignment(
            event,
            split_config=split_config,
            event_split_overrides=data_module.split_overrides,
            source_split_overrides=data_module.source_split_overrides,
        )
        for event in events
    }
    leaked = sorted(uid for uid, assigned in assigned_splits.items() if assigned == "train")
    wrong_split = sorted(uid for uid, assigned in assigned_splits.items() if assigned != split)
    if leaked:
        raise ValueError(f"evaluation UIDs are assigned to training: {leaked[:5]}")
    if wrong_split:
        raise ValueError(
            f"evaluation UIDs do not belong to requested {split} split: {wrong_split[:5]}"
        )

    architecture = ModelArchitecture.from_dict(payload["architecture"])
    ablation = str(training_config.get("ablation", "full_revised"))
    if ablation not in ALL_ABLATIONS:
        raise ValueError(f"checkpoint has unknown architecture ablation {ablation!r}")
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
        type_conditioned_daughter_relation_bias=(
            architecture.type_conditioned_daughter_relation_bias
        ),
        pid_kinematics_mode=pid_mode,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_runtime_feature_normalizer(
        RuntimeFeatureNormalizer(
            common_mean=data_module.normalizers["common"].mean,
            common_std=data_module.normalizers["common"].std,
            composite_mean=data_module.normalizers["composite"].mean,
            composite_std=data_module.normalizers["composite"].std,
            common_count=data_module.normalizers["common"].count,
            composite_count=data_module.normalizers["composite"].count,
        )
    )
    device = torch.device(device)
    model.to(device).eval()
    policy = ReconstructionConstraintPolicy.from_dict(
        contract["reconstruction_constraint_policy"]
    )
    if model.pid_kinematics_mode != pid_mode:
        raise ValueError("restored model PID mode differs from feature contract")
    rollout_pid_mode = str(
        training_config.get(
            "rollout_pid_kinematics_mode", "soft_decision_hard_construction"
        )
    )
    metadata = {
        "git_sha": _git_sha(),
        "checkpoint_git_sha": payload.get("git_commit", "unknown"),
        "checkpoint_step": int(payload.get("step", 0)),
        "checkpoint_training_split_hash": payload.get("split_manifest_hash", ""),
        "checkpoint_dataset_index_hash": training_index_hash,
        "dataset_index_hash": index_hash,
        "preprocessing_schema_version": checkpoint_schema,
        "feature_spec_hash": current_spec["feature_spec_hash"],
        "model_feature_contract_hash": current_spec["model_feature_contract_hash"],
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "architecture": payload["architecture"],
        "reconstruction_constraint_policy": policy.to_dict(),
        "pid_kinematics_mode": model.pid_kinematics_mode,
        "rollout_pid_kinematics_mode": rollout_pid_mode,
        "evaluation_split": split,
        "evaluation_source_categories": sorted(requested_categories),
        "evaluation_event_selection": resolved_event_selection,
        "checkpoint_event_cohort_size": (
            len(cohort_uids) if cohort_uids is not None else None
        ),
        "evaluated_event_uids": evaluated_uids,
        "evaluated_event_uid_sha256": _uid_sequence_sha256(evaluated_uids),
        "evaluated_uid_split_assignments": assigned_splits,
        "evaluation_uid_train_overlap": [],
        "external_independent_sample_override": bool(
            diagnostic_allow_external_independent_sample
        ),
        "frame_dependent_observables_available": False,
        "frame_dependent_observables_reason": (
            "no verified frame, beam-energy, and channel-specific missing-particle contract"
        ),
    }
    return TrainedEvaluationContext(
        model=model,
        data_module=data_module,
        events=tuple(events),
        constraint_policy=policy,
        checkpoint=payload,
        split=split,
        pid_kinematics_mode=model.pid_kinematics_mode,
        rollout_pid_kinematics_mode=rollout_pid_mode,
        report_metadata=metadata,
    )


def _evaluation_split_assignment(
    event: HeterogeneousEvent,
    *,
    split_config: SourceAwareSplitConfig,
    event_split_overrides: dict[str, str],
    source_split_overrides: dict[str, str],
) -> str:
    """Repeat the loader's split decision without bypassing source-role manifests."""

    assigned = event_split_overrides.get(str(event.event_uid))
    if assigned is not None:
        return assigned
    assigned = source_split_overrides.get(str(event.source_file))
    if assigned is not None:
        return assigned
    return stable_split_name(
        {
            "event_uid": event.event_uid,
            "source_file": event.source_file,
            "source_category": event.source_category,
        },
        split_config,
    )


def _resolve_checkpoint_event_selection(
    payload: dict[str, Any],
    *,
    data_module: RealDataModule,
    split: str,
    requested: str,
) -> tuple[tuple[str, ...] | None, str]:
    selection = payload.get("validation_selection", {})
    has_checkpoint_cohort = (
        split == "validation"
        and selection.get("split") == "validation"
        and bool(selection.get("rollout_event_uids"))
    )
    resolved = (
        "checkpoint_rollout"
        if requested == "auto" and has_checkpoint_cohort
        else "stream" if requested == "auto" else requested
    )
    if resolved == "stream":
        return None, "stream_order_diagnostic"
    if split != "validation" or selection.get("split") != split:
        raise ValueError(
            f"{resolved} event selection is unavailable for split {split!r}"
        )
    if not bool(selection.get("deterministic", False)):
        raise ValueError("checkpoint validation event selection is not deterministic")
    checkpoint_manifest_hash = str(selection.get("selection_manifest_hash", ""))
    data_manifest_hash = str(data_module.selection_manifest_hash or "")
    if checkpoint_manifest_hash != data_manifest_hash:
        raise ValueError(
            "checkpoint validation cohort belongs to a different training-selection "
            "manifest"
        )
    field = (
        "rollout_event_uids"
        if resolved == "checkpoint_rollout"
        else "event_uids"
    )
    values = tuple(str(value) for value in selection.get(field, ()))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"checkpoint {field} must be nonempty and unique")
    return values, resolved


def _select_evaluation_events(
    data_module: RealDataModule,
    *,
    split: str,
    max_events: int,
    requested_categories: set[str],
    cohort_uids: tuple[str, ...] | None,
) -> list[HeterogeneousEvent]:
    if cohort_uids is None:
        output: list[HeterogeneousEvent] = []
        for event in data_module.iter_events(split, shuffle=False):
            if requested_categories and event.source_category not in requested_categories:
                continue
            output.append(event)
            if len(output) >= max_events:
                break
        return output

    rank = {uid: index for index, uid in enumerate(cohort_uids)}
    found: dict[str, HeterogeneousEvent] = {}
    seen: set[str] = set()
    cutoff: int | None = None
    for event in data_module.iter_events(split, shuffle=False):
        uid = str(event.event_uid)
        if uid not in rank:
            continue
        seen.add(uid)
        if not requested_categories or event.source_category in requested_categories:
            found[uid] = event
        ordered_found = sorted(found, key=rank.__getitem__)
        if len(ordered_found) >= max_events:
            cutoff = rank[ordered_found[max_events - 1]]
            if all(uid in seen for uid in cohort_uids[: cutoff + 1]):
                break
    ordered = sorted(found.values(), key=lambda event: rank[str(event.event_uid)])
    selected = ordered[:max_events]
    required = cohort_uids[: cutoff + 1] if cutoff is not None else cohort_uids
    missing_prefix = [uid for uid in required if uid not in seen]
    if missing_prefix:
        raise ValueError(
            "checkpoint validation cohort UIDs are absent from the bound "
            f"dataset: {missing_prefix[:5]}"
        )
    return selected


def _require_source_role_manifest_binding(
    data: str | Path | Sequence[str | Path],
    index_payload: dict[str, Any],
) -> None:
    """Prevent rehashing raw paths against a source-role-bound index."""

    _shared_source_role_preflight(data, index_payload)


def _preflight_evaluation_data_binding(
    data: str | Path | Sequence[str | Path],
    dataset_index: str | Path,
) -> dict[str, object]:
    """Delegate evaluation preflight to the shared metadata-only invariant."""

    return preflight_dataset_index_data_binding(data, dataset_index)


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"checkpoint {name} mismatch: {observed!r} != {expected!r}")


def _uid_sequence_sha256(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


__all__ = ["TrainedEvaluationContext", "load_trained_evaluation_context"]
