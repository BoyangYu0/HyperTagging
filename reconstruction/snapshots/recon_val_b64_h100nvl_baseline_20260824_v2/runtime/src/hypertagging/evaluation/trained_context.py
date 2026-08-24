"""Contract-checked preparation for trained held-out reconstruction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Sequence

import torch

from hypertagging.data.heterogeneous import (
    HeterogeneousEvent,
    collate_heterogeneous_events,
)
from hypertagging.data.streaming import RuntimeFeatureNormalizer
from hypertagging.data.dataset_index import load_dataset_index
from hypertagging.data.splitting import SourceAwareSplitConfig, stable_split_name
from hypertagging.models.ablation import ALL_ABLATIONS, build_ablation_model
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.data_module import RealDataModule, build_real_data_module
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
    target_policy = str(training_config.get("target_policy", "complete_only"))
    index_payload = load_dataset_index(dataset_index)
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

    events: list[HeterogeneousEvent] = []
    for event in data_module.iter_events(split, shuffle=False):
        events.append(event)
        if len(events) >= max_events:
            break
    if not events:
        raise ValueError(f"held-out {split} selection is empty")
    evaluated_uids = [event.event_uid for event in events]
    if len(evaluated_uids) != len(set(evaluated_uids)):
        raise ValueError("evaluation event UIDs are not unique")
    assigned_splits = {
        event.event_uid: stable_split_name(
            {
                "event_uid": event.event_uid,
                "source_file": event.source_file,
                "source_category": event.source_category,
            },
            split_config,
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
    pid_mode = str(training_config.get("pid_kinematics_mode", model.pid_kinematics_mode))
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
        "pid_kinematics_mode": pid_mode,
        "rollout_pid_kinematics_mode": rollout_pid_mode,
        "evaluation_split": split,
        "evaluated_event_uids": evaluated_uids,
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
        pid_kinematics_mode=pid_mode,
        rollout_pid_kinematics_mode=rollout_pid_mode,
        report_metadata=metadata,
    )


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"checkpoint {name} mismatch: {observed!r} != {expected!r}")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


__all__ = ["TrainedEvaluationContext", "load_trained_evaluation_context"]
