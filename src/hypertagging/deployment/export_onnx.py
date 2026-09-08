"""Export a trained full-decay checkpoint as a basf2 ONNX bundle.

The deployment boundary is deliberately one-way: this exporter is allowed to
load the trusted PyTorch training checkpoint, while event processing consumes
only the resulting JSON manifest and per-level ONNX graphs.  Every graph has a
fixed batch size and capacity so the same bundle can be loaded by the minimal
basf2 runtime without importing the training environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import torch
from torch import nn

from hypertagging.deployment.onnx_contract import (
    BUNDLE_FORMAT_VERSION,
    FEATURE_NAMES,
    MAX_EXACT_PROPOSALS_PER_HYPOTHESIS,
    MODEL_FAMILY,
    MODEL_INPUT_NAMES,
    MODEL_OUTPUT_NAMES,
    file_sha256,
    load_bundle_manifest,
    with_manifest_hash,
)
from hypertagging.models.ablation import ALL_ABLATIONS, build_ablation_model
from hypertagging.preprocessing.pid_filter import (
    PDG_TOKENS,
    PID_VOCABULARY_VERSION,
)
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v4 import (
    CATEGORICAL_COMMON_FEATURE_NAMES,
    CONTINUOUS_COMMON_INDICES,
    DYNAMIC_COMPOSITE_INDICES,
    LEAF_MODE_TO_ID,
    SCHEMA_VERSION_V4,
    TARGET_COMPOSITE_METADATA_INDICES,
    feature_spec_v4,
)
from hypertagging.reconstruction.constraints import (
    REDUCED_TOKEN_CHARGE,
    ReconstructionConstraintPolicy,
)
from hypertagging.reconstruction.pid_state import COMPOSITE_TYPE_SOURCE_TO_ID
from hypertagging.training.checkpointing import load_training_checkpoint
from hypertagging.training.model_config import ModelArchitecture


DEFAULT_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
DEFAULT_BASF2_RELEASES: tuple[str, ...] = ("light-2607-kasei",)
DEFAULT_ONNX_OPSET = 18
EXPORTER_VERSION = "hypertagging-fixed-shape-onnx-v1"
NORMALIZED_BLOCKS: tuple[str, ...] = (
    "common",
    "track",
    "cluster",
    "composite",
)


@dataclass(frozen=True)
class ExportConfiguration:
    """Fixed-shape graph and host-side beam-search configuration."""

    levels: tuple[int, ...] = DEFAULT_LEVELS
    max_nodes: int = 128
    max_sources: int = 128
    beam_width: int = 4
    max_proposals_per_hypothesis: int = 12
    object_threshold: float = 0.5
    pointer_threshold: float | None = None
    confidence_threshold: float = 0.0
    type_probability_threshold: float | None = None
    opset_version: int = DEFAULT_ONNX_OPSET
    basf2_releases: tuple[str, ...] = DEFAULT_BASF2_RELEASES
    use_learned_confidence: bool | None = None

    def validated(self) -> "ExportConfiguration":
        levels = tuple(int(level) for level in self.levels)
        if not levels or levels != tuple(sorted(set(levels))):
            raise ValueError("levels must be a non-empty sorted unique sequence")
        if levels != tuple(range(1, levels[-1] + 1)):
            raise ValueError("full-decay deployment levels must be contiguous from 1")
        if self.max_nodes < max(6, levels[-1] + 4):
            raise ValueError(
                "max_nodes must leave room for detector leaves and one example "
                "composite at every lower level"
            )
        if self.max_sources <= 0:
            raise ValueError("max_sources must be positive")
        for name in ("beam_width", "max_proposals_per_hypothesis"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            int(self.max_proposals_per_hypothesis)
            > MAX_EXACT_PROPOSALS_PER_HYPOTHESIS
        ):
            raise ValueError(
                "max_proposals_per_hypothesis exceeds the exact-search limit "
                f"of {MAX_EXACT_PROPOSALS_PER_HYPOTHESIS}"
            )
        for name in (
            "object_threshold",
            "confidence_threshold",
            "type_probability_threshold",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when configured")
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.pointer_threshold is not None and not math.isfinite(
            float(self.pointer_threshold)
        ):
            raise ValueError("pointer_threshold must be finite when configured")
        if self.pointer_threshold is not None and not 0.0 <= float(
            self.pointer_threshold
        ) <= 1.0:
            raise ValueError("pointer_threshold must be in [0, 1]")
        if int(self.opset_version) < 18:
            raise ValueError("ONNX opset 18 or newer is required")
        releases = tuple(str(value).strip() for value in self.basf2_releases)
        if any(not value for value in releases):
            raise ValueError("basf2 release names must not be empty")
        return ExportConfiguration(
            levels=levels,
            max_nodes=int(self.max_nodes),
            max_sources=int(self.max_sources),
            beam_width=int(self.beam_width),
            max_proposals_per_hypothesis=int(
                self.max_proposals_per_hypothesis
            ),
            object_threshold=float(self.object_threshold),
            pointer_threshold=(
                None
                if self.pointer_threshold is None
                else float(self.pointer_threshold)
            ),
            confidence_threshold=float(self.confidence_threshold),
            type_probability_threshold=(
                None
                if self.type_probability_threshold is None
                else float(self.type_probability_threshold)
            ),
            opset_version=int(self.opset_version),
            basf2_releases=releases,
            use_learned_confidence=self.use_learned_confidence,
        )


class _LevelOnnxWrapper(nn.Module):
    """Expose one target-level invocation through the versioned ONNX ABI."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_level: int,
        pid_kinematics_mode: str,
        pid_temperature: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_level = int(target_level)
        self.pid_kinematics_mode = str(pid_kinematics_mode)
        self.pid_temperature = float(pid_temperature)

    def forward(  # noqa: PLR0913 - the signature is the deployment ABI
        self,
        common_features: torch.Tensor,
        common_availability: torch.Tensor,
        track_features: torch.Tensor,
        track_availability: torch.Tensor,
        cluster_features: torch.Tensor,
        cluster_availability: torch.Tensor,
        klm_features: torch.Tensor,
        klm_availability: torch.Tensor,
        composite_features: torch.Tensor,
        composite_availability: torch.Tensor,
        daughter_input_pid_histogram: torch.Tensor,
        daughter_input_pid_histogram_available: torch.Tensor,
        node_kind_ids: torch.Tensor,
        leaf_kinematics_mode_ids: torch.Tensor,
        pid_labels: torch.Tensor,
        level_ids: torch.Tensor,
        p4: torch.Tensor,
        charge: torch.Tensor,
        parent_ids: torch.Tensor,
        daughter_adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        active: torch.Tensor,
        copied: torch.Tensor,
        node_ids: torch.Tensor,
        reco_ids: torch.Tensor,
        source_node_ids: torch.Tensor,
        recursive_leaf_source_mask: torch.Tensor,
        copied_from: torch.Tensor,
        runtime_composite_type_source_ids: torch.Tensor,
        allowed_type_mask: torch.Tensor,
        type_logit_bias: torch.Tensor,
        pointer_validity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Provenance is intentionally fixed inside the exported graph.  It is
        # used only by the training-side truth-leakage assertion, never as a
        # learned feature.  Runtime inputs are reconstructed data (ID 3), while
        # truth supervision is explicitly unavailable (ID 0).
        input_provenance = torch.full_like(level_ids, 3)
        unavailable_truth = torch.zeros_like(level_ids)
        batch = {
            "common_features": common_features,
            "common_availability": common_availability,
            "track_features": track_features,
            "track_availability": track_availability,
            "cluster_features": cluster_features,
            "cluster_availability": cluster_availability,
            "klm_features": klm_features,
            "klm_availability": klm_availability,
            "composite_features": composite_features,
            "composite_availability": composite_availability,
            "daughter_input_pid_histogram": daughter_input_pid_histogram,
            "daughter_input_pid_histogram_available": (
                daughter_input_pid_histogram_available
            ),
            # The encoder keeps these compatibility aliases, but both point to
            # the explicit truth-free input histogram in deployment.
            "daughter_pid_histogram": daughter_input_pid_histogram,
            "daughter_pid_histogram_available": (
                daughter_input_pid_histogram_available
            ),
            "node_kind_ids": node_kind_ids,
            "leaf_kinematics_mode_ids": leaf_kinematics_mode_ids,
            "pid_labels": pid_labels,
            "level_ids": level_ids,
            "p4": p4,
            "charge": charge,
            "parent_ids": parent_ids,
            "daughter_adjacency": daughter_adjacency,
            "node_mask": node_mask,
            "active": active,
            "copied": copied,
            "node_ids": node_ids,
            "reco_ids": reco_ids,
            "source_node_ids": source_node_ids,
            "recursive_leaf_source_mask": recursive_leaf_source_mask,
            "copied_from": copied_from,
            "runtime_composite_type_source_ids": (
                runtime_composite_type_source_ids
            ),
            "allowed_type_mask": allowed_type_mask,
            "type_logit_bias": type_logit_bias,
            "pointer_validity_mask": pointer_validity_mask,
            "model_input_source_ids": input_provenance,
            "daughter_input_pid_source_ids": input_provenance,
            "truth_supervision_source_ids": unavailable_truth,
            "daughter_truth_pid_source_ids": unavailable_truth,
        }
        output = self.model(
            batch,
            target_level=self.target_level,
            pid_kinematics_mode_override=self.pid_kinematics_mode,
            pid_temperature_override=self.pid_temperature,
            return_attention=False,
        )
        if output.leaf_pid_logits is None or output.current_p4 is None:
            raise RuntimeError(
                "full-decay deployment requires leaf PID logits and current p4"
            )
        return (
            output.pointer.object_logits,
            output.pointer.type_logits,
            output.pointer.pointer_logits,
            output.pointer.cardinality_logits,
            output.pointer.confidence_logits,
            output.leaf_pid_logits,
            output.current_p4,
        )


class _OnnxRuntimeFeatureNormalizer(nn.Module):
    """Export-friendly equivalent of ``RuntimeFeatureNormalizer``.

    The training implementation uses indexed in-place updates.  PyTorch's
    legacy ONNX lowering emits invalid placeholder inputs for that pattern, so
    deployment expresses the exact same fixed-width transform with broadcast
    masks and ``where`` operations.
    """

    def __init__(self, normalizers: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        super().__init__()
        common = normalizers["common"]
        common_observed = common["count"] > 0
        common_mean = torch.where(
            common_observed, common["mean"], torch.zeros_like(common["mean"])
        )
        common_std = torch.where(
            common_observed,
            common["standard_deviation"],
            torch.ones_like(common["standard_deviation"]),
        )
        common_continuous = torch.zeros(
            len(FEATURE_NAMES["common"]), dtype=torch.bool
        )
        common_continuous[list(CONTINUOUS_COMMON_INDICES)] = True
        common_allowed = torch.ones_like(common_continuous)
        for name in CATEGORICAL_COMMON_FEATURE_NAMES:
            common_allowed[FEATURE_NAMES["common"].index(name)] = False

        composite = normalizers["composite"]
        composite_observed = composite["count"] > 0
        composite_mean = torch.where(
            composite_observed,
            composite["mean"],
            torch.zeros_like(composite["mean"]),
        )
        composite_std = torch.where(
            composite_observed,
            composite["standard_deviation"],
            torch.ones_like(composite["standard_deviation"]),
        )
        composite_dynamic = torch.zeros(
            len(FEATURE_NAMES["composite"]), dtype=torch.bool
        )
        composite_dynamic[list(DYNAMIC_COMPOSITE_INDICES)] = True
        composite_allowed = torch.ones_like(composite_dynamic)
        composite_allowed[list(TARGET_COMPOSITE_METADATA_INDICES)] = False

        for name, value in (
            ("common_mean", common_mean),
            ("common_std", common_std),
            ("common_continuous", common_continuous),
            ("common_allowed", common_allowed),
            ("composite_mean", composite_mean),
            ("composite_std", composite_std),
            ("composite_dynamic", composite_dynamic),
            ("composite_allowed", composite_allowed),
        ):
            self.register_buffer(name, value.detach().clone())

    def normalize_runtime(
        self,
        common: torch.Tensor,
        common_availability: torch.Tensor,
        composite: torch.Tensor,
        composite_availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        common_normalized = (common - self.common_mean) / self.common_std
        common_out = torch.where(
            self.common_continuous, common_normalized, common
        )
        common_mask = common_availability & self.common_allowed
        common_out = torch.where(
            common_mask, common_out, torch.zeros_like(common_out)
        )

        composite_normalized = (
            composite - self.composite_mean
        ) / self.composite_std
        composite_out = torch.where(
            self.composite_dynamic, composite_normalized, composite
        )
        composite_mask = composite_availability & self.composite_allowed
        composite_out = torch.where(
            composite_mask, composite_out, torch.zeros_like(composite_out)
        )
        return common_out, common_mask, composite_out, composite_mask


def inspect_checkpoint_for_export(
    checkpoint_path: str | Path,
    *,
    configuration: ExportConfiguration = ExportConfiguration(),
) -> dict[str, Any]:
    """Validate a trusted checkpoint and return a JSON-compatible export plan."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    config = configuration.validated()
    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    model, architecture, policy, pid_mode, pid_temperature = (
        _restore_deployment_model(payload)
    )
    query_count, max_cardinality = _uniform_decoder_capacity(
        model, config.levels
    )
    del model
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "levels": list(config.levels),
        "max_nodes": config.max_nodes,
        "max_sources": config.max_sources,
        "n_queries": query_count,
        "max_cardinality": max_cardinality,
        "architecture": architecture.to_dict(),
        "constraint_policy": policy.to_dict(),
        "pid_kinematics_mode": pid_mode,
        "pid_temperature": pid_temperature,
        "confidence_head_trained": bool(
            payload.get("confidence_head_trained", False)
        ),
        "opset_version": config.opset_version,
        "basf2_releases": list(config.basf2_releases),
    }


def export_checkpoint_bundle(
    checkpoint_path: str | Path,
    output_directory: str | Path,
    *,
    configuration: ExportConfiguration = ExportConfiguration(),
    force: bool = False,
) -> Path:
    """Export and atomically publish a hash-checked ONNX model bundle.

    By default the destination must not exist or must be empty.  ``force`` may
    replace only a directory that already contains a valid HyperTagging bundle;
    it never authorizes removal of an unrelated non-empty directory.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    destination = Path(output_directory).expanduser().resolve()
    if destination == checkpoint or destination in checkpoint.parents:
        raise ValueError("output directory must not contain the source checkpoint")
    config = configuration.validated()
    _validate_destination(destination, force=force)

    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    model, architecture, policy, pid_mode, pid_temperature = (
        _restore_deployment_model(payload)
    )
    query_count, max_cardinality = _uniform_decoder_capacity(
        model, config.levels
    )
    normalizers = _deployment_normalizers(payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.export-",
            dir=destination.parent,
        )
    )
    try:
        models: dict[str, dict[str, Any]] = {}
        for level in config.levels:
            filename = f"level-{level:02d}.onnx"
            graph_path = staging / filename
            wrapper = _LevelOnnxWrapper(
                model,
                target_level=level,
                pid_kinematics_mode=pid_mode,
                pid_temperature=pid_temperature,
            ).eval()
            example = _example_inputs(config, policy, target_level=level)
            _export_level_graph(
                wrapper,
                example,
                graph_path,
                opset_version=config.opset_version,
            )
            _rewrite_boolean_where_for_onnxruntime(graph_path)
            runtime_inputs = _validate_onnx_graph(
                graph_path,
                max_nodes=config.max_nodes,
                query_count=query_count,
                max_cardinality=max_cardinality,
            )
            models[str(level)] = {
                "file": filename,
                "sha256": file_sha256(graph_path),
                "runtime_inputs": runtime_inputs,
            }

        manifest = _build_manifest(
            checkpoint=checkpoint,
            payload=payload,
            architecture=architecture,
            policy=policy,
            configuration=config,
            models=models,
            normalizers=normalizers,
            query_count=query_count,
            max_cardinality=max_cardinality,
            pid_mode=pid_mode,
            pid_temperature=pid_temperature,
        )
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_bundle_manifest(manifest_path)
        _publish_directory(staging, destination, force=force)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _restore_deployment_model(
    payload: Mapping[str, Any],
) -> tuple[
    nn.Module,
    ModelArchitecture,
    ReconstructionConstraintPolicy,
    str,
    float,
]:
    _validate_checkpoint_contract(payload)
    architecture = ModelArchitecture.from_dict(payload["architecture"])
    training_config = _mapping(payload.get("config"), "checkpoint config")
    ablation = str(training_config.get("ablation", "full_revised"))
    if ablation not in ALL_ABLATIONS:
        raise ValueError(f"checkpoint has unknown ablation {ablation!r}")
    feature_contract = _mapping(
        payload.get("feature_contract"), "checkpoint feature_contract"
    )
    contract_pid_mode = feature_contract.get("pid_reconstruction_mode")
    if not isinstance(contract_pid_mode, str) or not contract_pid_mode:
        raise ValueError("checkpoint lacks an authoritative PID reconstruction mode")
    configured_pid_mode = training_config.get("pid_kinematics_mode")
    if (
        configured_pid_mode is not None
        and str(configured_pid_mode) != contract_pid_mode
    ):
        raise ValueError(
            "checkpoint training configuration conflicts with its PID feature contract"
        )
    pid_mode = contract_pid_mode
    rollout_pid_mode = str(
        training_config.get(
            "rollout_pid_kinematics_mode", "soft_decision_hard_construction"
        )
    )
    if rollout_pid_mode != "soft_decision_hard_construction":
        raise ValueError(
            "the v1 basf2 exporter requires soft decisions with hard rollout "
            f"construction, found {rollout_pid_mode!r}"
        )
    pid_temperature = float(
        training_config.get(
            "rollout_pid_temperature",
            feature_contract.get("pid_temperature", 1.0),
        )
    )
    if not math.isfinite(pid_temperature) or pid_temperature <= 0:
        raise ValueError("checkpoint PID temperature must be finite and positive")
    model = build_ablation_model(
        ablation,
        n_features=len(FEATURE_NAMES["common"]),
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
        pid_temperature=pid_temperature,
    )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model_state_dict is missing")
    model.load_state_dict(state, strict=True)
    normalizers = _checkpoint_normalizer_tensors(payload)
    runtime_normalizer = _OnnxRuntimeFeatureNormalizer(normalizers)
    model.set_runtime_feature_normalizer(runtime_normalizer)
    raw_policy = feature_contract.get("reconstruction_constraint_policy")
    if not isinstance(raw_policy, Mapping):
        raise ValueError(
            "checkpoint feature contract lacks reconstruction_constraint_policy"
        )
    policy = ReconstructionConstraintPolicy.from_dict(raw_policy)
    return model.cpu().eval(), architecture, policy, pid_mode, pid_temperature


def _validate_checkpoint_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("preprocessing_schema_version") != SCHEMA_VERSION_V4:
        raise ValueError(
            "only schema-v4 full-decay checkpoints can be exported: "
            f"{payload.get('preprocessing_schema_version')!r} != {SCHEMA_VERSION_V4!r}"
        )
    if payload.get("pid_vocabulary_version") != PID_VOCABULARY_VERSION:
        raise ValueError("checkpoint PID vocabulary differs from the deployment code")
    specification = _mapping(
        payload.get("feature_specification"), "checkpoint feature_specification"
    )
    current_specification = feature_spec_v4()
    if specification.get("feature_spec_hash") != current_specification.get(
        "feature_spec_hash"
    ):
        raise ValueError("checkpoint feature-specification hash is not current")
    feature_contract = _mapping(
        payload.get("feature_contract"), "checkpoint feature_contract"
    )
    if feature_contract.get(
        "model_feature_contract_hash"
    ) != current_specification.get("model_feature_contract_hash"):
        raise ValueError("checkpoint model-feature contract hash is not current")
    if specification.get("model_feature_contract_hash") != current_specification.get(
        "model_feature_contract_hash"
    ):
        raise ValueError(
            "checkpoint serialized feature specification has a stale model contract"
        )
    blocks = {
        "common": specification.get("common"),
        "track": specification.get("track"),
        "cluster": specification.get("ecl_cluster"),
        "composite": specification.get("composite"),
    }
    for name in ("common", "track", "cluster", "composite"):
        expected = FEATURE_NAMES[name]
        actual = blocks.get(name)
        if not isinstance(actual, (tuple, list)) or list(actual) != list(expected):
            raise ValueError(
                f"checkpoint {name} feature order differs from ONNX contract"
            )
    runtime_contracts = current_specification.get("runtime_model_contracts")
    if specification.get("runtime_model_contracts") != runtime_contracts:
        raise ValueError("checkpoint feature specification has stale runtime contracts")
    if payload.get("runtime_model_contracts") != runtime_contracts:
        raise ValueError("checkpoint top-level runtime contracts are not current")
    if specification.get("pid_tokens") != list(PDG_TOKENS):
        raise ValueError("checkpoint feature specification has stale PID tokens")
    if float(payload.get("legacy_conflated_fraction", 0.0)) != 0.0:
        raise ValueError("legacy-conflated checkpoints cannot be deployed")
    if not payload.get("data_compatible_performance", False):
        raise ValueError("checkpoint is not marked data-compatible")
    if not isinstance(payload.get("architecture"), Mapping):
        raise ValueError("checkpoint architecture is missing")


def _uniform_decoder_capacity(
    model: nn.Module, levels: Sequence[int]
) -> tuple[int, int]:
    capacities: set[tuple[int, int]] = set()
    for level in levels:
        decoder = (
            model.level_decoders[str(level)]
            if str(level) in model.level_decoders
            else model.decoder
        )
        capacities.add((int(decoder.n_queries), int(decoder.max_cardinality)))
    if len(capacities) != 1:
        raise ValueError(
            "the v1 fixed-shape runtime requires identical query/cardinality "
            f"capacity at every exported level, found {sorted(capacities)}"
        )
    return next(iter(capacities))


def _checkpoint_normalizer_tensors(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    raw = _mapping(payload.get("normalizer_state"), "checkpoint normalizer_state")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for block in NORMALIZED_BLOCKS:
        feature_names = FEATURE_NAMES[block]
        state = _mapping(raw.get(block), f"normalizer_state.{block}")
        tensors: dict[str, torch.Tensor] = {}
        for name in ("count", "mean", "standard_deviation"):
            value = state.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"normalizer_state.{block}.{name} must be a tensor")
            tensor = value.detach().cpu().float()
            if tensor.shape != (len(feature_names),):
                raise ValueError(
                    f"normalizer_state.{block}.{name} has width {tensor.numel()}, "
                    f"expected {len(feature_names)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"normalizer_state.{block}.{name} is not finite")
            tensors[name] = tensor
        if (tensors["count"] < 0).any():
            raise ValueError(f"normalizer_state.{block}.count is negative")
        if (tensors["standard_deviation"] <= 0).any():
            raise ValueError(
                f"normalizer_state.{block}.standard_deviation is not positive"
            )
        result[block] = tensors
    return result


def _deployment_normalizers(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, list[float]]]:
    normalizers = _checkpoint_normalizer_tensors(payload)
    return {
        block: {
            "mean": [float(value) for value in state["mean"].tolist()],
            "standard_deviation": [
                float(value) for value in state["standard_deviation"].tolist()
            ],
            "count": [float(value) for value in state["count"].tolist()],
        }
        for block, state in normalizers.items()
    }


def _example_inputs(
    config: ExportConfiguration,
    policy: ReconstructionConstraintPolicy,
    *,
    target_level: int,
) -> dict[str, torch.Tensor]:
    """Construct a trace example covering every truth-free runtime branch."""

    batch_size = 1
    node_count = config.max_nodes
    source_count = config.max_sources
    values: dict[str, torch.Tensor] = {}
    for block, feature_names in FEATURE_NAMES.items():
        values[f"{block}_features"] = torch.zeros(
            (batch_size, node_count, len(feature_names)), dtype=torch.float32
        )
        values[f"{block}_availability"] = torch.zeros(
            (batch_size, node_count, len(feature_names)), dtype=torch.bool
        )
    values["daughter_input_pid_histogram"] = torch.zeros(
        (batch_size, node_count, len(PDG_TOKENS)), dtype=torch.float32
    )
    values["daughter_input_pid_histogram_available"] = torch.zeros(
        (batch_size, node_count), dtype=torch.bool
    )
    for name, fill in (
        ("node_kind_ids", NODE_KIND_TO_ID["unknown"]),
        ("leaf_kinematics_mode_ids", LEAF_MODE_TO_ID["legacy_conflated"]),
        ("pid_labels", 0),
        ("level_ids", -1),
        ("parent_ids", -1),
        ("node_ids", -1),
        ("reco_ids", -1),
        ("source_node_ids", -1),
        ("copied_from", -1),
        (
            "runtime_composite_type_source_ids",
            COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"],
        ),
    ):
        values[name] = torch.full(
            (batch_size, node_count), fill, dtype=torch.int64
        )
    values["p4"] = torch.zeros(
        (batch_size, node_count, 4), dtype=torch.float32
    )
    values["charge"] = torch.zeros(
        (batch_size, node_count), dtype=torch.float32
    )
    values["daughter_adjacency"] = torch.zeros(
        (batch_size, node_count, node_count), dtype=torch.bool
    )
    for name in ("node_mask", "active", "copied"):
        values[name] = torch.zeros(
            (batch_size, node_count), dtype=torch.bool
        )
    values["recursive_leaf_source_mask"] = torch.zeros(
        (batch_size, node_count, source_count), dtype=torch.bool
    )

    # Two oppositely charged raw tracks exercise both PID/p4 branches.  ECL and
    # KLM leaves keep all heterogeneous adapters in the graph.  A masked node
    # remains available to prove padding is not baked into learned outputs.
    leaf_kinds = (
        NODE_KIND_TO_ID["track"],
        NODE_KIND_TO_ID["track"],
        NODE_KIND_TO_ID["ecl_cluster"],
        NODE_KIND_TO_ID["klm_cluster"],
    )
    leaf_modes = (
        LEAF_MODE_TO_ID["raw_track_predicted_pid"],
        LEAF_MODE_TO_ID["raw_track_predicted_pid"],
        LEAF_MODE_TO_ID["ecl_cluster"],
        LEAF_MODE_TO_ID["klm_cluster"],
    )
    leaf_tokens = (0, 0, 2, 3)
    charges = (1.0, -1.0, 0.0, 0.0)
    momenta = (
        (0.20, 0.10, 0.30, 0.40),
        (-0.15, 0.08, -0.25, 0.35),
        (0.12, -0.04, 0.20, 0.24),
        (-0.05, -0.02, 0.10, 0.12),
    )
    for position in range(4):
        values["node_kind_ids"][0, position] = leaf_kinds[position]
        values["leaf_kinematics_mode_ids"][0, position] = leaf_modes[position]
        values["pid_labels"][0, position] = leaf_tokens[position]
        values["level_ids"][0, position] = 0
        values["node_ids"][0, position] = position
        values["reco_ids"][0, position] = position
        values["source_node_ids"][0, position] = position
        values["p4"][0, position] = torch.tensor(momenta[position])
        values["charge"][0, position] = charges[position]
        values["node_mask"][0, position] = True
        values["active"][0, position] = True
        values["recursive_leaf_source_mask"][
            0, position, position % source_count
        ] = True
        values["common_features"][0, position, :4] = values["p4"][0, position]
        values["common_features"][0, position, 5] = charges[position]
        values["common_availability"][0, position, :6] = True
    values["track_availability"][0, :2] = True
    values["cluster_availability"][0, 2] = True
    values["klm_availability"][0, 3] = True

    # Lower-level predicted composites force the exported level-L graph to
    # retain the recurrent p4 and PID-histogram rebuild for every generation.
    for lower_level in range(1, target_level):
        position = 3 + lower_level
        values["node_kind_ids"][0, position] = NODE_KIND_TO_ID["composite"]
        values["leaf_kinematics_mode_ids"][0, position] = LEAF_MODE_TO_ID[
            "composite"
        ]
        values["pid_labels"][0, position] = 4
        values["level_ids"][0, position] = lower_level
        values["node_ids"][0, position] = position
        values["source_node_ids"][0, position] = position
        values["runtime_composite_type_source_ids"][0, position] = (
            COMPOSITE_TYPE_SOURCE_TO_ID["predicted"]
        )
        values["node_mask"][0, position] = True
        values["active"][0, position] = True
        values["daughter_adjacency"][0, position, 0] = True
        values["daughter_adjacency"][0, position, 1] = True
        values["recursive_leaf_source_mask"][0, position, 0] = True
        if source_count > 1:
            values["recursive_leaf_source_mask"][0, position, 1] = True
        values["daughter_input_pid_histogram"][0, position, 0] = 2.0
        values["daughter_input_pid_histogram_available"][0, position] = True
        values["composite_availability"][0, position, :9] = True

    allowed, bias = policy.type_constraints(target_level, device=torch.device("cpu"))
    values["allowed_type_mask"] = allowed
    values["type_logit_bias"] = bias
    values["pointer_validity_mask"] = values["node_mask"] & (
        values["level_ids"] < int(target_level)
    )
    return values


def _validate_onnx_graph(
    path: Path,
    *,
    max_nodes: int,
    query_count: int,
    max_cardinality: int,
) -> list[str]:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - CLI environment failure
        raise RuntimeError("the onnx package is required to validate exports") from exc
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value.name for value in model.graph.input if value.name not in initializers]
    if not set(inputs).issubset(MODEL_INPUT_NAMES):
        raise ValueError(f"ONNX graph contains unexpected runtime inputs: {inputs}")
    canonical_inputs = [name for name in MODEL_INPUT_NAMES if name in inputs]
    if len(inputs) != len(set(inputs)) or inputs != canonical_inputs:
        raise ValueError(
            "ONNX runtime inputs must be unique and retain canonical contract order: "
            f"{inputs} != {canonical_inputs}"
        )
    outputs = [value.name for value in model.graph.output]
    if outputs != list(MODEL_OUTPUT_NAMES):
        raise ValueError(
            f"ONNX output contract mismatch: {outputs} != {list(MODEL_OUTPUT_NAMES)}"
        )
    expected_shapes = {
        "object_logits": (1, query_count),
        "type_logits": (1, query_count, len(PDG_TOKENS)),
        "pointer_logits": (1, query_count, max_nodes),
        "cardinality_logits": (1, query_count, max_cardinality + 1),
        "confidence_logits": (1, query_count),
        "leaf_pid_logits": (1, max_nodes, len(PDG_TOKENS)),
        "current_p4": (1, max_nodes, 4),
    }
    for output in model.graph.output:
        dimensions = tuple(
            int(item.dim_value) if item.HasField("dim_value") else -1
            for item in output.type.tensor_type.shape.dim
        )
        if dimensions != expected_shapes[output.name]:
            raise ValueError(
                f"ONNX output {output.name} shape {dimensions} != "
                f"{expected_shapes[output.name]}"
            )
    return inputs


def _rewrite_boolean_where_for_onnxruntime(path: Path) -> None:
    """Lower boolean ``Where`` to logical ops supported by CVMFS ORT.

    The ONNX Runtime shipped by the target light release does not provide a
    CPU kernel for ``Where(bool, bool, bool)``.  The equivalent logical form is
    standard ONNX and preserves multidirectional broadcasting:

    ``(condition AND when_true) OR ((NOT condition) AND when_false)``.
    """

    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:  # pragma: no cover - CLI environment failure
        raise RuntimeError("the onnx package is required to finalize exports") from exc
    model = onnx.load(str(path), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    element_types: dict[str, int] = {}
    for collection in (
        inferred.graph.input,
        inferred.graph.output,
        inferred.graph.value_info,
    ):
        for value in collection:
            if value.type.HasField("tensor_type"):
                element_types[value.name] = value.type.tensor_type.elem_type
    for initializer in inferred.graph.initializer:
        element_types[initializer.name] = initializer.data_type

    rewritten = []
    for index, node in enumerate(model.graph.node):
        boolean_where = (
            node.op_type == "Where"
            and len(node.input) == 3
            and element_types.get(node.input[1]) == TensorProto.BOOL
            and element_types.get(node.input[2]) == TensorProto.BOOL
        )
        if not boolean_where:
            rewritten.append(node)
            continue
        condition, when_true, when_false = node.input
        output = node.output[0]
        prefix = f"{output}.bool_where_{index}"
        rewritten.extend(
            (
                helper.make_node(
                    "Not", [condition], [f"{prefix}.not"], name=f"{node.name}/Not"
                ),
                helper.make_node(
                    "And",
                    [condition, when_true],
                    [f"{prefix}.true"],
                    name=f"{node.name}/AndTrue",
                ),
                helper.make_node(
                    "And",
                    [f"{prefix}.not", when_false],
                    [f"{prefix}.false"],
                    name=f"{node.name}/AndFalse",
                ),
                helper.make_node(
                    "Or",
                    [f"{prefix}.true", f"{prefix}.false"],
                    [output],
                    name=f"{node.name}/Or",
                ),
            )
        )
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _export_level_graph(
    wrapper: nn.Module,
    example: Mapping[str, torch.Tensor],
    graph_path: Path,
    *,
    opset_version: int,
) -> None:
    """Run the stable exporter with the one missing standard-op lowering.

    PyTorch's legacy exporter does not currently register ``aten::asinh`` even
    though ONNX has exposed the equivalent standard ``Asinh`` operator since
    opset 9.  Hyperbolic distance uses that operation, so register the exact
    one-to-one lowering only for the duration of this export.
    """

    inverse_hyperbolic = {
        "aten::asinh": "Asinh",
        "aten::atanh": "Atanh",
    }
    for aten_name, onnx_name in inverse_hyperbolic.items():
        def symbolic(graph: Any, value: Any, *, _onnx_name: str = onnx_name) -> Any:
            return graph.op(_onnx_name, value)

        torch.onnx.register_custom_op_symbolic(
            aten_name, symbolic, int(opset_version)
        )
    try:
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(
                wrapper,
                tuple(example[name] for name in MODEL_INPUT_NAMES),
                graph_path,
                export_params=True,
                opset_version=int(opset_version),
                do_constant_folding=True,
                input_names=list(MODEL_INPUT_NAMES),
                output_names=list(MODEL_OUTPUT_NAMES),
                dynamo=False,
            )
    finally:
        for aten_name in inverse_hyperbolic:
            torch.onnx.unregister_custom_op_symbolic(aten_name, int(opset_version))


def _build_manifest(
    *,
    checkpoint: Path,
    payload: Mapping[str, Any],
    architecture: ModelArchitecture,
    policy: ReconstructionConstraintPolicy,
    configuration: ExportConfiguration,
    models: Mapping[str, Mapping[str, Any]],
    normalizers: Mapping[str, Mapping[str, list[float]]],
    query_count: int,
    max_cardinality: int,
    pid_mode: str,
    pid_temperature: float,
) -> dict[str, Any]:
    confidence_trained = bool(payload.get("confidence_head_trained", False))
    use_learned = (
        confidence_trained
        if configuration.use_learned_confidence is None
        else bool(configuration.use_learned_confidence)
    )
    if use_learned and not confidence_trained:
        raise ValueError(
            "learned-confidence scoring was requested but the checkpoint marks "
            "the confidence head untrained"
        )
    pointer_threshold = (
        policy.minimum_pointer_probability
        if configuration.pointer_threshold is None
        else configuration.pointer_threshold
    )
    exported_levels = set(configuration.levels)
    allowed_by_level = [
        [int(level), sorted({int(token) for token in tokens})]
        for level, tokens in sorted(policy.allowed_mother_types_by_level)
        if int(level) in exported_levels
    ]
    if not policy.reject_recursive_source_conflicts:
        raise ValueError(
            "basf2 deployment requires recursive reconstructed-source conflicts "
            "to be rejected"
        )
    if not policy.require_lower_level_context:
        raise ValueError(
            "basf2 deployment requires strictly lower-level pointer context"
        )
    reconstruction_policy: dict[str, Any] = {
        **_jsonable(policy.to_dict()),
        "allowed_mother_types_by_level": allowed_by_level,
        "static_allowed_mother_tokens": sorted(
            {int(token) for token in policy.static_allowed_mother_tokens}
        ),
        "valid_leaf_node_kinds": sorted(
            {int(kind) for kind in policy.valid_leaf_node_kinds}
        ),
        "valid_composite_node_kinds": sorted(
            {int(kind) for kind in policy.valid_composite_node_kinds}
        ),
        "beam_score": "sum_confidence",
        "beam_width": configuration.beam_width,
        "max_proposals_per_hypothesis": (
            configuration.max_proposals_per_hypothesis
        ),
        "object_threshold": configuration.object_threshold,
        "pointer_threshold": float(pointer_threshold),
        "confidence_threshold": configuration.confidence_threshold,
        "type_probability_threshold": configuration.type_probability_threshold,
        "minimum_daughters": policy.minimum_daughters,
        "root_tokens": [1],
        # Preserve the current offline evaluation convention: a reconstructed
        # Upsilon(4S) may complete the signal tree even when unrelated FSPs are
        # left in the event forest.
        "root_requires_all_sources": False,
        "token_charge": [float(value) for value in REDUCED_TOKEN_CHARGE],
        "use_learned_confidence": use_learned,
        "confidence_head_trained": confidence_trained,
        "rollout_pid_kinematics_mode": "soft_decision_hard_construction",
    }
    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_family": MODEL_FAMILY,
        "exporter": {
            "version": EXPORTER_VERSION,
            "onnx_opset": configuration.opset_version,
            "torch_version": str(torch.__version__),
        },
        "source_checkpoint": {
            "file_name": checkpoint.name,
            "sha256": file_sha256(checkpoint),
            "step": int(payload.get("step", 0)),
            "epoch": int(payload.get("epoch", 0)),
            "git_commit": str(payload.get("git_commit", "unknown")),
            "split_manifest_hash": str(payload.get("split_manifest_hash", "")),
        },
        "contract": {
            "max_nodes": configuration.max_nodes,
            "max_sources": configuration.max_sources,
            "n_queries": query_count,
            "max_cardinality": max_cardinality,
            "levels": list(configuration.levels),
            "pid_tokens": list(PDG_TOKENS),
            "pid_vocabulary_version": PID_VOCABULARY_VERSION,
            "preprocessing_schema_version": SCHEMA_VERSION_V4,
            "feature_names": {
                name: list(values) for name, values in FEATURE_NAMES.items()
            },
            "model_input_names": list(MODEL_INPUT_NAMES),
            "model_output_names": list(MODEL_OUTPUT_NAMES),
            "feature_contract": _jsonable(payload.get("feature_contract", {})),
            "runtime_model_contracts": _jsonable(
                payload.get("runtime_model_contracts", {})
            ),
        },
        "architecture": _jsonable(architecture.to_dict()),
        "pid_inference": {
            "decision_mode": pid_mode,
            "temperature": float(pid_temperature),
            "construction_mode": "hard_argmax_charge_compatible",
        },
        "normalization": _jsonable(normalizers),
        "reconstruction_policy": reconstruction_policy,
        "models": _jsonable(models),
        "runtime": {
            "engine": "onnxruntime",
            "providers": ["CPUExecutionProvider"],
            "batch_size": 1,
            "basf2_releases": list(configuration.basf2_releases),
        },
    }
    return with_manifest_hash(manifest)


def _validate_destination(destination: Path, *, force: bool) -> None:
    if not destination.exists():
        return
    if not destination.is_dir():
        raise FileExistsError(f"ONNX bundle destination is not a directory: {destination}")
    entries = tuple(destination.iterdir())
    if not entries:
        return
    if not force:
        raise FileExistsError(
            f"ONNX bundle destination is not empty: {destination}; pass force=True "
            "only to replace an existing valid bundle"
        )
    manifest = destination / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            "force refuses to replace a non-empty directory that is not a "
            "validated HyperTagging ONNX bundle"
        )
    existing = load_bundle_manifest(manifest)
    expected_entries = {
        "manifest.json",
        *(str(entry["file"]) for entry in existing["models"].values()),
    }
    actual_entries = {entry.name for entry in entries}
    if actual_entries != expected_entries or any(not entry.is_file() for entry in entries):
        raise ValueError(
            "force target contains entries outside its validated bundle: "
            f"expected {sorted(expected_entries)}, found {sorted(actual_entries)}"
        )
    for level, entry in existing["models"].items():
        graph = (destination / entry["file"]).resolve()
        if graph.parent != destination or not graph.is_file():
            raise ValueError(
                f"force target has no in-bundle ONNX graph for level {level}"
            )
        if file_sha256(graph) != entry["sha256"]:
            raise ValueError(
                f"force target ONNX hash mismatch for existing level {level}"
            )


def _publish_directory(staging: Path, destination: Path, *, force: bool) -> None:
    if destination.exists() and not any(destination.iterdir()):
        destination.rmdir()
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not force:
        raise FileExistsError(f"destination appeared during export: {destination}")
    # Export can take minutes. Revalidate immediately before moving the old
    # directory so a file added after the initial preflight is never deleted.
    _validate_destination(destination, force=True)
    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.backup-", dir=destination.parent
        )
    )
    backup.rmdir()
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _jsonable(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


__all__ = [
    "DEFAULT_BASF2_RELEASES",
    "DEFAULT_LEVELS",
    "DEFAULT_ONNX_OPSET",
    "EXPORTER_VERSION",
    "ExportConfiguration",
    "export_checkpoint_bundle",
    "inspect_checkpoint_for_export",
]
