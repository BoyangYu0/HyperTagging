"""Strict offline hierarchical inference from reconstructed detector FSPs.

This module is the information boundary between a collated, normalized
schema-v4 direct-mDST batch and free reconstruction rollout.  It deliberately
does not accept the historical GraFEI ``pairs`` representation.  Only track,
ECL-cluster, and KLM-cluster records are copied into the inference state;
truth topology and truth PID fields remain available only in the caller's
original batch for later evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

import torch
from torch import nn

from hypertagging.data.heterogeneous import (
    MODEL_INPUT_SOURCE_TO_ID,
    TRUTH_SUPERVISION_SOURCE_TO_ID,
)
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import PDG_TOKENS, TOKENIZE_DICT
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v3 import (
    V3_CLUSTER_FEATURE_NAMES,
    V3_COMMON_FEATURE_NAMES as COMMON_FEATURE_NAMES,
    V3_COMPOSITE_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES,
)
from hypertagging.preprocessing.schema_v4 import (
    KLM_FEATURE_NAMES,
    LEAF_MODE_TO_ID,
    SCHEMA_VERSION_V4,
)
from hypertagging.reconstruction.level_rollout import (
    BatchedRolloutResult,
    RolloutConfig,
    batched_free_rollout,
)
from hypertagging.reconstruction.pid_state import COMPOSITE_TYPE_SOURCE_TO_ID
from hypertagging.utils.tensor_contractions import boolean_matmul


InferenceScope = Literal["full", "half"]
OFFLINE_INFERENCE_POLICY_VERSION = (
    "fsp-forest-root-empty-level-soft-type-pid-parity-v2"
)

FULL_ROOT_TOKEN = TOKENIZE_DICT[300553]
DEFAULT_HALF_ROOT_TOKENS: tuple[int, ...] = tuple(
    TOKENIZE_DICT[pdg] for pdg in (511, 521, -511, -521)
)
DETECTOR_FSP_KIND_IDS: tuple[int, ...] = (
    NODE_KIND_TO_ID["track"],
    NODE_KIND_TO_ID["ecl_cluster"],
    NODE_KIND_TO_ID["klm_cluster"],
)

_REQUIRED_INPUT_FIELDS = frozenset(
    {
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
        "b_side",
        "model_input_source_ids",
        "daughter_input_pid_source_ids",
        "runtime_features_are_raw",
    }
)

_REMOVED_TRUTH_TARGET_FIELDS = (
    "pid_target_labels",
    "truth_pid_labels",
    "truth_pid_available",
)


@dataclass(frozen=True)
class HierarchicalInferenceConfig:
    """Offline inference scope plus the existing rollout policy.

    Full-event inference always stops on reduced Upsilon(4S) token ``1``.
    Half-event inference deliberately supplies no rollout root token, so the
    decoder cannot stop when it first emits a B.  B subtrees (or, for a
    continuum-like event without a B, terminal reconstructed forest roots)
    are exposed as evaluation slices after exhaustion or ``max_level``.
    """

    scope: InferenceScope = "full"
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    max_level: int | None = None
    half_root_tokens: tuple[int, ...] = DEFAULT_HALF_ROOT_TOKENS

    def __post_init__(self) -> None:
        if self.scope not in {"full", "half"}:
            raise ValueError("inference scope must be 'full' or 'half'")
        if self.max_level is not None and self.max_level <= 0:
            raise ValueError("max_level must be positive")
        if any(token <= 0 or token >= len(PDG_TOKENS) for token in self.half_root_tokens):
            raise ValueError("half_root_tokens contain an invalid reduced PID token")
        if not self.rollout_config.exclusive_final:
            raise ValueError(
                "strict hierarchical inference requires exclusive_final=True "
                "to preserve one parent per reconstructed node"
            )

    def resolved_rollout_config(self) -> RolloutConfig:
        maximum = (
            self.rollout_config.max_level
            if self.max_level is None
            else int(self.max_level)
        )
        roots = (FULL_ROOT_TOKEN,) if self.scope == "full" else ()
        return replace(
            self.rollout_config,
            max_level=maximum,
            root_types=roots,
            # Checkpoint-policy contraction can leave empty stored generations.
            # The target-level embedding still changes at the next generation,
            # so an empty proposal set is not a proof that no later root exists.
            continue_through_empty_levels=True,
        )


@dataclass(frozen=True)
class FSPInputAudit:
    """Read-only accounting for the destructive-in-spirit input projection."""

    schema_version: str
    batch_size: int
    source_node_width: int
    projected_node_width: int
    fsp_counts: tuple[int, ...]
    discarded_active_node_counts: tuple[int, ...]
    discarded_higher_level_node_counts: tuple[int, ...]
    track_counts: tuple[int, ...]
    ecl_cluster_counts: tuple[int, ...]
    klm_cluster_counts: tuple[int, ...]
    detector_source_counts: tuple[int, ...]
    detector_source_conflict_pair_counts: tuple[int, ...]
    original_fsp_positions: tuple[tuple[int, ...], ...]
    original_fsp_node_ids: tuple[tuple[int, ...], ...]
    original_fsp_reco_ids: tuple[tuple[int, ...], ...]
    original_fsp_source_node_ids: tuple[tuple[int, ...], ...]
    evaluation_fsp_source_keys: tuple[tuple[int, ...], ...]
    removed_truth_target_fields: tuple[str, ...]
    cpu_validated: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_size": self.batch_size,
            "source_node_width": self.source_node_width,
            "projected_node_width": self.projected_node_width,
            "fsp_counts": list(self.fsp_counts),
            "discarded_active_node_counts": list(
                self.discarded_active_node_counts
            ),
            "discarded_higher_level_node_counts": list(
                self.discarded_higher_level_node_counts
            ),
            "track_counts": list(self.track_counts),
            "ecl_cluster_counts": list(self.ecl_cluster_counts),
            "klm_cluster_counts": list(self.klm_cluster_counts),
            "detector_source_counts": list(self.detector_source_counts),
            "detector_source_conflict_pair_counts": list(
                self.detector_source_conflict_pair_counts
            ),
            "original_fsp_positions": [
                list(positions) for positions in self.original_fsp_positions
            ],
            "original_fsp_node_ids": [
                list(values) for values in self.original_fsp_node_ids
            ],
            "original_fsp_reco_ids": [
                list(values) for values in self.original_fsp_reco_ids
            ],
            "original_fsp_source_node_ids": [
                list(values) for values in self.original_fsp_source_node_ids
            ],
            "evaluation_fsp_source_keys": [
                list(values) for values in self.evaluation_fsp_source_keys
            ],
            "removed_truth_target_fields": list(
                self.removed_truth_target_fields
            ),
            "cpu_validated": self.cpu_validated,
        }


@dataclass(frozen=True)
class FSPProjection:
    """Physically compacted detector-only batch and its source audit."""

    batch: dict[str, torch.Tensor]
    evaluation_leaf_source_keys: torch.Tensor
    audit: FSPInputAudit


@dataclass(frozen=True)
class HierarchicalInferenceResult:
    """Free-rollout result plus masks defining event/half evaluation units."""

    scope: InferenceScope
    rollout: BatchedRolloutResult
    projected_fsp_batch: dict[str, torch.Tensor]
    input_audit: FSPInputAudit
    forest_root_mask: torch.Tensor
    b_root_mask: torch.Tensor
    continuum_root_mask: torch.Tensor
    evaluation_slice_root_mask: torch.Tensor
    evaluation_slice_multiplicity: torch.Tensor

    @property
    def batch(self) -> dict[str, torch.Tensor]:
        """Compatibility convenience for the reconstructed runtime tree."""

        return self.rollout.batch


def project_schema_v4_fsps(
    full_batch: Mapping[str, torch.Tensor],
) -> FSPProjection:
    """Compact a normalized schema-v4 batch to detector FSPs on CPU.

    The returned dictionary is newly allocated.  It contains no input truth
    PID targets, no B-side labels, no input tree links, and no retained mother
    records.  Runtime-required truth-named compatibility tensors are present
    only as all-zero unavailable sentinels.
    """

    batch_size, source_width, fsp_mask = _validate_schema_v4_input(full_batch)
    positions = tuple(
        tuple(
            int(position)
            for position in fsp_mask[batch_index]
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        for batch_index in range(batch_size)
    )
    counts = tuple(len(event_positions) for event_positions in positions)
    if any(count == 0 for count in counts):
        missing = [index for index, count in enumerate(counts) if count == 0]
        raise ValueError(
            "every inference event must contain at least one reconstructed "
            f"track/ECL/KLM FSP; empty event indices: {missing}"
        )
    width = max(counts)

    kinds = _gather_nodes(full_batch["node_kind_ids"], positions, width)
    modes = _gather_nodes(full_batch["leaf_kinematics_mode_ids"], positions, width)
    pid_labels = _gather_nodes(full_batch["pid_labels"], positions, width)
    p4 = _gather_nodes(full_batch["p4"], positions, width)
    charge = _gather_nodes(full_batch["charge"], positions, width)
    node_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    for batch_index, count in enumerate(counts):
        node_mask[batch_index, :count] = True

    common = _gather_nodes(full_batch["common_features"], positions, width)
    common_available = _gather_nodes(
        full_batch["common_availability"], positions, width
    ).bool()
    common = common.clone()
    common[..., :4] = p4
    common[..., 4] = (
        p4[..., 3].square() - p4[..., :3].square().sum(dim=-1)
    ).clamp_min(0).sqrt()
    common[..., 5] = charge
    common[..., 6] = pid_labels.to(common.dtype)
    common[..., 7] = 0
    common[..., 8] = node_mask.to(common.dtype)
    common[..., 9] = 0
    common[..., 10] = 0
    # candidate_confidence (slot 11) is reconstructed-object input and is
    # intentionally retained.  Categorical slots are represented by explicit
    # embeddings and never exposed as continuous values.
    for categorical in ("reduced_pid", "level", "active", "copied"):
        index = COMMON_FEATURE_NAMES.index(categorical)
        common_available[..., index] = False
    common = torch.where(common_available, common, torch.zeros_like(common))

    detector_blocks: dict[str, torch.Tensor] = {}
    kind_for_block = {
        "track": NODE_KIND_TO_ID["track"],
        "cluster": NODE_KIND_TO_ID["ecl_cluster"],
        "klm": NODE_KIND_TO_ID["klm_cluster"],
    }
    for block, expected_kind in kind_for_block.items():
        values = _gather_nodes(full_batch[f"{block}_features"], positions, width)
        available = _gather_nodes(
            full_batch[f"{block}_availability"], positions, width
        ).bool()
        correct_kind = (kinds == expected_kind) & node_mask
        available &= correct_kind.unsqueeze(-1)
        detector_blocks[f"{block}_features"] = torch.where(
            available, values, torch.zeros_like(values)
        )
        detector_blocks[f"{block}_availability"] = available

    histogram_width = full_batch["daughter_input_pid_histogram"].shape[-1]
    if histogram_width != len(PDG_TOKENS):
        raise ValueError("daughter input PID histogram width is not schema-v4")
    composite_width = full_batch["composite_features"].shape[-1]
    dense_ids = torch.full((batch_size, width), -1, dtype=torch.long)
    for batch_index, count in enumerate(counts):
        dense_ids[batch_index, :count] = torch.arange(count, dtype=torch.long)

    # Preserve detector-object source provenance for the selected FSP rows,
    # while dropping every source column unused by those rows.  This retains
    # ECL/KLM association conflicts learned by the current model without
    # consulting or retaining any higher-level particle record.
    original_recursive_sources = full_batch["recursive_leaf_source_mask"].bool()
    compact_source_rows: list[torch.Tensor] = []
    source_counts: list[int] = []
    for batch_index, event_positions in enumerate(positions):
        index = torch.tensor(event_positions, dtype=torch.long)
        rows = original_recursive_sources[batch_index, index]
        used_columns = rows.any(dim=0)
        compact = rows[:, used_columns]
        if not bool(compact.any(dim=-1).all()):
            raise ValueError(
                "every detector FSP requires reconstructed source provenance"
            )
        compact_source_rows.append(compact)
        source_counts.append(int(compact.shape[-1]))
    provenance_width = max(source_counts)
    recursive_sources = torch.zeros(
        (batch_size, width, provenance_width), dtype=torch.bool
    )
    for batch_index, compact in enumerate(compact_source_rows):
        recursive_sources[
            batch_index, : compact.shape[0], : compact.shape[1]
        ] = compact
    source_overlap = boolean_matmul(
        recursive_sources, recursive_sources.transpose(1, 2)
    )
    source_conflicts = source_overlap & ~torch.eye(
        width, dtype=torch.bool
    ).unsqueeze(0)
    source_conflicts &= node_mask[:, :, None] & node_mask[:, None, :]
    evaluation_source_keys = torch.full_like(dense_ids, -1)
    original_reco_ids = _gather_nodes(full_batch["reco_ids"], positions, width)
    original_source_node_ids = _gather_nodes(
        full_batch["source_node_ids"], positions, width
    )
    original_node_ids = _gather_nodes(full_batch["node_ids"], positions, width)
    evaluation_source_keys = torch.where(
        original_reco_ids >= 0,
        original_reco_ids,
        torch.where(
            original_source_node_ids >= 0,
            original_source_node_ids,
            original_node_ids,
        ),
    )
    evaluation_source_keys = torch.where(
        node_mask, evaluation_source_keys, torch.full_like(evaluation_source_keys, -1)
    )

    unavailable_truth_source = TRUTH_SUPERVISION_SOURCE_TO_ID["unavailable"]
    native_input_source = MODEL_INPUT_SOURCE_TO_ID["native_v4_reconstructed"]
    input_sources = torch.where(
        node_mask,
        torch.full_like(dense_ids, native_input_source),
        torch.zeros_like(dense_ids),
    )
    unavailable_truth_sources = torch.full_like(
        dense_ids, unavailable_truth_source
    )
    input_fixed_sources = torch.full_like(
        dense_ids, COMPOSITE_TYPE_SOURCE_TO_ID["input_fixed"]
    )
    zeros_long = torch.zeros_like(dense_ids)
    minus_one_long = torch.full_like(dense_ids, -1)
    zeros_bool = torch.zeros_like(node_mask)
    zeros_float = torch.zeros(
        (batch_size, width, histogram_width), dtype=p4.dtype
    )

    projected: dict[str, torch.Tensor] = {
        "common_features": common,
        "common_availability": common_available & node_mask.unsqueeze(-1),
        **detector_blocks,
        "composite_features": torch.zeros(
            (batch_size, width, composite_width), dtype=p4.dtype
        ),
        "composite_availability": torch.zeros(
            (batch_size, width, composite_width), dtype=torch.bool
        ),
        "daughter_input_pid_histogram": zeros_float.clone(),
        "daughter_input_pid_histogram_available": zeros_bool.clone(),
        "daughter_truth_pid_histogram": zeros_float.clone(),
        "daughter_truth_pid_histogram_available": zeros_bool.clone(),
        "node_kind_ids": torch.where(kinds >= 0, kinds, zeros_long),
        "leaf_kinematics_mode_ids": torch.where(
            node_mask,
            modes,
            torch.full_like(modes, LEAF_MODE_TO_ID["legacy_conflated"]),
        ),
        "pid_labels": torch.where(node_mask, pid_labels, zeros_long),
        "level_ids": torch.where(node_mask, zeros_long, minus_one_long),
        "p4": torch.where(node_mask.unsqueeze(-1), p4, torch.zeros_like(p4)),
        "charge": torch.where(node_mask, charge, torch.zeros_like(charge)),
        "parent_ids": minus_one_long.clone(),
        "daughter_adjacency": torch.zeros(
            (batch_size, width, width), dtype=torch.bool
        ),
        "node_mask": node_mask,
        "active": node_mask.clone(),
        "copied": zeros_bool.clone(),
        "node_ids": dense_ids.clone(),
        "reco_ids": dense_ids.clone(),
        "source_node_ids": dense_ids.clone(),
        "recursive_leaf_source_mask": recursive_sources,
        "source_conflict_matrix": source_conflicts,
        "copied_from": minus_one_long.clone(),
        "b_side": minus_one_long.clone(),
        "model_input_source_ids": input_sources.clone(),
        "daughter_input_pid_source_ids": input_sources.clone(),
        "truth_supervision_source_ids": unavailable_truth_sources.clone(),
        "daughter_truth_pid_source_ids": unavailable_truth_sources.clone(),
        "runtime_composite_type_source_ids": input_fixed_sources,
        "full_truth_daughter_count": minus_one_long.clone(),
        "retained_truth_daughter_count_expected": minus_one_long.clone(),
        "retained_daughter_count": minus_one_long.clone(),
        "reconstructed_daughter_count": zeros_long.clone(),
        "complete_truth_decay": zeros_bool.clone(),
        "complete_reconstructable_decay": zeros_bool.clone(),
        "recursive_reconstructable_complete": zeros_bool.clone(),
        "partial_missing_daughters": zeros_bool.clone(),
        "contracted_intermediate": zeros_bool.clone(),
        "valid_reconstruction_target": zeros_bool.clone(),
        "truth_root_distance": minus_one_long.clone(),
        "full_event_max_level": minus_one_long.clone(),
        "runtime_structurally_valid": zeros_bool.clone(),
        "ancestor_descendant_relation": torch.zeros(
            (batch_size, width, width), dtype=torch.bool
        ),
        "lca_node_id": torch.full(
            (batch_size, width, width), -1, dtype=torch.long
        ),
        "edges_to_lca_from_i": torch.full(
            (batch_size, width, width), -1, dtype=torch.long
        ),
        "edges_to_lca_from_j": torch.full(
            (batch_size, width, width), -1, dtype=torch.long
        ),
        "exact_tree_path_distance": torch.full(
            (batch_size, width, width), -1, dtype=torch.long
        ),
        "lca_depth": torch.full(
            (batch_size, width, width), -1, dtype=torch.long
        ),
        "depth_from_retained_root": minus_one_long.clone(),
        "distance_to_nearest_retained_root": minus_one_long.clone(),
        "runtime_features_are_raw": torch.tensor(True),
    }
    projected["daughter_pid_histogram"] = projected[
        "daughter_input_pid_histogram"
    ]
    projected["daughter_pid_histogram_available"] = projected[
        "daughter_input_pid_histogram_available"
    ]
    projected["node_features"] = projected["common_features"]
    if "event_ids" in full_batch:
        projected["event_ids"] = full_batch["event_ids"].detach().clone()

    active = full_batch["node_mask"].bool()
    levels = full_batch["level_ids"]
    audit = FSPInputAudit(
        schema_version=SCHEMA_VERSION_V4,
        batch_size=batch_size,
        source_node_width=source_width,
        projected_node_width=width,
        fsp_counts=counts,
        discarded_active_node_counts=tuple(
            int((active[index] & ~fsp_mask[index]).sum())
            for index in range(batch_size)
        ),
        discarded_higher_level_node_counts=tuple(
            int((active[index] & (levels[index] > 0)).sum())
            for index in range(batch_size)
        ),
        track_counts=_kind_counts(fsp_mask, full_batch["node_kind_ids"], "track"),
        ecl_cluster_counts=_kind_counts(
            fsp_mask, full_batch["node_kind_ids"], "ecl_cluster"
        ),
        klm_cluster_counts=_kind_counts(
            fsp_mask, full_batch["node_kind_ids"], "klm_cluster"
        ),
        detector_source_counts=tuple(source_counts),
        detector_source_conflict_pair_counts=tuple(
            int(torch.triu(source_conflicts[index], diagonal=1).sum())
            for index in range(batch_size)
        ),
        original_fsp_positions=positions,
        # Dense runtime index i maps to each tuple's i-th original key.  These
        # evaluation-only keys are carried beside ``projected`` but are never
        # read by the encoder or decoder.
        original_fsp_node_ids=_values_at_positions(
            full_batch["node_ids"], positions
        ),
        original_fsp_reco_ids=_values_at_positions(
            full_batch["reco_ids"], positions
        ),
        original_fsp_source_node_ids=_values_at_positions(
            full_batch["source_node_ids"], positions
        ),
        evaluation_fsp_source_keys=tuple(
            tuple(int(value) for value in evaluation_source_keys[index, :count])
            for index, count in enumerate(counts)
        ),
        removed_truth_target_fields=tuple(
            name for name in _REMOVED_TRUTH_TARGET_FIELDS if name in full_batch
        ),
    )
    _validate_projected_fsp_batch(projected, evaluation_source_keys)
    return FSPProjection(
        batch=projected,
        evaluation_leaf_source_keys=evaluation_source_keys,
        audit=audit,
    )


def reconstruct_full_tree_from_fsps(
    model: LevelAutoregressiveReconstructor,
    full_batch: Mapping[str, torch.Tensor],
    *,
    config: HierarchicalInferenceConfig | None = None,
    scope: InferenceScope | None = None,
) -> HierarchicalInferenceResult:
    """Run CPU-only free reconstruction from a preprocessed-mDST batch.

    ``model`` must be an independent CPU evaluation instance.  This function
    never moves it, toggles its train/eval state, or mutates ``full_batch``;
    consequently it is safe to use alongside a separate training process.
    """

    if config is None:
        config = HierarchicalInferenceConfig(scope=scope or "full")
    elif scope is not None:
        config = replace(config, scope=scope)
    _validate_cpu_evaluation_model(model)
    projection = project_schema_v4_fsps(full_batch)
    rollout_config = config.resolved_rollout_config()
    with torch.inference_mode():
        rollout = batched_free_rollout(
            model,
            projection.batch,
            config=rollout_config,
        )

    reconstructed = rollout.batch
    # Attach stable FSP identities only after every model call has finished.
    # They are metric metadata, never an input tensor.
    reconstructed["evaluation_leaf_source_keys"] = (
        projection.evaluation_leaf_source_keys.clone()
    )
    _require_cpu_tensors(reconstructed, owner="reconstructed rollout")
    nodes = reconstructed["node_mask"].bool()
    composite = nodes & (
        reconstructed["node_kind_ids"] == NODE_KIND_TO_ID["composite"]
    )
    forest = nodes & (reconstructed["parent_ids"] < 0)
    pid = reconstructed["pid_labels"]
    b_tokens = torch.tensor(config.half_root_tokens, dtype=torch.long)
    b_roots = composite & (
        (pid[..., None] == b_tokens).any(dim=-1)
        if b_tokens.numel()
        else torch.zeros_like(nodes)
    )
    continuum_roots = composite & forest & ~b_roots
    if config.scope == "full":
        slices = composite & (pid == FULL_ROOT_TOKEN)
    else:
        has_b = b_roots.any(dim=-1, keepdim=True)
        slices = torch.where(has_b, b_roots, continuum_roots)
    multiplicity = slices.sum(dim=-1).to(torch.long)
    return HierarchicalInferenceResult(
        scope=config.scope,
        rollout=rollout,
        projected_fsp_batch=projection.batch,
        input_audit=projection.audit,
        forest_root_mask=forest,
        b_root_mask=b_roots,
        continuum_root_mask=continuum_roots,
        evaluation_slice_root_mask=slices,
        evaluation_slice_multiplicity=multiplicity,
    )


def _validate_schema_v4_input(
    full_batch: Mapping[str, torch.Tensor],
) -> tuple[int, int, torch.Tensor]:
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(full_batch))
    if missing:
        raise ValueError(
            "schema-v4 collated/normalized inference batch is missing: "
            + ", ".join(missing)
        )
    _require_cpu_tensors(full_batch, owner="inference input")
    raw_marker = full_batch["runtime_features_are_raw"]
    if raw_marker.shape != torch.Size([]) or not bool(raw_marker.item()):
        raise ValueError(
            "schema-v4 inference expects the training data-module contract: "
            "static detector blocks normalized, runtime common/composite blocks raw"
        )
    node_mask = full_batch["node_mask"]
    if node_mask.ndim != 2:
        raise ValueError("node_mask must have shape [B,N]")
    batch_size, node_width = node_mask.shape
    if batch_size == 0 or node_width == 0:
        raise ValueError("inference batch must contain at least one stored node")
    feature_widths = {
        "common": len(COMMON_FEATURE_NAMES),
        "track": len(V3_TRACK_FEATURE_NAMES),
        "cluster": len(V3_CLUSTER_FEATURE_NAMES),
        "klm": len(KLM_FEATURE_NAMES),
        "composite": len(V3_COMPOSITE_FEATURE_NAMES),
    }
    for block, width in feature_widths.items():
        expected = (batch_size, node_width, width)
        for suffix in ("features", "availability"):
            name = f"{block}_{suffix}"
            if full_batch[name].shape != expected:
                raise ValueError(f"{name} must have shape [B,N,{width}]")
    histogram_width = len(PDG_TOKENS)
    if full_batch["daughter_input_pid_histogram"].shape != (
        batch_size,
        node_width,
        histogram_width,
    ):
        raise ValueError(
            "daughter_input_pid_histogram must have shape "
            f"[B,N,{histogram_width}]"
        )
    scalar_fields = (
        "daughter_input_pid_histogram_available",
        "node_kind_ids",
        "leaf_kinematics_mode_ids",
        "pid_labels",
        "level_ids",
        "charge",
        "parent_ids",
        "node_mask",
        "active",
        "copied",
        "node_ids",
        "reco_ids",
        "source_node_ids",
        "copied_from",
        "b_side",
        "model_input_source_ids",
        "daughter_input_pid_source_ids",
    )
    for name in scalar_fields:
        if full_batch[name].shape != (batch_size, node_width):
            raise ValueError(f"{name} must have shape [B,N]")
    provenance = full_batch["recursive_leaf_source_mask"]
    if (
        provenance.ndim != 3
        or provenance.shape[:2] != (batch_size, node_width)
        or provenance.shape[2] <= 0
    ):
        raise ValueError("recursive_leaf_source_mask must have shape [B,N,S>0]")
    if full_batch["daughter_adjacency"].shape != (
        batch_size,
        node_width,
        node_width,
    ):
        raise ValueError("daughter_adjacency must have shape [B,N,N]")
    if full_batch["p4"].shape != (batch_size, node_width, 4):
        raise ValueError("p4 must have shape [B,N,4]")
    kinds = full_batch["node_kind_ids"]
    bool_fields = (
        "common_availability",
        "track_availability",
        "cluster_availability",
        "klm_availability",
        "composite_availability",
        "daughter_input_pid_histogram_available",
        "daughter_adjacency",
        "node_mask",
        "active",
        "copied",
        "recursive_leaf_source_mask",
    )
    long_fields = (
        "node_kind_ids",
        "leaf_kinematics_mode_ids",
        "pid_labels",
        "level_ids",
        "parent_ids",
        "node_ids",
        "reco_ids",
        "source_node_ids",
        "copied_from",
        "b_side",
        "model_input_source_ids",
        "daughter_input_pid_source_ids",
    )
    float_fields = (
        "common_features",
        "track_features",
        "cluster_features",
        "klm_features",
        "composite_features",
        "daughter_input_pid_histogram",
        "p4",
        "charge",
    )
    for name in bool_fields:
        if full_batch[name].dtype != torch.bool:
            raise TypeError(f"schema-v4 {name} must have dtype bool")
    for name in long_fields:
        if full_batch[name].dtype != torch.long:
            raise TypeError(f"schema-v4 {name} must have dtype int64")
    for name in float_fields:
        if full_batch[name].dtype != torch.float32:
            raise TypeError(f"schema-v4 {name} must have dtype float32")
    if raw_marker.dtype != torch.bool:
        raise TypeError("runtime_features_are_raw must have dtype bool")
    active = node_mask.bool()
    native_source = MODEL_INPUT_SOURCE_TO_ID["native_v4_reconstructed"]
    for name in ("model_input_source_ids", "daughter_input_pid_source_ids"):
        if active.any() and (full_batch[name][active] != native_source).any():
            raise ValueError(
                f"{name} is not native {SCHEMA_VERSION_V4} reconstructed input"
            )
    physical_kind = torch.zeros_like(active)
    for kind in DETECTOR_FSP_KIND_IDS:
        physical_kind |= kinds == kind
    fsp_mask = active & physical_kind
    if (fsp_mask & (full_batch["level_ids"] != 0)).any():
        raise ValueError("detector FSP records must be stored at level zero")
    if (fsp_mask & full_batch["daughter_adjacency"].any(dim=-1)).any():
        raise ValueError("detector FSP records must not have stored daughters")
    if not fsp_mask.any():
        raise ValueError("batch has no reconstructed track/ECL/KLM FSP records")

    raw_track = fsp_mask & (
        full_batch["leaf_kinematics_mode_ids"]
        == LEAF_MODE_TO_ID["raw_track_predicted_pid"]
    )
    if raw_track.any() and (full_batch["pid_labels"][raw_track] != 0).any():
        raise ValueError("raw-track FSP inputs must have unknown PID token zero")
    if not torch.isfinite(full_batch["p4"][fsp_mask]).all():
        raise ValueError("detector FSP four-momenta must be finite")
    if not torch.isfinite(full_batch["charge"][fsp_mask]).all():
        raise ValueError("detector FSP charges must be finite")
    for block in ("common", "track", "cluster", "klm"):
        values = full_batch[f"{block}_features"]
        available = full_batch[f"{block}_availability"].bool()
        visible = available & fsp_mask.unsqueeze(-1)
        if visible.any() and not torch.isfinite(values[visible]).all():
            raise ValueError(f"available detector {block} features must be finite")
    return batch_size, node_width, fsp_mask


def _validate_projected_fsp_batch(
    batch: Mapping[str, torch.Tensor],
    evaluation_source_keys: torch.Tensor,
) -> None:
    _require_cpu_tensors(batch, owner="projected FSP batch")
    active = batch["node_mask"].bool()
    active_kinds = batch["node_kind_ids"][active]
    allowed = torch.zeros_like(active_kinds, dtype=torch.bool)
    for kind in DETECTOR_FSP_KIND_IDS:
        allowed |= active_kinds == kind
    if not bool(allowed.all()):
        raise AssertionError("projected inference input is not detector-FSP-only")
    if (batch["level_ids"][active] != 0).any():
        raise AssertionError("projected FSP levels were not reset")
    if (batch["parent_ids"][active] != -1).any() or batch[
        "daughter_adjacency"
    ].any():
        raise AssertionError("projected topology was not reset")
    if (batch["b_side"][active] != -1).any():
        raise AssertionError("projected B-side truth labels were not scrubbed")
    if any(name in batch for name in _REMOVED_TRUTH_TARGET_FIELDS):
        raise AssertionError("truth PID target field survived FSP projection")
    unavailable = TRUTH_SUPERVISION_SOURCE_TO_ID["unavailable"]
    if (
        batch["truth_supervision_source_ids"][active] != unavailable
    ).any() or (
        batch["daughter_truth_pid_source_ids"][active] != unavailable
    ).any():
        raise AssertionError("projected truth provenance is not unavailable")
    if batch["daughter_truth_pid_histogram_available"].any() or batch[
        "daughter_truth_pid_histogram"
    ].count_nonzero():
        raise AssertionError("truth daughter-PID state survived FSP projection")
    for event_index in range(active.shape[0]):
        count = int(active[event_index].sum())
        expected = torch.arange(count, dtype=torch.long)
        for name in ("node_ids", "reco_ids", "source_node_ids"):
            if not torch.equal(batch[name][event_index, :count], expected):
                raise AssertionError(f"{name} are not dense after FSP projection")
        sources = batch["recursive_leaf_source_mask"][event_index, :count]
        if not bool(sources.any(dim=-1).all()):
            raise AssertionError("projected FSP detector provenance is empty")
        overlap = boolean_matmul(sources, sources.transpose(0, 1))
        expected_conflicts = overlap & ~torch.eye(count, dtype=torch.bool)
        observed_conflicts = batch["source_conflict_matrix"][
            event_index, :count, :count
        ]
        if not torch.equal(observed_conflicts, expected_conflicts):
            raise AssertionError("projected detector-source conflicts are inconsistent")
        keys = evaluation_source_keys[event_index, :count]
        if (keys < 0).any() or keys.unique().numel() != count:
            raise ValueError(
                "evaluation FSP source keys must be nonnegative and unique per event"
            )


def _validate_cpu_evaluation_model(model: object) -> None:
    if not isinstance(model, nn.Module):
        return
    if model.training:
        raise ValueError(
            "offline inference requires a separate model in eval mode; refusing "
            "to mutate a possibly training model"
        )
    devices = {
        tensor.device.type
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if devices - {"cpu"}:
        raise ValueError(
            "offline inference requires a separate CPU model; refusing to move "
            "or interfere with a GPU training model"
        )


def _require_cpu_tensors(
    values: Mapping[str, torch.Tensor], *, owner: str
) -> None:
    non_cpu = sorted(
        name
        for name, value in values.items()
        if isinstance(value, torch.Tensor) and value.device.type != "cpu"
    )
    if non_cpu:
        raise ValueError(
            f"{owner} must be CPU-only; non-CPU tensors: {', '.join(non_cpu)}"
        )


def _gather_nodes(
    value: torch.Tensor,
    positions: tuple[tuple[int, ...], ...],
    width: int,
) -> torch.Tensor:
    output = torch.zeros(
        (len(positions), width, *value.shape[2:]), dtype=value.dtype
    )
    for batch_index, event_positions in enumerate(positions):
        count = len(event_positions)
        if count:
            index = torch.tensor(event_positions, dtype=torch.long)
            output[batch_index, :count] = value[batch_index, index]
    return output


def _kind_counts(
    fsp_mask: torch.Tensor,
    node_kind_ids: torch.Tensor,
    kind_name: str,
) -> tuple[int, ...]:
    selected = fsp_mask & (node_kind_ids == NODE_KIND_TO_ID[kind_name])
    return tuple(int(selected[index].sum()) for index in range(selected.shape[0]))


def _values_at_positions(
    values: torch.Tensor,
    positions: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(values[batch_index, position]) for position in event_positions)
        for batch_index, event_positions in enumerate(positions)
    )


# Descriptive alias for callers constructing their own evaluation pipeline.
project_preprocessed_mdst_fsps = project_schema_v4_fsps


__all__ = [
    "DEFAULT_HALF_ROOT_TOKENS",
    "DETECTOR_FSP_KIND_IDS",
    "FULL_ROOT_TOKEN",
    "FSPInputAudit",
    "FSPProjection",
    "HierarchicalInferenceConfig",
    "HierarchicalInferenceResult",
    "InferenceScope",
    "OFFLINE_INFERENCE_POLICY_VERSION",
    "project_preprocessed_mdst_fsps",
    "project_schema_v4_fsps",
    "reconstruct_full_tree_from_fsps",
]
