"""Source-aligned metrics for evaluation-only full decay reconstruction.

The functions in this module deliberately do not use generated node IDs, node
ordering, PID, or four-momentum to align trees.  Stable final-state-particle
(FSP) keys come from level-zero reconstructed-object IDs, or from the explicit
evaluation-only ``evaluation_leaf_source_keys`` field.  Composite membership
is rebuilt from the evaluated adjacency.  It is intentionally separate from
``recursive_leaf_source_mask``, whose columns encode detector-source overlap
and may be multi-hot for associated ECL/KLM FSPs.
PID and kinematics are read only *after* source-set alignment.

``LCAG`` means lowest-common-ancestor generation.  Leaves have generation zero
and a mother has one plus the maximum generation of its daughters.  Returning
the LCAG as a mapping keyed by unordered FSP pairs makes the comparison
invariant to tensor/node order and to generated composite IDs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID


B_ROOT_TOKENS: frozenset[int] = frozenset((21, 22, 38, 39))
UPSILON_TOKEN = 1
TargetPolicy = Literal[
    "complete_only", "reconstructable_partial", "diagnostic_all"
]
TruthTopologyMode = Literal["checkpoint_direct", "contracted_diagnostic"]


@dataclass(frozen=True)
class RatioMetric:
    """An aggregation-safe scalar represented by its sufficient statistics."""

    numerator: float = 0.0
    denominator: float = 0.0

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "value": self.value,
            "numerator": float(self.numerator),
            "denominator": float(self.denominator),
        }


@dataclass(frozen=True)
class KinematicErrorMetrics:
    """Absolute kinematic errors for source-set-aligned composite particles.

    ``p4_l1`` is averaged over the four components. ``p3`` is the Euclidean
    distance between momentum vectors.  ``momentum_magnitude`` is the absolute
    error on ``|p|``.  No value in this structure participates in alignment.
    """

    alignment_coverage: RatioMetric = RatioMetric()
    p4_l1: RatioMetric = RatioMetric()
    p3: RatioMetric = RatioMetric()
    relative_p3: RatioMetric = RatioMetric()
    p3_squared_error: RatioMetric = RatioMetric()
    px_bias: RatioMetric = RatioMetric()
    py_bias: RatioMetric = RatioMetric()
    pz_bias: RatioMetric = RatioMetric()
    energy_bias: RatioMetric = RatioMetric()
    px_mae: RatioMetric = RatioMetric()
    py_mae: RatioMetric = RatioMetric()
    pz_mae: RatioMetric = RatioMetric()
    energy: RatioMetric = RatioMetric()
    px_squared_error: RatioMetric = RatioMetric()
    py_squared_error: RatioMetric = RatioMetric()
    pz_squared_error: RatioMetric = RatioMetric()
    energy_squared_error: RatioMetric = RatioMetric()
    mass: RatioMetric = RatioMetric()
    mass_bias: RatioMetric = RatioMetric()
    mass_squared_error: RatioMetric = RatioMetric()
    momentum_magnitude: RatioMetric = RatioMetric()

    def as_dict(self) -> dict[str, dict[str, float | None]]:
        return {
            "alignment_coverage": self.alignment_coverage.as_dict(),
            "p4_l1_error": self.p4_l1.as_dict(),
            "p3_error": self.p3.as_dict(),
            "relative_p3_error": self.relative_p3.as_dict(),
            "p3_rmse": _rmse_dict(self.p3_squared_error),
            "px_bias": self.px_bias.as_dict(),
            "py_bias": self.py_bias.as_dict(),
            "pz_bias": self.pz_bias.as_dict(),
            "energy_bias": self.energy_bias.as_dict(),
            "px_mae": self.px_mae.as_dict(),
            "py_mae": self.py_mae.as_dict(),
            "pz_mae": self.pz_mae.as_dict(),
            "energy_mae": self.energy.as_dict(),
            "px_rmse": _rmse_dict(self.px_squared_error),
            "py_rmse": _rmse_dict(self.py_squared_error),
            "pz_rmse": _rmse_dict(self.pz_squared_error),
            "energy_rmse": _rmse_dict(self.energy_squared_error),
            "energy_error": self.energy.as_dict(),
            "mass_error": self.mass.as_dict(),
            "mass_bias": self.mass_bias.as_dict(),
            "mass_rmse": _rmse_dict(self.mass_squared_error),
            "momentum_magnitude_error": self.momentum_magnitude.as_dict(),
        }


@dataclass(frozen=True)
class DecayEvaluation:
    """One independently weighted full-decay or B-half evaluation row."""

    scope: str
    unit_index: int
    available: bool
    unavailable_reason: str | None
    truth_root_position: int | None
    predicted_root_position: int | None
    truth_sources: tuple[int, ...]
    predicted_sources: tuple[int, ...]
    source_recall: RatioMetric
    source_precision: RatioMetric
    structurally_valid: RatioMetric
    target_representable: RatioMetric
    lcag_pair_accuracy: RatioMetric
    perfect_lcag: RatioMetric
    strict_missing_one_leaf: RatioMetric
    leave_one_out_lcag: RatioMetric
    leaf_pid_accuracy: RatioMetric
    mother_pid_accuracy: RatioMetric
    mother_pid_coverage: RatioMetric
    root_pid_accuracy: RatioMetric
    kinematics: KinematicErrorMetrics
    omitted_source_key: int | None = None
    truth_mother_count: int = 0
    predicted_mother_count: int = 0
    matched_mother_count: int = 0
    leaf_pid_confusion: tuple[tuple[int, int, int], ...] = ()
    mother_pid_confusion: tuple[tuple[int, int, int], ...] = ()
    truth_retained_depth: int | None = None
    predicted_retained_depth: int | None = None
    truth_topology_mode: str = "checkpoint_direct"
    target_unrepresentable_reasons: tuple[str, ...] = ()
    kinematic_reference: str = "reconstructed_fsp_daughter_sum"
    physical_momentum_error_available: bool = False
    physical_momentum_error_unavailable_reason: str | None = (
        "preprocessed_training_schema_does_not_retain_mc_composite_p4"
    )

    @property
    def perfectLCAG(self) -> bool | None:  # noqa: N802 - analysis column name
        value = self.perfect_lcag.value
        return bool(value) if value is not None else None

    @property
    def missing_one_particle(self) -> bool | None:
        value = self.strict_missing_one_leaf.value
        return bool(value) if value is not None else None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, flat analysis row."""

        output: dict[str, Any] = {
            "scope": self.scope,
            "unit_index": int(self.unit_index),
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "truth_root_position": self.truth_root_position,
            "predicted_root_position": self.predicted_root_position,
            "truth_sources": list(self.truth_sources),
            "predicted_sources": list(self.predicted_sources),
            "truth_leaf_count": len(self.truth_sources),
            "predicted_leaf_count": len(self.predicted_sources),
            "missing_source_keys": sorted(
                set(self.truth_sources) - set(self.predicted_sources)
            ),
            "extra_source_keys": sorted(
                set(self.predicted_sources) - set(self.truth_sources)
            ),
            "omitted_source_key": self.omitted_source_key,
            "truth_mother_count": int(self.truth_mother_count),
            "predicted_mother_count": int(self.predicted_mother_count),
            "matched_mother_count": int(self.matched_mother_count),
            "truth_retained_depth": self.truth_retained_depth,
            "predicted_retained_depth": self.predicted_retained_depth,
            "truth_topology_mode": self.truth_topology_mode,
            "kinematic_reference": self.kinematic_reference,
            "physical_momentum_error_available": (
                self.physical_momentum_error_available
            ),
            "physical_momentum_error_unavailable_reason": (
                self.physical_momentum_error_unavailable_reason
            ),
            "unmatched_truth_mother_count": int(
                self.truth_mother_count - self.matched_mother_count
            ),
            "unmatched_predicted_mother_count": int(
                self.predicted_mother_count - self.matched_mother_count
            ),
        }
        _flatten_ratio(output, "source_recall", self.source_recall)
        _flatten_ratio(output, "source_precision", self.source_precision)
        _flatten_ratio(
            output, "structurally_valid", self.structurally_valid, boolean=True
        )
        _flatten_ratio(
            output,
            "target_representable",
            self.target_representable,
            boolean=True,
        )
        output["target_unrepresentable_reasons"] = list(
            self.target_unrepresentable_reasons
        )
        _flatten_ratio(output, "lcag_pair_accuracy", self.lcag_pair_accuracy)
        _flatten_ratio(output, "perfectLCAG", self.perfect_lcag, boolean=True)
        _flatten_ratio(
            output,
            "strict_missing_one_leaf",
            self.strict_missing_one_leaf,
            boolean=True,
        )
        output["missing_one_particle"] = output["strict_missing_one_leaf"]
        _flatten_ratio(
            output,
            "leave_one_out_lcag",
            self.leave_one_out_lcag,
            boolean=True,
        )
        _flatten_ratio(output, "leaf_pid_accuracy", self.leaf_pid_accuracy)
        output["leaf_pid_unavailable_count"] = max(
            len(self.truth_sources) - int(self.leaf_pid_accuracy.denominator), 0
        )
        _flatten_ratio(output, "mother_pid_accuracy", self.mother_pid_accuracy)
        _flatten_ratio(output, "mother_pid_coverage", self.mother_pid_coverage)
        _flatten_ratio(output, "root_pid_accuracy", self.root_pid_accuracy)
        output["leaf_pid_confusion"] = [
            {
                "truth_token": truth_token,
                "predicted_token": predicted_token,
                "count": count,
            }
            for truth_token, predicted_token, count in self.leaf_pid_confusion
        ]
        output["mother_pid_confusion"] = [
            {
                "truth_token": truth_token,
                "predicted_token": predicted_token,
                "count": count,
            }
            for truth_token, predicted_token, count in self.mother_pid_confusion
        ]
        for name, metric in self.kinematics.as_dict().items():
            output[name] = metric["value"]
            output[f"{name}_numerator"] = metric["numerator"]
            output[f"{name}_denominator"] = metric["denominator"]
        return output


@dataclass(frozen=True)
class HalfDecayEvaluation:
    """Multiplicity-weighted B halves or explicit continuum root components."""

    available: bool
    unavailable_reason: str | None
    halves: tuple[DecayEvaluation, ...]
    both_halves_perfect_lcag: RatioMetric
    both_halves_leave_one_out_lcag: RatioMetric
    predicted_b_root_count: int = 0
    assigned_predicted_component_count: int = 0
    unassigned_predicted_b_root_count: int = 0
    unit_semantics: str = "b_halves"
    predicted_component_root_count: int = 0
    unassigned_predicted_component_root_count: int = 0

    @property
    def rows(self) -> tuple[DecayEvaluation, ...]:
        return self.halves

    def as_rows(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.halves]

    def as_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "unit_count": len(self.halves),
            "unit_semantics": self.unit_semantics,
            "predicted_b_root_count": self.predicted_b_root_count,
            "assigned_predicted_component_count": (
                self.assigned_predicted_component_count
            ),
            "unassigned_predicted_b_root_count": (
                self.unassigned_predicted_b_root_count
            ),
            "predicted_component_root_count": self.predicted_component_root_count,
            "unassigned_predicted_component_root_count": (
                self.unassigned_predicted_component_root_count
            ),
            "rows": self.as_rows(),
        }
        _flatten_ratio(
            output,
            "both_halves_perfectLCAG",
            self.both_halves_perfect_lcag,
            boolean=True,
        )
        _flatten_ratio(
            output,
            "both_halves_leave_one_out_lcag",
            self.both_halves_leave_one_out_lcag,
            boolean=True,
        )
        return output


@dataclass(frozen=True)
class _TreeView:
    active: torch.Tensor
    adjacency: torch.Tensor
    sources: torch.Tensor
    source_keys: tuple[int, ...]
    pid: torch.Tensor
    truth_pid_available: torch.Tensor | None
    p4: torch.Tensor | None
    b_side: torch.Tensor | None = None
    detector_sources: torch.Tensor | None = None

    @property
    def positions(self) -> list[int]:
        return [int(value) for value in self.active.nonzero(as_tuple=False).flatten()]

    def children(self, node: int) -> list[int]:
        return [
            int(value)
            for value in (self.adjacency[node] & self.active)
            .nonzero(as_tuple=False)
            .flatten()
        ]

    def source_set(self, node: int) -> frozenset[int]:
        return frozenset(
            self.source_keys[int(value)]
            for value in self.sources[node].nonzero(as_tuple=False).flatten()
        )


def source_keyed_lcag(
    batch: Mapping[str, torch.Tensor],
    event_index: int = 0,
    root_position: int | None = None,
    *,
    source_keys: Sequence[int] | torch.Tensor | None = None,
) -> dict[tuple[int, int], int]:
    """Return ``{(source_i, source_j): LCA generation}`` for one rooted tree.

    Pair keys are sorted and diagonal entries are intentionally omitted.  PID,
    momentum, ``node_ids``, and node order are never consulted.
    """

    view = _tree_view(
        batch, event_index, truth=False, source_keys_override=source_keys
    )
    root = _choose_largest_root(view) if root_position is None else int(root_position)
    if root is None:
        return {}
    _require_active_position(view, root, "root_position")
    return _source_keyed_lcag_view(view, root)


def canonical_fsp_membership(
    batch: Mapping[str, torch.Tensor],
    event_index: int = 0,
    *,
    source_keys: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return topology-derived node-to-unique-FSP membership and its keys.

    This is intentionally distinct from ``recursive_leaf_source_mask``.  The
    latter is detector-resource provenance used to reject conflicts and can
    legitimately be multi-hot for one reconstructed FSP.
    """

    view = _tree_view(
        batch,
        event_index,
        truth=False,
        source_keys_override=source_keys,
    )
    return view.sources.clone(), view.source_keys


def truth_target_policy_diagnostics(
    batch: Mapping[str, torch.Tensor],
    event_index: int = 0,
    *,
    target_policy: TargetPolicy = "complete_only",
    minimum_daughters: int = 2,
) -> dict[str, int | str]:
    """Account for retained and suppressed truth mothers under a checkpoint policy."""

    active_value = batch["node_mask"] if "node_mask" in batch else batch["active"]
    active = _event_tensor(active_value, event_index, 1).bool()
    adjacency = _event_tensor(
        batch["daughter_adjacency"], event_index, 2
    ).bool()[: active.numel(), : active.numel()]
    raw_mothers = active & (adjacency & active[None, :]).any(dim=-1)
    retained = _truth_target_mask(
        batch,
        event_index,
        target_policy=target_policy,
        minimum_daughters=minimum_daughters,
    )
    invalid = (
        raw_mothers
        & ~_event_tensor(
            batch["valid_reconstruction_target"], event_index, 1
        ).bool()[: active.numel()]
        if "valid_reconstruction_target" in batch
        else torch.zeros_like(raw_mothers)
    )
    incomplete = (
        raw_mothers
        & ~_event_tensor(
            batch["recursive_reconstructable_complete"], event_index, 1
        ).bool()[: active.numel()]
        if "recursive_reconstructable_complete" in batch
        else torch.zeros_like(raw_mothers)
    )
    return {
        "target_policy": target_policy,
        "minimum_daughters": int(minimum_daughters),
        "raw_truth_mother_count": int(raw_mothers.sum()),
        "retained_target_mother_count": int(retained.sum()),
        "suppressed_truth_mother_count": int((raw_mothers & ~retained).sum()),
        "invalid_target_mother_count": int(invalid.sum()),
        "incomplete_target_mother_count": int(incomplete.sum()),
    }


def _source_keyed_lcag_view(
    view: _TreeView, root: int
) -> dict[tuple[int, int], int]:
    return {
        pair: signature[1]
        for pair, signature in _source_keyed_lca_signatures(view, root).items()
    }


def _source_keyed_lca_signatures(
    view: _TreeView,
    root: int,
    *,
    retained_sources: Iterable[int] | None = None,
) -> dict[tuple[int, int], tuple[tuple[int, ...], int]]:
    """Canonical pair LCA clade and induced generation.

    Intersecting all clades with ``retained_sources`` and rebuilding their
    inclusion hierarchy is the induced-tree operation needed by the
    missing-one metric.  It correctly suppresses unary chains created when a
    leaf is omitted; simply comparing original generations does not.
    """

    descendants = _descendants(view, root)
    retained = (
        set(view.source_set(root))
        if retained_sources is None
        else set(int(value) for value in retained_sources)
        & set(view.source_set(root))
    )
    clades: set[frozenset[int]] = set()
    for node in descendants:
        clade = frozenset(set(view.source_set(node)) & retained)
        if clade:
            clades.add(clade)
    clades.update(frozenset((source,)) for source in retained)

    heights: dict[frozenset[int], int] = {}

    def height(clade: frozenset[int]) -> int:
        if clade in heights:
            return heights[clade]
        proper = [candidate for candidate in clades if candidate < clade]
        maximal = [
            candidate
            for candidate in proper
            if not any(candidate < other < clade for other in proper)
        ]
        value = 0 if len(clade) <= 1 else 1 + max(
            (height(candidate) for candidate in maximal), default=0
        )
        heights[clade] = value
        return value

    output: dict[tuple[int, int], tuple[tuple[int, ...], int]] = {}
    for left, right in combinations(sorted(retained), 2):
        candidates = [
            clade for clade in clades if left in clade and right in clade
        ]
        if not candidates:
            continue
        lca = min(candidates, key=lambda value: (len(value), tuple(sorted(value))))
        output[(left, right)] = (tuple(sorted(lca)), height(lca))
    return output


def evaluate_full_decay(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    event_index: int = 0,
    *,
    truth_root_position: int | None = None,
    predicted_root_position: int | None = None,
    truth_source_keys: Sequence[int] | torch.Tensor | None = None,
    predicted_source_keys: Sequence[int] | torch.Tensor | None = None,
    target_policy: TargetPolicy = "complete_only",
    minimum_daughters: int = 2,
    truth_topology_mode: TruthTopologyMode = "checkpoint_direct",
) -> DecayEvaluation:
    """Evaluate one full event, preferring the truth Upsilon(4S) root.

    The predicted root is selected by source-set overlap with the chosen truth
    root.  Its predicted PID is not used during this selection.
    """

    _validate_truth_topology_mode(truth_topology_mode)

    raw_truth_view = _tree_view(
        truth,
        event_index,
        truth=True,
        source_keys_override=truth_source_keys,
    )
    declared_roots = _truth_full_roots(raw_truth_view)
    if truth_root_position is None and not declared_roots:
        return _unavailable_row(
            "full",
            0,
            "truth_root_not_found",
            truth_topology_mode=truth_topology_mode,
        )
    if truth_root_position is None and len(declared_roots) != 1:
        return _unavailable_row(
            "full",
            0,
            f"expected_one_truth_root_found_{len(declared_roots)}",
            truth_topology_mode=truth_topology_mode,
        )
    truth_root = (
        int(declared_roots[0])
        if truth_root_position is None
        else int(truth_root_position)
    )
    _require_active_position(raw_truth_view, truth_root, "truth_root_position")
    target_mask = _truth_target_mask(
        truth,
        event_index,
        target_policy=target_policy,
        minimum_daughters=minimum_daughters,
    )
    if not bool(target_mask[truth_root]):
        return _unavailable_row(
            "full",
            0,
            f"truth_root_not_eligible_for_{target_policy}",
            truth_root_position=truth_root,
            truth_sources=tuple(sorted(raw_truth_view.source_set(truth_root))),
            truth_topology_mode=truth_topology_mode,
        )
    target_representable, representability_reasons = _unit_representability(
        truth,
        event_index,
        raw_truth_view,
        truth_root,
        target_mask,
        truth_topology_mode=truth_topology_mode,
    )
    truth_view = (
        _tree_view(
            truth,
            event_index,
            truth=True,
            source_keys_override=truth_source_keys,
            target_policy=target_policy,
            minimum_daughters=minimum_daughters,
        )
        if truth_topology_mode == "contracted_diagnostic"
        else raw_truth_view
    )
    predicted_view = _tree_view(
        predicted,
        event_index,
        truth=False,
        source_keys_override=predicted_source_keys,
    )
    if predicted_root_position is None:
        predicted_root = _best_root_for_sources(
            predicted_view,
            truth_view.source_set(truth_root),
            candidates=[
                root
                for root in _root_positions(predicted_view)
                if predicted_view.children(root)
            ],
        )
    else:
        predicted_root = int(predicted_root_position)
        _require_active_position(predicted_view, predicted_root, "predicted_root_position")
    return _evaluate_root_pair(
        predicted,
        truth,
        predicted_view,
        truth_view,
        predicted_root,
        truth_root,
        event_index=event_index,
        scope="full",
        unit_index=0,
        truth_topology_mode=truth_topology_mode,
        truth_target_mask=target_mask,
        target_representable=target_representable,
        target_unrepresentable_reasons=representability_reasons,
    )


def evaluate_half_decays(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    event_index: int = 0,
    *,
    source_category: str | None = None,
    truth_source_keys: Sequence[int] | torch.Tensor | None = None,
    predicted_source_keys: Sequence[int] | torch.Tensor | None = None,
    target_policy: TargetPolicy = "complete_only",
    minimum_daughters: int = 2,
    truth_topology_mode: TruthTopologyMode = "checkpoint_direct",
) -> HalfDecayEvaluation:
    """Evaluate B halves or explicit continuum roots as multiplicity units.

    Signal-like events contribute exactly two unordered B rows.  Continuum
    events contribute one row per explicit top-level truth composite; no
    artificial two-hemisphere split or ``both halves`` trial is invented.
    """

    _validate_truth_topology_mode(truth_topology_mode)

    if _is_continuum_category(source_category):
        return _evaluate_continuum_components(
            predicted,
            truth,
            event_index=event_index,
            truth_source_keys=truth_source_keys,
            predicted_source_keys=predicted_source_keys,
            target_policy=target_policy,
            minimum_daughters=minimum_daughters,
            truth_topology_mode=truth_topology_mode,
        )
    raw_truth_view = _tree_view(
        truth,
        event_index,
        truth=True,
        source_keys_override=truth_source_keys,
    )
    b_roots = _truth_b_roots(raw_truth_view)
    if len(b_roots) != 2:
        reason = (
            "continuum_sides_unavailable_in_preprocessed_mdst_schema"
            if _is_continuum_category(source_category)
            else f"expected_two_truth_b_roots_found_{len(b_roots)}"
        )
        return _unavailable_halves(reason)
    target_mask = _truth_target_mask(
        truth,
        event_index,
        target_policy=target_policy,
        minimum_daughters=minimum_daughters,
    )
    truth_view = (
        _tree_view(
            truth,
            event_index,
            truth=True,
            source_keys_override=truth_source_keys,
            target_policy=target_policy,
            minimum_daughters=minimum_daughters,
        )
        if truth_topology_mode == "contracted_diagnostic"
        else raw_truth_view
    )
    predicted_view = _tree_view(
        predicted,
        event_index,
        truth=False,
        source_keys_override=predicted_source_keys,
    )
    # Canonical truth ordering is by source set, never tensor order or B charge.
    b_roots = sorted(
        b_roots,
        key=lambda node: (tuple(sorted(raw_truth_view.source_set(node))), node),
    )
    policy_eligible = {root: bool(target_mask[root]) for root in b_roots}
    representability = {
        root: _unit_representability(
            truth,
            event_index,
            raw_truth_view,
            root,
            target_mask,
            truth_topology_mode=truth_topology_mode,
        )
        for root in b_roots
    }
    eligible_roots = [root for root in b_roots if policy_eligible[root]]
    candidate_nodes = [
        node
        for node in predicted_view.positions
        if predicted_view.children(node)
        and any(
            predicted_view.source_set(node) & truth_view.source_set(root)
            for root in eligible_roots
        )
    ]
    assigned = _assign_two_by_sources(
        predicted_view,
        candidate_nodes,
        truth_view,
        eligible_roots,
    )
    assignment_by_root = dict(zip(eligible_roots, assigned, strict=True))
    predicted_b_roots = set(_truth_b_roots(predicted_view))
    assigned_components = {
        candidate for candidate in assigned if candidate is not None
    }
    rows = tuple(
        (
            _evaluate_root_pair(
                predicted,
                truth,
                predicted_view,
                truth_view,
                assignment_by_root[truth_root],
                truth_root,
                event_index=event_index,
                scope="b_half",
                unit_index=index,
                truth_topology_mode=truth_topology_mode,
                truth_target_mask=target_mask,
                target_representable=representability[truth_root][0],
                target_unrepresentable_reasons=representability[truth_root][1],
            )
            if policy_eligible[truth_root]
            else _unavailable_row(
                "b_half",
                index,
                f"truth_b_root_not_eligible_for_{target_policy}",
                truth_root_position=truth_root,
                truth_sources=tuple(
                    sorted(raw_truth_view.source_set(truth_root))
                ),
                truth_topology_mode=truth_topology_mode,
            )
        )
        for index, truth_root in enumerate(b_roots)
    )
    both_eligible = all(row.available for row in rows)
    both_topology_eligible = both_eligible and all(
        row.perfect_lcag.denominator > 0 for row in rows
    )
    both_perfect = both_topology_eligible and all(
        row.perfectLCAG is True for row in rows
    )
    both_loo = both_eligible and all(
        row.leave_one_out_lcag.value == 1.0 for row in rows
    )
    loo_available = both_eligible and all(
        row.leave_one_out_lcag.denominator > 0 for row in rows
    )
    return HalfDecayEvaluation(
        available=both_eligible,
        unavailable_reason=(
            None
            if both_eligible
            else f"one_or_more_truth_b_roots_not_eligible_for_{target_policy}"
        ),
        halves=rows,
        both_halves_perfect_lcag=RatioMetric(
            float(both_perfect) if both_topology_eligible else 0.0,
            1.0 if both_topology_eligible else 0.0,
        ),
        both_halves_leave_one_out_lcag=RatioMetric(
            float(both_loo) if loo_available else 0.0,
            1.0 if loo_available else 0.0,
        ),
        predicted_b_root_count=len(predicted_b_roots),
        assigned_predicted_component_count=len(assigned_components),
        unassigned_predicted_b_root_count=len(
            predicted_b_roots - assigned_components
        ),
        unit_semantics="b_halves",
        predicted_component_root_count=len(
            [root for root in _root_positions(predicted_view) if predicted_view.children(root)]
        ),
        unassigned_predicted_component_root_count=len(
            {
                root
                for root in _root_positions(predicted_view)
                if predicted_view.children(root)
            }
            - assigned_components
        ),
    )


def _evaluate_continuum_components(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    *,
    event_index: int,
    truth_source_keys: Sequence[int] | torch.Tensor | None,
    predicted_source_keys: Sequence[int] | torch.Tensor | None,
    target_policy: TargetPolicy,
    minimum_daughters: int,
    truth_topology_mode: TruthTopologyMode,
) -> HalfDecayEvaluation:
    """Evaluate explicit top-level continuum particles as multiplicity units."""

    raw_truth = _tree_view(
        truth,
        event_index,
        truth=True,
        source_keys_override=truth_source_keys,
    )
    roots = sorted(
        [root for root in _root_positions(raw_truth) if raw_truth.children(root)],
        key=lambda root: (tuple(sorted(raw_truth.source_set(root))), root),
    )
    if not roots:
        return _unavailable_halves(
            "no_explicit_truth_continuum_component_roots",
            unit_semantics="continuum_components",
        )
    target_mask = _truth_target_mask(
        truth,
        event_index,
        target_policy=target_policy,
        minimum_daughters=minimum_daughters,
    )
    truth_view = (
        _tree_view(
            truth,
            event_index,
            truth=True,
            source_keys_override=truth_source_keys,
            target_policy=target_policy,
            minimum_daughters=minimum_daughters,
        )
        if truth_topology_mode == "contracted_diagnostic"
        else raw_truth
    )
    predicted_view = _tree_view(
        predicted,
        event_index,
        truth=False,
        source_keys_override=predicted_source_keys,
    )
    policy_eligible = {root: bool(target_mask[root]) for root in roots}
    representability = {
        root: _unit_representability(
            truth,
            event_index,
            raw_truth,
            root,
            target_mask,
            truth_topology_mode=truth_topology_mode,
        )
        for root in roots
    }
    eligible_roots = [root for root in roots if policy_eligible[root]]
    all_predicted_component_roots = [
        root
        for root in _root_positions(predicted_view)
        if predicted_view.children(root)
    ]
    assignment_candidates = [
        root
        for root in all_predicted_component_roots
        if any(
            predicted_view.source_set(root) & truth_view.source_set(target)
            for target in eligible_roots
        )
    ]
    assigned = _assign_many_by_sources(
        predicted_view, assignment_candidates, truth_view, eligible_roots
    )
    assignment_by_root = dict(zip(eligible_roots, assigned, strict=True))
    assigned_components = {
        candidate for candidate in assigned if candidate is not None
    }

    rows: list[DecayEvaluation] = []
    for index, root in enumerate(roots):
        if policy_eligible[root]:
            rows.append(
                _evaluate_root_pair(
                    predicted,
                    truth,
                    predicted_view,
                    truth_view,
                    assignment_by_root[root],
                    root,
                    event_index=event_index,
                    scope="continuum_component",
                    unit_index=index,
                    truth_topology_mode=truth_topology_mode,
                    truth_target_mask=target_mask,
                    target_representable=representability[root][0],
                    target_unrepresentable_reasons=representability[root][1],
                )
            )
            continue
        rows.append(
            _unavailable_row(
                "continuum_component",
                index,
                f"truth_continuum_root_not_eligible_for_{target_policy}",
                truth_root_position=root,
                truth_sources=tuple(sorted(raw_truth.source_set(root))),
                truth_topology_mode=truth_topology_mode,
            )
        )
    available_count = sum(int(row.available) for row in rows)
    return HalfDecayEvaluation(
        available=available_count > 0,
        unavailable_reason=(
            None
            if available_count > 0
            else "no_representable_truth_continuum_component_roots"
        ),
        halves=tuple(rows),
        both_halves_perfect_lcag=RatioMetric(),
        both_halves_leave_one_out_lcag=RatioMetric(),
        unit_semantics="continuum_components",
        predicted_component_root_count=len(all_predicted_component_roots),
        assigned_predicted_component_count=len(assigned_components),
        unassigned_predicted_component_root_count=len(
            set(all_predicted_component_roots) - assigned_components
        ),
    )


def summarize_decay_evaluations(
    evaluations: Iterable[DecayEvaluation | HalfDecayEvaluation],
) -> dict[str, Any]:
    """Micro-aggregate full rows or multiplicity-weighted B-half rows."""

    units: list[DecayEvaluation] = []
    half_events: list[HalfDecayEvaluation] = []
    unavailable = 0
    for evaluation in evaluations:
        if isinstance(evaluation, HalfDecayEvaluation):
            half_events.append(evaluation)
            units.extend(evaluation.halves)
            unavailable += int(not evaluation.available)
        else:
            units.append(evaluation)
            unavailable += int(not evaluation.available)

    ratio_fields = (
        "source_recall",
        "source_precision",
        "structurally_valid",
        "target_representable",
        "lcag_pair_accuracy",
        "perfect_lcag",
        "strict_missing_one_leaf",
        "leave_one_out_lcag",
        "leaf_pid_accuracy",
        "mother_pid_accuracy",
        "mother_pid_coverage",
        "root_pid_accuracy",
    )
    output: dict[str, Any] = {
        "unit_count": len(units),
        "available_unit_count": sum(int(unit.available) for unit in units),
        "unavailable_unit_count": sum(int(not unit.available) for unit in units),
        "unavailable_evaluation_count": unavailable,
        "topology_outcome_counts": {
            "perfect": sum(int(unit.perfectLCAG is True) for unit in units),
            "strict_missing_one": sum(
                int(unit.missing_one_particle is True) for unit in units
            ),
            "other": sum(
                int(
                    unit.available
                    and unit.perfect_lcag.denominator > 0
                    and unit.perfectLCAG is not True
                    and unit.missing_one_particle is not True
                )
                for unit in units
            ),
        },
        "missing_source_count": sum(
            len(set(unit.truth_sources) - set(unit.predicted_sources))
            for unit in units
            if unit.available
        ),
        "extra_source_count": sum(
            len(set(unit.predicted_sources) - set(unit.truth_sources))
            for unit in units
            if unit.available
        ),
        "truth_mother_count": sum(unit.truth_mother_count for unit in units),
        "predicted_mother_count": sum(
            unit.predicted_mother_count for unit in units
        ),
        "matched_mother_count": sum(unit.matched_mother_count for unit in units),
        "target_unrepresentable_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for unit in units
                    for reason in unit.target_unrepresentable_reasons
                ).items()
            )
        ),
        "kinematic_reference": "reconstructed_fsp_daughter_sum",
        "physical_momentum_error_available": False,
        "physical_momentum_error_unavailable_reason": (
            "preprocessed_training_schema_does_not_retain_mc_composite_p4"
        ),
    }
    for field in ratio_fields:
        output[field] = _sum_ratios(getattr(unit, field) for unit in units).as_dict()
    for field in (
        "alignment_coverage",
        "p4_l1",
        "p3",
        "relative_p3",
        "p3_squared_error",
        "px_bias",
        "py_bias",
        "pz_bias",
        "energy_bias",
        "px_mae",
        "py_mae",
        "pz_mae",
        "energy",
        "px_squared_error",
        "py_squared_error",
        "pz_squared_error",
        "energy_squared_error",
        "mass",
        "mass_bias",
        "mass_squared_error",
        "momentum_magnitude",
    ):
        aggregate = _sum_ratios(getattr(unit.kinematics, field) for unit in units)
        output[field] = aggregate.as_dict()
    for component in ("p3", "px", "py", "pz", "energy", "mass"):
        squared = output[f"{component}_squared_error"]
        output[f"{component}_rmse"] = {
            **squared,
            "value": math.sqrt(squared["value"]) if squared["value"] is not None else None,
        }
    for name in ("leaf_pid_confusion", "mother_pid_confusion"):
        confusion: Counter[tuple[int, int]] = Counter()
        for unit in units:
            for truth_token, predicted_token, count in getattr(unit, name):
                confusion[(truth_token, predicted_token)] += count
        output[name] = [
            {
                "truth_token": truth_token,
                "predicted_token": predicted_token,
                "count": count,
            }
            for (truth_token, predicted_token), count in sorted(confusion.items())
        ]
    output["leaf_pid_unavailable_count"] = sum(
        max(len(unit.truth_sources) - int(unit.leaf_pid_accuracy.denominator), 0)
        for unit in units
        if unit.available
    )
    if half_events:
        output["half_event_count"] = len(half_events)
        output["predicted_b_root_count"] = sum(
            event.predicted_b_root_count for event in half_events
        )
        output["assigned_predicted_component_count"] = sum(
            event.assigned_predicted_component_count for event in half_events
        )
        output["unassigned_predicted_b_root_count"] = sum(
            event.unassigned_predicted_b_root_count for event in half_events
        )
        output["predicted_component_root_count"] = sum(
            event.predicted_component_root_count for event in half_events
        )
        output["unassigned_predicted_component_root_count"] = sum(
            event.unassigned_predicted_component_root_count
            for event in half_events
        )
        output["unit_semantics_counts"] = dict(
            sorted(Counter(event.unit_semantics for event in half_events).items())
        )
        output["both_halves_perfect_lcag"] = _sum_ratios(
            event.both_halves_perfect_lcag for event in half_events
        ).as_dict()
        output["both_halves_leave_one_out_lcag"] = _sum_ratios(
            event.both_halves_leave_one_out_lcag for event in half_events
        ).as_dict()
    return output


def _evaluate_root_pair(
    predicted_batch: Mapping[str, torch.Tensor],
    truth_batch: Mapping[str, torch.Tensor],
    predicted: _TreeView,
    truth: _TreeView,
    predicted_root: int | None,
    truth_root: int,
    *,
    event_index: int,
    scope: str,
    unit_index: int,
    truth_topology_mode: TruthTopologyMode,
    truth_target_mask: torch.Tensor,
    target_representable: bool,
    target_unrepresentable_reasons: tuple[str, ...],
) -> DecayEvaluation:
    truth_sources = tuple(sorted(truth.source_set(truth_root)))
    predicted_sources = (
        tuple(sorted(predicted.source_set(predicted_root)))
        if predicted_root is not None
        else ()
    )
    truth_lcag = _source_keyed_lca_signatures(truth, truth_root)
    try:
        predicted_lcag = (
            _source_keyed_lca_signatures(predicted, predicted_root)
            if predicted_root is not None
            else {}
        )
    except ValueError:
        predicted_lcag = {}
    pair_correct = sum(
        int(predicted_lcag.get(pair) == generation)
        for pair, generation in truth_lcag.items()
    )
    pair_metric = RatioMetric(float(pair_correct), float(len(truth_lcag)))
    structural_valid = _component_structurally_valid(predicted, predicted_root)
    topology_eligible = len(truth_sources) >= 2
    perfect = (
        topology_eligible
        and structural_valid
        and target_representable
        and bool(truth_sources)
        and predicted_sources == truth_sources
        and predicted_lcag == truth_lcag
    )
    strict_missing, leave_one_out, omitted_source = _one_leaf_metrics(
        truth,
        predicted,
        truth_root,
        predicted_root,
        truth_sources,
        predicted_sources,
        structurally_valid=structural_valid and target_representable,
    )
    truth_nodes = _descendants(truth, truth_root)
    predicted_nodes = (
        _descendants(predicted, predicted_root)
        if predicted_root is not None
        else set()
    )
    leaf_pid, leaf_confusion = _leaf_pid_metric(
        predicted, truth, truth_nodes
    )
    aligned_mothers, truth_mothers = _align_mothers_by_exact_sources(
        predicted,
        truth,
        predicted_nodes,
        truth_nodes,
        truth_mother_candidates={
            node
            for node in truth_nodes
            if node < truth_target_mask.numel() and bool(truth_target_mask[node])
        },
    )
    predicted_mothers = [
        node for node in predicted_nodes if predicted.children(node)
    ]
    truth_depth = _node_heights(truth, truth_nodes)[truth_root]
    try:
        predicted_depth = (
            _node_heights(predicted, predicted_nodes)[predicted_root]
            if predicted_root is not None
            else None
        )
    except ValueError:
        predicted_depth = None
    mother_pid, mother_pid_coverage, mother_confusion = _mother_pid_metric(
        predicted, truth, aligned_mothers, truth_mothers
    )
    root_pid = _root_pid_metric(predicted, truth, predicted_root, truth_root)
    kinematics = _kinematic_metrics(
        predicted, truth, aligned_mothers, len(truth_mothers)
    )
    return DecayEvaluation(
        scope=scope,
        unit_index=unit_index,
        available=True,
        unavailable_reason=None,
        truth_root_position=truth_root,
        predicted_root_position=predicted_root,
        truth_sources=truth_sources,
        predicted_sources=predicted_sources,
        source_recall=RatioMetric(
            float(len(set(truth_sources) & set(predicted_sources))),
            float(len(truth_sources)),
        ),
        source_precision=RatioMetric(
            float(len(set(truth_sources) & set(predicted_sources))),
            float(len(predicted_sources)),
        ),
        structurally_valid=RatioMetric(float(structural_valid), 1.0),
        target_representable=RatioMetric(float(target_representable), 1.0),
        lcag_pair_accuracy=pair_metric,
        perfect_lcag=RatioMetric(
            float(perfect) if topology_eligible else 0.0,
            1.0 if topology_eligible else 0.0,
        ),
        strict_missing_one_leaf=strict_missing,
        leave_one_out_lcag=leave_one_out,
        leaf_pid_accuracy=leaf_pid,
        mother_pid_accuracy=mother_pid,
        mother_pid_coverage=mother_pid_coverage,
        root_pid_accuracy=root_pid,
        kinematics=kinematics,
        omitted_source_key=omitted_source,
        truth_mother_count=len(truth_mothers),
        predicted_mother_count=len(predicted_mothers),
        matched_mother_count=len(aligned_mothers),
        leaf_pid_confusion=leaf_confusion,
        mother_pid_confusion=mother_confusion,
        truth_retained_depth=truth_depth,
        predicted_retained_depth=predicted_depth,
        truth_topology_mode=truth_topology_mode,
        target_unrepresentable_reasons=target_unrepresentable_reasons,
    )


def _one_leaf_metrics(
    truth: _TreeView,
    predicted: _TreeView,
    truth_root: int,
    predicted_root: int | None,
    truth_sources: Sequence[int],
    predicted_sources: Sequence[int],
    *,
    structurally_valid: bool,
) -> tuple[RatioMetric, RatioMetric, int | None]:
    truth_set = set(truth_sources)
    predicted_set = set(predicted_sources)
    # At least three comparable leaves must remain; otherwise a binary tree is
    # made trivially perfect by deleting one leaf.
    eligible = len(truth_set) >= 4
    if not eligible:
        return RatioMetric(), RatioMetric(), None
    missing = truth_set - predicted_set
    extras = predicted_set - truth_set
    shared = truth_set & predicted_set
    shared_exact = (
        structurally_valid
        and predicted_root is not None
        and _source_keyed_lca_signatures(
            truth, truth_root, retained_sources=shared
        )
        == _source_keyed_lca_signatures(
            predicted, predicted_root, retained_sources=shared
        )
    )
    strict = (
        len(missing) == 1
        and not extras
        and len(shared) >= 3
        and shared_exact
    )

    tolerant = False
    omitted_key: int | None = next(iter(missing)) if strict else None
    for omitted in sorted(truth_set):
        reduced_truth = truth_set - {omitted}
        reduced_predicted = predicted_set - {omitted}
        if reduced_truth != reduced_predicted or len(reduced_truth) < 3:
            continue
        if not structurally_valid or predicted_root is None:
            continue
        if _source_keyed_lca_signatures(
            truth, truth_root, retained_sources=reduced_truth
        ) == _source_keyed_lca_signatures(
            predicted, predicted_root, retained_sources=reduced_truth
        ):
            tolerant = True
            if omitted_key is None:
                omitted_key = omitted
            break
    return (
        RatioMetric(float(strict), 1.0),
        RatioMetric(float(tolerant), 1.0),
        omitted_key,
    )


def _leaf_pid_metric(
    predicted: _TreeView,
    truth: _TreeView,
    truth_nodes: set[int],
) -> tuple[RatioMetric, tuple[tuple[int, int, int], ...]]:
    truth_by_source: dict[int, int] = {}
    for node in sorted(truth_nodes):
        sources = truth.source_set(node)
        if truth.children(node) or len(sources) != 1:
            continue
        truth_by_source.setdefault(next(iter(sources)), node)
    # Leaf PID is a model metric independent of whether a leaf was linked into
    # the selected predicted root, so match against every predicted FSP.
    predicted_by_source: dict[int, int] = {}
    for node in predicted.positions:
        sources = predicted.source_set(node)
        if predicted.children(node) or len(sources) != 1:
            continue
        predicted_by_source.setdefault(next(iter(sources)), node)
    correct = denominator = 0
    confusion: Counter[tuple[int, int]] = Counter()
    for source, truth_node in truth_by_source.items():
        if truth.truth_pid_available is not None and not bool(
            truth.truth_pid_available[truth_node]
        ):
            continue
        denominator += 1
        predicted_node = predicted_by_source.get(source)
        truth_token = int(truth.pid[truth_node])
        predicted_token = (
            int(predicted.pid[predicted_node])
            if predicted_node is not None
            else -1
        )
        confusion[(truth_token, predicted_token)] += 1
        correct += int(
            predicted_node is not None
            and predicted_token == truth_token
        )
    return (
        RatioMetric(float(correct), float(denominator)),
        tuple(
            (truth_token, predicted_token, count)
            for (truth_token, predicted_token), count in sorted(confusion.items())
        ),
    )


def _align_mothers_by_exact_sources(
    predicted: _TreeView,
    truth: _TreeView,
    predicted_nodes: set[int],
    truth_nodes: set[int],
    *,
    truth_mother_candidates: set[int] | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    predicted_by_topology: dict[tuple[Any, ...], list[int]] = {}
    truth_by_topology: dict[tuple[Any, ...], list[int]] = {}
    predicted_heights = _node_heights(predicted, predicted_nodes) if predicted_nodes else {}
    truth_heights = _node_heights(truth, truth_nodes)
    truth_mothers = [
        node
        for node in truth_nodes
        if truth.children(node)
        and (
            truth_mother_candidates is None
            or node in truth_mother_candidates
        )
    ]
    for node in predicted_nodes:
        if predicted.children(node):
            key = _mother_topology_key(predicted, predicted_heights, node)
            predicted_by_topology.setdefault(key, []).append(node)
    for node in truth_mothers:
        key = _mother_topology_key(truth, truth_heights, node)
        truth_by_topology.setdefault(key, []).append(node)

    aligned: list[tuple[int, int]] = []
    for topology, truth_group in truth_by_topology.items():
        predicted_group = predicted_by_topology.get(topology, [])
        truth_group.sort()
        predicted_group.sort()
        aligned.extend(zip(predicted_group, truth_group, strict=False))
    return aligned, truth_mothers


def _mother_topology_key(
    view: _TreeView, heights: Mapping[int, int], node: int
) -> tuple[Any, ...]:
    return (
        tuple(sorted(view.source_set(node))),
        heights[node],
        tuple(
            sorted(
                tuple(sorted(view.source_set(child)))
                for child in view.children(node)
            )
        ),
    )


def _mother_pid_metric(
    predicted: _TreeView,
    truth: _TreeView,
    aligned: Sequence[tuple[int, int]],
    truth_mothers: Sequence[int],
) -> tuple[
    RatioMetric,
    RatioMetric,
    tuple[tuple[int, int, int], ...],
]:
    truth_eligible = [
        node
        for node in truth_mothers
        if truth.truth_pid_available is None or bool(truth.truth_pid_available[node])
    ]
    eligible_set = set(truth_eligible)
    eligible_aligned = [
        (pred_node, truth_node)
        for pred_node, truth_node in aligned
        if truth_node in eligible_set
    ]
    confusion: Counter[tuple[int, int]] = Counter(
        (
            int(truth.pid[truth_node]),
            int(predicted.pid[pred_node]),
        )
        for pred_node, truth_node in eligible_aligned
    )
    correct = sum(
        count
        for (truth_token, predicted_token), count in confusion.items()
        if truth_token == predicted_token
    )
    return (
        RatioMetric(float(correct), float(len(eligible_aligned))),
        RatioMetric(float(len(eligible_aligned)), float(len(truth_eligible))),
        tuple(
            (truth_token, predicted_token, count)
            for (truth_token, predicted_token), count in sorted(confusion.items())
        ),
    )


def _root_pid_metric(
    predicted: _TreeView,
    truth: _TreeView,
    predicted_root: int | None,
    truth_root: int,
) -> RatioMetric:
    if truth.truth_pid_available is not None and not bool(
        truth.truth_pid_available[truth_root]
    ):
        return RatioMetric()
    correct = int(
        predicted_root is not None
        and int(predicted.pid[predicted_root]) == int(truth.pid[truth_root])
    )
    return RatioMetric(float(correct), 1.0)


def _kinematic_metrics(
    predicted: _TreeView,
    truth: _TreeView,
    aligned: Sequence[tuple[int, int]],
    truth_mother_count: int,
) -> KinematicErrorMetrics:
    coverage = RatioMetric(float(len(aligned)), float(truth_mother_count))
    if predicted.p4 is None or truth.p4 is None or not aligned:
        return KinematicErrorMetrics(alignment_coverage=coverage)
    p4_l1 = p3 = relative_p3 = energy = mass = magnitude = 0.0
    mass_bias = mass_squared = 0.0
    relative_p3_count = 0
    component_bias = torch.zeros(4, dtype=torch.float64)
    component_absolute = torch.zeros(4, dtype=torch.float64)
    component_squared = torch.zeros(4, dtype=torch.float64)
    p3_squared = 0.0
    for predicted_node, truth_node in aligned:
        pred = predicted.p4[predicted_node].to(torch.float64)
        target = truth.p4[truth_node].to(torch.float64)
        delta = pred - target
        component_bias += delta
        component_absolute += delta.abs()
        component_squared += delta.square()
        p4_l1 += float(delta.abs().sum())
        delta_p3 = float(torch.linalg.vector_norm(delta[:3]))
        p3 += delta_p3
        p3_squared += delta_p3**2
        energy += abs(float(delta[3]))
        pred_momentum = float(torch.linalg.vector_norm(pred[:3]))
        truth_momentum = float(torch.linalg.vector_norm(target[:3]))
        if truth_momentum > 0.0:
            relative_p3 += delta_p3 / truth_momentum
            relative_p3_count += 1
        magnitude += abs(pred_momentum - truth_momentum)
        pred_mass = math.sqrt(max(float(pred[3] ** 2 - pred[:3].square().sum()), 0.0))
        truth_mass = math.sqrt(
            max(float(target[3] ** 2 - target[:3].square().sum()), 0.0)
        )
        mass_delta = pred_mass - truth_mass
        mass += abs(mass_delta)
        mass_bias += mass_delta
        mass_squared += mass_delta**2
    count = float(len(aligned))
    return KinematicErrorMetrics(
        alignment_coverage=coverage,
        p4_l1=RatioMetric(p4_l1, 4.0 * count),
        p3=RatioMetric(p3, count),
        relative_p3=RatioMetric(relative_p3, float(relative_p3_count)),
        p3_squared_error=RatioMetric(p3_squared, count),
        px_bias=RatioMetric(float(component_bias[0]), count),
        py_bias=RatioMetric(float(component_bias[1]), count),
        pz_bias=RatioMetric(float(component_bias[2]), count),
        energy_bias=RatioMetric(float(component_bias[3]), count),
        px_mae=RatioMetric(float(component_absolute[0]), count),
        py_mae=RatioMetric(float(component_absolute[1]), count),
        pz_mae=RatioMetric(float(component_absolute[2]), count),
        energy=RatioMetric(energy, count),
        px_squared_error=RatioMetric(float(component_squared[0]), count),
        py_squared_error=RatioMetric(float(component_squared[1]), count),
        pz_squared_error=RatioMetric(float(component_squared[2]), count),
        energy_squared_error=RatioMetric(float(component_squared[3]), count),
        mass=RatioMetric(mass, count),
        mass_bias=RatioMetric(mass_bias, count),
        mass_squared_error=RatioMetric(mass_squared, count),
        momentum_magnitude=RatioMetric(magnitude, count),
    )


def _tree_view(
    batch: Mapping[str, torch.Tensor],
    event_index: int,
    *,
    truth: bool,
    source_keys_override: Sequence[int] | torch.Tensor | None = None,
    target_policy: TargetPolicy | None = None,
    minimum_daughters: int = 2,
) -> _TreeView:
    if "node_mask" in batch:
        active = _event_tensor(batch["node_mask"], event_index, 1).bool()
    elif "active" in batch:
        active = _event_tensor(batch["active"], event_index, 1).bool()
    else:
        raise KeyError("batch requires node_mask or active")
    node_count = active.shape[0]
    if "daughter_adjacency" in batch:
        adjacency = _event_tensor(
            batch["daughter_adjacency"], event_index, 2
        ).bool()[:node_count, :node_count]
    elif "parent_ids" in batch:
        parents = _event_tensor(batch["parent_ids"], event_index, 1).long()
        adjacency = torch.zeros((node_count, node_count), dtype=torch.bool)
        for child, parent in enumerate(parents.tolist()):
            if 0 <= int(parent) < node_count:
                adjacency[int(parent), child] = True
    else:
        raise KeyError("batch requires daughter_adjacency or parent_ids")
    active = active[:node_count].detach().cpu()
    adjacency = adjacency.detach().cpu()
    adjacency &= active[:, None] & active[None, :]
    if truth and target_policy is not None:
        target_mask = _truth_target_mask(
            batch,
            event_index,
            target_policy=target_policy,
            minimum_daughters=minimum_daughters,
        ).detach().cpu()[:node_count]
        leaf_mask = _truth_leaf_mask(batch, event_index, active, adjacency)
        active, adjacency = _contract_truth_adjacency(
            active,
            adjacency,
            retained=leaf_mask | target_mask,
        )
    sources, source_keys = _canonical_leaf_membership(
        batch,
        event_index,
        active=active,
        adjacency=adjacency,
        override=source_keys_override,
    )

    if truth:
        if "truth_pid_labels" in batch:
            truth_labels = _event_tensor(
                batch["truth_pid_labels"], event_index, 1
            ).long()[:node_count]
            availability = (
                _event_tensor(batch["truth_pid_available"], event_index, 1)
                .bool()[:node_count]
                if "truth_pid_available" in batch
                else torch.ones(node_count, dtype=torch.bool)
            )
            fallback_key = "pid_target_labels" if "pid_target_labels" in batch else "pid_labels"
            fallback = _event_tensor(batch[fallback_key], event_index, 1).long()[:node_count]
            pid = torch.where(availability, truth_labels, fallback)
        else:
            key = "pid_target_labels" if "pid_target_labels" in batch else "pid_labels"
            pid = _event_tensor(batch[key], event_index, 1).long()[:node_count]
            availability = torch.ones(node_count, dtype=torch.bool)
    else:
        key = "current_pid_tokens" if "current_pid_tokens" in batch else "pid_labels"
        pid = _event_tensor(batch[key], event_index, 1).long()[:node_count]
        availability = None
    p4 = (
        _event_tensor(batch["p4"], event_index, 2).float()[:node_count]
        if "p4" in batch
        else None
    )
    if truth and target_policy is not None and p4 is not None:
        p4 = _daughter_sum_p4(p4.detach().cpu(), active, adjacency)
    b_side = (
        _event_tensor(batch["b_side"], event_index, 1).long()[:node_count]
        if "b_side" in batch
        else None
    )
    detector_sources = (
        _event_tensor(
            batch["recursive_leaf_source_mask"], event_index, 2
        ).bool()[:node_count]
        if "recursive_leaf_source_mask" in batch
        else None
    )
    return _TreeView(
        active=active,
        adjacency=adjacency,
        sources=sources.detach().cpu(),
        source_keys=source_keys,
        pid=pid.detach().cpu(),
        truth_pid_available=(availability.detach().cpu() if availability is not None else None),
        p4=(p4.detach().cpu() if p4 is not None else None),
        b_side=(b_side.detach().cpu() if b_side is not None else None),
        detector_sources=(
            detector_sources.detach().cpu()
            if detector_sources is not None
            else None
        ),
    )


def _event_tensor(
    value: torch.Tensor, event_index: int, unbatched_ndim: int
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == unbatched_ndim:
        if event_index != 0:
            raise IndexError("event_index must be zero for an unbatched event")
        return tensor
    if tensor.ndim == unbatched_ndim + 1:
        return tensor[event_index]
    raise ValueError(
        f"expected tensor rank {unbatched_ndim} or {unbatched_ndim + 1}, "
        f"found shape {tuple(tensor.shape)}"
    )


def _truth_leaf_mask(
    batch: Mapping[str, torch.Tensor],
    event_index: int,
    active: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Select the same detector FSP ontology used by strict inference."""

    if "level_ids" in batch:
        levels = _event_tensor(batch["level_ids"], event_index, 1).long()
        leaves = active & (levels[: active.numel()].detach().cpu() == 0)
    else:
        leaves = active & ~adjacency.any(dim=-1)
    if "node_kind_ids" in batch:
        kinds = (
            _event_tensor(batch["node_kind_ids"], event_index, 1)
            .long()[: active.numel()]
            .detach()
            .cpu()
        )
        detector_kinds = torch.zeros_like(leaves)
        for name in ("track", "ecl_cluster", "klm_cluster"):
            detector_kinds |= kinds == NODE_KIND_TO_ID[name]
        leaves &= detector_kinds
    return leaves


def _truth_target_mask(
    batch: Mapping[str, torch.Tensor],
    event_index: int,
    *,
    target_policy: TargetPolicy,
    minimum_daughters: int,
) -> torch.Tensor:
    """Return composites the checkpoint policy actually presents as targets."""

    if target_policy not in {
        "complete_only",
        "reconstructable_partial",
        "diagnostic_all",
    }:
        raise ValueError(f"unknown reconstruction target policy: {target_policy}")
    if minimum_daughters <= 0:
        raise ValueError("minimum_daughters must be positive")
    if "node_mask" in batch:
        active = _event_tensor(batch["node_mask"], event_index, 1).bool()
    else:
        active = _event_tensor(batch["active"], event_index, 1).bool()
    node_count = active.numel()
    adjacency = _event_tensor(
        batch["daughter_adjacency"], event_index, 2
    ).bool()[:node_count, :node_count]
    if "level_ids" in batch:
        higher = (
            _event_tensor(batch["level_ids"], event_index, 1).long()[:node_count]
            > 0
        )
    else:
        higher = adjacency.any(dim=-1)
    daughter_count = (adjacency & active[None, :]).sum(dim=-1)
    eligible = active & higher & (daughter_count >= minimum_daughters)
    if target_policy != "diagnostic_all" and "valid_reconstruction_target" in batch:
        eligible &= _event_tensor(
            batch["valid_reconstruction_target"], event_index, 1
        ).bool()[:node_count]
    if target_policy == "complete_only":
        complete_name = (
            "recursive_reconstructable_complete"
            if "recursive_reconstructable_complete" in batch
            else "complete_reconstructable_decay"
        )
        if complete_name in batch:
            eligible &= _event_tensor(
                batch[complete_name], event_index, 1
            ).bool()[:node_count]
    return eligible.detach().cpu()


def _validate_truth_topology_mode(mode: str) -> None:
    if mode not in {"checkpoint_direct", "contracted_diagnostic"}:
        raise ValueError(f"unknown truth topology mode: {mode}")


def _unit_ineligible_intermediate_count(
    view: _TreeView,
    root: int,
    target_mask: torch.Tensor,
) -> int:
    """Count direct-tree composites that free rollout cannot construct."""

    return sum(
        int(bool(view.children(node)) and not bool(target_mask[node]))
        for node in _descendants(view, root)
    )


def _unit_detector_source_conflict_count(
    batch: Mapping[str, torch.Tensor],
    event_index: int,
    view: _TreeView,
    root: int,
) -> int:
    """Count detector resources reused by two FSPs in one truth unit."""

    if "recursive_leaf_source_mask" not in batch:
        return 0
    provenance = _event_tensor(
        batch["recursive_leaf_source_mask"], event_index, 2
    ).bool().detach().cpu()
    nodes = _descendants(view, root)
    fsp_nodes = [
        node
        for node in nodes
        if not view.children(node) and len(view.source_set(node)) == 1
    ]
    if not fsp_nodes or provenance.shape[-1] == 0:
        return 0
    usage = provenance[fsp_nodes].sum(dim=0)
    return int((usage > 1).sum())


def _unit_representability(
    batch: Mapping[str, torch.Tensor],
    event_index: int,
    raw_truth: _TreeView,
    root: int,
    target_mask: torch.Tensor,
    *,
    truth_topology_mode: TruthTopologyMode,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a truth unit can be emitted under the scored contract.

    A policy-eligible root remains an available trial even when its direct
    topology cannot be emitted.  This prevents target incompatibilities from
    disappearing from the perfect-topology denominator.  Contracting an
    ineligible intermediate is allowed only in the explicitly diagnostic
    topology mode; detector-source reuse is unrepresentable in both modes.
    """

    reasons: list[str] = []
    if (
        truth_topology_mode == "checkpoint_direct"
        and _unit_ineligible_intermediate_count(raw_truth, root, target_mask) > 0
    ):
        reasons.append("ineligible_direct_intermediate")
    if _unit_detector_source_conflict_count(
        batch, event_index, raw_truth, root
    ) > 0:
        reasons.append("recursive_detector_source_conflict")
    return not reasons, tuple(reasons)


def _contract_truth_adjacency(
    active: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    retained: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress non-target intermediates and connect to their first frontier."""

    retained = retained.bool() & active
    output = torch.zeros_like(adjacency, dtype=torch.bool)
    memo: dict[int, tuple[int, ...]] = {}
    visiting: set[int] = set()

    def frontier(node: int) -> tuple[int, ...]:
        if bool(retained[node]):
            return (node,)
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError("cycle detected while contracting truth targets")
        visiting.add(node)
        values: set[int] = set()
        for child in (adjacency[node] & active).nonzero(
            as_tuple=False
        ).flatten().tolist():
            values.update(frontier(int(child)))
        visiting.remove(node)
        memo[node] = tuple(sorted(values))
        return memo[node]

    for mother in retained.nonzero(as_tuple=False).flatten().tolist():
        for child in (adjacency[mother] & active).nonzero(
            as_tuple=False
        ).flatten().tolist():
            for target in frontier(int(child)):
                if target != mother:
                    output[mother, target] = True
    output &= retained[:, None] & retained[None, :]
    return retained, output


def _daughter_sum_p4(
    p4: torch.Tensor,
    active: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Rebuild retained mother p4 from contracted detector-leaf inputs."""

    output = p4.clone()
    memo: dict[int, torch.Tensor] = {}
    visiting: set[int] = set()

    def value(node: int) -> torch.Tensor:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError("cycle detected while rebuilding truth p4")
        visiting.add(node)
        children = (adjacency[node] & active).nonzero(
            as_tuple=False
        ).flatten().tolist()
        resolved = (
            output[node]
            if not children
            else torch.stack([value(int(child)) for child in children]).sum(dim=0)
        )
        visiting.remove(node)
        memo[node] = resolved
        return resolved

    for node in active.nonzero(as_tuple=False).flatten().tolist():
        output[node] = value(int(node))
    return output


def _canonical_leaf_membership(
    batch: Mapping[str, torch.Tensor],
    event_index: int,
    *,
    active: torch.Tensor,
    adjacency: torch.Tensor,
    override: Sequence[int] | torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Build unique FSP membership from topology, not conflict provenance."""

    if "level_ids" in batch:
        levels = _event_tensor(batch["level_ids"], event_index, 1).long()
        leaf_mask = active & (levels[: active.numel()] == 0)
    else:
        leaf_mask = active & ~(adjacency & active[None, :]).any(dim=-1)
    leaf_positions = [
        int(value) for value in leaf_mask.nonzero(as_tuple=False).flatten()
    ]
    if not leaf_positions:
        return torch.zeros((active.numel(), 0), dtype=torch.bool), ()
    if any(bool((adjacency[position] & active).any()) for position in leaf_positions):
        raise ValueError("level-zero FSPs must not have daughters")

    explicit: torch.Tensor | None = None
    if override is not None:
        explicit = torch.as_tensor(override)
        if explicit.ndim == 2:
            explicit = explicit[event_index]
    else:
        for name in (
            "evaluation_leaf_source_keys",
            "evaluation_fsp_source_keys",
            "leaf_source_keys",
        ):
            if name in batch:
                explicit = _event_tensor(batch[name], event_index, 1)
                break
    if explicit is not None:
        if explicit.numel() < len(leaf_positions):
            raise ValueError(
                "evaluation leaf source keys are narrower than the active FSP set"
            )
        keys = tuple(
            int(value)
            for value in explicit[: len(leaf_positions)].detach().cpu()
        )
        if any(key < 0 for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("evaluation FSP keys must be nonnegative and unique")
    else:
        identity_fields: list[torch.Tensor] = []
        for name in ("reco_ids", "source_node_ids", "node_ids"):
            if name in batch:
                identity_fields.append(
                    _event_tensor(batch[name], event_index, 1).long().detach().cpu()
                )
        selected_keys: list[int] = []
        used: set[int] = set()
        for position in leaf_positions:
            key: int | None = None
            for values in identity_fields:
                candidate = int(values[position])
                if candidate >= 0 and candidate not in used:
                    key = candidate
                    break
            if key is None:
                candidate = int(position)
                while candidate in used:
                    candidate += active.numel()
                key = candidate
            selected_keys.append(key)
            used.add(key)
        keys = tuple(selected_keys)

    membership = torch.zeros(
        (active.numel(), len(leaf_positions)), dtype=torch.bool
    )
    leaf_column = {
        position: column for column, position in enumerate(leaf_positions)
    }
    memo: dict[int, torch.Tensor] = {}
    visiting: set[int] = set()

    def descendants(node: int) -> torch.Tensor:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError("cycle detected while constructing FSP membership")
        visiting.add(node)
        value = torch.zeros(len(leaf_positions), dtype=torch.bool)
        if node in leaf_column:
            value[leaf_column[node]] = True
        else:
            for child in (adjacency[node] & active).nonzero(
                as_tuple=False
            ).flatten().tolist():
                value |= descendants(int(child))
        visiting.remove(node)
        memo[node] = value
        return value

    for node in active.nonzero(as_tuple=False).flatten().tolist():
        membership[node] = descendants(int(node))
    return membership, keys


def _root_positions(view: _TreeView) -> list[int]:
    has_parent = (view.adjacency & view.active[:, None] & view.active[None, :]).any(dim=0)
    return [node for node in view.positions if not bool(has_parent[node])]


def _choose_largest_root(view: _TreeView) -> int | None:
    roots = _root_positions(view)
    if not roots:
        return None
    return max(
        roots,
        key=lambda node: (len(view.source_set(node)), tuple(sorted(view.source_set(node)))),
    )


def _truth_full_roots(view: _TreeView) -> list[int]:
    roots = set(_root_positions(view))
    # Full-event evaluation needs an explicit declared initial-state root.
    # Continuum schema-v4 forests do not have one, so choosing the largest
    # truth component would silently invent an event target.
    candidates = [
        node
        for node in view.positions
        if node in roots and int(view.pid[node]) == UPSILON_TOKEN
    ]
    return sorted(
        candidates,
        key=lambda node: (tuple(sorted(view.source_set(node))), node),
    )


def _choose_truth_root(view: _TreeView) -> int | None:
    """Compatibility helper for callers that only need the unique root."""

    roots = _truth_full_roots(view)
    return roots[0] if len(roots) == 1 else None


def _best_root_for_sources(
    view: _TreeView,
    target_sources: frozenset[int],
    *,
    candidates: Sequence[int],
) -> int | None:
    scored = [node for node in candidates if view.source_set(node) & target_sources]
    if not scored:
        return None

    def score(node: int) -> tuple[float, int, int, tuple[int, ...]]:
        sources = view.source_set(node)
        union = sources | target_sources
        return (
            len(sources & target_sources) / len(union),
            int(sources == target_sources),
            -len(sources ^ target_sources),
            tuple(sorted(sources)),
        )

    return max(scored, key=score)


def _truth_b_roots(view: _TreeView) -> list[int]:
    if view.b_side is not None and bool((view.b_side >= 0).any()):
        roots: list[int] = []
        for side in (0, 1):
            side_nodes = [
                node
                for node in view.positions
                if int(view.b_side[node]) == side
            ]
            side_set = set(side_nodes)
            side_roots = []
            for node in side_nodes:
                parents = (
                    view.adjacency[:, node] & view.active
                ).nonzero(as_tuple=False).flatten().tolist()
                if not any(int(parent) in side_set for parent in parents):
                    side_roots.append(node)
            if len(side_roots) != 1:
                return []
            root = side_roots[0]
            if int(view.pid[root]) not in B_ROOT_TOKENS:
                return []
            roots.append(root)
        if view.source_set(roots[0]) & view.source_set(roots[1]):
            return []
        return roots
    candidates = [node for node in view.positions if int(view.pid[node]) in B_ROOT_TOKENS]
    candidate_set = set(candidates)
    roots: list[int] = []
    for node in candidates:
        ancestors = _ancestors(view, node) - {node}
        if not (ancestors & candidate_set):
            roots.append(node)
    return roots


def _is_continuum_category(source_category: str | None) -> bool:
    if source_category is None:
        return False
    normalized = source_category.lower().replace("_", "").replace("-", "")
    return "continuum" in normalized or normalized in {
        "ccbar",
        "uubar",
        "ddbar",
        "ssbar",
        "qqbar",
    }


def _assign_two_by_sources(
    predicted: _TreeView,
    candidates: Sequence[int],
    truth: _TreeView,
    truth_roots: Sequence[int],
) -> tuple[int | None, ...]:
    if not truth_roots:
        return ()
    if len(truth_roots) == 1:
        return (
            _best_root_for_sources(
                predicted,
                truth.source_set(truth_roots[0]),
                candidates=candidates,
            ),
        )
    if len(truth_roots) != 2:
        raise ValueError("B-half source assignment supports exactly one or two roots")
    options: list[int | None] = [None, *candidates]
    best_assignment: tuple[int | None, int | None] = (None, None)
    best_score: tuple[int, float, int] = (-1, -1.0, -1)
    for left in options:
        for right in options:
            if left is not None and left == right:
                continue
            if (
                left is not None
                and right is not None
                and predicted.source_set(left) & predicted.source_set(right)
            ):
                continue
            assignment = (left, right)
            jaccard = exact = intersection = 0.0
            for candidate, target in zip(assignment, truth_roots, strict=True):
                if candidate is None:
                    continue
                candidate_sources = predicted.source_set(candidate)
                target_sources = truth.source_set(target)
                union = candidate_sources | target_sources
                jaccard += len(candidate_sources & target_sources) / len(union) if union else 1.0
                exact += int(candidate_sources == target_sources)
                intersection += len(candidate_sources & target_sources)
            # Preserve an exact half before improving aggregate overlap on the
            # other slot.  Otherwise two mixed candidates can outscore one
            # exact candidate plus one honest unmatched half.
            score = (int(exact), jaccard, int(intersection))
            if score > best_score:
                best_score = score
                best_assignment = assignment
    return best_assignment


def _assign_many_by_sources(
    predicted: _TreeView,
    candidates: Sequence[int],
    truth: _TreeView,
    truth_roots: Sequence[int],
) -> tuple[int | None, ...]:
    """One-to-one source-overlap assignment for continuum components."""

    if not truth_roots:
        return ()
    if not candidates:
        return (None,) * len(truth_roots)

    from hypertagging.losses.set_matching import hungarian_assignment

    intersections = torch.zeros(
        (len(candidates), len(truth_roots)), dtype=torch.float64
    )
    jaccards = torch.zeros_like(intersections)
    exact = torch.zeros_like(intersections)
    for candidate_index, candidate in enumerate(candidates):
        candidate_sources = predicted.source_set(candidate)
        for truth_index, target in enumerate(truth_roots):
            target_sources = truth.source_set(target)
            intersection = len(candidate_sources & target_sources)
            union = len(candidate_sources | target_sources)
            intersections[candidate_index, truth_index] = intersection
            jaccards[candidate_index, truth_index] = (
                intersection / union if union else 1.0
            )
            exact[candidate_index, truth_index] = float(
                candidate_sources == target_sources
            )

    # One exact component must dominate every possible aggregate secondary
    # score.  The last term only provides a deterministic overlap preference
    # after exactness and Jaccard similarity.
    truth_count = len(truth_roots)
    maximum_intersection = max(float(intersections.max()), 1.0)
    score = (
        exact * float(2 * truth_count + 1)
        + jaccards
        + intersections / maximum_intersection / float(truth_count + 1)
    )
    pairs = hungarian_assignment(
        -score,
        production=False,
        allow_bruteforce=True,
    )
    output: list[int | None] = [None] * truth_count
    for candidate_index, truth_index in pairs:
        if intersections[candidate_index, truth_index] > 0:
            output[truth_index] = int(candidates[candidate_index])
    return tuple(output)


def _descendants(view: _TreeView, root: int) -> set[int]:
    output: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in output:
            continue
        output.add(node)
        stack.extend(view.children(node))
    return output


def _component_structurally_valid(
    view: _TreeView, root: int | None
) -> bool:
    """Validate the selected rooted component independently of node IDs."""

    if root is None or root < 0 or root >= view.active.numel():
        return False
    if not bool(view.active[root]):
        return False
    nodes = _descendants(view, root)
    if not nodes or any(not view.source_set(node) for node in nodes):
        return False

    # Detect directed cycles explicitly; source-union construction alone can
    # otherwise hide a malformed repeated edge traversal.
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        if not all(visit(child) for child in view.children(node) if child in nodes):
            return False
        visiting.remove(node)
        visited.add(node)
        return True

    if not visit(root):
        return False

    for node in nodes:
        parents = [
            int(value)
            for value in (view.adjacency[:, node] & view.active)
            .nonzero(as_tuple=False)
            .flatten()
        ]
        internal_parents = [parent for parent in parents if parent in nodes]
        if node == root:
            # A B-half root may itself have one parent (for example an
            # Upsilon) outside the evaluated descendant component.
            if internal_parents or len(parents) > 1:
                return False
        elif len(internal_parents) != 1 or len(parents) != 1:
            return False
        children = [child for child in view.children(node) if child in nodes]
        child_sources = [view.source_set(child) for child in children]
        for left, right in combinations(child_sources, 2):
            if left & right:
                return False
        if children and frozenset().union(*child_sources) != view.source_set(node):
            return False
        if children and view.detector_sources is not None:
            detector_usage = view.detector_sources[children].sum(dim=0)
            if bool((detector_usage > 1).any()):
                return False
    return True


def _ancestors(view: _TreeView, node: int) -> set[int]:
    output: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in output:
            continue
        output.add(current)
        parents = (view.adjacency[:, current] & view.active).nonzero(as_tuple=False).flatten()
        stack.extend(int(parent) for parent in parents)
    return output


def _node_heights(view: _TreeView, nodes: set[int]) -> dict[int, int]:
    memo: dict[int, int] = {}
    visiting: set[int] = set()

    def height(node: int) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError("cycle detected while constructing source-keyed LCAG")
        visiting.add(node)
        children = [child for child in view.children(node) if child in nodes]
        value = 0 if not children else 1 + max(height(child) for child in children)
        visiting.remove(node)
        memo[node] = value
        return value

    for node in nodes:
        height(node)
    return memo


def _require_active_position(view: _TreeView, position: int, name: str) -> None:
    if position < 0 or position >= view.active.numel() or not bool(view.active[position]):
        raise IndexError(f"{name}={position} is not an active node position")


def _unavailable_row(
    scope: str,
    unit_index: int,
    reason: str,
    *,
    truth_root_position: int | None = None,
    truth_sources: tuple[int, ...] = (),
    truth_topology_mode: TruthTopologyMode = "checkpoint_direct",
) -> DecayEvaluation:
    return DecayEvaluation(
        scope=scope,
        unit_index=unit_index,
        available=False,
        unavailable_reason=reason,
        truth_root_position=truth_root_position,
        predicted_root_position=None,
        truth_sources=truth_sources,
        predicted_sources=(),
        source_recall=RatioMetric(),
        source_precision=RatioMetric(),
        structurally_valid=RatioMetric(),
        target_representable=RatioMetric(),
        lcag_pair_accuracy=RatioMetric(),
        perfect_lcag=RatioMetric(),
        strict_missing_one_leaf=RatioMetric(),
        leave_one_out_lcag=RatioMetric(),
        leaf_pid_accuracy=RatioMetric(),
        mother_pid_accuracy=RatioMetric(),
        mother_pid_coverage=RatioMetric(),
        root_pid_accuracy=RatioMetric(),
        kinematics=KinematicErrorMetrics(),
        truth_topology_mode=truth_topology_mode,
    )


def _unavailable_halves(
    reason: str,
    *,
    unit_semantics: str = "b_halves",
) -> HalfDecayEvaluation:
    return HalfDecayEvaluation(
        available=False,
        unavailable_reason=reason,
        halves=(),
        both_halves_perfect_lcag=RatioMetric(),
        both_halves_leave_one_out_lcag=RatioMetric(),
        unit_semantics=unit_semantics,
    )


def _flatten_ratio(
    output: dict[str, Any], name: str, metric: RatioMetric, *, boolean: bool = False
) -> None:
    value = metric.value
    output[name] = (bool(value) if value is not None else None) if boolean else value
    output[f"{name}_numerator"] = float(metric.numerator)
    output[f"{name}_denominator"] = float(metric.denominator)


def _sum_ratios(metrics: Iterable[RatioMetric]) -> RatioMetric:
    numerator = denominator = 0.0
    for metric in metrics:
        numerator += metric.numerator
        denominator += metric.denominator
    return RatioMetric(numerator, denominator)


def _rmse_dict(metric: RatioMetric) -> dict[str, float | None]:
    value = metric.value
    return {
        "value": math.sqrt(value) if value is not None else None,
        "numerator": float(metric.numerator),
        "denominator": float(metric.denominator),
    }


__all__ = [
    "B_ROOT_TOKENS",
    "UPSILON_TOKEN",
    "RatioMetric",
    "KinematicErrorMetrics",
    "DecayEvaluation",
    "HalfDecayEvaluation",
    "TargetPolicy",
    "TruthTopologyMode",
    "canonical_fsp_membership",
    "truth_target_policy_diagnostics",
    "source_keyed_lcag",
    "evaluate_full_decay",
    "evaluate_half_decays",
    "summarize_decay_evaluations",
]
