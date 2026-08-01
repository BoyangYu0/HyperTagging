"""Corrected, truth-separated direct-mDST schema-v3.

V1/v2 files remain readable through a conservative adapter.  Their historical
PID/charge semantics are explicitly marked as legacy-conflated; detector
features that were not stored are unavailable rather than fabricated.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import awkward as ak

from hypertagging.preprocessing.channels import event_channel_record
from hypertagging.preprocessing.levelize_tree import assign_levels, legacy_depth_samples, nodes_by_level
from hypertagging.preprocessing.mdst_tree_builder import EventTree, TreeNode
from hypertagging.preprocessing.pid_filter import (
    PDG_TOKENS,
    PID_VOCABULARY_VERSION,
    tokenize_pdg,
    validate_pid_token,
)
from hypertagging.preprocessing.schema_v2 import (
    CLUSTER_FEATURE_NAMES,
    COMMON_FEATURE_NAMES,
    COMPOSITE_FEATURE_NAMES,
    NODE_KINDS,
    NODE_KIND_TO_ID,
    SCHEMA_VERSION_V1,
    SCHEMA_VERSION_V2,
    TRACK_FEATURE_NAMES,
    adapt_payload_to_v2,
    infer_node_kind,
)
CHARGED_STABLE_NAMES: tuple[str, ...] = ("electron", "muon", "pion", "kaon", "proton")
CANONICAL_TRACK_HYPOTHESIS = "pion"


SCHEMA_VERSION_V3 = "direct-mdst-tree-v3"

V3_COMMON_FEATURE_NAMES = COMMON_FEATURE_NAMES
V3_TRACK_FEATURE_NAMES = TRACK_FEATURE_NAMES + tuple(
    f"pid_log_likelihood_{name}" for name in CHARGED_STABLE_NAMES
) + tuple(f"energy_hypothesis_{name}" for name in CHARGED_STABLE_NAMES)
V3_CLUSTER_FEATURE_NAMES = CLUSTER_FEATURE_NAMES
V3_COMPOSITE_FEATURE_NAMES = COMPOSITE_FEATURE_NAMES + (
    "full_truth_daughter_count",
    "retained_daughter_count",
    "reconstructed_daughter_count",
    "partial_missing_daughters",
)


def feature_spec_v3() -> dict[str, Any]:
    spec = {
        "schema_version": SCHEMA_VERSION_V3,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "pid_tokens": list(PDG_TOKENS),
        "common": list(V3_COMMON_FEATURE_NAMES),
        "track": list(V3_TRACK_FEATURE_NAMES),
        "ecl_cluster": list(V3_CLUSTER_FEATURE_NAMES),
        "composite": list(V3_COMPOSITE_FEATURE_NAMES),
        "node_kinds": list(NODE_KINDS),
        "canonical_track_hypothesis": CANONICAL_TRACK_HYPOTHESIS,
        "leaf_kinematics_modes": [
            "raw_track_predicted_pid",
            "fixed_hypothesis_candidate",
            "truth_topology_only",
            "legacy_conflated",
        ],
        "missing_value_policy": (
            "Zero is only tensor padding. Every heterogeneous feature has an "
            "explicit availability mask; missing detector quantities are never inferred."
        ),
        "reconstructed_p4_contract": (
            "track p3 from a reconstructed fit; track energy from a data-independent "
            "canonical or predicted PID mass; ECL p4 from cluster measurement; "
            "composite p4 is the recursive daughter sum"
        ),
    }
    spec["feature_spec_hash"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return spec


def export_trees_v3(
    trees: Iterable[EventTree],
    output: str | Path,
    *,
    summary: dict[str, object] | None = None,
    legacy_levels: bool = True,
    charge_conjugate_normalize: bool = False,
) -> Path:
    """Write bounded schema-v3 event records."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = feature_spec_v3()
    events: list[dict[str, Any]] = []
    level_records: list[dict[str, object]] = []
    for tree in trees:
        assign_levels(tree)
        events.append(
            _event_record_v3(
                tree,
                charge_conjugate_normalize=charge_conjugate_normalize,
                feature_spec_hash=str(spec["feature_spec_hash"]),
            )
        )
        if legacy_levels:
            level_records.extend(
                legacy_depth_samples(tree, channel=int(tree.metadata.get("channel", 0)))
            )
    payload = {
        "schema_version": SCHEMA_VERSION_V3,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": spec["feature_spec_hash"],
        "feature_spec_json": json.dumps(spec, sort_keys=True),
        "events": events,
        "legacy_levels": level_records,
        "summary_json": json.dumps(summary or {}, sort_keys=True),
    }
    ak.to_parquet(ak.Array([payload]), output_path)
    return output_path


def load_payload_v3(path: str | Path) -> dict[str, Any]:
    """Load v1/v2/v3 and normalize to the explicit v3 model contract."""

    payload = ak.to_list(ak.from_parquet(path))[0]
    return adapt_payload_to_v3(payload)


def adapt_payload_to_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    version = payload.get("schema_version")
    if version == SCHEMA_VERSION_V3:
        result = dict(payload)
        for event in result.get("events", []):
            for node in event.get("nodes", []):
                validate_pid_token(int(node["input_pid_token"]), name="input_pid_token")
                validate_pid_token(int(node["pid_target_token"]), name="pid_target_token")
        return result
    if version not in {SCHEMA_VERSION_V1, SCHEMA_VERSION_V2}:
        raise ValueError(f"Unsupported preprocessing schema: {version!r}")
    v2 = adapt_payload_to_v2(payload)
    events = [_adapt_legacy_event(event, source_version=str(version)) for event in v2["events"]]
    spec = feature_spec_v3()
    return {
        "schema_version": SCHEMA_VERSION_V3,
        "source_schema_version": version,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": spec["feature_spec_hash"],
        "feature_spec_json": json.dumps(spec, sort_keys=True),
        "events": events,
        "legacy_levels": v2.get("legacy_levels", []),
        "summary_json": v2.get("summary_json", "{}"),
    }


def _event_record_v3(
    tree: EventTree,
    *,
    charge_conjugate_normalize: bool,
    feature_spec_hash: str,
    include_v4_runtime_fields: bool = False,
) -> dict[str, Any]:
    reconstructable = event_channel_record(
        tree,
        charge_conjugate_normalize=charge_conjugate_normalize,
    )
    full = (
        tree.full_truth_channel_record_cc
        if charge_conjugate_normalize
        else tree.full_truth_channel_record
    ) or {}
    channel_fields = _dual_channel_fields(reconstructable, full)
    b1_counts = reconstructable.pop("b1_channel_counts")
    b2_counts = reconstructable.pop("b2_channel_counts")
    grouped = nodes_by_level(tree)
    return {
        "event_id": tree.event_id,
        "event_uid": str(tree.metadata.get("event_uid", tree.event_id)),
        "experiment": int(tree.metadata.get("experiment", -1)),
        "run": int(tree.metadata.get("run", -1)),
        "production": int(tree.metadata.get("production", -1)),
        "source_file": str(tree.metadata.get("source_file", "")),
        "source_category": str(tree.metadata.get("source_category", "")),
        "schema_version": SCHEMA_VERSION_V3,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": feature_spec_hash,
        "charge_conjugate_normalization": bool(charge_conjugate_normalize),
        "leaf_kinematics_mode": str(
            tree.metadata.get("leaf_kinematics_mode", "mixed_explicit_per_node")
        ),
        "root_ids": list(tree.root_ids),
        "levels": [{"level": level, "node_ids": ids} for level, ids in grouped.items()],
        "nodes": [
            _node_record_v3(
                tree,
                tree.nodes[node_id],
                include_v4_runtime_fields=include_v4_runtime_fields,
            )
            for node_id in sorted(tree.nodes)
        ],
        **reconstructable,
        **channel_fields,
        "b1_channel_count_array": _dense_pid_counts(b1_counts),
        "b2_channel_count_array": _dense_pid_counts(b2_counts),
        "b1_depth_pid_count_array": _dense_depth_pid_counts(b1_counts),
        "b2_depth_pid_count_array": _dense_depth_pid_counts(b2_counts),
        "b1_channel_summary_json": json.dumps(b1_counts, sort_keys=True),
        "b2_channel_summary_json": json.dumps(b2_counts, sort_keys=True),
        "metadata_json": json.dumps(dict(tree.metadata), sort_keys=True),
    }


def _node_record_v3(
    tree: EventTree,
    node: TreeNode,
    *,
    include_v4_runtime_fields: bool = False,
) -> dict[str, Any]:
    kind = "composite" if node.daughter_ids else (
        node.node_kind if node.node_kind in NODE_KIND_TO_ID else infer_node_kind(
            {"reco_id": node.reco_id or "", "daughter_ids": []}
        )
    )
    input_token = validate_pid_token(
        int(node.input_pid_token if node.input_pid_token is not None else 0),
        name="input_pid_token",
    )
    target_token = validate_pid_token(
        int(node.pid_target_token if node.pid_target_token is not None else node.token),
        name="pid_target_token",
    )
    common_values = {
        "px": node.p4.px,
        "py": node.p4.py,
        "pz": node.p4.pz,
        "energy": node.p4.energy,
        "mass": node.p4.mass,
        "charge": node.reco_charge if node.reco_charge is not None else node.charge,
        "reduced_pid": input_token,
        "level": node.level,
        "active": 1.0,
        "copied": float(node.copied_from is not None),
        "n_daughters": len(node.daughter_ids),
        "candidate_confidence": node.candidate_confidence,
    }
    track_values: dict[str, float | None] = dict(node.track_features) if kind == "track" else {}
    if kind == "track":
        track_values.update(
            {
                f"pid_log_likelihood_{name}": node.pid_likelihoods.get(name)
                for name in CHARGED_STABLE_NAMES
            }
        )
        track_values.update(
            {
                f"energy_hypothesis_{name}": node.track_energy_hypotheses.get(name)
                for name in CHARGED_STABLE_NAMES
            }
        )
    composite_values = _composite_values_v3(tree, node) if kind == "composite" else {}
    histogram = [0] * len(PDG_TOKENS)
    if kind == "composite":
        for daughter_id in node.daughter_ids:
            daughter_token = tree.nodes[daughter_id].pid_target_token
            histogram[validate_pid_token(int(daughter_token or 0))] += 1
    record = {
        # Explicit legacy display aliases. Model loading uses the unambiguous
        # fields below and never infers token semantics from these aliases.
        "pdg": int(node.truth_pdg if node.truth_pdg is not None else (node.raw_pdg or 0)),
        "token": target_token,
        "charge": float(node.reco_charge if node.reco_charge is not None else node.charge),
        "node_id": node.node_id,
        "reco_object_id": "" if node.reco_object_id is None else node.reco_object_id,
        "reco_id": "" if node.reco_id is None else node.reco_id,
        "source_node_id": -1 if node.source_node_id is None else node.source_node_id,
        "copied_from": -1 if node.copied_from is None else node.copied_from,
        "recursive_leaf_source_ids": list(node.recursive_leaf_source_ids),
        "mc_id": -1 if node.mc_id is None else node.mc_id,
        "raw_pdg": int(node.raw_pdg or 0),
        "reduced_pid_token": input_token,
        "input_pid_token": input_token,
        "pid_target_token": target_token,
        "truth_pdg": node.truth_pdg,
        "truth_pid_token": node.truth_pid_token,
        "node_kind": kind,
        "node_kind_id": NODE_KIND_TO_ID[kind],
        "reco_charge": node.reco_charge,
        "truth_charge": node.truth_charge,
        "px": node.p4.px,
        "py": node.p4.py,
        "pz": node.p4.pz,
        "energy": node.p4.energy,
        "reconstructed_energy": node.p4.energy,
        "mass": node.p4.mass,
        "energy_source": node.energy_source,
        "leaf_kinematics_mode": node.leaf_kinematics_mode,
        "level": node.level,
        "parent_id": -1 if node.parent_id is None else node.parent_id,
        "daughter_ids": list(node.daughter_ids),
        "active": True,
        "copied": node.copied_from is not None,
        "flags": sorted(node.flags),
        "candidate_confidence": node.candidate_confidence,
        "common_features": _values_record(V3_COMMON_FEATURE_NAMES, common_values),
        "common_availability": _mask_record(V3_COMMON_FEATURE_NAMES, common_values),
        "track_features": _values_record(V3_TRACK_FEATURE_NAMES, track_values),
        "track_availability": _track_mask(node, kind, track_values),
        "cluster_features": _values_record(
            V3_CLUSTER_FEATURE_NAMES,
            node.cluster_features if kind == "ecl_cluster" else {},
        ),
        "cluster_availability": _mask_record(
            V3_CLUSTER_FEATURE_NAMES,
            node.cluster_features if kind == "ecl_cluster" else {},
        ),
        "composite_features": _values_record(V3_COMPOSITE_FEATURE_NAMES, composite_values),
        "composite_availability": _mask_record(V3_COMPOSITE_FEATURE_NAMES, composite_values),
        "daughter_pid_histogram": histogram,
        "daughter_pid_histogram_available": kind == "composite",
        "pid_likelihoods": {
            name: float(node.pid_likelihoods.get(name, 0.0)) for name in CHARGED_STABLE_NAMES
        },
        "pid_likelihood_availability": {
            name: bool(node.pid_likelihood_availability.get(name, False))
            for name in CHARGED_STABLE_NAMES
        },
        "mass_hypothesis_energies": {
            name: float(node.track_energy_hypotheses.get(name, 0.0))
            for name in CHARGED_STABLE_NAMES
        },
        "mass_hypothesis_availability": {
            name: bool(node.track_energy_availability.get(name, False))
            for name in CHARGED_STABLE_NAMES
        },
        "full_truth_daughter_count": node.full_truth_daughter_count,
        "retained_daughter_count": node.retained_daughter_count,
        "reconstructed_daughter_count": node.reconstructed_daughter_count,
        "complete_truth_decay": node.complete_truth_decay,
        "complete_reconstructable_decay": node.complete_reconstructable_decay,
        "partial_missing_daughters": node.partial_missing_daughters,
        "contracted_intermediate": node.contracted_intermediate,
        "valid_reconstruction_target": node.valid_reconstruction_target,
        "mc_px": None if node.mc_p4 is None else node.mc_p4.px,
        "mc_py": None if node.mc_p4 is None else node.mc_p4.py,
        "mc_pz": None if node.mc_p4 is None else node.mc_p4.pz,
        "mc_energy": None if node.mc_p4 is None else node.mc_p4.energy,
    }
    if include_v4_runtime_fields:
        record.update(
            {
                "pid_likelihood_status": {
                    name: str(
                        node.pid_likelihood_status.get(name, "not_applicable")
                    )
                    for name in CHARGED_STABLE_NAMES
                },
                "pid_detector_availability": dict(
                    node.pid_detector_availability
                ),
                "track_fit_hypothesis": node.track_fit_hypothesis,
                "track_fit_selection_method": node.track_fit_selection_method,
                "track_fit_available": bool(node.track_fit_available),
                "track_fit_fallback_reason": node.track_fit_fallback_reason,
                "klm_features": dict(node.klm_features),
                "klm_availability": {
                    name: True for name in node.klm_features
                },
            }
        )
    return record


def _track_mask(
    node: TreeNode,
    kind: str,
    values: Mapping[str, Any],
) -> dict[str, bool]:
    mask = _mask_record(V3_TRACK_FEATURE_NAMES, values)
    if kind != "track":
        return mask
    for name in CHARGED_STABLE_NAMES:
        mask[f"pid_log_likelihood_{name}"] = bool(
            node.pid_likelihood_availability.get(name, False)
            and f"pid_log_likelihood_{name}" in values
            and values[f"pid_log_likelihood_{name}"] is not None
        )
        mask[f"energy_hypothesis_{name}"] = bool(
            node.track_energy_availability.get(name, False)
            and f"energy_hypothesis_{name}" in values
            and values[f"energy_hypothesis_{name}"] is not None
        )
    return mask


def _composite_values_v3(tree: EventTree, node: TreeNode) -> dict[str, float | None]:
    daughters = [tree.nodes[node_id] for node_id in node.daughter_ids]
    return {
        "daughter_sum_px": sum(daughter.p4.px for daughter in daughters),
        "daughter_sum_py": sum(daughter.p4.py for daughter in daughters),
        "daughter_sum_pz": sum(daughter.p4.pz for daughter in daughters),
        "daughter_sum_energy": sum(daughter.p4.energy for daughter in daughters),
        "summed_charge": sum(float(daughter.reco_charge or 0.0) for daughter in daughters),
        "daughter_count": len(daughters),
        "pointer_confidence_mean": None,
        "pointer_confidence_min": None,
        "copied_daughter_fraction": (
            sum(daughter.copied_from is not None for daughter in daughters) / len(daughters)
            if daughters else 0.0
        ),
        "full_truth_daughter_count": node.full_truth_daughter_count,
        "retained_daughter_count": node.retained_daughter_count,
        "reconstructed_daughter_count": node.reconstructed_daughter_count,
        "partial_missing_daughters": float(node.partial_missing_daughters),
    }


def _dual_channel_fields(
    reconstructable: Mapping[str, Any],
    full: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "full_truth_channel_available": bool(full),
        "reconstructable_channel_available": True,
    }
    for side in ("b1", "b2"):
        output[f"{side}_full_truth_channel_signature"] = full.get(f"{side}_channel_signature")
        output[f"{side}_full_truth_channel_id"] = int(full.get(f"{side}_channel_id", 0))
        output[f"{side}_reconstructable_channel_signature"] = reconstructable.get(
            f"{side}_channel_signature"
        )
        output[f"{side}_reconstructable_channel_id"] = int(
            reconstructable.get(f"{side}_channel_id", 0)
        )
    output["y4s_full_truth_channel_signature"] = full.get("y4s_channel_signature")
    output["y4s_full_truth_channel_id"] = int(full.get("y4s_channel_id", 0))
    output["y4s_reconstructable_channel_signature"] = reconstructable.get(
        "y4s_channel_signature"
    )
    output["y4s_reconstructable_channel_id"] = int(
        reconstructable.get("y4s_channel_id", 0)
    )
    return output


def _adapt_legacy_event(event: Mapping[str, Any], *, source_version: str) -> dict[str, Any]:
    adapted = dict(event)
    nodes: list[dict[str, Any]] = []
    for source in event.get("nodes", []):
        node = dict(source)
        token = validate_pid_token(int(node.get("token", 0)), name="legacy reduced token")
        kind = str(node.get("node_kind") or infer_node_kind(node))
        old_track = dict(node.get("track_features", {}))
        old_track_mask = dict(node.get("track_availability", {}))
        node.update(
            {
                "raw_pdg": int(node.get("pdg", 0)),
                "reduced_pid_token": token,
                "input_pid_token": token,
                "pid_target_token": token,
                "truth_pdg": None,
                "truth_pid_token": None,
                "node_kind": kind,
                "node_kind_id": NODE_KIND_TO_ID[kind],
                "reco_charge": float(node.get("charge", 0.0)),
                "truth_charge": None,
                "reconstructed_energy": float(node.get("energy", 0.0)),
                "energy_source": "legacy_conflated",
                "leaf_kinematics_mode": "legacy_conflated",
                "reco_object_id": str(node.get("reco_id", "")),
                "recursive_leaf_source_ids": (
                    [str(node.get("reco_id"))]
                    if not node.get("daughter_ids") and node.get("reco_id")
                    else []
                ),
                "track_features": {
                    name: float(old_track.get(name, 0.0)) for name in V3_TRACK_FEATURE_NAMES
                },
                "track_availability": {
                    name: bool(old_track_mask.get(name, False)) for name in V3_TRACK_FEATURE_NAMES
                },
                "composite_features": {
                    name: float(node.get("composite_features", {}).get(name, 0.0))
                    for name in V3_COMPOSITE_FEATURE_NAMES
                },
                "composite_availability": {
                    name: bool(node.get("composite_availability", {}).get(name, False))
                    for name in V3_COMPOSITE_FEATURE_NAMES
                },
                "pid_likelihoods": {name: 0.0 for name in CHARGED_STABLE_NAMES},
                "pid_likelihood_availability": {name: False for name in CHARGED_STABLE_NAMES},
                "mass_hypothesis_energies": {name: 0.0 for name in CHARGED_STABLE_NAMES},
                "mass_hypothesis_availability": {name: False for name in CHARGED_STABLE_NAMES},
                "full_truth_daughter_count": 0,
                "retained_daughter_count": len(node.get("daughter_ids", [])),
                "reconstructed_daughter_count": len(node.get("daughter_ids", [])),
                "complete_truth_decay": False,
                "complete_reconstructable_decay": bool(node.get("daughter_ids")),
                "partial_missing_daughters": False,
                "contracted_intermediate": False,
                "valid_reconstruction_target": len(node.get("daughter_ids", [])) >= 2,
                "legacy_leaf_contract": True,
            }
        )
        nodes.append(node)
    adapted["nodes"] = nodes
    adapted["schema_version"] = SCHEMA_VERSION_V3
    adapted["source_schema_version"] = source_version
    adapted["pid_vocabulary_version"] = PID_VOCABULARY_VERSION
    adapted.setdefault("b1_full_truth_channel_signature", None)
    adapted.setdefault("b2_full_truth_channel_signature", None)
    adapted.setdefault("b1_full_truth_channel_id", 0)
    adapted.setdefault("b2_full_truth_channel_id", 0)
    adapted["b1_reconstructable_channel_signature"] = adapted.get("b1_channel_signature")
    adapted["b2_reconstructable_channel_signature"] = adapted.get("b2_channel_signature")
    adapted["b1_reconstructable_channel_id"] = int(adapted.get("b1_channel_id", 0))
    adapted["b2_reconstructable_channel_id"] = int(adapted.get("b2_channel_id", 0))
    adapted["y4s_full_truth_channel_id"] = 0
    adapted["y4s_reconstructable_channel_id"] = int(adapted.get("y4s_channel_id", 0))
    return adapted


def _values_record(names: tuple[str, ...], values: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(values[name])
        if name in values and values[name] is not None and math.isfinite(float(values[name]))
        else 0.0
        for name in names
    }


def _mask_record(names: tuple[str, ...], values: Mapping[str, Any]) -> dict[str, bool]:
    return {
        name: bool(
            name in values
            and values[name] is not None
            and math.isfinite(float(values[name]))
        )
        for name in names
    }


def _dense_pid_counts(summary: Mapping[str, Any]) -> list[int]:
    output = [0] * len(PDG_TOKENS)
    for record in summary.get("pid_counts", []):
        output[validate_pid_token(int(record["token"]))] = int(record["count"])
    return output


def _dense_depth_pid_counts(summary: Mapping[str, Any]) -> list[list[int]]:
    maximum = int(summary.get("max_relative_depth", 0))
    output = [[0] * len(PDG_TOKENS) for _ in range(maximum + 1)]
    for record in summary.get("depth_pid_counts", []):
        output[int(record["depth"])][validate_pid_token(int(record["token"]))] = int(
            record["count"]
        )
    return output


__all__ = [
    "SCHEMA_VERSION_V3",
    "V3_CLUSTER_FEATURE_NAMES",
    "V3_COMMON_FEATURE_NAMES",
    "V3_COMPOSITE_FEATURE_NAMES",
    "V3_TRACK_FEATURE_NAMES",
    "adapt_payload_to_v3",
    "export_trees_v3",
    "feature_spec_v3",
    "load_payload_v3",
]
