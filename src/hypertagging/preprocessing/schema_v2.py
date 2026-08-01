"""Versioned heterogeneous direct-mDST schema and v1 compatibility adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import awkward as ak

from hypertagging.preprocessing.channels import event_channel_record
from hypertagging.preprocessing.levelize_tree import assign_levels, legacy_depth_samples, nodes_by_level
from hypertagging.preprocessing.mdst_tree_builder import EventTree, FourVector, TreeNode
from hypertagging.preprocessing.pid_filter import PDG_TOKENS


SCHEMA_VERSION_V1 = "direct-mdst-tree-v1"
SCHEMA_VERSION_V2 = "direct-mdst-tree-v2"

NODE_KINDS = (
    "unknown",
    "track",
    "ecl_cluster",
    "composite",
    "other",
    "klm_cluster",
)
NODE_KIND_TO_ID = {name: index for index, name in enumerate(NODE_KINDS)}

COMMON_FEATURE_NAMES = (
    "px",
    "py",
    "pz",
    "energy",
    "mass",
    "charge",
    "reduced_pid",
    "level",
    "active",
    "copied",
    "n_daughters",
    "candidate_confidence",
)
TRACK_FEATURE_NAMES = (
    "fit_p_value",
    "d0",
    "z0",
    "phi0",
    "omega",
    "tan_lambda",
)
CLUSTER_FEATURE_NAMES = (
    "cluster_energy",
    "theta",
    "phi",
    "time",
    "e9_over_e21",
    "n_crystals",
    "min_track_distance",
    "photon_hypothesis",
    "track_matched",
)
COMPOSITE_FEATURE_NAMES = (
    "daughter_sum_px",
    "daughter_sum_py",
    "daughter_sum_pz",
    "daughter_sum_energy",
    "summed_charge",
    "daughter_count",
    "pointer_confidence_mean",
    "pointer_confidence_min",
    "copied_daughter_fraction",
)


def feature_spec() -> dict[str, Any]:
    return {
        "common": list(COMMON_FEATURE_NAMES),
        "track": list(TRACK_FEATURE_NAMES),
        "ecl_cluster": list(CLUSTER_FEATURE_NAMES),
        "composite": list(COMPOSITE_FEATURE_NAMES),
        "daughter_pid_histogram_tokens": list(PDG_TOKENS),
        "node_kinds": list(NODE_KINDS),
        "missing_value_policy": (
            "Numeric storage uses 0.0 only as a tensor-safe placeholder; each value has an "
            "explicit availability mask and zero never implies availability."
        ),
    }


def export_trees_v2(
    trees: Iterable[EventTree],
    output: str | Path,
    *,
    summary: dict[str, object] | None = None,
    legacy_levels: bool = True,
    charge_conjugate_normalize: bool = False,
) -> Path:
    """Write v2 while preserving the legacy level view."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_records: list[dict[str, Any]] = []
    level_records: list[dict[str, object]] = []
    for tree in trees:
        assign_levels(tree)
        event_records.append(
            _event_record_v2(
                tree,
                charge_conjugate_normalize=charge_conjugate_normalize,
            )
        )
        if legacy_levels:
            level_records.extend(
                legacy_depth_samples(tree, channel=int(tree.metadata.get("channel", 0)))
            )
    payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "feature_spec_json": json.dumps(feature_spec(), sort_keys=True),
        "events": event_records,
        "legacy_levels": level_records,
        "summary_json": json.dumps(summary or {}, sort_keys=True),
    }
    ak.to_parquet(ak.Array([payload]), output_path)
    return output_path


def load_payload_v2(path: str | Path) -> dict[str, Any]:
    """Load v1 or v2 parquet and return one normalized v2 Python payload."""

    payload = ak.to_list(ak.from_parquet(path))[0]
    return adapt_payload_to_v2(payload)


def adapt_payload_to_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Conservatively adapt a v1 payload without fabricating detector fields."""

    version = payload.get("schema_version")
    if version == SCHEMA_VERSION_V2:
        return dict(payload)
    if version != SCHEMA_VERSION_V1:
        raise ValueError(f"Unsupported preprocessing schema: {version!r}")
    events = [_adapt_v1_event(event) for event in payload.get("events", [])]
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "source_schema_version": SCHEMA_VERSION_V1,
        "feature_spec_json": json.dumps(feature_spec(), sort_keys=True),
        "events": events,
        "legacy_levels": payload.get("legacy_levels", []),
        "summary_json": payload.get("summary_json", "{}"),
    }


def infer_node_kind(node: Mapping[str, Any]) -> str:
    """Infer only from explicit topology or reconstructed-object provenance."""

    if node.get("daughter_ids"):
        return "composite"
    reco_id = str(node.get("reco_id", ""))
    if reco_id.startswith("Track:"):
        return "track"
    if reco_id.startswith("ECLCluster:"):
        return "ecl_cluster"
    if reco_id:
        return "other"
    return "unknown"


def _event_record_v2(
    tree: EventTree,
    *,
    charge_conjugate_normalize: bool,
) -> dict[str, Any]:
    grouped = nodes_by_level(tree)
    channels = event_channel_record(
        tree,
        charge_conjugate_normalize=charge_conjugate_normalize,
    )
    b1_counts = channels.pop("b1_channel_counts")
    b2_counts = channels.pop("b2_channel_counts")
    return {
        "event_id": tree.event_id,
        "event_uid": str(tree.metadata.get("event_uid", tree.event_id)),
        "experiment": int(tree.metadata.get("experiment", -1)),
        "run": int(tree.metadata.get("run", -1)),
        "production": int(tree.metadata.get("production", -1)),
        "source_file": str(tree.metadata.get("source_file", "")),
        "source_category": str(tree.metadata.get("source_category", "")),
        "root_ids": list(tree.root_ids),
        "levels": [{"level": level, "node_ids": ids} for level, ids in grouped.items()],
        "nodes": [_node_record_v2(tree, tree.nodes[node_id]) for node_id in sorted(tree.nodes)],
        **channels,
        "b1_channel_count_array": _dense_pid_counts(b1_counts),
        "b2_channel_count_array": _dense_pid_counts(b2_counts),
        "b1_depth_pid_count_array": _dense_depth_pid_counts(b1_counts),
        "b2_depth_pid_count_array": _dense_depth_pid_counts(b2_counts),
        "b1_channel_summary_json": json.dumps(b1_counts, sort_keys=True),
        "b2_channel_summary_json": json.dumps(b2_counts, sort_keys=True),
        "metadata_json": json.dumps(dict(tree.metadata), sort_keys=True),
    }


def _node_record_v2(tree: EventTree, node: TreeNode) -> dict[str, Any]:
    kind = _validated_node_kind(node)
    common_values = {
        "px": node.p4.px,
        "py": node.p4.py,
        "pz": node.p4.pz,
        "energy": node.p4.energy,
        "mass": node.p4.mass,
        "charge": node.charge,
        "reduced_pid": node.token,
        "level": node.level,
        "active": 1.0,
        "copied": float(node.copied_from is not None),
        "n_daughters": len(node.daughter_ids),
        "candidate_confidence": node.candidate_confidence,
    }
    track = node.track_features if kind == "track" else {}
    cluster = node.cluster_features if kind == "ecl_cluster" else {}
    composite = _composite_values(tree, node) if kind == "composite" else {}
    daughter_pid_histogram = [0] * len(PDG_TOKENS)
    if kind == "composite":
        for daughter_id in node.daughter_ids:
            daughter_pid_histogram[tree.nodes[daughter_id].token] += 1
    return {
        **_legacy_node_fields(node),
        "node_kind": kind,
        "node_kind_id": NODE_KIND_TO_ID[kind],
        "active": True,
        "copied": node.copied_from is not None,
        "candidate_confidence": node.candidate_confidence,
        "common_features": _values_record(COMMON_FEATURE_NAMES, common_values),
        "common_availability": _mask_record(COMMON_FEATURE_NAMES, common_values),
        "track_features": _values_record(TRACK_FEATURE_NAMES, track),
        "track_availability": _mask_record(TRACK_FEATURE_NAMES, track),
        "cluster_features": _values_record(CLUSTER_FEATURE_NAMES, cluster),
        "cluster_availability": _mask_record(CLUSTER_FEATURE_NAMES, cluster),
        "composite_features": _values_record(COMPOSITE_FEATURE_NAMES, composite),
        "composite_availability": _mask_record(COMPOSITE_FEATURE_NAMES, composite),
        "daughter_pid_histogram": daughter_pid_histogram,
        "daughter_pid_histogram_available": kind == "composite",
    }


def _legacy_node_fields(node: TreeNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "pdg": node.pdg,
        "token": node.token,
        "charge": node.charge,
        "mass": node.p4.mass,
        "energy": node.p4.energy,
        "px": node.p4.px,
        "py": node.p4.py,
        "pz": node.p4.pz,
        "level": node.level,
        "parent_id": -1 if node.parent_id is None else node.parent_id,
        "daughter_ids": list(node.daughter_ids),
        "reco_id": "" if node.reco_id is None else node.reco_id,
        "mc_id": -1 if node.mc_id is None else node.mc_id,
        "copied_from": -1 if node.copied_from is None else node.copied_from,
        "source_node_id": -1 if node.source_node_id is None else node.source_node_id,
        "flags": sorted(node.flags),
        "prodTime": node.prod_time,
        "x": node.vertex[0],
        "y": node.vertex[1],
        "z": node.vertex[2],
        "nDaughters": node.n_daughters,
        "mc_px": None if node.mc_p4 is None else node.mc_p4.px,
        "mc_py": None if node.mc_p4 is None else node.mc_p4.py,
        "mc_pz": None if node.mc_p4 is None else node.mc_p4.pz,
        "mc_energy": None if node.mc_p4 is None else node.mc_p4.energy,
    }


def _adapt_v1_event(event: Mapping[str, Any]) -> dict[str, Any]:
    nodes = []
    for source in event.get("nodes", []):
        node = dict(source)
        kind = infer_node_kind(node)
        common_values = {
            "px": node["px"],
            "py": node["py"],
            "pz": node["pz"],
            "energy": node["energy"],
            "mass": node["mass"],
            "charge": node["charge"],
            "reduced_pid": node["token"],
            "level": node["level"],
            "active": 1.0,
            "copied": float(int(node.get("copied_from", -1)) >= 0),
            "n_daughters": node.get("nDaughters", len(node.get("daughter_ids", []))),
            "candidate_confidence": None,
        }
        # V1 has no detector-specific values. Even values derivable from the
        # common p4 remain unavailable in detector blocks.
        node.update(
            {
                "node_kind": kind,
                "node_kind_id": NODE_KIND_TO_ID[kind],
                "active": True,
                "copied": int(node.get("copied_from", -1)) >= 0,
                "candidate_confidence": None,
                "common_features": _values_record(COMMON_FEATURE_NAMES, common_values),
                "common_availability": _mask_record(COMMON_FEATURE_NAMES, common_values),
                "track_features": _values_record(TRACK_FEATURE_NAMES, {}),
                "track_availability": _mask_record(TRACK_FEATURE_NAMES, {}),
                "cluster_features": _values_record(CLUSTER_FEATURE_NAMES, {}),
                "cluster_availability": _mask_record(CLUSTER_FEATURE_NAMES, {}),
                "composite_features": _values_record(COMPOSITE_FEATURE_NAMES, {}),
                "composite_availability": _mask_record(COMPOSITE_FEATURE_NAMES, {}),
                "daughter_pid_histogram": [0] * len(PDG_TOKENS),
                "daughter_pid_histogram_available": False,
            }
        )
        nodes.append(node)
    by_id = {int(node["node_id"]): node for node in nodes}
    for node in nodes:
        if node["node_kind"] != "composite":
            continue
        daughters = [
            by_id[int(daughter_id)]
            for daughter_id in node.get("daughter_ids", [])
            if int(daughter_id) in by_id
        ]
        values = {
            "daughter_sum_px": sum(float(daughter["px"]) for daughter in daughters),
            "daughter_sum_py": sum(float(daughter["py"]) for daughter in daughters),
            "daughter_sum_pz": sum(float(daughter["pz"]) for daughter in daughters),
            "daughter_sum_energy": sum(float(daughter["energy"]) for daughter in daughters),
            "summed_charge": sum(float(daughter["charge"]) for daughter in daughters),
            "daughter_count": len(daughters),
            "pointer_confidence_mean": None,
            "pointer_confidence_min": None,
            "copied_daughter_fraction": (
                sum(int(daughter.get("copied_from", -1)) >= 0 for daughter in daughters)
                / len(daughters)
                if daughters
                else 0.0
            ),
        }
        node["composite_features"] = _values_record(COMPOSITE_FEATURE_NAMES, values)
        node["composite_availability"] = _mask_record(COMPOSITE_FEATURE_NAMES, values)
        histogram = [0] * len(PDG_TOKENS)
        for daughter in daughters:
            histogram[int(daughter["token"])] += 1
        node["daughter_pid_histogram"] = histogram
        node["daughter_pid_histogram_available"] = True
    adapted = dict(event)
    adapted["nodes"] = nodes
    adapted.setdefault("source_file", "")
    adapted.setdefault("source_category", "")
    adapted.update(_channel_fields_from_v1_event(event))
    return adapted


def _validated_node_kind(node: TreeNode) -> str:
    if node.daughter_ids:
        return "composite"
    if node.node_kind in NODE_KIND_TO_ID and node.node_kind != "composite":
        return node.node_kind
    if node.reco_id:
        return infer_node_kind({"reco_id": node.reco_id, "daughter_ids": []})
    return "unknown"


def _composite_values(tree: EventTree, node: TreeNode) -> dict[str, float | None]:
    daughters = [tree.nodes[node_id] for node_id in node.daughter_ids]
    copied_fraction = (
        sum(daughter.copied_from is not None for daughter in daughters) / len(daughters)
        if daughters
        else 0.0
    )
    return {
        "daughter_sum_px": sum(daughter.p4.px for daughter in daughters),
        "daughter_sum_py": sum(daughter.p4.py for daughter in daughters),
        "daughter_sum_pz": sum(daughter.p4.pz for daughter in daughters),
        "daughter_sum_energy": sum(daughter.p4.energy for daughter in daughters),
        "summed_charge": sum(daughter.charge for daughter in daughters),
        "daughter_count": len(daughters),
        "pointer_confidence_mean": None,
        "pointer_confidence_min": None,
        "copied_daughter_fraction": copied_fraction,
    }


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
        output[int(record["token"])] = int(record["count"])
    return output


def _dense_depth_pid_counts(summary: Mapping[str, Any]) -> list[list[int]]:
    max_depth = int(summary.get("max_relative_depth", 0))
    output = [[0] * len(PDG_TOKENS) for _ in range(max_depth + 1)]
    for record in summary.get("depth_pid_counts", []):
        output[int(record["depth"])][int(record["token"])] = int(record["count"])
    return output


def _empty_channel_fields() -> dict[str, Any]:
    return {
        "charge_conjugate_normalized": False,
        "b1_root_id": -1,
        "b2_root_id": -1,
        "b1_channel_signature": None,
        "b2_channel_signature": None,
        "b1_channel_id": 0,
        "b2_channel_id": 0,
        "b1_channel_count_array": [0] * len(PDG_TOKENS),
        "b2_channel_count_array": [0] * len(PDG_TOKENS),
        "b1_depth_pid_count_array": [[0] * len(PDG_TOKENS)],
        "b2_depth_pid_count_array": [[0] * len(PDG_TOKENS)],
        "b1_channel_summary_json": "{}",
        "b2_channel_summary_json": "{}",
        "exact_channel_equal": False,
        "structured_channel_similarity": 0.0,
        "same_event": True,
        "y4s_channel_signature": None,
        "y4s_channel_id": 0,
    }


def _channel_fields_from_v1_event(event: Mapping[str, Any]) -> dict[str, Any]:
    tree = EventTree(event_id=int(event["event_id"]))
    for record in event.get("nodes", []):
        tree.add_node(
            TreeNode(
                node_id=int(record["node_id"]),
                pdg=int(record["pdg"]),
                charge=float(record["charge"]),
                p4=FourVector(
                    float(record["px"]),
                    float(record["py"]),
                    float(record["pz"]),
                    float(record["energy"]),
                ),
                parent_id=(
                    None if int(record.get("parent_id", -1)) < 0 else int(record["parent_id"])
                ),
                daughter_ids=[int(value) for value in record.get("daughter_ids", [])],
                level=int(record.get("level", -1)),
                reco_id=str(record.get("reco_id", "")) or None,
                copied_from=(
                    None
                    if int(record.get("copied_from", -1)) < 0
                    else int(record["copied_from"])
                ),
                source_node_id=(
                    None
                    if int(record.get("source_node_id", -1)) < 0
                    else int(record["source_node_id"])
                ),
            )
        )
    tree.root_ids = [int(value) for value in event.get("root_ids", [])]
    channels = event_channel_record(tree)
    b1_counts = channels.pop("b1_channel_counts")
    b2_counts = channels.pop("b2_channel_counts")
    return {
        **channels,
        "b1_channel_count_array": _dense_pid_counts(b1_counts),
        "b2_channel_count_array": _dense_pid_counts(b2_counts),
        "b1_depth_pid_count_array": _dense_depth_pid_counts(b1_counts),
        "b2_depth_pid_count_array": _dense_depth_pid_counts(b2_counts),
        "b1_channel_summary_json": json.dumps(b1_counts, sort_keys=True),
        "b2_channel_summary_json": json.dumps(b2_counts, sort_keys=True),
    }


__all__ = [
    "CLUSTER_FEATURE_NAMES",
    "COMMON_FEATURE_NAMES",
    "COMPOSITE_FEATURE_NAMES",
    "NODE_KINDS",
    "NODE_KIND_TO_ID",
    "SCHEMA_VERSION_V1",
    "SCHEMA_VERSION_V2",
    "TRACK_FEATURE_NAMES",
    "adapt_payload_to_v2",
    "export_trees_v2",
    "feature_spec",
    "infer_node_kind",
    "load_payload_v2",
]
