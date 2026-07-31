"""Truth-clean, event-row, streamable direct-mDST schema-v4.

V4 deliberately does not reinterpret v3.  Native v4 records separate the
daughter PID histogram available to the model from truth-only diagnostics.
Older schemas are adapted with an explicit ``legacy_conflated`` marker.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from hypertagging.preprocessing.levelize_tree import assign_levels
from hypertagging.preprocessing.mdst_tree_builder import EventTree
from hypertagging.preprocessing.pid_filter import (
    PDG_TOKENS,
    PID_VOCABULARY_VERSION,
    validate_pid_token,
)
from hypertagging.preprocessing.schema_v3 import (
    SCHEMA_VERSION_V3,
    _event_record_v3,
    feature_spec_v3,
    load_payload_v3,
)


SCHEMA_VERSION_V4 = "direct-mdst-tree-v4"
FEATURE_SPEC_REVISION_V4 = "v4-runtime-normalized-categorical-separated-r2"

# These positions retain their stored compatibility values, but are never
# interpreted as continuous geometry. Dedicated embeddings/flags own them.
CATEGORICAL_COMMON_FEATURE_NAMES = (
    "reduced_pid",
    "level",
    "active",
    "copied",
)
CONTINUOUS_COMMON_FEATURE_NAMES = tuple(
    name
    for name in feature_spec_v3()["common"]
    if name not in CATEGORICAL_COMMON_FEATURE_NAMES
)
CONTINUOUS_COMMON_INDICES = tuple(
    feature_spec_v3()["common"].index(name)
    for name in CONTINUOUS_COMMON_FEATURE_NAMES
)
DYNAMIC_COMMON_FEATURE_NAMES = (
    "px",
    "py",
    "pz",
    "energy",
    "mass",
    "charge",
    "n_daughters",
)
DYNAMIC_COMMON_INDICES = tuple(
    feature_spec_v3()["common"].index(name)
    for name in DYNAMIC_COMMON_FEATURE_NAMES
)
STATIC_COMMON_FEATURE_NAMES = tuple(
    name
    for name in CONTINUOUS_COMMON_FEATURE_NAMES
    if name not in DYNAMIC_COMMON_FEATURE_NAMES
)
DYNAMIC_COMPOSITE_INDICES = tuple(range(len(feature_spec_v3()["composite"])))

LEAF_KINEMATICS_MODES: tuple[str, ...] = (
    "raw_track_predicted_pid",
    "fixed_hypothesis_candidate",
    "ecl_cluster",
    "composite",
    "truth_topology_only",
    "legacy_conflated",
)
LEAF_MODE_TO_ID = {name: index for index, name in enumerate(LEAF_KINEMATICS_MODES)}
LEAF_MODE_FROM_ID = dict(enumerate(LEAF_KINEMATICS_MODES))


def feature_spec_v4() -> dict[str, Any]:
    spec = dict(feature_spec_v3())
    spec.update(
        {
            "schema_version": SCHEMA_VERSION_V4,
            "feature_spec_revision": FEATURE_SPEC_REVISION_V4,
            "event_layout": "one-event-per-parquet-row",
            "leaf_kinematics_modes": list(LEAF_KINEMATICS_MODES),
            "daughter_pid_histograms": {
                "daughter_input_pid_histogram": (
                    "count-valued sum of data-available/runtime daughter PID distributions"
                ),
                "daughter_truth_pid_histogram": "truth-only target/diagnostic",
            },
            "legacy_daughter_pid_histogram": "not present in native v4",
            "continuous_common_features": list(CONTINUOUS_COMMON_FEATURE_NAMES),
            "categorical_common_features": {
                "reduced_pid": "PID embedding/current soft PID embedding",
                "level": "level embedding",
                "active": "binary active embedding",
                "copied": "binary copied embedding",
            },
            "runtime_dynamic_common_features": list(DYNAMIC_COMMON_FEATURE_NAMES),
            "runtime_static_common_features": list(STATIC_COMMON_FEATURE_NAMES),
            "runtime_dynamic_composite_indices": list(DYNAMIC_COMPOSITE_INDICES),
        }
    )
    spec.pop("feature_spec_hash", None)
    spec["feature_spec_hash"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return spec


class ParquetEventWriter:
    """Bounded, atomic event-row parquet writer with a metadata sidecar."""

    _schema = pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("event_uid", pa.string(), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_category", pa.string(), nullable=False),
            pa.field("event_json", pa.large_string(), nullable=False),
        ]
    )

    def __init__(
        self,
        output: str | Path,
        *,
        event_buffer_size: int = 128,
        row_group_size: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if event_buffer_size <= 0:
            raise ValueError("event_buffer_size must be positive")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.output.with_name(f".{self.output.name}.partial")
        self.sidecar = self.output.with_suffix(self.output.suffix + ".metadata.json")
        self.partial_sidecar = self.sidecar.with_name(f".{self.sidecar.name}.partial")
        self.completion_marker = self.output.with_suffix(self.output.suffix + ".complete")
        self.event_buffer_size = int(event_buffer_size)
        self.row_group_size = int(row_group_size or event_buffer_size)
        self.spec = feature_spec_v4()
        self.metadata = {
            "schema_version": SCHEMA_VERSION_V4,
            "pid_vocabulary_version": PID_VOCABULARY_VERSION,
            "feature_spec_hash": self.spec["feature_spec_hash"],
            "git_commit": _git_commit(),
            "leaf_kinematics_mode": "mixed_explicit_per_node",
            "charge_conjugate_normalization": False,
            "source_file": "",
            "entry_start": None,
            "entry_stop_exclusive": None,
            "category": "",
            "preprocessing_configuration": {},
            **dict(metadata or {}),
        }
        arrow_metadata = {
            key.encode("utf-8"): json.dumps(value, sort_keys=True).encode("utf-8")
            for key, value in self.metadata.items()
        }
        schema = self._schema.with_metadata(arrow_metadata)
        self._writer = pq.ParquetWriter(self.partial, schema, compression="zstd")
        self._buffer: list[dict[str, Any]] = []
        self._closed = False
        self._event_count = 0
        self._capacity = Counter()
        self._pid = Counter()
        self._completeness = Counter()
        self._feature_statistics = {
            block: _empty_feature_statistics(len(self.spec[block]))
            for block in ("common", "track", "ecl_cluster", "composite")
        }

    @property
    def buffered_events(self) -> int:
        return len(self._buffer)

    def write_event(self, event: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a finalized ParquetEventWriter")
        record = dict(event)
        if record.get("schema_version") != SCHEMA_VERSION_V4:
            raise ValueError("ParquetEventWriter accepts native schema-v4 events only")
        self._buffer.append(
            {
                "event_id": int(record["event_id"]),
                "event_uid": str(record["event_uid"]),
                "source_file": str(record.get("source_file", "")),
                "source_category": str(record.get("source_category", "")),
                "event_json": json.dumps(
                    record, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            }
        )
        self._event_count += 1
        self._update_statistics(record)
        if len(self._buffer) >= self.event_buffer_size:
            self.flush()

    def write_tree(
        self,
        tree: EventTree,
        *,
        charge_conjugate_normalize: bool = False,
    ) -> None:
        """Convert and append one tree without retaining previous events."""

        assign_levels(tree)
        event = _event_record_v3(
            tree,
            charge_conjugate_normalize=charge_conjugate_normalize,
            feature_spec_hash=str(self.spec["feature_spec_hash"]),
        )
        self.write_event(_native_v4_event(event))

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._writer.schema)
        self._writer.write_table(table, row_group_size=self.row_group_size)
        self._buffer.clear()

    def close(self) -> Path:
        if self._closed:
            return self.output
        try:
            self.flush()
            self._writer.close()
            final_metadata = {
                **self.metadata,
                "event_count": self._event_count,
                "aggregate_capacity_statistics": dict(sorted(self._capacity.items())),
                "aggregate_pid_statistics": dict(sorted(self._pid.items())),
                "aggregate_completeness_statistics": dict(
                    sorted(self._completeness.items())
                ),
                "aggregate_feature_welford": self._feature_statistics,
                "requested_collection_mode": self.metadata.get(
                    "leaf_kinematics_mode", "mixed_explicit_per_node"
                ),
                "actual_leaf_mode_distribution": {
                    key.removeprefix("leaf_mode_"): value
                    for key, value in sorted(self._capacity.items())
                    if key.startswith("leaf_mode_")
                },
                "actual_collection_mode": _actual_collection_mode(
                    {
                        key.removeprefix("leaf_mode_"): value
                        for key, value in self._capacity.items()
                        if key.startswith("leaf_mode_")
                    }
                ),
            }
            self.partial_sidecar.write_text(
                json.dumps(final_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(self.partial, self.output)
            os.replace(self.partial_sidecar, self.sidecar)
            marker_partial = self.completion_marker.with_name(
                f".{self.completion_marker.name}.partial"
            )
            marker_partial.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION_V4,
                        "event_count": self._event_count,
                        "feature_spec_hash": self.spec["feature_spec_hash"],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(marker_partial, self.completion_marker)
            self._closed = True
            return self.output
        except Exception:
            self.abort()
            raise

    finalize = close

    def abort(self) -> None:
        if not self._closed:
            try:
                self._writer.close()
            except Exception:
                pass
        for path in (self.partial, self.partial_sidecar):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._closed = True

    def __enter__(self) -> "ParquetEventWriter":
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _update_statistics(self, event: Mapping[str, Any]) -> None:
        nodes = event.get("nodes", [])
        self._capacity["events"] += 1
        self._capacity["nodes"] += len(nodes)
        mothers_by_level: Counter[int] = Counter()
        event_max_depth = 0
        for node in nodes:
            level = int(node.get("level", 0))
            daughters = len(node.get("daughter_ids", []))
            event_max_depth = max(event_max_depth, level)
            if daughters > 0 and bool(node.get("valid_reconstruction_target", False)):
                mothers_by_level[level] += 1
            self._capacity[f"nodes_level_{level}"] += 1
            self._capacity[f"mothers_level_{level}"] += int(daughters > 0)
            self._capacity[f"daughter_cardinality_{daughters}"] += int(daughters > 0)
            if daughters > 0 and bool(node.get("valid_reconstruction_target", False)):
                target_token = int(node.get("pid_target_token", 0))
                self._capacity[
                    f"target_type_level_{level}_token_{target_token}"
                ] += 1
            self._capacity["max_depth"] = max(self._capacity["max_depth"], level)
            self._pid[str(int(node.get("input_pid_token", 0)))] += 1
            mode = str(node.get("leaf_kinematics_mode", "legacy_conflated"))
            self._capacity[f"leaf_mode_{mode}"] += 1
            self._completeness["valid_targets"] += int(
                bool(node.get("valid_reconstruction_target", False))
            )
            self._completeness["partial_targets"] += int(
                bool(node.get("partial_missing_daughters", False))
            )
            self._completeness["recursive_complete"] += int(
                bool(node.get("recursive_reconstructable_complete", False))
            )
            for block, record_key, availability_key in (
                ("common", "common_features", "common_availability"),
                ("track", "track_features", "track_availability"),
                ("ecl_cluster", "cluster_features", "cluster_availability"),
                ("composite", "composite_features", "composite_availability"),
            ):
                names = self.spec[block]
                values = node.get(record_key, {})
                available = node.get(availability_key, {})
                for index, name in enumerate(names):
                    if bool(available.get(name, False)):
                        _update_feature_statistic(
                            self._feature_statistics[block], index, float(values[name])
                        )
        self._capacity[f"depth_{event_max_depth}"] += 1
        for level in range(1, event_max_depth + 1):
            self._capacity[
                f"mother_count_level_{level}_value_{mothers_by_level[level]}"
            ] += 1


def _empty_feature_statistics(width: int) -> dict[str, list[float]]:
    return {
        "count": [0.0] * width,
        "mean": [0.0] * width,
        "m2": [0.0] * width,
    }


def _actual_collection_mode(distribution: Mapping[str, int]) -> str:
    populated = sorted(name for name, count in distribution.items() if int(count) > 0)
    return populated[0] if len(populated) == 1 else "mixed_explicit_per_node"


def _update_feature_statistic(
    statistics: dict[str, list[float]], index: int, value: float
) -> None:
    count = statistics["count"][index] + 1.0
    delta = value - statistics["mean"][index]
    mean = statistics["mean"][index] + delta / count
    statistics["m2"][index] += delta * (value - mean)
    statistics["mean"][index] = mean
    statistics["count"][index] = count


def export_trees_v4(
    trees: Iterable[EventTree],
    output: str | Path,
    *,
    summary: Mapping[str, Any] | None = None,
    charge_conjugate_normalize: bool = False,
    event_buffer_size: int = 128,
    row_group_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write trees incrementally as one v4 event per parquet row."""

    spec = feature_spec_v4()
    writer_metadata = {
        "charge_conjugate_normalization": bool(charge_conjugate_normalize),
        "preprocessing_configuration": dict(summary or {}),
        **dict(metadata or {}),
    }
    with ParquetEventWriter(
        output,
        event_buffer_size=event_buffer_size,
        row_group_size=row_group_size,
        metadata=writer_metadata,
    ) as writer:
        for tree in trees:
            writer.write_tree(
                tree,
                charge_conjugate_normalize=charge_conjugate_normalize,
            )
    return Path(output)


def iter_event_records_v4(
    path: str | Path,
    *,
    batch_size: int = 64,
    worker_id: int = 0,
    worker_count: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield v4 records lazily, or explicitly adapted legacy records."""

    source = Path(path)
    parquet = pq.ParquetFile(source)
    names = set(parquet.schema_arrow.names)
    if "event_json" in names:
        metadata = parquet.schema_arrow.metadata or {}
        schema_version = json.loads(
            metadata.get(b"schema_version", b'"direct-mdst-tree-v4"').decode("utf-8")
        )
        if schema_version != SCHEMA_VERSION_V4:
            raise ValueError(f"event-row parquet has unsupported schema {schema_version!r}")
        if worker_count <= 0 or not 0 <= worker_id < worker_count:
            raise ValueError("invalid worker row-group partition")
        for row_group in range(worker_id, parquet.num_row_groups, worker_count):
            table = parquet.read_row_group(row_group, columns=["event_json"])
            for record_batch in table.to_batches(max_chunksize=batch_size):
                for text in record_batch.column(0).to_pylist():
                    yield json.loads(text)
        return
    payload = load_payload_v3(source)
    source_version = str(payload.get("source_schema_version", SCHEMA_VERSION_V3))
    for event in payload["events"]:
        yield _legacy_event_to_v4(event, source_version=source_version)


def load_payload_v4(path: str | Path) -> dict[str, Any]:
    """Materializing compatibility API; production code should use the iterator."""

    source = Path(path)
    records = list(iter_event_records_v4(source))
    spec = feature_spec_v4()
    sidecar = source.with_suffix(source.suffix + ".metadata.json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    summary = {
        "capacity": dict(metadata.get("aggregate_capacity_statistics", {})),
        "pid": dict(metadata.get("aggregate_pid_statistics", {})),
        "completeness": dict(
            metadata.get("aggregate_completeness_statistics", {})
        ),
        "leaf_modes": dict(metadata.get("actual_leaf_mode_distribution", {})),
    }
    summary.setdefault("n_events", len(records))
    source_version = (
        records[0].get("source_schema_version", SCHEMA_VERSION_V4)
        if records
        else SCHEMA_VERSION_V4
    )
    return {
        "schema_version": SCHEMA_VERSION_V4,
        "source_schema_version": source_version,
        "pid_vocabulary_version": PID_VOCABULARY_VERSION,
        "feature_spec_hash": spec["feature_spec_hash"],
        "feature_spec_json": json.dumps(spec, sort_keys=True),
        "summary_json": json.dumps(summary, sort_keys=True),
        "events": records,
    }


def _native_v4_event(event: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(event)
    output["schema_version"] = SCHEMA_VERSION_V4
    output["source_schema_version"] = SCHEMA_VERSION_V4
    nodes = [dict(node) for node in event.get("nodes", [])]
    node_by_id = {int(node["node_id"]): node for node in nodes}
    full_event_max_level = max(
        (int(node.get("level", 0)) for node in nodes), default=0
    )
    output["full_event_max_level"] = full_event_max_level
    for node in nodes:
        node["leaf_kinematics_mode_id"] = _leaf_mode_id(node)
        input_hist = [0.0] * len(PDG_TOKENS)
        truth_hist = [0.0] * len(PDG_TOKENS)
        for daughter_id in node.get("daughter_ids", []):
            daughter = node_by_id[int(daughter_id)]
            input_hist[
                validate_pid_token(int(daughter.get("input_pid_token", 0)))
            ] += 1.0
            truth_token = daughter.get("truth_pid_token")
            if truth_token is None:
                truth_token = daughter.get("pid_target_token")
            if truth_token is not None:
                truth_hist[validate_pid_token(int(truth_token))] += 1.0
        available = bool(node.get("daughter_ids"))
        node["daughter_input_pid_histogram"] = input_hist
        node["daughter_truth_pid_histogram"] = truth_hist
        node["daughter_input_pid_histogram_available"] = available
        node["daughter_truth_pid_histogram_available"] = available
        node.pop("daughter_pid_histogram", None)
        node.pop("daughter_pid_histogram_available", None)
        common_availability = dict(node.get("common_availability", {}))
        for name in CATEGORICAL_COMMON_FEATURE_NAMES:
            common_availability[name] = False
        node["common_availability"] = common_availability
        node.setdefault(
            "retained_truth_daughter_count_expected",
            int(node.get("full_truth_daughter_count", len(node.get("daughter_ids", [])))),
        )
        node["truth_level_id"] = int(node.get("level", 0))
        node["full_event_max_level"] = full_event_max_level
        node["truth_root_distance"] = (
            full_event_max_level - int(node.get("level", 0))
        )
    _annotate_recursive_completeness(nodes)
    output["nodes"] = nodes
    output["legacy_conflated_fraction"] = 0.0
    return output


def _legacy_event_to_v4(event: Mapping[str, Any], *, source_version: str) -> dict[str, Any]:
    output = dict(event)
    output["schema_version"] = SCHEMA_VERSION_V4
    output["source_schema_version"] = source_version
    nodes = [dict(node) for node in event.get("nodes", [])]
    legacy_count = 0
    full_event_max_level = max(
        (int(candidate.get("level", 0)) for candidate in nodes), default=0
    )
    output["full_event_max_level"] = full_event_max_level
    for node in nodes:
        legacy_count += 1
        ambiguous = list(node.get("daughter_pid_histogram", [0] * len(PDG_TOKENS)))
        node["daughter_input_pid_histogram"] = ambiguous
        node["daughter_truth_pid_histogram"] = ambiguous
        available = bool(node.get("daughter_pid_histogram_available", False))
        node["daughter_input_pid_histogram_available"] = available
        node["daughter_truth_pid_histogram_available"] = available
        node["legacy_daughter_pid_histogram"] = ambiguous
        node["leaf_kinematics_mode"] = "legacy_conflated"
        node["leaf_kinematics_mode_id"] = LEAF_MODE_TO_ID["legacy_conflated"]
        node.setdefault(
            "retained_truth_daughter_count_expected",
            int(node.get("full_truth_daughter_count", len(node.get("daughter_ids", [])))),
        )
        node["truth_level_id"] = int(node.get("level", 0))
        node["full_event_max_level"] = full_event_max_level
        node["truth_root_distance"] = (
            full_event_max_level - int(node.get("level", 0))
        )
    _annotate_recursive_completeness(nodes)
    output["nodes"] = nodes
    output["legacy_conflated_fraction"] = legacy_count / max(len(nodes), 1)
    return output


def _leaf_mode_id(node: Mapping[str, Any]) -> int:
    if node.get("daughter_ids"):
        mode = "composite"
    elif str(node.get("node_kind")) == "ecl_cluster":
        mode = "ecl_cluster"
    else:
        mode = str(node.get("leaf_kinematics_mode", "truth_topology_only"))
    if mode not in LEAF_MODE_TO_ID:
        mode = "truth_topology_only"
    return LEAF_MODE_TO_ID[mode]


def _annotate_recursive_completeness(nodes: list[dict[str, Any]]) -> None:
    by_id = {int(node["node_id"]): node for node in nodes}
    memo: dict[int, bool] = {}

    def complete(node_id: int, visiting: set[int]) -> bool:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            memo[node_id] = False
            return False
        node = by_id[node_id]
        daughters = [
            int(value) for value in node.get("daughter_ids", []) if int(value) in by_id
        ]
        if not daughters:
            result = bool(
                node.get("reco_object_id")
                or node.get("reco_id")
                or str(node.get("node_kind")) == "ecl_cluster"
            )
        else:
            result = bool(node.get("complete_reconstructable_decay", False)) and all(
                complete(child, visiting | {node_id}) for child in daughters
            )
        memo[node_id] = result
        return result

    for node in nodes:
        node["recursive_reconstructable_complete"] = complete(int(node["node_id"]), set())


def _git_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


__all__ = [
    "LEAF_KINEMATICS_MODES",
    "LEAF_MODE_FROM_ID",
    "LEAF_MODE_TO_ID",
    "ParquetEventWriter",
    "SCHEMA_VERSION_V4",
    "export_trees_v4",
    "feature_spec_v4",
    "iter_event_records_v4",
    "load_payload_v4",
]
