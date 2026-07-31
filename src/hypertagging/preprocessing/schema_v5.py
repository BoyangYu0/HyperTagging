"""Native nested Arrow event rows and a bounded JSON-v4/native benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import tracemalloc
import resource
from typing import Any, Iterable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from hypertagging.preprocessing.schema_v4 import (
    ParquetEventWriter,
    feature_spec_v4,
    iter_event_records_v4,
)
from hypertagging.preprocessing.schema_v3 import CHARGED_STABLE_NAMES


SCHEMA_VERSION_V5 = "direct-mdst-tree-v5-native-nested"


class NativeNestedEventWriter:
    """Bounded experimental writer with an explicit feature-spec schema."""

    def __init__(self, output: str | Path, *, event_buffer_size: int = 128) -> None:
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.output.with_name(f".{self.output.name}.partial")
        self.event_buffer_size = int(event_buffer_size)
        self.buffer: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema = native_nested_schema_v5()
        self.event_count = 0

    def write_event(self, event: Mapping[str, Any]) -> None:
        record = dict(event)
        record["schema_version"] = SCHEMA_VERSION_V5
        _validate_native_record(record, self.schema)
        self.buffer.append(record)
        self.event_count += 1
        if len(self.buffer) >= self.event_buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.partial, self.schema, compression="zstd")
        assert self.writer is not None and self.schema is not None
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.event_buffer_size)
        self.buffer.clear()

    def close(self) -> Path:
        try:
            self.flush()
            if self.writer is None:
                raise ValueError("cannot publish an empty native nested dataset")
            self.writer.close()
            os.replace(self.partial, self.output)
            marker = self.output.with_suffix(self.output.suffix + ".complete")
            marker.write_text(
                json.dumps(
                    {"schema_version": SCHEMA_VERSION_V5, "event_count": self.event_count}
                )
                + "\n"
            )
            return self.output
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
        self.partial.unlink(missing_ok=True)

    def __enter__(self) -> "NativeNestedEventWriter":
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        self.close() if exc_type is None else self.abort()


def native_nested_schema_v5() -> pa.Schema:
    """Build the stable v5 schema from the versioned feature specification."""

    spec = feature_spec_v4()
    feature_struct = lambda names, value_type: pa.struct(
        [pa.field(name, value_type) for name in names]
    )
    node_fields: list[pa.Field] = []
    integer_names = {
        "node_id", "mc_id", "raw_pdg", "reduced_pid_token", "input_pid_token",
        "pid_target_token", "node_kind_id", "leaf_kinematics_mode_id", "level",
        "parent_id", "copied_from", "source_node_id", "pdg", "token",
        "truth_pdg", "truth_pid_token", "full_truth_daughter_count",
        "retained_truth_daughter_count_expected", "retained_daughter_count",
        "reconstructed_daughter_count", "truth_level_id", "full_event_max_level",
        "truth_root_distance",
    }
    float_names = {
        "charge", "reco_charge", "truth_charge", "px", "py", "pz", "energy",
        "reconstructed_energy", "mass", "candidate_confidence", "mc_px", "mc_py",
        "mc_pz", "mc_energy",
    }
    bool_names = {
        "active", "copied", "daughter_input_pid_histogram_available",
        "daughter_truth_pid_histogram_available", "complete_truth_decay",
        "complete_reconstructable_decay", "recursive_reconstructable_complete",
        "partial_missing_daughters", "contracted_intermediate",
        "valid_reconstruction_target",
    }
    string_names = {
        "reco_object_id", "reco_id", "node_kind", "leaf_kinematics_mode", "energy_source",
    }
    for name in sorted(integer_names): node_fields.append(pa.field(name, pa.int64()))
    for name in sorted(float_names): node_fields.append(pa.field(name, pa.float64()))
    for name in sorted(bool_names): node_fields.append(pa.field(name, pa.bool_()))
    for name in sorted(string_names): node_fields.append(pa.field(name, pa.string()))
    node_fields.append(pa.field("daughter_ids", pa.list_(pa.int64())))
    node_fields.append(pa.field("recursive_leaf_source_ids", pa.list_(pa.string())))
    node_fields.append(pa.field("flags", pa.list_(pa.string())))
    for block, prefix in (
        ("common", "common"), ("track", "track"),
        ("ecl_cluster", "cluster"), ("composite", "composite"),
    ):
        node_fields.append(pa.field(f"{prefix}_features", feature_struct(spec[block], pa.float64())))
        node_fields.append(pa.field(f"{prefix}_availability", feature_struct(spec[block], pa.bool_())))
    for name in ("daughter_input_pid_histogram", "daughter_truth_pid_histogram"):
        node_fields.append(pa.field(name, pa.list_(pa.float64())))
    for name, value_type in (
        ("pid_likelihoods", pa.float64()),
        ("pid_likelihood_availability", pa.bool_()),
        ("mass_hypothesis_energies", pa.float64()),
        ("mass_hypothesis_availability", pa.bool_()),
    ):
        node_fields.append(pa.field(name, feature_struct(CHARGED_STABLE_NAMES, value_type)))
    top_fields: list[pa.Field] = [
        pa.field("event_id", pa.int64()), pa.field("event_uid", pa.string()),
        pa.field("schema_version", pa.string()), pa.field("source_schema_version", pa.string()),
        pa.field("source_file", pa.string()), pa.field("source_category", pa.string()),
        pa.field("experiment", pa.int64()), pa.field("run", pa.int64()),
        pa.field("production", pa.int64()), pa.field("feature_spec_hash", pa.string()),
        pa.field("pid_vocabulary_version", pa.string()), pa.field("leaf_kinematics_mode", pa.string()),
        pa.field("metadata_json", pa.large_string()),
        pa.field("nodes", pa.list_(pa.struct(node_fields))),
        pa.field("levels", pa.list_(pa.struct([pa.field("level", pa.int64()), pa.field("node_ids", pa.list_(pa.int64()))]))),
        pa.field("root_ids", pa.list_(pa.int64())),
    ]
    for side in ("b1", "b2"):
        top_fields.extend([
            pa.field(f"{side}_channel_count_array", pa.list_(pa.int64())),
            pa.field(f"{side}_depth_pid_count_array", pa.list_(pa.list_(pa.int64()))),
            pa.field(f"{side}_channel_summary_json", pa.large_string()),
            pa.field(f"{side}_root_id", pa.int64()),
        ])
    for prefix in ("b1", "b2", "y4s"):
        for variant in ("", "full_truth_", "reconstructable_"):
            top_fields.append(pa.field(f"{prefix}_{variant}channel_id", pa.int64()))
            top_fields.append(pa.field(f"{prefix}_{variant}channel_signature", pa.string()))
    for name in (
        "b_root_discovery_fallback", "b_root_discovery_valid", "charge_conjugate_normalization",
        "charge_conjugate_normalized", "exact_channel_equal", "full_truth_channel_available",
        "reconstructable_channel_available", "same_event",
    ):
        top_fields.append(pa.field(name, pa.bool_()))
    for name in ("structured_channel_similarity", "legacy_conflated_fraction"):
        top_fields.append(pa.field(name, pa.float64()))
    top_fields.append(pa.field("full_event_max_level", pa.int64()))
    metadata = {
        b"schema_version": json.dumps(SCHEMA_VERSION_V5).encode(),
        b"event_layout": b'"native-nested-one-event-per-row"',
        b"feature_spec_hash": json.dumps(spec["feature_spec_hash"]).encode(),
        b"experimental_default_off": b"true",
    }
    return pa.schema(top_fields, metadata=metadata)


def _validate_native_record(record: Mapping[str, Any], schema: pa.Schema) -> None:
    unknown = set(record) - set(schema.names)
    if unknown:
        raise ValueError(f"unknown native-v5 event field(s): {sorted(unknown)}")
    node_type = schema.field("nodes").type.value_type
    allowed_nodes = {field.name for field in node_type}
    for index, node in enumerate(record.get("nodes", ())):
        unknown_node = set(node) - allowed_nodes
        if unknown_node:
            raise ValueError(
                f"unknown native-v5 node field(s) at node {index}: {sorted(unknown_node)}"
            )
        for field_name in (
            "common_features", "common_availability", "track_features",
            "track_availability", "cluster_features", "cluster_availability",
            "composite_features", "composite_availability", "pid_likelihoods",
            "pid_likelihood_availability", "mass_hypothesis_energies",
            "mass_hypothesis_availability",
        ):
            values = node.get(field_name)
            if values is None:
                continue
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"native-v5 node {index} field {field_name} must be a mapping"
                )
            field_type = node_type.field(field_name).type
            unknown_features = set(values) - {
                field.name for field in field_type
            }
            if unknown_features:
                raise ValueError(
                    f"unknown native-v5 {field_name} field(s) at node {index}: "
                    f"{sorted(unknown_features)}"
                )


def iter_native_nested_v5(
    path: str | Path, *, columns: list[str] | None = None
) -> Iterator[dict[str, Any]]:
    dataset = pq.ParquetFile(path)
    for row_group in range(dataset.num_row_groups):
        table = dataset.read_row_group(row_group, columns=columns)
        yield from table.to_pylist()


def benchmark_storage_formats(
    v4_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    max_events: int = 1000,
) -> dict[str, float | int | str]:
    """Bounded synthetic/pilot benchmark; never intended as a 10M launch."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    decode_start = time.perf_counter()
    decode_cpu_start = time.process_time()
    for path in v4_paths:
        for record in iter_event_records_v4(path):
            records.append(record)
            if len(records) >= max_events:
                break
        if len(records) >= max_events:
            break
    json_decode_seconds = max(time.perf_counter() - decode_start, 1e-9)
    source_json_decode_cpu_seconds = max(time.process_time() - decode_cpu_start, 0.0)
    if not records:
        raise ValueError("storage benchmark requires at least one event")
    node_count = sum(len(record.get("nodes", ())) for record in records)
    json_path = output / "event-json-v4.parquet"
    json_write_start = time.perf_counter()
    with ParquetEventWriter(
        json_path, event_buffer_size=min(128, len(records))
    ) as writer:
        for record in records:
            writer.write_event(record)
    json_write_seconds = max(time.perf_counter() - json_write_start, 1e-9)
    native_path = output / "native-v5.parquet"
    write_start = time.perf_counter()
    with NativeNestedEventWriter(native_path, event_buffer_size=min(128, len(records))) as writer:
        for record in records:
            writer.write_event(record)
    native_write_seconds = max(time.perf_counter() - write_start, 1e-9)
    tracemalloc.start()
    read_start = time.perf_counter()
    native_cpu_start = time.process_time()
    native_records = list(iter_native_nested_v5(native_path))
    native_read_seconds = max(time.perf_counter() - read_start, 1e-9)
    native_decode_cpu_seconds = max(time.process_time() - native_cpu_start, 0.0)
    _, native_peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    projected_start = time.perf_counter()
    projected = list(iter_native_nested_v5(native_path, columns=["event_uid"]))
    projected_seconds = max(time.perf_counter() - projected_start, 1e-9)
    tracemalloc.start()
    comparable_json_start = time.perf_counter()
    comparable_json_cpu_start = time.process_time()
    comparable_json_records = list(iter_event_records_v4(json_path))
    comparable_json_seconds = max(time.perf_counter() - comparable_json_start, 1e-9)
    comparable_json_cpu_seconds = max(
        time.process_time() - comparable_json_cpu_start, 0.0
    )
    _, json_peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result: dict[str, float | int | str] = {
        "event_count": len(records),
        "node_count": node_count,
        "json_file_size_bytes": json_path.stat().st_size,
        "native_file_size_bytes": native_path.stat().st_size,
        "json_write_events_per_second": len(records) / json_write_seconds,
        "json_write_nodes_per_second": node_count / json_write_seconds,
        "json_write_bytes_per_second": json_path.stat().st_size / json_write_seconds,
        "json_read_events_per_second": len(comparable_json_records)
        / comparable_json_seconds,
        "json_read_nodes_per_second": node_count / comparable_json_seconds,
        "json_full_read_bytes_per_second": json_path.stat().st_size / comparable_json_seconds,
        "native_write_events_per_second": len(records) / native_write_seconds,
        "native_write_nodes_per_second": node_count / native_write_seconds,
        "native_write_bytes_per_second": native_path.stat().st_size / native_write_seconds,
        "native_read_events_per_second": len(native_records) / native_read_seconds,
        "native_read_nodes_per_second": node_count / native_read_seconds,
        "native_full_read_bytes_per_second": native_path.stat().st_size / native_read_seconds,
        "native_projected_read_events_per_second": len(projected) / projected_seconds,
        "native_projected_read_bytes_per_second": native_path.stat().st_size / projected_seconds,
        "json_decode_cpu_seconds": comparable_json_cpu_seconds,
        "source_sample_decode_seconds": json_decode_seconds,
        "source_json_decode_cpu_seconds": source_json_decode_cpu_seconds,
        "native_decode_cpu_seconds": native_decode_cpu_seconds,
        "json_peak_python_bytes": json_peak_python_bytes,
        "native_peak_python_bytes": native_peak_python_bytes,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "arrow_allocation_caveat": (
            "tracemalloc excludes Arrow C++ allocations; process peak RSS is process-wide "
            "and may include earlier allocations"
        ),
        "review_required_before_10m": "true",
    }
    (output / "storage_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "NativeNestedEventWriter",
    "SCHEMA_VERSION_V5",
    "benchmark_storage_formats",
    "iter_native_nested_v5",
    "native_nested_schema_v5",
]
