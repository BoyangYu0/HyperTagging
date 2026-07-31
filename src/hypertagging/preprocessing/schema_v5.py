"""Native nested Arrow event rows and a bounded JSON-v4/native benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any, Iterable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from hypertagging.preprocessing.schema_v4 import (
    ParquetEventWriter,
    iter_event_records_v4,
)


SCHEMA_VERSION_V5 = "direct-mdst-tree-v5-native-nested"


class NativeNestedEventWriter:
    """Bounded native-Arrow writer; schema is frozen from the first buffer."""

    def __init__(self, output: str | Path, *, event_buffer_size: int = 128) -> None:
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.output.with_name(f".{self.output.name}.partial")
        self.event_buffer_size = int(event_buffer_size)
        self.buffer: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.event_count = 0

    def write_event(self, event: Mapping[str, Any]) -> None:
        record = dict(event)
        record["schema_version"] = SCHEMA_VERSION_V5
        self.buffer.append(record)
        self.event_count += 1
        if len(self.buffer) >= self.event_buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if self.schema is None:
            inferred = pa.Table.from_pylist(self.buffer)
            metadata = {
                b"schema_version": json.dumps(SCHEMA_VERSION_V5).encode(),
                b"event_layout": b'"native-nested-one-event-per-row"',
            }
            self.schema = inferred.schema.with_metadata(metadata)
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
    for path in v4_paths:
        for record in iter_event_records_v4(path):
            records.append(record)
            if len(records) >= max_events:
                break
        if len(records) >= max_events:
            break
    json_decode_seconds = max(time.perf_counter() - decode_start, 1e-9)
    if not records:
        raise ValueError("storage benchmark requires at least one event")
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
    native_records = list(iter_native_nested_v5(native_path))
    native_read_seconds = max(time.perf_counter() - read_start, 1e-9)
    _, native_peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    projected_start = time.perf_counter()
    projected = list(iter_native_nested_v5(native_path, columns=["event_uid"]))
    projected_seconds = max(time.perf_counter() - projected_start, 1e-9)
    tracemalloc.start()
    comparable_json_start = time.perf_counter()
    comparable_json_records = list(iter_event_records_v4(json_path))
    comparable_json_seconds = max(time.perf_counter() - comparable_json_start, 1e-9)
    _, json_peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result: dict[str, float | int | str] = {
        "event_count": len(records),
        "json_file_size_bytes": json_path.stat().st_size,
        "native_file_size_bytes": native_path.stat().st_size,
        "json_write_events_per_second": len(records) / json_write_seconds,
        "json_read_events_per_second": len(comparable_json_records)
        / comparable_json_seconds,
        "native_write_events_per_second": len(records) / native_write_seconds,
        "native_read_events_per_second": len(native_records) / native_read_seconds,
        "native_projected_read_events_per_second": len(projected) / projected_seconds,
        "json_decode_cpu_seconds": comparable_json_seconds,
        "source_sample_decode_seconds": json_decode_seconds,
        "native_decode_cpu_seconds": native_read_seconds,
        "json_peak_python_bytes": json_peak_python_bytes,
        "native_peak_python_bytes": native_peak_python_bytes,
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
]
