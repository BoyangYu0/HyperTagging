import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from hypertagging.data.notebook_fixtures import (
    write_notebook_fixture_v3,
    write_notebook_fixture_v4,
)
from hypertagging.preprocessing.schema_v4 import (
    SCHEMA_VERSION_V4,
    iter_event_records_v4,
)
from hypertagging.training.data_module import build_real_data_module


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mdst_batch_production_v4", ROOT / "scripts" / "mdst_batch_production.py"
)
assert SPEC and SPEC.loader
production = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(production)


def test_v4_is_event_row_default_and_metadata_matches(tmp_path, monkeypatch):
    path = write_notebook_fixture_v4(tmp_path / "events.parquet")
    parquet = pq.ParquetFile(path)
    assert parquet.metadata.num_rows == 2
    assert parquet.num_row_groups == 2
    records = list(iter_event_records_v4(path))
    assert all(record["schema_version"] == SCHEMA_VERSION_V4 for record in records)
    sidecar = json.loads(
        path.with_suffix(".parquet.metadata.json").read_text(encoding="utf-8")
    )
    assert sidecar["schema_version"] == SCHEMA_VERSION_V4
    assert sidecar["event_count"] == 2

    monkeypatch.setattr(production, "root_event_count", lambda _path: 2)
    manifest, _ = production.build_manifest_records(
        [tmp_path / "input.root"],
        output_root=tmp_path,
        target_events=2,
        events_per_task=2,
        charge_conjugate_normalization=True,
    )
    assert manifest[0]["schema_version"] == SCHEMA_VERSION_V4
    assert manifest[0]["charge_conjugate_normalization"] is True
    assert manifest[0]["event_buffer_size"] > 0


def test_legacy_gate_requires_explicit_diagnostic_opt_in(tmp_path):
    legacy = write_notebook_fixture_v3(tmp_path / "legacy.parquet")
    with pytest.raises(ValueError, match="legacy-conflated"):
        build_real_data_module(legacy)
    with pytest.warns(RuntimeWarning, match="DIAGNOSTIC ONLY"):
        module = build_real_data_module(
            legacy,
            allow_legacy_conflated=True,
            pilot_split_repair=True,
        )
    assert module.legacy_conflated_fraction == 1.0
