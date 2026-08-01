from hypertagging.data.dataset_index import (
    build_dataset_index,
    build_dataset_index_from_sidecars,
    load_dataset_index,
)
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.preprocessing.schema_v5 import benchmark_storage_formats
from hypertagging.preprocessing.schema_v4 import (
    ParquetEventWriter,
    iter_event_records_v4,
)
from hypertagging.data.splitting import SourceAwareSplitConfig


def test_index_and_native_storage_benchmark(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet", row_group_size=1)
    index_path = build_dataset_index(
        [shard],
        tmp_path / "dataset_index.json",
        split_config=SourceAwareSplitConfig(
            train_fraction=1.0, validation_fraction=0.0, test_fraction=0.0
        ),
    )
    index = load_dataset_index(index_path)
    assert index["event_count"] == 2
    assert index["schema_versions"] == ["direct-mdst-tree-v4"]
    collision = index["full_truth_to_reconstructable_channel_collisions"]
    assert collision["collision_group_count"] >= 0
    assert "co-occurrence only" in collision["mechanism_scope"]
    result = benchmark_storage_formats([shard], tmp_path / "benchmark", max_events=3)
    assert result["event_count"] == 2
    assert result["json_write_events_per_second"] > 0
    assert result["json_read_events_per_second"] > 0
    assert result["native_read_events_per_second"] > 0
    assert result["native_projected_read_events_per_second"] > 0
    assert result["json_peak_python_bytes"] > 0
    assert result["native_peak_python_bytes"] > 0


def test_dataset_index_merges_published_sidecars_without_payload_scan(tmp_path):
    source = write_notebook_fixture_v4(tmp_path / "source.parquet", row_group_size=1)
    production = tmp_path / "production.parquet"
    with ParquetEventWriter(
        production,
        event_buffer_size=1,
        metadata={"source_file": "input.root", "category": "charged"},
    ) as writer:
        for event in iter_event_records_v4(source):
            writer.write_event(event)
    config = SourceAwareSplitConfig(
        train_fraction=1.0, validation_fraction=0.0, test_fraction=0.0
    )
    output = build_dataset_index_from_sidecars(
        [production], tmp_path / "sidecar-index.json", split_config=config
    )
    index = load_dataset_index(output)
    assert index["index_source"] == "merged_shard_sidecars"
    assert index["event_count"] == 2
    assert index["normalizer_state"]["common"]["count"][0] > 0
