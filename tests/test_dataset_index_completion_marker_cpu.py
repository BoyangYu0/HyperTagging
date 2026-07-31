import pytest

from hypertagging.data.dataset_index import build_dataset_index, load_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4


def test_changed_completion_marker_rejects_stale_index(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    marker = shard.with_suffix(shard.suffix + ".complete")
    marker.write_text("{}\n")
    with pytest.raises(ValueError, match="completion marker"):
        load_dataset_index(index_path)

