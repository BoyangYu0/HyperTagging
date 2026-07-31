import pytest

from hypertagging.data.dataset_index import build_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.training.data_module import build_real_data_module


def test_full_index_cannot_back_truncated_pilot(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    with pytest.raises(ValueError, match="max-events"):
        build_real_data_module(
            shard, dataset_index=index_path, max_events=1, pilot_split_repair=True
        )

