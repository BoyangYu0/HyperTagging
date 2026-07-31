import pytest

from hypertagging.data.dataset_index import build_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.training.data_module import build_real_data_module


def test_index_target_policy_must_equal_trainer_policy(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json", target_policy="complete_only")
    with pytest.raises(ValueError, match="target policy"):
        build_real_data_module(
            shard, dataset_index=index_path, target_policy="diagnostic_all",
            pilot_split_repair=True,
        )

