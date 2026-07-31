import pytest

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.training.data_module import build_real_data_module


def test_real_data_module_rejects_v4_without_completion_marker(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    shard.with_suffix(shard.suffix + ".complete").unlink()
    with pytest.raises(ValueError, match="incomplete schema-v4"):
        build_real_data_module(shard)

