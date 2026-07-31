from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.preprocessing.schema_v4 import ParquetEventWriter


def test_overwrite_invalidates_old_marker_before_new_publication(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    marker = shard.with_suffix(shard.suffix + ".complete")
    assert marker.exists()
    writer = ParquetEventWriter(shard)
    assert not marker.exists()
    writer.abort()
    assert shard.exists() and not marker.exists()

