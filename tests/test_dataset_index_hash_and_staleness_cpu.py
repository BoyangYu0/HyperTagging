import json

import pytest

from hypertagging.data.dataset_index import build_dataset_index, load_dataset_index
from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4


def test_index_hash_and_source_digest_are_verified(tmp_path):
    shard = write_notebook_fixture_v4(tmp_path / "events.parquet")
    index_path = build_dataset_index([shard], tmp_path / "index.json")
    assert load_dataset_index(index_path)["shards"][0]["source_digest"]
    payload = json.loads(index_path.read_text())
    payload["event_count"] += 1
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        load_dataset_index(index_path)

