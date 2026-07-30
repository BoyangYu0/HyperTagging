import torch

from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
from hypertagging.data.streaming import (
    BoundedShuffleBuffer,
    ParquetEventIterableDataset,
    StreamingMaskedFeatureNormalizer,
)
from hypertagging.data.splitting import MaskedFeatureNormalizer


def test_event_row_iteration_shuffle_and_early_stop(tmp_path):
    path = write_notebook_fixture_v4(tmp_path / "events.parquet", row_group_size=1)
    dataset = ParquetEventIterableDataset([path], max_events=2)
    records = list(dataset)
    assert len(records) == 2
    assert len({record["event_uid"] for record in records}) == 2
    first = list(BoundedShuffleBuffer(range(20), size=4, seed=9))
    second = list(BoundedShuffleBuffer(range(20), size=4, seed=9))
    assert first == second
    assert sorted(first) == list(range(20))


def test_streaming_normalizer_matches_batch_fit_and_merges():
    torch.manual_seed(3)
    values = torch.randn(23, 5)
    availability = torch.rand(23, 5) > 0.2
    expected = MaskedFeatureNormalizer().fit(values, availability)
    online = StreamingMaskedFeatureNormalizer()
    online.update(values[:7], availability[:7])
    right = StreamingMaskedFeatureNormalizer()
    right.update(values[7:], availability[7:])
    online.merge(right)
    torch.testing.assert_close(online.mean, expected.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        online.std, expected.standard_deviation, rtol=1e-5, atol=1e-6
    )
