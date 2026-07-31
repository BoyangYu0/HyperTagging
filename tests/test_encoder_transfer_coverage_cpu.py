import pytest

from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
from hypertagging.training.checkpointing import save_training_checkpoint
from hypertagging.training.pretrained_transfer import load_pretrained_encoder


def test_production_transfer_rejects_low_coverage(tmp_path):
    source = HeterogeneousNodeEncoder(d_model=16, hyper_dim=4)
    path = save_training_checkpoint(tmp_path / "source.pt", model=source, encoder=source)
    target = HeterogeneousNodeEncoder(d_model=32, hyper_dim=8)
    with pytest.raises(ValueError, match="coverage"):
        load_pretrained_encoder(target, path, minimum_coverage=0.9)

