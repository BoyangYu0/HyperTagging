import pytest
import torch

from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.preprocessing.pid_filter import (
    PDG_TOKENS,
    TOKENIZE_DICT,
    validate_pid_token,
    validate_pid_tokens,
)


def test_tokens_and_fixture_contract():
    assert TOKENIZE_DICT[22] == 2
    assert TOKENIZE_DICT[521] != TOKENIZE_DICT[22]
    for event in tiny_level_events():
        validate_pid_tokens(event.pid_labels)
        assert event.raw_pdg is not None
        assert any(abs(int(value)) > len(PDG_TOKENS) for value in event.raw_pdg)
    with pytest.raises(ValueError):
        validate_pid_token(-1)
    with pytest.raises(ValueError):
        validate_pid_tokens(torch.tensor([0, len(PDG_TOKENS)]))


def test_model_vocabulary_is_exact_and_invalid_is_not_repaired():
    model = LevelAutoregressiveReconstructor(n_features=8, n_types=len(PDG_TOKENS))
    assert model.decoder.type_head.out_features == len(PDG_TOKENS)
    assert model.encoder.pid_embedding.num_embeddings == len(PDG_TOKENS)
