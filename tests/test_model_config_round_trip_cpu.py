from hypertagging.training.model_config import MODEL_PRESETS, ModelArchitecture
from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
from hypertagging.data.level_collate import collate_level_events
from hypertagging.data.tiny_level_fixtures import tiny_level_events
from hypertagging.preprocessing.pid_filter import PDG_TOKENS


def test_all_model_presets_round_trip_exactly():
    assert {"tiny_cpu", "gpu_debug", "production_baseline"} <= set(MODEL_PRESETS)
    for architecture in MODEL_PRESETS.values():
        assert ModelArchitecture.from_dict(architecture.to_dict()) == architecture


def test_level_specific_query_and_cardinality_dimensions_are_effective():
    model = LevelAutoregressiveReconstructor(
        n_features=8, n_types=len(PDG_TOKENS), hidden_dim=16, hyper_dim=4,
        n_queries=5, max_cardinality=6, n_context_layers=1,
        n_queries_by_level=((2, 3),), max_cardinality_by_level=((2, 4),),
    )
    batch = collate_level_events([tiny_level_events()[1]], max_query_slots=5).to_dict()
    output = model(batch, target_level=2)
    assert output.pointer.object_logits.shape[1] == 3
    assert output.pointer.cardinality_logits.shape[-1] == 5
