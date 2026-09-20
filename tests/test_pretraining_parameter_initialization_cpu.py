"""Parameter initialization preserves fresh state and fails before partial mutation."""
from dataclasses import replace
import torch
import pytest
from hypertagging.training.pretrain_trainer import initialize_pretraining_parameters, PretrainConfig, train_hyperbolic_pretraining
from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
from hypertagging.preprocessing.schema_v4 import feature_spec_v4


class BufferedModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.register_buffer('normalizer', torch.tensor([5.0]))
        self.register_buffer('memory', torch.zeros(3))


def checkpoint(tmp_path, model):
    state = {k: torch.full_like(v, 9) for k, v in model.state_dict().items()}
    payload = {'architecture': {'test': 1}, 'pid_vocabulary_version': PID_VOCABULARY_VERSION,
               'feature_specification': feature_spec_v4(), 'preprocessing_schema_version': 'direct-mdst-tree-v4',
               'model_state_dict': state, 'step': 81, 'optimizer_state_dict': {'ignored': True}}
    path = tmp_path/'source.pt'
    torch.save(payload, path)
    return path, payload


def test_initialization_copies_all_parameters_and_preserves_buffers(tmp_path):
    model = BufferedModel()
    path, _ = checkpoint(tmp_path, model)
    before = path.read_bytes()
    result = initialize_pretraining_parameters(model, path, architecture={'test': 1})
    assert all(torch.equal(p, torch.full_like(p, 9)) for p in model.parameters())
    assert model.normalizer.item() == 5 and torch.equal(model.memory, torch.zeros(3))
    assert result == {'source_step': 81, 'parameter_count': 2, 'buffers_restored': False, 'optimizer_restored': False}
    assert path.read_bytes() == before


@pytest.mark.parametrize('fault', ['missing', 'shape', 'nonfinite', 'architecture', 'features', 'vocabulary', 'schema'])
def test_invalid_initialization_is_atomic(tmp_path, fault):
    model = BufferedModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    path, payload = checkpoint(tmp_path, model)
    if fault == 'missing': payload['model_state_dict'].pop('bias')
    if fault == 'shape': payload['model_state_dict']['bias'] = torch.zeros(3)
    if fault == 'nonfinite': payload['model_state_dict']['bias'][0] = float('nan')
    if fault == 'architecture': payload['architecture'] = {}
    if fault == 'features': payload['feature_specification'] = {}
    if fault == 'vocabulary': payload['pid_vocabulary_version'] = 'wrong'
    if fault == 'schema': payload['preprocessing_schema_version'] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError): initialize_pretraining_parameters(model, path, architecture={'test': 1})
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_resume_cannot_be_mixed_with_fresh_initialization(tmp_path):
    c = PretrainConfig(data='unused', output_dir=str(tmp_path/'out'), resume='unused', weights_initialization_checkpoint='unused')
    with pytest.raises(ValueError, match='mutually exclusive'): train_hyperbolic_pretraining(c)
    c = replace(c, resume=None, validation_event_uids=('same', 'same'), validation_events=2)
    with pytest.raises(ValueError, match='unique'): train_hyperbolic_pretraining(c)


def test_explicit_validation_cohort_controls_real_trainer(tmp_path):
    from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
    from hypertagging.training.checkpointing import load_training_checkpoint
    data = write_notebook_fixture_v3(tmp_path/'fixture.parquet')
    c = PretrainConfig(data=str(data), output_dir=str(tmp_path/'first'), max_steps=1,
                      batch_size=2, allow_legacy_conflated=True, validate_every=1,
                      validation_events=2, validation_batches=1, log_every=1)
    first = train_hyperbolic_pretraining(c)
    ids = tuple(reversed(load_training_checkpoint(first.checkpoint)['validation_selection']['event_uids']))
    second = train_hyperbolic_pretraining(replace(c, output_dir=str(tmp_path/'second'), validation_event_uids=ids, parent_ranking_weight=2.0))
    payload = load_training_checkpoint(second.checkpoint)
    assert payload['validation_selection']['event_uids'] == list(ids)
    assert payload['metrics']['pretraining_parent_ranking_weight'] == 2.0
