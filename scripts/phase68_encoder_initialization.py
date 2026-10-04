"""Build a parameter-only encoder ablation with identical refined PID and train buffers.

This is a hybrid transfer artifact, not an untrained model or resumed checkpoint.
Only encoder parameters are replaced by their authenticated Phase64 pre-refinement
values. The decoder is subsequently initialized afresh by the normal trainer.
"""
from pathlib import Path
import copy
import hashlib
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(refined, initialization, output, *, refined_sha256, initialization_sha256):
    refined, initialization, output = map(Path, (refined, initialization, output))
    if output.exists() or output.resolve() in (refined.resolve(), initialization.resolve()):
        raise FileExistsError(output)
    if digest(refined) != refined_sha256 or digest(initialization) != initialization_sha256:
        raise ValueError('Encoder initialization input hash mismatch')
    base = torch.load(refined, map_location='cpu', weights_only=False)
    initial = torch.load(initialization, map_location='cpu', weights_only=False)
    candidate = copy.deepcopy(base)
    changed = []
    for name, tensor in base['encoder_state_dict'].items():
        source = initial['model_state_dict']['encoder.' + name]
        if source.shape != tensor.shape or source.dtype != tensor.dtype or not torch.isfinite(source).all():
            raise ValueError('Incompatible encoder initialization: ' + name)
        # The encoder's sole persistent nonparameter buffer is a fixed physical scale.
        if name == 'klm_encoder.input_scales':
            if not torch.equal(source, tensor):
                raise ValueError('Physical encoder scales differ')
            continue
        if not torch.equal(source, tensor):
            changed.append(name)
        candidate['encoder_state_dict'][name] = source.clone()
        candidate['model_state_dict']['encoder.' + name] = source.clone()
    if not changed:
        raise ValueError('Ablation has no changed encoder parameters')
    for name, tensor in base['model_state_dict'].items():
        if not name.startswith('encoder.'):
            assert torch.equal(tensor, candidate['model_state_dict'][name])
    candidate['step'] = 0
    candidate['epoch'] = 0
    candidate['metrics'] = {}
    for name in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict','training_state','random_states','schedule_state','streaming_cursor'):
        candidate.pop(name, None)
    provenance = {'version':'phase68-encoder-parameter-ablation-v1',
        'refined_checkpoint_sha256':refined_sha256,
        'initialization_checkpoint_sha256':initialization_sha256,
        'changed_encoder_keys':changed, 'encoder_entries':len(base['encoder_state_dict']),
        'shared_pid_and_nonencoder_state':True, 'train_normalizer_preserved':True,
        'additional_pretraining_steps':0, 'parameter_only_not_resumable':True,
        'interpretation':'Incremental Phase64 encoder refinement at fixed refined PID and normalization; not pretraining versus no pretraining.'}
    candidate['phase68_encoder_ablation'] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        torch.save(candidate, f)
    assert digest(refined) == refined_sha256 and digest(initialization) == initialization_sha256
    return {**provenance, 'checkpoint_sha256':digest(output)}
