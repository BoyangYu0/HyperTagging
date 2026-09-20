"""Matched, parameter-only pretraining refinement before Phase55 reconstruction."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
from scripts.run_reconstruction_phase35 import atomic_json, finite_checkpoint, sha256


def validate_pretraining_contract(contract, prereg):
    expected = dict(prereg['pretraining_refinement'])
    arm = next(a for a in prereg['arms'] if a['role'] == contract['arm_role'])
    expected['parent_ranking_weight'] = arm['pretraining_parent_ranking_weight']
    if contract.get('pretraining_refinement') != expected:
        raise RuntimeError('Phase55 pretraining refinement differs from preregistration')
    c = expected['config']
    if (expected['parent_ranking_weight'] != (2.0 if contract['arm_role'] == 'stronger_parent_pretraining' else 1.0)
        or c['max_steps'] != 2188 or c['batch_size'] != 32
        or c['lr_schedule_total_steps'] != 2188 or c['seed'] != 20260922
        or c['curriculum_phase_steps'] != [547] * 4
        or c['truth_guided_structural_relation_inputs'] or c['resume']
        or c.get('weights_initialization_checkpoint') or c.get('validation_event_uids')
        or c['validation_events'] != 1000 or c['num_workers'] != 0
        or not c['scientific_mode'] or not c['pilot_objective_preflight']
        or c['pilot_objective_violation_action'] != 'fail'
        or expected['checkpoint_for_reconstruction'] != 'fixed_final_step_2188'
        or expected['initialization'] != 'parameters_only_fresh_train_normalization_optimizer_schedule_rng_and_memory'):
        raise RuntimeError('Phase55 matched pretraining budget or isolation changed')


def refinement_config(contract, runtime, cohort, output):
    c = dict(contract['pretraining_refinement']['config'])
    c.update(data=runtime['selection_manifest'], dataset_index=runtime['dataset_index'],
             output_dir=str(output), weights_initialization_checkpoint=runtime['checkpoint'],
             validation_event_uids=tuple(cohort['checkpoint_selection_event_uids'][:1000]),
             parent_ranking_weight=contract['pretraining_refinement']['parent_ranking_weight'])
    for key in ('curriculum', 'curriculum_phase_steps', 'curriculum_phase_events', 'leaf_pid_phase_weights', 'validation_views', 'n_queries_by_level', 'max_cardinality_by_level'):
        if key in c:
            c[key] = tuple(tuple(v) if isinstance(v, list) else v for v in c[key])
    return PretrainConfig(**c)


def run_pretraining_refinement(contract, runtime, cohort, output: Path):
    if output.exists():
        raise RuntimeError('Refusing to overwrite Phase55 pretraining refinement')
    config = refinement_config(contract, runtime, cohort, output)
    source = Path(runtime['checkpoint'])
    before = sha256(source)
    result = train_hyperbolic_pretraining(config)
    evidence = finite_checkpoint(result.checkpoint)
    payload = torch.load(result.checkpoint, map_location='cpu', weights_only=False)
    if (result.steps != 2188 or evidence['step'] != 2188 or not evidence['all_checkpoint_tensors_finite']
        or before != sha256(source)
        or payload['validation_selection']['event_uids'] != list(config.validation_event_uids)
        or payload['config']['parent_ranking_weight'] != config.parent_ranking_weight):
        raise RuntimeError('Phase55 pretraining refinement budget, lineage or cohort failed')
    rows = [json.loads(line) for line in result.log_path.read_text().splitlines()]
    training = [r for r in rows if r.get('split') != 'validation' and 'loss' in r]
    if len(training) != 2188 or any(r.get('pretraining_parent_ranking_weight') != config.parent_ranking_weight for r in training):
        raise RuntimeError('Phase55 pretraining history is incomplete')
    audit = {'status': 'COMPLETED', 'checkpoint': str(result.checkpoint.resolve()),
             'checkpoint_sha256': sha256(result.checkpoint), 'step': result.steps,
             'source_checkpoint_sha256': before, 'source_unchanged': True,
             'parameter_initialization': json.loads((output/'parameter-initialization.json').read_text()),
             'config': asdict(config), 'training_log_sha256': sha256(result.log_path),
             'training_log_records': len(training), 'all_checkpoint_tensors_finite': True,
             'validation_selection_exact': True, 'checkpoint_selection': 'fixed_final_step_2188'}
    atomic_json(output/'refinement-result.json', audit)
    return audit
