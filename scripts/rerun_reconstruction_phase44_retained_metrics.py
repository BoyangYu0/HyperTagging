#!/usr/bin/env python3
"""Rerun all Phase44 views with a frozen, additive retained-tree evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch

ARMS = ('late_adaptation_control', 'early_adaptation')
TRACKS = {
    'primary_complete_target_direct': 'best.pt',
    'independent_depth_direct': 'best_rollout_depth_fraction.pt',
    'independent_tree_validity_direct': 'best_rollout_tree_validity.pt',
    'primary_complete_target_beam_direct': 'best.pt',
    'primary_complete_target_repeat2_direct': 'best.pt',
    'independent_complete_target_direct': 'best_rollout_complete_target_efficiency.pt',
    'primary_complete_target_contracted_diagnostic': 'best.pt',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--evaluator-source', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.evaluator_source.resolve(strict=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise ValueError('Retained evaluator source must be clean and frozen')
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / 'provenance.json').write_text(json.dumps({
        'evaluator_revision': revision, 'legacy_root': str(args.legacy_root.resolve()),
        'classification': 'POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION',
        'original_jobs_remain_failed': True, 'sealed_test_accessed': False,
        'maximum_concurrent_cpu_processes': 4,
    }, indent=2) + '\n')
    plans = []
    for name, checkpoint in TRACKS.items():
        for arm in ARMS:
            old = args.legacy_root / arm / 'full-decay-reports'
            base = json.loads((old / 'primary_complete_target_direct.log.json').read_text())['command']
            command = list(base)
            training = Path(command[command.index('--reconstruction-checkpoint') + 1]).parent
            command[command.index('--reconstruction-checkpoint') + 1] = str(training / checkpoint)
            result = json.loads((args.legacy_root / arm / 'training-result.json').read_text())
            track = {'best.pt': 'best', 'best_rollout_depth_fraction.pt': 'best_depth', 'best_rollout_tree_validity.pt': 'best_tree_validity', 'best_rollout_complete_target_efficiency.pt': 'best_complete_target'}[checkpoint]
            assert hashlib.sha256((training / checkpoint).read_bytes()).hexdigest() == result['checkpoints'][track]['sha256']
            step = result['checkpoints'][track]['step']
            freeze_steps = torch.load(training / checkpoint, map_location='cpu', weights_only=False)['config']['freeze_pretrained_encoder_steps']
            if step <= freeze_steps and '--allow-finetuned-encoder' in command:
                command.remove('--allow-finetuned-encoder')
            if name.endswith('contracted_diagnostic'):
                command[command.index('--truth-topology-mode') + 1] = 'contracted_diagnostic'
            output = args.output_root / arm
            output.mkdir(exist_ok=True)
            if name.endswith('beam_direct'):
                cohort = json.loads((old / 'evaluation-cohort.json').read_text())
                cohort['event_uids'] = cohort['event_uids'][:20]
                cohort['event_uid_count'] = 20
                cohort['event_uids_sha256'] = hashlib.sha256(('\n'.join(cohort['event_uids']) + '\n').encode()).hexdigest()
                # Use the native cohort digest implementation below at execution.
                manifest = output / 'beam-evaluation-cohort.json'
                import sys
                sys.path.insert(0, str(source))
                from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256
                cohort['event_uids_sha256'] = uid_sequence_sha256(cohort['event_uids'])
                manifest.write_text(json.dumps(cohort, indent=2) + '\n')
                command[command.index('--event-uid-manifest') + 1] = str(manifest.resolve())
                command[command.index('--max-events') + 1] = '20'
                command += ['--beam-width', '4', '--beam-max-events', '20', '--beam-max-proposals', '12']
            command[command.index('--output') + 1] = str((output / f'{name}.json').resolve())
            plans.append((arm, name, command, output))
    (args.output_root / 'commands.json').write_text(json.dumps([{'arm': arm, 'view': name, 'command': cmd} for arm, name, cmd, _ in plans], indent=2) + '\n')

    def run(plan):
        arm, name, command, output = plan
        log = output / f'{name}.log'
        print(f'Starting {arm}/{name}', flush=True)
        with log.open('w') as handle:
            result = subprocess.run(command, cwd=source, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, timeout=7200)
        if result.returncode:
            raise RuntimeError(f'{arm}/{name} failed; see {log}')
        print(f'Completed {arm}/{name}', flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, plan) for plan in plans]
        for future in as_completed(futures):
            future.result()
    print('All 14 retained-tree evaluations completed.', flush=True)


if __name__ == '__main__':
    main()
