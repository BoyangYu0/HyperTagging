#!/usr/bin/env python3
"""Export all recorded Phase44 checkpoint and training-log scalar metrics."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase41_closeout import numeric_rows, digest  # noqa: E402
from scripts.build_reconstruction_phase44_closeout import ARMS, verify_receipt  # noqa: E402
from scripts.run_reconstruction_phase35 import finite_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'classification': 'POST_HOC_DIAGNOSTIC_WITH_PREREGISTRATION_SEED_DEVIATION',
                'original_job_status': 'FAILED', 'sealed_test_accessed': False,
                'checkpoint_aliases_are_separate_records_not_independent_trials': True,
                'evaluation_scope': 'recorded scalar metrics only; additional checkpoints are not additional strict evaluations',
                'arms': {}}
    fields = ['arm', 'record_kind', 'record', 'step', 'metric', 'value']
    with gzip.open(out/'phase44-training-history.csv.gz', 'wt', newline='') as csv_file, gzip.open(out/'phase44-training-history.jsonl.gz', 'wt') as json_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        def emit(arm, kind, record, step, values):
            count = 0
            for row in numeric_rows(values):
                item = dict(arm=arm, record_kind=kind, record=record, step=step, **row)
                writer.writerow(item)
                json_file.write(json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
                count += 1
            return count
        for arm, job in ARMS.items():
            receipt = source/f'artifacts/slurm/reconstruction-phase44/jobs/{job}/attempt-00/receipt.json'
            verify_receipt(json.loads(receipt.read_text()))
            training = source/f'artifacts/runs/ht-reconstruction-phase44-20260911/{arm}/{job}/training'
            log = training/'metrics.jsonl'
            record = {'receipt_sha256': digest(receipt), 'training_log_sha256': digest(log), 'log_records': 0, 'log_scalar_rows': 0, 'checkpoint_scalar_rows': 0, 'checkpoints': {}}
            for index, line in enumerate(log.read_text().splitlines(), 1):
                value = json.loads(line)
                record['log_scalar_rows'] += emit(arm, 'training_log', index, value.get('step'), value)
                record['log_records'] += 1
            for checkpoint in sorted(training.glob('*.pt')):
                evidence = finite_checkpoint(checkpoint)
                assert evidence['all_checkpoint_tensors_finite']
                record['checkpoint_scalar_rows'] += emit(arm, 'checkpoint', checkpoint.name, evidence['step'], evidence['metrics'])
                record['checkpoints'][checkpoint.name] = {k:evidence[k] for k in ('step','sha256','all_checkpoint_tensors_finite')}
            manifest['arms'][arm] = record
    manifest['total_scalar_rows'] = sum(r['log_scalar_rows']+r['checkpoint_scalar_rows'] for r in manifest['arms'].values())
    manifest['exports'] = {p.name: {'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.glob('*.gz'))}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'scalar_rows':manifest['total_scalar_rows'], 'arms':{a:{k:v for k,v in r.items() if k in ('log_records','log_scalar_rows','checkpoint_scalar_rows')} for a,r in manifest['arms'].items()}}))


if __name__ == '__main__':
    main()
