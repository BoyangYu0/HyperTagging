"""Authenticate, reserve and execute the separately versioned Phase69 policy study.

Raw UIDs, checkpoints and per-event reports belong in the supplied project-volume
output directory. This command never submits jobs or changes training contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.build_reconstruction_phase35_evaluation_cohort import (
    sha256, uid_sequence_sha256, uid_set_sha256, atomic_json,
)

CATEGORIES = ('charged', 'mixed', 'ccbar', 'uubar', 'ddbar', 'ssbar')
ARMS = ('refined_encoder', 'pre_refinement_encoder')
VERSION = 'phase69-policy-reevaluation-v1-20261006'


def load(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(Path(path))}


def choose_cohort(category_by_uid, excluded, quota=2000):
    """Rank identities only; never condition on truth or reconstruction success."""
    pools = {cat: [] for cat in CATEGORIES}
    for uid, cat in category_by_uid.items():
        if uid not in excluded and cat in pools:
            pools[cat].append(uid)
    counts = {cat: len(rows) for cat, rows in pools.items()}
    if any(n < quota for n in counts.values()):
        return counts, None
    selected = {cat: sorted(rows, key=lambda uid: (
        hashlib.sha256(f'{VERSION}:{uid}'.encode()).hexdigest(), uid))[:quota]
        for cat, rows in pools.items()}
    return counts, selected


def manifest(uids):
    return {'manifest_version': 'hypertagging-reconstruction-evaluation-cohort-v1',
            'role': 'validation', 'sealed_test_role_access': 'forbidden',
            'event_uid_count': len(uids), 'event_uids': uids,
            'event_uids_sha256': uid_sequence_sha256(uids)}


def prepare(out, historical_receipts):
    import pyarrow.parquet as pq
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise ValueError('Reservation already exists; do not overwrite')
    cohort_path = ROOT / 'configs/reconstruction/ht_reconstruction_phase70_validation_cohort_20261005.json'
    c = load(cohort_path)
    inputs = [binding(cohort_path)]
    for item in c['source_bindings'].values():
        p = ROOT / item['path']
        assert sha256(p) == item['sha256'], p
        inputs.append(binding(p))
    ledger = load(ROOT / c['source_bindings']['previous_validation_universe']['path'])
    assert uid_set_sha256(ledger['event_uids']) == ledger['event_uids_sha256']
    excluded = set(ledger['event_uids']) | set(c['event_uids']) | set(c['checkpoint_selection_event_uids'])
    assert len(excluded) == 61000
    universe = load(ROOT / c['source_bindings']['validation_universe']['path'])
    assert uid_set_sha256(universe['event_uids']) == universe['event_uids_sha256']
    selection, index_path = ROOT / c['selection_manifest'], ROOT / c['dataset_index']
    assert sha256(selection) == c['selection_manifest_sha256']
    assert sha256(index_path) == c['dataset_index_sha256']
    m, index = load(selection), load(index_path)
    assert not m['selection_includes_test'] and index['normalizer_scope'] == 'train'
    inputs.extend([binding(selection), binding(index_path), binding(historical_receipts)])
    categories, train, shards = {}, set(), []
    source_roles = {}
    for entry in m['entries']:
        role = entry['split']
        assert role in ('train', 'validation')
        source = entry['source_file']
        assert source_roles.setdefault(source, role) == role
        assert index['source_groups'][source] == role
        path = Path(m['data_root']) / entry['path']
        b = binding(path)
        assert b['sha256'] == entry['parquet_sha256_reference'], path
        shards.append({**b, 'split': role, 'category': entry['category']})
        table = pq.read_table(path, columns=['event_uid', 'source_category', 'source_file'], use_threads=False)
        assert table.num_rows == entry['event_count']
        for row in table.to_pylist():
            uid = row['event_uid']
            assert row['source_category'] == entry['category'] and row['source_file'] == source
            assert uid not in train and uid not in categories, 'Duplicate collision identity'
            if role == 'train':
                train.add(uid)
            else:
                categories[uid] = row['source_category']
    assert set(categories) == set(universe['event_uids']) and len(train) == 70000
    assert excluded <= set(categories) and not train & set(categories)
    counts, selected = choose_cohort(categories, excluded)
    coverage = {'version': VERSION, 'status': 'READY' if selected else 'BLOCKED_SHORTAGE',
                'validation_total': len(categories), 'prior_reserved': len(excluded),
                'unreserved_total': len(set(categories) - excluded),
                'required_per_category': 2000, 'available_by_category': counts,
                'validation_by_category': dict(Counter(categories.values())),
                'shortfall_by_category': {cat: max(0, 2000-counts[cat]) for cat in CATEGORIES},
                'selection_uses_truth': False, 'sealed_test_accessed': False,
                'source_bindings': inputs, 'shards': shards,
                'training_overlap': 0, 'prior_reservation_overlap': 0}
    atomic_json(out / 'coverage-authentication.json', coverage)
    if selected is None:
        raise SystemExit(2)
    flat = [uid for cat in CATEGORIES for uid in selected[cat]]
    atomic_json(out / 'cohort.json', {**manifest(flat), 'by_category': selected,
                'exclusion_uid_set_sha256': uid_set_sha256(excluded), 'version': VERSION})
    receipts = load(historical_receipts)
    arms = {}
    for arm in ARMS:
        r = next(r for r in receipts if r['arm'] == arm and r['view'] == 'primary_complete_target_direct')
        command = r['evaluations']['additive']['command']
        old_report = Path(r['evaluations']['additive']['report'])
        assert sha256(old_report) == r['evaluations']['additive']['report_sha256']
        checkpoints = {flag: binding(command[command.index('--'+flag)+1])
                       for flag in ('pretraining-checkpoint', 'reconstruction-checkpoint')}
        run = Path(checkpoints['reconstruction-checkpoint']['path']).parent.parent
        native = run.parents[3] / 'slurm/reconstruction-phase69/jobs' / run.name / 'attempt-00/receipt.json'
        from scripts.build_reconstruction_phase69_closeout import verify_receipt
        native_receipt = load(native)
        verify_receipt(native_receipt)
        pair_path = out / f'{arm}-checkpoint-pair.json'
        cmd = [sys.executable, str(ROOT/'scripts/validate_reconstruction_checkpoint_pair.py')]
        for flag, b in checkpoints.items():
            cmd += ['--'+flag, b['path']]
        cmd += ['--allow-finetuned-encoder', '--output', str(pair_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        pair = load(pair_path)
        assert pair['compatible'] and pair['reconstruction_step'] == 4000
        historical_pair = load(old_report)['checkpoint_pair']
        for kind in ('pretraining', 'reconstruction'):
            assert checkpoints[kind+'-checkpoint']['sha256'] == pair[kind+'_sha256'] == historical_pair[kind+'_sha256']
        source = run.parents[4]
        contract = load(run/'provenance/submitted-contract.json')
        canonical_contract = {k:v for k,v in contract.items() if k != 'contract_sha256'}
        assert hashlib.sha256(json.dumps(canonical_contract,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest() == contract['contract_sha256'] == native_receipt['contract_sha256']
        assert contract['arm_role'] == arm and contract['sealed_test_role_access'] == 'forbidden'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip() == contract['expected_git_sha']
        subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts'],cwd=source,check=True,stdout=subprocess.DEVNULL)
        arms[arm] = {**checkpoints, 'native_receipt': binding(native),
                     'historical_report': binding(old_report), 'checkpoint_pair': binding(pair_path)}
    tasks = []
    for cat in CATEGORIES:
        for part in range(4):
            uids = selected[cat][part*500:(part+1)*500]
            p = out / 'cohorts' / f'{cat}-{part}.json'
            atomic_json(p, manifest(uids))
            for arm in ARMS:
                tasks.append({'arm': arm, 'category': cat, 'view': 'greedy',
                              'cohort': binding(p), 'count': len(uids)})
    # A fixed, category-balanced 60-collision beam diagnostic; never quota compliant.
    beam = [uid for cat in CATEGORIES for uid in selected[cat][:10]]
    p = out / 'cohorts/beam-diagnostic.json'
    atomic_json(p, manifest(beam))
    for arm in ARMS:
        tasks.append({'arm': arm, 'category': 'stratified', 'view': 'beam_diagnostic',
                      'cohort': binding(p), 'count': len(beam)})
    atomic_json(out / 'plan.json', {'version': VERSION, 'cohort': binding(out/'cohort.json'),
                'coverage': binding(out/'coverage-authentication.json'), 'arms': arms,
                'selection': binding(selection), 'index': binding(index_path), 'tasks': tasks,
                'evaluator': binding(ROOT/'scripts/evaluate_full_decay.py'),
                'source_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})


def run_task(out, task_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Substantial evaluation requires a CPU Slurm allocation')
    plan = load(out/'plan.json'); task = plan['tasks'][task_id]
    subprocess.run(['git','diff','--exit-code',plan['source_head'],'--','src','scripts/evaluate_full_decay.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    for b in (task['cohort'], plan['selection'], plan['index'], plan['evaluator']):
        assert sha256(Path(b['path'])) == b['sha256']
    dest = out / 'tasks' / f'{task_id:03d}'
    dest.mkdir(parents=True, exist_ok=True)
    receipt_path = dest/'receipt.json'
    if receipt_path.exists():
        previous = load(receipt_path)
        if previous['returncode'] == 0 and sha256(dest/'report.json') == previous['report_sha256']:
            return
        raise RuntimeError('Prior failed task requires an explicit separate recovery attempt')
    cmd = [sys.executable, str(ROOT/'scripts/evaluate_full_decay.py')]
    for flag in ('pretraining-checkpoint','reconstruction-checkpoint'):
        b = plan['arms'][task['arm']][flag]
        assert sha256(Path(b['path'])) == b['sha256']
        cmd += ['--'+flag,b['path']]
    cmd += ['--data',plan['selection']['path'],'--dataset-index',plan['index']['path'],
            '--split','validation','--event-uid-manifest',task['cohort']['path'],
            '--scope','both','--truth-topology-mode','checkpoint_direct',
            '--max-events',str(task['count']),'--max-level','6','--object-threshold','0.6',
            '--pointer-threshold','0.35','--threads','1','--omit-trees',
            '--allow-finetuned-encoder','--deterministic-algorithms','--output',str(dest/'report.json')]
    if task['view'] == 'beam_diagnostic':
        cmd += ['--beam-search','--beam-width','2','--beam-max-candidate-expansions-per-query','32',
                '--beam-max-proposals-per-level','8','--beam-max-candidates-per-query','2']
    receipt = {'task': task, 'command': cmd, 'job_id': os.environ['SLURM_JOB_ID'],
               'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'), 'status': 'RUNNING'}
    atomic_json(dest/'running.json',receipt)
    with (dest/'evaluation.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(returncode=result.returncode,status='COMPLETED' if result.returncode == 0 else 'FAILED')
    if result.returncode == 0:
        receipt['report_sha256'] = sha256(dest/'report.json')
    atomic_json(receipt_path,receipt)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-receipts',type=Path);p.add_argument('--task-id',type=int)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.output.resolve(),a.historical_receipts)
    else: run_task(a.output.resolve(),a.task_id)

if __name__=='__main__': main()
