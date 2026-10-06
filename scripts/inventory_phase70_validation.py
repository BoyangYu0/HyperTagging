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
ARMS = ('type_bias', 'no_type_bias')
VERSION = 'phase70-policy-reevaluation-v1-20261006'


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
    registry = ROOT/'configs/reconstruction/supplementary_validation_reservations.json'
    inputs.append(binding(registry))
    for reservation in load(registry)['reservations']:
        bound = reservation['cohort_manifest']
        path = Path('/project/agkuhr/users/boyang/data/HyperTagging_artifacts')/bound['artifact_relative_path']
        assert sha256(path) == bound['sha256']
        doc = load(path); uids = set(doc['event_uids'])
        assert len(uids) == reservation['reserved_count'] and not uids & excluded
        excluded.update(uids); inputs.append(binding(path))
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
    return coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-receipts',type=Path,required=True)
    a=p.parse_args(); result=prepare(a.output.resolve(),a.historical_receipts)
    print(json.dumps({k:result[k] for k in ('status','available_by_category','shortfall_by_category')}))
if __name__=='__main__':main()
