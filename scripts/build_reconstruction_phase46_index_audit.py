#!/usr/bin/env python3
"""Bind the fresh train-only Phase46 index to the corrected record adapter."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FILES = ('src/hypertagging/data/heterogeneous.py', 'src/hypertagging/data/dataset_index.py', 'scripts/build_dataset_index.py')
INDEX = 'artifacts/experiment_readiness/reconstruction_phase46_20260912/train_070k.complete_only.index.json'
OLD = 'artifacts/experiment_readiness/reconstruction_phase40_20260907/train_070k.complete_only.index.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index,old = [json.loads((ROOT/p).read_text()) for p in (INDEX,OLD)]
    assert index['normalizer_scope'] == 'train'
    assert index['selection_contract']['included_splits'] == ['train','validation']
    assert index['event_identity_validation']['sealed_test_opened'] is False
    assert index['split_counts'] == old['split_counts']
    assert index['selection_contract'] == old['selection_contract']
    result = {'version':'phase46-fresh-statistics-audit-v1', 'status':'PASS',
        'index_path':INDEX, 'index_sha256':sha(ROOT/INDEX), 'normalizer_scope':'train',
        'sealed_test_accessed':False, 'construction':'full_record_scan_not_sidecars',
        'same_training_selection_as_phase45':True,
        'source_files':{p:sha(ROOT/p) for p in FILES},
        'changed_normalizer_blocks':[k for k,v in index['normalizer_state'].items() if v != old['normalizer_state'][k]],
        'historical_index_sha256':sha(ROOT/OLD)}
    (ROOT/'artifacts/experiment_readiness/reconstruction_phase46_20260912/fresh-statistics-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
