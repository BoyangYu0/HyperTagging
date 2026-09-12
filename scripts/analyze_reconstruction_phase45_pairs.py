#!/usr/bin/env python3
"""Paired event-cluster uncertainty and exact-component depth audit for Phase45."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reconstruction_phase45_closeout import ARMS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[json.loads((args.source_root/f'artifacts/runs/ht-reconstruction-phase45-20260911/{arm}/{job}/full-decay-reports/primary_complete_target_direct.json').read_text()) for arm,job in ARMS.items()]
    assert [e['event_uid'] for e in reports[0]['events']] == [e['event_uid'] for e in reports[1]['events']]
    rng=np.random.default_rng(20260912)
    samples=rng.integers(0,100,size=(10000,100))
    output={'version':'phase45-paired-event-bootstrap-v1','replicates':10000,'seed':20260912,'interval':'paired event-cluster percentile 95%, ratio of summed counts','limitation':'Conditional on two trained models; does not measure training-seed uncertainty. Degenerate all-zero intervals do not imply zero population uncertainty.','scopes':{},'perfect_components':{}}
    for scope in ('full','half'):
        output['scopes'][scope]={}
        for metric in ('lcag_pair_accuracy','mother_pid_coverage','source_recall','source_precision'):
            counts=np.array([[[sum(row[metric+'_'+part] for row in event['scopes'][scope]['retained_tree_metrics']['rows']) for part in ('numerator','denominator')] for event in report['events']] for report in reports],dtype=float)
            draws=counts[:,samples,:].sum(axis=2)
            delta=draws[1,:,0]/draws[1,:,1]-draws[0,:,0]/draws[0,:,1]
            total=counts.sum(axis=1)
            output['scopes'][scope][metric]={'candidate_minus_control':float(total[1,0]/total[1,1]-total[0,0]/total[0,1]),'ci95':np.quantile(delta,[.025,.975]).tolist(),'paired_event_counts_identical':bool(np.array_equal(counts[0],counts[1]))}
        for arm,report in zip(ARMS,reports,strict=True):
            shapes=Counter()
            for event in report['events']:
                for row in event['scopes'][scope]['retained_tree_metrics']['rows']:
                    if row.get('perfectLCAG_numerator',0):
                        shapes[(row['truth_leaf_count'],row['truth_mother_count'],row['truth_retained_depth'])]+=1
            output['perfect_components'][arm+'/'+scope]=[{'leaves':k[0],'mothers':k[1],'depth':k[2],'count':v} for k,v in sorted(shapes.items())]
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(output['scopes']))


if __name__=='__main__':main()
