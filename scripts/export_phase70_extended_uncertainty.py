"""Additional paired collision uncertainty for coherent pairs and continuum recovery.

Postprocess fixed reports only. No inference, ranking, fitting or resampling of
eligible events changes; the original supplemental aggregate remains immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.export_phase70_policy import paired_intervals


def event_ratios(event):
    values={}
    for scope,record in event['scopes'].items():
        tag=record['tag_efficiency']['greedy']
        for kind,item in [('exact',tag),('inclusive',tag['inclusive_fsp_grouping'])]:
            ratio=item['pool_at_k']['1'].get('b_reconstruction',{}).get('coherent_event_both_correct',{'numerator':0,'denominator':0})
            values[scope+'/'+kind+'/coherent_pair']=[ratio['numerator'],ratio['denominator']]
        for category in ('pooled_continuum','ccbar','uubar','ddbar','ssbar'):
            selected=category=='pooled_continuum' or event['source_category']==category
            for kind,item in [('exact',tag),('inclusive',tag['inclusive_fsp_grouping'])]:
                ratio=(item.get('continuum') or {}).get('retained_component_top1_correct') if selected else None
                values[f'{scope}/{category}/{kind}_component_recovery']=[ratio['numerator'],ratio['denominator']] if ratio else [0,0]
            for metric in ('top1_fake_b_slot_acceptance','top1_fake_b_event_acceptance'):
                ratio=(tag.get('continuum') or {}).get(metric) if selected else None
                values[f'{scope}/{category}/{metric}']=[ratio['numerator'],ratio['denominator']] if ratio else [0,0]
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.output/'plan.json').read_text());cohort=json.loads(Path(plan['cohort']['path']).read_text())
    expected={u:c for c,us in cohort['by_category'].items() for u in us}
    points={arm:{} for arm in plan['arms']};bindings=[]
    for i,task in enumerate(plan['tasks']):
        if task['view']!='greedy':continue
        folder=a.output/'tasks'/f'{i:03d}';raw=(folder/'report.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
        receipt=json.loads((folder/'receipt.json').read_text());assert receipt['returncode']==0 and receipt['report_sha256']==digest
        report=json.loads(raw)
        for event in report['events']:
            uid=event['event_uid'];assert expected[uid]==event['source_category'] and uid not in points[task['arm']]
            points[task['arm']][uid]=event_ratios(event)
        bindings.append({'task':i,'report_sha256':digest})
    result={'version':'phase70-additional-paired-uncertainty-v1','exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cohort_sha256':plan['cohort']['sha256'],'report_bindings':bindings,
            'bootstrap_source_sha256':hashlib.sha256((ROOT/'scripts/export_phase70_policy.py').read_bytes()).hexdigest(),
            'base_aggregate_sha256':hashlib.sha256((a.output/'aggregate.json').read_bytes()).hexdigest(),
            'uncertainty':paired_intervals(points['type_bias'],points['no_type_bias'],expected)}
    with (a.output/'extended-paired-uncertainty.json').open('x') as f:json.dump(result,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')
    print(json.dumps({'metrics':len(result['uncertainty']['metrics']),'reports':len(bindings)}))


if __name__=='__main__':main()
