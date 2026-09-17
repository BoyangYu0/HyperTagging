#!/usr/bin/env python3
"""Post-inference micro/macro and exact-source/PID/topology summaries."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.build_reconstruction_phase52_closeout import ARMS
from scripts.build_reconstruction_phase41_closeout import digest,numeric_rows

METRICS=('source_recall','source_precision','lcag_pair_accuracy','mother_pid_coverage','leaf_pid_accuracy','root_pid_accuracy','perfectLCAG')


def summarize(events):
    """Undefined denominators are unavailable; defined zero successes stay failures."""
    output={}
    rows=[r for event in events for r in event['rows']]
    for key in METRICS:
        def counts(rs):
            return sum(r[key+'_numerator'] for r in rs),sum(r[key+'_denominator'] for r in rs)
        num,den=counts(rows)
        units=[r[key+'_numerator']/r[key+'_denominator'] for r in rows if r[key+'_denominator']>0]
        ev=[n/d for event in events for n,d in [counts(event['rows'])] if d>0]
        output[key]={'micro_numerator':num,'micro_denominator':den,'micro':num/den if den else None,
            'unit_macro_sum':sum(units),'unit_macro_denominator':len(units),'unit_macro':sum(units)/len(units) if units else None,
            'unit_macro_unavailable':len(rows)-len(units),'event_macro_sum':sum(ev),'event_macro_denominator':len(ev),
            'event_macro':sum(ev)/len(ev) if ev else None,'event_macro_unavailable':len(events)-len(ev)}
    output['exact']={}
    for population in ('all_retained','nontrivial'):
        selected=[r for r in rows if population=='all_retained' or len(r['truth_sources'])>=2]
        numerators={k:0 for k in ('source','source_leaf_pid','source_topology','source_topology_all_pid')}
        for r in selected:
            source=bool(r['available'] and r['target_representable'] and r['structurally_valid'] and r['truth_sources'] and sorted(r['truth_sources'])==sorted(r['predicted_sources']))
            leaf=bool(r['leaf_pid_unavailable_count']==0 and r['leaf_pid_accuracy_numerator']==r['leaf_pid_accuracy_denominator']==r['truth_leaf_count'] and r['truth_leaf_count']>0)
            topology=bool(r['perfectLCAG_numerator'])
            all_pid=bool(leaf and r['mother_pid_coverage_numerator']==r['truth_mother_count']==r['predicted_mother_count'] and r['root_pid_accuracy_denominator']>0 and r['root_pid_accuracy_numerator']==r['root_pid_accuracy_denominator'])
            for k,v in [('source',source),('source_leaf_pid',source and leaf),('source_topology',topology),('source_topology_all_pid',topology and all_pid)]:numerators[k]+=int(v)
        output['exact'][population]={'unit_count':len(selected),'unavailable':sum(not r['available'] for r in selected),**{k:{'numerator':n,'denominator':len(selected),'value':n/len(selected) if selected else None} for k,n in numerators.items()}}
    return output


def build(source):
    payload={'version':'phase52-aggregation-supplement-v1','status':'COMPLETE','source_hashes':[],'metric_rows':[]}
    for arm,job in ARMS.items():
        for path in sorted((source/f'artifacts/runs/ht-reconstruction-phase52-20260917/{arm}/{job}/full-decay-reports').glob('*.json')):
            report=json.loads(path.read_text())
            if 'summaries' not in report:continue
            payload['source_hashes'].append(digest(path))
            for scope in ('full','half'):
                groups={'greedy':[e['scopes'][scope]['retained_tree_metrics'] for e in report['events']]}
                if report['beam_search']['events']:
                    for ranking in ('average_link_probability','learned_confidence_mean','learned_confidence_sum','normalized_joint_log_probability','oracle_diagnostic'):
                        selected=[]
                        for e in report['beam_search']['events']:
                            index=e['retained_tree_oracle_indices_by_scope'][scope] if ranking=='oracle_diagnostic' else e['ranking_selections'][ranking]
                            candidate=next(c for c in e['candidates'] if c['candidate_index']==index)
                            selected.append(candidate['retained_tree_metrics_by_scope'][scope])
                        groups[ranking]=selected
                for rank,events in groups.items():
                    payload['metric_rows'].extend({'arm':arm,'view':path.stem,**row} for row in numeric_rows(summarize(events),scope+'.'+rank))
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--csv',type=Path,required=True);a=parser.parse_args();d=build(a.source_root)
    a.output.write_text(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    with a.csv.open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','view','metric','value']);w.writeheader();w.writerows(d['metric_rows'])
    with a.csv.open() as f:
        for x,y in zip(csv.DictReader(f),d['metric_rows'],strict=True):assert all(x[k]==('' if v is None else str(v)) for k,v in y.items())
    registry=[[r['arm'],r['view'],r['metric']] for r in d['metric_rows']];assert len(set(map(tuple,registry)))==len(registry)
    (ROOT/'docs/_ext/phase52_aggregation_metric_registry.json').write_text(json.dumps(registry,separators=(',',':'))+'\n');print('PASS',len(registry))


if __name__=='__main__':main()
