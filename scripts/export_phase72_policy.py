"""Export authenticated supplementary Phase72 aggregates without event records.

Uses shared tag aggregation and sums tree sufficient statistics before division.
Only bootstrap resample batches, never all replicate-by-event arrays, are held.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.phase72_policy_evaluation import ARMS, CATEGORIES, VERSION, load, sha256, atomic_json
from hypertagging.evaluation.tag_efficiency import summarize_tag_efficiency_events, validate_tag_efficiency_report
from hypertagging.evaluation.retained_tree_checks import validate_retained_tree_report


def merge_statistics(rows, path=()):
    """Merge shared tree/inference summaries, rejecting unknown list semantics."""
    rows = list(rows)
    first = rows[0]
    if isinstance(first, dict):
        if {'numerator','denominator','value'} <= first.keys():
            n = sum(row['numerator'] for row in rows)
            d = sum(row['denominator'] for row in rows)
            value = n/d if d else None
            if path[-1].endswith('_rmse') and value is not None:
                value = math.sqrt(value)
            result = {'numerator':n,'denominator':d,'value':value}
            extras = set(first)-set(result)
            if extras == {'eligible'}:
                result['eligible'] = d>0
            elif extras:
                raise ValueError(f'Unknown ratio fields: {path}: {extras}')
            return result
        keys = sorted(set().union(*(row.keys() for row in rows)))
        return {key: merge_statistics([row[key] for row in rows if key in row],path+(key,)) for key in keys}
    if isinstance(first,list):
        if path[-1] not in ('leaf_pid_confusion','mother_pid_confusion'):
            raise ValueError(f'Unknown list semantics: {path}')
        count = Counter()
        for row in rows:
            for cell in row:
                count[cell['truth_token'],cell['predicted_token']] += cell['count']
        return [{'truth_token':a,'predicted_token':b,'count':n} for (a,b),n in sorted(count.items())]
    if type(first) in (int,float):
        return max(rows) if path[-1]=='maximum_p4_closure_residual' else sum(rows)
    if not all(row == first for row in rows):
        raise ValueError(f'Conflicting fixed metadata: {path}')
    return first


def event_ratios(event):
    values={}
    for scope,record in event['scopes'].items():
        for pop,metric in [('primary',record['metrics']),('retained',record['retained_tree_metrics'])]:
            rows=metric.get('rows',[metric])
            for name in ('lcag_pair_accuracy','perfectLCAG','source_precision','source_recall'):
                values[f'{scope}/{pop}/{name}'] = [sum(x[name+'_numerator'] for x in rows),sum(x[name+'_denominator'] for x in rows)]
        retained=record['retained_tree_metrics']
        for name in ('source_precision','source_recall','perfectLCAG'):
            rows=[row for row in retained['rows'] if row['truth_leaf_count']>=2 and row['truth_mother_count']>0]
            values[f'{scope}/nontrivial_retained/{name}']=[sum(row[name+'_numerator'] for row in rows),sum(row[name+'_denominator'] for row in rows)]
        forest=retained['coherent_retained_forest']
        values[f'{scope}/retained/coherent_forest']=[forest['numerator'],forest['denominator']]
        tag=record['tag_efficiency']['greedy']
        for pop,metrics in [('exact_tag',tag),('inclusive_group',tag['inclusive_fsp_grouping'])]:
            for name in ('per_b_correct','event_any_correct','event_both_correct'):
                ratio=metrics['top1'].get('b_reconstruction',{}).get(name,{'numerator':0,'denominator':0})
                values[f'{scope}/{pop}/{name}']=[ratio['numerator'],ratio['denominator']]
    return values


def paired_intervals(left,right,categories,repeats=10000,batch=16):
    """Stratified paired collision bootstrap with a bounded resample batch."""
    import numpy as np
    if set(left)!=set(right) or set(left)!=set(categories):
        raise ValueError('Paired collision identities differ')
    uids=sorted(left)
    names=sorted(left[uids[0]])
    a=np.asarray([[left[u][k] for k in names] for u in uids],dtype=float)
    b=np.asarray([[right[u][k] for k in names] for u in uids],dtype=float)
    rngs=[np.random.default_rng(seed) for seed in np.random.SeedSequence(20261007).spawn(len(CATEGORIES))]
    groups=[np.asarray([i for i,u in enumerate(uids) if categories[u]==cat]) for cat in CATEGORIES]
    estimates=np.full((repeats,len(names),2),np.nan)
    for start in range(0,repeats,batch):
        size=min(batch,repeats-start)
        sa=np.zeros((size,len(names),2)); sb=sa.copy()
        for group,rng in zip(groups,rngs):
            if not len(group): continue
            indices=group[rng.integers(0,len(group),size=(size,len(group)))]
            sa+=a[indices].sum(axis=1);sb+=b[indices].sum(axis=1)
        for j,data in enumerate((sa,sb)):
            np.divide(data[:,:,0],data[:,:,1],out=estimates[start:start+size,:,j],where=data[:,:,1]>0)
    totals=[a.sum(axis=0),b.sum(axis=0)]
    def ci(v):
        v=v[np.isfinite(v)]
        return {'percentile_95':np.quantile(v,[.025,.975]).tolist() if len(v) else None,
                'valid_resamples':len(v),'degenerate':bool(len(v) and v.min()==v.max())}
    result={}
    for i,name in enumerate(names):
        record={}
        points=[]
        for j,arm in enumerate(ARMS):
            n,d=totals[j][i];v=float(n/d) if d else None;points.append(v)
            record[arm]={'numerator':float(n),'denominator':float(d),'value':v,**ci(estimates[:,i,j])}
        record['overlap_on_minus_off']={'value':points[1]-points[0] if None not in points else None,**ci(estimates[:,i,1]-estimates[:,i,0])}
        result[name]=record
    return {'unit':'collision','stratified_by_source_category':True,'seed':20261007,'resamples':repeats,
            'resample_batch':batch,'numpy_version':np.__version__,'random_streams':'SeedSequence seed20261007 spawn6 category generators','metrics':result,
            'limitations':'Exploratory conditional comparison of two frozen fitted models; no training-seed uncertainty or multiplicity correction. Degenerate zero-success intervals do not prove zero population efficiency.'}


def aggregate(out):
    plan=load(out/'plan.json');cohort=load(plan['cohort']['path']);coverage=load(plan['coverage']['path'])
    for b in (plan['cohort'],plan['coverage'],plan['selection'],plan['index']):
        assert sha256(Path(b['path']))==b['sha256']
    expected={uid:cat for cat,uids in cohort['by_category'].items() for uid in uids}
    public={'version':VERSION,'status':'COMPLETE_GREEDY_WITH_DIAGNOSTIC_BEAM','exporter_sha256':sha256(Path(__file__)),
            'coverage':{k:v for k,v in coverage.items() if k not in ('shards','source_bindings')},
            'selected_checkpoint_steps':{arm:load(plan['arms'][arm]['checkpoint_pair']['path'])['reconstruction_step'] for arm in ARMS},'arms':{},'source_hashes':[],
            'historical_reports_replaced':False,'sealed_test_accessed':False,
            'physical_fei_comparison_ready':False,'cohort_sha256':plan['cohort']['sha256']}
    public['coverage']['views']={}
    public['coverage']['count_semantics']='Processed includes unavailable truth and unsuccessful reconstruction. Failed counts attempted collisions without completed evaluator output; invalid_rollout is a separate processed-event diagnostic. truth_unavailable refers to primary tree scope; retained and exact/inclusive availability are separate.'
    points={}; scientific_source=None
    final_views={arm:{scope:[] for scope in ('full','half')} for arm in ARMS}
    for arm in ARMS:
        tags=defaultdict(list);summaries=[];retained=[];category_summaries=defaultdict(list)
        guards={scope:Counter() for scope in ('full','half')}
        seen=set();counts={cat:{scope:{'requested':2000,'unique_attempted':0,'processed':0,'failed':0,
                                    'invalid_rollout':0,'truth_unavailable':0,'retained_tree_truth_unavailable':0,'exact_tag_truth_unavailable_events':0,
                                    'inclusive_membership_unavailable_events':0} for scope in ('full','half')} for cat in CATEGORIES}
        points[arm]={};sources=[];beam=None;shapes=Counter();forests=Counter()
        for i,task in enumerate(plan['tasks']):
            if task['arm']!=arm:continue
            folder=out/'tasks'/f'{i:03d}';receipt=load(folder/'receipt.json')
            assert receipt['returncode']==0 and receipt['task']==task
            assert sha256(folder/'report.json')==receipt['report_sha256']
            report=load(folder/'report.json')
            prov=report['evaluator_code_provenance']
            assert prov['provenance_complete'] and not prov['worktree_dirty']
            if scientific_source is None:scientific_source=prov['git_head']
            assert prov['git_head']==scientific_source
            assert report['device']=='cpu' and report['torch_num_threads']==1
            assert report['torch_deterministic_algorithms_enabled'] and report['checkpoint_pair']['compatible']
            policy={key:report['configuration'][key] for key in ('max_level','object_threshold','pointer_threshold','confidence_threshold','type_probability_threshold','use_cardinality','use_learned_confidence','confidence_head_trained','p4_closure_tolerance','rollout_pid_kinematics_mode','rollout_pid_temperature','truth_topology_mode')}
            policy['offline_inference_policy_version']=report['offline_inference_policy_version']
            policy['target_policy']=report['retained_tree_checks']['training_target_policy']
            if 'inference_policy' not in public:public['inference_policy']=policy
            assert policy==public['inference_policy'], 'Inference policy differs across arms/tasks'
            if 'runtime_environment' not in public:public['runtime_environment']=report['runtime_environment']
            assert public['runtime_environment']==report['runtime_environment']
            public['execution']={'device':'cpu','threads':1,'deterministic_algorithms':True}
            validate_tag_efficiency_report(report);validate_retained_tree_report(report)
            selected=load(task['cohort']['path'])['event_uids']
            uids=[e['event_uid'] for e in report['events']]
            assert len(uids)==len(set(uids))==task['count'] and set(uids)==set(selected)
            source={'task':i,'arm':arm,'view':task['view'],'report_sha256':receipt['report_sha256'],'cohort_sha256':task['cohort']['sha256']}
            sources.append(source)
            if task['view']=='beam_diagnostic':
                beam={key:report[key] for key in ('summaries','summaries_by_source_category','summaries_by_target_shape','retained_tree_checks','tag_efficiency','configuration')}
                # Configuration contains data/checkpoint provenance paths; publish only search settings.
                beam['configuration']={k:v for k,v in report['configuration'].items() if k in ('beam_search','scope','max_level','object_threshold','pointer_threshold','truth_topology_mode')}
                beam['coverage']={'status':'DIAGNOSTIC_NOT_QUOTA_COMPLIANT','processed_by_category':dict(Counter(e['source_category'] for e in report['events'])),'distinct_collisions':len(uids),'required_per_category':2000,'views':{}}
                for scope in ('full','half'):
                    for method in ('greedy','model_top1','retained_beam_pool'):
                        groups={}
                        for cat in CATEGORIES:
                            records=[e['scopes'][scope] for e in report['events'] if e['source_category']==cat]
                            groups[cat]={'requested':10,'unique_attempted':len(records),'processed':len(records),'failed':0,
                                         'truth_unavailable':sum(not r['metrics']['available'] for r in records),
                                         'invalid_rollout':sum((not all(c['inference']['rollout_event_valid'] for c in r['beam']['candidates'])) if method=='retained_beam_pool' else (not (r['inference'] if method=='greedy' else r['beam']['top1_inference'])['rollout_event_valid']) for r in records)}
                        beam['coverage']['views'][scope+'/'+method]=groups
                beam['candidate_accounting']={scope:{'returned_candidates':sum(e['scopes'][scope]['beam']['candidate_count'] for e in report['events']),
                    'collision_count':len(uids),'candidates_do_not_multiply_coverage':True} for scope in ('full','half')}
                beam['candidate_rank_scores']={};beam['implementation_guards']={}
                for scope in ('full','half'):
                    candidates=[c for e in report['events'] for c in e['scopes'][scope]['beam']['candidates']]
                    beam['candidate_rank_scores'][scope]={}
                    for rank in sorted({c['rank'] for c in candidates}):
                        ranked=[c for c in candidates if c['rank']==rank]
                        beam['candidate_rank_scores'][scope][str(rank)]={key:{'count':len(ranked),'sum':sum(c[key] for c in ranked),
                            'mean':sum(c[key] for c in ranked)/len(ranked),'minimum':min(c[key] for c in ranked),'maximum':max(c[key] for c in ranked)}
                            for key in ('score','log_score_sum','scored_candidate_count','scored_decision_count')}
                    beam['implementation_guards'][scope]={key:sum(c['inference'][key] for c in candidates)
                        for key in ('committed_forest_source_conflict_count','source_conflicting_mother_count','source_conflicting_detector_resource_count')}
                    beam['implementation_guards'][scope]['p4_closure_failed_mothers']=sum(c['inference']['p4_closure_denominator']-c['inference']['p4_closure_numerator'] for c in candidates)
                continue
            assert not seen.intersection(uids);seen.update(uids)
            summaries.append(report['summaries']);retained.append(report['retained_tree_checks'])
            for cat,data in report['summaries_by_source_category'].items():category_summaries[cat].append(data)
            for event in report['events']:
                uid=event['event_uid'];cat=event['source_category'];assert expected[uid]==cat
                points[arm][uid]=event_ratios(event)
                for scope,record in event['scopes'].items():
                    final_views[arm][scope].append({'event_uid':uid,'source_category':cat,'attempted':True,'processed':True,'truth_unavailable':not record['metrics']['available']})
                    count=counts[cat][scope];count['unique_attempted']+=1;count['processed']+=1
                    diag=record['inference'];g=guards[scope];g['processed']+=1
                    for key in ('committed_forest_source_conflict_count','source_conflicting_mother_count','source_conflicting_detector_resource_count'):
                        g[key]+=diag[key]
                    g['p4_closure_failed_mothers']+=diag['p4_closure_denominator']-diag['p4_closure_numerator']
                    g['committed_forest_disjoint_events']+=int(diag['committed_forest_detector_sources_disjoint'])
                    count['invalid_rollout']+=int(not record['inference']['rollout_event_valid'])
                    count['truth_unavailable']+=int(not record['metrics']['available'])
                    count['retained_tree_truth_unavailable']+=int(not record['retained_tree_metrics']['available'])
                    tree=record['retained_tree_metrics']
                    for row in tree['rows']:
                        if row['perfectLCAG_numerator']:
                            shapes[(scope,cat,row['truth_leaf_count'],row['truth_mother_count'],row['truth_retained_depth'])]+=1
                    if tree['coherent_retained_forest']['numerator']:
                        forests[(scope,cat,max((r['truth_retained_depth'] for r in tree['rows']),default=0),sum(r['truth_mother_count'] for r in tree['rows']))]+=1
                    tag=record['tag_efficiency']['greedy'];tags[f'{scope}/greedy'].append(tag)
                    count['exact_tag_truth_unavailable_events']+=int(tag['top1'].get('b_reconstruction',{}).get('unknown_truth_trials',0)>0 or (tag.get('continuum') or {}).get('unknown_retained_component_count',0)>0)
                    count['inclusive_membership_unavailable_events']+=int(tag['inclusive_fsp_grouping']['top1'].get('b_reconstruction',{}).get('unknown_truth_trials',0)>0 or (tag['inclusive_fsp_grouping'].get('continuum') or {}).get('unknown_retained_component_count',0)>0)
        from scripts.validate_category_coverage import validate_coverage
        validate_coverage([{'event_uid':uid,'source_category':expected[uid],
                            'attempted':True,'processed':True} for uid in seen])
        assert seen==set(expected) and len(seen)==12000
        assert all(c['processed']>=2000 for s in counts.values() for c in s.values())
        assert beam is not None
        public['coverage']['views'][arm]=counts
        public['arms'][arm]={'greedy':{'implementation_guards':{scope:dict(counts) for scope,counts in guards.items()},'summaries':merge_statistics(summaries),
            'summaries_by_source_category':{cat:merge_statistics(rows) for cat,rows in category_summaries.items()},
            'retained_tree_checks':merge_statistics(retained),
            'tag_efficiency':{'version':'tag-efficiency-study-v1','replaces_preregistered_metrics':False,
                              'summaries':{key:summarize_tag_efficiency_events(rows) for key,rows in tags.items()}}},
            'beam_diagnostic':beam,
            'exact_component_shapes':[{'scope':s,'category':c,'leaves':l,'mothers':m,'depth':d,'count':n} for (s,c,l,m,d),n in sorted(shapes.items())],
            'coherent_forest_shapes':[{'scope':s,'category':c,'maximum_depth':d,'mothers':m,'count':n} for (s,c,d,m),n in sorted(forests.items())]}
        from scipy.stats import beta
        bounds={}
        for key,rows in tags.items():
            for category in ('charged','mixed'):
                selected=[r for r in rows if r['source_category']==category]
                for kind in ('exact','inclusive'):
                    family=[r if kind=='exact' else r['inclusive_fsp_grouping'] for r in selected]
                    success=sum(r['top1']['b_reconstruction']['event_any_correct']['numerator'] for r in family)
                    trials=sum(r['top1']['b_reconstruction']['event_any_correct']['denominator'] for r in family)
                    bounds[key+'/'+category+'/'+kind]={'successes':success,'collision_trials':trials,
                      'one_sided_95_upper':float(beta.ppf(.95,success+1,trials-success)) if success<trials else 1.,
                      'lower_95_two_sided':float(beta.ppf(.025,success,trials-success+1)) if success else 0.,
                      'upper_95_two_sided':float(beta.ppf(.975,success+1,trials-success)) if success<trials else 1.}
        public['arms'][arm]['sparse_event_uncertainty']={'method':'Clopper-Pearson per-category collision event-any proven-success; two Bs stay clustered; conditional iid collisions within source category, not source-domain uncertainty or simultaneous intervals; unavailable truth limits identification of full-truth efficiency','bounds':bounds}

        public['source_hashes']+=sources
        atomic_json(out/f'{arm}-aggregate.json',public['arms'][arm])
    for cat in CATEGORIES:
        for scope in ('full','half'):
            a=public['coverage']['views'][ARMS[0]][cat][scope]
            b=public['coverage']['views'][ARMS[1]][cat][scope]
            for key in ('requested','unique_attempted','processed','failed','truth_unavailable','retained_tree_truth_unavailable','exact_tag_truth_unavailable_events','inclusive_membership_unavailable_events'):
                assert a[key]==b[key], 'Compared arms have different truth/coverage populations'
    from scripts.validate_phase72_final_coverage import validate_final_coverage
    final_coverage=validate_final_coverage(cohort,final_views)
    atomic_json(out/'final-coverage-validation.json',final_coverage)
    public['evaluator_revision']=scientific_source
    public['checkpoint_hashes']={arm:{key:plan['arms'][arm][key]['sha256'] for key in ('pretraining-checkpoint','reconstruction-checkpoint')} for arm in ARMS}
    public['data_hashes']={key:plan[key]['sha256'] for key in ('selection','index','cohort','coverage')}
    public['uncertainty']=paired_intervals(points[ARMS[0]],points[ARMS[1]],expected)
    atomic_json(out/'aggregate.json',public)
    return public


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    aggregate(a.output)

if __name__=='__main__':main()
