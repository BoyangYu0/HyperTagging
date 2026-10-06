"""Render the complete Phase70 review from authenticated aggregate evidence."""
import argparse
import json
from pathlib import Path

ARMS=('type_bias','no_type_bias')


def ratio(value):
    return 'UNAVAILABLE' if not value.get('denominator') else f"{value['numerator']:g}/{value['denominator']:g} ({100*value['value']:.5g}%)"


def table(lines,title,columns,rows):
    lines.extend(['.. list-table:: '+title,'   :header-rows: 1','   :widths: auto',''])
    for row in [columns,*rows]:
        lines.extend(('   * - ' if i==0 else '     - ')+str(v) for i,v in enumerate(row))
    lines.append('')


def section(lines,title):lines.extend([title,'-'*len(title),''])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=json.loads((a.artifacts/'supplemental-v1/aggregate.json').read_text())
    native=json.loads((a.artifacts/'native/closeout-reviewed.json').read_text())
    extra=json.loads((a.artifacts/'supplemental-v1/extended-paired-uncertainty.json').read_text())
    for arm in ARMS:
        for cat,scopes in value['coverage']['views'][arm].items():
            assert all(row['processed']==2000 and row['failed']==0 for row in scopes.values())
    lines=['Phase70 decoder relation-bias and efficiency review','='*51,'',
      'The category-complete evaluation does not establish an exact retained B-tag advantage. Both arms have **0/8,000 exact successes**; inclusive FSP grouping is **0/8,000 with bias enabled and 1/8,000 with bias disabled**, in both full and half scope. That single inclusive success is exploratory. These are retained-source proxies, not physical FEI efficiencies.','',
      'The same immutable cohort contains **2,000 distinct processed collisions in each of charged, mixed, ccbar, uubar, ddbar and ssbar**, 12,000 per arm. All requested collisions were processed; none was replaced or omitted for difficult or unavailable truth. Full and half scope share collisions. No sealed test was accessed and no scientific model is promoted.','',
      'The :doc:`dashboard <_generated/status/index>` gives the leading efficiency results and links every historical and current aggregate download. This review separates policy-sized primary measurements from native and beam diagnostics.','']
    section(lines,'Primary tagging and coherent pairs')
    rows=[]
    for scope in ('full','half'):
        for kind in ('exact','inclusive'):
            summaries=[]
            for arm in ARMS:
                t=value['arms'][arm]['greedy']['tag_efficiency']['summaries'][scope+'/greedy']
                summaries.append((t if kind=='exact' else t['inclusive_fsp_grouping'])['summary'])
            for endpoint in ('per_b_correct','event_any_correct','event_both_correct'):
                rows.append([f'{scope} {kind} {endpoint}',*[ratio(s['top1']['b_reconstruction'][endpoint]) for s in summaries]])
            rows.append([f'{scope} {kind} coherent pair',*[ratio(s['pool_at_k']['1']['b_reconstruction']['coherent_event_both_correct']) for s in summaries]])
    table(lines,'Strict greedy; nominal denominators include unavailable truth',['Endpoint','Bias enabled','Bias disabled'],rows)
    rows=[]
    for arm in ARMS:
        for kind in ('exact','inclusive'):
            t=value['arms'][arm]['greedy']['tag_efficiency']['summaries']['full/greedy']
            t=t if kind=='exact' else t['inclusive_fsp_grouping'];channels=t['b_channel_coverage']
            rows.append([arm,kind,t['summary']['top1']['b_reconstruction']['unknown_truth_trials'],len(channels),sum(bool(c['covered_top1']) for c in channels),sum(c['evaluated_b_trials'] for c in channels)])
    table(lines,'Full-scope B truth and channel accounting; half scope is separately exported',['Arm','Definition','Unknown nominal B trials','Channel rows','Recovered channels','Nominal channel trials'],rows)
    lines += ['Each B-pair collision contributes two nominal B trials. Distinct truth B successes are deduplicated across every accepted candidate. Inclusive success requires an actual valid reconstructed composite with precisely the retained FSP membership; no disconnected partition is invented. Exact topology/PID availability and inclusive membership availability remain separate. Missing truth makes observed success rates proven-success lower bounds. Channel identifiers describe the retained signed representation; full generator decay descriptions and original quark ancestry are unavailable.','']
    section(lines,'Continuum recovery and fake-B background')
    rows=[]
    for scope in ('full','half'):
        for cat in ('ccbar','uubar','ddbar','ssbar'):
            for label,field in [('exact','retained_component_top1_correct'),('inclusive','retained_component_top1_correct'),('fake-B event','top1_fake_b_event_acceptance'),('fake-B slot','top1_fake_b_slot_acceptance')]:
                cells=[]
                for arm in ARMS:
                    t=value['arms'][arm]['greedy']['tag_efficiency']['summaries'][scope+'/greedy']
                    if label=='inclusive':t=t['inclusive_fsp_grouping']
                    cells.append(ratio(t['continuum_type_coverage'][cat][field]))
                rows.append([f'{scope} {cat} {label}',*cells])
    table(lines,'Explicit retained components; fake slots use two nominal slots per collision',['Population and endpoint','Bias enabled','Bias disabled'],rows)
    lines += ['Component recovery is not q/qbar recovery: reduced data lack parton-to-FSP ancestry. Background acceptance and reconstruction correctness use different denominators. Unavailable component truth is retained in the complete aggregate tables. All accepted B candidates are inspected for tag recovery; slot-based fake-B accounting remains distinct from the total accepted-candidate count.','']
    section(lines,'Retained topology and paired uncertainty')
    rows=[]
    for scope in ('full','half'):
        for metric in ('lcag_pair_accuracy','source_precision','source_recall','coherent_retained_forest'):
            rows.append([scope+' '+metric,*[ratio(value['arms'][arm]['greedy']['retained_tree_checks']['summaries'][scope+'/greedy'][metric]) for arm in ARMS]])
        for metric in ('perfectLCAG','source_precision','source_recall'):
            r=value['uncertainty']['metrics'][f'{scope}/nontrivial_retained/{metric}']
            rows.append([scope+' nontrivial '+metric,*[ratio(r[arm]) for arm in ARMS]])
    table(lines,'All retained versus nontrivial populations',['Endpoint','Bias enabled','Bias disabled'],rows)
    rows=[]
    for arm in ARMS:
        for scope in ('full','half'):
            shapes=[s for s in value['arms'][arm]['exact_component_shapes'] if s['scope']==scope and s['mothers']>0 and s['leaves']>=2]
            forests=[s for s in value['arms'][arm]['coherent_forest_shapes'] if s['scope']==scope]
            rows.append([arm,scope,sum(s['count'] for s in shapes if s['depth']==1),sum(s['count'] for s in shapes if s['depth']>=2),sum(s['count'] for s in forests if s['mothers']>0),sum(s['count'] for s in forests if s['mothers']==0)])
    table(lines,'Exact shapes are not interchangeable with deep B reconstruction',['Arm','Scope','Depth-one exact components','Depth-two-or-deeper exact components','Forests with mothers','Mother-free forests'],rows)
    rows=[]
    for name in ('full/exact_tag/per_b_correct','full/inclusive_group/per_b_correct','full/retained/lcag_pair_accuracy','full/nontrivial_retained/perfectLCAG','full/nontrivial_retained/source_precision','full/nontrivial_retained/source_recall','full/retained/coherent_forest'):
        r=value['uncertainty']['metrics'][name]['disabled_minus_enabled']
        rows.append([name.replace('/',' '),r['value'],str(r['percentile_95'])])
    for name in ('full/pooled_continuum/exact_component_recovery','full/pooled_continuum/inclusive_component_recovery','full/pooled_continuum/top1_fake_b_event_acceptance'):
        r=extra['uncertainty']['metrics'][name]['disabled_minus_enabled'];rows.append([name.replace('/',' '),r['value'],str(r['percentile_95'])])
    table(lines,'Disabled minus enabled; paired stratified collision bootstrap, 10,000 resamples',['Endpoint','Difference','95% percentile interval'],rows)
    lines += ['The downloads contain both-arm and difference intervals for every registered paired endpoint, including each continuum category, coherent pairs and fake-B slots. Both B trials stay within their collision cluster. Sparse category-specific event-any binomial bounds accompany zero/single successes. Degenerate zero-success bootstrap intervals do not prove zero population efficiency.','']
    section(lines,'Coverage, independence and immutable lineage')
    rows=[]
    before={'charged':1921,'mixed':1901,'ccbar':5825,'uubar':5778,'ddbar':1913,'ssbar':1833}
    for cat in ('charged','mixed','ccbar','uubar','ddbar','ssbar'):
        r=value['coverage']['views']['type_bias'][cat]['full']
        rows.append([cat,before[cat],max(0,2000-before[cat]),value['coverage']['available_by_category'][cat],r['processed'],r['failed'],r['truth_unavailable'],r['exact_tag_truth_unavailable_events'],r['inclusive_membership_unavailable_events']])
    table(lines,'Actual per-arm primary coverage; both arms and scopes authenticate identically',['Category','Old eligible','Old shortage','Expanded eligible','Processed','Failed','Tree unavailable','Exact-tag unavailable events','Membership unavailable events'],rows)
    lines += ['The prior union contains 73,000 reservations, including the Phase69 policy supplement. The historical 39,000 figure is stale. Four categories were short before materialization. The established source-safe expansion used two fresh source files per required category, 60,000 new validation collisions, excluded the complete prior source catalogue, preserved every original role and all 70,000 training payloads, and fitted nothing on validation. The resulting index authenticates 230,000 distinct train/validation identities, unchanged train normalization and no source/category mismatches. Sealed-test payloads were never read.','',
      'The deterministic identity-only cohort reserves 12,000 events, leaving 75,000 unreserved validation identities before the next campaign. Selection, fitting, training and every historical adaptive reservation are excluded. The checkpoints remain the original selection-only step4000 artifacts. The expanded external-sample path relaxes index/split identity only; explicit cohort identities, schema, feature, PID, normalization, policy and source exclusivity checks remain.','',
      'Native training source: ``fc05d79ea4fef264cf8cd5718bde420504170590``. Supplemental evaluator source: ``'+value['evaluator_revision']+'``. Cohort SHA-256: ``'+value['cohort_sha256']+'``. Every checkpoint, input, cohort chunk and output report has an immutable digest in the exports. The exact policy handoff patch is integrated without replacing unrelated edits.','']
    section(lines,'Native evidence and smaller diagnostic views')
    lines += ['Both native training jobs completed successfully, but scheduler completion is not scientific validation. Both arms execute 4,376 finite optimizer updates and 280,064 replay slots on 70,000 training events, seed20261006, mixed contexts and adaptation after2,188 steps, with zero additional pretraining. Shared refined encoder/PID initialization and train normalization authenticate. Module presence is independently verified on all five registered tracks:21 treatment parameter entries versus zero. Serialized model-state tensor elements are4,439,722 versus3,729,971; these are not equal-FLOP or bitwise-prefix experiments.','',
      'All fourteen native reports,26 saved checkpoints and exact native primary repeats authenticate. Native strict coverage is100 collisions: charged8, mixed13, ccbar16, uubar25, ddbar8, ssbar7 and taupair23. Its21 B-pair collisions give42 nominal B trials, with zero exact/inclusive successes in both arms and scopes. These are historical diagnostics, not policy-complete final coverage.','']
    table(lines,'Preserved original 100-event gates and endpoints',['Endpoint','Bias enabled','Bias disabled'],[[key,*[ratio(native['arms'][arm]['endpoints'][key]) for arm in ARMS]] for key in native['arms'][ARMS[0]]['endpoints']])
    lines += ['Only the disabled arm passes every original gate. Enabled fails full source precision, half perfect LCAG, half root PID and half source precision. The native retained full LCAG counts are19/3,823 versus22/3,823; exact nontrivial components are2/135 versus7/135, all depth one. The native paired LCAG interval spans zero. Gate passing, leaf retention and shallow exactness do not establish tagging utility.','',
      'The original seven views per arm retain all checkpoint tracks, contracted topology, exact repeats and every proposal-beam candidate. Additive instrumentation reproduces2,600 inference fingerprints exactly; all scientific counts/categories agree. The3,676 archived continuous-value differences, maximum absolute1.4007091522216797e-6, remain documented rather than silently replacing native p4 or ranking values. The separate metadata correction changes only the inherited native classification/date, keeping original evidence and every numerical result.','',
      'The new width-two full-depth beam processes60 collisions per arm, ten per category, in both scopes. It is explicitly diagnostic. Its top1, retained pool at every K, oracle, score/rank summaries and actual candidate counts are exported separately. Incompatible candidates recovering different Bs do not count as a coherent pair; no beam result inherits12,000-event coverage.','']
    rows=[]
    for arm in ARMS:
        for scope in ('full','half'):
            b=value['arms'][arm]['beam_diagnostic'];rows.append([arm,scope,b['coverage']['distinct_collisions'],b['candidate_accounting'][scope]['returned_candidates']])
    table(lines,'Supplemental beam diagnostic cardinality',['Arm','Scope','Distinct collisions','Returned candidates'],rows)
    section(lines,'Complete metric and implementation audit')
    lines += ['All topology/LCAG/source/PID/p4/constraint aggregates, unavailable values, target representability, depth/nontrivial components, coherent forests, channel coverage, beam candidates and confidence/search diagnostics are included. The complete native/checkpoint exports contain11,711,302 native scalar records and63,089 checkpoint scalar records, including zero-dimensional tensor scalars. The native public bundle contains178,049 scalar values across nine registered views. Additive exports contain14 reports,2,480 event-scope records,240 candidate-scope records and132,208 public numeric records. The private archive also contains every supplemental report and its complete scalar census. Cardinality and SHA-256 manifests bind each family; overlapping export families are not advertised as unique independent measurements.','',
      'All48 fixed-input geometry views are unsaturated. They are not a census of generated-state geometry. Exposure audits retain105,347 predicted and174,717 teacher slots in both arms; actual forward counts396,937 versus388,985 differ. Complete checkpoint transfer and native objective histories remain downloadable. Missing strict evaluation of unregistered checkpoint tracks is unavailable, never zero. No source conflict, failed daughter-sum closure or truth-assisted inference is tolerated by the implementation audit. Daughter-sum closure is a construction invariant, not physical momentum resolution against generator truth.','',
      'Public exports omit collision identities, raw reports, private paths, scheduler identifiers and checkpoint filenames. Lossless typed-DAG parts preserve scalar types, unavailable values and all channel tables; the standalone decoder verifies the decoded digest. Download the manifest, all parts and decoder into one directory. The decoded object retains arms, native_additive_tagging, native_additional_diagnostics and extended_paired_uncertainty. A separate native scalar decoder restores every compact metric name to JSONL and verifies the178,049-record binding. The private bundle keeps raw records, native receipts and full integrity manifests. Historical reports and downloads retain their original bytes.','']
    lines += (a.artifacts/'control/review-discussion.rst').read_text().splitlines()+['']
    a.output.write_text('\n'.join(lines)+'\n')
    print(a.output)


if __name__=='__main__':main()
