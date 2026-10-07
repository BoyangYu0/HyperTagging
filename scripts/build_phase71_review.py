"""Render the complete Phase71 review from authenticated aggregate evidence."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path

ARMS=('aux_teacher_050','aux_teacher_100')


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
    value=json.loads((a.artifacts/'primary-v1/aggregate.json').read_text())
    native=json.loads((a.artifacts/'native/closeout.json').read_text())
    extra=json.loads((a.artifacts/'primary-v1/extended-paired-uncertainty.json').read_text())
    diagnostic=json.loads((a.artifacts/'native/public-additional-diagnostics-v3.json').read_text())
    for arm in ARMS:
        for scopes in value['coverage']['views'][arm].values():assert all(r['processed']==2000 and r['failed']==0 for r in scopes.values())
    decision=json.loads((a.artifacts/'phase72-decision.json').read_text())
    title='Phase71 auxiliary teacher objective and efficiency review'
    lines=[title,'='*len(title),'',
      'The same original reserved cohort supplies **12,000 processed collisions per arm**, **2,000 in each required category**, in full and half scope. Checkpoint selection uses a separate 1,000 collisions. All failures and unavailable truth retain their original denominators; no difficult event is replaced. The 70,000-event training set and fitted normalization are unchanged.','',
      'The contrast is auxiliary teacher objective weight 0.5 versus 1.0, with matched mixed contexts, enabled decoder bias, source-exclusive inference and bounded updates. These retained-source metrics are not physical FEI or original quark reconstruction efficiencies.','',
      'See the :doc:`dashboard <_generated/status/index>` and :doc:`complete download catalogue <_generated/status/downloads>`. Historical studies and all native diagnostic gates remain preserved.','']
    lines += [decision['public_result'],'']
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
    table(lines,'Strict greedy; nominal denominators include unavailable truth',['Endpoint','Teacher weight 0.5','Teacher weight 1.0'],rows)
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
    table(lines,'Explicit retained components; fake slots use two nominal slots per collision',['Population and endpoint','Teacher weight 0.5','Teacher weight 1.0'],rows)
    lines += ['Component recovery is not q/qbar recovery: reduced data lack parton-to-FSP ancestry. Background acceptance and reconstruction correctness use different denominators. Unavailable component truth is retained in the complete aggregate tables. All accepted B candidates are inspected for tag recovery; slot-based fake-B accounting remains distinct from the total accepted-candidate count.','']
    section(lines,'Retained topology and paired uncertainty')
    rows=[]
    for scope in ('full','half'):
        for metric in ('lcag_pair_accuracy','source_precision','source_recall','coherent_retained_forest'):
            rows.append([scope+' '+metric,*[ratio(value['arms'][arm]['greedy']['retained_tree_checks']['summaries'][scope+'/greedy'][metric]) for arm in ARMS]])
        for metric in ('perfectLCAG','source_precision','source_recall'):
            r=value['uncertainty']['metrics'][f'{scope}/nontrivial_retained/{metric}']
            rows.append([scope+' nontrivial '+metric,*[ratio(r[arm]) for arm in ARMS]])
    table(lines,'All retained versus nontrivial populations',['Endpoint','Teacher weight 0.5','Teacher weight 1.0'],rows)
    rows=[]
    for arm in ARMS:
        for scope in ('full','half'):
            shapes=[s for s in value['arms'][arm]['exact_component_shapes'] if s['scope']==scope and s['mothers']>0 and s['leaves']>=2]
            forests=[s for s in value['arms'][arm]['coherent_forest_shapes'] if s['scope']==scope]
            rows.append([arm,scope,sum(s['count'] for s in shapes if s['depth']==1),sum(s['count'] for s in shapes if s['depth']>=2),sum(s['count'] for s in forests if s['mothers']>0),sum(s['count'] for s in forests if s['mothers']==0)])
    table(lines,'Exact shapes are not interchangeable with deep B reconstruction',['Arm','Scope','Depth-one exact components','Depth-two-or-deeper exact components','Forests with mothers','Mother-free forests'],rows)
    rows=[]
    for name in ('full/exact_tag/per_b_correct','full/inclusive_group/per_b_correct','full/retained/lcag_pair_accuracy','full/nontrivial_retained/perfectLCAG','full/nontrivial_retained/source_precision','full/nontrivial_retained/source_recall','full/retained/coherent_forest'):
        r=value['uncertainty']['metrics'][name]['weight100_minus_weight050']
        rows.append([name.replace('/',' '),r['value'],str(r['percentile_95'])])
    for name in ('full/pooled_continuum/exact_component_recovery','full/pooled_continuum/inclusive_component_recovery','full/pooled_continuum/top1_fake_b_event_acceptance'):
        r=extra['uncertainty']['metrics'][name]['weight100_minus_weight050'];rows.append([name.replace('/',' '),r['value'],str(r['percentile_95'])])
    table(lines,'Weight1.0 minus0.5; paired stratified collision bootstrap, 10,000 resamples',['Endpoint','Difference','95% percentile interval'],rows)
    lines += ['The downloads contain both-arm and difference intervals for every registered paired endpoint, including each continuum category, coherent pairs and fake-B slots. Both B trials stay within their collision cluster. Sparse category-specific event-any binomial bounds accompany zero/single successes. Degenerate zero-success bootstrap intervals do not prove zero population efficiency.','']
    section(lines,'Coverage, unavailable truth and immutable lineage')
    rows=[]
    for cat in ('charged','mixed','ccbar','uubar','ddbar','ssbar'):
        for scope in ('full','half'):
            r=value['coverage']['views'][ARMS[0]][cat][scope]
            rows.append([cat+' '+scope,*[r[k] for k in ('requested','unique_attempted','processed','failed','truth_unavailable','exact_tag_truth_unavailable_events','inclusive_membership_unavailable_events')]])
    table(lines,'Identical authenticated populations in both arms and scopes',['Category and scope','Requested','Attempted','Processed','Failed','Tree unavailable','Exact-tag unavailable events','Membership unavailable events'],rows)
    lines += ['The primary full-root tree metric is unavailable for continuum collisions because it does not invent a B-pair root. Explicit retained-component metrics remain separately measurable. Half-scope direct-target compatibility and exact/inclusive tag availability have their own unavailable counts; none of these events is removed.','', 'All 85,000 prior reservations and the separate 1,000 selection events are excluded from this original 12,000-event primary cohort. The supplementary registry authenticates the combined 13,000-event reservation. Input hashes cover 70,000 training and 160,000 validation identities, with no sealed-test payload access. Original training payloads and train-only normalization authenticate against the Phase70 lineage. Expanded independent validation is not training growth.','',
      'The original selection-only checkpoints are step 3,000 for weight 0.5 and step 4,000 for weight 1.0. No primary result is used to select another checkpoint. Shared cardinality limits 17 globally, 12 at level 1 and 17 at level 2 admit the four previously exposed targets in both arms. Their recorded repair prevents causal Phase70-to-Phase71 attribution.','',
      'The immutable native cohort also retains descriptive pre-repair index metadata and an inherited selection path. Actual native job contracts and the primary evaluator bind the repaired index and expanded selection independently by hash. The discrepancy is recorded, not silently edited; original memberships and training payloads are preserved. Phase72 writes current descriptive bindings.','',
      'Native source: ``1545d8ee80f26bc291498ae8718cb6e4a984f0bb``. Primary evaluator: ``'+value['evaluator_revision']+'``. Cohort digest: ``'+value['cohort_sha256']+'``. Every input, report, checkpoint and chunk is hash-bound. Full and half scopes, repeated reports and candidates do not multiply collision coverage.','']
    section(lines,'Native diagnostic tracks, geometry and execution')
    rows=[]
    for arm in ARMS:
        r=native['arms'][arm]
        rows.append([arm,r['optimizer_steps'],r['selected']['step'] if 'step' in r['selected'] else str(r['selected']),r['all_gates_passed'],', '.join(k for k,v in r['gates'].items() if not v) or 'none',r['primary_repeat_identical']])
    table(lines,'Original gates are retained, not replaced by primary efficiency',['Arm','Updates','Selected checkpoint','All original gates pass','Failed gates','Exact repeat'],rows)
    rows=[]
    for arm in ARMS:
        for view,c in diagnostic['arms'][arm]['diagnostic_category_counts'].items():
            rows.append([arm,view,c['distinct_collisions'],', '.join(f'{k}:{v}' for k,v in sorted(c['processed_by_category'].items()))])
    table(lines,'Registered native full/half views: actual diagnostic category counts',['Arm','View','Collisions','Category counts'],rows)
    rows=[]
    for arm in ARMS:
        totals=diagnostic['arms'][arm]['context_and_encoder_execution']['context_totals']
        rows.append([arm,*[totals[k] for k in ('sampled_teacher_count','sampled_predicted_count','truth_target_count','representable_target_count','unrepresentable_target_count','model_forward_count')]])
    table(lines,'Actual training exposure; matched updates are not equal FLOPs',['Arm','Teacher slots','Predicted slots','Target exposures','Representable','Unrepresentable','Model forwards'],rows)
    lines += ['Both arms execute 4,376 updates and 280,064 replay slots, seed 20261007, with zero additional pretraining. All 26 saved checkpoints retain exhaustive scalar exports, finite-state and frozen-PID lineage checks. Missing strict evaluations of unregistered tracks remain unavailable. All registered auxiliary full/half reports and every returned candidate are preserved.','',
      'Independent current-CPU replays preserve all original scientific counts and prediction fingerprints. The original native repeat gates pass exactly. Against the archived runtime, 9,446 continuous kinematic or score values differ, with maximum absolute difference 1.36e-5; these bounded differences and original values are preserved separately. No bitwise archived tensor equality is claimed.','',
      'Geometry diagnostics use 128 fixed training and 32 previously used development collisions across four curriculum views and initial/selected/final encoders. They are diagnostic, not independent primary coverage or a census of generated states. Their exact sample counts, per-level variance, radial derivatives, saturation and gradient/transfer diagnostics remain in the complete bundle. Persistent p4 is an exact daughter sum; closure is not generator momentum resolution.','']
    table(lines,'Geometry input samples: actual category counts',['Population','Collisions','Category counts'],[[k,sum(v.values()),str(v)] for k,v in diagnostic['geometry_category_counts'].items()])
    section(lines,'Diagnostic full-depth beam and candidate accounting')
    rows=[]
    for arm in ARMS:
        beam=value['arms'][arm]['beam_diagnostic']
        for scope in ('full','half'):
            rows.append([arm,scope,beam['coverage']['distinct_collisions'],str(beam['coverage']['processed_by_category']),beam['candidate_accounting'][scope]['returned_candidates']])
    table(lines,'Width-two beam: 60 collisions, 10/category; not policy-complete',['Arm','Scope','Collisions','Category counts','Returned candidates'],rows)
    rows=[]
    for arm in ARMS:
        for scope in ('full','half'):
            for kind in ('exact','inclusive'):
                summaries=value['arms'][arm]['beam_diagnostic']['tag_efficiency']['summaries']
                for mode,key,pool in [('greedy','greedy','1'),('model top1','full_depth_beam','1'),('retained pool@2','full_depth_beam','2')]:
                    t=summaries[scope+'/'+key]
                    if kind=='inclusive':t=t['inclusive_fsp_grouping']
                    b=t['summary']['pool_at_k'][pool]['b_reconstruction']
                    rows.append([arm,scope+' '+kind,mode,*[ratio(b[k]) for k in ('per_b_correct','event_any_correct','event_both_correct','coherent_event_both_correct')]])
    table(lines,'Beam subset efficiencies:40 nominal B trials,20 B-pair collisions',['Arm','Scope and definition','Selection','Per B','Any B','Both B pooled','Coherent pair'],rows)
    lines += ['Greedy, model-ranked top1 and retained pool at each K remain separate. Pool recovery deduplicates each true B and inspects every accepted candidate; incompatible successes are not a coherent pair. The complete download includes all beam/oracle, rank/score, candidate and uncertainty records, plus native 20-event proposal-beam diagnostics. Neither small subset inherits primary coverage.','']
    section(lines,'Synthesis of all studies')
    history=Path('docs/wiki/phase70.rst').read_text()
    history=history[history.index('.. list-table:: Study families and negative evidence'):history.index('Training growth and pretraining recommendation')]
    history=history.replace('The module intervention executes, but original small-cohort gate passing and shallow component recovery alone cannot establish tagging improvement. The separately reserved category-sized results above own the final efficiency claims.', 'The module intervention executes; its category-sized exact tags are0/8000 in both arms and inclusive tags0 versus1/8000. Better shallow recovery trades against recall and background. No joint efficiency winner.')
    lines += history.splitlines()+['']
    lines += ['Phase71 adds a category-sized auxiliary teacher objective contrast to that evidence. Its two arms can be compared within their shared cohort; historical cohorts, seeds, source domains, repaired capacities and evaluator versions must not be pooled as causal effects. Negative replications, failed controls and unavailable outcomes remain evidence.','']
    section(lines,'Decision, uncertainty and one bounded successor')
    decision=json.loads((a.artifacts/'phase72-decision.json').read_text())
    lines += [decision['recommendation'], '', decision['rationale'], '',
      'The paired intervals resample whole collisions within source category and condition on two fitted models. They do not estimate training-seed variation or provide multiplicity-adjusted discovery. Zero-width empirical bootstrap intervals do not show zero population efficiency: the downloads also report collision-level exact-binomial sparse-success bounds. Missing truth means proven-success lower bounds. Equal-category pooled results are not physical mixture estimates.','',
      'Phase72 tests the **presence of matched-daughter soft-Jaccard supervision**, coefficient 0 versus 1, alongside unchanged focal pointer supervision. The surrogate penalizes foreign and omitted daughters jointly on existing legal, matched target sets. It does not alter truth-free inference, matching assignments, source exclusivity or target eligibility. It is local daughter membership, not an end-to-end exact B objective.','',
      'Both arms use 70,000 training events, the same reconditioned encoder and frozen PID, mixed context 0.5, auxiliary teacher weight 0.5, enabled bias, late adaptation after 2,188 updates and 4,376 total updates/280,064 slots; seed 20261008. No extra pretraining is purchased. Resources are bounded to one H100 NVL, 8 CPUs, 64 GiB and 36 hours per arm, with no requeue.','',
      'Primary endpoints are exact/inclusive per-B and any/both/coherent-pair efficiencies with channel coverage, continuum recovery, fake-B acceptance, source precision and recursive structural guards. The useful-effect target is at least 4 additional successes per 8,000 nominal B trials with positive paired lower bounds and no background/precision/structure deterioration; this is not a power guarantee. Native diagnostic gates remain unchanged.','',
      'A fresh 12,000-event cohort, 2,000/category, and separate 1,000 selection events exclude the 98,000 prior reservations. Reservation alone is not future processed coverage. Admission, frozen-source hashes and accepted scheduling are recorded separately. There is no automatic successor, sealed-test access or scientific model promotion.','']
    snapshot=a.artifacts/'phase72/scheduling-snapshot.json'
    if snapshot.exists():
        scheduling=json.loads(snapshot.read_text())
        local=datetime.fromisoformat(scheduling['observed_utc']).astimezone(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M %Z')
        admission=json.loads((a.artifacts/'phase72/native-preflight.json').read_text())
        states=', '.join(j['arm']+': '+j['state']+' ('+j['reason']+')' for j in scheduling['jobs'])
        lines += [f"At the {local} admission snapshot, both Phase72 jobs are accepted: {states}. Frozen source ``{scheduling['jobs'][0]['source_sha']}`` passes {admission['passed_tests']} CPU admission tests and both native preflights. Startup and full training completion are not yet verified; scheduling is not scientific validation. Immutable private receipts bind both jobs, resources, configurations, checkpoint and fresh cohorts.",'']
    section(lines,'Complete aggregate downloads and integrity')
    lines += [':download:`Complete policy and scientific bundle <_generated/status/phase71-policy-reevaluation-v1.json>`; :download:`integrity <_generated/status/phase71-policy-integrity.json>`; :download:`standalone decoder <_generated/status/phase71-policy-decoder.txt>`.','',
      ':download:`All native aggregate scalars <_generated/status/phase71-native-aggregates.json>`; :download:`native integrity <_generated/status/phase71-native-integrity.json>`; :download:`native decoder <_generated/status/phase71-native-decoder.txt>`.','']
    binding=json.loads(Path('artifacts/codex/phase71_policy_reevaluation_20261007/binding.json').read_text())
    for part in binding['parts']:lines += [f":download:`{part['filename']} <_generated/status/{part['filename']}>`.",'']
    lines += ['The complete policy/scientific aggregate retains 1,756,039 numeric records; the native download retains 173,335 aggregate scalar records. Private scalar censuses retain 13,236,906 native values, 59,686 saved-checkpoint metadata values and 12,536,947 additive values, plus every policy report. These are exported records, not independent observations.','', 'The lossless compact formats retain numeric counts, channel/type labels and null availability. Public archives omit collision identifiers, native machine paths, scheduler records, logs and weights. Private evidence retains full reports, logs, scalar censuses, native failures, hash inventories and reproducible receipts. Historical downloads and their immutable bytes remain preserved.','']
    a.output.write_text('\n'.join(lines).rstrip()+'\n')
if __name__=='__main__':main()
