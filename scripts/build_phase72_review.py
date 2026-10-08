"""Render the complete Phase72 review from authenticated aggregate evidence."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path

ARMS=('set_overlap_off','set_overlap_on')


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
    decision=json.loads((a.artifacts/'phase73-decision.json').read_text())
    title='Phase72 set-overlap objective and efficiency review'
    lines=[title,'='*len(title),'',
      'The same original reserved cohort supplies **12,000 processed collisions per arm**, **2,000 in each required category**, in full and half scope. Checkpoint selection uses a separate 1,000 collisions. All failures and unavailable truth retain their original denominators; no difficult event is replaced. The 70,000-event training set and fitted normalization are unchanged.','',
      'The contrast adds matched-daughter soft-Jaccard weight 0 versus 1 alongside the same focal pointer objective, auxiliary teacher weight 0.5, mixed contexts and enabled decoder bias. The shared seed is 20261008, with 4,376 updates and no extra pretraining. Inference and target eligibility are unchanged. These retained-source metrics are not physical FEI or original quark reconstruction efficiencies.','',
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
    table(lines,'Strict greedy; nominal denominators include unavailable truth',['Endpoint','Set overlap off','Set overlap on'],rows)
    rows=[]
    for arm in ARMS:
        for kind in ('exact','inclusive'):
            t=value['arms'][arm]['greedy']['tag_efficiency']['summaries']['full/greedy']
            t=t if kind=='exact' else t['inclusive_fsp_grouping'];channels=t['b_channel_coverage']
            rows.append([arm,kind,t['summary']['top1']['b_reconstruction']['unknown_truth_trials'],len(channels),sum(bool(c['covered_top1']) for c in channels),sum(c['evaluated_b_trials'] for c in channels)])
    table(lines,'Full-scope B truth and channel accounting; half scope is separately exported',['Arm','Definition','Unknown nominal B trials','Channel rows','Recovered channels','Nominal channel trials'],rows)
    lines += ['Each B-pair collision contributes two nominal B trials. Distinct truth B successes are deduplicated across every accepted candidate. Inclusive success requires an actual valid reconstructed composite with precisely the retained FSP membership; no disconnected partition is invented. Exact topology/PID availability and inclusive membership availability remain separate. Missing truth makes observed success rates proven-success lower bounds. Channel identifiers describe the retained signed representation; full generator decay descriptions and original quark ancestry are unavailable.','']
    rows=[]
    for category in ('charged','mixed'):
        for scope in ('full','half'):
            for kind in ('exact','inclusive'):
                cells=[]
                for arm in ARMS:
                    t=value['arms'][arm]['greedy']['tag_efficiency']['summaries'][scope+'/greedy']
                    if kind=='inclusive':t=t['inclusive_fsp_grouping']
                    cells.append(ratio(t['by_source_category'][category]['top1']['b_reconstruction']['per_b_correct']))
                rows.append([category+' '+scope+' '+kind,*cells])
    table(lines,'Per-category nominal B tagging',['Population','Set overlap off','Set overlap on'],rows)
    lines += ['All three positive inclusive memberships reproduce exactly in separate tree-retaining replays. Inspection identifies actual depth-one composites containing two retained FSPs, predicted as D0 or anti-D0, rather than exact B trees. Source sets match the corresponding retained B memberships; no partition or disconnected union is invented. The charged success is shared between arms; only the off arm adds a mixed-category success. These very small retained groups reinforce the limitation of treating inclusive success as physical B-tagging efficiency.', '']
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
    table(lines,'Explicit retained components; fake slots use two nominal slots per collision',['Population and endpoint','Set overlap off','Set overlap on'],rows)
    lines += ['Component recovery is not q/qbar recovery: reduced data lack parton-to-FSP ancestry. Background acceptance and reconstruction correctness use different denominators. Unavailable component truth is retained in the complete aggregate tables. All accepted B candidates are inspected for tag recovery; slot-based fake-B accounting remains distinct from the total accepted-candidate count.','']
    section(lines,'Retained topology and paired uncertainty')
    rows=[]
    for scope in ('full','half'):
        for metric in ('lcag_pair_accuracy','source_precision','source_recall','coherent_retained_forest'):
            rows.append([scope+' '+metric,*[ratio(value['arms'][arm]['greedy']['retained_tree_checks']['summaries'][scope+'/greedy'][metric]) for arm in ARMS]])
        for metric in ('perfectLCAG','source_precision','source_recall'):
            r=value['uncertainty']['metrics'][f'{scope}/nontrivial_retained/{metric}']
            rows.append([scope+' nontrivial '+metric,*[ratio(r[arm]) for arm in ARMS]])
    table(lines,'All retained versus nontrivial populations',['Endpoint','Set overlap off','Set overlap on'],rows)
    rows=[]
    for arm in ARMS:
        for scope in ('full','half'):
            shapes=[s for s in value['arms'][arm]['exact_component_shapes'] if s['scope']==scope and s['mothers']>0 and s['leaves']>=2]
            forests=[s for s in value['arms'][arm]['coherent_forest_shapes'] if s['scope']==scope]
            rows.append([arm,scope,sum(s['count'] for s in shapes if s['depth']==1),sum(s['count'] for s in shapes if s['depth']>=2),sum(s['count'] for s in forests if s['mothers']>0),sum(s['count'] for s in forests if s['mothers']==0)])
    table(lines,'Exact shapes are not interchangeable with deep B reconstruction',['Arm','Scope','Depth-one exact components','Depth-two-or-deeper exact components','Forests with mothers','Mother-free forests'],rows)
    rows=[]
    for name in ('full/exact_tag/per_b_correct','full/inclusive_group/per_b_correct','full/retained/lcag_pair_accuracy','full/nontrivial_retained/perfectLCAG','full/nontrivial_retained/source_precision','full/nontrivial_retained/source_recall','full/retained/coherent_forest'):
        r=value['uncertainty']['metrics'][name]['overlap_on_minus_off']
        rows.append([name.replace('/',' '),r['value'],str(r['percentile_95'])])
    for name in ('full/pooled_continuum/exact_component_recovery','full/pooled_continuum/inclusive_component_recovery','full/pooled_continuum/top1_fake_b_event_acceptance'):
        r=extra['uncertainty']['metrics'][name]['overlap_on_minus_off'];rows.append([name.replace('/',' '),r['value'],str(r['percentile_95'])])
    table(lines,'On minus off; paired stratified collision bootstrap, 10,000 resamples',['Endpoint','Difference','95% percentile interval'],rows)
    lines += ['The downloads contain both-arm and difference intervals for every registered paired endpoint, including each continuum category, coherent pairs and fake-B slots. Both B trials stay within their collision cluster. Sparse category-specific event-any binomial bounds accompany zero/single successes. Degenerate zero-success bootstrap intervals do not prove zero population efficiency.','']
    section(lines,'Coverage, unavailable truth and immutable lineage')
    rows=[]
    for cat in ('charged','mixed','ccbar','uubar','ddbar','ssbar'):
        for scope in ('full','half'):
            r=value['coverage']['views'][ARMS[0]][cat][scope]
            rows.append([cat+' '+scope,*[r[k] for k in ('requested','unique_attempted','processed','failed','truth_unavailable','exact_tag_truth_unavailable_events','inclusive_membership_unavailable_events')]])
    table(lines,'Identical authenticated populations in both arms and scopes',['Category and scope','Requested','Attempted','Processed','Failed','Tree unavailable','Exact-tag unavailable events','Membership unavailable events'],rows)
    lines += ['Full-root tree metrics do not invent a B-pair root for continuum collisions. Explicit retained components remain separately measurable. Direct-target incompatibility, exact-tag availability and membership availability have distinct denominators; missing truth never causes replacement.', '',
      'The original primary cohort excludes all 98,000 prior reservations and 1,000 checkpoint-selection collisions. It is disjoint from the 70,000-event training set and train-fitted normalization. The source-safe 160,000-event validation expansion is not training-data growth. Both original selection-only checkpoints are step 4,000; primary outcomes never choose another checkpoint.', '',
      'Native source: ``38124a9e05ca63cd5ac5469e0687952eec2b14dc``. Evaluator: ``'+value['evaluator_revision']+'``. Cohort digest: ``'+value['cohort_sha256']+'``. Selection/index digests and every checkpoint, chunk and receipt are retained. Frozen contracts and original failed gates are unchanged.', '']
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
    receipts=json.loads((a.artifacts/'additive/authenticated-evaluation-receipts.json').read_text())
    differences=[d for r in receipts for d in r['archived_floating_point_differences']]
    lines += ['Both arms execute 4,376 updates and 280,064 replay slots, seed 20261008, with zero additional pretraining. All 26 saved checkpoints have scalar censuses, finite-state and frozen-PID transfer checks. Strict results of unregistered checkpoint tracks remain unavailable.', '',
      f"Independent current-CPU replays retain all scientific counts and prediction fingerprints. Native repeat gates pass exactly. Against the archived runtime, {len(differences):,} continuous values differ, with maximum absolute difference {max((x['absolute_difference'] for x in differences),default=0):.6g}; original values and differences remain archived. No unsupported bitwise archived equality is claimed.", '',
      'Geometry uses 128 fixed training and 32 previously used development collisions across four curriculum views and initial/selected/final encoders. These are diagnostic inputs, not primary coverage or a generated-state census. The complete bundle retains variance, radial derivatives, saturation, transfer and objective execution diagnostics. Daughter-sum p4 closure is an implementation invariant, not physical resolution.', '']
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
    section(lines,'Synthesis of all studies and allocation recommendation')
    history=Path('docs/wiki/phase70.rst').read_text()
    history=history[history.index('.. list-table:: Study families and negative evidence'):history.index('Interpretation of uncertainty\n')]
    history=history.replace('The module intervention executes, but original small-cohort gate passing and shallow component recovery alone cannot establish tagging improvement. The separately reserved category-sized results above own the final efficiency claims.', 'Category-sized exact tags are 0/8000 in both arms and inclusive tags 0 versus 1/8000. Better shallow recovery trades against recall and higher fake-B acceptance; no joint winner.')
    history=history.replace('one next bounded controlled study','next bounded development study')
    lines += history.splitlines()+['']
    lines += ['Phase71 auxiliary weights 0.5/1.0 yield 0/8000 exact and inclusive tags in both arms. Nontrivial exact components fall 308/18536 to 148/18536, while fake-B events change 293/8000 to 281/8000 with a paired interval spanning zero. This ends auxiliary dose tuning. Phase72 tests a different local set objective; it does not estimate the benefit of training growth, encoder width or more pretraining.', '', decision['public_result'], '', decision['recommendation'], '', decision['rationale'], '']
    section(lines,'Latest structural policy and one bounded development campaign')
    lines += ['The integrated policy is structural-reconstruction-study-policy-v2. The uncommitted source snapshot and its evidence were authenticated before integration; later remote/live-source checks preserve lineage. Historical study contracts remain immutable.', '',
      'Five search conditions on the same reused 60 development collisions (40 B trials) all recover 0 exact and inclusive B groups at top1 and in the pool. Broader search generates more shallow proposals, but no correct complete B pool. Reranking-only work lacks the required pool/top1 gap. Optimistic clean-root coverage is a necessary bound, not legal decoder reachability.', '',
      'The frozen head memorizes 32/32 raw tiny memberships, of which 28 pass candidate guards; both tiny and 384-event fits remain 0/40 held out. This is learnability evidence, not generalization or proof that encoder capacity is sufficient. The pilot has a separate 64-wide head bottleneck.', '',
      'Phase73 therefore tests the policy’s direct-membership joint-learning family in development: the same historical encoder frozen versus adapted, a common 128-wide head, frozen PID parameters, identical initialization, identities and presentation budgets. The original 384 training identities and 60 reused development collisions are disjoint from Phase72 primary and selection. Fixed final checkpoints and presence threshold 0.5 avoid tuning on development. Each arm has a 1,000-update tiny-head assay and 1,000-update main fit; seed 20261008. Two CPU-only jobs each have 2 CPUs, 32 GiB, 24 hours and no requeue.', '',
      'The proposed 128/256 contextual-width by existing/assembly-targeted pretraining factorial remains an unexecuted representation plan. It must hold hyperbolic width 32, depth and geometry fixed, use comparable pretraining histories, a controlled downstream interface and measured compute. Historical pretrained128 versus fresh256 cannot measure a width effect. Phase73 is an adaptation contrast and makes no capacity or pretraining claim.', '',
      'Before any primary scale-up or hierarchy, require at least95% tiny raw membership plus a count-backed held-out membership gain over control, background comparison and source validity. Lower loss, memorization, extra candidates and shallow recovery alone cannot pass. No new primary cohort is reserved; the sealed test stays closed. There is no automatic second campaign or model promotion. Policy planning validation is separate from runtime/data/resource admission.', '',
      'Prioritize assembly-aligned objectives and representations before purchasing uncontrolled training growth or simply longer unchanged pretraining. This allocation recommendation does not establish that better pretraining beats data growth. A later growth study must separate more unique examples from optimizer presentations and compute; repeated70,000-event holds do not prove scaling useless.', '']
    snapshot=a.artifacts/'phase73/scheduling-snapshot.json'
    if snapshot.exists():
        scheduling=json.loads(snapshot.read_text())
        local=datetime.fromisoformat(scheduling['observed_utc']).astimezone(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M %Z')
        states=', '.join(j['arm']+': '+j['state'] for j in scheduling['jobs'])
        lines += [f"At {local}, the two development jobs are accepted: {states}. Frozen source ``{scheduling['source_sha']}``. Runtime admission authenticates the checkpoint, data, source, cohort exclusions and fixed resource budget. Job completion and scientific effectiveness are not required or inferred from this scheduling snapshot.", '']
    section(lines,'Uncertainty and interpretation limits')
    lines += ['All paired intervals resample whole collisions within category, keeping both B trials together. They condition on two fitted models; single-seed uncertainty and multiplicity are not covered. Exact binomial collision bounds accompany sparse or zero successes. A zero-width empirical bootstrap is not zero population uncertainty. Missing truth remains nominal, and equal-category pooled rates are not physical mixture estimates.', '']
    rows=[]
    for arm in ARMS:
        for key,b in value['arms'][arm]['sparse_event_uncertainty']['bounds'].items():
            if key.startswith('full/'):
                rows.append([arm,key,b['successes'],b['collision_trials'],b['one_sided_95_upper']])
    table(lines,'Sparse bounds for collision event-any proven success',['Arm','Population','Successes','Collisions','One-sided95% upper'],rows)
    section(lines,'Complete metric downloads and integrity')
    lines += [':doc:`Download all latest and historical metric files <_generated/status/downloads>`. Policy manifests, all parts, standard-library decoders and integrity/cardinality records are preserved together. The policy bundle includes all full/half primary aggregates, channel/type coverage, exact/inclusive tagging, beam top1/pool/coherent accounting, paired uncertainty and native diagnostics.', '',
      'New payload files use bounded gzip/base32 JSON envelopes of authenticated public aggregates. The supplied standalone decoders verify compressed and decoded hashes, expansion limits and cardinalities. This lossless transport preserves every historical download and retains the fixed site/file/privacy limits.', '',
      'Every per-step training and checkpoint metadata scalar is also downloadable in the four ``phase72-set-overlap-*-scalars.json`` files, with ``phase72-scalar-integrity.json`` and the bounded ``phase72-scalar-decoder.txt``. Native aggregate records: ``phase72-native-aggregates.json``; integrity and decoder: ``phase72-native-integrity.json`` and ``phase72-native-decoder.txt``. Policy entrypoint: ``phase72-policy-reevaluation-v1.json``; integrity and decoder: ``phase72-policy-integrity.json`` and ``phase72-policy-decoder.txt``. Download every referenced part before decoding.', '']
    native_export=json.loads((a.artifacts/'native/export-manifest.json').read_text())
    cp=json.loads((a.artifacts/'checkpoint-exhaustive/checkpoint-export-manifest.json').read_text())
    combined=json.loads((a.artifacts/'complete-public-v1/combination-manifest.json').read_text())
    nb=json.loads(Path('artifacts/codex/phase72_native_20261008/binding.json').read_text())
    lines += [f"The complete public scientific bundle contains {combined['numeric_records']:,} numeric records; the native bundle contains {nb['scalar_records']:,} aggregate scalar records. Full private censuses preserve {native_export['native_scalar_count']:,} native scalars and {cp['additional_native_scalar_count']:,} checkpoint metadata scalars, plus every primary report, candidate, per-step training scalar and repeat. Counts measure exported records, not independent observations.", '',
      'Public aggregates retain null availability and all reviewed count/label fields while excluding event identifiers, private paths, logs, scheduler IDs and weights. Full private evidence and hash manifests remain in the task artifact bundle. No historical download or original failed gate has been removed.', '']
    a.output.write_text('\n'.join(lines).rstrip()+'\n')
if __name__=='__main__':main()
