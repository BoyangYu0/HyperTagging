"""Bounded aggregate-only publication of the supplementary Phase69 policy study."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = 'artifacts/codex/phase69_policy_reevaluation_20261006'
BUNDLE = 'phase69-policy-reevaluation-v1.json'
INTEGRITY = 'phase69-policy-integrity.json'
DECODER = 'phase69-policy-decoder.txt'
MAX_PART_BYTES = 4 * 1024 * 1024
MAX_PARTS = 4
MAX_NODES = 200_000
MAX_DECODED_BYTES = 128 * 1024 * 1024


def sibling(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()


def decoded_size(nodes, root):
    """Bound DAG expansion before materializing or scanning repeated subtrees."""
    if len(nodes)>MAX_NODES:raise ValueError('Policy DAG exceeds node budget')
    sizes=[]
    for node in nodes:
        def ref(index):
            if type(index) is not int or not 0<=index<len(sizes):raise ValueError('Invalid DAG reference')
            return sizes[index]
        if node[0]=='dict':
            size=2+max(0,len(node[1])-1)+sum(ref(k)+1+ref(v) for k,v in node[1])
        elif node[0]=='list':
            size=2+max(0,len(node[1])-1)+sum(ref(i) for i in node[1])
        elif node[0]=='null':size=4
        elif node[0] in ('str','bool','int','float'):
            size=len(json.dumps(node[1],separators=(',', ':'),allow_nan=False).encode())
        else:raise ValueError('Unknown DAG node')
        if size+1>MAX_DECODED_BYTES:raise ValueError('Policy DAG exceeds decoded byte budget')
        sizes.append(size)
    if type(root) is not int or not 0<=root<len(sizes):raise ValueError('Invalid DAG root')
    return sizes[root]+1


def validate(value):
    if value['version']!='phase69-policy-reevaluation-v1-20261006' or value['status']!='COMPLETE_GREEDY_WITH_DIAGNOSTIC_BEAM':
        raise ValueError('Unknown policy study publication')
    if value['historical_reports_replaced'] or value['sealed_test_accessed'] or value['physical_fei_comparison_ready']:
        raise ValueError('Invalid scientific boundary')
    categories={'charged','mixed','ccbar','uubar','ddbar','ssbar'}
    arms={'refined_encoder','pre_refinement_encoder'}
    if set(value['arms'])!=arms or set(value['coverage']['views'])!=arms:
        raise ValueError('Incomplete arm coverage')
    for arm in arms:
        counts=value['coverage']['views'][arm]
        if set(counts)!=categories:raise ValueError('Incomplete category coverage')
        for scopes in counts.values():
            if set(scopes)!={'full','half'}:raise ValueError('Incomplete scope coverage')
            for row in scopes.values():
                if not (row['requested']==row['unique_attempted']==row['processed']==2000 and row['failed']==0):
                    raise ValueError('Quota not satisfied')
                if not 0<=row['truth_unavailable']<=row['processed']:
                    raise ValueError('Invalid truth availability count')
        beam=value['arms'][arm]['beam_diagnostic']['coverage']
        if beam['status']!='DIAGNOSTIC_NOT_QUOTA_COMPLIANT' or beam['processed_by_category']!=dict.fromkeys(categories,10):
            raise ValueError('Beam diagnostic scope changed')
        greedy=value['arms'][arm]['greedy']
        if set(greedy['summaries_by_source_category'])!=categories:
            raise ValueError('Tree category aggregates incomplete')
        for scope in ('full','half'):
            if greedy['summaries'][scope]['inference']['event_count']!=12000:
                raise ValueError('Tree collision count mismatch')
            if any(greedy['summaries_by_source_category'][cat][scope]['inference']['event_count']!=2000 for cat in categories):
                raise ValueError('Tree category count mismatch')
            tag=greedy['tag_efficiency']['summaries'][scope+'/greedy']
            for summary in (tag,tag['inclusive_fsp_grouping']):
                if summary['summary']['bbbar_trial_count']!=8000 or summary['summary']['event_count']!=12000:
                    raise ValueError('Nominal collision/B accounting changed')
                if set(summary['by_source_category'])!=categories or any(row['event_count']!=2000 for row in summary['by_source_category'].values()):
                    raise ValueError('Tag category coverage incomplete')
                if sum(c['evaluated_b_trials'] for c in summary['b_channel_coverage'])!=8000:
                    raise ValueError('Channel coverage incomplete')
    def event_data(item):
        if isinstance(item,dict):
            return any(k in {'event_uid','event_uids','source_file','b_units','truth_sources','predicted_sources','truth_root_position'} or event_data(v) for k,v in item.items())
        return any(event_data(v) for v in item) if isinstance(item,list) else False
    if event_data(value):raise ValueError('Private event data cannot publish')


def generate(root,output):
    source=root/SOURCE
    if not source.exists():return None
    for path in (source,source/'binding.json',*source.parents):
        if path.is_symlink():raise ValueError('Unsafe policy source')
    binding=json.loads((source/'binding.json').read_text())
    if binding['version']!='phase69-policy-dag-binding-v1' or not 1<=len(binding['parts'])<=MAX_PARTS:
        raise ValueError('Unknown or oversized policy binding')
    codec=sibling('wiki_phase69_efficiencies')
    nodes=[];contents=[]
    for i,part in enumerate(binding['parts']):
        if part['source']!=f'aggregates-{i}.json.gz' or part['filename']!=f'phase69-policy-part-{i}.json':
            raise ValueError('Invalid policy part identity')
        path=source/part['source']
        if path.is_symlink() or path.stat().st_size>MAX_PART_BYTES:raise ValueError('Unsafe policy part')
        with gzip.open(path,'rb') as handle:
            data=handle.read(MAX_PART_BYTES+1)
        if len(data)>MAX_PART_BYTES or len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:
            raise ValueError('Policy part binding or capacity mismatch')
        payload=json.loads(data)
        if payload['encoding']!='typed-json-dag-part-v1' or payload['part']!=i:
            raise ValueError('Policy part order mismatch')
        nodes.extend(payload['nodes']);contents.append((part['filename'],data))
    if len(nodes)!=binding['nodes']:raise ValueError('Policy DAG node count mismatch')
    if decoded_size(nodes,binding['root'])!=binding['decoded_bytes']:raise ValueError('Policy decoded byte count mismatch')
    encoded={'encoding':'typed-json-dag-v1','root':binding['root'],'nodes':nodes}
    value=codec.unpack(encoded)
    if hashlib.sha256(canonical(value)).hexdigest()!=binding['decoded_sha256']:
        raise ValueError('Policy decoded digest mismatch')
    validate(value)
    privacy=sibling('wiki_privacy')
    decoded=canonical(value).decode()
    if privacy._contains_private_fields(value) or privacy.redact(decoded)!=decoded:
        raise ValueError('Private policy aggregate fields')
    decoder=(Path(__file__).parents[1]/'wiki/phase69_policy_decoder.txt').read_bytes()
    main={'version':'phase69-policy-aggregate-download-v1','encoding':'typed-json-dag-parts-v1',
          'root':binding['root'],'nodes':binding['nodes'],'decoded_sha256':binding['decoded_sha256'],
          'parts':[{k:v for k,v in part.items() if k!='source'} for part in binding['parts']]}
    main_data=canonical(main)
    manifest={'version':'phase69-policy-integrity-v1','decoded_sha256':binding['decoded_sha256'],
              'files':[{'filename':BUNDLE,'sha256':hashlib.sha256(main_data).hexdigest(),'bytes':len(main_data)},
                       *main['parts'],
                       {'filename':DECODER,'sha256':hashlib.sha256(decoder).hexdigest(),'bytes':len(decoder)}],
              'source_binding_sha256':hashlib.sha256((source/'binding.json').read_bytes()).hexdigest(),
              'historical_evidence_preserved':True}
    for filename,content in ((BUNDLE,main_data),(INTEGRITY,canonical(manifest)),(DECODER,decoder),*contents):
        target=output/filename
        if target.is_symlink() or (target.exists() and target.stat().st_nlink!=1):raise ValueError('Unsafe policy destination')
        target.write_bytes(content)
    # Only compact review fields enter the dashboard status file; channels stay in the lossless download.
    return {'version':value['version'],'status':value['status'],'coverage':value['coverage'],
            'evaluator_revision':value['evaluator_revision'],'downloads':manifest['files'],
            'arms':{arm:{'tree':value['arms'][arm]['greedy']['summaries'],
                         'retained':value['arms'][arm]['greedy']['retained_tree_checks']['summaries'],
                         'tag':{scope:{'exact':value['arms'][arm]['greedy']['tag_efficiency']['summaries'][scope+'/greedy']['summary'],
                                      'inclusive':value['arms'][arm]['greedy']['tag_efficiency']['summaries'][scope+'/greedy']['inclusive_fsp_grouping']['summary']}
                                for scope in ('full','half')}} for arm in value['arms']}}


def render(record):
    if not record:return []
    def ratio(value):
        n,d=value['numerator'],value['denominator']
        return f'{n:g}/{d:g}' if d else 'UNAVAILABLE (0/0)'
    lines=['Phase69 supplementary category-policy evaluation','~'*48,'',
           'The frozen step4000 pair has now processed 2,000 distinct validation collisions in each of charged, mixed, ccbar, uubar, ddbar and ssbar: 12,000 per arm in each full/half view. The shared cohort excludes training, normalization and all 61,000 prior reservations through Phase70. Historical 100-event and 20-event reports remain unchanged.','',
           '.. list-table:: Frozen checkpoint comparison on the new cohort','   :header-rows: 1','',
           '   * - Endpoint','     - Refined encoder','     - Pre-refinement encoder']
    for scope in ('full','half'):
        for label,group,key in [('primary LCAG','tree','lcag_pair_accuracy'),('retained LCAG','retained','lcag_pair_accuracy'),('retained exact component','retained','perfect_lcag')]:
            vals=[]
            for arm in ('refined_encoder','pre_refinement_encoder'):
                data=record['arms'][arm][group]
                metric=data[scope]['decay_metrics'][key] if group=='tree' else data[scope+'/greedy'][key]
                vals.append(ratio(metric))
            lines += [f'   * - {scope} {label}',f'     - {vals[0]}',f'     - {vals[1]}']
        for label,kind in [('exact retained B tagging','exact'),('inclusive FSP B grouping','inclusive')]:
            vals=[ratio(record['arms'][arm]['tag'][scope][kind]['top1']['b_reconstruction']['per_b_correct']) for arm in ('refined_encoder','pre_refinement_encoder')]
            lines += [f'   * - {scope} {label}',f'     - {vals[0]}',f'     - {vals[1]}']
    lines += ['', 'A separately bounded width-two beam uses 60 collisions per arm (10 per category). Its model-ranked top1 and retained-pool/oracle results are diagnostic and do not meet the quota. Missing truth remains in nominal tagging denominators. These are retained proxies, not physical FEI efficiencies.','',
              f':download:`Supplementary full aggregate bundle <{BUNDLE}>`; :download:`integrity manifest <{INTEGRITY}>`; :download:`standalone decoder <{DECODER}>`.','',
              'The lossless bundle retains primary full/half trees, exact/inclusive per-B and per-event tagging, every evaluated channel/type, category coverage, continuum fake-B/component rates, beam top1/pool distinctions, search bounds, and paired collision uncertainty. See :doc:`../../phase69_policy` for interpretation.','']
    for part in record['downloads']:
        if part['filename'].startswith('phase69-policy-part-'):
            lines += [f":download:`Aggregate data {part['filename']} <{part['filename']}>`.",'']
    return lines
