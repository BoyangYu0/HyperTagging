"""Bounded aggregate-only publication of the supplementary Phase72 policy study."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = 'artifacts/codex/phase72_policy_reevaluation_20261008'
BUNDLE = 'phase72-policy-reevaluation-v1.json'
INTEGRITY = 'phase72-policy-integrity.json'
DECODER = 'phase72-policy-decoder.txt'
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
    if value['version']!='phase72-policy-evaluation-v1-20261007' or value['status']!='COMPLETE_GREEDY_WITH_DIAGNOSTIC_BEAM':
        raise ValueError('Unknown policy study publication')
    if value['historical_reports_replaced'] or value['sealed_test_accessed'] or value['physical_fei_comparison_ready']:
        raise ValueError('Invalid scientific boundary')
    categories={'charged','mixed','ccbar','uubar','ddbar','ssbar'}
    arms={'set_overlap_off','set_overlap_on'}
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
    if binding['version']!='phase72-policy-dag-binding-v1' or not 1<=len(binding['parts'])<=MAX_PARTS:
        raise ValueError('Unknown or oversized policy binding')
    codec=sibling('wiki_phase69_efficiencies')
    nodes=[];contents=[]
    for i,part in enumerate(binding['parts']):
        if part['source']!=f'aggregates-{i}.json.gz' or part['filename']!=f'phase72-policy-part-{i}.json':
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
    decoder=(Path(__file__).parents[1]/'wiki/phase72_policy_decoder.txt').read_bytes()
    main={'version':'phase72-policy-aggregate-download-v1','encoding':'typed-json-dag-parts-v1',
          'root':binding['root'],'nodes':binding['nodes'],'decoded_sha256':binding['decoded_sha256'],
          'parts':[{k:v for k,v in part.items() if k!='source'} for part in binding['parts']]}
    main_data=canonical(main)
    manifest={'version':'phase72-policy-integrity-v1','decoded_sha256':binding['decoded_sha256'],
              'files':[{'filename':BUNDLE,'sha256':hashlib.sha256(main_data).hexdigest(),'bytes':len(main_data)},
                       *main['parts'],
                       {'filename':DECODER,'sha256':hashlib.sha256(decoder).hexdigest(),'bytes':len(decoder)}],
              'source_binding_sha256':hashlib.sha256((source/'binding.json').read_bytes()).hexdigest(),
              'historical_evidence_preserved':True}
    for filename,content in ((BUNDLE,main_data),(INTEGRITY,canonical(manifest)),(DECODER,decoder),*contents):
        target=output/filename
        if target.is_symlink() or (target.exists() and target.stat().st_nlink!=1):raise ValueError('Unsafe policy destination')
        target.write_bytes(content)
    native_source=root/'artifacts/codex/phase72_native_20261008'
    native_binding=json.loads((native_source/'binding.json').read_text())
    with gzip.open(native_source/'native.json.gz','rb') as stream:native_data=stream.read(10*1024*1024+1)
    if len(native_data)!=native_binding['bytes'] or hashlib.sha256(native_data).hexdigest()!=native_binding['sha256'] or len(native_data)>10*1024*1024:
        raise ValueError('Native binding or capacity mismatch')
    native=json.loads(native_data)
    if native['version']!='phase72-native-aggregate-bundle-v1' or native['arms']!=['set_overlap_off','set_overlap_on'] or len(native['records'])!=native_binding['scalar_records']:
        raise ValueError('Native scalar coverage mismatch')
    identities={(v,a,m) for v,a,m,x in native['records']}
    if len(identities)!=len(native['records']):raise ValueError('Duplicate native metric')
    if privacy._contains_private_fields(native) or privacy.redact(native_data.decode())!=native_data.decode():raise ValueError('Private native aggregate')
    (output/'phase72-native-aggregates.json').write_bytes(native_data)
    native_decoder=(Path(__file__).parents[1]/'wiki/phase72_native_decoder.txt').read_bytes()
    (output/'phase72-native-decoder.txt').write_bytes(native_decoder)
    (output/'phase72-native-integrity.json').write_bytes(canonical({**native_binding,'decoder':{'filename':'phase72-native-decoder.txt','sha256':hashlib.sha256(native_decoder).hexdigest(),'bytes':len(native_decoder)}}))
    scalar_files=[]
    for folder in ('phase72_histories_20261008','phase72_checkpoint_scalars_20261008'):
        scalar_source=root/'artifacts/codex'/folder
        scalar_binding=json.loads((scalar_source/'binding.json').read_text())
        if scalar_binding['version']!='phase72-training-scalar-binding-v1' or len(scalar_binding['files'])!=2:
            raise ValueError('Invalid scalar binding')
        for item in scalar_binding['files']:
            if item['source'] not in ('set_overlap_off.json.gz','set_overlap_on.json.gz'):
                raise ValueError('Invalid scalar source')
            allowed_names={f'phase72-set-overlap-{arm}-{kind}-scalars.json' for arm in ('off','on') for kind in ('training','checkpoint')}
            if item['filename'] not in allowed_names:raise ValueError('Invalid scalar destination')
            path=scalar_source/item['source']
            if path.is_symlink():raise ValueError('Unsafe scalar source')
            with gzip.open(path,'rb') as stream:data=stream.read(10*1024*1024+1)
            if len(data)>10*1024*1024 or len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('Scalar binding or capacity mismatch')
            value_scalar=json.loads(data)
            if value_scalar['scalar_count']!=item['scalar_records'] or value_scalar['decoded_scalar_sha256']!=item['decoded_scalar_sha256']:
                raise ValueError('Scalar cardinality mismatch')
            if privacy._contains_private_fields(value_scalar) or privacy.redact(data.decode())!=data.decode():raise ValueError('Private scalar fields')
            (output/item['filename']).write_bytes(data)
            scalar_files.append({k:v for k,v in item.items() if k!='source'})
    scalar_decoder=(Path(__file__).parents[1]/'wiki/phase72_scalar_decoder.txt').read_bytes()
    (output/'phase72-scalar-decoder.txt').write_bytes(scalar_decoder)
    (output/'phase72-scalar-integrity.json').write_bytes(canonical({'version':'phase72-all-scalar-integrity-v1','files':scalar_files,'decoder_sha256':hashlib.sha256(scalar_decoder).hexdigest()}))
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
    def ratio(v):return f"{v['numerator']:g}/{v['denominator']:g}" if v['denominator'] else 'UNAVAILABLE'
    lines=['Phase72 efficiency evaluation','-----------------------------','',
      'The same immutable cohort supplies 2,000 distinct processed collisions in each required category, 12,000 per arm in full and half scope. The 4,000 B-pair collisions give 8,000 nominal B trials. Missing truth remains in nominal denominators. These retained proxies are not physical FEI efficiencies.','',
      '.. list-table:: Primary strict greedy tagging','   :header-rows: 1','',
      '   * - Endpoint','     - Set overlap off','     - Set overlap on']
    for scope in ('full','half'):
        for kind in ('exact','inclusive'):
            for key in ('per_b_correct','event_any_correct','event_both_correct'):
                vals=[ratio(record['arms'][arm]['tag'][scope][kind]['top1']['b_reconstruction'][key]) for arm in ('set_overlap_off','set_overlap_on')]
                lines += [f'   * - {scope} {kind} {key}',f'     - {vals[0]}',f'     - {vals[1]}']
    lines += ['', 'The 60-collision width-two beam (10 per category) is diagnostic only. All native 100-event reports and 20-event proposal-beam evidence remain diagnostic and retain their original gates. Full and half views do not multiply sample size. See :doc:`../../phase72` for continuum, channel, uncertainty and study decisions.','',
              f':download:`Complete supplemental aggregates <{BUNDLE}>`; :download:`integrity <{INTEGRITY}>`; :download:`standalone decoder <{DECODER}>`.','',
              ':download:`All native training/checkpoint scalars and decoder <phase72-scalar-integrity.json>`; :download:`scalar decoder <phase72-scalar-decoder.txt>`. Full file links are in the download catalogue.', '',
              ':download:`Complete native aggregate scalar records <phase72-native-aggregates.json>`; :download:`native integrity <phase72-native-integrity.json>`; :download:`native scalar decoder <phase72-native-decoder.txt>`.','']
    for part in record['downloads']:
        if part['filename'].startswith('phase72-policy-part-'):
            lines += [f":download:`{part['filename']} <{part['filename']}>`.",'']
    return lines
