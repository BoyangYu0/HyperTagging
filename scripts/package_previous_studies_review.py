"""Publish complete supported Phase79–83 metrics with bounded lossless transport.

Private identities and operational locations are excluded; numeric metrics, list
order/cardinality, histories, event records, and uncertainty are retained. Run
with an explicit private evidence root. Never launches scientific work.
"""
from __future__ import annotations
import argparse
import base64
import lzma
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'e2a17446d1854e2043073d6e1e03b4a530343773'
LIMIT = 5_000_000

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'docs/_ext'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sources(base):
    phases = {n: next(base.glob(f'phase{n}_*_20261009')) for n in range(79,84)}
    chosen = []
    for n in (79,82):
        chosen += [(n, phases[n]/'diagnostic-v1'/f) for f in ('summary.json','event-metrics-private.json')]
    chosen += [(80,p) for p in sorted((phases[80]/'diagnostic-v1').glob('*.json')) if p.name!='selection-private.json']
    chosen += [(81,phases[81]/f) for f in ('training-screen-review.json','pair-separability-review.json','validation-and-corrections.json','initialization-verification.json')]
    for arm in ('native','pair_supervised'):
        chosen += [(81,phases[81]/'campaign-v1/runs'/arm/f) for f in ('summary.json','tiny-evaluation.json','downstream-evaluation.json','terminal.json','downstream-metrics.jsonl','tiny-metrics.jsonl')]
    chosen += [(83,phases[83]/'terminal-metrics-review.json'),(83,phases[83]/'replay-failure-analysis.json')]
    chosen += [(83,p) for p in sorted((phases[83]/'replay-corrected-v2/diagnostic-v1').glob('*.json')) if p.name!='selection-private.json']
    for n,p in chosen:
        yield f'phase{n}.'+str(p.relative_to(phases[n])).replace('/','.'),p

# Operational leaves are not scientific endpoints. Keep numeric counts such as
# unique_optimizer_identities; remove only actual UID values and path metadata.
EXCLUDE = re.compile(r'^(?:uid|uids|event_uid|event_uids|job_id|failed_attempt_job_id|scheduler_id|hostname|username|source_root|output_root|path|checkpoint_path|authority_path|NodeList|JobID)$',re.I)

def sanitize(value, privacy, exclusions, location=()):
    if isinstance(value, dict):
        result = {}
        for key,item in value.items():
            if EXCLUDE.fullmatch(key) or privacy._SENSITIVE_KEY.search(key):
                exclusions['excluded_key:'+key] += 1
                continue
            newkey = privacy.redact(key)
            if newkey != key:
                exclusions['redacted_dictionary_key'] += 1
                # Unique stable ordinal keeps all values even for private keys.
                newkey = f'private_key_ordinal_{len(result)}'
                while newkey in value or newkey in result:newkey+='_'
            result[newkey] = sanitize(item,privacy,exclusions,location+(key,))
        return result
    if isinstance(value,list):
        return [sanitize(x,privacy,exclusions,location+(i,)) for i,x in enumerate(value)]
    if isinstance(value,str):
        # Event UIDs are not covered by generic publication privacy regexes.
        out = re.sub(r'(?<!\d)\d+:\d+:\d+:\d+(?!\d)','[redacted event identity]',value)
        out = privacy.redact(out)
        if out != value: exclusions['redacted_string_value'] += 1
        return out
    return value

def stats(value):
    c=Counter()
    def walk(v):
        if isinstance(v,dict):
            c['objects']+=1;c['object_fields']+=len(v)
            for x in v.values():walk(x)
        elif isinstance(v,list):
            c['arrays']+=1;c['array_items']+=len(v)
            for x in v:walk(x)
        else:
            c['scalars']+=1
            if isinstance(v,(int,float)) and not isinstance(v,bool):c['numeric_scalars']+=1
    walk(value);return dict(c)

def fragments(source,value,path=()):
    entry={'source':source,'path':list(path),'value':value}
    if len(canonical(entry))<4_500_000:
        yield entry;return
    if isinstance(value,dict):
        for key,item in value.items():yield from fragments(source,item,path+(key,))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from fragments(source,item,path+(index,))
    else:raise ValueError('Oversize scalar')

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'artifacts/codex/previous_studies_review_20261009');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    privacy=module('wiki_privacy');codec=module('wiki_phase72_compact')
    from functools import lru_cache
    privacy.redact=lru_cache(maxsize=100000)(privacy.redact)
    inventory=[];parts=[];current=[];current_size=100
    public_sources={}
    subtree_counts=Counter()
    def count_subtrees(value):
        if isinstance(value,(dict,list)):
            data=canonical(value)
            if len(data)>=128:subtree_counts[digest(data)]+=1
            for child in (value.values() if isinstance(value,dict) else value):count_subtrees(child)
    # Reuse exact already-public Phase78 subtrees without rewriting historical
    # downloads. The public decoder resolves only these four authenticated
    # adjacent envelope files, never private source locations.
    external_subtrees={};external_documents={};external_sources=[]
    historical=ROOT/'artifacts/codex/phase78_review_20261009'
    def index_external(value,filename,path=()):
        if isinstance(value,(dict,list)):
            data=canonical(value)
            if len(data)>=128:
                external_subtrees.setdefault(digest(data),(filename,list(path)))
            for key,child in (value.items() if isinstance(value,dict) else enumerate(value)):
                index_external(child,filename,path+(key,))
    historical_names=['phase78-events-0.json','phase78-events-1.json','phase78-events-2.json','phase78-review.json']
    for filename in historical_names:
        historical_file=historical/(filename+'.gz')
        packed=historical_file.read_bytes()
        frozen=subprocess.run(['git','show',SOURCE_SHA+':'+historical_file.relative_to(ROOT).as_posix()],cwd=ROOT,check=True,capture_output=True).stdout
        if frozen!=packed:raise ValueError('Historical public bytes differ from frozen source')
        decoder=__import__('zlib').decompressobj(31)
        data=decoder.decompress(packed,LIMIT+1)
        if len(data)>LIMIT or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('Historical bounded gzip mismatch')
        envelope=codec.encode(data,packed,LIMIT)
        external_sources.append({'filename':filename,'envelope_sha256':digest(envelope),'decoded_sha256':digest(data),'decoded_bytes':len(data)})
        value=json.loads(data);external_documents[filename]=value
        index_external(value,filename)
    external_definitions=[];external_indices={}
    def externalize(value):
        if not isinstance(value,(dict,list)):return value
        data=canonical(value);key=digest(data)
        if len(data)>=128 and key in external_subtrees:
            if key not in external_indices:
                external_indices[key]=len(external_definitions)
                filename,path=external_subtrees[key]
                external_definitions.append({'file':filename,'path':path,'sha256':key})
            return {'$external':external_indices[key]}
        return {k:externalize(v) for k,v in value.items()} if isinstance(value,dict) else [externalize(v) for v in value]
    definitions={}
    def intern(value):
        if not isinstance(value,(dict,list)):return value
        data=canonical(value);key=digest(data)
        if len(data)>=128 and subtree_counts[key]>1:
            if key not in definitions:
                index=len(definitions);definitions[key]={'index':index,'value':None}
                transformed={k:intern(v) for k,v in value.items()} if isinstance(value,dict) else [intern(v) for v in value]
                definitions[key]['value']=transformed
            return {'$ref':definitions[key]['index']}
        return {k:intern(v) for k,v in value.items()} if isinstance(value,dict) else [intern(v) for v in value]
    def flush():
        nonlocal current,current_size
        if not current:return
        payload={'version':'previous-studies-lossless-fragments-v1','fragments':current}
        fragment_data=canonical(payload)
        # Keep the original fragment JSON under the same privacy and 5MB limits.
        codec.encode(fragment_data,gzip.compress(fragment_data,compresslevel=9,mtime=0),LIMIT)
        inner=lzma.compress(fragment_data,format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,preset=6)
        inner_decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=64*1024*1024)
        restored=inner_decoder.decompress(inner,max_length=LIMIT+1)
        if restored!=fragment_data or not inner_decoder.eof or inner_decoder.unused_data or inner_decoder.check!=lzma.CHECK_CRC64:
            raise ValueError('Bounded inner XZ roundtrip failed')
        wrapper={'encoding':'bounded-xz-base32-fragments-v1','decoded_bytes':len(fragment_data),'decoded_sha256':digest(fragment_data),'data':base64.b32encode(inner).decode('ascii')}
        data=canonical(wrapper);packed=gzip.compress(data,compresslevel=9,mtime=0)
        codec.encode(data,packed,LIMIT)
        name=f'previous-studies-metrics-{len(parts):03d}.json.gz'
        (args.output/name).write_bytes(packed)
        parts.append({'file':name,'compressed_sha256':digest(packed),'decoded_sha256':digest(data),'decoded_bytes':len(data),'fragment_decoded_bytes':len(fragment_data),'fragment_decoded_sha256':digest(fragment_data),'fragments':len(current)})
        current=[];current_size=100
    evidence_manifest=json.loads((args.evidence_root/'phase78_partition_20261009/sequential-improvement-v1/final-manifest.json').read_text())
    authenticated={entry['path']:entry['sha256'] for entry in evidence_manifest['artifacts']}
    for logical,path in sources(args.evidence_root):
        raw=path.read_bytes();original=[json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix=='.jsonl' else json.loads(raw);excluded=Counter();value=sanitize(original,privacy,excluded)
        data=canonical(value)
        if str(path) not in authenticated or digest(raw)!=authenticated[str(path)]:
            raise ValueError('Metric evidence missing or changed from frozen final manifest: '+logical)
        before=stats(original);after=stats(value)
        inventory.append({'source':logical,'evidence_sha256':digest(raw),'sanitized_sha256':digest(data),'sanitized_bytes':len(data),'original_cardinality':before,'public_cardinality':after,'exclusions':[{'field_or_reason':k,'occurrences':v} for k,v in sorted(excluded.items())]})
        public_sources[logical]=value
    reduced_sources={logical:externalize(value) for logical,value in public_sources.items()}
    for value in reduced_sources.values():count_subtrees(value)
    transformed={logical:intern(value) for logical,value in reduced_sources.items()}
    transformed['shared-subtree-definitions']=[item['value'] for item in definitions.values()]
    transformed['external-subtree-definitions']=external_definitions
    shared=transformed['shared-subtree-definitions']
    @lru_cache(maxsize=None)
    def expand_ref(index):
        return expand(shared[index])
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={'$ref'}:return expand_ref(value['$ref'])
            if set(value)=={'$external'}:
                entry=external_definitions[value['$external']]
                result=external_documents[entry['file']]
                for key in entry['path']:result=result[key]
                if digest(canonical(result))!=entry['sha256']:raise ValueError('External subtree hash mismatch')
                return result
            return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        return value
    for logical,original in public_sources.items():
        if canonical(expand(transformed[logical]))!=canonical(original):
            raise ValueError('Lossless shared-subtree roundtrip failed')
    for logical,value in transformed.items():
        for fragment in fragments(logical,value):
            size=len(canonical(fragment))+1
            if current_size+size>4_950_000:flush()
            current.append(fragment);current_size+=size
    flush()
    inv={'version':'previous-studies-public-inventory-v1','metric_sources':inventory,'external_sources':external_sources,'external_reference_count':len(external_definitions),'deduplication':{'format':'exact-shared-subtrees-v1','definitions':len(definitions),'reference_key':'$ref','definitions_source':'shared-subtree-definitions'},'policy':'Only actual identities, operational metadata and private string substrings excluded; all remaining metric values and array order retained. Fragment paths reconstruct intermediate sources; replace each singleton $ref object with its indexed shared-subtree definition recursively and resolve singleton $external objects by authenticated historical public envelope plus JSON path and subtree hash to reconstruct exact sanitized sources. Historical files remain unchanged. No physics endpoints synthesized.'}
    invdata=canonical(inv);invpacked=gzip.compress(invdata,compresslevel=9,mtime=0)
    (args.output/'inventory.json.gz').write_bytes(invpacked)
    binding={'version':'previous-studies-public-binding-v1','source_sha':SOURCE_SHA,'source_role':'training','first_phase':79,'last_phase':83,'fresh_validation_events':0,'primary_eligible':False,'parts':parts,'inner_encoding':'bounded-xz-base32-fragments-v1','external_sources':external_sources,'inventory':{'file':'inventory.json.gz','compressed_sha256':digest(invpacked),'decoded_bytes':len(invdata),'sha256':digest(invdata)},'public_summary':{'phase79':{'probe_auc':0.549412,'native_auc':0.545198,'difference_ci95':[-0.00671,0.01409]},'phase80':{'harmful_native_steps':12,'harmful_half_steps':12,'steps_per_condition':12,'exact_numerator':0,'exact_denominator':64},'phase81':{'native_raw_exact':0,'pair_raw_exact':0,'native_accepted_exact':0,'pair_accepted_exact':0,'b_trials':1024,'b_collisions':512,'main_events':1536,'events_per_category':256,'native_proposal_correct':3437,'pair_proposal_correct':4304,'b_nodes':9437,'native_proposal_background':2981,'pair_proposal_background':5362,'background_nodes':52590,'tiny_raw':32,'tiny_trials':32,'tiny_accepted':27},'phase82':{'nonlinear_auc':0.534863,'native_auc':0.545198,'difference_ci95':[-0.025873,0.004259],'readability_gate':False},'phase83':{'qualifying_steps':0,'required_steps':9,'steps':12,'raw_exact':0,'accepted_exact':0,'b_trials':64,'audit_events':96,'events_per_category':16,'scientific_benefit':False},'scheduler_cpu_seconds_including_failures':7596.282,'exposure_confound':'tiny8000 presentations across24 views versus main12000 across1536 remains unresolved; not proof of undertraining','physical_trees_p4_beam_endpoints':'unavailable','all_zero_intervals':'do not prove equivalence'}}
    (args.output/'binding.json').write_bytes(canonical(binding))
    print(json.dumps({'sources':len(inventory),'parts':len(parts),'gzip_bytes':sum((args.output/x['file']).stat().st_size for x in parts),'decoded_bytes':sum(x['decoded_bytes'] for x in parts)},indent=2))
if __name__=='__main__':main()
