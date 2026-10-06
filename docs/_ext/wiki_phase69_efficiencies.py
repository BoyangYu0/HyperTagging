"""Lossless additive Phase69 publication; original downloads remain byte-identical."""
import gzip
import hashlib
import json
from pathlib import Path

SOURCE = 'artifacts/codex/phase69_efficiencies_20261006/efficiencies.json.gz'
BUNDLE = 'phase69-complete-aggregate-metrics-v3.json'
MANIFEST = 'phase69-efficiencies-integrity.json'
DECODER = 'phase69-efficiencies-decoder.txt'


def pack(value):
    """Typed, topologically ordered DAG; share repeated channel tables losslessly."""
    nodes, lookup = [], {}
    def visit(item):
        if isinstance(item, dict):
            node = ['dict', [[visit(key), visit(val)] for key, val in sorted(item.items())]]
        elif isinstance(item, list):
            node = ['list', [visit(val) for val in item]]
        elif item is None:
            node = ['null']
        elif type(item) in (str, bool, int, float):
            node = [type(item).__name__, item]
        else:
            raise ValueError('Unsupported aggregate scalar')
        identity = json.dumps(node, separators=(',', ':'), allow_nan=False)
        if identity not in lookup:
            lookup[identity] = len(nodes)
            nodes.append(node)
        return lookup[identity]
    root = visit(value)
    return {'encoding': 'typed-json-dag-v1', 'root': root, 'nodes': nodes}


def unpack(value):
    if set(value) != {'encoding', 'root', 'nodes'} or value['encoding'] != 'typed-json-dag-v1':
        raise ValueError('Unknown encoding')
    decoded = []
    for node in value['nodes']:
        kind = node[0]
        def ref(index):
            if type(index) is not int or not 0 <= index < len(decoded):
                raise ValueError('Invalid DAG reference')
            return decoded[index]
        if kind == 'dict':
            pairs = [(ref(key), ref(val)) for key, val in node[1]]
            if any(type(key) is not str for key, _ in pairs) or len({key for key, _ in pairs}) != len(pairs):
                raise ValueError('Invalid object keys')
            item = dict(pairs)
        elif kind == 'list':
            item = [ref(index) for index in node[1]]
        elif kind == 'null' and len(node) == 1:
            item = None
        elif kind in ('str', 'bool', 'int', 'float') and len(node) == 2 and type(node[1]).__name__ == kind:
            item = node[1]
        else:
            raise ValueError('Invalid typed node')
        decoded.append(item)
    root = value['root']
    if type(root) is not int or not 0 <= root < len(decoded):
        raise ValueError('Invalid root')
    return decoded[root]


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def generate(root, output, native_bundle):
    import importlib.util
    spec = importlib.util.spec_from_file_location("wiki_privacy", Path(__file__).with_name("wiki_privacy.py"))
    privacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(privacy)
    path = root / SOURCE
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Unsafe efficiency source')
    binding = json.loads(path.with_name('binding.json').read_text())
    data = gzip.decompress(path.read_bytes())
    if len(data) != binding['bytes'] or hashlib.sha256(data).hexdigest() != binding['sha256']:
        raise ValueError('Efficiency export binding mismatch')
    value = json.loads(data)
    if value['version'] != 'phase69-additive-tag-efficiencies-v1' or value['counts'] != binding['counts']:
        raise ValueError('Efficiency export schema mismatch')
    counts = value['counts']
    if (counts['registered_reports'] != 14 or counts['event_scope_records'] != 2480
            or counts['retained_candidate_scope_records'] != 262 or len(value['views']) != 7
            or any(set(arms) != {'refined_encoder', 'pre_refinement_encoder'} for arms in value['views'].values())):
        raise ValueError('Incomplete efficiency coverage')
    def forbidden(item):
        if isinstance(item, dict):
            return any(key in {'event_uid', 'event_uids', 'source_file', 'b_units', 'truth_root_position'}
                       or forbidden(val) for key, val in item.items())
        return any(forbidden(val) for val in item) if isinstance(item, list) else False
    if forbidden(value):
        raise ValueError('Event-level efficiency data cannot publish')
    if privacy._contains_private_fields(value) or privacy.redact(data.decode()) != data.decode():
        raise ValueError('Private efficiency export')
    encoded = pack(value)
    if canonical(unpack(encoded)) != data:
        raise ValueError('Efficiency encoding is not lossless')
    bundle = {'version': 'phase69-complete-aggregate-bundle-v3',
              'native_aggregate_bundle': native_bundle,
              'native_aggregate_bundle_sha256': hashlib.sha256(canonical(native_bundle)).hexdigest(),
              'additive_efficiencies': encoded,
              'additive_decoded_sha256': binding['sha256'],
              'additive_counts': binding['counts']}
    exported = canonical(bundle)
    if len(exported) >= 9 * 1024 * 1024:
        raise ValueError('Efficiency bundle requires capacity review')
    # Project only this reviewed public decoder, as a scanned text download.
    # Arbitrary source-script staging remains forbidden.
    decoder_path = root / 'docs/wiki/phase69_efficiencies_decoder.py'
    if any(p.is_symlink() for p in (decoder_path, *decoder_path.parents)):
        raise ValueError('Unsafe decoder source')
    decoder_data = decoder_path.read_bytes()
    if privacy.redact(decoder_data.decode()) != decoder_data.decode():
        raise ValueError('Private decoder text')
    manifest = {'version': 'phase69-efficiencies-integrity-v1', 'files': [
        {'filename': BUNDLE, 'bytes': len(exported), 'sha256': hashlib.sha256(exported).hexdigest()}],
        'decoded_efficiencies': binding, 'native_bundle_preserved': True,
        'decoder': {'filename': DECODER,
                    'sha256': hashlib.sha256(decoder_data).hexdigest(),
                    'bytes': len(decoder_data)}}
    for filename, content in ((BUNDLE, exported), (MANIFEST, canonical(manifest)), (DECODER, decoder_data)):
        target = output / filename
        if target.is_symlink() or (target.exists() and target.stat().st_nlink != 1):
            raise ValueError('Unsafe efficiency destination')
        target.write_bytes(content)
    return manifest
