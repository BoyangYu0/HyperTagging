"""Authenticated, bounded CR032 aggregate publication; no private source reads."""
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = 'artifacts/codex/cr032_review_20261015'
METRIC_FILES = {'cr032-summary.json', 'cr009-metrics.json', 'cr009-paired.json',
         'cr009-resources.json', 'cr032-inventory.json'}
FILES = METRIC_FILES | {'cr032-decoder.txt'}
LIMIT = 5_000_000
SOURCE_SHA = '6e1b5bd4a34a9a70174baf1099a01847ba7c0a1f'


def _codec():
    spec = importlib.util.spec_from_file_location('cr032_compact', Path(__file__).with_name('wiki_phase72_compact.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def generate(root, output):
    """Preserve recorded gzip bytes and publish existing privacy-checked codec."""
    source = root / SOURCE
    binding_path = source / 'binding.json'
    if binding_path.is_symlink() or binding_path.stat().st_size > 32_768:
        raise ValueError('Unsafe CR032 binding')
    binding = json.loads(binding_path.read_bytes())
    if (binding['schema'] != 'cr032-public-binding-v1' or binding['source_sha'] != SOURCE_SHA
            or binding['no_new_science'] is not True
            or binding['complete_post_study_evaluation'] is not False
            or len(binding['files']) != len(METRIC_FILES)
            or {row['download'] for row in binding['files']} != METRIC_FILES):
        raise ValueError('CR032 publication scope differs')
    codec = _codec()
    documents, downloads = {}, []
    for row in binding['files']:
        name = row['download']
        expected_file = ('inventory' if name == 'cr032-inventory.json' else name[:-5])+'.json.gz'
        if row['file'] != expected_file:
            raise ValueError('Unexpected CR032 source filename')
        path = source / row['file']
        if path.is_symlink() or path.stat().st_size > LIMIT:
            raise ValueError('Unsafe CR032 metric file')
        packed = path.read_bytes()
        if len(packed) != row['compressed_bytes'] or hashlib.sha256(packed).hexdigest() != row['compressed_sha256']:
            raise ValueError('CR032 compressed binding differs')
        # decode enforces exact expansion size, stream completion and digest.
        import base64
        data = codec.decode({'encoding': codec.ENCODING, 'data': base64.b32encode(packed).decode('ascii'),
                             'decoded_bytes': row['decoded_bytes'], 'decoded_sha256': row['decoded_sha256']}, LIMIT)
        envelope = codec.encode(data, packed, LIMIT)
        if len(envelope) != row['envelope_bytes'] or hashlib.sha256(envelope).hexdigest() != row['envelope_sha256']:
            raise ValueError('CR032 envelope binding differs')
        documents[name] = json.loads(data)
        downloads.append({'filename': name, 'bytes': len(envelope), 'sha256': row['envelope_sha256']})
        (output / name).write_bytes(envelope)
    (output / 'cr032-decoder.txt').write_text(DECODER)
    downloads.append({'filename': 'cr032-decoder.txt', 'bytes': len(DECODER.encode()),
                      'sha256': hashlib.sha256(DECODER.encode()).hexdigest()})
    summary = documents['cr032-summary.json']
    if (summary['summary'] != binding['public_summary']
            or summary['summary']['accepted_upper'] != 50
            or summary['summary']['eligible'] != 64
            or summary['summary']['processed'] != 64
            or summary['summary']['failed'] != 0
            or summary['summary']['membership_unavailable'] != 0
            or summary['summary']['raw_upper'] != 64
            or summary['summary']['raw_observed'] != 1
            or summary['summary']['accepted_observed'] != 1
            or summary['limits']['tiny_gate_accepted'] != 61
            or summary['limits']['tiny_gate_raw'] != 61
            or summary['limits']['fresh_evaluation']
            or summary['charge_obstruction_histogram'] != {'-3': 2, '-2': 5, '2': 6, '4': 1}
            or summary['proof_population'] != {'units': 64, 'events': 32}):
        raise ValueError('CR032 scientific scope differs')
    return {'summary': summary, 'context': {'reports': {key: {field: report[field] for field in
                ('raw_correct', 'accepted_correct', 'scientific_gate')}
                for key, report in documents['cr009-paired.json']['reports'].items()},
                'seed_uncertainty': documents['cr009-paired.json']['seed_uncertainty']},
            'metric_downloads': downloads, 'source_sha': SOURCE_SHA}


def render(value):
    return [
        'CR032: fixed acceptance gate infeasible',
        '---------------------------------------', '',
        'All 64 nominal B trials in 32 TRAIN collisions were processed; none failed or lacked membership truth.',
        'Fourteen target source groups have immutable detector charge outside the allowed mother-charge union',
        'of −1, 0 and +1. Exact daughter sums therefore cap accepted recovery at 50/64 (78.125%),',
        'below the unchanged 61/64 gate. The raw bound remains 64/64; the remaining 50 are not proved reachable.', '',
        'Archived mixed-control recovery remains 1/64 raw and accepted. This is a fixed-panel feasibility',
        'finding, not improved reconstruction or a population estimate. The original denominator and physics',
        'constraints remain unchanged. Further tuning solely to pass this impossible gate is not justified.', '',
        'The completed CR009 sampling comparison provides context: mixed recovery was 1/64 raw and accepted;',
        'generic-B recovery was 5/64 raw and 3/64 accepted, with fake-event acceptance rising from 10/64 to 27/64.',
        'All original tiny gates failed. These are adaptive single-seed TRAIN diagnostics, not fresh confirmation.', '',
        'See :doc:`../../cr032` for complete interpretation, scoped counts, background and uncertainty.',
        'Downloads use the existing bounded gzip/base32 JSON envelope; decoded aggregate field schemas and source hashes',
        'are in the inventory. Private event and source identities are excluded.', '',
        *[':download:`'+label+' <'+name+'>`.' for name, label in (
            ('cr032-summary.json', 'CR032 complete aggregate certificate and coverage tables'),
            ('cr009-metrics.json', 'CR009 complete contextual metric and exposure tables'),
            ('cr009-paired.json', 'CR009 saved paired uncertainty and tagging reports'),
            ('cr009-resources.json', 'CR009 resource accounting including the failed attempt'),
            ('cr032-inventory.json', 'Explicit public field schema, source hashes and exclusions'),
            ('cr032-decoder.txt', 'Standalone bounded decoder'))], '',
    ]


DECODER = r'''# Usage: python cr032-decoder.txt compact-download.json restored.json
# Standard library only. Output is created exclusively and never overwritten.
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

LIMIT = 5_000_000

def decode(raw):
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError('Input byte limit')
    value = json.loads(raw)
    if set(value) != {'encoding', 'decoded_bytes', 'decoded_sha256', 'data'} or value['encoding'] != 'bounded-gzip-base32-json-v1':
        raise ValueError('Unknown compact envelope')
    size = value['decoded_bytes']
    if type(size) is not int or not 0 <= size <= LIMIT or not isinstance(value['data'], str) or len(value['data']) > 10 * 1024 * 1024:
        raise ValueError('Compact byte limit')
    packed = base64.b32decode(value['data'], casefold=False)
    decoder = zlib.decompressobj(31)
    decoded = decoder.decompress(packed, size + 1)
    if len(decoded) != size or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError('Compact expansion or stream mismatch')
    if hashlib.sha256(decoded).hexdigest() != value['decoded_sha256']:
        raise ValueError('Compact decoded digest mismatch')
    return decoded

if __name__ == '__main__':
    source, output = map(Path, sys.argv[1:])
    with source.open('rb') as stream:
        raw = stream.read(10 * 1024 * 1024 + 1)
    decoded = decode(raw)
    with output.open('xb') as stream:
        stream.write(decoded)
'''
