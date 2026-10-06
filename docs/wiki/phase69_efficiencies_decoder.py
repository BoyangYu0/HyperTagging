"""Decode the complete Phase69 v3 JSON bundle without project dependencies.
Usage: python phase69-efficiencies-decoder.txt bundle.json decoded.json
Preserves the original native v2 bundle and expands additive efficiencies.
"""
import hashlib
import json
import sys
from pathlib import Path

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


if __name__ == '__main__':
    bundle = json.loads(Path(sys.argv[1]).read_text())
    assert bundle['version'] == 'phase69-complete-aggregate-bundle-v3'
    native = (json.dumps(bundle['native_aggregate_bundle'], sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    assert hashlib.sha256(native).hexdigest() == bundle['native_aggregate_bundle_sha256']
    value = unpack(bundle['additive_efficiencies'])
    canonical = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    assert hashlib.sha256(canonical).hexdigest() == bundle['additive_decoded_sha256']
    result = {'native_aggregate_bundle': bundle['native_aggregate_bundle'], 'additive_efficiencies': value}
    with Path(sys.argv[2]).open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
