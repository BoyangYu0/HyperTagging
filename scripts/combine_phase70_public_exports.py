"""Combine authenticated supplemental and native additive aggregates losslessly."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def numeric_count(value):
    if isinstance(value, dict):
        return sum(numeric_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(numeric_count(v) for v in value)
    return int(type(value) in (int, float, bool))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('phase70', root/'docs/_ext/wiki_phase70.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {'extended_paired_uncertainty': args.artifacts/'supplemental-v1/extended-paired-uncertainty.json',
             'supplemental': args.artifacts/'supplemental-v1/aggregate.json',
             'native_additive_tagging': args.artifacts/'additive/efficiencies.json.gz',
             'native_additional_diagnostics': args.artifacts/'native/public-additional-diagnostics-v3.json'}
    values, bindings = {}, {}
    for name, path in files.items():
        raw = path.read_bytes()
        data = gzip.decompress(raw) if path.suffix == '.gz' else raw
        values[name] = json.loads(data)
        bindings[name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                          'decoded_sha256': hashlib.sha256(data).hexdigest(),
                          'numeric_records': numeric_count(values[name]), 'bytes': len(raw)}
    result = values.pop('supplemental')
    module.validate(result)
    result.update(values)
    result['complete_export_cardinality'] = bindings
    module.validate(result)
    privacy = module.sibling('wiki_privacy')
    data = module.canonical(result)
    if privacy._contains_private_fields(result) or privacy.redact(data.decode()) != data.decode():
        raise ValueError('Private content in combined public exports')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'control').mkdir()
    (args.output/'aggregate.json').write_bytes(data)
    (args.output/'combination-manifest.json').write_bytes(module.canonical({
        'version': 'phase70-complete-public-combination-v1', 'inputs': bindings,
        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
        'numeric_records': numeric_count(result), 'private_event_records_published': False}))
    print(json.dumps({'bytes': len(data), 'numeric_records': numeric_count(result)}))


if __name__ == '__main__':
    main()
