"""Necessary cohort exclusion gate; never grants runtime or submission admission.

Authenticate a declared development cohort and every supplied reservation before
counting overlap. A diagnostic label does not remove primary/selection membership.
The caller must supply the complete exclusion registry for a passing full audit;
this tool can establish a failure using a smaller authenticated exclusion union.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def authenticate(binding: dict) -> dict:
    path = Path(binding['path'])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding['sha256']:
        raise ValueError(f'Binding hash mismatch: {path.name}')
    return json.loads(payload)


def identities(document: dict) -> set[str]:
    values = document.get('event_uids')
    if (not isinstance(values, list) or not values
            or any(not isinstance(uid, str) or not uid for uid in values)
            or len(set(values)) != len(values)):
        raise ValueError('event_uids must be nonempty unique strings')
    return set(values)


def check_isolation(candidate: dict, exclusions: list[dict]) -> dict:
    """Inputs are authenticated separately; return counts without exposing UIDs."""
    if candidate.get('role') != 'validation':
        raise ValueError('Development cohort must have validation role')
    sealed = candidate.get('sealed_test_role_access')
    if sealed is not False and sealed != 'forbidden':
        raise ValueError('Sealed test access must be explicitly false')
    dev = identities(candidate)
    if not exclusions:
        raise ValueError('An exclusion union is required')
    union: set[str] = set()
    overlaps = []
    for exclusion in exclusions:
        reserved = identities(exclusion)
        union.update(reserved)
        overlaps.append({'reserved_count': len(reserved), 'overlap_count': len(dev & reserved)})
    overlap = dev & union
    return {
        'status': 'FAIL' if overlap else 'PASS_NECESSARY_CHECK_ONLY',
        'development_count': len(dev), 'excluded_development_count': len(overlap),
        'eligible_development_count': len(dev - union),
        'exclusion_union_count': len(union), 'overlaps': overlaps,
        'runtime_admitted': False, 'submission_authorized_by_check': False,
        'scope': 'supplied_authenticated_exclusions_only_not_full_cohort_or_runtime_admission',
    }


def run(request: dict) -> dict:
    candidate = authenticate(request['candidate'])
    exclusions = [authenticate(binding) for binding in request['exclusions']]
    result = check_isolation(candidate, exclusions)
    result['bindings'] = request
    result['observed_utc'] = datetime.now(timezone.utc).isoformat()
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'bindings'}, indent=2))
    raise SystemExit(2 if result['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
