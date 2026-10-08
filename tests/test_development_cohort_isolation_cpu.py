import copy
import hashlib
import json

import pytest

from scripts.check_development_cohort_isolation import check_isolation, run


def candidate():
    return {'event_uids': ['d1', 'd2'], 'role': 'validation', 'sealed_test_role_access': False}


def test_diagnostic_label_does_not_override_primary_reservation():
    document = {**candidate(), 'cohort_role': 'development_diagnostic'}
    result = check_isolation(document, [{'event_uids': ['d1', 'd2', 'p3']}])
    assert result['status'] == 'FAIL'
    assert result['excluded_development_count'] == 2
    assert result['eligible_development_count'] == 0
    assert not result['runtime_admitted']


def test_overlap_union_is_not_double_counted():
    result = check_isolation(candidate(), [{'event_uids': ['d1']}, {'event_uids': ['d1', 'p1']}])
    assert result['excluded_development_count'] == 1
    assert result['exclusion_union_count'] == 2


def test_disjoint_subset_does_not_authorize_submission():
    result = check_isolation(candidate(), [{'event_uids': ['p1']}])
    assert result['status'] == 'PASS_NECESSARY_CHECK_ONLY'
    assert not result['runtime_admitted']
    assert not result['submission_authorized_by_check']


@pytest.mark.parametrize('values', [[], ['d1', 'd1'], ['d1', 2], None, ['']])
def test_invalid_candidate_or_exclusion_is_rejected(values):
    with pytest.raises(ValueError):
        check_isolation({**candidate(), 'event_uids': values}, [{'event_uids': ['p1']}])
    with pytest.raises(ValueError):
        check_isolation(candidate(), [{'event_uids': values}])


@pytest.mark.parametrize('patch', [{'role': 'test'}, {'role': 'train'}, {'sealed_test_role_access': True}, {'sealed_test_role_access': None}])
def test_role_and_sealed_boundary(patch):
    with pytest.raises(ValueError):
        check_isolation({**candidate(), **patch}, [{'event_uids': ['p1']}])


def test_missing_exclusions_fails():
    with pytest.raises(ValueError):
        check_isolation(candidate(), [])


def test_both_candidate_and_exclusion_are_hash_bound(tmp_path):
    def binding(name, document):
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    request = {'candidate': binding('candidate.json', candidate()),
               'exclusions': [binding('primary.json', {'event_uids': ['d1']})]}
    assert run(request)['status'] == 'FAIL'
    for key in ('candidate', 'exclusions'):
        changed = copy.deepcopy(request)
        item = changed[key] if key == 'candidate' else changed[key][0]
        item['sha256'] = '0' * 64
        with pytest.raises(ValueError, match='hash mismatch'):
            run(changed)


def test_canonical_forbidden_sealed_role_is_supported():
    result = check_isolation({**candidate(), 'sealed_test_role_access': 'forbidden'},
                             [{'event_uids': ['d1']}])
    assert result['status'] == 'FAIL'
