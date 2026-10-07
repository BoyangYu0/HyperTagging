"""Reject a final Phase72 closeout without all arm/scope processed identities."""
from scripts.validate_category_coverage import validate_coverage


def validate_final_coverage(plan, views):
    expected={uid:cat for cat,uids in plan['by_category'].items() for uid in uids}
    if len(expected)!=12000 or set(views)!={'set_overlap_off','set_overlap_on'}:
        raise ValueError('Incomplete Phase72 arm/cohort identity')
    result={}
    for arm,scopes in views.items():
        if set(scopes)!={'full','half'}:raise ValueError('Incomplete Phase72 scopes')
        result[arm]={}
        for scope,rows in scopes.items():
            if {r['event_uid']:r['source_category'] for r in rows}!=expected:
                raise ValueError('Final evaluation differs from immutable primary cohort')
            result[arm][scope]=validate_coverage(rows,quota=2000)
    return result
