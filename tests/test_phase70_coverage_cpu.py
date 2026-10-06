import pytest
from scripts.validate_category_coverage import validate_coverage,CATEGORIES
from scripts.phase70_policy_reevaluation import choose_cohort

def test_quota_is_per_category_and_success_independent():
    rows=[dict(event_uid=str(i),source_category=c,attempted=True,processed=True,truth_unavailable=True) for i,c in enumerate(CATEGORIES)]
    assert validate_coverage(rows,quota=1)["complete"]
    rows[-1]["processed"]=False
    with pytest.raises(ValueError,match="Incomplete"):validate_coverage(rows,quota=1)
    assert not validate_coverage(rows,quota=1,diagnostic=True)["complete"]

def test_duplicate_does_not_increase_coverage():
    row=dict(event_uid="x",source_category="charged",attempted=True,processed=True)
    with pytest.raises(ValueError,match="Duplicate"):validate_coverage([row,row],diagnostic=True)

def test_reservation_excludes_history_and_preserves_categories():
    cats={f"{cat}{i}":cat for cat in CATEGORIES for i in range(3)}
    used={f"{cat}0" for cat in CATEGORIES}
    counts,cohort=choose_cohort(cats,used,2)
    assert all(n==2 for n in counts.values()) and all(len(x)==2 for x in cohort.values())
    assert not used.intersection(u for rows in cohort.values() for u in rows)
    assert choose_cohort(cats,used|{"charged1"},2)[1] is None
