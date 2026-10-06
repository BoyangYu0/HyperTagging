"""Reservation checks do not open data, train models or inspect truth."""
from scripts.phase69_policy_reevaluation import CATEGORIES, choose_cohort, manifest
from scripts.build_reconstruction_phase35_evaluation_cohort import uid_sequence_sha256


def test_stratified_reservation_excludes_history_and_is_order_independent():
    rows = {f'{cat}-{i}': cat for cat in CATEGORIES for i in range(9)}
    excluded = {f'{cat}-0' for cat in CATEGORIES}
    counts, chosen = choose_cohort(rows, excluded, quota=5)
    assert counts == dict.fromkeys(CATEGORIES, 8)
    assert chosen == choose_cohort(dict(reversed(list(rows.items()))), excluded, quota=5)[1]
    selected = [uid for cat in CATEGORIES for uid in chosen[cat]]
    assert len(selected) == len(set(selected)) == 30
    assert not set(selected) & excluded
    assert all(rows[uid] == cat for cat in CATEGORIES for uid in chosen[cat])
    assert manifest(selected)['event_uids_sha256'] == uid_sequence_sha256(selected)


def test_one_category_shortage_cannot_borrow_or_repeat():
    rows = {f'{cat}-{i}': cat for cat in CATEGORIES for i in range(10)}
    excluded = {f'charged-{i}' for i in range(8)}
    counts, chosen = choose_cohort(rows, excluded, quota=3)
    assert counts['charged'] == 2
    assert counts['mixed'] == 10
    assert chosen is None
