"""Focused contracts for backend-independent bounded proposal-set search."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hypertagging.reconstruction.bounded_search import bounded_conflict_free_sets


@dataclass(frozen=True)
class _Option:
    name: str
    identity: str
    source_ids: frozenset[int]
    score: float
    terminal: bool = False


def _search(
    groups: tuple[tuple[_Option, ...], ...],
    *,
    omitted: tuple[float, ...],
    width: int,
):
    return bounded_conflict_free_sets(
        groups,
        sources=lambda option: option.source_ids,
        identity=lambda option: option.identity,
        chosen_score=lambda option: option.score,
        omitted_scores=omitted,
        width=width,
        singleton=lambda option: option.terminal,
    )


def _names(items: tuple[_Option, ...]) -> tuple[str, ...]:
    return tuple(item.name for item in items)


def test_late_terminal_singleton_survives_pruned_empty_prefix():
    first = _Option("first", "first", frozenset({1}), -0.1)
    second = _Option("second", "second", frozenset({2}), -0.1)
    root = _Option("root", "root", frozenset({1, 2}), -3.0, terminal=True)

    retained, diagnostics = _search(
        ((first,), (second,), (root,)),
        omitted=(-4.0, -4.0, -4.0),
        width=2,
    )

    # The ordinary width-two prefix beam has already discarded the all-null
    # path before it reaches the third query. The root must nevertheless be
    # scored as a singleton with the first two queries omitted and returned
    # ahead of a higher-scoring unfinished set.
    assert [_names(items) for _score, items in retained] == [
        ("root",),
        ("first", "second"),
    ]
    assert retained[0][0] == pytest.approx(-11.0)
    assert retained[1][0] == pytest.approx(-4.2)
    assert len(retained) == 2
    assert diagnostics["proposal_sets_pruned"] > 0


def test_multiple_terminal_queries_are_deduplicated_ranked_and_bounded():
    side = _Option("side", "side", frozenset({9}), 10.0)
    root_z = _Option("root-z", "root-z", frozenset({1}), -2.0, terminal=True)
    root_a_early = _Option(
        "root-a-early", "root-a", frozenset({2}), -1.0, terminal=True
    )
    root_b = _Option("root-b", "root-b", frozenset({3}), -1.0, terminal=True)
    root_a_late = _Option("root-a-late", "root-a", frozenset({4}), -0.5, terminal=True)
    root_c = _Option("root-c", "root-c", frozenset({5}), -3.0, terminal=True)
    groups = (
        (side, root_z),
        (root_a_early, root_b),
        (root_a_late, root_c),
    )

    first, diagnostics = _search(
        groups,
        omitted=(-0.5, -0.5, -0.5),
        width=3,
    )
    second, _ = _search(
        groups,
        omitted=(-0.5, -0.5, -0.5),
        width=3,
    )

    assert [_names(items) for _score, items in first] == [
        ("root-a-late",),
        ("root-b",),
        ("root-z",),
    ]
    assert [score for score, _items in first] == pytest.approx([-1.5, -2.0, -3.0])
    assert first == second
    assert all(len(items) == 1 and items[0].terminal for _score, items in first)
    assert len(first) == 3
    assert diagnostics["proposal_sets_deduplicated"] >= 1
    assert diagnostics["proposal_sets_pruned"] > 0


def test_no_singleton_path_preserves_score_order_conflicts_and_width():
    first = _Option("first", "first", frozenset({1}), 2.0)
    other = _Option("other", "other", frozenset({2}), 1.0)
    conflict = _Option("conflict", "conflict", frozenset({1}), 3.0)
    free = _Option("free", "free", frozenset({3}), 0.5)

    retained, diagnostics = _search(
        ((first, other), (conflict, free)),
        omitted=(-0.25, -0.5),
        width=3,
    )

    assert [score for score, _items in retained] == pytest.approx([4.0, 2.75, 2.5])
    assert [_names(items) for _score, items in retained] == [
        ("other", "conflict"),
        ("conflict",),
        ("first", "free"),
    ]
    assert len(retained) == 3
    assert diagnostics == {
        "proposal_sets_expanded": 11,
        "proposal_sets_pruned": 5,
        "proposal_sets_deduplicated": 0,
        "source_conflicts_rejected": 1,
    }
