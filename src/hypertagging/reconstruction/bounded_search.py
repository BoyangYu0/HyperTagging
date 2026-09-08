"""Backend-independent, deterministic bounded search primitives.

No tensor library is imported here, so deployment runtimes can reuse these
primitives without importing the training stack.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T")


def stable_bounded_unique(
    items: Iterable[T],
    *,
    key: Callable[[T], Hashable],
    order: Callable[[T], object],
    limit: int,
) -> tuple[list[T], int, int]:
    """Keep the best representative per identity, then truncate stably.

    Return retained items, duplicate count, and count removed by the bound.
    Callers bound the iterable before invoking this helper.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("search limit must be a positive integer")
    unique: dict[Hashable, T] = {}
    duplicates = 0
    for item in items:
        identity = key(item)
        previous = unique.get(identity)
        if previous is not None:
            duplicates += 1
        if previous is None or order(item) < order(previous):
            unique[identity] = item
    ranked = sorted(unique.values(), key=order)
    return ranked[:limit], duplicates, max(0, len(ranked) - limit)


def bounded_conflict_free_sets(
    groups: Iterable[Iterable[T]],
    *,
    sources: Callable[[T], frozenset],
    identity: Callable[[T], object],
    chosen_score: Callable[[T], float],
    omitted_scores: Iterable[float],
    width: int,
    singleton: Callable[[T], bool] = lambda _item: False,
    max_items: int | None = None,
) -> tuple[list[tuple[float, tuple[T, ...]]], dict[str, int]]:
    """Incremental query-by-query beam, including the no-object alternative.

    Each group contains alternatives for exactly one query. At most ``width``
    partial sets survive each query; no powerset is ever materialized.

    ``singleton`` marks terminal alternatives which must be the sole selected
    item (for example a reconstructed event root).  Such alternatives are
    scored with every other query omitted and retained in a separate bounded
    lane.  This prevents an early partial-beam prune from making a root in a
    later query unreachable.  The final merge gives terminal singletons the
    same priority they receive as completed hypotheses in the outer beam,
    while still returning at most ``width`` sets in total.
    """
    if max_items is not None and (
        isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1
    ):
        raise ValueError("max_items must be a positive integer when supplied")
    grouped = tuple(
        (tuple(group), float(omitted))
        for group, omitted in zip(groups, omitted_scores, strict=True)
    )
    omitted_total = sum(omitted for _group, omitted in grouped)
    partial: list[tuple[float, tuple[T, ...], frozenset]] = [(0.0, (), frozenset())]
    terminal: list[tuple[float, tuple[T, ...], frozenset]] = []
    diagnostics = {
        "proposal_sets_expanded": 0,
        "proposal_sets_pruned": 0,
        "proposal_sets_deduplicated": 0,
        "source_conflicts_rejected": 0,
    }

    def identity_key(
        item: tuple[float, tuple[T, ...], frozenset],
    ) -> tuple[object, ...]:
        return tuple(sorted(identity(option) for option in item[1]))

    def score_order(
        item: tuple[float, tuple[T, ...], frozenset],
    ) -> tuple[float, tuple[object, ...]]:
        return -item[0], identity_key(item)

    for options, omitted in grouped:
        regular_options: list[T] = []
        singleton_options: list[T] = []
        for item in options:
            (singleton_options if singleton(item) else regular_options).append(item)

        if singleton_options and (max_items is None or max_items >= 1):
            direct = [
                (
                    omitted_total - omitted + chosen_score(item),
                    (item,),
                    sources(item),
                )
                for item in singleton_options
            ]
            diagnostics["proposal_sets_expanded"] += len(direct)
            terminal, duplicate, pruned = stable_bounded_unique(
                (*terminal, *direct),
                key=identity_key,
                order=score_order,
                limit=width,
            )
            diagnostics["proposal_sets_deduplicated"] += duplicate
            diagnostics["proposal_sets_pruned"] += pruned

        expanded = []
        for score, accepted, used in partial:
            expanded.append((score + omitted, accepted, used))
            for item in regular_options:
                if max_items is not None and len(accepted) >= max_items:
                    continue
                item_sources = sources(item)
                if used & item_sources:
                    diagnostics["source_conflicts_rejected"] += 1
                    continue
                expanded.append(
                    (
                        score + chosen_score(item),
                        accepted + (item,),
                        used | item_sources,
                    )
                )
        diagnostics["proposal_sets_expanded"] += len(expanded)
        partial, duplicate, pruned = stable_bounded_unique(
            expanded,
            key=identity_key,
            order=score_order,
            limit=width,
        )
        diagnostics["proposal_sets_deduplicated"] += duplicate
        diagnostics["proposal_sets_pruned"] += pruned

    if terminal:
        tagged = [
            (True, score, accepted, used) for score, accepted, used in terminal
        ] + [(False, score, accepted, used) for score, accepted, used in partial]
        retained, duplicate, pruned = stable_bounded_unique(
            tagged,
            key=lambda item: identity_key((item[1], item[2], item[3])),
            order=lambda item: (
                not item[0],
                -item[1],
                identity_key((item[1], item[2], item[3])),
            ),
            limit=width,
        )
        diagnostics["proposal_sets_deduplicated"] += duplicate
        diagnostics["proposal_sets_pruned"] += pruned
        return [
            (score, accepted) for _is_terminal, score, accepted, _used in retained
        ], diagnostics
    return [(score, accepted) for score, accepted, _used in partial], diagnostics
