"""Deterministic validation-cohort selection without source-order prefixes."""

from __future__ import annotations

import hashlib
import heapq
from typing import Iterable, TypeVar


T = TypeVar("T")
FIXED_VALIDATION_VERSION = "manifest-role-uid-hash-v1"


def select_validation_events(
    events: Iterable[T],
    *,
    limit: int,
    scientific_mode: bool,
    selection_manifest_hash: str | None,
    seed: int,
    restored_event_uids: tuple[str, ...] = (),
) -> tuple[list[T], tuple[str, ...], dict[str, object]]:
    """Select a fixed role cohort, or retain prefix behavior for explicit CI mode."""

    if limit <= 0:
        raise ValueError("validation event limit must be positive")
    if scientific_mode and not selection_manifest_hash:
        raise ValueError(
            "scientific fixed validation requires a training-selection manifest"
        )
    if restored_event_uids:
        requested = set(restored_event_uids)
        selected = {
            str(getattr(event, "event_uid")): event
            for event in events
            if str(getattr(event, "event_uid")) in requested
        }
        missing = requested - set(selected)
        if missing:
            raise ValueError(
                "saved fixed-validation UIDs are absent from the validation role: "
                f"{sorted(missing)[:3]}"
            )
        ordered = [selected[uid] for uid in restored_event_uids]
        return ordered, restored_event_uids, {
            "version": FIXED_VALIDATION_VERSION,
            "mode": "restored_manifest_role_uids",
            "selection_manifest_hash": selection_manifest_hash or "",
            "seed": int(seed),
        }
    if not scientific_mode:
        selected = []
        for event in events:
            if len(selected) >= limit:
                break
            selected.append(event)
        uids = tuple(str(getattr(event, "event_uid")) for event in selected)
        return selected, uids, {
            "version": "ci-source-prefix-v1",
            "mode": "non_scientific_ci_prefix",
            "selection_manifest_hash": selection_manifest_hash or "",
            "seed": int(seed),
        }
    ranked: list[tuple[int, int, str, T]] = []
    for index, event in enumerate(events):
        uid = str(getattr(event, "event_uid"))
        rank = int.from_bytes(hashlib.sha256(
            f"{FIXED_VALIDATION_VERSION}:{seed}:{uid}".encode("utf-8")
        ).digest(), byteorder="big")
        entry = (-rank, index, uid, event)
        if len(ranked) < limit:
            heapq.heappush(ranked, entry)
        elif rank < -ranked[0][0]:
            heapq.heapreplace(ranked, entry)
    chosen = sorted(ranked, key=lambda item: (-item[0], item[2]))
    return [item[3] for item in chosen], tuple(item[2] for item in chosen), {
        "version": FIXED_VALIDATION_VERSION,
        "mode": "manifest_validation_role_uid_hash",
        "selection_manifest_hash": selection_manifest_hash,
        "seed": int(seed),
    }


__all__ = ["FIXED_VALIDATION_VERSION", "select_validation_events"]
