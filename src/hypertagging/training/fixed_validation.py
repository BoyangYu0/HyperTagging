"""Deterministic validation-cohort selection without source-order prefixes."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")
FIXED_VALIDATION_VERSION = "manifest-role-uid-hash-v1"
EXCLUDED_EVENT_UID_HASH_SCHEME = "sha256-u64be-length-prefixed-utf8-v1"


def excluded_event_uids_contract(
    event_uids: Iterable[str],
) -> dict[str, int | str]:
    """Return an order-independent identity for an excluded validation set."""

    normalized = tuple(sorted({str(uid) for uid in event_uids}))
    digest = hashlib.sha256()
    for uid in normalized:
        encoded = uid.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return {
        "excluded_event_uid_count": len(normalized),
        "excluded_event_uids_sha256": digest.hexdigest(),
        "excluded_event_uids_hash_scheme": EXCLUDED_EVENT_UID_HASH_SCHEME,
    }


def select_validation_events(
    events: Iterable[T],
    *,
    limit: int,
    scientific_mode: bool,
    selection_manifest_hash: str | None,
    seed: int,
    restored_event_uids: tuple[str, ...] = (),
    excluded_event_uids: tuple[str, ...] = (),
) -> tuple[list[T], tuple[str, ...], dict[str, object]]:
    """Select a fixed role cohort, or retain prefix behavior for explicit CI mode."""

    if limit <= 0:
        raise ValueError("validation event limit must be positive")
    if scientific_mode and not selection_manifest_hash:
        raise ValueError(
            "scientific fixed validation requires a training-selection manifest"
        )
    excluded = {str(uid) for uid in excluded_event_uids}
    exclusion_contract = excluded_event_uids_contract(excluded)
    if restored_event_uids:
        overlap = set(restored_event_uids) & excluded
        if overlap:
            raise ValueError(
                "saved fixed-validation UIDs intersect the excluded validation set: "
                f"{sorted(overlap)[:3]}"
            )
        requested = set(restored_event_uids)
        selected = {
            str(event.event_uid): event
            for event in events
            if str(event.event_uid) in requested
            and str(event.event_uid) not in excluded
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
            **exclusion_contract,
        }
    if not scientific_mode:
        selected = []
        for event in events:
            if str(event.event_uid) in excluded:
                continue
            if len(selected) >= limit:
                break
            selected.append(event)
        uids = tuple(str(event.event_uid) for event in selected)
        return selected, uids, {
            "version": "ci-source-prefix-v1",
            "mode": "non_scientific_ci_prefix",
            "selection_manifest_hash": selection_manifest_hash or "",
            "seed": int(seed),
            **exclusion_contract,
        }
    ranked: list[tuple[int, int, str, T]] = []
    for index, event in enumerate(events):
        uid = str(event.event_uid)
        if uid in excluded:
            continue
        rank = int.from_bytes(hashlib.sha256(
            f"{FIXED_VALIDATION_VERSION}:{seed}:{uid}".encode()
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
        **exclusion_contract,
    }


__all__ = [
    "EXCLUDED_EVENT_UID_HASH_SCHEME",
    "FIXED_VALIDATION_VERSION",
    "excluded_event_uids_contract",
    "select_validation_events",
]
