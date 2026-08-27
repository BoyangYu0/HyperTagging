"""Production API for the auth-false v13 verifier v3."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import _v13_router as _router


def verify_chain(
    repo: str,
    terminal_ref: Mapping[str, Any],
    pair_ref: Mapping[str, Any],
    locator_ref: Mapping[str, Any],
    pilot_ref: Mapping[str, Any],
    ladder_ref: Mapping[str, Any],
    pointer_ref: Mapping[str, Any],
    hpo_ref: Mapping[str, Any],
    final_ref: Mapping[str, Any],
    *,
    trusted_roots: Iterable[str],
) -> object:
    """Verify exactly eight file-backed nodes and return a private capability."""
    return _router.verify_nodes(
        repo,
        (terminal_ref, pair_ref, locator_ref, pilot_ref, ladder_ref, pointer_ref, hpo_ref, final_ref),
        tuple(trusted_roots),
    )


def decision(capability: object) -> str:
    """Consume a live same-process verifier capability exactly once."""
    return _router.decide(capability)


__all__ = ["verify_chain", "decision"]
