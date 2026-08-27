"""The only production entry points for v13 terminal evidence."""
from __future__ import annotations

from typing import Any, Mapping

from . import _v13_core as _core


def verify_chain(repo: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen and fail-closed validate a v13 evidence chain."""
    return _core.validate_chain(repo, evidence)


def decision(chain: Mapping[str, Any]) -> str:
    """Return the immutable v13 decision; no authorization is inferred."""
    # v13 deliberately has no final promotion expectation: absent the complete
    # downstream shape/authority evidence, the only safe result is this block.
    if not isinstance(chain, Mapping):
        raise _core.VerificationError("chain is not a mapping")
    return "E_FEASIBILITY_SHAPE_AUTHORITY"


__all__ = ["verify_chain", "decision"]
