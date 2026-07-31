"""Legacy HyperTagging PID vocabulary and pruning rules.

The historical ``HyperTagging/DataProd.py`` kept MCParticles only when the
Belle II particle name was in ``particle_list`` and ``isPrimaryParticle()`` was
true.  This module keeps that behavior in one place and exposes PDG-token
mapping for the new direct-mDST preprocessing path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol


LEGACY_PARTICLE_NAMES: tuple[str, ...] = (
    "Upsilon(4S)",
    "gamma",
    "K_L0",
    "pi0",
    "J/psi",
    "K_S0",
    "e+",
    "K+",
    "pi+",
    "mu+",
    "p+",
    "Lambda0",
    "Sigma+",
    "D+",
    "D0",
    "D_s+",
    "Lambda_c+",
    "D*+",
    "D*0",
    "D_s*+",
    "B0",
    "B+",
    "B_s0",
    "e-",
    "K-",
    "pi-",
    "mu-",
    "anti-p-",
    "anti-Lambda0",
    "anti-Sigma-",
    "D-",
    "anti-D0",
    "D_s-",
    "anti-Lambda_c-",
    "D*-",
    "anti-D*0",
    "D_s*-",
    "anti-B0",
    "B-",
    "anti-B_s0",
)

PDG_TOKENS: tuple[int, ...] = (
    0,
    300553,
    22,
    130,
    111,
    443,
    310,
    -11,
    321,
    211,
    -13,
    2212,
    3122,
    3222,
    411,
    421,
    431,
    4122,
    413,
    423,
    433,
    511,
    521,
    531,
    11,
    -321,
    -211,
    13,
    -2212,
    -3122,
    -3222,
    -411,
    -421,
    -431,
    -4122,
    -413,
    -423,
    -433,
    -511,
    -521,
    -531,
)
PID_VOCABULARY_VERSION = "legacy-reduced-pdg-v1"

# Static reconstruction ontology.  These are reduced-PID species which can be
# retained composite mothers after the documented pruning/contraction step.
# Stable detector leaves (gamma, K_L, e, mu, pi, K, p) and unknown token 0 are
# deliberately absent.  Empirical frequency priors may further constrain this
# set, but can never make a leaf-only or unknown token reconstructable.
MOTHER_ONTOLOGY_VERSION = "reduced-mother-ontology-v1"
STATIC_MOTHER_PDGS: tuple[int, ...] = (
    300553,
    111,
    443,
    310,
    3122,
    3222,
    411,
    421,
    431,
    4122,
    413,
    423,
    433,
    511,
    521,
    531,
    -3122,
    -3222,
    -411,
    -421,
    -431,
    -4122,
    -413,
    -423,
    -433,
    -511,
    -521,
    -531,
)

TOKENIZE_DICT: dict[int, int] = {pdg: index for index, pdg in enumerate(PDG_TOKENS)}
DETOKENIZE_DICT: dict[int, int] = {index: pdg for pdg, index in TOKENIZE_DICT.items()}
STATIC_MOTHER_TOKENS: tuple[int, ...] = tuple(
    TOKENIZE_DICT[pdg] for pdg in STATIC_MOTHER_PDGS
)
ALLOWED_PDGS: frozenset[int] = frozenset(PDG_TOKENS[1:])
ALLOWED_NAMES: frozenset[str] = frozenset(LEGACY_PARTICLE_NAMES)


class ParticleLike(Protocol):
    """Minimal protocol shared by basf2 objects and test doubles."""

    def getName(self) -> str: ...

    def isPrimaryParticle(self) -> bool: ...


@dataclass
class FilterDecision:
    """Decision returned by :class:`PidFilter`."""

    keep: bool
    reason: str = ""
    mapped_pdg: int = 0
    token: int = 0


@dataclass
class FilterSummary:
    """Counters accumulated during pruning."""

    kept: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    pdg_before: Counter[int] = field(default_factory=Counter)
    pdg_after: Counter[int] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, dict[str, int]]:
        """Return a JSON-serializable summary."""

        return {
            "kept": dict(self.kept),
            "dropped": dict(self.dropped),
            "pdg_before": {str(k): v for k, v in self.pdg_before.items()},
            "pdg_after": {str(k): v for k, v in self.pdg_after.items()},
        }


class PidFilter:
    """Apply the legacy HyperTagging pruning and reduced PID vocabulary."""

    def __init__(self, *, require_primary: bool = True) -> None:
        self.require_primary = require_primary
        self.summary = FilterSummary()

    def decide(
        self,
        *,
        pdg: int,
        name: str | None = None,
        is_primary: bool | None = None,
    ) -> FilterDecision:
        """Return whether a particle should be retained.

        ``name`` and ``is_primary`` are available for basf2 MCParticle objects.
        For reconstructed-only inputs, PDG membership is sufficient unless the
        caller passes ``is_primary=False``.
        """

        self.summary.pdg_before[int(pdg)] += 1
        if name is not None and name not in ALLOWED_NAMES:
            self.summary.dropped["name_not_allowed"] += 1
            return FilterDecision(False, "name_not_allowed")
        if self.require_primary and is_primary is False:
            self.summary.dropped["not_primary"] += 1
            return FilterDecision(False, "not_primary")
        if int(pdg) not in TOKENIZE_DICT:
            self.summary.dropped["pdg_not_in_vocabulary"] += 1
            return FilterDecision(False, "pdg_not_in_vocabulary")

        token = TOKENIZE_DICT[int(pdg)]
        self.summary.kept["kept"] += 1
        self.summary.pdg_after[int(pdg)] += 1
        return FilterDecision(True, mapped_pdg=int(pdg), token=token)

    def decide_basf2(self, particle: ParticleLike) -> FilterDecision:
        """Apply the historical basf2 ``toSkip`` equivalent."""

        return self.decide(
            pdg=int(particle.getPDG()),  # type: ignore[attr-defined]
            name=particle.getName(),
            is_primary=bool(particle.isPrimaryParticle()),
        )


def tokenize_pdg(pdg: int, *, unknown: int = 0) -> int:
    """Map a full PDG code to the legacy reduced HyperTagging token."""

    return TOKENIZE_DICT.get(int(pdg), unknown)


def detokenize_pdg(token: int, *, unknown: int = 0) -> int:
    """Map a legacy reduced HyperTagging token back to PDG."""

    return DETOKENIZE_DICT.get(int(token), unknown)


def validate_pid_token(token: int, *, name: str = "PID token") -> int:
    """Validate and return one model-internal reduced PID token.

    Invalid labels are deliberately not clamped or converted with ``abs``:
    doing so can turn a raw PDG code into a different valid class.
    """

    value = int(token)
    if value < 0 or value >= len(PDG_TOKENS):
        raise ValueError(
            f"{name}={value} is outside the reduced PID vocabulary "
            f"[0, {len(PDG_TOKENS)})"
        )
    return value


def validate_pid_tokens(tokens: object, *, name: str = "PID tokens") -> None:
    """Validate an integer tensor/array/iterable without importing PyTorch."""

    if hasattr(tokens, "numel") and hasattr(tokens, "min") and hasattr(tokens, "max"):
        if int(tokens.numel()) == 0:  # type: ignore[call-arg]
            return
        minimum = int(tokens.min().item())  # type: ignore[union-attr]
        maximum = int(tokens.max().item())  # type: ignore[union-attr]
        if minimum < 0 or maximum >= len(PDG_TOKENS):
            raise ValueError(
                f"{name} contains values [{minimum}, {maximum}] outside the reduced "
                f"PID vocabulary [0, {len(PDG_TOKENS)})"
            )
        return
    for index, token in enumerate(tokens):  # type: ignore[union-attr]
        validate_pid_token(int(token), name=f"{name}[{index}]")
