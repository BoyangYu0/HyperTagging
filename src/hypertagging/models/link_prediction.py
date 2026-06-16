"""Link-prediction model definitions."""

from hypertagging.models.common import Linker, SimpleInteractor, linearLinker
from hypertagging.models.gpt_like import EmbLinker

StandardLinker = linearLinker
CorrectedLinker = linearLinker

__all__ = [
    "CorrectedLinker",
    "EmbLinker",
    "Linker",
    "SimpleInteractor",
    "StandardLinker",
    "linearLinker",
]
