"""Toy-MC HyperTagging model aliases.

Phase 6 keeps the historical class names and constructor shapes. The classes
are imported from the mechanically migrated common model definitions where the
GraFEI/toy implementations share code.
"""

from hypertagging.models.common import HypDecoder, InteractingLayer, pretrain_HTR

__all__ = ["HypDecoder", "InteractingLayer", "pretrain_HTR"]
