"""Export direct-mDST trees as level-autoregressive datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hypertagging.preprocessing.export_dataset import export_trees
from hypertagging.preprocessing.mdst_tree_builder import EventTree
from hypertagging.preprocessing.schema import SCHEMA_VERSION


def export_level_dataset(trees: Iterable[EventTree], output: str | Path, *, summary: dict[str, object] | None = None) -> Path:
    """Compatibility wrapper around the canonical tree exporter."""

    merged_summary = {"level_schema_version": SCHEMA_VERSION, **(summary or {})}
    return export_trees(trees, output, summary=merged_summary, legacy_levels=True)
