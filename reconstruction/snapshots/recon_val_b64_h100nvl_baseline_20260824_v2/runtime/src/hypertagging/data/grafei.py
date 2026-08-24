"""GraFEI preprocessing adapters."""

from __future__ import annotations

from pathlib import Path
import subprocess

from hypertagging.data.preprocessing import (
    DEFAULT_GRAFEI_INPUT_ROOT,
    DryRunResult,
    prepare_legacy_dataset,
)


def prepare_grafei(
    chunk_index: int,
    *,
    input_root: Path | str = DEFAULT_GRAFEI_INPUT_ROOT,
    output_root: Path | str | None = None,
    reduced: bool = True,
    dry_run: bool = True,
) -> DryRunResult | subprocess.CompletedProcess[str]:
    """Prepare or run the legacy GraFEI pair-data preprocessing script."""

    return prepare_legacy_dataset(
        "grafei",
        chunk_index,
        input_root=input_root,
        output_root=output_root,
        reduced=reduced,
        dry_run=dry_run,
    )


__all__ = ["prepare_grafei"]
