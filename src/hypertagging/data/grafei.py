"""GraFEI preprocessing adapters."""

from __future__ import annotations

from pathlib import Path
import subprocess

from hypertagging.data.preprocessing import (
    DEFAULT_GRAFEI_INPUT_ROOT,
    DryRunResult,
    build_grafei_plan,
    run_legacy_preprocessing,
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

    plan = build_grafei_plan(
        chunk_index,
        input_root=input_root,
        output_root=output_root,
        reduced=reduced,
    )
    result = run_legacy_preprocessing(plan, dry_run=dry_run)
    return result


__all__ = ["prepare_grafei"]
