"""Toy-MC preprocessing adapters."""

from __future__ import annotations

from pathlib import Path
import subprocess

from hypertagging.data.preprocessing import (
    DEFAULT_TOY_MC_INPUT_ROOT,
    DryRunResult,
    prepare_legacy_dataset,
)


def prepare_toy_mc_dataprod(
    channel_index: int,
    job_id: int | str,
    *,
    input_root: Path | str = DEFAULT_TOY_MC_INPUT_ROOT,
    output_root: Path | str | None = None,
    dry_run: bool = True,
) -> DryRunResult | subprocess.CompletedProcess[str]:
    """Prepare or run the legacy BASF2 Toy-MC DataProd script."""

    return prepare_legacy_dataset(
        "toy_mc_dataprod",
        channel_index,
        job_id,
        input_root=input_root,
        output_root=output_root,
        dry_run=dry_run,
    )


def prepare_toy_mc(
    job_id: int | str,
    *,
    awkward_output: bool = True,
    input_root: Path | str = DEFAULT_TOY_MC_INPUT_ROOT,
    output_root: Path | str | None = None,
    dry_run: bool = True,
) -> DryRunResult | subprocess.CompletedProcess[str]:
    """Prepare or run the legacy Toy-MC preprocessing script."""

    return prepare_legacy_dataset(
        "toy_mc_preprocess",
        job_id,
        awkward_output=awkward_output,
        input_root=input_root,
        output_root=output_root,
        dry_run=dry_run,
    )


__all__ = ["prepare_toy_mc", "prepare_toy_mc_dataprod"]
