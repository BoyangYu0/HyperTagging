"""Subprocess adapters for historical preprocessing scripts.

Phase 4 intentionally does not reimplement preprocessing algorithms. These
adapters record the intended new input roots and call the legacy scripts as
subprocesses when requested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
from typing import Any, Literal


DEFAULT_TOY_MC_INPUT_ROOT = Path("/home/boyang/data/MC")
DEFAULT_GRAFEI_INPUT_ROOT = Path("/home/boyang/data/graFEI")


PreprocessingKind = Literal["toy_mc_dataprod", "toy_mc_preprocess", "grafei", "gpt_like"]


@dataclass(frozen=True)
class PreprocessingPlan:
    """A dry-run description of a legacy preprocessing invocation."""

    kind: PreprocessingKind
    legacy_script: Path
    argv: tuple[str, ...]
    input_root: Path
    output_root: Path | None = None
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def command(self, python_executable: str = "python") -> tuple[str, ...]:
        """Return the subprocess command for this legacy invocation."""

        return (python_executable, str(self.legacy_script), *self.argv)


@dataclass(frozen=True)
class DryRunResult:
    """Returned by adapters when ``dry_run=True``."""

    plan: PreprocessingPlan
    command: tuple[str, ...]


def repository_root(start: Path | None = None) -> Path:
    """Return the historical-repository workspace root used by this checkout."""

    if start is None:
        start = Path(__file__).resolve()
    return start.parents[4]


def run_legacy_preprocessing(
    plan: PreprocessingPlan,
    *,
    dry_run: bool = True,
    python_executable: str = "python",
    check: bool = True,
    extra_env: Mapping[str, str] | None = None,
) -> DryRunResult | subprocess.CompletedProcess[str]:
    """Run or dry-run a legacy preprocessing script.

    The adapters expose planned roots through environment variables for future
    wrapper-aware scripts. Current historical scripts mostly use hard-coded
    paths, so these variables are metadata unless the script is updated later.
    """

    command = plan.command(python_executable)
    if dry_run:
        return DryRunResult(plan=plan, command=command)

    env = {
        **os.environ,
        "HYPERTAGGING_INPUT_ROOT": str(plan.input_root),
        **dict(plan.environment),
        **dict(extra_env or {}),
    }
    if plan.output_root is not None:
        env["HYPERTAGGING_OUTPUT_ROOT"] = str(plan.output_root)

    return subprocess.run(
        command,
        cwd=plan.cwd,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _script(*parts: str) -> Path:
    return repository_root().joinpath(*parts)


def build_toy_mc_dataprod_plan(
    channel_index: int,
    job_id: int | str,
    *,
    input_root: Path | str = DEFAULT_TOY_MC_INPUT_ROOT,
    output_root: Path | str | None = None,
) -> PreprocessingPlan:
    """Plan the BASF2-dependent Toy-MC DataProd legacy script."""

    return PreprocessingPlan(
        kind="toy_mc_dataprod",
        legacy_script=_script("HyperTagging", "DataProd.py"),
        argv=(str(channel_index), str(job_id)),
        input_root=Path(input_root),
        output_root=Path(output_root) if output_root is not None else None,
        cwd=_script("HyperTagging"),
        notes=(
            "Requires BASF2/ROOT; not suitable for generic CPU-only test execution.",
            "Historical script expects argv: channel index, job id.",
            "Default planned input root is after BASF2 generation and before preprocessing.",
        ),
    )


def build_toy_mc_preprocess_plan(
    job_id: int | str,
    *,
    awkward_output: bool = True,
    input_root: Path | str = DEFAULT_TOY_MC_INPUT_ROOT,
    output_root: Path | str | None = None,
) -> PreprocessingPlan:
    """Plan the Toy-MC preprocessing script for HDF5 or awkward parquet output."""

    script = ("HyperTagging", "ak", "preprocess_ak.py") if awkward_output else ("HyperTagging", "preprocess.py")
    return PreprocessingPlan(
        kind="toy_mc_preprocess",
        legacy_script=_script(*script),
        argv=(str(job_id),),
        input_root=Path(input_root),
        output_root=Path(output_root) if output_root is not None else None,
        cwd=_script(*script[:-1]),
        notes=(
            "Historical script expects argv: job id.",
            "Non-Colab Toy-MC planned input root is /home/boyang/data/MC.",
            "Adapter does not alter tokenization, repeat copying, event numbering, or E_Rec logic.",
        ),
    )


def build_grafei_plan(
    chunk_index: int,
    *,
    input_root: Path | str = DEFAULT_GRAFEI_INPUT_ROOT,
    output_root: Path | str | None = None,
    reduced: bool = True,
) -> PreprocessingPlan:
    """Plan the GraFEI pair-data preprocessing legacy script."""

    repo = "graFEI_reduced" if reduced else "graFEI"
    return PreprocessingPlan(
        kind="grafei",
        legacy_script=_script(repo, "produce_train_data_grafei.py"),
        argv=(str(chunk_index),),
        input_root=Path(input_root),
        output_root=Path(output_root) if output_root is not None else None,
        cwd=_script(repo),
        notes=(
            "Historical script expects argv: chunk index.",
            "Default planned input root is the original GraFEI dataset before preprocessing.",
            "Adapter does not reimplement LCAG reconstruction, energy sorting, mass token rounding, or pattern construction.",
        ),
    )


def build_gpt_like_plan(
    chunk_index: int,
    *,
    input_root: Path | str = DEFAULT_GRAFEI_INPUT_ROOT,
    output_root: Path | str | None = None,
) -> PreprocessingPlan:
    """Plan the GPT-like GraFEI preprocessing legacy script."""

    return PreprocessingPlan(
        kind="gpt_like",
        legacy_script=_script("graFEI_gpt", "produce_train_data_grafei.py"),
        argv=(str(chunk_index),),
        input_root=Path(input_root),
        output_root=Path(output_root) if output_root is not None else None,
        cwd=_script("graFEI_gpt"),
        notes=(
            "Historical script expects argv: chunk index.",
            "Default planned input root is the original GraFEI dataset before preprocessing.",
            "Adapter does not reimplement GPT-like masking or autoregressive tree flattening.",
        ),
    )


def assert_legacy_scripts_exist(plans: Sequence[PreprocessingPlan]) -> None:
    """Validate that all referenced historical scripts are present."""

    missing = [str(plan.legacy_script) for plan in plans if not plan.legacy_script.exists()]
    if missing:
        raise FileNotFoundError("Missing legacy preprocessing scripts: " + ", ".join(missing))
