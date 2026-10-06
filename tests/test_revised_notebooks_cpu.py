"""Fast notebook guards; full execution runs in the scheduled notebook workflow."""
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "environment", "message"),
    (
        (
            "inspect_trained_physics_validation.ipynb",
            ("HYPERTAGGING_REAL_PARQUET", "HYPERTAGGING_TRAINED_CHECKPOINT"),
            "REAL INPUT REQUIRED",
        ),
        (
            "inspect_real_mdst_pilot.ipynb",
            ("HYPERTAGGING_REAL_PILOT",),
            "fixture substitution is forbidden",
        ),
    ),
)
def test_real_only_notebooks_fail_clearly_without_inputs(
    monkeypatch, name, environment, message
):
    for variable in environment:
        monkeypatch.delenv(variable, raising=False)
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    guard = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and message in cell.source
    )
    with pytest.raises(RuntimeError, match=message):
        exec(compile(guard, f"{name}:guard", "exec"), {})


def test_real_pilot_report_covers_categories_and_fit_policy_diagnostics():
    notebook = nbformat.read(
        ROOT / "notebooks" / "inspect_real_mdst_pilot.ipynb", as_version=4
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    for field in (
        "category_aware_summaries",
        "track_fit_pid_conditioned_energy_differences",
        "track_fit_composite_mass_shifts",
        "track_fit_unavailable_fraction",
        "pion_comparison_unavailable_fraction",
        "level1_pointer_logit_comparison",
        "incomplete_reconstructable_branch",
        "copied_or_shared_sources",
    ):
        assert field in source
    assert "'status':'NOT_RUN'" in source
    assert "inspect_trained_physics_validation.ipynb" in source
