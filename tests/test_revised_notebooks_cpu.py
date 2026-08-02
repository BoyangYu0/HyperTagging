from pathlib import Path
import subprocess
import sys

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_revised_notebooks_generate_and_execute_on_cpu_fixtures(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/execute_notebook_smoke_tests.py",
            "--timeout",
            "180",
            "--keep-output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        env={
            **__import__("os").environ,
            "CUDA_VISIBLE_DEVICES": "",
            "JUPYTER_CONFIG_DIR": str(tmp_path / "jupyter"),
            "IPYTHONDIR": str(tmp_path / "ipython"),
        },
    )
    assert "Executed 15 notebooks on CPU fixtures" in completed.stdout
    for name in (
        "inspect_leaf_input_pid_contract.ipynb",
        "inspect_leaf_pid_and_composite_inputs.ipynb",
        "inspect_streaming_dataset.ipynb",
        "inspect_preprocessed_dataset.ipynb",
        "inspect_hyperbolic_pretraining.ipynb",
        "inspect_exact_tree_geometry_and_loss_scales.ipynb",
        "inspect_rollout_search_and_calibration.ipynb",
        "inspect_runtime_scaling.ipynb",
        "inspect_query_capacity_and_losses.ipynb",
        "inspect_training_pipeline.ipynb",
        "inspect_level_autoregressive_reconstruction.ipynb",
        "preprocessing_qa_report.ipynb",
        "inspect_production_manifest.ipynb",
        "preprocessing_four_momentum_validation.ipynb",
        "inspect_preprocessed_parquet_and_gpt_like.ipynb",
    ):
        path = tmp_path / name
        notebook = nbformat.read(path, as_version=4)
        assert path.exists()
        assert all(
            output.get("output_type") != "error"
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        )


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
