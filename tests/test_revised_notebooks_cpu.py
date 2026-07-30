from pathlib import Path
import subprocess
import sys

import nbformat


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
    assert "Executed 12 notebooks on CPU fixtures" in completed.stdout
    for name in (
        "inspect_leaf_input_pid_contract.ipynb",
        "inspect_leaf_pid_and_composite_inputs.ipynb",
        "inspect_streaming_dataset.ipynb",
        "inspect_preprocessed_dataset.ipynb",
        "inspect_hyperbolic_pretraining.ipynb",
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
