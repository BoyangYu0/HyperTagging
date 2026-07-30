#!/usr/bin/env python
"""Generate and execute revised notebooks on deterministic CPU fixtures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "leaf_composite": (
        REPO_ROOT / "scripts" / "create_leaf_pid_composite_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_leaf_pid_and_composite_inputs.ipynb",
        ("input versus truth daughter histograms", "pointer response", "gradient"),
    ),
    "streaming": (
        REPO_ROOT / "scripts" / "create_streaming_dataset_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_streaming_dataset.ipynb",
        ("event-row parquet", "bounded iteration", "online normalization"),
    ),
    "leaf_pid": (
        REPO_ROOT / "scripts" / "create_leaf_input_pid_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_leaf_input_pid_contract.ipynb",
        ("reduced-PID contract", "energy hypotheses", "leakage check"),
    ),
    "dataset": (
        REPO_ROOT / "scripts" / "create_dataset_inspection_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_preprocessed_dataset.ipynb",
        ("Dataset/schema overview", "PID inspection", "Decay-tree visualization", "Channel inspection"),
    ),
    "hyperbolic": (
        REPO_ROOT / "scripts" / "create_hyperbolic_inspection_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_hyperbolic_pretraining.ipynb",
        ("Embedding projections", "Radius/depth validation", "Anti-collapse diagnostics", "Channel embeddings"),
    ),
    "capacity": (
        REPO_ROOT / "scripts" / "create_query_capacity_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_query_capacity_and_losses.ipynb",
        ("query usage", "focal weighting", "Hungarian"),
    ),
    "training": (
        REPO_ROOT / "scripts" / "create_training_pipeline_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_training_pipeline.ipynb",
        ("train-only normalizer", "Encoder transfer", "resume"),
    ),
    "reconstruction": (
        REPO_ROOT / "scripts" / "create_reconstruction_inspection_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_level_autoregressive_reconstruction.ipynb",
        ("stair-causal mask", "decoded proposals", "teacher-forced", "forward/loss/backward"),
    ),
    "qa": (
        REPO_ROOT / "scripts" / "create_preprocessing_qa_notebook.py",
        REPO_ROOT / "notebooks" / "preprocessing_qa_report.ipynb",
        ("Preprocessing QA report", "Machine-readable", "daughter-summed"),
    ),
    "manifest": (
        REPO_ROOT / "scripts" / "create_production_manifest_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_production_manifest.ipynb",
        ("Production manifest inspection", "memory", "global UID"),
    ),
    "four_vector": (
        REPO_ROOT / "scripts" / "create_preprocessing_visualization_notebook.py",
        REPO_ROOT / "notebooks" / "preprocessing_four_momentum_validation.ipynb",
        ("four-momentum validation", "truth-comparable", "MC diagnostic"),
    ),
    "direct_gpt": (
        REPO_ROOT / "scripts" / "create_parquet_gpt_inspection_notebook.py",
        REPO_ROOT / "notebooks" / "inspect_preprocessed_parquet_and_gpt_like.ipynb",
        ("direct-mDST parquet", "attention mask", "optimizer step"),
    ),
}


def execute_one(
    name: str,
    generator: Path,
    source: Path,
    expected_sections: tuple[str, ...],
    *,
    work_root: Path,
    timeout: int,
) -> Path:
    subprocess.run([sys.executable, str(generator)], cwd=REPO_ROOT, check=True)
    notebook_text = source.read_text(encoding="utf-8")
    for section in expected_sections:
        if section.lower() not in notebook_text.lower():
            raise AssertionError(f"{source.name} misses expected section text: {section}")
    executed_path = work_root / source.name
    shutil.copy2(source, executed_path)
    notebook = nbformat.read(executed_path, as_version=4)
    figure_dir = work_root / f"{name}_figures"
    old_environment = os.environ.copy()
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HYPERTAGGING_FIGURE_DIR": str(figure_dir),
            "HYPERTAGGING_QA_JSON": str(work_root / "preprocessing_qa.json"),
            "HYPERTAGGING_NOTEBOOK_SEED": "20260730",
        }
    )
    os.environ.pop("HYPERTAGGING_PARQUET", None)
    os.environ.pop("HYPERTAGGING_PREPROCESS_OUTPUT", None)
    os.environ.pop("HYPERTAGGING_CHECKPOINT", None)
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
    nbformat.write(notebook, executed_path)
    error_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if error_outputs:
        raise AssertionError(f"{source.name} produced notebook errors: {error_outputs}")
    if not list(figure_dir.glob("*.png")):
        raise AssertionError(f"{source.name} did not produce expected figures")
    required_artifacts = {
        "leaf_composite": ("leaf_composite_contract.json",),
        "streaming": ("streaming_report.json",),
        "leaf_pid": ("leaf_pid_token_range.csv", "leaf_input_leakage_check.json"),
        "capacity": ("capacity_report.json", "sparse_loss_table.csv"),
        "training": ("training_pipeline_pass.json", "reconstruction/checkpoint.pt"),
        "qa": (),
        "manifest": ("production_manifest_report.json",),
    }.get(name, ())
    for relative in required_artifacts:
        if not (figure_dir / relative).exists():
            raise AssertionError(f"{source.name} missed required artifact {relative}")
    if name == "leaf_composite":
        import json

        report = json.loads(
            (figure_dir / "leaf_composite_contract.json").read_text(encoding="utf-8")
        )
        if not report.get("truth_clean_composite_input") or not report.get(
            "raw_track_unknown_input_pass"
        ):
            raise AssertionError(f"leaf/composite contract failed: {report}")
    if name == "streaming":
        import json

        report = json.loads(
            (figure_dir / "streaming_report.json").read_text(encoding="utf-8")
        )
        if not report.get("manifest_output_file_resolved") or not report.get(
            "source_leakage_pass"
        ):
            raise AssertionError(f"streaming report failed: {report}")
    if name == "qa":
        qa_path = work_root / "preprocessing_qa.json"
        if not qa_path.exists():
            raise AssertionError("QA notebook did not write its JSON summary")
        import json
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if not qa.get("p4_closure_pass"):
            raise AssertionError(f"QA JSON did not pass p4 closure: {qa}")
    return executed_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--keep-output", type=Path, default=None)
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(NOTEBOOKS),
        help="Execute only the named notebook group; repeat to select several.",
    )
    args = parser.parse_args(argv)
    if args.keep_output:
        work_root = args.keep_output.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="hypertagging-notebooks-")
        work_root = Path(temporary.name)
    executed = []
    selected = set(args.only or NOTEBOOKS)
    for name, (generator, source, sections) in NOTEBOOKS.items():
        if name not in selected:
            continue
        executed.append(
            execute_one(
                name,
                generator,
                source,
                sections,
                work_root=work_root,
                timeout=args.timeout,
            )
        )
    print(f"Executed {len(executed)} notebooks on CPU fixtures")
    for path in executed:
        print(path)
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
