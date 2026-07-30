#!/usr/bin/env python
"""Generate the schema-v3 leaf-input/PID-contract inspection notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_leaf_input_pid_contract.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook():
    cells = [
        md(
            """
            # Leaf input and reduced-PID contract

            This notebook audits data-compatible track/cluster inputs separately from MC
            supervision. Fixture results test software behavior only.
            """
        ),
        md("## Setup and schema-v3 validation"),
        code(
            """
            from pathlib import Path
            import json, os, sys
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            ROOT = Path.cwd()
            sys.path.insert(0, str(ROOT / "src"))
            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
            from hypertagging.preprocessing.schema_v3 import load_payload_v3, SCHEMA_VERSION_V3
            from hypertagging.preprocessing.pid_filter import PDG_TOKENS
            from hypertagging.reconstruction.kinematics import track_energy_hypotheses
            SEED = int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730"))
            torch.manual_seed(SEED); np.random.seed(SEED)
            requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
            FIXTURE_MODE = not bool(requested)
            INPUT = Path(requested) if requested else Path("/tmp/hypertagging_leaf_pid_v3.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v3(INPUT)
            payload = load_payload_v3(INPUT)
            if payload["schema_version"] != SCHEMA_VERSION_V3: raise ValueError("schema-v3 adaptation failed")
            OUT = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging_figures/leaf_pid"))
            OUT.mkdir(parents=True, exist_ok=True)
            print("TINY FIXTURE — NOT REAL DATA" if FIXTURE_MODE else "REAL PREPROCESSED SAMPLE")
            """
        ),
        md("## Raw p3, canonical input energy, and e/mu/pi/K/p hypotheses"),
        code(
            """
            tracks = [node for event in payload["events"] for node in event["nodes"] if node["node_kind"] == "track"]
            rows = []
            for node in tracks:
                p3 = torch.tensor([node["px"], node["py"], node["pz"]], dtype=torch.float64)
                hypotheses = track_energy_hypotheses(p3).tolist()
                rows.append({
                    "node_id": node["node_id"], "px": node["px"], "py": node["py"], "pz": node["pz"],
                    "canonical_energy": node["reconstructed_energy"],
                    **dict(zip(["E_e", "E_mu", "E_pi", "E_K", "E_p"], hypotheses)),
                    "reco_charge": node["reco_charge"], "truth_charge": node["truth_charge"],
                    "input_pid_token": node["input_pid_token"], "truth_pid_token": node["truth_pid_token"],
                    "leaf_mode": node["leaf_kinematics_mode"],
                })
            frame = pd.DataFrame(rows)
            display(frame)
            assert frame.input_pid_token.between(0, len(PDG_TOKENS)-1).all()
            frame.to_csv(OUT / "leaf_pid_token_range.csv", index=False)
            frame[["E_e","E_mu","E_pi","E_K","E_p"]].plot.hist(alpha=.45, bins=15, figsize=(9,5))
            plt.title("Data-independent track energy hypotheses"); plt.tight_layout()
            plt.savefig(OUT / "track_energy_hypotheses.png"); plt.show()
            """
        ),
        md("## PID likelihood availability and input/target separation"),
        code(
            """
            availability = pd.DataFrame([node["pid_likelihood_availability"] for node in tracks])
            display(availability)
            display(pd.crosstab(frame.input_pid_token, frame.truth_pid_token, margins=True))
            fallback_rate = float((frame.input_pid_token == 0).mean())
            print("Unknown input PID fallback rate:", fallback_rate)
            availability.mean().plot.bar(title="PIDLikelihood availability fraction")
            plt.tight_layout(); plt.savefig(OUT / "pid_likelihood_availability.png"); plt.show()
            """
        ),
        md("## Explicit MC-present/MC-absent leakage check"),
        code(
            """
            p3 = torch.tensor([[0.3, -0.2, 0.4]])
            before = track_energy_hypotheses(p3)
            changed_truth_pid = 8
            after = track_energy_hypotheses(p3)
            leakage_pass = bool(torch.equal(before, after))
            report = {
                "no_truth_leakage_pass": leakage_pass,
                "changed_truth_pid_token": changed_truth_pid,
                "input_pid_token": 0,
                "canonical_hypothesis": "pion",
                "mc_absent_input_identical": leakage_pass,
            }
            (OUT / "leaf_input_leakage_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(report)
            assert leakage_pass
            """
        ),
        md("## Optional trained leaf PID head"),
        code(
            """
            checkpoint = os.environ.get("HYPERTAGGING_CHECKPOINT", "")
            print("No checkpoint supplied; PID confusion matrix is diagnostic-only." if not checkpoint else checkpoint)
            confusion = pd.crosstab(frame.truth_pid_token, frame.input_pid_token)
            display(confusion)
            plt.imshow(confusion.to_numpy(), cmap="Blues"); plt.title("Input versus truth PID token")
            plt.colorbar(); plt.tight_layout(); plt.savefig(OUT / "pid_confusion.png"); plt.show()
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
