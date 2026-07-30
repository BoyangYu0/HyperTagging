#!/usr/bin/env python
"""Create the reproducible direct-mDST four-momentum comparison notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_INPUT = Path("/data/dust/user/boyangyu/hypertagging/preprocess_mc16ri_run2_100.parquet")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "preprocessing_four_momentum_validation.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook(default_input: Path) -> nbf.NotebookNode:
    cells = [
        markdown(
            """
            # Direct-mDST four-momentum validation

            ## tl;dr

            This notebook checks deterministic daughter-summed reconstructed four-vectors
            and keeps optional reco-versus-MC comparisons explicitly diagnostic. If no
            parquet path is configured it creates a tiny labelled software fixture; no
            fixture number is a physics-performance result.
            """
        ),
        markdown(
            """
            ## Context & Methods

            This notebook validates `direct-mdst-tree-v1` or `direct-mdst-tree-v2` output produced by
            `scripts/preprocess_mdst.py`. Reconstructed/computed four-vectors and diagnostic
            MC four-vectors remain separate throughout.

            ### Key Assumptions

            - A **truth-comparable node** has diagnostic MC four-momentum and is not a
              copied node.
            - A **matched final-state particle** is a truth-comparable leaf with a
              reconstructed object ID.
            - Particle residuals use matched final-state particles only.
            - Event totals sum those same matched final-state particles within each event,
              so reconstructed and MC totals use identical membership.
            - Composite computed four-momenta are recursive daughter sums. They appear in
              the all-node distribution comparison, but not in final-state totals.
            - Invariant mass is recomputed as
              $m=\\sqrt{\\max(E^2-p_x^2-p_y^2-p_z^2, 0)}$.
            - Histogram axes show the central 99% to keep bulk structure readable; tables
              report untrimmed residual metrics.
            """
        ),
        markdown("## Data"),
        markdown("### 1. Load the verified preprocessing output"),
        code(
            f"""
            from pathlib import Path
            import json
            import os
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            REPO_ROOT = Path.cwd()
            if not (REPO_ROOT / "src").exists():
                REPO_ROOT = Path("..").resolve()
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from hypertagging.data.notebook_fixtures import write_notebook_fixture
            from hypertagging.preprocessing.schema_v2 import load_payload_v2

            configured_path = (
                os.environ.get("HYPERTAGGING_PARQUET")
                or os.environ.get("HYPERTAGGING_PREPROCESS_OUTPUT")
            )
            if configured_path:
                INPUT_PATH = Path(configured_path)
                if not INPUT_PATH.exists():
                    raise FileNotFoundError(f"Configured parquet does not exist: {{INPUT_PATH}}")
                FIXTURE_MODE = False
            else:
                INPUT_PATH = Path("/tmp/hypertagging-four-vector-fixture.parquet")
                write_notebook_fixture(INPUT_PATH)
                FIXTURE_MODE = True
            FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging-four-vector-figures"))
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            np.random.seed(int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730")))

            payload = load_payload_v2(INPUT_PATH)
            required = {{"schema_version", "events", "summary_json", "feature_spec_json"}}
            missing = sorted(required - set(payload))
            if missing:
                raise KeyError(f"Missing required parquet fields: {{missing}}")
            events = payload["events"]
            production_summary = json.loads(payload["summary_json"])
            feature_spec = json.loads(payload["feature_spec_json"])

            plt.rcParams.update({{
                "figure.figsize": (12, 8),
                "figure.dpi": 110,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.22,
                "font.size": 10,
            }})

            COLORS = {{
                "computed": "#2C6EBA",
                "mc": "#D4A72C",
                "residual": "#D97706",
                "zero": "#374151",
            }}

            print(f"Input: {{INPUT_PATH}}")
            print("Mode:", "SOFTWARE FIXTURE — NOT PHYSICS PERFORMANCE" if FIXTURE_MODE else "REAL PREPROCESSED DATA")
            print(f"Schema: {{payload['schema_version']}}")
            print(f"Source schema: {{payload.get('source_schema_version', payload['schema_version'])}}")
            print("Feature groups:", list(feature_spec))
            print(f"Events: {{len(events)}}")
            """
        ),
        markdown("### 2. Build comparable node and final-state tables"),
        code(
            """
            def invariant_mass(energy, px, py, pz):
                mass_squared = np.asarray(energy) ** 2 - np.asarray(px) ** 2 - np.asarray(py) ** 2 - np.asarray(pz) ** 2
                return np.sqrt(np.clip(mass_squared, 0.0, None))


            node_rows = []
            max_parent_closure = 0.0
            for event in events:
                nodes_by_id = {int(node["node_id"]): node for node in event["nodes"]}
                for node in event["nodes"]:
                    daughters = node["daughter_ids"]
                    if daughters:
                        for field in ("energy", "px", "py", "pz"):
                            daughter_sum = sum(float(nodes_by_id[child][field]) for child in daughters)
                            max_parent_closure = max(max_parent_closure, abs(float(node[field]) - daughter_sum))

                    if node["mc_energy"] is None or int(node["copied_from"]) != -1:
                        continue
                    node_rows.append(
                        {
                            "event_id": int(event["event_id"]),
                            "node_id": int(node["node_id"]),
                            "pdg": int(node["pdg"]),
                            "is_leaf": not daughters,
                            "has_reco": bool(node["reco_id"]),
                            "E": float(node["energy"]),
                            "px": float(node["px"]),
                            "py": float(node["py"]),
                            "pz": float(node["pz"]),
                            "mc_E": float(node["mc_energy"]),
                            "mc_px": float(node["mc_px"]),
                            "mc_py": float(node["mc_py"]),
                            "mc_pz": float(node["mc_pz"]),
                        }
                    )

            comparable_nodes = pd.DataFrame(node_rows)
            comparable_nodes["mass"] = invariant_mass(
                comparable_nodes["E"], comparable_nodes["px"], comparable_nodes["py"], comparable_nodes["pz"]
            )
            comparable_nodes["mc_mass"] = invariant_mass(
                comparable_nodes["mc_E"],
                comparable_nodes["mc_px"],
                comparable_nodes["mc_py"],
                comparable_nodes["mc_pz"],
            )

            matched_particles = comparable_nodes.query("is_leaf and has_reco").copy()
            components = ["E", "px", "py", "pz", "mass"]
            for component in components:
                matched_particles[f"delta_{component}"] = (
                    matched_particles[component] - matched_particles[f"mc_{component}"]
                )

            event_totals = matched_particles.groupby("event_id", as_index=True)[
                ["E", "px", "py", "pz", "mc_E", "mc_px", "mc_py", "mc_pz"]
            ].sum()
            event_totals["mass"] = invariant_mass(
                event_totals["E"], event_totals["px"], event_totals["py"], event_totals["pz"]
            )
            event_totals["mc_mass"] = invariant_mass(
                event_totals["mc_E"], event_totals["mc_px"], event_totals["mc_py"], event_totals["mc_pz"]
            )
            for component in components:
                event_totals[f"delta_{component}"] = event_totals[component] - event_totals[f"mc_{component}"]

            assert len(comparable_nodes) > 0
            assert len(matched_particles) > 0
            assert np.isfinite(comparable_nodes.select_dtypes("number").to_numpy()).all()
            assert max_parent_closure < 1e-8

            pd.DataFrame(
                {
                    "value": [
                        len(events),
                        len(comparable_nodes),
                        len(matched_particles),
                        len(event_totals),
                        production_summary.get("unmatched_reco", 0),
                        max_parent_closure,
                    ]
                },
                index=[
                    "input events",
                    "truth-comparable retained nodes",
                    "matched final-state particles",
                    "events with matched particles",
                    "unmatched reco objects excluded from residuals",
                    "maximum parent p4 closure error",
                ],
            )
            """
        ),
        markdown("## Results"),
        markdown("### 3. Computed and MC component distributions for all comparable nodes"),
        code(
            """
            component_labels = {
                "E": "Energy [GeV]",
                "px": "$p_x$ [GeV/$c$]",
                "py": "$p_y$ [GeV/$c$]",
                "pz": "$p_z$ [GeV/$c$]",
                "mass": "Invariant mass [GeV/$c^2$]",
            }


            def central_limits(*series, lower=0.005, upper=0.995, symmetric=False):
                values = np.concatenate([np.asarray(item, dtype=float) for item in series])
                values = values[np.isfinite(values)]
                if symmetric:
                    bound = np.quantile(np.abs(values), upper)
                    return (-bound, bound) if bound > 0 else (-1.0, 1.0)
                low, high = np.quantile(values, [lower, upper])
                if not high > low:
                    return low - 0.5, high + 0.5
                padding = 0.03 * (high - low)
                return low - padding, high + padding


            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            for axis, component in zip(axes.flat, components):
                computed = comparable_nodes[component]
                mc = comparable_nodes[f"mc_{component}"]
                low, high = central_limits(computed, mc)
                bins = np.linspace(low, high, 55)
                axis.hist(
                    computed,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    color=COLORS["computed"],
                    label="Computed / reco-derived",
                )
                axis.hist(
                    mc,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    linestyle="--",
                    color=COLORS["mc"],
                    label="MC diagnostic",
                )
                axis.set_title(f"{component_labels[component].split(' [')[0]} distribution")
                axis.set_xlabel(component_labels[component])
                axis.set_ylabel("Density")
            axes.flat[0].legend(frameon=False)
            axes.flat[-1].axis("off")
            fig.suptitle("Computed and MC four-momentum components", fontsize=15, y=1.01)
            fig.text(
                0.5,
                0.01,
                f"All {len(comparable_nodes):,} truth-comparable retained nodes; central 99% shown. Source: {INPUT_PATH.name}",
                ha="center",
                color="#4B5563",
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(FIGURE_DIR / "computed_vs_mc_components.png")
            plt.show()
            """
        ),
        markdown("### 4. Particle-by-particle component differences"),
        code(
            """
            def residual_metrics(frame, prefix):
                records = []
                for component in components:
                    values = frame[f"delta_{component}"].to_numpy()
                    records.append(
                        {
                            "scope": prefix,
                            "component": component,
                            "count": len(values),
                            "mean": np.mean(values),
                            "median": np.median(values),
                            "RMSE": np.sqrt(np.mean(values**2)),
                            "95% |difference|": np.quantile(np.abs(values), 0.95),
                        }
                    )
                return pd.DataFrame(records)


            particle_metrics = residual_metrics(matched_particles, "particle")
            particle_metrics.style.format(
                {"mean": "{:.4f}", "median": "{:.4f}", "RMSE": "{:.4f}", "95% |difference|": "{:.4f}"}
            )
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            for axis, component in zip(axes.flat, components):
                residual = matched_particles[f"delta_{component}"]
                low, high = central_limits(residual, symmetric=True)
                axis.hist(
                    residual,
                    bins=np.linspace(low, high, 55),
                    color=COLORS["residual"],
                    alpha=0.78,
                    edgecolor="#92400E",
                    linewidth=0.35,
                )
                axis.axvline(0.0, color=COLORS["zero"], linestyle="--", linewidth=1.2)
                axis.set_title(f"Particle $\\Delta${component}")
                axis.set_xlabel(f"Computed − MC {component_labels[component]}")
                axis.set_ylabel("Particles")
            axes.flat[-1].axis("off")
            fig.suptitle("Particle-by-particle four-momentum differences", fontsize=15, y=1.01)
            fig.text(
                0.5,
                0.01,
                f"{len(matched_particles):,} matched, non-copied final-state particles; central 99% shown.",
                ha="center",
                color="#4B5563",
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(FIGURE_DIR / "particle_mc_diagnostic_residuals.png")
            plt.show()
            """
        ),
        markdown("### 5. Event-by-event total component differences"),
        code(
            """
            event_metrics = residual_metrics(event_totals, "event")
            event_metrics.style.format(
                {"mean": "{:.4f}", "median": "{:.4f}", "RMSE": "{:.4f}", "95% |difference|": "{:.4f}"}
            )
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            for axis, component in zip(axes.flat, components):
                residual = event_totals[f"delta_{component}"]
                low, high = central_limits(residual, symmetric=True)
                axis.hist(
                    residual,
                    bins=np.linspace(low, high, 36),
                    color=COLORS["computed"],
                    alpha=0.78,
                    edgecolor="#1E3A5F",
                    linewidth=0.4,
                )
                axis.axvline(0.0, color=COLORS["zero"], linestyle="--", linewidth=1.2)
                axis.set_title(f"Event total $\\Delta${component}")
                axis.set_xlabel(f"Computed − MC {component_labels[component]}")
                axis.set_ylabel("Events")
            axes.flat[-1].axis("off")
            fig.suptitle("Event-by-event matched final-state total differences", fontsize=15, y=1.01)
            fig.text(
                0.5,
                0.01,
                f"{len(event_totals):,} events; identical matched-particle membership on computed and MC sides; central 99% shown.",
                ha="center",
                color="#4B5563",
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(FIGURE_DIR / "event_mc_diagnostic_residuals.png")
            plt.show()
            """
        ),
        markdown("## Takeaways"),
        code(
            """
            summary_table = pd.concat([particle_metrics, event_metrics], ignore_index=True)
            particle_energy = summary_table.query("scope == 'particle' and component == 'E'").iloc[0]
            event_energy = summary_table.query("scope == 'event' and component == 'E'").iloc[0]
            event_mass = summary_table.query("scope == 'event' and component == 'mass'").iloc[0]

            print(
                f"• Structural QA: {len(events)} event trees passed; maximum parent p4 closure error "
                f"was {max_parent_closure:.3g}."
            )
            print(
                f"• Particle level: {len(matched_particles):,} matched leaves have mean ΔE "
                f"{particle_energy['mean']:.3f} GeV and energy RMSE {particle_energy['RMSE']:.3f} GeV."
            )
            print(
                f"• Event level: mean ΔE is {event_energy['mean']:.3f} GeV and mean Δmass is "
                f"{event_mass['mean']:.3f} GeV/c² for the matched final-state subset."
            )
            print(
                "• Interpretation boundary: unmatched reconstructed objects remain in the production file "
                "but are excluded here because a truth residual would be undefined."
            )
            """
        ),
        markdown(
            """
            The negative event-level energy and mass differences are descriptive validation
            results, not by themselves evidence of a preprocessing error. They can include
            detector response, reconstruction inefficiency, acceptance, and MC-matching
            selection effects. A physics-performance conclusion would require explicit
            particle selections and efficiency/purity studies beyond this preprocessing audit.
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (HyperTagging CPU)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Default parquet input recorded in the notebook.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Notebook path to create.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(args.input.resolve())
    nbf.write(notebook, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
