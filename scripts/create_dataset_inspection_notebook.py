#!/usr/bin/env python
"""Generate the schema-v1/v2/v3 dataset inspection notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_preprocessed_dataset.ipynb"


def _md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def _code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        _md(
            """
            # Preprocessed HyperTagging dataset inspection

            ## Goal

            Inspect a `direct-mdst-tree-v1`, v2, or corrected v3 parquet as a
            physicist-facing artifact: schema, PIDs, heterogeneous features, levels,
            retained decay trees, reconstructed four-vector closure, and two-B channel
            representations. Fixture output is explicitly labelled and is not a
            real-data physics result.
            """
        ),
        _md("## Setup"),
        _code(
            """
            from pathlib import Path
            import json, os, sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            REPO_ROOT = Path.cwd()
            if not (REPO_ROOT / "src").exists():
                REPO_ROOT = Path("..").resolve()
            sys.path.insert(0, str(REPO_ROOT / "src"))

            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v3
            from hypertagging.preprocessing.pid_filter import PDG_TOKENS, DETOKENIZE_DICT
            from hypertagging.preprocessing.schema_v3 import load_payload_v3

            SEED = int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730"))
            np.random.seed(SEED)
            requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
            FIXTURE_MODE = not bool(requested)
            INPUT_PATH = Path(requested) if requested else Path("/tmp/hypertagging_notebook_fixture_v3.parquet")
            if FIXTURE_MODE:
                write_notebook_fixture_v3(INPUT_PATH)
            if not INPUT_PATH.exists():
                raise FileNotFoundError(f"Required parquet does not exist: {INPUT_PATH}")
            FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging_figures/dataset"))
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            payload = load_payload_v3(INPUT_PATH)
            required_top = {"schema_version", "events", "summary_json", "feature_spec_json"}
            missing_top = required_top - payload.keys()
            if missing_top:
                raise KeyError(f"Missing required top-level columns: {sorted(missing_top)}")
            events = payload["events"]
            if not events:
                raise ValueError("Dataset has no events")
            required_node = {"node_id", "pdg", "token", "level", "daughter_ids", "node_kind", "common_availability"}
            for index, event in enumerate(events):
                if not event["nodes"]:
                    raise ValueError(f"Event {index} has no nodes")
                missing = required_node - event["nodes"][0].keys()
                if missing:
                    raise KeyError(f"Event {index} nodes miss required columns: {sorted(missing)}")
            MODE_LABEL = "TINY SOFTWARE FIXTURE — NOT REAL DATA" if FIXTURE_MODE else "REAL PREPROCESSED SAMPLE"
            print(MODE_LABEL)
            print("Input:", INPUT_PATH)
            print("Figures:", FIGURE_DIR)
            """
        ),
        _md("## Dataset/schema overview"),
        _code(
            """
            feature_spec = json.loads(payload["feature_spec_json"])
            summary = json.loads(payload["summary_json"])
            event_uids = [event["event_uid"] for event in events]
            schema_overview = pd.Series({
                "mode": MODE_LABEL,
                "schema_version": payload["schema_version"],
                "source_schema_version": payload.get("source_schema_version", payload["schema_version"]),
                "file_or_shard_count": 1,
                "events": len(events),
                "unique_event_uid": len(set(event_uids)),
                "duplicate_event_uid": len(event_uids) - len(set(event_uids)),
                "nodes": sum(len(event["nodes"]) for event in events),
                "edges": sum(len(node["daughter_ids"]) for event in events for node in event["nodes"]),
                "legacy_level_rows": len(payload.get("legacy_levels", [])),
            }, name="value")
            display(schema_overview.to_frame())
            display(pd.DataFrame({
                "feature_group": ["common", "track", "ecl_cluster", "composite"],
                "fields": [", ".join(feature_spec[name]) for name in ("common", "track", "ecl_cluster", "composite")],
            }))
            display(pd.DataFrame([{
                "event_uid": event["event_uid"],
                "source_file": event.get("source_file", ""),
                "source_category": event.get("source_category", ""),
                "experiment": event.get("experiment", -1),
                "run": event.get("run", -1),
                "production": event.get("production", -1),
            } for event in events]).head(20))
            assert schema_overview["duplicate_event_uid"] == 0, "Duplicate event_uid values detected"
            """
        ),
        _md("## PID inspection"),
        _code(
            """
            node_rows = []
            for event in events:
                for node in event["nodes"]:
                    node_rows.append({
                        "event_uid": event["event_uid"], "node_id": node["node_id"],
                        "pdg": node["pdg"], "token": node["token"], "node_kind": node["node_kind"],
                        "level": node["level"], "charge": node["charge"], "mass": node["mass"],
                        "energy": node["energy"], "px": node["px"], "py": node["py"], "pz": node["pz"],
                        "n_daughters": len(node["daughter_ids"]), "copied": node["copied"],
                        "unmatched": "unmatched_reco" in node["flags"],
                    })
            nodes = pd.DataFrame(node_rows)
            print("Complete reduced PID vocabulary (token -> PDG):")
            display(pd.DataFrame({"token": range(len(PDG_TOKENS)), "pdg": PDG_TOKENS}))
            pid_counts = nodes["pdg"].value_counts().rename_axis("pdg").reset_index(name="count")
            token_counts = nodes["token"].value_counts().sort_index()
            display(pid_counts)
            display(nodes.groupby(["node_kind", "pdg"]).size().rename("count").reset_index())
            display(nodes.groupby(["level", "pdg"]).size().rename("count").reset_index())
            unknown_rate = float((nodes["token"] == 0).mean())
            print(f"Unknown/fallback PID rate: {unknown_rate:.3%}")
            print("Top rare PIDs:", pid_counts.sort_values(["count", "pdg"]).head(15).to_dict("records"))
            conjugate = pd.DataFrame({
                "abs_pdg": nodes["pdg"].abs(),
                "sign": np.where(nodes["pdg"] < 0, "negative", "nonnegative"),
            }).groupby(["abs_pdg", "sign"]).size().unstack(fill_value=0)
            display(conjugate)
            production_pid = summary.get("pid_summary", {})
            print("Full PDG before reduction/pruning (when stored):", production_pid.get("pdg_before", "not stored"))
            print("Retained PDG counts:", production_pid.get("pdg_after", "not stored"))
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            pid_counts.head(20).plot.bar(x="pdg", y="count", ax=axes[0], legend=False, title="Full retained PDG distribution")
            token_counts.plot.bar(ax=axes[1], title="Reduced PID/token distribution")
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "pid_distributions.png"); plt.show()
            """
        ),
        _md("## Node-kind and feature inspection"),
        _code(
            """
            event_kind = nodes.groupby(["event_uid", "node_kind"]).size().unstack(fill_value=0)
            display(event_kind)
            event_kind.plot.bar(stacked=True, figsize=(10, 4), title="Tracks, clusters, composites per event")
            plt.tight_layout(); plt.savefig(FIGURE_DIR / "node_kinds_per_event.png"); plt.show()

            availability_rows = []
            feature_rows = {"common": [], "track": [], "cluster": [], "composite": []}
            for event in events:
                for node in event["nodes"]:
                    for group in ("common", "track", "cluster", "composite"):
                        values = node[f"{group}_features"]
                        masks = node[f"{group}_availability"]
                        feature_rows[group].append({name: values[name] if masks[name] else np.nan for name in values})
                        availability_rows.append({
                            "event_uid": event["event_uid"], "node_id": node["node_id"],
                            "node_kind": node["node_kind"], "group": group,
                            "available_fraction": np.mean(list(masks.values())),
                        })
            availability = pd.DataFrame(availability_rows)
            missingness = availability.pivot_table(index="node_kind", columns="group", values="available_fraction", aggfunc="mean")
            display(missingness)
            fig, ax = plt.subplots(figsize=(8, 3))
            image = ax.imshow(missingness.fillna(0), vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(missingness.columns)), missingness.columns)
            ax.set_yticks(range(len(missingness.index)), missingness.index)
            ax.set_title("Feature availability by node kind"); fig.colorbar(image, ax=ax)
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "feature_missingness.png"); plt.show()
            for group, rows in feature_rows.items():
                frame = pd.DataFrame(rows)
                print(group, "feature distributions")
                display(frame.describe().T)
            numeric = nodes.select_dtypes("number").to_numpy()
            print({"nan_count": int(np.isnan(numeric).sum()), "infinite_count": int(np.isinf(numeric).sum())})
            assert np.isfinite(numeric).all(), "NaN or infinite common node values detected"
            """
        ),
        _md("## Level definition inspection"),
        _code(
            """
            level_rows, violations = [], []
            cross_level_combinations = 0
            for event in events:
                by_id = {int(node["node_id"]): node for node in event["nodes"]}
                for mother in event["nodes"]:
                    daughters = [by_id[int(child)] for child in mother["daughter_ids"]]
                    if daughters:
                        expected = 1 + max(int(daughter["level"]) for daughter in daughters)
                        differences = [int(mother["level"]) - int(daughter["level"]) for daughter in daughters]
                        cross_level_combinations += int(len(set(difference for difference in differences if difference > 1)) > 0)
                        if int(mother["level"]) != expected:
                            violations.append({
                                "event_uid": event["event_uid"], "mother_id": mother["node_id"],
                                "stored_level": mother["level"], "expected_level": expected,
                                "daughter_ids": mother["daughter_ids"],
                            })
                        for daughter in daughters:
                            level_rows.append({
                                "event_uid": event["event_uid"], "mother_level": mother["level"],
                                "child_level": daughter["level"],
                                "difference": mother["level"] - daughter["level"],
                                "multiplicity": len(daughters),
                            })
            level_edges = pd.DataFrame(level_rows)
            display(nodes.groupby(["level", "node_kind"]).size().rename("nodes").reset_index())
            display(nodes.groupby(["level", "pdg"]).size().rename("nodes").reset_index())
            display(nodes.groupby("event_uid")["level"].max().rename("maximum_depth"))
            display(nodes[nodes.n_daughters > 0]["n_daughters"].describe())
            print("Cross-level daughter combinations:", cross_level_combinations)
            print("All L_mother = 1 + max L_d violations:")
            display(pd.DataFrame(violations))
            assert not violations, f"Found {len(violations)} level-definition violations"
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            nodes.groupby("level").size().plot.bar(ax=axes[0], title="Nodes per reconstruction level")
            level_edges["difference"].plot.hist(ax=axes[1], bins=np.arange(0.5, level_edges.difference.max()+1.5), title="Parent level − child level")
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "level_definitions.png"); plt.show()
            """
        ),
        _md("## Decay-tree visualization"),
        _code(
            """
            def plot_tree(event, *, branch_root=None, title=None):
                import networkx as nx
                selected = event["nodes"]
                by_id = {int(node["node_id"]): node for node in selected}
                if branch_root is not None:
                    keep, stack = set(), [int(branch_root)]
                    while stack:
                        node_id = stack.pop()
                        if node_id in keep or node_id not in by_id: continue
                        keep.add(node_id); stack.extend(by_id[node_id]["daughter_ids"])
                    selected = [node for node in selected if int(node["node_id"]) in keep]
                    by_id = {int(node["node_id"]): node for node in selected}
                graph = nx.DiGraph()
                labels, colors, shapes = {}, [], {}
                palette = {"track": "#3B82F6", "ecl_cluster": "#F59E0B", "composite": "#10B981", "unknown": "#9CA3AF", "other": "#8B5CF6"}
                for node in selected:
                    node_id = int(node["node_id"]); graph.add_node(node_id)
                    labels[node_id] = (
                        f"{node_id}: PDG {node['pdg']} / tok {node['token']}\\n"
                        f"L{node['level']} m={node['mass']:.3f}\\n"
                        f"p4=({node['px']:.2f},{node['py']:.2f},{node['pz']:.2f};{node['energy']:.2f})"
                        f"\\nreco={node['reco_id'] or '-'} src={node['source_node_id']} copy={node['copied_from']}"
                        + ("\\nCOPIED" if node["copied"] else "")
                        + ("\\nUNMATCHED" if "unmatched_reco" in node["flags"] else "")
                        + ("\\nTRUTH-TOPOLOGY-ONLY" if "truth_topology_only" in node["flags"] else "")
                    )
                    colors.append(palette.get(node["node_kind"], "#9CA3AF"))
                    shapes[node_id] = {"track": "o", "ecl_cluster": "s", "composite": "D"}.get(node["node_kind"], "^")
                    for daughter in node["daughter_ids"]:
                        if int(daughter) in by_id: graph.add_edge(node_id, int(daughter))
                levels = {int(node["node_id"]): int(node["level"]) for node in selected}
                positions = {}
                for level in sorted(set(levels.values())):
                    ids = sorted(node_id for node_id, value in levels.items() if value == level)
                    for offset, node_id in enumerate(ids):
                        positions[node_id] = (offset - (len(ids)-1)/2, level)
                try:
                    graphviz_positions = nx.nx_agraph.graphviz_layout(graph, prog="dot")
                    if graphviz_positions:
                        positions = graphviz_positions
                except Exception:
                    pass  # deterministic reconstruction-level matplotlib layout above
                fig, ax = plt.subplots(figsize=(max(9, len(selected)*1.3), 5))
                for shape in sorted(set(shapes.values())):
                    ids = [node_id for node_id in graph.nodes if shapes[node_id] == shape]
                    nx.draw_networkx_nodes(graph, positions, nodelist=ids, node_shape=shape,
                                           node_color=[palette.get(by_id[node_id]["node_kind"], "#9CA3AF") for node_id in ids],
                                           node_size=1900, ax=ax)
                nx.draw_networkx_edges(graph, positions, arrows=True, arrowsize=16, ax=ax)
                nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=ax)
                ax.set_title(title or f"Retained tree: {event['event_uid']}")
                ax.set_axis_off(); fig.tight_layout()
                return fig

            multiplicities = {event["event_uid"]: len(event["nodes"]) for event in events}
            depths = {event["event_uid"]: max(node["level"] for node in event["nodes"]) for event in events}
            representatives = {
                "lowest_multiplicity": min(events, key=lambda event: multiplicities[event["event_uid"]]),
                "high_multiplicity": max(events, key=lambda event: multiplicities[event["event_uid"]]),
                "maximum_depth": max(events, key=lambda event: depths[event["event_uid"]]),
                "copied_nodes": next((event for event in events if any(node["copied"] for node in event["nodes"])), events[0]),
                "unmatched_reco": next((event for event in events if any("unmatched_reco" in node["flags"] for node in event["nodes"])), events[0]),
                "two_b_branches": next((event for event in events if event.get("b1_root_id", -1) >= 0 and event.get("b2_root_id", -1) >= 0), events[0]),
                "random_valid": events[int(np.random.default_rng(SEED).integers(len(events)))],
            }
            for label, event in representatives.items():
                fig = plot_tree(event, title=f"{label}: {event['event_uid']}")
                fig.savefig(FIGURE_DIR / f"tree_{label}.png"); plt.show()
            two_b = representatives["two_b_branches"]
            for side in ("b1_root_id", "b2_root_id"):
                if int(two_b.get(side, -1)) >= 0:
                    fig = plot_tree(two_b, branch_root=two_b[side], title=f"{side} branch")
                    fig.savefig(FIGURE_DIR / f"tree_{side}.png"); plt.show()
            """
        ),
        _md("## Four-vector validation"),
        _code(
            """
            closure_rows = []
            for event in events:
                by_id = {int(node["node_id"]): node for node in event["nodes"]}
                for mother in event["nodes"]:
                    daughters = [by_id[int(child)] for child in mother["daughter_ids"]]
                    if not daughters: continue
                    residual = {}
                    for name in ("energy", "px", "py", "pz"):
                        residual[name] = float(mother[name]) - sum(float(daughter[name]) for daughter in daughters)
                    scale = max(abs(float(mother["energy"])), 1e-12)
                    closure_rows.append({
                        "event_uid": event["event_uid"], "mother_id": mother["node_id"],
                        "level": mother["level"], "multiplicity": len(daughters),
                        "delta_E": residual["energy"], "delta_px": residual["px"],
                        "delta_py": residual["py"], "delta_pz": residual["pz"],
                        "relative_max": max(abs(value) for value in residual.values()) / scale,
                    })
            closure = pd.DataFrame(closure_rows)
            display(closure.describe().T)
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            for axis, field in zip(axes.flat, ["delta_E", "delta_px", "delta_py", "delta_pz", "relative_max"]):
                closure[field].plot.hist(ax=axis, bins=30, title=field)
            closure.plot.scatter(x="level", y="relative_max", ax=axes.flat[-1], title="Closure residual vs level")
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "p4_closure.png"); plt.show()
            display(closure.groupby("multiplicity")["relative_max"].describe())
            assert closure[["delta_E", "delta_px", "delta_py", "delta_pz"]].abs().to_numpy().max() < 1e-8
            print("MC mother p4, when present, is diagnostic only; it is not a reconstructed target.")
            """
        ),
        _md("## Channel inspection"),
        _code(
            """
            channel_rows = [{
                "event_uid": event["event_uid"],
                "b1_signature": event.get("b1_channel_signature"),
                "b2_signature": event.get("b2_channel_signature"),
                "b1_id": event.get("b1_channel_id", 0),
                "b2_id": event.get("b2_channel_id", 0),
                "exact_equal": event.get("exact_channel_equal", False),
                "structured_similarity": event.get("structured_channel_similarity", 0.0),
                "y4s_signature": event.get("y4s_channel_signature"),
                "y4s_id": event.get("y4s_channel_id", 0),
                "charge_conjugate_normalized": event.get("charge_conjugate_normalized", False),
                "b1_counts": event.get("b1_channel_count_array"),
                "b2_counts": event.get("b2_channel_count_array"),
            } for event in events]
            channels = pd.DataFrame(channel_rows)
            display(channels)
            frequencies = pd.concat([channels.b1_id, channels.b2_id]).value_counts()
            display(frequencies.rename("frequency").to_frame())
            print("Rare channels:", frequencies[frequencies <= 2].index.tolist())
            print("Examples labelled identical/similar/dissimilar:")
            display(channels.assign(category=pd.cut(channels.structured_similarity, [-.01, .25, .99, 1.01], labels=["dissimilar", "similar", "identical"])))
            channels.structured_similarity.plot.hist(bins=np.linspace(0, 1, 11), title="Structured channel similarity")
            plt.tight_layout(); plt.savefig(FIGURE_DIR / "channel_similarity.png"); plt.show()
            print("Charge-conjugate normalization is configurable at export; this file stores:",
                  channels.charge_conjugate_normalized.unique().tolist())
            """
        ),
        _md("## Schema-v3 leaf, partial-decay, duplicate, and capacity diagnostics"),
        _code(
            """
            v3_rows = [{
                "event_uid": event["event_uid"],
                "node_id": node["node_id"],
                "raw_pdg": node["raw_pdg"],
                "input_pid_token": node["input_pid_token"],
                "pid_target_token": node["pid_target_token"],
                "node_kind": node["node_kind"],
                "leaf_mode": node["leaf_kinematics_mode"],
                "energy_source": node["energy_source"],
                "complete_truth_decay": node["complete_truth_decay"],
                "complete_reconstructable_decay": node["complete_reconstructable_decay"],
                "partial_missing_daughters": node["partial_missing_daughters"],
                "valid_target": node["valid_reconstruction_target"],
                "recursive_leaf_source_ids": node["recursive_leaf_source_ids"],
                "relation_flags": ",".join(flag for flag in node["flags"] if flag in
                    {"unique_match","duplicate_track","split_cluster","ambiguous_relation","unmatched_reco"}),
            } for event in events for node in event["nodes"]]
            v3 = pd.DataFrame(v3_rows)
            display(v3)
            assert v3.input_pid_token.between(0, len(PDG_TOKENS)-1).all()
            capacity = v3[v3.valid_target].groupby(["event_uid"]).size()
            cardinality = pd.Series(
                [len(node["daughter_ids"]) for event in events for node in event["nodes"] if node["daughter_ids"]],
                name="daughter_cardinality",
            )
            print("Maximum target mothers/event:", int(capacity.max()) if len(capacity) else 0)
            print("Maximum daughter cardinality:", int(cardinality.max()) if len(cardinality) else 0)
            display(v3.groupby(["node_kind","relation_flags"]).size().rename("count").reset_index())
            """
        ),
        _md("## Full-truth versus reconstructable channel identity"),
        _code(
            """
            dual_channels = pd.DataFrame([{
                "event_uid": event["event_uid"],
                "b1_full": event.get("b1_full_truth_channel_signature"),
                "b1_reconstructable": event.get("b1_reconstructable_channel_signature"),
                "b2_full": event.get("b2_full_truth_channel_signature"),
                "b2_reconstructable": event.get("b2_reconstructable_channel_signature"),
                "y4s_full_id": event.get("y4s_full_truth_channel_id",0),
                "y4s_reconstructable_id": event.get("y4s_reconstructable_channel_id",0),
            } for event in events])
            display(dual_channels)
            print("Detector inefficiency is not generator channel identity; both concepts are stored.")
            """
        ),
        _md(
            """
            ## Takeaways

            A successful top-to-bottom run verifies schema availability, unique event IDs,
            explicit heterogeneous missingness, the exact level recurrence, deterministic
            daughter-summed mother four-vectors, and inspectable two-B channel fields.
            Fixture results validate software integration only.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3"}
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
