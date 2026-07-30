#!/usr/bin/env python
"""Generate the hyperbolic-pretraining inspection notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_hyperbolic_pretraining.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Hyperbolic pretraining inspection

            ## Goal

            Inspect an untrained tiny shared encoder, or an optional supplied
            checkpoint, in the Poincare ball and tangent space. Two-dimensional
            views are projections and do not exactly represent the full geometry.
            """
        ),
        md("## Setup"),
        code(
            """
            from pathlib import Path
            import os, sys
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch

            REPO_ROOT = Path.cwd()
            if not (REPO_ROOT / "src").exists(): REPO_ROOT = Path("..").resolve()
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from hypertagging.data.heterogeneous import load_heterogeneous_events, collate_heterogeneous_events
            from hypertagging.data.notebook_fixtures import write_notebook_fixture
            from hypertagging.losses.hyperbolic_pretraining import (
                build_tree_relation_targets, collapse_diagnostics, pool_b_branch_embeddings,
            )
            from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
            from hypertagging.models.hyperbolic import distance, logmap0, radius

            SEED = int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730"))
            torch.manual_seed(SEED); np.random.seed(SEED)
            requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
            FIXTURE_MODE = not bool(requested)
            INPUT_PATH = Path(requested) if requested else Path("/tmp/hypertagging_notebook_fixture_v2.parquet")
            if FIXTURE_MODE: write_notebook_fixture(INPUT_PATH)
            if not INPUT_PATH.exists(): raise FileNotFoundError(INPUT_PATH)
            FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging_figures/hyperbolic"))
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            events = load_heterogeneous_events(INPUT_PATH, limit=8, max_nodes=128)
            if not events: raise ValueError("No usable events")
            batch = collate_heterogeneous_events(events)
            model = HeterogeneousNodeEncoder(d_model=24, hyper_dim=4)
            checkpoint = os.environ.get("HYPERTAGGING_CHECKPOINT", "").strip()
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
            model.eval()
            with torch.no_grad(): encoded = model(batch)
            h, z, tangent = encoded.node_embeddings, encoded.hyperbolic_embeddings, logmap0(encoded.hyperbolic_embeddings)
            mask = batch["node_mask"]
            print("TINY SOFTWARE FIXTURE — UNTRAINED MODEL" if FIXTURE_MODE and not checkpoint else "SUPPLIED DATA/CHECKPOINT INSPECTION")
            print("Events/nodes:", len(events), int(mask.sum()))
            """
        ),
        md("## Embedding projections"),
        code(
            """
            flat = tangent[mask].numpy()
            centered = flat - flat.mean(0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            pca = centered @ vh[:2].T
            levels = batch["level_ids"][mask].numpy()
            kinds = batch["node_kind_ids"][mask].numpy()
            pids = batch["pid_labels"][mask].numpy()
            sides = batch["b_side"][mask].numpy()
            parents = batch["parent_ids"][mask].numpy()
            channel_by_node = torch.where(
                batch["b_side"] == 0,
                batch["b1_channel_ids"][:, None],
                torch.where(batch["b_side"] == 1, batch["b2_channel_ids"][:, None], torch.zeros_like(batch["b_side"])),
            )[mask].numpy()
            figures = [
                ("reconstruction level", levels),
                ("node kind", kinds),
                ("B side", sides),
                ("reduced PID", pids),
                ("immediate mother position", parents),
                ("selected channel family", channel_by_node),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            for axis, (label, color) in zip(axes.flat, figures):
                scatter = axis.scatter(pca[:, 0], pca[:, 1], c=color, cmap="tab20", s=30)
                axis.set_title(f"Tangent-space PCA by {label}"); fig.colorbar(scatter, ax=axis)
            disk = z[mask][:, :2].numpy()
            disk_axis = axes.flat[len(figures)]
            disk_axis.scatter(disk[:, 0], disk[:, 1], c=levels, cmap="viridis")
            circle = plt.Circle((0, 0), 1, fill=False, color="black"); disk_axis.add_patch(circle)
            disk_axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal",
                              title="First two coordinates (projected Poincare disk)")
            axes.flat[-1].set_axis_off()
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "embedding_projections.png"); plt.show()
            print("Channel family uses exact channel IDs at the pooled-branch level below, not per-node classification.")
            """
        ),
        md("## Radius/depth validation"),
        code(
            """
            radii = radius(z)[mask].numpy()
            truth_distance = []
            for batch_index in range(mask.shape[0]):
                for node_index in mask[batch_index].nonzero().flatten().tolist():
                    distance_to_root, current, seen = 0, node_index, set()
                    while int(batch["parent_ids"][batch_index, current]) >= 0:
                        if current in seen: raise ValueError("Cycle while computing truth distance to root")
                        seen.add(current); current = int(batch["parent_ids"][batch_index, current]); distance_to_root += 1
                    truth_distance.append(distance_to_root)
            radius_frame = pd.DataFrame({"radius": radii, "level": levels, "truth_distance_to_root": truth_distance})
            correlation = radius_frame.corr().loc["radius", "level"]
            root_distance_correlation = radius_frame.corr().loc["radius", "truth_distance_to_root"]
            medians = radius_frame.groupby("level").radius.median()
            monotone_fraction = float(np.mean(np.diff(medians.values) <= 0)) if len(medians) > 1 else 1.0
            display(radius_frame.groupby("level").radius.describe())
            print({"radius_level_correlation": correlation, "radius_truth_root_distance_correlation": root_distance_correlation,
                   "monotone_decrease_fraction": monotone_fraction})
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            radius_frame.plot.scatter(x="level", y="radius", ax=axes[0], title="Hyperbolic radius vs reconstruction level")
            radius_frame.boxplot(column="radius", by="level", ax=axes[1]); axes[1].set_title("Radius distribution by level")
            radius_frame.plot.scatter(x="truth_distance_to_root", y="radius", ax=axes[2],
                                      title="Radius vs truth distance to retained root")
            fig.suptitle(""); fig.tight_layout(); fig.savefig(FIGURE_DIR / "radius_depth.png"); plt.show()
            leaf_median = radius_frame.query("level == 0").radius.median()
            root_level = radius_frame.level.max()
            root_median = radius_frame.query("level == @root_level").radius.median()
            print("Convention check (trained expectation): leaves farther than roots:", leaf_median, root_median)
            if checkpoint: assert leaf_median > root_median, "Checkpoint reverses the documented radius convention"
            """
        ),
        md("## Relation separation"),
        code(
            """
            pair_distance = distance(z[:, :, None, :], z[:, None, :, :])
            categories = {"same immediate mother": [], "same branch": [], "different B sides": [],
                          "unrelated": [], "true parent-child": [], "hard negative": []}
            for b in range(mask.shape[0]):
                valid = mask[b].nonzero().flatten().tolist()
                for i in valid:
                    for j in valid:
                        if i >= j: continue
                        value = float(pair_distance[b, i, j])
                        if batch["parent_ids"][b, i] >= 0 and batch["parent_ids"][b, i] == batch["parent_ids"][b, j]:
                            categories["same immediate mother"].append(value)
                        elif batch["b_side"][b, i] >= 0 and batch["b_side"][b, i] == batch["b_side"][b, j]:
                            categories["same branch"].append(value)
                        elif batch["b_side"][b, i] >= 0 and batch["b_side"][b, j] >= 0:
                            categories["different B sides"].append(value)
                        else: categories["unrelated"].append(value)
                        if batch["parent_ids"][b, i] == j or batch["parent_ids"][b, j] == i:
                            categories["true parent-child"].append(value)
                for child in valid:
                    parent = int(batch["parent_ids"][b, child])
                    negatives = [index for index in valid if index not in (child, parent)]
                    if parent >= 0 and negatives:
                        categories["hard negative"].append(min(float(pair_distance[b, child, index]) for index in negatives))
            fig, ax = plt.subplots(figsize=(11, 5))
            plotted = [values for values in categories.values() if values]
            labels = [name for name, values in categories.items() if values]
            ax.hist(plotted, bins=20, histtype="step", label=labels, density=True)
            ax.set_title("Hyperbolic relation-distance separation"); ax.legend()
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "relation_distances.png"); plt.show()
            display(pd.DataFrame({name: pd.Series(values) for name, values in categories.items()}).describe().T)
            """
        ),
        md("## Anti-collapse diagnostics"),
        code(
            """
            diagnostics = collapse_diagnostics(z, mask, level_ids=batch["level_ids"], b_side=batch["b_side"])
            diagnostics = {name: float(value) for name, value in diagnostics.items()}
            valid = tangent[mask]
            standard_deviation = valid.std(dim=0, unbiased=False).numpy()
            centered = valid - valid.mean(0)
            covariance = (centered.T @ centered / max(len(centered)-1, 1)).numpy()
            singular_values = torch.linalg.svdvals(centered).numpy()
            pairwise = torch.cdist(valid, valid).numpy()
            nearly_constant_fraction = float((standard_deviation < 1e-3).mean())
            status = (
                "FAIL" if diagnostics["effective_rank"] < 1.5 or nearly_constant_fraction > 0.5
                else "WARN" if diagnostics["min_dimension_std"] < 1e-3 or diagnostics["boundary_fraction"] > 0.1
                else "PASS"
            )
            print({"status": status, **diagnostics, "nearly_constant_fraction": nearly_constant_fraction})
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            axes[0,0].bar(np.arange(len(standard_deviation)), standard_deviation); axes[0,0].set_title("Per-dimension tangent std")
            image = axes[0,1].imshow(covariance, cmap="coolwarm"); axes[0,1].set_title("Tangent covariance"); fig.colorbar(image, ax=axes[0,1])
            axes[1,0].plot(singular_values, marker="o"); axes[1,0].set_title("Singular-value spectrum")
            axes[1,1].hist(pairwise[np.triu_indices_from(pairwise, k=1)], bins=25); axes[1,1].set_title("Pairwise tangent distance")
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "anti_collapse.png"); plt.show()
            print("Criteria: FAIL for effective rank <1.5 or >50% constant dimensions; WARN for min std <1e-3 or >10% near boundary.")
            """
        ),
        md("## Channel embeddings"),
        code(
            """
            branch_embeddings, branch_mask = pool_b_branch_embeddings(encoded.channel_projection, batch["b_side"], mask)
            cosine = torch.nn.functional.cosine_similarity(branch_embeddings[:,0], branch_embeddings[:,1]).detach().numpy()
            channel_frame = pd.DataFrame({
                "event": batch["event_ids"].numpy(),
                "embedding_similarity": cosine,
                "structured_channel_similarity": batch["channel_similarity"].numpy(),
                "b1_channel": batch["b1_channel_ids"].numpy(),
                "b2_channel": batch["b2_channel_ids"].numpy(),
            })
            channel_frame["exact_same_channel"] = channel_frame.b1_channel == channel_frame.b2_channel
            display(channel_frame)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(channel_frame.structured_channel_similarity, channel_frame.embedding_similarity,
                       c=channel_frame.exact_same_channel.astype(int), cmap="coolwarm")
            ax.set(xlabel="Dictionary/tree channel similarity", ylabel="B-embedding cosine similarity",
                   title="B-branch representation vs channel similarity")
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "channel_embeddings.png"); plt.show()
            event_embeddings = branch_embeddings.mean(dim=1)
            similarity = torch.nn.functional.normalize(event_embeddings, dim=-1) @ torch.nn.functional.normalize(event_embeddings, dim=-1).T
            print("Unordered Upsilon event-embedding similarity:")
            display(pd.DataFrame(similarity.detach().numpy(), index=batch["event_ids"].tolist(), columns=batch["event_ids"].tolist()))
            print("Nearest-neighbor retrieval and rare-channel examples:")
            for index, event_id in enumerate(batch["event_ids"].tolist()):
                order = torch.argsort(similarity[index], descending=True).tolist()
                print(event_id, [batch["event_ids"][neighbor].item() for neighbor in order[:3]])
            """
        ),
        md(
            """
            ## Takeaways

            These plots diagnose geometry and collapse. An untrained fixture run is
            expected to have no scientific relation separation or channel retrieval
            quality; only a trained checkpoint may be interpreted as a learned model.
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
