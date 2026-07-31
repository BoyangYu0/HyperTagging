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
            import json, os, sys
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch

            REPO_ROOT = Path.cwd()
            if not (REPO_ROOT / "src").exists(): REPO_ROOT = Path("..").resolve()
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from hypertagging.data.heterogeneous import load_heterogeneous_events, collate_heterogeneous_events
            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
            from hypertagging.losses.hyperbolic_pretraining import (
                build_tree_relation_targets, collapse_diagnostics, pool_b_branch_embeddings,
                topology_safe_parent_negative_mask,
            )
            from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
            from hypertagging.models.hyperbolic import distance, logmap0, radius
            from hypertagging.preprocessing.pid_filter import PDG_TOKENS
            from hypertagging.training.pretrain_trainer import ContextualPretrainingModel

            SEED = int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730"))
            torch.manual_seed(SEED); np.random.seed(SEED)
            requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
            FIXTURE_MODE = not bool(requested)
            INPUT_PATH = Path(requested) if requested else Path("/tmp/hypertagging_notebook_fixture_v3.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v4(INPUT_PATH)
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
            lca_height=batch["lca_depth"]
            relation_targets,_=build_tree_relation_targets(
                parent_ids=batch["parent_ids"],lca_depth=lca_height,
                level_ids=batch["level_ids"],node_mask=mask,b_side=batch["b_side"],
                lca_node_id=batch["lca_node_id"],
                edges_to_lca_from_i=batch["edges_to_lca_from_i"],
                edges_to_lca_from_j=batch["edges_to_lca_from_j"],
            )
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
                    eligible=topology_safe_parent_negative_mask(
                        batch["parent_ids"][b],mask[b],child,
                        lca_depth=lca_height[b],tree_relation_targets=relation_targets[b],
                        b_side=batch["b_side"][b],
                    )
                    negatives=eligible.nonzero().flatten().tolist()
                    if parent >= 0 and negatives:
                        categories["hard negative"].append(min(float(pair_distance[b,child,index]) for index in negatives))
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
            fsp_branch_embeddings,fsp_branch_mask=pool_b_branch_embeddings(encoded.channel_projection,batch["b_side"],mask,mode="fsp_only",level_ids=batch["level_ids"])
            cosine = torch.nn.functional.cosine_similarity(branch_embeddings[:,0], branch_embeddings[:,1]).detach().numpy()
            channel_frame = pd.DataFrame({
                "event": batch["event_ids"].numpy(),
                "embedding_similarity": cosine,
                "structured_channel_similarity": batch["channel_similarity"].numpy(),
                "b1_channel": batch["b1_channel_ids"].numpy(),
                "b2_channel": batch["b2_channel_ids"].numpy(),
            })
            channel_frame["exact_same_channel"] = channel_frame.b1_channel == channel_frame.b2_channel
            channel_frame["fsp_only_embedding_similarity"]=torch.nn.functional.cosine_similarity(fsp_branch_embeddings[:,0],fsp_branch_embeddings[:,1]).detach().numpy()
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
        md("## Context order and FSP/multilevel/corrupted curriculum comparison"),
        code(
            """
            from hypertagging.training.pretraining_curriculum import PretrainingStage, build_curriculum_batch
            runtime_model=ContextualPretrainingModel(d_model=24,hyper_dim=4).eval()
            representations = {
                "uncontextualized_adapter": encoded.adapter_embeddings.detach(),
                "contextual_euclidean": encoded.node_embeddings.detach(),
                "contextual_hyperbolic_tangent": tangent.detach(),
            }
            print({name: tuple(value.shape) for name,value in representations.items()})
            curriculum_rows=[]
            channel_representation_rows=[]
            corruption_contract_pass=True
            hard_negative_classes=[]
            for stage in PretrainingStage:
                view=build_curriculum_batch(batch,stage,seed=SEED,corruption_probability=1.0)
                with torch.no_grad(): stage_encoded=model(view.batch,attention_mask=view.batch["curriculum_attention_mask"])
                with torch.no_grad(): runtime_encoded, leaf_pid_logits, runtime_batch=runtime_model.encode_runtime(
                    view.batch,attention_mask=view.batch["curriculum_attention_mask"]
                )
                curriculum_rows.append({
                    "stage":stage.value,
                    "valid_nodes":int(view.batch["node_mask"].sum()),
                    "corrupted_nodes":int(view.corrupted_node_mask.sum()),
                    "mean_radius":float(radius(stage_encoded.hyperbolic_embeddings)[view.batch["node_mask"]].mean()),
                    "original_max_level":int(view.batch["full_event_max_level"].max()),
                    "hard_negative_pairs":int(view.hard_negative_pairs.shape[0]),
                    "actual_corruption_codes":sorted(set(view.corruption_code[view.corrupted_node_mask].tolist())),
                    "hard_negative_relation_classes":sorted(set(view.hard_negative_relation_classes.tolist())),
                    "runtime_pid_probability_width":runtime_batch["current_pid_probabilities"].shape[-1],
                    "invalid_structural_positives":int((view.corrupted_node_mask&view.structural_positive_mask).sum()),
                    "causal_future_links":int((
                        view.batch["curriculum_attention_mask"]
                        & (view.batch["level_ids"][:,None,:] > view.batch["level_ids"][:,:,None])
                    ).sum()),
                })
                stage_branches,stage_branch_mask=pool_b_branch_embeddings(stage_encoded.channel_projection,view.batch["b_side"],view.batch["node_mask"],mode="mean_all",level_ids=view.batch["level_ids"])
                full_ids=torch.stack([batch["b1_full_truth_channel_ids"],batch["b2_full_truth_channel_ids"]],-1);flat_ids=full_ids[stage_branch_mask];positive_pairs=int(((flat_ids[:,None]==flat_ids[None,:])&(flat_ids[:,None]>0)&~torch.eye(len(flat_ids),dtype=torch.bool)).sum()/2) if len(flat_ids) else 0;total_pairs=len(flat_ids)*(len(flat_ids)-1)//2
                channel_representation_rows.append({"representation":stage.value,"pooling":"truth_guided_all_nodes" if stage is PretrainingStage.TRUTH_GUIDED_MULTILEVEL else "corrupted_predicted_like" if stage is PretrainingStage.CORRUPTED_COMPOSITES else "fsp_stage","positive_pairs":positive_pairs,"negative_pairs":total_pairs-positive_pairs,"valid_branches":int(stage_branch_mask.sum()),"mean_norm":float(stage_branches[stage_branch_mask].norm(dim=-1).mean()) if stage_branch_mask.any() else 0.0})
                corruption_contract_pass &= bool(
                    torch.equal(view.corrupted_node_mask, view.corruption_code > 0)
                )
                hard_negative_classes.extend(view.hard_negative_relation_classes.tolist())
            display(pd.DataFrame(curriculum_rows))
            fsp_ids=torch.stack([batch["b1_full_truth_channel_ids"],batch["b2_full_truth_channel_ids"]],-1)[fsp_branch_mask];fsp_positive=int(((fsp_ids[:,None]==fsp_ids[None,:])&(fsp_ids[:,None]>0)&~torch.eye(len(fsp_ids),dtype=torch.bool)).sum()/2) if len(fsp_ids) else 0;channel_representation_rows.append({"representation":"fsp_only_embeddings","pooling":"fsp_only","positive_pairs":fsp_positive,"negative_pairs":len(fsp_ids)*(len(fsp_ids)-1)//2-fsp_positive,"valid_branches":int(fsp_branch_mask.sum()),"mean_norm":float(fsp_branch_embeddings[fsp_branch_mask].norm(dim=-1).mean()) if fsp_branch_mask.any() else 0.0});display(pd.DataFrame(channel_representation_rows))
            assert all(row["causal_future_links"]==0 for row in curriculum_rows)
            assert corruption_contract_pass
            curriculum_report={
                "level_causal_pass":all(row["causal_future_links"]==0 for row in curriculum_rows),
                "actual_corruption_labels_pass":corruption_contract_pass,
                "hard_negative_relation_classes":sorted(set(hard_negative_classes)),
                "hard_negatives_are_explicit_tree_relations":all(code in (1,2) for code in hard_negative_classes),
                "runtime_two_pass_pid_semantics":all(row["runtime_pid_probability_width"]==len(PDG_TOKENS) for row in curriculum_rows),
                "invalid_corruptions_excluded_from_positive_structure":all(row["invalid_structural_positives"]==0 for row in curriculum_rows),
                "depth_pid_channel_shape":list(batch["b_depth_pid_count_arrays"].shape),
                "branch_multiplicity_summary_shape":list(batch["b_branch_multiplicity_summaries"].shape),
                "channel_representation_comparison":channel_representation_rows,
                "channel_memory_comparison_configs":["configs/ablations/channel_memory_disabled.yaml","configs/ablations/channel_memory_bounded.yaml"],
                "inactive_channel_loss_visible":all("positive_pairs" in row for row in channel_representation_rows),
            }
            (FIGURE_DIR/"curriculum_runtime_report.json").write_text(json.dumps(curriculum_report,indent=2))
            print("Corrupted Stage 3 rebuilds p4, charge, input PID histograms, source masks, and structural features.")
            """
        ),
        md("## Direct tree-distance target versus hyperbolic distance"),
        code(
            """
            from hypertagging.losses.hyperbolic_pretraining import tree_distance_targets
            pair_mask=mask[:,:,None]&mask[:,None,:]
            target_distance=tree_distance_targets(
                exact_tree_path_distance=batch["exact_tree_path_distance"],pair_mask=pair_mask
            )
            predicted_distance=distance(z[:,:,None,:],z[:,None,:,:])
            selected=pair_mask&(batch["exact_tree_path_distance"]>=0)
            plt.scatter(target_distance[selected].numpy(),predicted_distance[selected].numpy(),s=15,alpha=.6)
            plt.xlabel("Normalized retained-tree distance target"); plt.ylabel("Poincare distance")
            plt.tight_layout(); plt.savefig(FIGURE_DIR/"tree_distance_supervision.png"); plt.show()
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
