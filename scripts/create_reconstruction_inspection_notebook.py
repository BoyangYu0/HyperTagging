#!/usr/bin/env python
"""Generate the level-autoregressive reconstruction inspection notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_level_autoregressive_reconstruction.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Hierarchical level-autoregressive reconstruction inspection

            ## Goal

            Follow one event through relation-aware contextualization, mother queries,
            daughter pointers, deterministic composite construction, teacher forcing,
            and free rollout. The default is an untrained CPU fixture and is not a
            physics-performance result.
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
            from hypertagging.evaluation.hierarchical_metrics import edge_set, summarize_rollout
            from hypertagging.losses.level_reconstruction import level_reconstruction_loss
            from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor, compare_pid_kinematics_modes
            from hypertagging.models.stair_masks import stair_attention_mask
            from hypertagging.preprocessing.pid_filter import PDG_TOKENS
            from hypertagging.reconstruction.level_rollout import RolloutConfig, cached_context_for_level, hard_decode_proposals, level_rollout, proposal_ambiguity_metrics
            from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy

            SEED = int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730"))
            torch.manual_seed(SEED); np.random.seed(SEED)
            requested = os.environ.get("HYPERTAGGING_PARQUET", "").strip()
            FIXTURE_MODE = not bool(requested)
            INPUT_PATH = Path(requested) if requested else Path("/tmp/hypertagging_notebook_fixture_v4.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v4(INPUT_PATH)
            if not INPUT_PATH.exists(): raise FileNotFoundError(INPUT_PATH)
            FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging_figures/reconstruction"))
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            events = load_heterogeneous_events(INPUT_PATH, limit=1, max_nodes=128)
            if not events: raise ValueError("No usable events")
            batch = collate_heterogeneous_events(events)
            model = LevelAutoregressiveReconstructor(
                n_features=batch["node_features"].shape[-1], n_types=len(PDG_TOKENS),
                hidden_dim=24, hyper_dim=6, n_queries=6, n_heads=4, n_context_layers=2,
                use_hyperbolic_relation_refinement=True,
            )
            checkpoint = os.environ.get("HYPERTAGGING_CHECKPOINT", "").strip()
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
            print("TINY FIXTURE / UNTRAINED CPU MODEL" if FIXTURE_MODE and not checkpoint else "SUPPLIED SAMPLE/CHECKPOINT")
            print("Input nodes:", int(batch["node_mask"].sum()))
            """
        ),
        md("## Step 1: input nodes at $S_{\\leq t}$"),
        code(
            """
            input_table = pd.DataFrame({
                "position": range(batch["node_mask"].shape[1]),
                "node_id": batch["node_ids"][0].tolist(),
                "level": batch["level_ids"][0].tolist(),
                "kind": batch["node_kind_ids"][0].tolist(),
                "pid": batch["pid_labels"][0].tolist(),
                "active": batch["node_mask"][0].tolist(),
            })
            display(input_table)
            """
        ),
        md("## Steps 2–4: stair-causal mask and separate Stage-A/Stage-B attention"),
        code(
            """
            model.eval()
            with torch.no_grad(): output = model(batch, target_level=1)
            stair = stair_attention_mask(batch["level_ids"], batch["node_mask"])[0].numpy()
            physical_bias = output.physical_relation_bias[0].numpy()
            physical_attention = output.physical_attention_weights[0].mean(0).numpy()
            hyperbolic_bias = output.hyperbolic_relation_bias[0].numpy()
            hyperbolic_attention = output.hyperbolic_attention_weights[0].mean(0).numpy()
            fig, axes = plt.subplots(1, 5, figsize=(24, 4))
            for axis, matrix, title in zip(
                axes, [stair, physical_bias, physical_attention, hyperbolic_bias, hyperbolic_attention],
                ["Stair-causal allowed mask", "Stage-A physical bias", "Stage-A attention", "Stage-B hyperbolic bias", "Stage-B attention"],
            ):
                image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
                axis.set_title(title); axis.set_xlabel("key"); axis.set_ylabel("query")
                fig.colorbar(image, ax=axis)
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "mask_bias_attention.png"); plt.show()
            assert not stair[0, batch["level_ids"][0].argmax()], "A leaf query leaked a higher-level target"
            """
        ),
        md("## Steps 5–7: mother queries, types, and daughter pointers"),
        code(
            """
            pointer = output.pointer
            object_scores = torch.sigmoid(pointer.object_logits[0]).numpy()
            confidence = torch.sigmoid(pointer.confidence_logits[0]).numpy()
            type_probability = torch.softmax(pointer.type_logits[0], dim=-1).numpy()
            pointer_probability = torch.sigmoid(pointer.pointer_logits[0]).numpy()
            query_table = pd.DataFrame({
                "query": range(len(object_scores)),
                "object_score": object_scores,
                "confidence": confidence,
                "argmax_type": type_probability.argmax(-1),
                "type_probability": type_probability.max(-1),
                "predicted_cardinality": pointer.cardinality_logits[0].argmax(-1).numpy(),
            })
            display(query_table)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            axes[0].bar(query_table["query"], query_table["object_score"]); axes[0].set_title("Mother-query object scores")
            image = axes[1].imshow(pointer_probability, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            axes[1].set(title="Daughter-pointer probabilities", xlabel="node position", ylabel="query")
            fig.colorbar(image, ax=axes[1]); fig.tight_layout()
            fig.savefig(FIGURE_DIR / "query_pointer_predictions.png"); plt.show()
            """
        ),
        md("## Steps 8–10: decoded proposals, composite p4, and updated tree"),
        code(
            """
            rollout_config = RolloutConfig(
                max_level=4, root_types=(), object_threshold=0.4,
                pointer_threshold=0.4, use_cardinality=False, exclusive_final=True,
            )
            decoded = hard_decode_proposals(output, batch, rollout_config)
            display(pd.DataFrame([proposal.__dict__ for proposal in decoded]))
            teacher = level_rollout(model, batch, mode="teacher_forced",
                                    config=RolloutConfig(max_level=4, root_types=(), exclusive_final=False))
            predicted = level_rollout(model, batch, mode="predicted", config=rollout_config)
            bounded = level_rollout(
                model, batch, mode="predicted",
                config=RolloutConfig(
                    max_level=4, root_types=(), object_threshold=0.4,
                    pointer_threshold=0.4, use_cardinality=False,
                    exclusive_final=True, exclusive_resolution="weighted_set_packing",
                    max_resolution_proposals=12,
                ),
            )
            print("Teacher-forced levels:", [(step.target_level, len(step.accepted)) for step in teacher.steps])
            print("Predicted levels:", [(step.target_level, len(step.accepted)) for step in predicted.steps])
            print("Stop reasons:", teacher.stop_reason, predicted.stop_reason, bounded.stop_reason)
            constructed_rows = []
            for step in teacher.steps:
                for node_id in step.appended_node_ids:
                    position = (teacher.batch["node_ids"][0] == node_id).nonzero().flatten()
                    if position.numel():
                        index = int(position[0])
                        constructed_rows.append({
                            "level": step.target_level, "node_id": node_id,
                            "type": int(teacher.batch["pid_labels"][0,index]),
                            "px": float(teacher.batch["p4"][0,index,0]),
                            "py": float(teacher.batch["p4"][0,index,1]),
                            "pz": float(teacher.batch["p4"][0,index,2]),
                            "E": float(teacher.batch["p4"][0,index,3]),
                        })
            display(pd.DataFrame(constructed_rows))
            """
        ),
        md("## Teacher-forced versus free-rollout tree errors"),
        code(
            """
            truth_edges = edge_set(batch)
            teacher_edges = edge_set(teacher.batch)
            predicted_edges = edge_set(predicted.batch)
            edge_comparison = pd.DataFrame({
                "edge": sorted(truth_edges | predicted_edges),
            })
            edge_comparison["truth"] = edge_comparison.edge.isin(truth_edges)
            edge_comparison["predicted"] = edge_comparison.edge.isin(predicted_edges)
            edge_comparison["status"] = np.select(
                [edge_comparison.truth & edge_comparison.predicted,
                 edge_comparison.truth & ~edge_comparison.predicted,
                 ~edge_comparison.truth & edge_comparison.predicted],
                ["correct", "missing", "extra"], default="absent")
            display(edge_comparison)
            def mother_hypotheses(tree_batch):
                hypotheses = {}
                for mother in tree_batch["daughter_adjacency"][0].any(-1).nonzero().flatten().tolist():
                    daughters = tree_batch["daughter_adjacency"][0, mother].nonzero().flatten()
                    sources = tuple(sorted(int(tree_batch["source_node_ids"][0, daughter]) for daughter in daughters))
                    hypotheses[sources] = int(tree_batch["pid_labels"][0, mother])
                return hypotheses
            truth_hypotheses, predicted_hypotheses = mother_hypotheses(batch), mother_hypotheses(predicted.batch)
            hypothesis_rows = []
            for daughters in sorted(truth_hypotheses.keys() | predicted_hypotheses.keys()):
                truth_type, predicted_type = truth_hypotheses.get(daughters), predicted_hypotheses.get(daughters)
                hypothesis_rows.append({
                    "daughter_sources": daughters, "truth_type": truth_type, "predicted_type": predicted_type,
                    "status": "missing" if predicted_type is None else ("extra" if truth_type is None else
                              ("correct" if truth_type == predicted_type else "wrong mother type")),
                })
            display(pd.DataFrame(hypothesis_rows))
            import networkx as nx
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for axis, tree_batch, label in [(axes[0], batch, "Truth tree"), (axes[1], predicted.batch, "Predicted rollout")]:
                edges = edge_set(tree_batch)
                graph = nx.DiGraph(); graph.add_edges_from(edges)
                graph.add_nodes_from(int(value) for value in tree_batch["node_ids"][0, tree_batch["node_mask"][0]].tolist())
                position = {}
                for index, node_id in enumerate(tree_batch["node_ids"][0, tree_batch["node_mask"][0]].tolist()):
                    level = int(tree_batch["level_ids"][0, index])
                    same_level = [int(value) for value in tree_batch["node_ids"][0, tree_batch["node_mask"][0]].tolist()
                                  if int(tree_batch["level_ids"][0, (tree_batch["node_ids"][0] == value).nonzero()[0]]) == level]
                    position[int(node_id)] = (same_level.index(int(node_id)), level)
                nx.draw_networkx(graph, position, ax=axis, node_size=700, font_size=8, arrows=True)
                axis.set_title(label); axis.set_axis_off()
            fig.tight_layout(); fig.savefig(FIGURE_DIR / "truth_predicted_trees.png"); plt.show()
            display(pd.Series(summarize_rollout(predicted.batch, batch), name="predicted_rollout"))
            first_divergence = next(
                (step.target_level for step in predicted.steps
                 if {(proposal.mother_type, proposal.daughter_positions) for proposal in step.accepted}
                 != {(proposal.mother_type, proposal.daughter_positions) for proposal in teacher.steps[step.target_level-1].accepted}
                 if step.target_level-1 < len(teacher.steps)),
                None,
            )
            print("First rollout divergence level:", first_divergence)
            print("Missing/extra edges and wrong types above show error propagation to later levels.")
            ambiguity_rows=[]
            for step in predicted.steps:
                context=cached_context_for_level(predicted,step.target_level)
                if "recursive_leaf_source_mask" not in context: continue
                row={"level":step.target_level,**proposal_ambiguity_metrics(
                    list(step.proposals),list(step.accepted),
                    total_queries=step.model_output.pointer.object_logits.shape[1],
                    recursive_leaf_source_mask=context["recursive_leaf_source_mask"][0],
                )}
                ambiguity_rows.append(row)
            display(pd.DataFrame(ambiguity_rows))
            print({
                "greedy_tree":summarize_rollout(predicted.batch,batch),
                "bounded_set_packing_tree":summarize_rollout(bounded.batch,batch),
                "bounded_alternative_is_evaluation_only":True,
            })
            """
        ),
        md("## Leaf PID, level conditioning, confidence, and recursive-source audit"),
        code(
            """
            from hypertagging.evaluation.hierarchical_metrics import canonical_tree_metrics
            from hypertagging.training.scheduled_sampling import (
                TeacherForcingSchedule, aligned_level_targets,
                combine_sampled_context_losses, resolve_unrepresentable_target_policy,
            )
            with torch.no_grad():
                level1=model(batch,target_level=1)
                level2=model(batch,target_level=2)
            leaf_mask=batch["node_mask"]&(batch["node_kind_ids"]==1)
            leaf_pid_prediction=level1.leaf_pid_logits.softmax(-1).argmax(-1)
            display(pd.DataFrame({
                "position":leaf_mask[0].nonzero().flatten().tolist(),
                "predicted_leaf_pid_token":leaf_pid_prediction[0,leaf_mask[0]].tolist(),
                "truth_pid_token":batch["truth_pid_labels"][0,leaf_mask[0]].tolist(),
            }))
            print("Target-level query conditioning changes logits:",
                  not torch.allclose(level1.pointer.pointer_logits,level2.pointer.pointer_logits))
            scheduled=level_rollout(model,batch,mode="scheduled",config=RolloutConfig(max_level=4,root_types=(),scheduled_sampling_probability=.5))
            canonical=canonical_tree_metrics(teacher.batch,batch)
            schedule=TeacherForcingSchedule(kind="linear",start_probability=1.0,end_probability=.2,duration_steps=1000)
            aligned=aligned_level_targets(batch, predicted.batch, target_level=1)
            fallback=resolve_unrepresentable_target_policy(
                "fallback_teacher", truth_target_count=1, representable_target_count=0
            )
            constraint_policy=ReconstructionConstraintPolicy(
                empirical_type_prior_mode="soft", mother_charge_compatibility="soft_train_hard_rollout"
            )
            constraint_round_trip=(
                ReconstructionConstraintPolicy.from_dict(constraint_policy.to_dict()) == constraint_policy
            )
            primary, primary_metrics=combine_sampled_context_losses(
                torch.tensor([1.0]),torch.tensor([3.0]),
                choose_teacher=torch.tensor([False]),auxiliary_teacher_weight=0.0,
            )
            scheduled_report={
                "sampled_context":"predicted",
                "primary_loss":float(primary),
                "teacher_event_double_counted":False,
                "auxiliary_teacher_weight":0.0,
                "representable_targets":aligned.representable_count,
                "unrepresentable_targets":aligned.truth_target_count-aligned.representable_count,
                "fallback_teacher_on_unrepresentable":fallback.use_teacher_context,
                "optimizer_step_10_teacher_probability":schedule.probability(10),
                "schedule_not_at_endpoint":schedule.probability(10)>schedule.end_probability,
                "constraint_policy_round_trip":constraint_round_trip,
                "cached_state_count":len(teacher.cached_states),
                "rollout_forward_count":len(teacher.steps),
                **primary_metrics,
            }
            pid_mismatch=compare_pid_kinematics_modes(model,batch,target_level=1,temperature=.5)
            scheduled_report.update(pid_mismatch)
            scheduled_report["duplicate_metrics_by_level"]=ambiguity_rows
            scheduled_report["greedy_rollout"]=summarize_rollout(predicted.batch,batch)
            scheduled_report["bounded_set_packing_rollout"]=summarize_rollout(bounded.batch,batch)
            scheduled_report["evaluation_slices"]={
                "channel_pair":[int(batch["b1_full_truth_channel_ids"][0]),int(batch["b2_full_truth_channel_ids"][0])],
                "leaf_multiplicity":int(((batch["level_ids"]==0)&batch["node_mask"]).sum()),
                "truth_depth":int(batch["level_ids"][batch["node_mask"]].max()),
            }
            (FIGURE_DIR/"scheduled_context_report.json").write_text(json.dumps(scheduled_report,indent=2))
            print({
                "confidence_scores":torch.sigmoid(level1.pointer.confidence_logits).tolist(),
                "recursive_leaf_source_shape":tuple(batch["recursive_leaf_source_mask"].shape),
                "canonical_teacher_exact":canonical.full_tree_exact_match,
                "scheduled_stop":scheduled.stop_reason,
                "partial_target_policy":"complete/reconstructable targets with min_daughters=2",
                "scheduled_teacher_probability_at_optimizer_step_10":schedule.probability(10),
                "cached_context_max_level_for_target_3":int(cached_context_for_level(teacher,3)["level_ids"].max()),
                "representable_targets":aligned.representable_count,
                "unrepresentable_targets":aligned.truth_target_count-aligned.representable_count,
                "first_context_divergence_level":canonical.first_divergence_level,
                "primary_sampled_loss":float(primary),
            })
            assert canonical.full_tree_exact_match
            """
        ),
        md("## Complete CPU forward/loss/backward/optimizer smoke step"),
        code(
            """
            torch.manual_seed(SEED)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            train_output = model(batch, target_level=1)
            loss_output = level_reconstruction_loss(train_output.pointer, batch, target_level=1)
            loss_output.total.backward()
            finite_gradients = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            relation_gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in model.relation_bias.parameters()
                if parameter.grad is not None
            )
            optimizer.step()
            print({
                "total_loss": float(loss_output.total.detach()),
                "components": {name: float(value.detach()) for name, value in loss_output.components.items()},
                "hungarian_matches": loss_output.matches,
                "finite_gradients": finite_gradients,
                "relation_bias_gradient_norm_l1": relation_gradient,
            })
            assert torch.isfinite(loss_output.total)
            assert finite_gradients
            assert relation_gradient > 0
            """
        ),
        md(
            """
            ## Takeaways

            The smoke step proves that relation features alter trainable attention,
            mother queries are matched as an unordered set, and all appended mother
            four-vectors are daughter sums. Free-rollout output from an untrained model
            is a software diagnostic only.
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
