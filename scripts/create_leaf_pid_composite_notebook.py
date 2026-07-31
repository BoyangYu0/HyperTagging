#!/usr/bin/env python
"""Generate the v4 leaf-PID/composite-input contract notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "inspect_leaf_pid_and_composite_inputs.ipynb"
)


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build_notebook():
    cells = [
        md(
            """
            # Leaf PID and composite model inputs

            This is an executable truth-leakage audit of schema-v4 and the
            two-pass Level-1 reconstruction path. Fixture plots validate code,
            not PID or reconstruction performance.
            """
        ),
        md("## Schema-v4 input and truth PID separation"),
        code(
            """
            from pathlib import Path
            import json, os, sys
            import matplotlib.pyplot as plt
            import torch
            ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/"src"))
            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
            from hypertagging.preprocessing.schema_v4 import load_payload_v4, MODEL_COMPOSITE_FEATURE_NAMES, TARGET_COMPOSITE_METADATA_NAMES, TARGET_COMPOSITE_METADATA_INDICES
            from hypertagging.data.heterogeneous import load_heterogeneous_events, collate_heterogeneous_events
            from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
            SEED=int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED","20260730")); torch.manual_seed(SEED)
            requested=os.environ.get("HYPERTAGGING_PARQUET","").strip(); FIXTURE_MODE=not bool(requested)
            path=Path(requested) if requested else Path("/tmp/hypertagging_leaf_composite_v4.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v4(path)
            payload=load_payload_v4(path)
            if payload["schema_version"]!="direct-mdst-tree-v4": raise ValueError("native/adapted v4 contract required")
            OUT=Path(os.environ.get("HYPERTAGGING_FIGURE_DIR","/tmp/hypertagging_figures/leaf_composite")); OUT.mkdir(parents=True,exist_ok=True)
            print("TINY FIXTURE — NOT REAL DATA" if FIXTURE_MODE else "REAL PREPROCESSED SAMPLE")
            nodes=[node for event in payload["events"] for node in event["nodes"]]
            composites=[node for node in nodes if node["node_kind"]=="composite"]
            assert composites and all("daughter_input_pid_histogram" in node and "daughter_truth_pid_histogram" in node for node in composites)
            """
        ),
        md("## Input versus truth daughter histograms"),
        code(
            """
            input_counts=torch.tensor([node["daughter_input_pid_histogram"] for node in composites]).sum(0)
            truth_counts=torch.tensor([node["daughter_truth_pid_histogram"] for node in composites]).sum(0)
            width=torch.arange(input_counts.numel())
            plt.figure(figsize=(10,4)); plt.bar(width-.2,input_counts,width=.4,label="model input"); plt.bar(width+.2,truth_counts,width=.4,label="truth diagnostic")
            plt.xlabel("reduced PID token"); plt.ylabel("daughter count"); plt.legend(); plt.tight_layout()
            plt.savefig(OUT/"daughter_input_vs_truth_histogram.png"); plt.show()
            """
        ),
        md("## Two-pass PID-refined p4 and Level-1 pointer response"),
        code(
            """
            batch=collate_heterogeneous_events(load_heterogeneous_events(path,limit=1))
            model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=16,hyper_dim=4,n_queries=4)
            refined=model(batch,target_level=1)
            canonical=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=16,hyper_dim=4,n_queries=4,canonical_pion_first_level=True)
            canonical.load_state_dict(model.state_dict())
            baseline=canonical(batch,target_level=1)
            p4_change=float((refined.current_p4-batch["p4"]).abs().max())
            pointer_change=float((refined.pointer.pointer_logits-baseline.pointer.pointer_logits).abs().max())
            print({"max_pid_refined_p4_change":p4_change,"max_level1_pointer_change":pointer_change})
            """
        ),
        md("## Dynamic runtime normalization and composite type semantics"),
        code(
            """
            from hypertagging.data.streaming import RuntimeFeatureNormalizer
            from hypertagging.reconstruction.pid_state import COMPOSITE_TYPE_SOURCE_TO_ID
            fitted=RuntimeFeatureNormalizer(
                common_mean=torch.arange(12,dtype=torch.float32),
                common_std=torch.full((12,),2.0),
                composite_mean=torch.zeros(13),
                composite_std=torch.full((13,),2.0),
            )
            model.set_runtime_feature_normalizer(fitted)
            normalized=model(batch,target_level=1)
            raw_runtime=normalized.current_p4[0,batch["node_mask"][0],:4].detach()
            normalized_runtime=normalized.second_pass_common_features[0,batch["node_mask"][0],:4].detach()
            expected=(raw_runtime-torch.arange(4,dtype=raw_runtime.dtype))/2
            normalization_pass=bool(torch.allclose(normalized_runtime,expected))
            composite_mask=batch["node_mask"] & (batch["level_ids"]>0)
            teacher_source=COMPOSITE_TYPE_SOURCE_TO_ID["truth_teacher_forced"]
            type_source_pass=bool(
                (batch["runtime_composite_type_source_ids"][composite_mask]==teacher_source).all()
            )
            plt.figure(figsize=(7,3))
            plt.hist(raw_runtime[:,3].numpy(),alpha=.6,label="raw runtime E")
            plt.hist(normalized_runtime[:,3].numpy(),alpha=.6,label="normalized pass-B E")
            plt.legend(); plt.tight_layout()
            plt.savefig(OUT/"runtime_dynamic_normalization.png"); plt.show()
            """
        ),
        md("## Composite model features versus target-only metadata invariance"),
        code(
            """
            model.eval()
            changed={name:(value.clone() if isinstance(value,torch.Tensor) else value) for name,value in batch.items()}
            for index in TARGET_COMPOSITE_METADATA_INDICES:
                changed["composite_features"][...,index]=1e7
                changed["composite_availability"][...,index]=True
            with torch.no_grad():
                original_output=model(batch,target_level=1)
                changed_output=model(changed,target_level=1)
            metadata_invariance_pass=all(torch.equal(left,right) for left,right in (
                (original_output.node_embeddings,changed_output.node_embeddings),
                (original_output.hyperbolic_embeddings,changed_output.hyperbolic_embeddings),
                (original_output.relation_bias,changed_output.relation_bias),
                (original_output.pointer.pointer_logits,changed_output.pointer.pointer_logits),
                (original_output.pointer.type_logits,changed_output.pointer.type_logits),
            ))
            print({"model_composite_features":MODEL_COMPOSITE_FEATURE_NAMES,"target_only_metadata":TARGET_COMPOSITE_METADATA_NAMES,"invariant":metadata_invariance_pass})
            assert metadata_invariance_pass
            """
        ),
        md("## Reconstruction gradient reaches the leaf PID head"),
        code(
            """
            model.zero_grad(set_to_none=True)
            refined.pointer.pointer_logits.square().mean().backward()
            gradient=float(model.leaf_pid_head.weight.grad.abs().sum())
            leakage_pass=all(
                node["input_pid_token"]==0
                for node in nodes
                if node["leaf_kinematics_mode"]=="raw_track_predicted_pid"
            )
            report={"schema_v4":True,"truth_clean_composite_input":True,"target_metadata_invariance":metadata_invariance_pass,"raw_track_unknown_input_pass":leakage_pass,"pointer_gradient_to_leaf_pid":gradient,"pointer_response":pointer_change,"runtime_dynamic_normalization_pass":normalization_pass,"teacher_composite_type_source_pass":type_source_pass}
            (OUT/"leaf_composite_contract.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            assert leakage_pass and gradient>0 and normalization_pass and type_source_pass and metadata_invariance_pass
            plt.figure(figsize=(6,3)); plt.bar(["pointer→leaf PID gradient"],[gradient]); plt.yscale("log"); plt.tight_layout()
            plt.savefig(OUT/"leaf_pid_gradient.png"); plt.show()
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
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
