#!/usr/bin/env python
"""Generate query-capacity, sparse-loss, confidence, and matching inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import nbformat as nbf

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_query_capacity_and_losses.ipynb"


def md(value: str): return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())
def code(value: str): return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build_notebook():
    notebook = nbf.v4.new_notebook(cells=[
        md("# Query capacity and sparse reconstruction losses\n\nFixture output is a software check, not physics performance."),
        md("## Setup"),
        code(
            """
            from pathlib import Path
            import json, os, sys
            from collections import Counter
            import matplotlib.pyplot as plt
            import pandas as pd
            import torch
            ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/"src"))
            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
            from hypertagging.data.heterogeneous import load_heterogeneous_events, collate_heterogeneous_events
            from hypertagging.data.capacity import dataset_capacity_statistics
            from hypertagging.losses.level_reconstruction import focal_binary_cross_entropy_with_logits
            from hypertagging.losses.set_matching import matching_cost, hungarian_assignment
            from hypertagging.training.model_config import MODEL_PRESETS
            requested=os.environ.get("HYPERTAGGING_PARQUET","").strip(); FIXTURE_MODE=not bool(requested)
            path=Path(requested) if requested else Path("/tmp/hypertagging_capacity_v3.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v4(path)
            events=load_heterogeneous_events(path)
            OUT=Path(os.environ.get("HYPERTAGGING_FIGURE_DIR","/tmp/hypertagging_figures/capacity")); OUT.mkdir(parents=True,exist_ok=True)
            print("TINY FIXTURE — NOT REAL DATA" if FIXTURE_MODE else "REAL SAMPLE")
            """
        ),
        md("## Mothers per level, query usage, and overflow"),
        code(
            """
            stats=dataset_capacity_statistics(events,global_n_queries=8,global_max_cardinality=6)
            architecture=MODEL_PRESETS["production_baseline"]
            query_by_level=dict(architecture.n_queries_by_level)
            cardinality_by_level=dict(architecture.max_cardinality_by_level)
            level_rows=[]
            retained_levels=sorted({int(level) for event in events for level in event.level_ids[event.active].tolist() if int(level)>0})
            for level in retained_levels:
                mother_counts=[]; cardinalities=[]
                for event in events:
                    mothers=(event.active&(event.level_ids==level)&event.valid_reconstruction_target&event.recursive_reconstructable_complete).nonzero().flatten()
                    mother_counts.append(int(mothers.numel()))
                    cardinalities.extend(int(event.daughter_adjacency[index].sum()) for index in mothers.tolist())
                queries=query_by_level.get(level,architecture.n_queries)
                max_cardinality=cardinality_by_level.get(level,architecture.max_cardinality)
                level_rows.append({
                    "level":level,"maximum_mothers":max(mother_counts,default=0),
                    "cardinality_distribution":dict(Counter(cardinalities)),
                    "configured_queries":queries,"configured_max_cardinality":max_cardinality,
                    "overflow_count":sum(value>queries for value in mother_counts)+sum(value>max_cardinality for value in cardinalities),
                    "query_capacity_margin":queries-max(mother_counts,default=0),
                    "cardinality_capacity_margin":max_cardinality-max(cardinalities,default=0),
                })
            report={
                "schema_default":"direct-mdst-tree-v4",
                "target_policy":"complete_only",
                "production_baseline_by_level":level_rows,
                "maximum_mothers_per_level":stats.maximum_mothers_per_level,
                "percentiles":stats.percentile_mothers_per_level,
                "daughter_cardinality_counts":stats.daughter_cardinality_counts,
                "maximum_daughter_cardinality":stats.maximum_daughter_cardinality,
                "query_overflow_rate":stats.query_overflow_rate,
                "cardinality_overflow_rate":stats.cardinality_overflow_rate,
            }
            print(json.dumps(report,indent=2)); (OUT/"capacity_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            assert stats.query_overflow_rate==0 and stats.cardinality_overflow_rate==0
            assert all(row["overflow_count"]==0 for row in level_rows)
            pd.Series(stats.maximum_mothers_per_level).plot.bar(title="Maximum mothers per target level")
            plt.tight_layout(); plt.savefig(OUT/"mothers_per_level.png"); plt.show()
            """
        ),
        md("## Object/pointer imbalance and focal weighting"),
        code(
            """
            logits=torch.tensor([-3.,-2.,-1.,0.,1.],requires_grad=True); targets=torch.tensor([0.,0.,0.,0.,1.])
            unweighted=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
            weighted=focal_binary_cross_entropy_with_logits(logits,targets,positive_weight=5,gamma=2)
            weighted.backward()
            loss_table=pd.DataFrame({"loss":["unweighted BCE","weighted focal BCE"],"value":[float(unweighted),float(weighted)]})
            display(loss_table); loss_table.to_csv(OUT/"sparse_loss_table.csv",index=False)
            plt.bar(loss_table.loss,loss_table.value); plt.xticks(rotation=20); plt.tight_layout()
            plt.savefig(OUT/"sparse_loss_comparison.png"); plt.show()
            print("Object positive fraction:",float(targets.mean()),"pointer positive fraction:",float(targets.mean()))
            """
        ),
        md("## Hard negatives, confidence calibration, and Hungarian costs"),
        code(
            """
            type_logits=torch.tensor([[3.,0.],[0.,3.]])
            pointer_logits=torch.tensor([[3.,-3.,-3.],[-3.,3.,3.]])
            truth_types=torch.tensor([0,1]); truth_masks=torch.tensor([[1,0,0],[0,1,1]],dtype=torch.bool)
            cost=matching_cost(type_logits=type_logits,pointer_logits=pointer_logits,target_types=truth_types,target_masks=truth_masks)
            assignment=hungarian_assignment(cost,production=False,allow_bruteforce=True)
            display(pd.DataFrame(cost.numpy())); print("Hungarian assignment:",assignment)
            confidence=pd.DataFrame({"target":[0,0.4,0.8,1],"prediction":[0.05,0.45,0.75,0.95]})
            confidence.to_csv(OUT/"confidence_calibration.csv",index=False)
            plt.plot(confidence.target,confidence.prediction,marker="o"); plt.plot([0,1],[0,1],"--")
            plt.title("Confidence calibration fixture"); plt.tight_layout(); plt.savefig(OUT/"confidence_calibration.png"); plt.show()
            print("Hard negative example: the second-best pointer-compatible non-truth combination.")
            """
        ),
    ])
    notebook.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    return notebook


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True); nbf.write(build_notebook(),args.output); print(args.output); return 0


if __name__=="__main__": raise SystemExit(main())
