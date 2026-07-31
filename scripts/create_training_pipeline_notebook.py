#!/usr/bin/env python
"""Generate a real-pipeline one-step CPU training and resume notebook."""

from __future__ import annotations
import argparse
from pathlib import Path
import textwrap
import nbformat as nbf

DEFAULT_OUTPUT=Path(__file__).resolve().parents[1]/"notebooks"/"inspect_training_pipeline.ipynb"
def md(x): return nbf.v4.new_markdown_cell(textwrap.dedent(x).strip())
def code(x): return nbf.v4.new_code_cell(textwrap.dedent(x).strip())


def build_notebook():
    notebook=nbf.v4.new_notebook(cells=[
        md("# Real parquet training pipeline\n\nThis executes the real trainers on a tiny CPU parquet; it is not a fixture-only dry-run."),
        md("## Data, stable split, and train-only normalizer"),
        code(
            """
            from pathlib import Path
            import json, os, sys
            import matplotlib.pyplot as plt
            ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/"src"))
            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
            from hypertagging.training.data_module import build_real_data_module
            from hypertagging.training.pretrain_trainer import PretrainConfig, train_hyperbolic_pretraining
            from hypertagging.training.reconstruction_trainer import ReconstructionConfig, train_level_reconstruction
            requested=os.environ.get("HYPERTAGGING_PARQUET","").strip(); FIXTURE_MODE=not bool(requested)
            data=Path(requested) if requested else Path("/tmp/hypertagging_training_v4.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v4(data)
            OUT=Path(os.environ.get("HYPERTAGGING_FIGURE_DIR","/tmp/hypertagging_figures/training")); OUT.mkdir(parents=True,exist_ok=True)
            module=build_real_data_module(data,seed=20260730,pilot_split_repair=True,max_events=2)
            print("TINY FIXTURE — NOT REAL DATA" if FIXTURE_MODE else "REAL PILOT SAMPLE")
            print({name:[event.event_uid for event in events] for name,events in module.splits.items()})
            assert module.split_counts["train"] > 0
            """
        ),
        md("## One real pretraining step and checkpoint"),
        code(
            """
            pre_dir=OUT/"pretrain"
            pre=train_hyperbolic_pretraining(PretrainConfig(data=str(data),output_dir=str(pre_dir),device="cpu",max_steps=1,batch_size=2,seed=7))
            assert pre.checkpoint.exists()
            print(pre.checkpoint,pre.metrics)
            """
        ),
        md("## Encoder transfer and one all-level reconstruction step"),
        code(
            """
            reco_dir=OUT/"reconstruction"
            reco=train_level_reconstruction(ReconstructionConfig(data=str(data),output_dir=str(reco_dir),pretrained_encoder=str(pre.checkpoint),transfer_leaf_pid_head=True,device="cpu",max_steps=1,batch_size=2,seed=11))
            assert reco.checkpoint.exists() and reco.transfer_report and reco.transfer_report.loaded_keys
            print(reco.transfer_report); print(reco.metrics)
            """
        ),
        md("## Teacher-forced, scheduled, free validation and resume"),
        code(
            """
            resumed=train_level_reconstruction(ReconstructionConfig(data=str(data),output_dir=str(OUT/"resumed"),pretrained_encoder=str(pre.checkpoint),transfer_leaf_pid_head=True,device="cpu",max_steps=2,batch_size=2,seed=11,resume=str(reco.checkpoint)))
            records=[json.loads(line) for line in reco.log_path.read_text().splitlines()]
            display(records)
            assert resumed.steps==2
            losses=[record["loss"] for record in records]
            plt.plot(range(1,len(losses)+1),losses,marker="o"); plt.title("Real trainer JSONL loss")
            plt.tight_layout(); plt.savefig(OUT/"training_loss.png"); plt.show()
            (OUT/"training_pipeline_pass.json").write_text(json.dumps({"pass":True,"checkpoint":str(reco.checkpoint),"resume_step":resumed.steps},indent=2),encoding="utf-8")
            """
        ),
    ])
    notebook.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    return notebook


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True); nbf.write(build_notebook(),a.output); print(a.output); return 0
if __name__=="__main__": raise SystemExit(main())
