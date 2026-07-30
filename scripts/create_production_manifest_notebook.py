#!/usr/bin/env python
"""Generate the production-manifest planning/validation notebook."""

from __future__ import annotations
import argparse
from pathlib import Path
import textwrap
import nbformat as nbf

DEFAULT_OUTPUT=Path(__file__).resolve().parents[1]/"notebooks"/"inspect_production_manifest.ipynb"
def md(x): return nbf.v4.new_markdown_cell(textwrap.dedent(x).strip())
def code(x): return nbf.v4.new_code_cell(textwrap.dedent(x).strip())


def build_notebook():
    notebook=nbf.v4.new_notebook(cells=[
        md("# Production manifest inspection\n\nRead-only planning and shard QA. This notebook never submits jobs."),
        md("## Manifest/config setup"),
        code(
            """
            from pathlib import Path
            import json, os, sys
            import matplotlib.pyplot as plt
            import pandas as pd
            ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/"src"))
            from hypertagging.preprocessing.pid_filter import PID_VOCABULARY_VERSION
            from hypertagging.preprocessing.schema_v3 import SCHEMA_VERSION_V3,feature_spec_v3
            OUT=Path(os.environ.get("HYPERTAGGING_FIGURE_DIR","/tmp/hypertagging_figures/manifest")); OUT.mkdir(parents=True,exist_ok=True)
            requested=os.environ.get("HYPERTAGGING_MANIFEST","").strip()
            if requested:
                records=[json.loads(line) for line in Path(requested).read_text().splitlines() if line.strip()]
                MODE="REAL PRODUCTION MANIFEST"
            else:
                records=[
                    {"task_id":0,"input_file":"fixture-a.root","physics_category":"charged","entry_start":0,"entry_stop_exclusive":10,"planned_events":10,"output_file":"/tmp/missing-0.parquet"},
                    {"task_id":1,"input_file":"fixture-a.root","physics_category":"charged","entry_start":10,"entry_stop_exclusive":20,"planned_events":10,"output_file":"/tmp/missing-1.parquet"},
                ]
                for row in records: row.update(schema_version=SCHEMA_VERSION_V3,pid_vocabulary_version=PID_VOCABULARY_VERSION,feature_spec_hash=feature_spec_v3()["feature_spec_hash"],leaf_kinematics_mode="raw_track_predicted_pid",charge_conjugate_normalization=False,git_commit="fixture")
                MODE="TINY MANIFEST FIXTURE — NOT REAL PRODUCTION"
            print(MODE)
            frame=pd.DataFrame(records); display(frame)
            """
        ),
        md("## Planned counts, categories, ranges, and memory"),
        code(
            """
            overlaps=[]
            for source,group in frame.groupby("input_file"):
                ordered=group.sort_values("entry_start")
                previous=None
                for row in ordered.itertuples():
                    if previous is not None and previous>row.entry_start: overlaps.append((source,previous,row.entry_start))
                    previous=row.entry_stop_exclusive
            frame["estimated_memory_gb"]=frame.planned_events*0.00005
            summary={"planned_events":int(frame.planned_events.sum()),"tasks":len(frame),"overlaps":overlaps,"schema_versions":sorted(frame.schema_version.unique()),"feature_hashes":sorted(frame.feature_spec_hash.unique()),"pid_versions":sorted(frame.pid_vocabulary_version.unique()),"completed":int(frame.output_file.map(lambda x:Path(x).exists()).sum()),"missing":int((~frame.output_file.map(lambda x:Path(x).exists())).sum())}
            print(json.dumps(summary,indent=2)); assert not overlaps and summary["schema_versions"]==[SCHEMA_VERSION_V3]
            frame.groupby("physics_category").planned_events.sum().plot.bar(title="Planned category distribution")
            plt.tight_layout(); plt.savefig(OUT/"manifest_categories.png"); plt.show()
            """
        ),
        md("## Completed/missing/invalid shards and global UID checks"),
        code(
            """
            status=frame[["task_id","output_file"]].copy(); status["exists"]=status.output_file.map(lambda x:Path(x).exists())
            display(status)
            report={"pass":not overlaps,"global_uid_check":"requires completed shards" if not status.exists.any() else "run validator","summary":summary}
            (OUT/"production_manifest_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print("For real manifests run: python scripts/mdst_batch_production.py validate --manifest",requested or "<manifest.jsonl>")
            """
        ),
    ])
    notebook.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    return notebook
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True); nbf.write(build_notebook(),a.output); print(a.output); return 0
if __name__=="__main__": raise SystemExit(main())
