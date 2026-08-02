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
            from hypertagging.preprocessing.schema_v4 import SCHEMA_VERSION_V4,feature_spec_v4
            from scripts import mdst_batch_production as production
            OUT=Path(os.environ.get("HYPERTAGGING_FIGURE_DIR","/tmp/hypertagging_figures/manifest")); OUT.mkdir(parents=True,exist_ok=True)
            requested=os.environ.get("HYPERTAGGING_MANIFEST","").strip()
            if requested:
                records=[json.loads(line) for line in Path(requested).read_text().splitlines() if line.strip()]
                MODE="REAL PRODUCTION MANIFEST"
            else:
                records=[
                    {"task_id":0,"input_file":"fixture-a.root","physics_category":"charged","entry_start":0,"entry_stop_exclusive":10,"planned_events":10,"output_file":"/tmp/hypertagging-manifest-fixture/missing-0.parquet"},
                    {"task_id":1,"input_file":"fixture-a.root","physics_category":"charged","entry_start":10,"entry_stop_exclusive":20,"planned_events":10,"output_file":"/tmp/hypertagging-manifest-fixture/missing-1.parquet"},
                ]
                spec=feature_spec_v4()
                for row in records:
                    row.update(manifest_schema_version=production.MANIFEST_SCHEMA_VERSION,campaign_id="fixture-campaign",campaign_config_digest="c"*64,source_git_commit="a"*40,source_git_tree="b"*40,source_state="clean",input_file_size=-1,input_file_mtime_ns=-1,input_file_identity="fixture",input_file_sha256=None,source_entries=20,entry_sequence=f"{row['entry_start']}:{row['entry_stop_exclusive']-1}",schema_version=SCHEMA_VERSION_V4,pid_vocabulary_version=PID_VOCABULARY_VERSION,feature_spec_hash=spec["feature_spec_hash"],model_feature_contract_hash=spec["model_feature_contract_hash"],leaf_kinematics_mode="raw_track_predicted_pid",track_fit_policy="max_p_value_then_pion_fallback-v1",charge_conjugate_normalization=False,event_buffer_size=128,row_group_size=128,campaign_stage="pilot",klm_training_scope="unresolved",production_readiness_report_sha256=None,git_commit="a"*40)
                    row["task_record_hash"]=production.task_record_hash(row)
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
            valid_task_hashes=all(row["task_record_hash"]==production.task_record_hash(row) for row in records)
            summary={"planned_events":int(frame.planned_events.sum()),"tasks":len(frame),"overlaps":overlaps,"schema_versions":sorted(frame.schema_version.unique()),"feature_hashes":sorted(frame.feature_spec_hash.unique()),"model_feature_contract_hashes":sorted(frame.model_feature_contract_hash.unique()),"pid_versions":sorted(frame.pid_vocabulary_version.unique()),"campaign_ids":sorted(frame.campaign_id.unique()),"campaign_config_digests":sorted(frame.campaign_config_digest.unique()),"source_git_commits":sorted(frame.source_git_commit.unique()),"valid_task_hashes":valid_task_hashes,"completed":int(frame.output_file.map(lambda x:Path(x).exists()).sum()),"missing":int((~frame.output_file.map(lambda x:Path(x).exists())).sum())}
            print(json.dumps(summary,indent=2)); assert not overlaps and valid_task_hashes and summary["schema_versions"]==[SCHEMA_VERSION_V4]
            frame.groupby("physics_category").planned_events.sum().plot.bar(title="Planned category distribution")
            plt.tight_layout(); plt.savefig(OUT/"manifest_categories.png"); plt.show()
            """
        ),
        md("## Completed/missing/invalid shards and global UID checks"),
        code(
            """
            status=frame[["task_id","output_file"]].copy(); status["classification"]=[production.classify_shard(Path(row.output_file),**production._validation_kwargs(row._asdict()))["classification"] for row in frame.itertuples(index=False)]
            display(status)
            report={
                "pass":not overlaps,
                "campaign_source_task_provenance_pass":valid_task_hashes and len(summary["campaign_ids"])==len(summary["source_git_commits"])==1,
                "completion_marker_hash_validation":"requires completed shards" if not (status.classification=="COMPLETE_VALID").any() else "verified by classify_shard",
                "global_uid_check":"requires completed shards" if not (status.classification=="COMPLETE_VALID").any() else "run validator",
                "global_uid_validation_passes":0,
                "publication_contract":"parquet + hashed metadata sidecar + parsed completion marker published last",
                "overwrite_invalidates_old_marker_first":True,
                "invalid_task_classifications":status.groupby("classification").size().to_dict(),
                "summary":summary,
            }
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
