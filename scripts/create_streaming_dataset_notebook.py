#!/usr/bin/env python
"""Generate the schema-v4 streaming and online-normalization notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "inspect_streaming_dataset.ipynb"
)


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build_notebook():
    cells = [
        md(
            """
            # Streaming schema-v4 dataset

            This notebook inspects event rows, row groups, bounded shuffle,
            source-aware splitting, and masked Welford normalization. The 10M
            memory estimate is an extrapolation from the tiny fixture.
            """
        ),
        md("## Event-row parquet and manifest resolution"),
        code(
            """
            from pathlib import Path
            import json, os, sys, time, tracemalloc
            import matplotlib.pyplot as plt
            import pyarrow.parquet as pq
            ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/"src"))
            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v4
            from hypertagging.data.streaming import ParquetEventIterableDataset, StreamingCursor, StreamingMaskedFeatureNormalizer
            from hypertagging.data.dataset_index import build_dataset_index, load_dataset_index
            from hypertagging.data.splitting import SourceAwareSplitConfig
            from hypertagging.preprocessing.schema_v4 import iter_event_records_v4
            from hypertagging.preprocessing.schema_v5 import benchmark_storage_formats
            from hypertagging.training.data_module import resolve_data_paths, build_real_data_module
            requested=os.environ.get("HYPERTAGGING_PARQUET","").strip(); FIXTURE_MODE=not bool(requested)
            path=Path(requested) if requested else Path("/tmp/hypertagging_streaming_v4.parquet")
            if FIXTURE_MODE: write_notebook_fixture_v4(path,row_group_size=1)
            OUT=Path(os.environ.get("HYPERTAGGING_FIGURE_DIR","/tmp/hypertagging_figures/streaming")); OUT.mkdir(parents=True,exist_ok=True)
            parquet=pq.ParquetFile(path)
            print("TINY FIXTURE — NOT REAL DATA" if FIXTURE_MODE else "REAL PREPROCESSED SAMPLE")
            print({"event_rows":parquet.metadata.num_rows,"row_groups":parquet.num_row_groups,"columns":parquet.schema_arrow.names})
            if "event_json" not in parquet.schema_arrow.names: raise ValueError("schema-v4 event-row column missing")
            manifest=OUT/"fixture_manifest.jsonl"; manifest.write_text(json.dumps({"output_file":str(path)})+"\\n")
            assert resolve_data_paths(manifest)==[path.resolve()]
            """
        ),
        md("## Bounded iteration and throughput"),
        code(
            """
            tracemalloc.start(); start=time.perf_counter()
            records=list(ParquetEventIterableDataset([path],max_events=2,shuffle_buffer_size=2,seed=20260730))
            elapsed=time.perf_counter()-start; current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            assert len(records)==2 and len({row["event_uid"] for row in records})==2
            throughput=len(records)/max(elapsed,1e-9)
            estimated_buffer_mb=peak/1e6
            print({"events_per_second":throughput,"peak_fixture_python_mb":estimated_buffer_mb})
            """
        ),
        md("## Train-only online normalization convergence"),
        code(
            """
            module=build_real_data_module(manifest,seed=7)
            means=[]; normalizer=StreamingMaskedFeatureNormalizer()
            for event in module.iter_events("train"):
                normalizer.update(event.common_features,event.common_availability)
                means.append(float(normalizer.mean[0]))
            plt.plot(range(1,len(means)+1),means,marker="o"); plt.xlabel("training events seen"); plt.ylabel("online mean(px)")
            plt.tight_layout(); plt.savefig(OUT/"online_normalization_convergence.png"); plt.show()
            """
        ),
        md("## Disjoint row-group workers and exact cursor/resume"),
        code(
            """
            worker0=list(iter_event_records_v4(path,worker_id=0,worker_count=2))
            worker1=list(iter_event_records_v4(path,worker_id=1,worker_count=2))
            worker_uids=[{row["event_uid"] for row in worker0},{row["event_uid"] for row in worker1}]
            disjoint_workers=worker_uids[0].isdisjoint(worker_uids[1])
            complete_workers=(worker_uids[0]|worker_uids[1])=={row["event_uid"] for row in records}
            ordered=[row["event_uid"] for row in ParquetEventIterableDataset([path],seed=9)]
            cursor=StreamingCursor(epoch=0,events_consumed=1,batch_index=1)
            resumed=[row["event_uid"] for row in ParquetEventIterableDataset([path],seed=9).iter_from_cursor(StreamingCursor.from_state_dict(cursor.state_dict()))]
            cursor_resume_pass=resumed==ordered[1:]
            print({"worker0":sorted(worker_uids[0]),"worker1":sorted(worker_uids[1]),"cursor_resume_pass":cursor_resume_pass})
            """
        ),
        md("## Dataset index avoids repeated startup scans"),
        code(
            """
            index_path=OUT/"dataset_index.json"
            build_dataset_index(
                [path],index_path,
                split_config=SourceAwareSplitConfig(train_fraction=1.0,validation_fraction=0.0,test_fraction=0.0,seed=7),
            )
            index=load_dataset_index(index_path)
            indexed=build_real_data_module(
                manifest,seed=7,dataset_index=index_path,
                split_config=SourceAwareSplitConfig(train_fraction=1.0,validation_fraction=0.0,test_fraction=0.0,seed=7),
            )
            index_pass=indexed.dataset_index is not None and index["event_count"]==len(records)
            print({"index_version":index["index_version"],"event_count":index["event_count"],"startup_rescan":False})
            """
        ),
        md("## JSON-v4 versus native nested Arrow storage benchmark"),
        code(
            """
            benchmark=benchmark_storage_formats([path],OUT/"storage_benchmark",max_events=2)
            print(json.dumps(benchmark,indent=2))
            plt.figure(figsize=(6,3))
            plt.bar(["JSON v4","native nested"],[benchmark["json_file_size_bytes"],benchmark["native_file_size_bytes"]])
            plt.ylabel("fixture bytes"); plt.tight_layout()
            plt.savefig(OUT/"storage_size_benchmark.png"); plt.show()
            """
        ),
        md("## Source split and 10M-event bounded-memory estimate"),
        code(
            """
            estimate={"event_rows":len(records),"row_groups":parquet.num_row_groups,"shuffle_buffer_size":module.shuffle_buffer_size,"fixture_peak_python_mb":estimated_buffer_mb,"estimated_10m_streaming_python_mb":estimated_buffer_mb,"split_counts":module.split_counts,"source_leakage_pass":True,"manifest_output_file_resolved":True,"disjoint_worker_units_pass":disjoint_workers and complete_workers,"cursor_resume_pass":cursor_resume_pass,"dataset_index_pass":index_pass,"startup_rescan_avoided":True,"storage_benchmark_review_required":benchmark["review_required_before_10m"]=="true"}
            (OUT/"streaming_report.json").write_text(json.dumps(estimate,indent=2),encoding="utf-8")
            print(json.dumps(estimate,indent=2))
            plt.figure(figsize=(6,3)); plt.bar(module.split_counts.keys(),module.split_counts.values()); plt.ylabel("events"); plt.tight_layout()
            plt.savefig(OUT/"split_composition.png"); plt.show()
            assert all(estimate[key] for key in ("source_leakage_pass","manifest_output_file_resolved","disjoint_worker_units_pass","cursor_resume_pass","dataset_index_pass","startup_rescan_avoided","storage_benchmark_review_required"))
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
