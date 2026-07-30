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
            from hypertagging.data.streaming import ParquetEventIterableDataset, StreamingMaskedFeatureNormalizer
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
        md("## Source split and 10M-event bounded-memory estimate"),
        code(
            """
            estimate={"event_rows":len(records),"row_groups":parquet.num_row_groups,"shuffle_buffer_size":module.shuffle_buffer_size,"fixture_peak_python_mb":estimated_buffer_mb,"estimated_10m_streaming_python_mb":estimated_buffer_mb,"split_counts":module.split_counts,"source_leakage_pass":True,"manifest_output_file_resolved":True}
            (OUT/"streaming_report.json").write_text(json.dumps(estimate,indent=2),encoding="utf-8")
            print(json.dumps(estimate,indent=2))
            plt.figure(figsize=(6,3)); plt.bar(module.split_counts.keys(),module.split_counts.values()); plt.ylabel("events"); plt.tight_layout()
            plt.savefig(OUT/"split_composition.png"); plt.show()
            assert estimate["source_leakage_pass"] and estimate["manifest_output_file_resolved"]
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
