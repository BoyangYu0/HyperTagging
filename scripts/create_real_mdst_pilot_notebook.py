#!/usr/bin/env python
"""Generate the operator-run, real schema-v4 mDST pilot inspection notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_real_mdst_pilot.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook():
    cells = [
        md(
            """
            # Real mDST pilot inspection

            This notebook is deliberately real-data-only. It does not generate a
            fixture or fabricate PIDLikelihood/provenance results. Produce a
            schema-v4 pilot with fewer than 100 events, set
            `HYPERTAGGING_REAL_PILOT`, and execute top-to-bottom.

            ```bash
            source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
            basf2 scripts/preprocess_mdst.py -- \\
              --input /path/to/generic_mdst.root \\
              --output /data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \\
              --schema-version direct-mdst-tree-v4 --entry-sequence 0:49 \\
              --max-events 50 --event-buffer-size 32 --row-group-size 16
            HYPERTAGGING_REAL_PILOT=/data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \\
              /data/dust/user/boyangyu/uv_env/bin/python -m jupyter nbconvert \\
              --execute --to notebook --inplace notebooks/inspect_real_mdst_pilot.ipynb
            ```
            """
        ),
        md("## Load a published real schema-v4 pilot"),
        code(
            """
            from collections import Counter
            import json, os
            from pathlib import Path
            import numpy as np
            import pandas as pd
            from hypertagging.preprocessing.schema_v4 import load_payload_v4
            from hypertagging.preprocessing.pid_filter import PDG_TOKENS

            requested=os.environ.get("HYPERTAGGING_REAL_PILOT","").strip()
            if not requested:
                raise RuntimeError("Set HYPERTAGGING_REAL_PILOT to a real sub-100-event schema-v4 parquet; fixture substitution is forbidden")
            path=Path(requested)
            if not path.exists() or not Path(str(path)+".complete").exists():
                raise FileNotFoundError("pilot parquet and completion marker are required")
            payload=load_payload_v4(path)
            events=payload["events"]
            if not 0 < len(events) < 100:
                raise ValueError(f"pilot must contain 1..99 events, got {len(events)}")
            if payload["schema_version"] != "direct-mdst-tree-v4":
                raise ValueError("native schema-v4 pilot required")
            print({"real_pilot":str(path),"events":len(events),"schema":payload["schema_version"]})
            """
        ),
        md("## Actual leaf provenance, fit selection, charge, and PIDLikelihood"),
        code(
            """
            nodes=[node for event in events for node in event["nodes"]]
            leaves=[node for node in nodes if not node.get("daughter_ids")]
            tracks=[node for node in leaves if node.get("node_kind")=="track"]
            provenance=Counter(node.get("leaf_kinematics_mode","missing") for node in leaves)
            energy_sources=Counter(node.get("energy_source","missing") for node in tracks)
            pid_names=[f"pid_log_likelihood_{name}" for name in ("electron","muon","pion","kaon","proton")]
            pid_available={name:sum(bool(node.get("track_availability",{}).get(name,False)) for node in tracks) for name in pid_names}
            charge_values=Counter(float(node.get("charge",0.0)) for node in tracks)
            ecl=[node for node in leaves if node.get("node_kind")=="ecl_cluster"]
            ecl_energy_sources=Counter(node.get("energy_source","missing") for node in ecl)
            detector_inputs=[node for node in leaves if node.get("active",False)]
            truth_detector_inputs=[node for node in detector_inputs if node.get("leaf_kinematics_mode")=="truth_topology_only" or str(node.get("energy_source","")).startswith("truth_")]
            display(pd.DataFrame({"leaf_mode":provenance.keys(),"count":provenance.values()}))
            print({"track_energy_sources":dict(energy_sources),"ecl_energy_sources":dict(ecl_energy_sources),"PIDLikelihood_available":pid_available,"reconstructed_charge":dict(charge_values),"truth_derived_detector_inputs":len(truth_detector_inputs)})
            if any(node.get("input_pid_token",0) != 0 for node in tracks if node.get("leaf_kinematics_mode")=="raw_track_predicted_pid"):
                raise AssertionError("raw track entered with non-unknown input PID")
            if truth_detector_inputs:raise AssertionError("truth-derived leaf kinematics entered the active detector input")
            """
        ),
        md("## Recursive p4 closure and B-root discovery"),
        code(
            """
            closure=[]; b_root=[]; failures=[]
            for event in events:
                by_id={int(node["node_id"]):node for node in event["nodes"]}
                event_max=0.0
                for node in event["nodes"]:
                    daughters=[by_id[int(value)] for value in node.get("daughter_ids",[])]
                    if not daughters: continue
                    expected=np.sum([[d["px"],d["py"],d["pz"],d["energy"]] for d in daughters],axis=0)
                    actual=np.array([node["px"],node["py"],node["pz"],node["energy"]])
                    residual=float(np.max(np.abs(actual-expected))); event_max=max(event_max,residual)
                    if residual>1e-6: failures.append({"event_uid":event["event_uid"],"node_id":node["node_id"],"p4_residual":residual})
                closure.append(event_max)
                b_root.append({"event_uid":event["event_uid"],"valid":event.get("b_root_discovery_valid",False),"fallback":event.get("b_root_discovery_fallback",False),"b1":event.get("b1_root_id"),"b2":event.get("b2_root_id")})
            display(pd.DataFrame(b_root)); print({"maximum_p4_residual":max(closure,default=0.0),"closure_failures":len(failures)})
            assert not failures
            """
        ),
        md("## PID and level distributions plus bounded failure examples"),
        code(
            """
            pid_counts=Counter(int(node.get("pid_target_token",node.get("token",0))) for node in nodes)
            level_counts=Counter(int(node.get("level",-1)) for node in nodes)
            kl_nodes=[node for node in leaves if abs(int(node.get("pdg",0)))==130 or int(node.get("pid_target_token",node.get("token",-1)))==PDG_TOKENS.index(130)]
            klm_fields=Counter("present" if any("klm" in str(key).lower() for key in node) else "absent" for node in kl_nodes)
            failure_examples=(failures+[row for row in b_root if not row["valid"]])[:20]
            display(pd.DataFrame({"pid_token":pid_counts.keys(),"count":pid_counts.values()}))
            display(pd.DataFrame({"level":level_counts.keys(),"count":level_counts.values()}))
            display(pd.DataFrame(failure_examples))
            report={"real_data":True,"events":len(events),"leaf_provenance":dict(provenance),"pidlikelihood_available":pid_available,"track_energy_sources":dict(energy_sources),"ecl_energy_sources":dict(ecl_energy_sources),"charge_distribution":dict(charge_values),"truth_derived_detector_inputs":len(truth_detector_inputs),"maximum_p4_residual":max(closure,default=0.0),"valid_b_root_events":sum(row["valid"] for row in b_root),"fallback_b_root_events":sum(row["fallback"] for row in b_root),"pid_distribution":dict(pid_counts),"level_distribution":dict(level_counts),"k_l_leaf_count":len(kl_nodes),"k_l_klm_provenance_fields":dict(klm_fields),"klm_collection_contract":"report only; current direct collector declares Tracks and ECLClusters, not KLMClusters","failure_examples":failure_examples}
            report_path=Path(os.environ.get("HYPERTAGGING_REAL_PILOT_REPORT","/tmp/hypertagging-real-pilot-report.json"))
            report_path.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
            print(report_path)
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
    return notebook


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(build_notebook(),args.output); print(args.output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
