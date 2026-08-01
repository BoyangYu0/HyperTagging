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
            `HYPERTAGGING_REAL_PILOT`, and execute top-to-bottom. The variable
            accepts one path or a comma-separated bounded category map such as
            `charged=/path/a.parquet,mixed=/path/b.parquet`; the combined event
            count must remain below 100.

            ```bash
            source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
            python3 scripts/preprocess_mdst.py \\
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
            import json, os, subprocess
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import hypertagging
            from hypertagging.preprocessing.schema_v4 import load_payload_v4
            from hypertagging.preprocessing.pid_filter import PDG_TOKENS

            REPO_ROOT=Path(hypertagging.__file__).resolve().parents[2]
            os.chdir(REPO_ROOT)

            requested=os.environ.get("HYPERTAGGING_REAL_PILOT","").strip()
            if not requested:
                report_path=Path(os.environ.get("HYPERTAGGING_REAL_PILOT_REPORT","/tmp/hypertagging-real-pilot-report.json"))
                report_path.write_text(json.dumps({"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"schema_version":"direct-mdst-tree-v4","fixture_or_real":"real_only","data_path_or_fixture_name":"HYPERTAGGING_REAL_PILOT not set","checkpoint_path_or_none":"none","seed":int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED","20260730")),"pass_fail_status":"NOT RUN"},indent=2),encoding="utf-8")
                raise RuntimeError("Set HYPERTAGGING_REAL_PILOT to a real sub-100-event schema-v4 parquet; fixture substitution is forbidden")
            specifications=[]
            for value in requested.split(','):
                label,raw_path=(value.split('=',1) if '=' in value else ('unspecified',value))
                specifications.append((label.strip(),Path(raw_path.strip())))
            events=[];paths=[];input_categories=Counter();track_fit_policies={}
            for label,path in specifications:
                if not path.exists() or not Path(str(path)+".complete").exists():
                    raise FileNotFoundError(f"pilot parquet and completion marker are required: {path}")
                payload=load_payload_v4(path)
                if payload["schema_version"] != "direct-mdst-tree-v4":
                    raise ValueError("native schema-v4 pilot required")
                metadata_path=Path(str(path)+'.metadata.json')
                metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                track_fit_policies[str(path)]=metadata.get('track_fit_policy','NOT_RECORDED')
                for original in payload['events']:
                    event=dict(original);event['_pilot_input_category']=label;events.append(event)
                    input_categories[str(event.get('source_category') or label or 'unknown')]+=1
                paths.append(path)
            if not 0 < len(events) < 100:
                raise ValueError(f"combined pilot must contain 1..99 events, got {len(events)}")
            path=paths[0]
            print({"real_pilots":[str(value) for value in paths],"events":len(events),"schema":"direct-mdst-tree-v4","categories":dict(input_categories),"track_fit_policies":track_fit_policies})
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
                b_root.append({"event_uid":event["event_uid"],"valid":event.get("b_root_discovery_valid",False),"fallback":event.get("b_root_discovery_fallback",False),"reason":event.get("b_root_missing_reason","missing"),"strict_count":event.get("strict_b_root_count",0),"fallback_count":event.get("fallback_b_root_count",0),"valid_b_side_labels":event.get("valid_b_side_label_count",0),"active_channel_loss_branches":event.get("active_channel_loss_branch_count",0),"b1":event.get("b1_root_id"),"b2":event.get("b2_root_id")})
            display(pd.DataFrame(b_root)); print({"maximum_p4_residual":max(closure,default=0.0),"closure_failures":len(failures)})
            assert not failures
            """
        ),
        md("## Retained-tree plots and structural invariants"),
        code(
            """
            import matplotlib.pyplot as plt
            cycle_failures=[];missing_links=[];level_failures=[];closure_by_level_multiplicity=[]
            def event_summary(event):
                nodes=event['nodes']; leaves=[node for node in nodes if not node.get('daughter_ids')]
                return {
                    'depth':max((int(node.get('level',0)) for node in nodes),default=0),
                    'fsp':len(leaves),
                    'neutral':sum(float(node.get('charge',0.0))==0.0 for node in leaves),
                    'copied':sum(bool(node.get('copied',False)) for node in nodes),
                    'unmatched':sum('unmatched_reco' in set(node.get('flags',[])) for node in nodes),
                    'klm':sum(node.get('node_kind')=='klm_cluster' for node in leaves),
                    'ecl_klm':sum(bool(node.get('associated_reco_id')) for node in leaves),
                    'contracted':sum(bool(node.get('contracted_intermediate',False)) for node in nodes),
                    'strict_b':bool(event.get('b_root_discovery_valid',False) and not event.get('b_root_discovery_fallback',False)),
                    'missing_b':not bool(event.get('b_root_discovery_valid',False)),
                }
            summaries=[event_summary(event) for event in events]
            representative={}
            for key in ('depth','fsp','neutral'):
                representative[f'maximum_{key}']=int(np.argmax([row[key] for row in summaries]))
            for key in ('copied','unmatched','klm','ecl_klm','contracted','strict_b','missing_b'):
                representative[key]=next((index for index,row in enumerate(summaries) if row[key]),None)
            selected_indices=[]
            for index in representative.values():
                if index is not None and index not in selected_indices:selected_indices.append(index)
            selected_tree_events=[events[index] for index in selected_indices[:10]]
            print({'representative_event_indices':representative,'selected_event_uids':[event['event_uid'] for event in selected_tree_events]})
            fig,axes=plt.subplots(len(selected_tree_events),1,figsize=(12,max(3,3*len(selected_tree_events))),squeeze=False)
            for axis,event in zip(axes[:,0],selected_tree_events):
                by_id={int(node['node_id']):node for node in event['nodes']};colors=[]
                for node in event['nodes']:
                    node_id=int(node['node_id']);level=int(node.get('level',-1));axis.scatter(node_id,level);axis.text(node_id,level,str(node.get('pdg',node.get('pid_target_token',0))),fontsize=7)
                    daughters=[int(value) for value in node.get('daughter_ids',[])]
                    for daughter in daughters:
                        if daughter not in by_id:missing_links.append({'event_uid':event['event_uid'],'mother':node_id,'missing_daughter':daughter});continue
                        axis.plot([node_id,daughter],[level,int(by_id[daughter].get('level',-1))],color='gray',alpha=.5)
                        if int(by_id[daughter].get('level',-1))>=level:level_failures.append({'event_uid':event['event_uid'],'mother':node_id,'daughter':daughter})
                    expected_level=0 if not daughters else 1+max(int(by_id[d]['level']) for d in daughters if d in by_id)
                    if level!=expected_level:level_failures.append({'event_uid':event['event_uid'],'node':node_id,'stored':level,'expected':expected_level})
                parent={int(node['node_id']):node.get('parent_id') for node in event['nodes']}
                for start in parent:
                    seen=set();current=start
                    while current is not None and int(current)>=0:
                        current=int(current)
                        if current in seen:cycle_failures.append({'event_uid':event['event_uid'],'start':start,'cycle_at':current});break
                        seen.add(current);current=parent.get(current)
                axis.set(title=event['event_uid'],xlabel='node id',ylabel='retained level')
            fig.tight_layout();tree_plot_path=Path(os.environ.get('HYPERTAGGING_REAL_PILOT_TREE_PLOT','/tmp/hypertagging-real-pilot-trees.png'));fig.savefig(tree_plot_path);plt.show()
            if cycle_failures or missing_links or level_failures:raise AssertionError({'cycles':cycle_failures,'missing_links':missing_links,'level_invariants':level_failures})
            """
        ),
        md("## Detector availability, topology categories, and local denominators"),
        code(
            """
            track_availability=Counter();ecl_availability=Counter();klm_availability=Counter();pid_by_fit_hypothesis=Counter();fit_choices=Counter();fit_policy_comparisons=[];two_body_fit_mass_differences=[];unmatched_examples=[];truth_only_examples=[];contracted=Counter();denominators=Counter();p4_by_level_mult=[];ecl_klm_associations=[]
            for event in events:
                by_id={int(node['node_id']):node for node in event['nodes']}
                for node in event['nodes']:
                    if node.get('node_kind')=='track':
                        for name,available in node.get('track_availability',{}).items():track_availability[(name,bool(available))]+=1
                        fit=str(node.get('track_fit_selection_method','missing'));fit_hypothesis=str(node.get('track_fit_hypothesis','missing'));fit_choices[(fit,fit_hypothesis,bool(node.get('track_fit_available',False)),str(node.get('track_fit_fallback_reason')))] += 1
                        for hypothesis in ('electron','muon','pion','kaon','proton'):
                            available=bool(node.get('track_availability',{}).get(f'pid_log_likelihood_{hypothesis}',False));status=node.get('pid_likelihood_status',{}).get(hypothesis,'missing');pid_by_fit_hypothesis[(fit,hypothesis,available,status)]+=1
                        if node.get('track_fit_policy_diagnostics'):fit_policy_comparisons.append({'event_uid':event['event_uid'],'node_id':node['node_id'],**node['track_fit_policy_diagnostics']})
                    if node.get('node_kind')=='ecl_cluster':
                        for name,available in node.get('cluster_availability',{}).items():ecl_availability[(name,bool(available))]+=1
                    if node.get('node_kind')=='klm_cluster':
                        for name,available in node.get('klm_availability',{}).items():klm_availability[(name,bool(available))]+=1
                        if node.get('associated_reco_id'):
                            associated=next((candidate for candidate in event['nodes'] if candidate.get('reco_object_id')==node.get('associated_reco_id') or candidate.get('reco_id')==node.get('associated_reco_id')),None)
                            shared=set(node.get('recursive_leaf_source_ids',[])) & set((associated or {}).get('recursive_leaf_source_ids',[]))
                            ecl_klm_associations.append({'event_uid':event['event_uid'],'klm_node':node['node_id'],'associated_reco_id':node.get('associated_reco_id'),'ecl_node':None if associated is None else associated['node_id'],'shared_source_conflict':bool(shared)})
                    flags=set(node.get('flags',[]))
                    if 'unmatched_reco' in flags or node.get('leaf_kinematics_mode')=='unmatched_reco':unmatched_examples.append({'event_uid':event['event_uid'],'node_id':node['node_id'],'pdg':node.get('pdg')})
                    if 'truth_topology_only' in flags or node.get('leaf_kinematics_mode')=='truth_topology_only':truth_only_examples.append({'event_uid':event['event_uid'],'node_id':node['node_id'],'pdg':node.get('pdg')})
                    contracted[bool(node.get('contracted_intermediate',False) or 'contracted_intermediate_path' in flags)]+=1
                    daughters=[by_id[int(value)] for value in node.get('daughter_ids',[]) if int(value) in by_id]
                    if daughters:
                        denominators['reconstructable_partial']+=int(bool(node.get('valid_reconstruction_target',False)))
                        denominators['complete_only']+=int(bool(node.get('valid_reconstruction_target',False) and node.get('recursive_reconstructable_complete',False)))
                        expected=np.sum([[d['px'],d['py'],d['pz'],d['energy']] for d in daughters],axis=0);actual=np.array([node['px'],node['py'],node['pz'],node['energy']]);p4_by_level_mult.append({'level':int(node['level']),'daughter_multiplicity':len(daughters),'max_abs_residual':float(np.max(np.abs(actual-expected)))})
                comparable=[node for node in event['nodes'] if node.get('node_kind')=='track' and node.get('track_fit_policy_diagnostics',{}).get('pion_comparison_available')]
                pion_mass=0.13957039
                for left_index,left in enumerate(comparable):
                    for right in comparable[left_index+1:]:
                        selected=[];pion=[]
                        for node in (left,right):
                            selected_p=np.array([node['px'],node['py'],node['pz']],dtype=float);diagnostic=node['track_fit_policy_diagnostics'];pion_p=np.array([diagnostic['pion_px'],diagnostic['pion_py'],diagnostic['pion_pz']],dtype=float)
                            selected.append(np.r_[selected_p,np.sqrt(np.dot(selected_p,selected_p)+pion_mass**2)]);pion.append(np.r_[pion_p,np.sqrt(np.dot(pion_p,pion_p)+pion_mass**2)])
                        selected_sum=np.sum(selected,axis=0);pion_sum=np.sum(pion,axis=0);selected_mass=np.sqrt(max(selected_sum[3]**2-np.dot(selected_sum[:3],selected_sum[:3]),0.0));pion_fit_mass=np.sqrt(max(pion_sum[3]**2-np.dot(pion_sum[:3],pion_sum[:3]),0.0));two_body_fit_mass_differences.append({'event_uid':event['event_uid'],'left_node_id':left['node_id'],'right_node_id':right['node_id'],'selected_mass':float(selected_mass),'pion_closest_mass':float(pion_fit_mass),'delta_mass':float(selected_mass-pion_fit_mass)})
            display(pd.DataFrame([{'feature':k[0],'available':k[1],'count':v} for k,v in track_availability.items()]));display(pd.DataFrame([{'feature':k[0],'available':k[1],'count':v} for k,v in ecl_availability.items()]));display(pd.DataFrame([{'feature':k[0],'available':k[1],'count':v} for k,v in klm_availability.items()]));display(pd.DataFrame([{'fit':k[0],'hypothesis':k[1],'available':k[2],'status':k[3],'count':v} for k,v in pid_by_fit_hypothesis.items()]));display(pd.DataFrame(fit_policy_comparisons));display(pd.DataFrame(two_body_fit_mass_differences));display(pd.DataFrame(ecl_klm_associations));display(pd.DataFrame(p4_by_level_mult).groupby(['level','daughter_multiplicity']).agg(count=('max_abs_residual','size'),max_abs_residual=('max_abs_residual','max')));display(pd.DataFrame(unmatched_examples[:20]));display(pd.DataFrame(truth_only_examples[:20]));print({'fit_choices':{str(k):v for k,v in fit_choices.items()},'energy_sources':dict(energy_sources),'denominators':dict(denominators),'contracted_intermediate':dict(contracted),'strict_b_roots':sum(row['valid'] and not row['fallback'] for row in b_root),'fallback_b_roots':sum(row['fallback'] for row in b_root),'missing_root_reasons':dict(Counter(row['reason'] for row in b_root)),'active_channel_loss_branches':sum(row['active_channel_loss_branches'] for row in b_root)})
            """
        ),
        md("## PID and level distributions plus bounded failure examples"),
        code(
            """
            pid_counts=Counter(int(node.get("pid_target_token",node.get("token",0))) for node in nodes)
            level_counts=Counter(int(node.get("level",-1)) for node in nodes)
            kl_nodes=[node for node in leaves if abs(int(node.get("pdg",0)))==130 or int(node.get("pid_target_token",node.get("token",-1)))==PDG_TOKENS.index(130)]
            klm_nodes=[node for node in leaves if node.get("node_kind")=="klm_cluster"]
            klm_fields=Counter("present" if node.get("klm_features") else "absent" for node in kl_nodes)
            capacity_rows=[]
            for event in events:
                active=[node for node in event['nodes'] if node.get('active',False)]
                for level in sorted({int(node.get('level',0)) for node in active if int(node.get('level',0))>0}):
                    mothers=[node for node in active if int(node.get('level',0))==level and node.get('valid_reconstruction_target',False)]
                    capacity_rows.append({'event_uid':event['event_uid'],'source_category':event.get('source_category',''),'level':level,'target_mothers':len(mothers),'maximum_daughter_cardinality':max((len(node.get('daughter_ids',[])) for node in mothers),default=0),'event_multiplicity':len(active),'neutral_multiplicity':sum(float(node.get('charge',0.0))==0.0 for node in active if not node.get('daughter_ids')),'channel_pair':str(sorted((event.get('b1_reconstructable_channel_id',0),event.get('b2_reconstructable_channel_id',0))))})
            capacity_frame=pd.DataFrame(capacity_rows)
            display(capacity_frame.groupby('level').agg(maximum_mothers=('target_mothers','max'),p50_mothers=('target_mothers','median'),p95_mothers=('target_mothers',lambda values:values.quantile(.95)),maximum_daughter_cardinality=('maximum_daughter_cardinality','max')) if len(capacity_frame) else capacity_frame)
            failure_examples=(failures+[row for row in b_root if not row["valid"]])[:20]
            display(pd.DataFrame({"pid_token":pid_counts.keys(),"count":pid_counts.values()}))
            display(pd.DataFrame({"level":level_counts.keys(),"count":level_counts.values()}))
            display(pd.DataFrame(failure_examples))
            report={"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"schema_version":"direct-mdst-tree-v4","fixture_or_real":"real_only","data_path_or_fixture_name":[str(value) for value in paths],"checkpoint_path_or_none":"none","seed":int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED","20260730")),"pass_fail_status":"PASS","real_data":True,"events":len(events),"input_category_counts":dict(input_categories),"track_fit_policies":track_fit_policies,"leaf_provenance":dict(provenance),"track_feature_availability":{str(k):v for k,v in track_availability.items()},"ecl_feature_availability":{str(k):v for k,v in ecl_availability.items()},"klm_feature_availability":{str(k):v for k,v in klm_availability.items()},"ecl_klm_associations":ecl_klm_associations,"track_fit_policy_comparisons":fit_policy_comparisons,"track_fit_two_body_mass_differences":two_body_fit_mass_differences,"query_cardinality_capacity_rows":capacity_rows,"representative_event_indices":representative,"pidlikelihood_available":pid_available,"pidlikelihood_by_fit_and_hypothesis":{str(k):v for k,v in pid_by_fit_hypothesis.items()},"fit_choice_distribution":{str(k):v for k,v in fit_choices.items()},"track_energy_sources":dict(energy_sources),"ecl_energy_sources":dict(ecl_energy_sources),"charge_distribution":dict(charge_values),"truth_derived_detector_inputs":len(truth_detector_inputs),"cycles":cycle_failures,"missing_links":missing_links,"level_invariant_failures":level_failures,"maximum_p4_residual":max(closure,default=0.0),"p4_closure_by_level_and_daughter_multiplicity":p4_by_level_mult,"valid_b_root_events":sum(row["valid"] for row in b_root),"strict_b_root_events":sum(row["valid"] and not row["fallback"] for row in b_root),"fallback_b_root_events":sum(row["fallback"] for row in b_root),"missing_b_root_reason_histogram":dict(Counter(row["reason"] for row in b_root)),"valid_b_side_label_count":sum(row["valid_b_side_labels"] for row in b_root),"active_channel_loss_branch_count":sum(row["active_channel_loss_branches"] for row in b_root),"complete_only_and_reconstructable_partial_denominators":dict(denominators),"unmatched_reco_examples":unmatched_examples[:20],"truth_topology_only_examples":truth_only_examples[:20],"contracted_intermediate_frequency":dict(contracted),"pid_distribution":dict(pid_counts),"level_distribution":dict(level_counts),"k_l_leaf_count":len(kl_nodes),"klm_node_count":len(klm_nodes),"k_l_klm_provenance_fields":dict(klm_fields),"klm_collection_contract":"KLMClusters are explicit masked model inputs; associated ECL/KLM nodes share recursive source identity and conflict","selected_tree_plot":str(tree_plot_path),"failure_examples":failure_examples}
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
