#!/usr/bin/env python
"""Generate guarded real-parquet plus trained-checkpoint physics validation."""

from __future__ import annotations
import argparse
from pathlib import Path
import textwrap
import nbformat as nbf

DEFAULT_OUTPUT=Path(__file__).resolve().parents[1]/'notebooks'/'inspect_trained_physics_validation.ipynb'
def md(x):return nbf.v4.new_markdown_cell(textwrap.dedent(x).strip())
def code(x):return nbf.v4.new_code_cell(textwrap.dedent(x).strip())

def build_notebook():
    cells=[
      md("""# Trained physics validation

      ## tl;dr

      This notebook has **no fixture fallback**. It requires an explicit real
      schema-v4 parquet and a trained reconstruction checkpoint. Missing inputs
      are a hard, clear failure so fixture output cannot be mistaken for physics.
      """),
      md("## Context & Methods\n\nSet `HYPERTAGGING_REAL_PARQUET`, `HYPERTAGGING_DATASET_INDEX`, and `HYPERTAGGING_TRAINED_CHECKPOINT`. The reusable evaluation loader validates every checkpoint/data contract, restores all four normalizers, and proves a held-out validation/test selection. Evaluation is bounded by `HYPERTAGGING_VALIDATION_EVENTS` (default 16)."),
      code("""
      import json,os,subprocess,sys
      from pathlib import Path
      import matplotlib.pyplot as plt
      import pandas as pd
      import torch
      ROOT=Path.cwd();ROOT=ROOT if (ROOT/'src').exists() else Path('..').resolve();sys.path.insert(0,str(ROOT/'src'))
      parquet=os.environ.get('HYPERTAGGING_REAL_PARQUET','').strip();checkpoint=os.environ.get('HYPERTAGGING_TRAINED_CHECKPOINT','').strip();dataset_index=os.environ.get('HYPERTAGGING_DATASET_INDEX','').strip()
      if not parquet or not checkpoint or not dataset_index:
          out=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/trained_physics'));out.mkdir(parents=True,exist_ok=True)
          (out/'trained_physics_validation_summary.json').write_text(json.dumps({'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'schema_version':'direct-mdst-tree-v4','fixture_or_real':'real_only','data_path_or_fixture_name':parquet or 'HYPERTAGGING_REAL_PARQUET not set','checkpoint_path_or_none':checkpoint or 'none','seed':int(os.environ.get('HYPERTAGGING_NOTEBOOK_SEED','20260730')),'pass_fail_status':'NOT RUN'},indent=2))
          raise RuntimeError('REAL INPUT REQUIRED: set HYPERTAGGING_REAL_PARQUET, HYPERTAGGING_DATASET_INDEX, and HYPERTAGGING_TRAINED_CHECKPOINT; fixture fallback is intentionally disabled')
      PARQUET=Path(parquet);CHECKPOINT=Path(checkpoint);DATASET_INDEX=Path(dataset_index)
      if not PARQUET.exists() or not CHECKPOINT.exists() or not DATASET_INDEX.exists():raise FileNotFoundError(f'required real parquet/index/checkpoint missing: {PARQUET}, {DATASET_INDEX}, {CHECKPOINT}')
      from hypertagging.evaluation import load_trained_evaluation_context
      from hypertagging.evaluation.hierarchical_metrics import summarize_rollout,canonical_tree_signatures
      from hypertagging.preprocessing.pid_filter import TOKENIZE_DICT
      from hypertagging.reconstruction.level_rollout import RolloutConfig,bounded_beam_rollout,level_rollout
      limit=int(os.environ.get('HYPERTAGGING_VALIDATION_EVENTS','16'));test_limit=int(os.environ.get('HYPERTAGGING_TEST_EVENTS',str(limit)))
      validation_context=load_trained_evaluation_context(checkpoint=CHECKPOINT,data=PARQUET,dataset_index=DATASET_INDEX,split='validation',max_events=limit)
      context=load_trained_evaluation_context(checkpoint=CHECKPOINT,data=PARQUET,dataset_index=DATASET_INDEX,split='test',max_events=test_limit)
      payload=context.checkpoint;model=context.model;events=context.events;policy=context.constraint_policy
      if all('fixture' in (event.source_file or '').lower() for event in events):raise RuntimeError('fixture-like source provenance rejected by real-only physics notebook')
      OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/trained_physics'));OUT.mkdir(parents=True,exist_ok=True)
      B_TOKENS={TOKENIZE_DICT[pdg] for pdg in (511,-511,521,-521)}
      """),
      md("## Data\n\nThe report records actual source/schema, multiplicity, levels, PID/channel frequency, and partial-topology denominators."),
      code("""
      rows=[];mass_rows=[];pid_slice_rows=[];level_slice_rows=[];b_side_rows=[];channel_rows=[];b_candidates=[]
      for event_index,event in enumerate(events):
          truth=context.collated_event_batch(event_index);teacher=level_rollout(model,truth,mode='teacher_forced',config=RolloutConfig(max_level=8,constraint_policy=policy,rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode));free=level_rollout(model,truth,mode='predicted',config=RolloutConfig(max_level=8,constraint_policy=policy,rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode))
          teacher_metrics=summarize_rollout(teacher.batch,truth);free_metrics=summarize_rollout(free.batch,truth)
          rows.append({'event_uid':event.event_uid,'nodes':int(truth['node_mask'].sum()),'max_level':int(truth['level_ids'][truth['node_mask']].max()),'partial_topology':bool(truth['partial_missing_daughters'].any()),'complete_only_targets':int((truth['valid_reconstruction_target']&truth['recursive_reconstructable_complete']).sum()),'reconstructable_partial_targets':int(truth['valid_reconstruction_target'].sum()),'channel_id':event.b1_reconstructable_channel_id,'teacher_edge_purity':teacher_metrics['edge_precision'],'teacher_edge_efficiency':teacher_metrics['edge_recall'],'free_edge_purity':free_metrics['edge_precision'],'free_edge_efficiency':free_metrics['edge_recall'],'teacher_edge_f1':teacher_metrics['edge_f1'],'free_edge_f1':free_metrics['edge_f1'],'teacher_tree_exact':teacher_metrics['full_tree_exact_match'],'free_tree_exact':free_metrics['full_tree_exact_match'],'first_divergence_level':free_metrics['first_divergence_level']})
          # Local denominators: one row per truth/predicted mother or edge, never one event metric copied to every PID merely present.
          truth_sig=canonical_tree_signatures(truth);pred_sig=canonical_tree_signatures(free.batch)
          truth_edges=[(int(p),int(c)) for p,c in truth['daughter_adjacency'][0].nonzero().tolist()];pred_edges=[(int(p),int(c)) for p,c in free.batch['daughter_adjacency'][0].nonzero().tolist()]
          truth_edge_keys={(truth_sig[p],truth_sig[c]) for p,c in truth_edges};pred_edge_keys={(pred_sig[p],pred_sig[c]) for p,c in pred_edges}
          for p,c in truth_edges:
              child_pid=int(truth['pid_target_labels'][0,c]);pid_slice_rows.append({'slice_kind':'per_child_pid','pid_token':child_pid,'event_uid':event.event_uid,'tp':int((truth_sig[p],truth_sig[c]) in pred_edge_keys),'truth_denominator':1,'predicted_denominator':0})
          for p,c in pred_edges:
              child_pid=int(free.batch.get('current_pid_tokens',free.batch['pid_labels'])[0,c]);pid_slice_rows.append({'slice_kind':'per_child_pid','pid_token':child_pid,'event_uid':event.event_uid,'tp':int((pred_sig[p],pred_sig[c]) in truth_edge_keys),'truth_denominator':0,'predicted_denominator':1})
          pred_mothers=free.batch['daughter_adjacency'][0].any(-1).nonzero().flatten().tolist()
          for mother in truth['daughter_adjacency'][0].any(-1).nonzero().flatten().tolist():
              token=int(truth['pid_target_labels'][0,mother]);level=int(truth['level_ids'][0,mother]);truth_children={truth_sig[c] for c in truth['daughter_adjacency'][0,mother].nonzero().flatten().tolist()}
              candidates=[]
              for predicted_mother in pred_mothers:
                  predicted_children={pred_sig[c] for c in free.batch['daughter_adjacency'][0,predicted_mother].nonzero().flatten().tolist()};union=truth_children|predicted_children;candidates.append((len(truth_children&predicted_children)/max(len(union),1),predicted_mother,predicted_children))
              _,matched_mother,predicted_children=max(candidates,default=(0.0,None,set()),key=lambda row:row[0]);pointer_tp=len(truth_children&predicted_children);type_correct=int(matched_mother is not None and int(free.batch.get('current_pid_tokens',free.batch['pid_labels'])[0,matched_mother])==token);node_correct=int(type_correct and truth_children==predicted_children)
              local={'target_level':level,'mother_pid_token':token,'event_uid':event.event_uid,'node_correct':node_correct,'type_correct':type_correct,'pointer_true_positive':pointer_tp,'pointer_truth_denominator':len(truth_children),'pointer_predicted_denominator':len(predicted_children),'mother_denominator':1};level_slice_rows.append(local)
              side=int(truth['b_side'][0,mother]);b_side_rows.append({'b_side':side,**local})
          channel_rows.append({'event_uid':event.event_uid,'b1_channel_id':event.b1_reconstructable_channel_id,'b2_channel_id':event.b2_reconstructable_channel_id,'unordered_y4s_pair':tuple(sorted((event.b1_reconstructable_channel_id,event.b2_reconstructable_channel_id))),'free_tree_exact':free_metrics['full_tree_exact_match']})
          for label,state in [('truth',truth),('teacher',teacher.batch),('free',free.batch)]:
              composite=state['node_mask'][0]&state['daughter_adjacency'][0].any(-1);p4=state['p4'][0,composite];mass=(p4[:,3].square()-p4[:,:3].square().sum(-1)).clamp_min(0).sqrt()
              for value in mass.tolist():mass_rows.append({'event_uid':event.event_uid,'mode':label,'mass_GeV':value})
              types=state.get('current_pid_tokens',state['pid_labels'])[0,composite]
              for vector,token in zip(p4.tolist(),types.tolist()):
                  if int(token) in B_TOKENS:b_candidates.append({'event_uid':event.event_uid,'mode':label,'pid_token':int(token),'px':vector[0],'py':vector[1],'pz':vector[2],'energy':vector[3]})
      frame=pd.DataFrame(rows);masses=pd.DataFrame(mass_rows);pid_slices=pd.DataFrame(pid_slice_rows);level_slices=pd.DataFrame(level_slice_rows);b_sides=pd.DataFrame(b_side_rows);channels=pd.DataFrame(channel_rows)
      display(frame.describe(include='all'));display(frame.groupby(['partial_topology']).agg(events=('event_uid','count'),free_edge_purity=('free_edge_purity','mean'),free_edge_efficiency=('free_edge_efficiency','mean'),free_tree_exact=('free_tree_exact','mean')))
      display(frame.assign(multiplicity_bin=pd.cut(frame.nodes,[0,8,16,32,10**9])).groupby('multiplicity_bin',observed=True).agg(events=('event_uid','count'),free_edge_purity=('free_edge_purity','mean'),free_edge_efficiency=('free_edge_efficiency','mean')))
      display(level_slices.groupby(['target_level','mother_pid_token']).agg(node_correct=('node_correct','sum'),type_correct=('type_correct','sum'),pointer_true_positive=('pointer_true_positive','sum'),pointer_truth_denominator=('pointer_truth_denominator','sum'),pointer_predicted_denominator=('pointer_predicted_denominator','sum'),mother_denominator=('mother_denominator','sum')))
      display(pid_slices.groupby(['slice_kind','pid_token']).agg(true_positive=('tp','sum'),truth_edges=('truth_denominator','sum'),predicted_edges=('predicted_denominator','sum')))
      display(b_sides.groupby(['b_side','mother_pid_token']).agg(node_correct=('node_correct','sum'),type_correct=('type_correct','sum'),pointer_true_positive=('pointer_true_positive','sum'),pointer_truth_denominator=('pointer_truth_denominator','sum'),pointer_predicted_denominator=('pointer_predicted_denominator','sum'),mother_denominator=('mother_denominator','sum')));display(channels)
      """),
      md("## Results\n\nEdge/tree efficiency and purity are sliced by multiplicity, level, PID/channel availability, frequency, and partial topology. Mass, B-level Mbc/DeltaE, missing mass, and rare/unseen-channel fields are reported only where their required beam/channel metadata exist."),
      code("""
      # Validation-only threshold/calibration sweep. Final test evaluation must use
      # the selected validation threshold without re-selection.
      threshold_rows=[];mode_rows=[];search_rows=[]
      sweep_events=min(len(validation_context.events),4)
      for object_threshold in (0.3,0.5,0.7):
          for pointer_threshold in (0.3,0.5,0.7):
              for confidence_threshold in (0.0,0.3,0.6):
                  for type_probability_threshold in (None,0.3,0.5):
                      values=[]
                      for event_index in range(sweep_events):
                          truth=validation_context.collated_event_batch(event_index);result=level_rollout(validation_context.model,truth,mode='predicted',config=RolloutConfig(max_level=8,constraint_policy=validation_context.constraint_policy,object_threshold=object_threshold,pointer_threshold=pointer_threshold,confidence_threshold=confidence_threshold,type_probability_threshold=type_probability_threshold,rollout_pid_kinematics_mode=validation_context.rollout_pid_kinematics_mode));values.append(summarize_rollout(result.batch,truth)['edge_f1'])
                      threshold_rows.append({'object_threshold':object_threshold,'pointer_threshold':pointer_threshold,'confidence_threshold':confidence_threshold,'type_probability_threshold':type_probability_threshold,'validation_edge_f1':sum(values)/max(len(values),1)})
      selected_threshold=max(threshold_rows,key=lambda row:row['validation_edge_f1']) if threshold_rows else None
      final_test_selected_threshold=[]
      for event_index,event in enumerate(events):
          truth=context.collated_event_batch(event_index);result=level_rollout(model,truth,mode='predicted',config=RolloutConfig(max_level=8,constraint_policy=policy,object_threshold=selected_threshold['object_threshold'],pointer_threshold=selected_threshold['pointer_threshold'],confidence_threshold=selected_threshold['confidence_threshold'],type_probability_threshold=selected_threshold['type_probability_threshold'],rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode));final_test_selected_threshold.append({'event_uid':event.event_uid,**summarize_rollout(result.batch,truth)})
      for pid_mode in ('soft_decision_hard_construction','hard','temperature_softmax','straight_through_hard'):
          truth=context.collated_event_batch(0);result=level_rollout(model,truth,mode='predicted',config=RolloutConfig(max_level=8,constraint_policy=policy,rollout_pid_kinematics_mode=pid_mode));mode_rows.append({'pid_kinematics_mode':pid_mode,**summarize_rollout(result.batch,truth)})
      for event_index in range(min(len(events),4)):
          truth=context.collated_event_batch(event_index)
          for resolver in ('greedy','weighted_set_packing'):
              result=level_rollout(model,truth,mode='predicted',config=RolloutConfig(max_level=8,constraint_policy=policy,exclusive_resolution=resolver,max_resolution_proposals=12,rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode));search_rows.append({'event_uid':events[event_index].event_uid,'resolver':resolver,**summarize_rollout(result.batch,truth)})
          beam=bounded_beam_rollout(model,truth,config=RolloutConfig(max_level=2,constraint_policy=policy,max_resolution_proposals=12,rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode),beam_width=3,lookahead_levels=2)
          if beam:search_rows.append({'event_uid':events[event_index].event_uid,'resolver':'bounded_beam_best','beam_width':3,'lookahead_levels':2,**summarize_rollout(beam[0].batch,truth)})
      display(pd.DataFrame(threshold_rows));display(pd.DataFrame(mode_rows));display(pd.DataFrame(search_rows))
      """),
      code("""
      fig,axes=plt.subplots(1,2,figsize=(10,4));masses.groupby('mode').mass_GeV.plot.hist(alpha=.4,bins=30,ax=axes[0],legend=True);frame.plot.scatter(x='nodes',y='free_edge_f1',c='max_level',ax=axes[1],title='Free rollout by multiplicity/depth');fig.tight_layout();fig.savefig(OUT/'trained_physics_validation.png');plt.show()
      # Beam-energy metadata is not assumed. Never invent Mbc/DeltaE or missing mass.
      feature_contract=payload.get('feature_contract',{});beam_energy=feature_contract.get('beam_energy_GeV');frame_contract=feature_contract.get('four_vector_frame');b_metrics={'available':False,'reason':'validated CMS beam energy/frame absent from checkpoint contract'}
      if beam_energy is not None and frame_contract=='cms':
          b_frame=pd.DataFrame(b_candidates);ebeam=float(beam_energy)
          if not b_frame.empty:
              b_frame['mbc_GeV']=(ebeam**2-b_frame[['px','py','pz']].pow(2).sum(axis=1)).clip(lower=0).pow(.5);b_frame['delta_e_GeV']=b_frame.energy-ebeam
              b_metrics={'available':True,'beam_energy_GeV':ebeam,'four_vector_frame':'cms','entries':b_frame.to_dict('records')}
      training_channel_counts=feature_contract.get('training_channel_counts');rare_unseen={'available':False,'reason':'training-channel frequency map absent from checkpoint'}
      if training_channel_counts:
          eval_counts=frame.channel_id.value_counts().to_dict();rare_unseen={'available':True,'evaluation_channel_counts':{str(k):int(v) for k,v in eval_counts.items()},'training_channel_counts':training_channel_counts,'unseen_evaluation_channels':[int(channel) for channel in eval_counts if str(channel) not in training_channel_counts and channel not in training_channel_counts]}
      summary={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'schema_version':'direct-mdst-tree-v4','fixture_or_real':'real_only','data_path_or_fixture_name':str(PARQUET),'checkpoint_path_or_none':str(CHECKPOINT),'seed':int(os.environ.get('HYPERTAGGING_NOTEBOOK_SEED','20260730')),'pass_fail_status':'PASS',**context.report_metadata,'validation_selection':validation_context.report_metadata,'real_input':str(PARQUET),'dataset_index':str(DATASET_INDEX),'trained_checkpoint':str(CHECKPOINT),'events':len(events),'edge_tree_metrics':rows,'per_child_pid_edge_metrics':pid_slice_rows,'per_mother_pid_and_target_level_metrics':level_slice_rows,'per_b_side_metrics':b_side_rows,'both_b_channel_ids_and_unordered_y4s_pair':channel_rows,'complete_only_and_reconstructable_partial_denominators':frame[['event_uid','complete_only_targets','reconstructable_partial_targets']].to_dict('records'),'mass_entries':len(mass_rows),'b_level_mbc_deltae':b_metrics,'missing_mass':{'available':False,'reason':'no channel-specific missing-particle and initial-state contract supplied'},'rare_unseen_channel':rare_unseen,'teacher_forced_vs_free':{'teacher_edge_purity':float(frame.teacher_edge_purity.mean()),'teacher_edge_efficiency':float(frame.teacher_edge_efficiency.mean()),'free_edge_purity':float(frame.free_edge_purity.mean()),'free_edge_efficiency':float(frame.free_edge_efficiency.mean())},'bounded_set_packing':search_rows,'pid_kinematics_modes':mode_rows,'validation_threshold_sweep':threshold_rows,'selected_validation_threshold':selected_threshold,'final_held_out_test_at_selected_threshold':final_test_selected_threshold,'final_test_threshold_selection_performed':False,'calibration':{'validation_reliability_surface':threshold_rows,'selection_metric':'edge_f1','selection_split':'validation','application_split':'test'},'physics_claim_ready':False}
      (OUT/'trained_physics_validation_summary.json').write_text(json.dumps(summary,indent=2))
      """),
      md("## Takeaways\n\nOnly the executed real/checkpoint artifact may support physics conclusions. Missing beam, missing-particle, or training-frequency contracts remain explicitly unavailable rather than inferred."),
    ]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};return nb
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);nbf.write(build_notebook(),a.output);print(a.output);return 0
if __name__=='__main__':raise SystemExit(main())
