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
      md("## Context & Methods\n\nSet `HYPERTAGGING_REAL_PARQUET` and `HYPERTAGGING_TRAINED_CHECKPOINT`. Evaluation is bounded by `HYPERTAGGING_VALIDATION_EVENTS` (default 16)."),
      code("""
      import json,os,sys
      from pathlib import Path
      import matplotlib.pyplot as plt
      import pandas as pd
      import torch
      ROOT=Path.cwd();ROOT=ROOT if (ROOT/'src').exists() else Path('..').resolve();sys.path.insert(0,str(ROOT/'src'))
      parquet=os.environ.get('HYPERTAGGING_REAL_PARQUET','').strip();checkpoint=os.environ.get('HYPERTAGGING_TRAINED_CHECKPOINT','').strip()
      if not parquet or not checkpoint:
          raise RuntimeError('REAL INPUT REQUIRED: set HYPERTAGGING_REAL_PARQUET and HYPERTAGGING_TRAINED_CHECKPOINT; fixture fallback is intentionally disabled')
      PARQUET=Path(parquet);CHECKPOINT=Path(checkpoint)
      if not PARQUET.exists() or not CHECKPOINT.exists():raise FileNotFoundError(f'required real parquet/checkpoint missing: {PARQUET}, {CHECKPOINT}')
      from hypertagging.data.heterogeneous import load_heterogeneous_events,collate_heterogeneous_events
      from hypertagging.evaluation.hierarchical_metrics import summarize_rollout
      from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
      from hypertagging.preprocessing.pid_filter import TOKENIZE_DICT
      from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
      from hypertagging.reconstruction.level_rollout import RolloutConfig,level_rollout
      from hypertagging.training.model_config import ModelArchitecture
      payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);architecture=ModelArchitecture.from_dict(payload['architecture'])
      model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=architecture.d_model,hyper_dim=architecture.hyper_dim,n_queries=architecture.n_queries,n_heads=architecture.n_heads,n_context_layers=architecture.n_context_layers,curvature=architecture.curvature,ffn_dim=architecture.ffn_dim,dropout=0.0,max_cardinality=architecture.max_cardinality,n_queries_by_level=architecture.n_queries_by_level,max_cardinality_by_level=architecture.max_cardinality_by_level,hyper_projection_init_scale=architecture.hyper_projection_init_scale,tangent_scale_mode=architecture.tangent_scale_mode)
      model.load_state_dict(payload['model_state_dict'],strict=True);model.eval();limit=int(os.environ.get('HYPERTAGGING_VALIDATION_EVENTS','16'));events=load_heterogeneous_events(PARQUET,limit=limit,max_nodes=512)
      if not events:raise RuntimeError('real parquet contained no usable events')
      if all('fixture' in (event.source_file or '').lower() for event in events):raise RuntimeError('fixture-like source provenance rejected by real-only physics notebook')
      OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/trained_physics'));OUT.mkdir(parents=True,exist_ok=True)
      B_TOKENS={TOKENIZE_DICT[pdg] for pdg in (511,-511,521,-521)}
      """),
      md("## Data\n\nThe report records actual source/schema, multiplicity, levels, PID/channel frequency, and partial-topology denominators."),
      code("""
      rows=[];mass_rows=[];pid_slice_rows=[];level_slice_rows=[];b_candidates=[];policy=ReconstructionConstraintPolicy.from_dict(payload.get('feature_contract',{}).get('reconstruction_constraint_policy',{})) if payload.get('feature_contract',{}).get('reconstruction_constraint_policy') else ReconstructionConstraintPolicy()
      for event in events:
          truth=collate_heterogeneous_events([event]);teacher=level_rollout(model,truth,mode='teacher_forced',config=RolloutConfig(max_level=8,constraint_policy=policy,rollout_pid_kinematics_mode='hard'));free=level_rollout(model,truth,mode='predicted',config=RolloutConfig(max_level=8,constraint_policy=policy,rollout_pid_kinematics_mode='hard'))
          teacher_metrics=summarize_rollout(teacher.batch,truth);free_metrics=summarize_rollout(free.batch,truth)
          rows.append({'event_uid':event.event_uid,'nodes':int(truth['node_mask'].sum()),'max_level':int(truth['level_ids'][truth['node_mask']].max()),'partial_topology':bool(truth['partial_missing_daughters'].any()),'complete_only_targets':int((truth['valid_reconstruction_target']&truth['recursive_reconstructable_complete']).sum()),'reconstructable_partial_targets':int(truth['valid_reconstruction_target'].sum()),'channel_id':event.b1_reconstructable_channel_id,'teacher_edge_purity':teacher_metrics['edge_precision'],'teacher_edge_efficiency':teacher_metrics['edge_recall'],'free_edge_purity':free_metrics['edge_precision'],'free_edge_efficiency':free_metrics['edge_recall'],'teacher_edge_f1':teacher_metrics['edge_f1'],'free_edge_f1':free_metrics['edge_f1'],'teacher_tree_exact':teacher_metrics['full_tree_exact_match'],'free_tree_exact':free_metrics['full_tree_exact_match'],'first_divergence_level':free_metrics['first_divergence_level']})
          present_pids=truth['pid_target_labels'][truth['node_mask']].unique().tolist()
          for token in present_pids:pid_slice_rows.append({'pid_token':int(token),'event_uid':event.event_uid,'free_edge_purity':free_metrics['edge_precision'],'free_edge_efficiency':free_metrics['edge_recall'],'free_tree_exact':free_metrics['full_tree_exact_match']})
          for level in truth['level_ids'][truth['node_mask']].unique().tolist():level_slice_rows.append({'retained_level_present':int(level),'event_uid':event.event_uid,'free_edge_purity':free_metrics['edge_precision'],'free_edge_efficiency':free_metrics['edge_recall'],'free_tree_exact':free_metrics['full_tree_exact_match']})
          for label,state in [('truth',truth),('teacher',teacher.batch),('free',free.batch)]:
              composite=state['node_mask'][0]&state['daughter_adjacency'][0].any(-1);p4=state['p4'][0,composite];mass=(p4[:,3].square()-p4[:,:3].square().sum(-1)).clamp_min(0).sqrt()
              for value in mass.tolist():mass_rows.append({'event_uid':event.event_uid,'mode':label,'mass_GeV':value})
              types=state.get('current_pid_tokens',state['pid_labels'])[0,composite]
              for vector,token in zip(p4.tolist(),types.tolist()):
                  if int(token) in B_TOKENS:b_candidates.append({'event_uid':event.event_uid,'mode':label,'pid_token':int(token),'px':vector[0],'py':vector[1],'pz':vector[2],'energy':vector[3]})
      frame=pd.DataFrame(rows);masses=pd.DataFrame(mass_rows);pid_slices=pd.DataFrame(pid_slice_rows);level_slices=pd.DataFrame(level_slice_rows)
      display(frame.describe(include='all'));display(frame.groupby(['partial_topology']).agg(events=('event_uid','count'),free_edge_purity=('free_edge_purity','mean'),free_edge_efficiency=('free_edge_efficiency','mean'),free_tree_exact=('free_tree_exact','mean')))
      display(frame.assign(multiplicity_bin=pd.cut(frame.nodes,[0,8,16,32,10**9])).groupby('multiplicity_bin',observed=True).agg(events=('event_uid','count'),free_edge_purity=('free_edge_purity','mean'),free_edge_efficiency=('free_edge_efficiency','mean')))
      display(level_slices.groupby('retained_level_present').agg(events=('event_uid','nunique'),free_edge_purity=('free_edge_purity','mean'),free_edge_efficiency=('free_edge_efficiency','mean')))
      display(pid_slices.groupby('pid_token').agg(events=('event_uid','nunique'),free_edge_purity=('free_edge_purity','mean'),free_edge_efficiency=('free_edge_efficiency','mean')))
      """),
      md("## Results\n\nEdge/tree efficiency and purity are sliced by multiplicity, level, PID/channel availability, frequency, and partial topology. Mass, B-level Mbc/DeltaE, missing mass, and rare/unseen-channel fields are reported only where their required beam/channel metadata exist."),
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
      summary={'real_input':str(PARQUET),'trained_checkpoint':str(CHECKPOINT),'checkpoint_git_commit':payload.get('git_commit','unknown'),'events':len(events),'edge_tree_metrics':rows,'pid_slices':pid_slice_rows,'level_slices':level_slice_rows,'mass_entries':len(mass_rows),'b_level_mbc_deltae':b_metrics,'missing_mass':{'available':False,'reason':'no channel-specific missing-particle and initial-state contract supplied'},'rare_unseen_channel':rare_unseen,'teacher_forced_vs_free':{'teacher_edge_purity':float(frame.teacher_edge_purity.mean()),'teacher_edge_efficiency':float(frame.teacher_edge_efficiency.mean()),'free_edge_purity':float(frame.free_edge_purity.mean()),'free_edge_efficiency':float(frame.free_edge_efficiency.mean())},'physics_claim_ready':False}
      (OUT/'trained_physics_validation_summary.json').write_text(json.dumps(summary,indent=2))
      """),
      md("## Takeaways\n\nOnly the executed real/checkpoint artifact may support physics conclusions. Missing beam, missing-particle, or training-frequency contracts remain explicitly unavailable rather than inferred."),
    ]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};return nb
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);nbf.write(build_notebook(),a.output);print(a.output);return 0
if __name__=='__main__':raise SystemExit(main())
