#!/usr/bin/env python
"""Generate rollout PID/search/calibration fixture diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import nbformat as nbf

DEFAULT_OUTPUT=Path(__file__).resolve().parents[1]/'notebooks'/'inspect_rollout_search_and_calibration.ipynb'

def md(x): return nbf.v4.new_markdown_cell(textwrap.dedent(x).strip())
def code(x): return nbf.v4.new_code_cell(textwrap.dedent(x).strip())

def build_notebook():
    cells=[
      md("""# Rollout search and calibration

      ## tl;dr

      CPU fixtures make PID conditioning, hard construction, greedy/set-packing/
      bounded-beam differences, duplicate/overlap metrics, candidate survival,
      first divergence, and reliability-bin mechanics explicit. Random fixture
      scores are not calibration or physics measurements.
      """),
      md("## Context & Methods\n\nGreedy remains the reproducible baseline. Set packing and the one/two-level beam are bounded evaluation comparators."),
      code("""
      import json,os,sys
      from pathlib import Path
      import matplotlib.pyplot as plt
      import numpy as np
      import pandas as pd
      import torch
      ROOT=Path.cwd(); ROOT=ROOT if (ROOT/'src').exists() else Path('..').resolve(); sys.path.insert(0,str(ROOT/'src'))
      from hypertagging.data.heterogeneous import collate_heterogeneous_events,heterogeneous_from_level_event
      from hypertagging.data.tiny_level_fixtures import tiny_level_events
      from hypertagging.evaluation.hierarchical_metrics import summarize_rollout
      from hypertagging.losses.level_reconstruction import level_reconstruction_loss,confidence_calibration_metrics
      from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
      from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
      from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
      from hypertagging.reconstruction.level_rollout import CompositeProposal,RolloutConfig,level_rollout,proposal_ambiguity_metrics,resolve_exclusive_proposals,resolve_weighted_set_packing,bounded_beam_proposal_sets,resolver_difference_rate,rollout_search_metrics
      torch.manual_seed(20260731); OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/rollout_search')); OUT.mkdir(parents=True,exist_ok=True)
      event=heterogeneous_from_level_event(tiny_level_events()[1]); batch=collate_heterogeneous_events([event]); leaves=batch['node_mask']&(batch['level_ids']==0); batch['leaf_kinematics_mode_ids'][leaves]=LEAF_MODE_TO_ID['raw_track_predicted_pid']; batch['pid_labels'][leaves]=0
      model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=16,hyper_dim=4,n_queries=8,n_context_layers=1).eval(); policy=ReconstructionConstraintPolicy(mother_charge_compatibility='off')
      """),
      md("## Data\n\nA deterministic retained B-decay fixture is decoded under each explicit rollout PID contract."),
      code("""
      pid_rows=[]; rollout_results={}
      expectations={'soft_decision_hard_construction':('soft_expectation','hard'),'hard':('hard','hard'),'temperature_softmax':('temperature_softmax','temperature_softmax'),'straight_through_hard':('straight_through_hard','straight_through_hard')}
      for mode,(forward,construction) in expectations.items():
          cfg=RolloutConfig(max_level=3,root_types=(),constraint_policy=policy,rollout_pid_kinematics_mode=mode,seed=17)
          result=level_rollout(model,batch,mode='teacher_forced',config=cfg); repeat=level_rollout(model,batch,mode='teacher_forced',config=cfg); rollout_results[mode]=result
          pid_rows.append({'rollout_mode':mode,'relation_mode':result.steps[0].model_output.relation_pid_kinematics_mode,'decision_mode':result.steps[0].model_output.decision_pid_kinematics_mode,'construction_mode':result.steps[0].appended_mother_p4_pid_kinematics_mode,'reproducible':bool(torch.equal(result.batch['p4'],repeat.batch['p4'])),'p4_closure':summarize_rollout(result.batch,batch)['p4_closure_rate']})
      pid_frame=pd.DataFrame(pid_rows); display(pid_frame); assert pid_frame.reproducible.all() and (pid_frame.p4_closure==1).all()
      """),
      md("## Results\n\nSynthetic competing proposals isolate resolver behavior without claiming trained quality."),
      code("""
      sources=torch.eye(3,dtype=torch.bool)
      proposals=[CompositeProposal(0,4,(0,1),1,.9),CompositeProposal(1,4,(0,),1,.6),CompositeProposal(2,4,(1,),1,.6),CompositeProposal(3,5,(2,),1,.4),CompositeProposal(4,4,(0,),1,.5)]
      greedy=resolve_exclusive_proposals(proposals,recursive_leaf_source_mask=sources)
      packed=resolve_weighted_set_packing(proposals,recursive_leaf_source_mask=sources,max_proposals=8)
      beam=bounded_beam_proposal_sets(proposals,recursive_leaf_source_mask=sources,beam_width=3,max_proposals=8)
      ambiguity=proposal_ambiguity_metrics(proposals,greedy,total_queries=8,recursive_leaf_source_mask=sources)
      search=rollout_search_metrics(rollout_results['hard'],batch)
      resolver={'greedy':[x.query_id for x in greedy],'set_packing':[x.query_id for x in packed],'beam':[[x.query_id for x in h] for h in beam],'greedy_set_packing_difference':resolver_difference_rate(greedy,packed),'greedy_beam_best_difference':resolver_difference_rate(greedy,beam[0])}
      display(pd.DataFrame({'resolver':['greedy','set packing','beam best'],'queries':[resolver['greedy'],resolver['set_packing'],resolver['beam'][0]]}))

      output=model(batch,target_level=1,pid_kinematics_mode_override='hard'); losses=level_reconstruction_loss(output.pointer,batch,target_level=1,constraint_policy=policy)
      confidence=confidence_calibration_metrics(output.pointer.confidence_logits,losses.confidence_targets,n_bins=5)
      object_probability=torch.sigmoid(output.pointer.object_logits).detach().flatten(); object_target=torch.zeros_like(output.pointer.object_logits)
      for event_matches in losses.matches:
          for query,_target in event_matches: object_target[0,query]=1
      object_target=object_target.flatten()
      def reliability(prob,target,bins=5):
          rows=[]
          for index in range(bins):
              lo,hi=index/bins,(index+1)/bins; selected=(prob>=lo)&((prob<=hi) if index==bins-1 else (prob<hi))
              rows.append({'bin':index,'count':int(selected.sum()),'confidence':float(prob[selected].mean()) if selected.any() else None,'accuracy':float(target[selected].float().mean()) if selected.any() else None})
          return rows
      reliability_tables={'object':reliability(object_probability,object_target),'confidence':reliability(torch.sigmoid(output.pointer.confidence_logits).detach().flatten(),losses.confidence_targets.detach().flatten())}
      # Type and pointer reliability are defined on Hungarian-matched queries.
      type_prob=[];type_ok=[];pointer_prob=[];pointer_ok=[]
      truth_types=batch['pid_target_labels'][0]
      for query,target_row in losses.matches[0]:
          context=batch['node_mask'][0]&(batch['level_ids'][0]<1); truth_nodes=(batch['node_mask'][0]&(batch['level_ids'][0]==1)).nonzero().flatten(); truth_node=truth_nodes[target_row]
          tp=torch.softmax(output.pointer.type_logits[0,query],-1); type_prob.append(float(tp.max())); type_ok.append(int(tp.argmax()==truth_types[truth_node]))
          pp=torch.sigmoid(output.pointer.pointer_logits[0,query,context]).detach(); pt=batch['daughter_adjacency'][0,truth_node,context]; pointer_prob.extend(pp.tolist()); pointer_ok.extend(pt.int().tolist())
      reliability_tables['type']=reliability(torch.tensor(type_prob),torch.tensor(type_ok)) if type_prob else []
      reliability_tables['pointer']=reliability(torch.tensor(pointer_prob),torch.tensor(pointer_ok)) if pointer_prob else []
      fig,axes=plt.subplots(1,4,figsize=(14,3))
      for ax,(name,rows) in zip(axes,reliability_tables.items()):
          visible=[row for row in rows if row['count']]; ax.plot([row['confidence'] for row in visible],[row['accuracy'] for row in visible],marker='o'); ax.plot([0,1],[0,1],'--',color='gray'); ax.set(title=name,xlabel='confidence',ylabel='observed')
      fig.tight_layout(); fig.savefig(OUT/'reliability_diagrams.png'); plt.show()
      summary={'pid_modes':pid_rows,'ambiguity':ambiguity,'resolver':resolver,'candidate_search':search,'reliability':reliability_tables,'confidence_metrics':confidence,'first_divergence_level':summarize_rollout(rollout_results['hard'].batch,batch)['first_divergence_level'],'evaluation_only_batch_size_one':True,'batched_rollout_design':'pad active states per beam/event, mask query and node axes, segmented append, then compact between levels'}
      (OUT/'rollout_search_calibration_summary.json').write_text(json.dumps(summary,indent=2))
      """),
      md("## Takeaways\n\nThe hybrid default is soft-conditioned for neural decisions and hard only for construction. Beam/set packing stay evaluation-only until held-out measurements justify any promotion; calibration diagrams from random fixtures validate mechanics only."),
    ]
    nb=nbf.v4.new_notebook(cells=cells); nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}; return nb

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);nbf.write(build_notebook(),a.output);print(a.output);return 0
if __name__=='__main__': raise SystemExit(main())
