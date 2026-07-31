#!/usr/bin/env python
"""Generate bounded CPU/GPU runtime scaling diagnostics for model presets."""

from __future__ import annotations
import argparse
from pathlib import Path
import textwrap
import nbformat as nbf

DEFAULT_OUTPUT=Path(__file__).resolve().parents[1]/'notebooks'/'inspect_runtime_scaling.ipynb'
def md(x): return nbf.v4.new_markdown_cell(textwrap.dedent(x).strip())
def code(x): return nbf.v4.new_code_cell(textwrap.dedent(x).strip())

def build_notebook():
    cells=[
      md("""# Runtime scaling

      ## tl;dr

      One bounded deterministic event is timed for each architecture preset.
      These measurements expose mechanics and relative scaling only; they do not
      establish full-data throughput or ten-million-event readiness.
      """),
      md("## Context & Methods\n\nCPU is mandatory. GPU memory is reported only when an already-allowed CUDA device is visible; no device or Condor job is requested."),
      code("""
      import gc,json,os,resource,sys,time
      from pathlib import Path
      import matplotlib.pyplot as plt
      import pandas as pd
      import torch
      ROOT=Path.cwd();ROOT=ROOT if (ROOT/'src').exists() else Path('..').resolve();sys.path.insert(0,str(ROOT/'src'))
      from hypertagging.data.heterogeneous import collate_heterogeneous_events,heterogeneous_from_level_event
      from hypertagging.data.tree_geometry import build_exact_tree_geometry
      from hypertagging.data.tiny_level_fixtures import tiny_level_events
      from hypertagging.losses.level_reconstruction import level_reconstruction_loss
      from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
      import hypertagging.models.level_autoregressive as level_model_module
      from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
      from hypertagging.reconstruction.level_rollout import RolloutConfig,level_rollout
      from hypertagging.training.model_config import MODEL_PRESETS
      torch.manual_seed(20260731);torch.set_num_threads(1)
      OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/runtime'));OUT.mkdir(parents=True,exist_ok=True)
      batch=collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[1])]);policy=ReconstructionConstraintPolicy(mother_charge_compatibility='off')
      """),
      md("## Data\n\nThe fixture has explicit node, level, query, and attention-size accounting."),
      code("""
      def expanded_leaf_batch(template,n):
          old=template['node_mask'].shape[1];result={}
          pair_names={'daughter_adjacency','source_conflict_matrix','lca_depth','lca_node_id','edges_to_lca_from_i','edges_to_lca_from_j','exact_tree_path_distance','ancestor_descendant_relation'}
          for key,value in template.items():
              if not isinstance(value,torch.Tensor):continue
              if key=='recursive_leaf_source_mask':result[key]=torch.eye(n,dtype=torch.bool).unsqueeze(0)
              elif key in pair_names:result[key]=torch.zeros((1,n,n),dtype=value.dtype)
              elif value.ndim>=2 and value.shape[0]==1 and value.shape[1]==old:result[key]=value[:,0:1].expand(1,n,*value.shape[2:]).clone()
              else:result[key]=value.clone()
          result['node_mask']=result['active']=torch.ones((1,n),dtype=torch.bool);result['level_ids'].zero_();result['parent_ids'].fill_(-1);result['node_ids']=torch.arange(n).unsqueeze(0);result['source_node_ids']=torch.arange(n).unsqueeze(0);result['p4'][...,0]=torch.linspace(-.2,.2,n);result['p4'][...,3]=.5
          return result
      rows=[];preset=MODEL_PRESETS['tiny_cpu'];torch.manual_seed(101)
      model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=preset.d_model,hyper_dim=preset.hyper_dim,n_queries=preset.n_queries,n_heads=preset.n_heads,n_context_layers=preset.n_context_layers,curvature=preset.curvature,ffn_dim=preset.ffn_dim,dropout=0.0,max_cardinality=preset.max_cardinality,hyper_projection_init_scale=preset.hyper_projection_init_scale,tangent_scale_mode=preset.tangent_scale_mode).eval()
      for nodes in (32,64,100,160):
          gc.collect();parent=torch.full((nodes,),-1,dtype=torch.long);start=time.perf_counter();geometry=build_exact_tree_geometry(parent);sized=expanded_leaf_batch(batch,nodes);collation_geometry=time.perf_counter()-start
          contextual=[];relations=[];pid_times=[];pointer=[]
          encoder_starts=[];relation_starts=[];decoder_starts=[]
          hooks=[model.encoder.register_forward_pre_hook(lambda *args:encoder_starts.append(time.perf_counter())),model.encoder.register_forward_hook(lambda *args:contextual.append(time.perf_counter()-encoder_starts.pop(0))),model.encoder.physical_relation_bias.register_forward_pre_hook(lambda *args:relation_starts.append(time.perf_counter())),model.encoder.physical_relation_bias.register_forward_hook(lambda *args:relations.append(time.perf_counter()-relation_starts.pop(0))),model.decoder.register_forward_pre_hook(lambda *args:decoder_starts.append(time.perf_counter())),model.decoder.register_forward_hook(lambda *args:pointer.append(time.perf_counter()-decoder_starts.pop(0)))]
          original_rebuild=level_model_module.rebuild_runtime_pid_state
          def timed_rebuild(*args,**kwargs):
              started=time.perf_counter();out=original_rebuild(*args,**kwargs);pid_times.append(time.perf_counter()-started);return out
          level_model_module.rebuild_runtime_pid_state=timed_rebuild
          try:
              with torch.no_grad():output=model(sized,target_level=1,pid_kinematics_mode_override='hard')
          finally:
              level_model_module.rebuild_runtime_pid_state=original_rebuild
              for hook in hooks:hook.remove()
          rollout_batch=expanded_leaf_batch(batch,nodes);start=time.perf_counter();rollout=level_rollout(model,rollout_batch,mode='predicted',config=RolloutConfig(max_level=1,root_types=(),constraint_policy=policy,rollout_pid_kinematics_mode='hard'));one_rollout_level=time.perf_counter()-start
          rows.append({'preset':'tiny_cpu','nodes_per_event':nodes,'collation_geometry_seconds':collation_geometry,'first_contextual_pass_seconds':contextual[0],'pid_rebuild_seconds':sum(pid_times),'second_contextual_pass_seconds':contextual[1],'relation_bias_seconds':sum(relations),'pointer_decoder_seconds':sum(pointer),'one_rollout_level_seconds':one_rollout_level,'attention_memory_bytes_estimate':preset.n_context_layers*preset.n_heads*nodes*nodes*4,'peak_cpu_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'batch_size':1,'fixture_timing_only':True})
      frame=pd.DataFrame(rows);display(frame)
      """),
      md("## Results"),
      code("""
      timing_columns=['collation_geometry_seconds','first_contextual_pass_seconds','pid_rebuild_seconds','second_contextual_pass_seconds','relation_bias_seconds','pointer_decoder_seconds','one_rollout_level_seconds'];fig,axes=plt.subplots(1,2,figsize=(12,4));frame.plot(x='nodes_per_event',y=timing_columns,marker='o',ax=axes[0],title='Bounded stage latency by N');frame.plot(x='nodes_per_event',y='attention_memory_bytes_estimate',marker='o',ax=axes[1],legend=False,title='Attention-logit byte estimate');fig.tight_layout();fig.savefig(OUT/'runtime_scaling.png');plt.show()
      summary={'device':'cpu','torch_threads':torch.get_num_threads(),'node_counts':[32,64,100,160],'separate_stage_timings':timing_columns,'measurements':rows,'gpu_available_in_guarded_environment':bool(torch.cuda.is_available()),'throughput_claim':False,'fixture_timing_only':True,'batched_rollout_design':'ragged event/beam states -> padded node/query blocks -> masked batched forward -> segmented append and compaction between levels'}
      (OUT/'runtime_scaling_summary.json').write_text(json.dumps(summary,indent=2))
      """),
      md("## Takeaways\n\nThe free rollout implementation is explicitly evaluation-only at batch size one. A production batched design requires padded/ragged event and beam axes with segmented append/compaction. Representative scale benchmarking remains deferred."),
    ]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};return nb
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);nbf.write(build_notebook(),a.output);print(a.output);return 0
if __name__=='__main__':raise SystemExit(main())
