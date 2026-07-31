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
      from hypertagging.data.tiny_level_fixtures import tiny_level_events
      from hypertagging.losses.level_reconstruction import level_reconstruction_loss
      from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
      from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
      from hypertagging.reconstruction.level_rollout import RolloutConfig,level_rollout
      from hypertagging.training.model_config import MODEL_PRESETS
      torch.manual_seed(20260731);torch.set_num_threads(1)
      OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/runtime'));OUT.mkdir(parents=True,exist_ok=True)
      batch=collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[1])]);policy=ReconstructionConstraintPolicy(mother_charge_compatibility='off')
      """),
      md("## Data\n\nThe fixture has explicit node, level, query, and attention-size accounting."),
      code("""
      rows=[];nodes=int(batch['node_mask'].sum());levels=int(batch['level_ids'][batch['node_mask']].max())+1
      for name,preset in MODEL_PRESETS.items():
          gc.collect();torch.manual_seed(100+preset.hyper_dim)
          model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=preset.d_model,hyper_dim=preset.hyper_dim,n_queries=preset.n_queries,n_heads=preset.n_heads,n_context_layers=preset.n_context_layers,curvature=preset.curvature,ffn_dim=preset.ffn_dim,dropout=0.0,max_cardinality=preset.max_cardinality,n_queries_by_level=preset.n_queries_by_level,max_cardinality_by_level=preset.max_cardinality_by_level,hyper_projection_init_scale=preset.hyper_projection_init_scale,tangent_scale_mode=preset.tangent_scale_mode).eval()
          start=time.perf_counter();output=model(batch,target_level=1,pid_kinematics_mode_override='hard');forward=time.perf_counter()-start
          loss=(output.pointer.object_logits.square().mean()+output.pointer.type_logits.square().mean()+output.pointer.pointer_logits.square().mean()+output.hyperbolic_embeddings.square().mean())
          model.zero_grad(set_to_none=True);start=time.perf_counter();loss.backward();backward=time.perf_counter()-start
          start=time.perf_counter();rollout=level_rollout(model,batch,mode='teacher_forced',config=RolloutConfig(max_level=3,root_types=(),constraint_policy=policy,rollout_pid_kinematics_mode='hard'));rollout_time=time.perf_counter()-start
          queries={str(level):dict(preset.n_queries_by_level).get(level,preset.n_queries) for level in range(1,levels)}
          attention_bytes=preset.n_context_layers*preset.n_heads*nodes*nodes*4
          rows.append({'preset':name,'nodes_per_event':nodes,'levels_per_event':levels,'queries_by_level':queries,'attention_memory_bytes_estimate':attention_bytes,'forward_seconds':forward,'backward_seconds':backward,'full_rollout_seconds':rollout_time,'rollout_steps':len(rollout.steps),'peak_cpu_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'peak_gpu_memory_mb':None,'batch_size':1,'rollout_scope':'evaluation_only_batch_size_one'})
      frame=pd.DataFrame(rows);display(frame.drop(columns=['queries_by_level']))
      """),
      md("## Results"),
      code("""
      fig,axes=plt.subplots(1,2,figsize=(10,4));frame.plot.bar(x='preset',y=['forward_seconds','backward_seconds','full_rollout_seconds'],ax=axes[0],title='Bounded fixture latency');frame.plot.bar(x='preset',y='attention_memory_bytes_estimate',ax=axes[1],legend=False,title='Attention-logit byte estimate');fig.tight_layout();fig.savefig(OUT/'runtime_scaling.png');plt.show()
      summary={'device':'cpu','torch_threads':torch.get_num_threads(),'measurements':rows,'gpu_available_in_guarded_environment':bool(torch.cuda.is_available()),'throughput_claim':False,'batched_rollout_design':'ragged event/beam states -> padded node/query blocks -> masked batched forward -> segmented append and compaction between levels'}
      (OUT/'runtime_scaling_summary.json').write_text(json.dumps(summary,indent=2))
      """),
      md("## Takeaways\n\nThe free rollout implementation is explicitly evaluation-only at batch size one. A production batched design requires padded/ragged event and beam axes with segmented append/compaction. Representative scale benchmarking remains deferred."),
    ]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'};return nb
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);nbf.write(build_notebook(),a.output);print(a.output);return 0
if __name__=='__main__':raise SystemExit(main())
