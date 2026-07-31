#!/usr/bin/env python
"""Generate exact retained-tree geometry and hyperbolic-scale diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_exact_tree_geometry_and_loss_scales.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build_notebook():
    cells = [
        md("""
        # Exact tree geometry and hyperbolic loss scales

        ## tl;dr

        Deterministic CPU diagnostics separate autoregressive reconstruction
        height from exact retained-tree edge distance, verify connected two-B
        parent negatives, and inspect initialization scale for every model preset.
        Fixture results are software evidence only, not physics performance.
        """),
        md("## Context & Methods\n\nThe exact geometry contract uses only `parent_ids`; fixed scale contracts are event-independent."),
        code("""
        import json, os, sys
        from pathlib import Path
        import matplotlib.pyplot as plt
        import pandas as pd
        import torch
        ROOT=Path.cwd()
        if not (ROOT/'src').exists(): ROOT=Path('..').resolve()
        sys.path.insert(0,str(ROOT/'src'))
        from hypertagging.data.tree_geometry import build_exact_tree_geometry, EXACT_TREE_GEOMETRY_CONTRACT_VERSION
        from hypertagging.data.level_collate import build_lca_depth
        from hypertagging.data.heterogeneous import collate_heterogeneous_events, heterogeneous_from_level_event
        from hypertagging.data.tiny_level_fixtures import tiny_level_events
        from hypertagging.losses.hyperbolic_pretraining import build_tree_relation_targets, build_topology_safe_parent_negative_mask, topology_safe_parent_negative_mask, parent_negative_coverage_statistics, hyperbolic_pretraining_loss, dimension_aware_tangent_variance_target, TREE_DISTANCE_CONTRACT_VERSION, HYPERBOLIC_SCALE_CONTRACT_VERSION
        from hypertagging.models.heterogeneous import HeterogeneousNodeEncoder
        from hypertagging.models.hyperbolic import radius, logmap0
        from hypertagging.training.model_config import MODEL_PRESETS
        torch.manual_seed(20260731)
        OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/exact_geometry'))
        OUT.mkdir(parents=True,exist_ok=True)
        """),
        md("## Data\n\nAn explicitly unbalanced tree and a connected Upsilon(4S) → B1+B2 retained tree are constructed in memory."),
        code("""
        unbalanced_parent=torch.tensor([3,2,3,-1])
        unbalanced_level=torch.tensor([0,0,3,4])
        unbalanced=build_exact_tree_geometry(unbalanced_parent)
        lca_height=build_lca_depth(unbalanced_parent,unbalanced_level)
        assert int(unbalanced.exact_tree_path_distance[0,3])==1
        assert int(lca_height[0,3]-unbalanced_level[0])==4
        display(pd.DataFrame({
            'node':range(4),'reconstruction_level':unbalanced_level.tolist(),
            'depth_from_root':unbalanced.depth_from_retained_root.tolist(),
        }))
        display(pd.DataFrame(unbalanced.lca_node_id.numpy(),columns=range(4)))
        display(pd.DataFrame(unbalanced.exact_tree_path_distance.numpy(),columns=range(4)))

        parents=torch.tensor([4,4,0,7,6,6,8,8,-1]); levels=torch.tensor([1,0,0,0,2,0,3,1,4])
        sides=torch.tensor([0,0,0,1,0,0,0,1,-1]); mask=torch.ones(9,dtype=torch.bool)
        geometry=build_exact_tree_geometry(parents); lca=build_lca_depth(parents,levels)
        targets,pair_mask=build_tree_relation_targets(
            parent_ids=parents[None],lca_depth=lca[None],level_ids=levels[None],node_mask=mask[None],b_side=sides[None],
            lca_node_id=geometry.lca_node_id[None],edges_to_lca_from_i=geometry.edges_to_lca_from_i[None],edges_to_lca_from_j=geometry.edges_to_lca_from_j[None])
        eligible=topology_safe_parent_negative_mask(parents,mask,0,lca_depth=lca,tree_relation_targets=targets[0],b_side=sides)
        coverage={k:float(v) for k,v in parent_negative_coverage_statistics(parents[None],mask[None],lca_depth=lca[None],tree_relation_targets=targets,b_side=sides[None]).items()}
        assert eligible[3] and eligible[7] and set(targets[0,0,eligible].tolist())=={4}
        display(pd.DataFrame({'candidate':range(9),'relation_class':targets[0,0].tolist(),'eligible':eligible.tolist()}))
        """),
        md("## Results\n\nInitialization diagnostics exercise finite outputs, component losses, gradients, radii, boundary occupancy, and tangent spread."),
        code("""
        batch=collate_heterogeneous_events([heterogeneous_from_level_event(tiny_level_events()[1])])
        preset_rows=[]
        for name,preset in MODEL_PRESETS.items():
            torch.manual_seed(100+preset.hyper_dim)
            model=HeterogeneousNodeEncoder(d_model=preset.d_model,hyper_dim=preset.hyper_dim,n_heads=preset.n_heads,n_context_layers=preset.n_context_layers,ffn_dim=preset.ffn_dim,dropout=0.0,curvature=preset.curvature,hyper_projection_init_scale=preset.hyper_projection_init_scale,tangent_scale_mode=preset.tangent_scale_mode)
            encoded=model(batch); z=encoded.hyperbolic_embeddings
            loss=hyperbolic_pretraining_loss(z=z,parent_ids=batch['parent_ids'],level_ids=batch['level_ids'],node_mask=batch['node_mask'],exact_tree_path_distance=batch['exact_tree_path_distance'],tangent_variance_target=preset.tangent_variance_target,curvature=preset.curvature)
            loss.total.backward()
            radii=radius(z,curvature=preset.curvature)[batch['node_mask']].detach()
            tangent=logmap0(z,curvature=preset.curvature)[batch['node_mask']].detach()
            boundary=float(((preset.curvature**0.5)*torch.linalg.norm(z[batch['node_mask']].detach(),dim=-1)>=.95).float().mean())
            preset_rows.append({
              'preset':name,'hyper_dim':preset.hyper_dim,'variance_target':preset.tangent_variance_target,
              'dimension_aware_default':dimension_aware_tangent_variance_target(preset.hyper_dim),
              'init_scale':preset.hyper_projection_init_scale,'radius_p05':float(radii.quantile(.05)),
              'radius_p50':float(radii.quantile(.5)),'radius_p95':float(radii.quantile(.95)),
              'boundary_fraction':boundary,'tangent_std_mean':float(tangent.std(0,unbiased=False).mean()),
              'loss_total':float(loss.total.detach()),'loss_components':{k:float(v.detach()) for k,v in loss.components.items()},
              'hyper_projection_gradient_norm':float(model.hyper_projection.weight.grad.norm()),
              'finite_z':bool(torch.isfinite(z).all()),'finite_gradients':bool(torch.isfinite(model.hyper_projection.weight.grad).all()),
            })
        frame=pd.DataFrame([{k:v for k,v in row.items() if k!='loss_components'} for row in preset_rows]); display(frame)
        tiny=MODEL_PRESETS['tiny_cpu'];radius_model=HeterogeneousNodeEncoder(d_model=tiny.d_model,hyper_dim=tiny.hyper_dim,n_heads=tiny.n_heads,n_context_layers=tiny.n_context_layers,ffn_dim=tiny.ffn_dim,dropout=0.0,curvature=tiny.curvature).eval();radius_encoded=radius_model(batch);relation_targets,relation_mask=build_tree_relation_targets(parent_ids=batch['parent_ids'],lca_depth=batch['lca_depth'],level_ids=batch['level_ids'],node_mask=batch['node_mask'],b_side=batch['b_side'],lca_node_id=batch['lca_node_id'],edges_to_lca_from_i=batch['edges_to_lca_from_i'],edges_to_lca_from_j=batch['edges_to_lca_from_j']);negative_mask=build_topology_safe_parent_negative_mask(relation_targets,batch['node_mask'],batch['ancestor_descendant_relation']);radius_rows=[]
        for mode in ('generation_height_radius','exact_root_depth_radius','weak_or_learned_radius'):
            result=hyperbolic_pretraining_loss(z=radius_encoded.hyperbolic_embeddings,parent_ids=batch['parent_ids'],level_ids=batch['level_ids'],node_mask=batch['node_mask'],tree_relation_targets=relation_targets,tree_relation_mask=relation_mask,parent_negative_mask=negative_mask,exact_tree_path_distance=batch['exact_tree_path_distance'],radius_target_mode=mode,depth_from_retained_root=batch['depth_from_retained_root'],distance_to_nearest_retained_root=batch['distance_to_nearest_retained_root']);depth_grad=torch.autograd.grad(result.components['depth'],radius_model.hyper_projection.weight,retain_graph=True)[0].flatten();tree_grad=torch.autograd.grad(result.components['tree_distance'],radius_model.hyper_projection.weight,retain_graph=True)[0].flatten();cosine=float(torch.nn.functional.cosine_similarity(depth_grad,tree_grad,dim=0,eps=1e-12));radius_rows.append({'radius_target_mode':mode,'radius_loss':float(result.components['depth']),'tree_distance_loss':float(result.components['tree_distance']),'gradient_cosine_on_hyperbolic_projection':cosine,'direct_leaf_to_root_example':{'reconstruction_height':4,'exact_root_depth':1},'level1_pointer_metrics':'NOT RUN: requires matched trained held-out checkpoints','full_rollout_metrics':'NOT RUN: requires matched trained held-out checkpoints'})
        display(pd.DataFrame(radius_rows))
        assert frame.finite_z.all() and frame.finite_gradients.all() and (frame.boundary_fraction<=.01).all()
        fig,axes=plt.subplots(1,2,figsize=(10,4)); frame.plot.bar(x='preset',y='radius_p95',ax=axes[0],legend=False,title='Initialization radius p95'); frame.plot.bar(x='preset',y=['variance_target','dimension_aware_default'],ax=axes[1],title='Tangent per-dimension targets'); fig.tight_layout(); fig.savefig(OUT/'exact_geometry_loss_scales.png'); plt.show()
        summary={'tree_geometry_contract':EXACT_TREE_GEOMETRY_CONTRACT_VERSION,'tree_distance_contract':TREE_DISTANCE_CONTRACT_VERSION,'hyperbolic_scale_contract':HYPERBOLIC_SCALE_CONTRACT_VERSION,'direct_leaf_root_edges':int(unbalanced.exact_tree_path_distance[0,3]),'rejected_height_difference':int(lca_height[0,3]-unbalanced_level[0]),'eligible_different_b_positions':eligible.nonzero().flatten().tolist(),'parent_coverage':coverage,'presets':preset_rows,'radius_target_ablations':radius_rows,'default_radius_target_unchanged':'generation_height_radius'}
        (OUT/'exact_geometry_scale_summary.json').write_text(json.dumps(summary,indent=2))
        """),
        md("## Takeaways\n\nThe direct daughter is one edge from the root; connected different-B nodes remain directed negatives; preset initialization checks only numerical contracts. Scientific weights still require held-out ablations."),
    ]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    return notebook


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True); nbf.write(build_notebook(),args.output); print(args.output); return 0


if __name__ == '__main__':
    raise SystemExit(main())
