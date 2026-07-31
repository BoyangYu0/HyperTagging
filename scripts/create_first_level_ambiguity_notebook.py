#!/usr/bin/env python
"""Generate the bounded Level-0-to-1 ambiguity diagnostic notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / "inspect_first_level_ambiguity.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build_notebook():
    cells = [
        md("""
        # First-level set ambiguity

        ## tl;dr

        This fixture diagnostic evaluates the current independent multi-label
        pointer factorization at Level 0 -> 1. It does not promote a more complex
        decoder or make a physics-performance claim.
        """),
        md("## Context & Methods\n\nAll metrics use proposal-, daughter-, or truth-mother-local denominators. The soft type-conditioned query-to-node relation bias is exercised through the real model pointer path and remains disabled by default. Whole-set and iterative-pointer ideas are deferred designs, not runnable configs; no unrestricted enumerator is used."),
        code("""
        import json,math,os,sys
        from pathlib import Path
        import matplotlib.pyplot as plt
        import pandas as pd
        import torch
        ROOT=Path.cwd();ROOT=ROOT if (ROOT/'src').exists() else Path('..').resolve();sys.path.insert(0,str(ROOT/'src'))
        from hypertagging.data.heterogeneous import collate_heterogeneous_events,heterogeneous_from_level_event
        from hypertagging.data.tiny_level_fixtures import tiny_level_events
        from hypertagging.losses.level_reconstruction import level_reconstruction_loss,targets_for_level
        from hypertagging.models.level_autoregressive import LevelAutoregressiveReconstructor
        seed=int(os.environ.get('HYPERTAGGING_NOTEBOOK_SEED','20260730'));torch.manual_seed(seed);OUT=Path(os.environ.get('HYPERTAGGING_FIGURE_DIR','/tmp/hypertagging_figures/first_level_ambiguity'));OUT.mkdir(parents=True,exist_ok=True)
        events=tiny_level_events();batch=collate_heterogeneous_events([heterogeneous_from_level_event(event) for event in events]);model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=24,hyper_dim=8,n_queries=8,n_context_layers=1).eval();output=model(batch,target_level=1)
        torch.manual_seed(seed);relation_model=LevelAutoregressiveReconstructor(n_features=12,n_types=41,hidden_dim=24,hyper_dim=8,n_queries=8,n_context_layers=1,type_conditioned_daughter_relation_bias=True).train();relation_output=relation_model(batch,target_level=1);relation_change=float((relation_output.pointer.pointer_logits-output.pointer.pointer_logits).abs().mean().detach());relation_output.pointer.pointer_logits[relation_output.context_mask[:,None].expand_as(relation_output.pointer.pointer_logits)].sum().backward();relation_table_gradient=float(relation_model.decoder.type_relation_table.grad.abs().sum());compatibility_gradient=float(relation_model.decoder.compatibility_node.weight.grad.abs().sum())
        """),
        md("## Results"),
        code("""
        target_types,target_masks,target_p4,target_charge=targets_for_level(batch,1,target_policy='diagnostic_all');loss=level_reconstruction_loss(output.pointer,batch,target_level=1,target_policy='diagnostic_all')
        rows=[]
        for event_index,event in enumerate(events):
            fsp=batch['node_mask'][event_index]&(batch['level_ids'][event_index]==0);pointer_prob=torch.sigmoid(output.pointer.pointer_logits[event_index,:,fsp]);object_prob=torch.sigmoid(output.pointer.object_logits[event_index]);pointer_entropy=-(pointer_prob*pointer_prob.clamp_min(1e-9).log()+(1-pointer_prob)*(1-pointer_prob).clamp_min(1e-9).log()).mean(-1);object_entropy=-(object_prob*object_prob.clamp_min(1e-9).log()+(1-object_prob)*(1-object_prob).clamp_min(1e-9).log())
            matched=dict(loss.matches[event_index]);proposal_sets=[];overlap_count=duplicates=valid_sets=0;tp=predicted=truth_total=0
            for query in range(pointer_prob.shape[0]):
                selected=set(pointer_prob[query].ge(.5).nonzero().flatten().tolist());proposal_sets.append(tuple(sorted(selected)));predicted+=len(selected)
                if query in matched:
                    truth=set(target_masks[event_index][matched[query]].nonzero().flatten().tolist());tp+=len(selected&truth);truth_total+=len(truth)
                    exact=int(selected==truth)
                else:exact=0
                charge=float(batch['charge'][event_index,fsp][list(selected)].sum()) if selected else 0.;p4=batch['p4'][event_index,fsp][list(selected)].sum(0) if selected else torch.zeros(4);mass=float((p4[3].square()-p4[:3].square().sum()).clamp_min(0).sqrt());valid_sets+=int(len(selected)>=2 and math.isfinite(mass) and abs(charge)<=2)
                daughter_pid_composition=tuple(sorted(int(batch['pid_labels'][event_index,fsp][index]) for index in selected))
                rows.append({'event_id':event.event_id,'fsp_multiplicity':int(fsp.sum()),'target_cardinality':int(target_masks[event_index][matched[query]].sum()) if query in matched else 0,'query':query,'pointer_entropy':float(pointer_entropy[query]),'object_query_entropy':float(object_entropy[query]),'exact_daughter_set_match':exact,'whole_set_physical_validity':int(len(selected)>=2 and math.isfinite(mass) and abs(charge)<=2),'mass':mass,'charge':charge,'mother_type':int(output.pointer.type_logits[event_index,query].argmax()),'daughter_pid_composition':daughter_pid_composition,'predicted_daughters':len(selected)})
            duplicates=len(proposal_sets)-len(set(proposal_sets));source_overlap=sum(bool(set(left)&set(right)) for i,left in enumerate(proposal_sets) for right in proposal_sets[i+1:] if left and right)
            rows.append({'event_id':event.event_id,'fsp_multiplicity':int(fsp.sum()),'target_cardinality':sum(int(mask.sum()) for mask in target_masks[event_index]),'query':-1,'pointer_entropy':float(pointer_entropy.mean()),'object_query_entropy':float(object_entropy.mean()),'exact_daughter_set_match':float(sum(row['exact_daughter_set_match'] for row in rows if row['event_id']==event.event_id and row['query']>=0)/max(len(target_types[event_index]),1)),'whole_set_physical_validity':valid_sets/max(len(proposal_sets),1),'mass':None,'charge':None,'mother_type':-1,'predicted_daughters':predicted,'per_daughter_marginal_precision':tp/max(predicted,1),'per_daughter_marginal_recall':tp/max(truth_total,1),'source_overlap':source_overlap,'duplicate_proposal_rate':duplicates/max(len(proposal_sets),1)})
        frame=pd.DataFrame(rows);event_rows=frame[frame['query']==-1];display(event_rows);display(frame[frame['query']>=0].groupby(['mother_type','predicted_daughters']).agg(proposals=('query','size'),exact_set_rate=('exact_daughter_set_match','mean'),mass_validity=('whole_set_physical_validity','mean')));display(event_rows.groupby(['fsp_multiplicity','target_cardinality']).agg(pointer_entropy=('pointer_entropy','mean'),object_entropy=('object_query_entropy','mean'),exact_set=('exact_daughter_set_match','mean'),marginal_precision=('per_daughter_marginal_precision','mean'),marginal_recall=('per_daughter_marginal_recall','mean'),source_overlap=('source_overlap','mean'),duplicate_rate=('duplicate_proposal_rate','mean')))
        """),
        code("""
        fig,axes=plt.subplots(1,2,figsize=(10,4));event_rows.plot.scatter(x='fsp_multiplicity',y='pointer_entropy',ax=axes[0],title='Pointer entropy vs FSP multiplicity');event_rows.plot.scatter(x='target_cardinality',y='exact_daughter_set_match',ax=axes[1],title='Exact set match vs target cardinality');fig.tight_layout();fig.savefig(OUT/'first_level_ambiguity.png');plt.show()
        report={'factorization':'independent_multi_label_pointer','level_transition':'0_to_1','metrics':event_rows.to_dict('records'),'mother_type_conditioned_daughter_composition':frame[frame['query']>=0].to_dict('records'),'optional_ablations':{'whole_set_compatibility_scorer':{'status':'DEFERRED_DESIGN_NO_RUNNABLE_CONFIG'},'iterative_within_mother_pointer':{'status':'DEFERRED_DESIGN_NO_RUNNABLE_CONFIG'},'type_conditioned_daughter_relation_bias':{'enabled':True,'actual_model_pointer_path':relation_output.pointer.query_node_compatibility_bias is not None,'pointer_logit_mean_absolute_change':relation_change,'type_relation_table_gradient':relation_table_gradient,'compatibility_projection_gradient':compatibility_gradient}},'unrestricted_combinatorial_enumerator':False,'physics_claim':False};assert relation_change>0 and relation_table_gradient>0 and compatibility_gradient>0;(OUT/'first_level_ambiguity_summary.json').write_text(json.dumps(report,indent=2))
        """),
        md("## Takeaways\n\nUse the local precision/recall, entropy, overlap, duplication, charge/mass validity, and exact-set diagnostics to decide whether the independent factorization is the bottleneck. Any decoder promotion requires a matched held-out study."),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
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
