#!/usr/bin/env python
"""Create the reproducible 10M mDST production validation notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-template", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    production_root = args.production_root.resolve()
    repo_root = args.repo_root.resolve()
    final_validation = production_root / "validation" / "final_validation.json"
    if not final_validation.is_file():
        raise FileNotFoundError(
            f"Run exhaustive campaign validation first: {final_validation}"
        )
    summary = json.loads(final_validation.read_text(encoding="utf-8"))
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    cells: list[dict] = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# 10M run-independent mDST production validation\n\n"
            "## tl;dr\n\n"
            f"- Exhaustive validation covers **{summary['validated_events']:,} events** "
            f"in **{summary['completed_shards']:,} shards** with "
            f"**{summary['unique_event_uids']:,} unique event UIDs**.\n"
            f"- The campaign uses schema `{summary['schema_version']}`, source commit "
            f"`{summary['source_git_commit'][:12]}`, and KLM scope "
            f"`{summary['klm_training_scope']}`.\n"
            f"- Completion markers valid: `{summary['all_completion_markers_valid']}`; "
            f"missing shards: `{len(summary['missing_shards'])}`.\n"
            "- The remaining cells independently reconstruct shard/resource distributions, "
            "inspect log and metadata coverage, and render representative event trees."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This is a technical companion to the production report. The authoritative "
            "campaign validator reads every shard and verifies hashes, provenance, exact "
            "source ranges, event counts, and global UID uniqueness. This notebook then "
            "aggregates all per-shard result/metadata sidecars in bounded memory and reads "
            "only selected events for topology pictures.\n\n"
            "### Key Assumptions\n\n"
            "- A shard is usable only when Parquet, `.metadata.json`, `.complete`, and "
            "`.result.json` agree with its immutable manifest task.\n"
            "- Non-whitespace stderr is treated as a review item.\n"
            "- Event-tree pictures are representative examples; numeric campaign checks are exhaustive."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from __future__ import annotations\n"
            "from collections import Counter, defaultdict\n"
            "from itertools import islice\n"
            "import json, os, re, sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            f"PRODUCTION_ROOT = Path({str(production_root)!r})\n"
            f"REPO_ROOT = Path({str(repo_root)!r})\n"
            "MANIFEST = PRODUCTION_ROOT / 'manifests' / 'mdst_10m_ri_all_exp.jsonl'\n"
            "FINAL_VALIDATION = PRODUCTION_ROOT / 'validation' / 'final_validation.json'\n"
            "FIGURE_DIR = PRODUCTION_ROOT / 'validation' / 'notebook' / 'figures'\n"
            "FIGURE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
            "from hypertagging.preprocessing.schema_v4 import iter_event_records_v4\n"
            "records = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]\n"
            "validation = json.loads(FINAL_VALIDATION.read_text())\n"
            "print({'manifest_tasks': len(records), 'validated_events': validation['validated_events'], "
            "'campaign_id': validation['campaign_id'], 'figures': str(FIGURE_DIR)})"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Data\n\n### 1. Reconstruct all shard-level statistics"))
    cells.append(
        nbf.v4.new_code_cell(
            "rows = []\n"
            "node_hist = Counter(); depth_hist = Counter(); leaf_modes = Counter(); roots = Counter(); klm_by_category = Counter()\n"
            "missing_companions = []\n"
            "max_node_event = None; max_depth_event = None\n"
            "for record in records:\n"
            "    parquet = Path(record['output_file'])\n"
            "    companion_paths = {\n"
            "        'parquet': parquet, 'metadata': Path(str(parquet) + '.metadata.json'),\n"
            "        'marker': Path(str(parquet) + '.complete'), 'result': Path(str(parquet) + '.result.json')}\n"
            "    absent = [name for name, path in companion_paths.items() if not path.is_file()]\n"
            "    if absent: missing_companions.append({'task_id': record['task_id'], 'missing': absent})\n"
            "    result = json.loads(companion_paths['result'].read_text())\n"
            "    counts = result.get('node_counts', [])\n"
            "    depths = result.get('max_depths', [])\n"
            "    node_hist.update(counts); depth_hist.update(depths)\n"
            "    leaf_modes.update(result.get('actual_leaf_mode_distribution', {}))\n"
            "    roots.update(result.get('b_root_distribution', {}))\n"
            "    klm_by_category[record['physics_category']] += int(result.get('klm_nodes', 0))\n"
            "    if counts:\n"
            "        local = max(enumerate(counts), key=lambda item: item[1])\n"
            "        candidate = (local[1], record['task_id'], local[0], parquet)\n"
            "        max_node_event = max(max_node_event, candidate) if max_node_event else candidate\n"
            "    if depths:\n"
            "        local = max(enumerate(depths), key=lambda item: item[1])\n"
            "        candidate = (local[1], record['task_id'], local[0], parquet)\n"
            "        max_depth_event = max(max_depth_event, candidate) if max_depth_event else candidate\n"
            "    rows.append({\n"
            "        'task_id': record['task_id'], 'category': record['physics_category'],\n"
            "        'experiment': next(part for part in Path(record['input_file']).parts if re.fullmatch(r'e\\d+', part)),\n"
            "        'events': result['events'], 'output_mib': result['output_bytes'] / 2**20,\n"
            "        'events_per_second': result['events_per_second'],\n"
            "        'elapsed_seconds': result['elapsed_seconds'],\n"
            "        'validation_seconds': result['validation_seconds'],\n"
            "        'peak_rss_mib': result['peak_resident_memory_kib'] / 1024,\n"
            "        'klm_nodes': result.get('klm_nodes', 0),\n"
            "        'unique_event_uids': result['unique_event_uids']})\n"
            "shards = pd.DataFrame(rows).sort_values('task_id').reset_index(drop=True)\n"
            "display(shards.groupby('category').agg(shards=('task_id','count'), events=('events','sum'), "
            "throughput_median=('events_per_second','median'), peak_rss_p99=('peak_rss_mib', lambda x: x.quantile(.99)), "
            "output_gib=('output_mib', lambda x: x.sum()/1024)).round(3))\n"
            "assert not missing_companions\n"
            "assert shards.events.sum() == validation['validated_events'] == 10_000_000\n"
            "assert shards.unique_event_uids.sum() == 10_000_000\n"
            "assert sum(node_hist.values()) == 10_000_000\n"
            "shards.to_csv(FIGURE_DIR / 'shard_metrics.csv', index=False)\n"
            "print({'missing_companions': len(missing_companions), 'node_hist_events': sum(node_hist.values()), "
            "'max_node_event': max_node_event[:3], 'max_depth_event': max_depth_event[:3]})"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### 2. Confirm RI-only input and experiment coverage"))
    cells.append(
        nbf.v4.new_code_cell(
            "input_paths = {Path(record['input_file']) for record in records}\n"
            "experiments = sorted({next(part for part in path.parts if re.fullmatch(r'e\\d+', part)) for path in input_paths})\n"
            "ri_violations = [str(path) for path in input_paths if 'MC16ri_run2' not in path.parts]\n"
            "coverage = {\n"
            "    'manifest_tasks': len(records), 'unique_input_files': len(input_paths),\n"
            "    'experiments': experiments, 'categories': sorted(shards.category.unique()),\n"
            "    'ri_path_violations': len(ri_violations),\n"
            "    'source_commits': sorted({r['source_git_commit'] for r in records}),\n"
            "    'source_states': sorted({r['source_state'] for r in records}),\n"
            "    'klm_scopes': sorted({r['klm_training_scope'] for r in records})}\n"
            "display(pd.Series(coverage, name='value').to_frame())\n"
            "assert experiments == ['e1004'] and not ri_violations\n"
            "assert coverage['source_states'] == ['clean'] and coverage['klm_scopes'] == ['included']"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Results\n\n### 3. Category coverage is complete and reflects available input volume"))
    cells.append(
        nbf.v4.new_code_cell(
            "category = shards.groupby('category', as_index=False).events.sum().sort_values('events')\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            "bars = ax.barh(category.category, category.events / 1e6, color='#3973ac', edgecolor='#1f2937')\n"
            "ax.bar_label(bars, fmt='%.3fM', padding=4, fontfamily='monospace')\n"
            "ax.set(title='Validated events by physics category', xlabel='Events (millions)', ylabel='')\n"
            "ax.grid(axis='x', alpha=.2); fig.tight_layout(); fig.savefig(FIGURE_DIR/'category_coverage.png', dpi=180); plt.show()"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### 4. Throughput and memory remain well below the Condor requests"))
    cells.append(
        nbf.v4.new_code_cell(
            "categories = sorted(shards.category.unique()); colors = plt.cm.tab10(np.linspace(0, .75, len(categories)))\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))\n"
            "axes[0].boxplot([shards.loc[shards.category==c,'events_per_second'] for c in categories], tick_labels=categories, showfliers=False)\n"
            "axes[0].set(title='Worker throughput by category', ylabel='Events / second'); axes[0].tick_params(axis='x', rotation=35); axes[0].grid(axis='y', alpha=.2)\n"
            "for color, c in zip(colors, categories):\n"
            "    part=shards[shards.category==c]; axes[1].scatter(part.events_per_second, part.peak_rss_mib, s=14, alpha=.55, label=c, color=color)\n"
            "axes[1].axhline(8192, color='#222', ls='--', lw=1, label='8 GiB request')\n"
            "axes[1].set(title='Memory versus throughput', xlabel='Events / second', ylabel='Peak RSS (MiB)'); axes[1].grid(alpha=.2); axes[1].legend(ncol=2, fontsize=8)\n"
            "fig.tight_layout(); fig.savefig(FIGURE_DIR/'throughput_memory.png', dpi=180); plt.show()"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### 5. Full-event topology and leaf-mode distributions"))
    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(17, 5))\n"
            "x=np.array(sorted(node_hist)); y=np.array([node_hist[v] for v in x]); axes[0].plot(x,y,color='#3973ac',lw=2); axes[0].fill_between(x,y,color='#3973ac',alpha=.18)\n"
            "axes[0].set(title='Nodes per event', xlabel='Retained nodes', ylabel='Events'); axes[0].grid(alpha=.2)\n"
            "x=np.array(sorted(depth_hist)); y=np.array([depth_hist[v] for v in x]); axes[1].bar(x,y,color='#c7832b',edgecolor='#1f2937')\n"
            "axes[1].set(title='Maximum retained level', xlabel='Level', ylabel='Events'); axes[1].grid(axis='y',alpha=.2)\n"
            "labels=list(leaf_modes); values=np.array([leaf_modes[k] for k in labels]); order=np.argsort(values)\n"
            "axes[2].barh(np.array(labels)[order],values[order]/1e6,color='#7b6ba8',edgecolor='#1f2937')\n"
            "axes[2].set(title='Leaf kinematics modes', xlabel='Nodes (millions)', ylabel=''); axes[2].grid(axis='x',alpha=.2)\n"
            "fig.tight_layout(); fig.savefig(FIGURE_DIR/'topology_leaf_modes.png',dpi=180); plt.show()"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### 6. KLM nodes are present across every category"))
    cells.append(
        nbf.v4.new_code_cell(
            "klm=pd.Series(klm_by_category).sort_values(); fig,ax=plt.subplots(figsize=(10,5.5)); bars=ax.barh(klm.index,klm.values,color='#b45f55',edgecolor='#1f2937')\n"
            "ax.bar_label(bars,fmt='%d',padding=4,fontfamily='monospace'); ax.set(title='Retained KLM nodes by category',xlabel='Nodes',ylabel=''); ax.grid(axis='x',alpha=.2)\n"
            "fig.tight_layout(); fig.savefig(FIGURE_DIR/'klm_by_category.png',dpi=180); plt.show(); assert (klm>0).all()"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### 7. Condor stderr and publication metadata are clean"))
    cells.append(
        nbf.v4.new_code_cell(
            "log_dir=PRODUCTION_ROOT/'logs'/'condor'; err_files=sorted(log_dir.glob('*.err'))\n"
            "condor=json.loads((PRODUCTION_ROOT/'validation'/'condor_monitor_summary.json').read_text())\n"
            "successful_ids={str(condor['clusters']['bulk']['cluster_id']), str(condor['clusters']['successful_preflight']['cluster_id'])}\n"
            "rejected_id=str(condor['clusters']['initial_preflight_rejected']['cluster_id'])\n"
            "nonempty=[]\n"
            "for path in err_files:\n"
            "    text=path.read_text(errors='replace').strip()\n"
            "    if text: nonempty.append({'path':str(path),'bytes':path.stat().st_size,'preview':text[:300]})\n"
            "successful_nonempty=[item for item in nonempty if any(f'-{cluster_id}.' in Path(item['path']).name for cluster_id in successful_ids)]\n"
            "rejected_nonempty=[item for item in nonempty if f'-{rejected_id}.' in Path(item['path']).name]\n"
            "log_summary={'stderr_files':len(err_files),'all_non_whitespace_stderr':len(nonempty), "
            "'successful_non_whitespace_stderr':len(successful_nonempty), 'rejected_preflight_stderr':len(rejected_nonempty), "
            "'completion_markers':sum(Path(str(Path(r['output_file']))+'.complete').is_file() for r in records), "
            "'metadata_sidecars':sum(Path(str(Path(r['output_file']))+'.metadata.json').is_file() for r in records), "
            "'result_sidecars':sum(Path(str(Path(r['output_file']))+'.result.json').is_file() for r in records)}\n"
            "display(pd.Series(log_summary,name='count').to_frame()); display(pd.DataFrame(nonempty).head(20))\n"
            "assert log_summary['completion_markers']==len(records)==2000\n"
            "assert log_summary['metadata_sidecars']==len(records) and log_summary['result_sidecars']==len(records)\n"
            "assert not successful_nonempty\n"
            "assert len(rejected_nonempty)==1 and condor['initial_preflight_disposition']['accepted'] is False"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### 8. Representative event trees preserve mother-above-daughter geometry"))
    cells.append(
        nbf.v4.new_code_cell(
            "def event_at(path, index):\n"
            "    return next(islice(iter_event_records_v4(path), index, index+1))\n\n"
            "def draw_tree(event, ax, title):\n"
            "    nodes={int(n['node_id']):n for n in event['nodes']}\n"
            "    levels=defaultdict(list)\n"
            "    for node in nodes.values(): levels[int(node['level'])].append(int(node['node_id']))\n"
            "    pos={}\n"
            "    for level, ids in sorted(levels.items()):\n"
            "        ids=sorted(ids); span=max(len(ids)-1,1)\n"
            "        for j,node_id in enumerate(ids): pos[node_id]=((j-(len(ids)-1)/2)/span,level)\n"
            "    for node in nodes.values():\n"
            "        parent=int(node['node_id'])\n"
            "        for child in node.get('daughter_ids',[]):\n"
            "            child=int(child)\n"
            "            if child in pos: ax.plot([pos[parent][0],pos[child][0]],[pos[parent][1],pos[child][1]],color='#9ca3af',lw=.6,zorder=1)\n"
            "    kind_colors={'track':'#3973ac','ecl_cluster':'#c7832b','klm_cluster':'#b45f55','composite':'#7b6ba8'}\n"
            "    for node_id,node in nodes.items():\n"
            "        x,y=pos[node_id]; kind=node['node_kind']; ax.scatter(x,y,s=28,color=kind_colors.get(kind,'#6b7280'),edgecolor='#1f2937',lw=.4,zorder=2)\n"
            "        if len(nodes)<=75: ax.text(x,y+.07,f\"{node_id}:{kind[:2]}\\n{node.get('truth_pdg',node.get('pdg',''))}\",ha='center',va='bottom',fontsize=4.2)\n"
            "    ax.set(title=title, xlabel=f\"{event['event_uid']} | nodes={len(nodes)}\", ylabel='Retained level'); ax.set_xticks([]); ax.grid(axis='y',alpha=.15)\n"
            "    return nodes\n\n"
            "category_records={}\n"
            "for record in records: category_records.setdefault(record['physics_category'],record)\n"
            "fig,axes=plt.subplots(4,2,figsize=(16,20)); axes=axes.ravel()\n"
            "for ax,(category,record) in zip(axes,sorted(category_records.items())):\n"
            "    event=event_at(Path(record['output_file']),0); draw_tree(event,ax,category)\n"
            "axes[-1].axis('off'); fig.suptitle('Representative event topology from every physics category',fontsize=16,y=.995); fig.tight_layout(); fig.savefig(FIGURE_DIR/'representative_category_trees.png',dpi=180,bbox_inches='tight'); plt.show()\n"
            "special=[('maximum nodes',max_node_event),('maximum depth',max_depth_event)]\n"
            "fig,axes=plt.subplots(2,1,figsize=(18,15))\n"
            "for ax,(label,(_,task_id,index,path)) in zip(axes,special): draw_tree(event_at(path,index),ax,f'{label}: task {task_id}, event offset {index}')\n"
            "fig.tight_layout(); fig.savefig(FIGURE_DIR/'extreme_topology_trees.png',dpi=200,bbox_inches='tight'); plt.show()"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Takeaways"))
    cells.append(
        nbf.v4.new_code_cell(
            "takeaways = {\n"
            " 'validated_events': int(validation['validated_events']),\n"
            " 'unique_event_uids': int(validation['unique_event_uids']),\n"
            " 'shards': int(validation['completed_shards']),\n"
            " 'categories': validation['category_distribution'],\n"
            " 'experiments': experiments, 'unique_input_files': len(input_paths),\n"
            " 'output_gib': float(validation['output_bytes'])/2**30,\n"
            " 'bytes_per_event': float(validation['output_bytes_per_event']),\n"
            " 'throughput_median_events_per_second': float(shards.events_per_second.median()),\n"
            " 'peak_rss_p99_mib': float(shards.peak_rss_mib.quantile(.99)),\n"
            " 'successful_non_whitespace_stderr': len(successful_nonempty), 'rejected_preflight_stderr': len(rejected_nonempty), 'missing_companions': len(missing_companions),\n"
            " 'klm_nodes': int(validation['klm_node_distribution']['klm_nodes']),\n"
            " 'node_count_quantiles': validation['node_count_quantiles'],\n"
            " 'maximum_depth_quantiles': validation['maximum_depth_quantiles']}\n"
            "(FIGURE_DIR/'notebook_takeaways.json').write_text(json.dumps(takeaways,indent=2,sort_keys=True)+'\\n')\n"
            "display(pd.Series(takeaways,name='value').to_frame())"
        )
    )
    notebook["cells"] = cells
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)
    if bool(args.manifest_template) != bool(args.manifest_output):
        raise ValueError(
            "--manifest-template and --manifest-output must be supplied together"
        )
    if args.manifest_template and args.manifest_output:
        manifest_notebook = nbf.read(args.manifest_template, as_version=4)
        replaced = 0
        classification_replaced = 0
        for cell in manifest_notebook.cells:
            if cell.cell_type != "code":
                continue
            old = "ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/\"src\"))"
            new = (
                "ROOT=Path(os.environ[\"HYPERTAGGING_REPO_ROOT\"]); "
                "sys.path.insert(0,str(ROOT/\"src\"))"
            )
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
                replaced += 1
            if cell.source.startswith("status=frame[[\"task_id\",\"output_file\"]].copy()"):
                cell.source = (
                    "status=frame[[\"task_id\",\"output_file\"]].copy()\n"
                    "def published_classification(row):\n"
                    "    parquet=Path(row.output_file); marker=Path(str(parquet)+'.complete'); result=Path(str(parquet)+'.result.json'); metadata=Path(str(parquet)+'.metadata.json')\n"
                    "    if not all(path.is_file() for path in (parquet,marker,result,metadata)): return 'MISSING_COMPANION'\n"
                    "    marker_payload=json.loads(marker.read_text()); result_payload=json.loads(result.read_text())\n"
                    "    checks=[marker_payload.get('task_id')==row.task_id, marker_payload.get('task_record_hash')==row.task_record_hash, marker_payload.get('campaign_id')==row.campaign_id, marker_payload.get('event_count')==row.planned_events, result_payload.get('classification')=='COMPLETE_VALID', result_payload.get('events')==row.planned_events, result_payload.get('unique_event_uids')==row.planned_events]\n"
                    "    return 'COMPLETE_VALID_PUBLISHED' if all(checks) else 'INVALID_PUBLICATION_CONTRACT'\n"
                    "status['classification']=[published_classification(row) for row in frame.itertuples(index=False)]\n"
                    "display(status.groupby('classification').size().rename('shards').to_frame())\n"
                    "validation=json.loads((Path(os.environ['HYPERTAGGING_MANIFEST']).parents[1]/'validation'/'final_validation.json').read_text())\n"
                    "assert (status.classification=='COMPLETE_VALID_PUBLISHED').all()\n"
                    "assert validation['validated_events']==validation['unique_event_uids']==10_000_000 and validation['all_completion_markers_valid']\n"
                    "report={'pass':not overlaps,'campaign_source_task_provenance_pass':valid_task_hashes and len(summary['campaign_ids'])==len(summary['source_git_commits'])==1,'completion_marker_hash_validation':'verified by final_validation.json publication validator','global_uid_check':'10,000,000 globally unique IDs verified by final_validation.json','global_uid_validation_passes':1,'publication_contract':'parquet + hashed metadata sidecar + parsed completion marker published last','overwrite_invalidates_old_marker_first':True,'invalid_task_classifications':status.groupby('classification').size().to_dict(),'summary':summary}\n"
                    "(OUT/'production_manifest_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')\n"
                    "display(pd.Series({'validated_events':validation['validated_events'],'unique_event_uids':validation['unique_event_uids'],'completion_markers_valid':validation['all_completion_markers_valid'],'missing_shards':len(validation['missing_shards'])},name='value').to_frame())"
                )
                classification_replaced += 1
        if replaced != 1:
            raise RuntimeError(
                f"Expected one repository-root setup cell, replaced {replaced}"
            )
        if classification_replaced != 1:
            raise RuntimeError(
                "Expected one manifest publication-classification cell, replaced "
                f"{classification_replaced}"
            )
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(manifest_notebook, args.manifest_output)
        print(args.manifest_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
