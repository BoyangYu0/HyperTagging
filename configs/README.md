# Configs

`condor/` contains HTCondor resource defaults. `ablations/` contains the
ordered flat-to-full revised-model controls selected with `--ablation`.
`model_presets/` contains round-trippable `tiny_cpu`, `gpu_debug`,
`small_candidate`, and `production_baseline` architecture contracts.
`small_candidate` uses 128/32 dimensions with four heads and four context
layers; its copied conservative query/cardinality placeholders are explicitly
`capacity_report_required` and scientific reconstruction rejects the preset
without a checked dataset index. Local CLI defaults remain `tiny_cpu`.

`first_level_type_relation_bias.yaml` is a real, disabled-by-default soft
query-node scoring ablation and is serialized in the architecture contract.
Whole-set scoring and iterative pointer decoding have no runnable configs; they
are deferred in `docs/deferred_model_ablations.md` so no YAML can be accepted
and ignored. Level encoding configs are `learned_euclidean`,
`bounded_tangent_level_embedding`, and `none`.

The production query-repulsion weight remains zero; `query_repulsion_off`,
`weak_query_repulsion`, and `stronger_query_repulsion` are matched campaign
configs. PID configs cover soft expectation, temperature softmax,
straight-through hard, hard, and rollout soft-decision/hard-construction. The
four `pretrain_stage*` files provide staged-loss campaign variants, while
`hyperbolic_pretrain_pilot.yaml` requires objective-gradient preflight. None of
these fixture-executable surfaces is a claim of scientific optimality.

Expected data roots:

- Toy MC: `/home/boyang/data/MC`
- GraFEI: `/home/boyang/data/graFEI`

## Immutable scientific data selection

The canonical reduced-production selections are under
`training_selection/production_1m_20260812/`. Pass `train_035k.json`,
`train_100k.json`, or `train_250k.json` as the existing trainer `--data` value;
`build_real_data_module` detects the selection contract and applies its
source-level train/validation/test roles before any stable hash split. A
selection manifest cannot be combined with `--max-events`, and
`scientific_mode=True` rejects raw data paths entirely.
Scientific trainer validation hash-ranks UIDs from the manifest's validation
role and checkpoints the exact cohort. Source-order prefixes remain available
only when `scientific_mode=False`; neither selection path reads the sealed test
role.

Build the corresponding dataset index with
`scripts/build_dataset_index.py --selection-manifest ... --scientific-mode`.
Do not use raw `max_events=N` prefixes as scientific subsets. Legacy JSONL
production manifests and prefixes remain supported for explicitly diagnostic
CPU/CI runs. The manifest's UID status remains a promotion gate until a full
event-level index (not `--from-sidecars`) validates UID uniqueness and source
consistency.
