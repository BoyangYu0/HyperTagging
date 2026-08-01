# Configs

`condor/` contains HTCondor resource defaults. `ablations/` contains the
ordered flat-to-full revised-model controls selected with `--ablation`.
`model_presets/` contains round-trippable `tiny_cpu`, `gpu_debug`, and
`production_baseline` architecture contracts. The top-level pretraining and
reconstruction YAML files select the production preset for rendered HTCondor
jobs; local CLI defaults remain `tiny_cpu`.

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
