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

Expected data roots:

- Toy MC: `/home/boyang/data/MC`
- GraFEI: `/home/boyang/data/graFEI`
