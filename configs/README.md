# Configs

`condor/` contains HTCondor resource defaults. `ablations/` contains the
ordered flat-to-full revised-model controls selected with `--ablation`.
`model_presets/` contains round-trippable `tiny_cpu`, `gpu_debug`, and
`production_baseline` architecture contracts. The top-level pretraining and
reconstruction YAML files select the production preset for rendered HTCondor
jobs; local CLI defaults remain `tiny_cpu`.

Expected data roots:

- Toy MC: `/home/boyang/data/MC`
- GraFEI: `/home/boyang/data/graFEI`
