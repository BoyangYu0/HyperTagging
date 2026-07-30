# HTCondor Workflow

Full HyperTagging preprocessing and training should run through HTCondor.

1. Edit `configs/condor/default.yaml` to set the requested CPUs, memory, GPUs,
   and runtime.
2. Render a submit description and executable without submitting:

```bash
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml \
  --command 'uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --device cuda --max-steps 1000' \
  --output outputs/condor/level_reconstruction.sub
```

3. Review the generated `.sub` and `.sh` files and run `shellcheck` on the
   executable if available.
4. Submit explicitly with `condor_submit outputs/condor/level_reconstruction.sub`.
5. Monitor with `condor_q`.

The repository only calls `condor_submit` from a launcher when `--submit` is
given explicitly.

## Recommended revised-model ablation jobs

Render one submit description per ablation; do not submit during local
verification:

```bash
for ablation in flat_baseline heterogeneous_only hyperbolic_lca_parent \
  plus_radius_depth plus_variance_covariance plus_channel \
  plus_relation_attention full_revised
do
  python scripts/condor/render_condor_job.py \
    --config configs/condor/default.yaml \
    --command "python scripts/train_level_reconstruction.py --device cuda --ablation ${ablation} --max-steps 100000" \
    --output "outputs/condor/${ablation}.sub"
done
```

Run hyperbolic pretraining for the objective-bearing variants first, then load
their checkpoints in matched reconstruction runs. Recommended ordering is:

1. flat versus heterogeneous input control;
2. LCA plus true-distance parent ranking;
3. radius-depth direction;
4. variance/covariance anti-collapse;
5. pooled structured-channel supervision;
6. relation-aware attention;
7. the full revised configuration.

Review source-aware split manifests, normalization provenance, output/data
volume paths, GPU/runtime requests, and resume checkpoints before explicit
`condor_submit`. Compare identical splits and seeds. Record both teacher-forced
and free-rollout metrics; do not select only on fixture or next-level loss.
