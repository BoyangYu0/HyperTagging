# HTCondor Workflow

Full HyperTagging preprocessing and training should run through HTCondor.

1. Edit `configs/condor/default.yaml` to set the requested CPUs, memory, GPUs,
   and runtime.
2. Render a submit description and executable without submitting:

```bash
python scripts/condor/render_condor_job.py --config configs/condor/default.yaml \
  --command 'uv --cache-dir /tmp/uv-cache run python scripts/train_level_reconstruction.py --data /data/volume/manifest.jsonl --pretrained-encoder /data/volume/pretrain/checkpoint.pt --device cuda --max-steps 1000 --output-dir /data/volume/reconstruction' \
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
for ablation in flat_baseline heterogeneous_only contextual_euclidean \
  contextual_hyperbolic_parent_lca plus_radius_depth plus_variance_covariance \
  plus_cross_event_channel plus_hyperbolic_relation_attention plus_leaf_pid \
  plus_scheduled_sampling full_revised
do
  python scripts/condor/render_condor_job.py \
    --config configs/condor/default.yaml \
    --command "python scripts/train_level_reconstruction.py --data /data/volume/manifest.jsonl --pretrained-encoder /data/volume/pretrain/${ablation}/checkpoint.pt --device cuda --ablation ${ablation} --max-steps 100000 --output-dir /data/volume/reconstruction/${ablation}" \
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

Every real training command must include `--data` and a data-volume
`--output-dir`. Pretraining jobs should publish `checkpoint.pt`; matched
reconstruction jobs pass it through `--pretrained-encoder`. Rendering is
read-only. `condor_submit` remains a separate, explicit operator action.

Schema-v4 production manifests are authoritative. The rendered preprocessing
worker passes schema version, leaf kinematics mode, charge-conjugate channel
normalization, buffer size, and row-group size explicitly, then compares the
published sidecar metadata with the manifest. Training consumes the exact
`output_file` JSONL format emitted by `mdst_batch_production.py`.

Recommended first matched-split ablations are
`canonical_pion_first_level`, `contextual_euclidean`,
`plus_leaf_pid`, `plus_scheduled_sampling`, and `full_revised`. Render each
command first and inspect it; no repository command automatically submits a
job.

Build and review a dataset index before rendering a production trainer and add
`--dataset-index /data/volume/dataset_index.json` to the command. This avoids
repeated 10M-event startup scans. Use `--num-workers 0` when exact mid-epoch
resume is required; multiworker jobs partition I/O disjointly but do not claim
an exact prefetched-worker cursor.

Run the JSON-v4/native-nested storage benchmark on a representative pilot and
archive its JSON report before changing production format. Native v5 is
experimental and is never selected automatically.
