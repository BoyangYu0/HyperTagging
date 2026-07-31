# Final correctness and production-readiness report

Date: 2026-07-31 (Europe/Berlin)

## Revision and verification

- Branch/HEAD: `master` at `274d2df8a9df8b25142b68966dfe30c828538b90`
  (`code audit`). The repository metadata is read-only in this workspace, so
  the completed revision is an uncommitted worktree on that HEAD.
- Python: `/data/dust/user/boyangyu/uv_env/bin/python`.
- Initial tests: 183 passed, 8 skipped, 18 warnings in 389.01 s.
- Final tests: 223 passed, 8 skipped, 18 warnings in 315.00 s.
- Initial notebooks: 12/12 passed.
- Final notebooks: 12/12 passed; executed copies are in
  `/tmp/hypertagging-final-readiness-notebook-smoke`.
- GitHub Actions: the committed HEAD has a successful `CPU correctness` run
  (run 30617205526). No CI run can represent the uncommitted local worktree.
- HTCondor: two submit descriptions were rendered under `/tmp` and shell syntax
  was checked. No job was submitted.
- basf2: unavailable on this host, so no real mDST pilot was run.

## Implemented contracts

- Composite inputs now have separate `MODEL_COMPOSITE_FEATURE_NAMES` and
  `TARGET_COMPOSITE_METADATA_NAMES`. A versioned v4 adapter, model-owned
  normalizer masks, sidecar/index statistics, and checkpoint feature contract
  keep truth/completeness fields out of common/composite embeddings, relation
  bias, hyperbolic state, pointers, and type logits without regenerating v4.
- Scheduled targets use `fallback_teacher` by default. Skip, masked-only, and
  explicit recovery modes are available; unavailable mothers are masked from
  no-object loss. Schedule progress is optimizer-step-only, while level affects
  only deterministic sampling seeds.
- `ReconstructionConstraintPolicy` is serializable and shared by training,
  scheduled contexts, validation, and rollout. It controls level types,
  hard/soft/off empirical priors, context kinds, recursive conflicts, pointer
  thresholds, cardinality, reduced-token charge, and optional loose physical
  cuts. Fixed-hypothesis candidates remain accessible even with unknown kind.
- Dataset indexes recompute their own hash and validate PID/schema/feature,
  source path/size/digest, sidecar, marker content/hash, source ranges, target
  policy, and selection fingerprint. Capacity statistics are policy-specific
  and level-resolved; truncated pilots cannot use full-dataset indexes.
- Pretraining uses the runtime detector-context/PID/rebuilt-p4/second-context
  path. Teacher composites use explicit teacher type, corruptions use corrupted
  type, invalid candidates are excluded from positive structural losses, and
  invalid/fallback B roots are excluded from current-batch and memory channels.
- Predicted scheduled rollouts run once per event, cache every level state, and
  batch contexts by target level. The deterministic microbenchmark uses two
  cached forwards versus three independent-prefix forwards for the same two
  required prior levels. Leaf-PID loss is applied once per selected event.
- Exact resume stores and checks batch/shuffle/seed/max-events/index/split/
  target/curriculum/schedule/level-sampling/accumulation/worker settings plus
  actual epoch and batch index. It accurately claims single-worker replay by
  batch, not an unused physical row-group cursor.
- Validation uses its configured next-level batch size, target policy,
  per-event macro metrics, global and per-level summed micro statistics,
  representability, confidence, bounded channel summaries, multiplicity/depth
  slices, free-rollout tree metrics, and an eligible-target complete-efficiency
  denominator.
- Channel events now carry flat PID, depth-by-PID, branch-multiplicity,
  intermediate-particle, and exact full/reconstructable IDs through collation
  and pretraining. Similarity weights are explicit.
- Real v4 training requires parquet + sidecar + valid completion marker unless
  the fixture override is explicit. Overwrite invalidates the old marker first,
  publishes validated parquet/sidecar, and commits the marker last. Global UID
  count/uniqueness/digest share one event decode pass; SQLite resources clean up
  in `finally`.
- Experimental v5 stays default-off and uses a feature-spec-built Arrow schema,
  validates unknown nested fields, retains late optional fields, and reports
  size/write/full-read/projected-read/JSON-CPU/RSS metrics with the Arrow-memory
  caveat. The tiny two-event benchmark is diagnostic, not a production choice.
- `tiny_cpu`, `gpu_debug`, and `production_baseline` expose all requested model
  dimensions and per-level query/cardinality overrides. Pretraining and
  reconstruction checkpoints store the complete architecture. Production
  encoder transfer enforces configurable coverage and rejects shape/unexpected
  key mismatches unless explicitly overridden.

## Change surface

The implementation changes the v4/v5 schema and channel modules; heterogeneous
loading/normalization/index/capacity; constraint, rollout, scheduled-alignment,
loss, metrics, pretraining, reconstruction, transfer, checkpoint, and model
configuration modules; both real-training CLIs; production validation; six
notebook generators plus the deterministic runner; generated notebooks; model
and production YAML; training/config documentation; the baseline audit; and 33
focused CPU regression-test files named in the task specification.

## Operator commands (not run here)

Exact 50-event basf2 pilot:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/generic_mdst.root \
  --output /data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
  --schema-version direct-mdst-tree-v4 \
  --entry-sequence 0:49 --max-events 50 \
  --event-buffer-size 32 --row-group-size 16
```

Dry-run HTCondor rendering (these create files but do not submit):

```bash
DATA_MANIFEST=/data/volume/manifest.jsonl \
  OUTPUT_DIR=/data/volume/pretrain \
  scripts/condor/submit_hyperbolic_pretrain.sh \
  --output /tmp/hypertagging-pretrain.sub

DATA_MANIFEST=/data/volume/manifest.jsonl \
  PRETRAINED_ENCODER=/data/volume/pretrain/checkpoint.pt \
  OUTPUT_DIR=/data/volume/reconstruction \
  scripts/condor/submit_level_reconstruction.sh \
  --output /tmp/hypertagging-reconstruction.sub

scripts/condor/submit_mdst_production_10m.sh --dry-run
```

## Remaining scientific uncertainties

CPU fixtures establish software invariants, not physics performance. A real
sub-100-event basf2 pilot must still validate reconstructed provenance, PID
likelihood availability, charge and p4 behavior, and publication metadata.
Encoder/tree/channel objectives, calibration, rare-channel behavior, and
hyperbolic geometry require trained held-out studies. Ten-million-event I/O and
memory behavior remains unmeasured, and the two-event v5 benchmark is far too
small to justify replacing v4. No physics-performance improvement is claimed.

