# Final Runtime and Scale Audit

Audit date: 2026-07-31

Branch: `master`

Revision: `d2362847cf036599d955cd4c70b9d2d9a3d83a08`
(`fix data production`)

## Baseline

The worktree was clean. The literal `python -m pytest -q` selected
`/usr/bin/python`, which does not have pytest installed. The documented
repository environment completed the baseline:

```text
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
173 passed, 8 skipped, 18 warnings in 504.21s
```

The warnings are existing odd-head PyTorch nested-tensor warnings and explicit
legacy-conflated diagnostic warnings.

## Preserved invariants

- reconstructed raw-track p3, charge, input PID, and canonical energy remain
  independent of MC truth;
- input and truth daughter PID histograms remain distinct;
- raw, fixed-candidate, ECL, composite, truth-only, and legacy leaf modes remain
  explicit;
- reconstructed mother p4 remains a recursive daughter sum;
- detector contextualization precedes the principal hyperbolic projection;
- the reduced 41-token vocabulary remains the only model PID/type vocabulary;
- parent/LCA/tree-distance/radius/VICReg/channel objectives, Hungarian matching,
  constrained recursive-source decoding, confidence supervision, partial-target
  policies, and HTCondor safety remain intact;
- v1/v2/v3/v4 remain readable, with legacy training opt-in required.

## Confirmed issues

1. The data module normalizes the stored common/composite blocks, but runtime
   PID reconstruction overwrites p4/mass fields with raw values before pass B.
2. Common continuous inputs still include PID token, level, active, and copied
   numeric values despite dedicated categorical state.
3. Truth-guided composite input type stays unknown while predicted composites
   carry their generated mother type.
4. Scheduled training always computes teacher loss and then adds predicted
   losses, double-counting selected predicted events.
5. Checkpoints restore RNG and optimizer state but restart the event iterator
   at epoch zero.
6. Startup separately scans event payloads for split statistics, legacy state,
   allowed types, normalization, and capacity.
7. Native v4 places serialized JSON inside Parquet; no native nested benchmark
   exists.
8. DataLoader workers enumerate all events and discard modulo-assigned records,
   duplicating input parsing.
9. Corruption labels are inferred from corrupted-node ordinal rather than the
   operation actually applied; failed corruptions can remain labelled.
10. Hard-negative selection can include ancestors and close same-branch
    relatives and does not pass configured curvature to the distance.
11. Pretraining PID loss selects every track node rather than explicit raw-track
    mode.
12. Validation reports only a small rollout summary and limits all validation
    by the rollout-event count.
13. Requested global leaf mode is trusted as metadata even for mixed node
    collections.
14. V4 summary compatibility reads a nonexistent aggregate key, UID validation
    materializes records/sets, publication lacks an explicit completion marker,
    and large diagnostic materialization properties are not guarded.

## Compatibility and implementation plan

- Keep the stored v4 node contract readable and introduce a feature-spec
  revision that masks categorical common slots from continuous geometry.
- Add a model-side runtime feature normalizer, populated from train-fitted
  Welford state and used after every runtime rebuild.
- Add explicit runtime composite-type source IDs and use target mother type only
  for already generated teacher-forced composites; leaf truth PID remains
  target-only.
- Make scheduled context selection the primary per-event loss, with optional
  zero-default auxiliary teacher loss.
- Add serializable single-worker streaming cursors; reject unsupported exact
  multiworker resume rather than claiming determinism.
- Build/merge a versioned dataset index from shard metadata and Welford
  sufficient statistics, retaining `--rescan-dataset` as an explicit fallback.
- Add native nested Arrow v5 alongside v4 and benchmark both on deterministic
  synthetic fixtures before any 10M production decision.
- Assign disjoint row groups/files to workers, correct corruption and
  hard-negative labels, expand validation counts, and validate actual leaf-mode
  output distributions.

## Verification boundary

CPU fixtures can establish feature-contract consistency, deterministic
single-worker resume, bounded iteration, native-vs-JSON benchmark mechanics,
loss selection, and metric aggregation. They cannot establish 10M throughput,
real basf2 PIDLikelihood coverage, PID/confidence calibration, or physics
efficiency. No production job or Condor submission is authorized in this task.

## Implemented resolution

- `RuntimeFeatureNormalizer` is a checkpointed model module. Both contextual
  passes and every runtime p4/composite rebuild enter through its train-fitted
  common/composite transform; categorical common slots are handled by dedicated
  embeddings and flags.
- Previous teacher-forced composites carry the explicit
  `truth_teacher_forced` runtime type source and one-hot target mother type.
  Predicted/corrupted composites carry their generated/configured type. Leaf
  truth PID remains target-only.
- Scheduled reconstruction chooses one primary context per event and level. A
  teacher auxiliary is available but defaults to zero.
- Checkpoints store a logical streaming cursor. Single-worker resume replays
  the deterministic bounded stream to that cursor; exact multiworker resume is
  rejected explicitly.
- `hypertagging-dataset-index-v1` stores split/source counts, Welford state,
  allowed types, capacity/cardinality histograms, depth, completeness,
  category, schema/PID, and feature hashes. `--dataset-index` avoids startup
  payload rescans.
- Worker iteration partitions files, or row groups when workers outnumber
  files, before decoding. V4 publication now requires parquet, sidecar, and a
  completion marker. Global UID validation uses a temporary SQLite uniqueness
  index.
- Corruption labels record only operations that changed state. Hard negatives
  are explicit different-B/unrelated relations and exclude close relatives.
- V4 remains production. A native nested experimental v5 writer and bounded
  benchmark are supplied for review; it is not the 10M default.

## Acceptance tests

New focused tests cover runtime normalization/categorical separation,
composite runtime types, sampled-context loss selection, real streaming-trainer
resume, dataset indexing/native storage, corruption/hard-negative semantics,
and disjoint DataLoader worker row groups. The final full-suite result is
recorded in the completion report rather than this pre-edit audit.
