# Current repository audit status

This is the sole authoritative current audit report. It was prepared on
2026-07-31 against starting HEAD
`c29342e89237c197850d4d40f6bf25536643de28` on branch `master`. The starting
worktree was clean. The earlier `f064f49` reports describe an older state; the
patch they called uncommitted was subsequently committed in `c29342e` and is
not an uncommitted patch at this starting revision.

## Evidence boundary

The literal system `python -m pytest -q` could not run because `/usr/bin/python`
has no `pytest`. The repository environment
`/data/dust/user/boyangyu/uv_env/bin/python` was therefore used for executable
evidence. Before this audit's edits it produced `248 passed, 8 skipped, 19
warnings in 401.83s`; all 15 indexed default fixture notebooks also passed and
were retained under `/tmp/hypertagging-pre-fix-notebooks`.

GitHub has one exact-starting-SHA run: workflow `CPU correctness`, run
`30656603435`, completed successfully for `c29342e89237c197850d4d40f6bf25536643de28`.
That is historical evidence for the starting SHA only. No CI result can exist
for the current uncommitted worktree.

## Current implementation disposition

The authoritative item-by-item classification is
[`issue_ledger.yaml`](issue_ledger.yaml). Current source now:

- wires the optional type-conditioned query-to-node relation bias through typed
  config, model logits, gradients, checkpoints, CLI, and the diagnostic
  notebook;
- removes runnable whole-set and iterative-pointer YAMLs because those designs
  are not yet scientifically coherent;
- calls the alternative level representation
  `bounded_tangent_level_embedding`, with ordered leaf-outside tangent radii;
- propagates relation-attention dropout;
- enforces explicit model-input/truth-supervision provenance and tests cloned
  truth-only perturbation invariance;
- makes append-time composites persist daughter-summed physical features and
  defers contextual daughter pooling to the next encoder pass;
- retains `evaluation_reference_rollout` as the bounded batch-size-one
  correctness path and adds a vectorized `batched_level_step` for one complete
  masked target level;
- uses a checkpointed channel-memory ring buffer, vectorized channel anchors,
  and configurable multi-objective gradient diagnostics.

This does not make free rollout production-ready. Teacher-forced validation is
batched; the reference free rollout remains bounded and batch-size one; a full
multi-level batched production rollout remains open. Weighted set packing and
beam search are evaluation-only and bounded.

## Final verification

The final committed SHA remains
`c29342e89237c197850d4d40f6bf25536643de28`; this audit did not commit. The
worktree is intentionally dirty with the audit corrections and generated
notebook updates. Exact results:

- `/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q`:
  `263 passed, 8 skipped, 19 warnings in 367.96s (0:06:07)`;
- focused audit/config/model/checkpoint/leakage/composite/rollout/channel tests:
  `34 passed in 11.39s`;
- `scripts/validate_audit_integrity.py`: `PASS`, 13 archives, 71 ledger
  items, and one current-status document;
- `scripts/execute_notebook_smoke_tests.py --list`: 15 default deterministic
  CPU groups, derived from `notebooks/index.yaml`;
- requested post-fix notebook execution: all 15 passed, retained under
  `/tmp/hypertagging-post-fix-notebooks`;
- finalized first-level ambiguity diagnostic: one passed, retained under
  `/tmp/hypertagging-first-level-final`;
- 28 declared JSON reports in the post-fix suite were checked for all seven
  required provenance fields with no missing fields;
- `python -m compileall -q src scripts tests`, `git diff --check`, and staged
  rename diff checks pass.

No fixture timing is a throughput claim. Real-only notebooks were not executed.

## Ledger summary and remaining gaps

The 71-item ledger contains 55 `FIXED_AND_TESTED`, 6
`IMPLEMENTED_NOT_REAL_VERIFIED`, 5 `PARTIAL`, 4
`INTENTIONALLY_DEFERRED_SCIENCE`, 1 `OBSOLETE_OR_DUPLICATE`, and no `OPEN`
items. The principal partial items are full multi-level batched free rollout,
remaining bounded diagnostic/reference Python work and GPU measurement, KLM
input completeness, current-worktree remote CI, and human visual review.

The type-conditioned first-level ablation is proven to affect actual pointer
logits and gradients. Its YAML is accepted and consumed by the reconstruction
CLI; the switch round-trips through the architecture contract and trained
evaluation loader. `bounded_tangent_level_embedding` changes active outputs,
has ordered leaf-outside radii, receives gradients, and is padding invariant.
Whole-set and iterative-pointer runnable configs were removed, and unknown YAML
keys are rejected.

## Changed source surface

Core source changes are in `data/heterogeneous.py`,
`models/{ablation,first_level_ablations,heterogeneous,level_autoregressive,mother_pointer,relation_attention}.py`,
`losses/hyperbolic_pretraining.py`, `reconstruction/level_rollout.py`,
`training/{model_config,pretrain_trainer,reconstruction_trainer}.py`, and
`evaluation/trained_context.py`. CLI/notebook execution changes are in the two
training CLIs, four notebook generators, the indexed notebook runner, and the
audit-integrity validator. The tracked notebooks were regenerated from their
generators.

## Bounded real-mDST pilot commands

These commands were not run in this audit:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/generic_mdst.root \
  --output /data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
  --schema-version direct-mdst-tree-v4 \
  --entry-sequence 0:49 --max-events 50 \
  --event-buffer-size 32 --row-group-size 16

HYPERTAGGING_REAL_PILOT=/data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
HYPERTAGGING_REAL_PILOT_REPORT=/data/dust/user/boyangyu/hypertagging/pilot-v4-report.json \
  /data/dust/user/boyangyu/uv_env/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/inspect_real_mdst_pilot.ipynb
```

## First HTCondor ablation commands

First build and review a schema-v4 dataset index. Then render—without
submitting—the full revised pretrain and matched baseline/type-relation
reconstruction jobs:

```bash
DATA_MANIFEST=/data/volume/manifest.jsonl \
OUTPUT_DIR=/data/volume/pretrain/full_revised \
  scripts/condor/submit_hyperbolic_pretrain.sh \
  --output outputs/condor/pretrain_full_revised.sub

python scripts/condor/render_condor_job.py \
  --config configs/condor/default.yaml \
  --command "python scripts/train_level_reconstruction.py --config configs/level_reconstruction.yaml --data /data/volume/manifest.jsonl --dataset-index /data/volume/dataset_index.json --pretrained-encoder /data/volume/pretrain/full_revised/checkpoint.pt --device cuda --ablation full_revised --seed 11 --max-steps 100000 --output-dir /data/volume/reconstruction/full_revised" \
  --output outputs/condor/reconstruction_full_revised.sub

python scripts/condor/render_condor_job.py \
  --config configs/condor/default.yaml \
  --command "python scripts/train_level_reconstruction.py --config configs/level_reconstruction.yaml --data /data/volume/manifest.jsonl --dataset-index /data/volume/dataset_index.json --pretrained-encoder /data/volume/pretrain/full_revised/checkpoint.pt --device cuda --ablation first_level_type_relation_bias --seed 11 --max-steps 100000 --output-dir /data/volume/reconstruction/first_level_type_relation_bias" \
  --output outputs/condor/reconstruction_first_level_type_relation_bias.sub
```

Review `.sub`/`.sh`, data and output paths, split/index hashes, GPU resources,
and checkpoint compatibility. Only then may an operator explicitly run
`condor_submit` on the reviewed files. No job was submitted here.

## External evidence still required

The following were not available locally and remain explicitly unverified:

- a real sub-100-event basf2/schema-v4 mDST pilot;
- a trained checkpoint evaluated on matched held-out real data;
- current-worktree remote CI (requires a commit and push);
- guarded GPU correctness and representative GPU throughput;
- ten-million-event storage, memory, and runtime measurements;
- physics improvement, calibration, rare-channel performance, and optimal
  decoding conclusions.

The two real-only notebooks have no fixture fallback. Missing inputs produce a
machine-readable `NOT RUN` status.
