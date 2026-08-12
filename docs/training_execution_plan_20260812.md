# HyperTagging training execution plan — 2026-08-12

## Decision and evidence boundary

The first production path remains single-GPU, level-autoregressive set
reconstruction with 3–7 reconstruction levels. It is not an LLM-like long
sequence problem. The large unordered FSP input (about 40 median and up to 119
in the production report) makes first-level combinatorics the capacity and
memory bottleneck.

No convergence or held-out reconstruction performance has yet been
demonstrated. Publication/provenance completeness of the 10M campaign is not
scientific training evidence. The conservative decision is therefore to fix
and demonstrate the small-model path on immutable subsets before considering
1M, 10M, DDP, or publication claims.

This document distinguishes:

- **Software evidence:** deterministic selection, contract enforcement,
  numerical safety, restart fidelity, test results, and artifact hashes.
- **Scientific evidence:** learning relative to frozen baselines, free-rollout
  reconstruction, held-out category/channel robustness, calibration, and
  paired-seed ablations. Passing software tests is necessary but is not a
  scientific result.

## Live finding matrix

| Finding | Exact location | 2026-08-12 state | Why / required action |
|---|---|---|---|
| Clean branch integration preserves both histories | branch `training-integration-20260812`, merge `83776a2c4f9e7edd776915edf30816730881f9eb`; tags `pre-training-master-537d7cc`, `pre-training-focused-8e9d0b8` | Fixed | Use this branch lineage; do not rewrite either parent. |
| Production source object missing | `configs/training_selection/production_1m_20260812/provenance_status.json` | Blocking evidence-only issue | Commit `f4e54df23b5c60115e475c5d68df4651899d678e` is unavailable; expected tree `b6e3a4118b960e3a4676a61af9601438d56cef96` cannot be derived independently. CPU work may proceed; scientific Slurm work must fail closed. |
| 1M reduced data is complete but category-skewed | `/project/agkuhr/users/boyang/data/HyperTagging_uni`; `inventory.json` | Software-validated publication inventory | 200 v4 shards × 5,000 events; 64 mixed, 45 uubar, 21 taupair, 20 ccbar, 18 charged, 16 ddbar, 16 ssbar. It is every tenth production task, not the full 10M composition. |
| Raw `max_events=N` is a source-order prefix | `src/hypertagging/training/data_module.py`, `src/hypertagging/data/dataset_index.py` | Corrected for selection manifests | Scientific selections now use explicit whole-shard roles; a selection manifest rejects `max_events`. Legacy diagnostic manifests retain prefix compatibility. |
| Train-only normalization must follow immutable roles | `build_real_data_module`, `build_dataset_index[_from_sidecars]` | Corrected and CPU-tested | Source-role overrides are applied before hash splitting; only `train` contributes normalizer statistics. |
| Event UID uniqueness is not proven by sidecars | selection manifests: `uid_validation.status` | Required correction/gate | The later full index must scan selected event records, reject duplicate UIDs and source/task disagreement, and record the completed gate. Sidecars cannot prove it. |
| Cyclic pretraining curriculum | `src/hypertagging/training/pretrain_trainer.py` curriculum step selection | Correction needed next tranche | Replace step-modulo cycling with explicit progressive phase durations and persisted phase/cursor state. |
| Pilot routes to production-sized model | `configs/hyperbolic_pretrain_pilot.yaml: model_preset` | Correction needed next tranche | Route pilot to `gpu_debug`; production preset is promotion-only. |
| Four validation batches | pretraining config/trainer default `validation_batches` | Correction needed next tranche | CI-only sampling is too noisy. Use fixed manifest-backed validation cohorts and event budgets below. |
| No LR warmup | pretraining optimizer/scheduler construction | Correction needed next tranche | Add persisted step-based warmup and test resume equivalence at the warmup boundary. |
| Geometry/VIC under autocast | hyperbolic/VIC loss and diagnostics in pretraining trainer | Correction needed next tranche | Run log/exp maps, distances, covariance, variance, and associated diagnostics in explicit FP32 with autocast disabled. |
| Dominance warning 100:1 | `configs/hyperbolic_pretrain_pilot.yaml: objective_dominance_ratio` | Correction needed next tranche | Default warning target should be 20:1; screen 10–30:1 and require an explicit waiver for intentional dominance. |
| Reconstruction validation is CI-sized | `src/hypertagging/training/reconstruction_trainer.py`, `max_validation_events=32`, rollout 8 | Correction needed next tranche | Use fixed 2k/1k at screening and 5k/2k at candidate scale; final evaluates all 50k test events. |
| Primary checkpoint is teacher-forced loss | `configs/level_reconstruction.yaml: primary_metric=validation_loss_total` | Correction needed next tranche | Promote by a preregistered rollout composite/gates; retain teacher-forced loss as diagnostic. |
| Scheduled sampling tops at 25% | reconstruction schedule/config | Scientific ablation needed | Paired screening at 0/10/25/50%; do not assume 25% is optimal. |
| DDP/channel synchronization unproven | distributed trainer path/channel memory | Deferred | No DDP until single-GPU convergence, exact resume, synchronized memory, and metric equivalence are demonstrated. |

## Phase 0 — repository and immutable provenance

1. Keep `training-integration-20260812` based on merge `83776a2`; preserve both
   existing tags. Make future implementation commits small and reviewable. Do
   not merge either historical branch again.
2. Before an experiment, create an immutable annotated experiment tag from a
   reviewed clean commit and record commit and tree in `run_contract.json`.
   Archive the exact environment lock, rendered configuration, patch/diff
   status, and repository bundle or source tarball. A dirty tree is a STOP.
3. Recover `f4e54df...` from the campaign worktree, an object bundle, archive,
   or original submit host. Verify:

   ```bash
   git cat-file -e f4e54df23b5c60115e475c5d68df4651899d678e^{commit}
   git show -s --format='%H %T' f4e54df23b5c60115e475c5d68df4651899d678e
   ```

   The reported tree must exactly equal `b6e3a411...`. Do not synthesize a
   commit with the expected tree and call that recovery.
4. Validate the machine status before CPU work; require the fail-closed form
   before any scientific Slurm submission:

   ```bash
   .venv/bin/python scripts/validate_training_provenance.py
   .venv/bin/python scripts/validate_training_provenance.py --require-scientific-slurm-ready
   ```

   The second command is expected to fail today. Submission wrappers added in
   the Slurm tranche must call it and refuse to continue.

## Dataset, roles, subsets, and index gate

Canonical location:
`configs/training_selection/production_1m_20260812/`.

The generator allocates held-out roles first using SHA-256 rank of seed
`20260812`, category, task, source, and task-record hash. Whole source/task
shards are indivisible. Validation and sealed test each contain 10 shards
(50k); stress contains one shard from every category (35k); the remaining 173
shards (865k) form the training pool. No source or task may appear in two roles.

The held-out quotas for validation and test are each ccbar 2, charged 1, ddbar
1, mixed 1, ssbar 1, taupair 2, and uubar 2. Stress is one of each category.
The nested training subsets are:

| Selection | Train shards/events | Train category shards (ccbar, charged, ddbar, mixed, ssbar, taupair, uubar) | Use |
|---|---:|---|---|
| `train_035k.json` | 7 / 35k | 1, 1, 1, 1, 1, 1, 1 | GPU debug and broad ablation screening |
| `train_100k.json` | 20 / 100k | 4, 3, 2, 2, 2, 3, 4 | Candidate confirmation |
| `train_250k.json` | 50 / 250k | 11, 7, 4, 6, 4, 8, 10 | Promoted confirmation only |

The full training pool has category shard counts 15, 15, 13, 61, 13, 16, 40.
A future 500k or 865k selection may be described explicitly, but must not be
labeled category-representative: this reduced campaign is intrinsically
source-task sampled and skewed. Report macro-category metrics and production-
mixture-weighted metrics separately; never hide the denominators.

Regenerate and validate without reading Parquet payloads:

```bash
.venv/bin/python scripts/build_training_selection.py \
  --data-root /project/agkuhr/users/boyang/data/HyperTagging_uni \
  --output-dir configs/training_selection/production_1m_20260812
.venv/bin/python scripts/build_training_selection.py \
  --output-dir configs/training_selection/production_1m_20260812 \
  --validate-only
```

Generation checks the Parquet footer and schema, hashes the small sidecar and
marker, and checks the marker/sidecar contract. It records the marker-bound
Parquet SHA-256 instead of rereading 12 GiB. Before scientific training, build
a full event-level index for the exact selection (template; no current run):

```bash
.venv/bin/python scripts/build_dataset_index.py \
  --selection-manifest configs/training_selection/production_1m_20260812/train_035k.json \
  --output artifacts/index/train_035k.complete_only.json \
  --target-policy complete_only --scientific-mode
```

Do not use `--from-sidecars` for the final UID gate. Extend the full builder in
the next tranche to reject duplicate event UIDs, verify each record's source
and category against its selected publication, and emit UID/source count and
hash evidence. Fit and serialize normalizers only from explicit `train` roles;
freeze them for validation, test, resume, and all paired ablations.

## Corrections and focused implementation tests before GPU

Next-tranche code targets:

1. `src/hypertagging/training/pretrain_trainer.py`: progressive persisted
   phase durations; LR warmup; explicit-FP32 hyperbolic/VIC computations;
   fixed validation event sampler; objective-gradient ratios and 20:1 warning.
2. `configs/hyperbolic_pretrain_pilot.yaml`: `gpu_debug`, warmup, fixed
   validation cohort, and corrected dominance threshold.
3. `src/hypertagging/training/reconstruction_trainer.py`: fixed validation and
   rollout UID lists; rollout-based checkpoint gate; configurable scheduled
   sampling endpoint; all-level denominators and calibration output.
4. `configs/level_reconstruction.yaml`: immutable selection/index references,
   rollout primary metric, and non-CI validation budgets.
5. Dataset index: the UID/source/category scan gate described above plus
   ingestion sampling checks for tree validity, level consistency,
   daughter-summed p4, finite tensors, and per-shard anomalies.

Required CPU tests are implementation evidence only: phase boundary sequence,
warmup/resume equivalence, FP32 dtype under mocked autocast, gradient-ratio
warning, fixed validation UID stability, rollout checkpoint tie-breaking,
scheduled-sampling endpoints 0/10/25/50%, duplicate UID rejection, source-role
isolation, train-only normalizers, and exact resume/cursor behavior. Run focused
tests, then the existing dataset-index, data-contract, resume, and information-
boundary tests. Do not run the expensive full suite as a first response.

## Mandatory node-local V100 admission guard

The local Tesla V100 observation (125 MiB, 0%, no process) was timestamped and
is not permission for a later run. Immediately before every bounded microtest,
use `scripts/slurm/v100_local_admission.py admit` to collect exactly three
samples at 10-second intervals, followed only by its monitored `run` subcommand.
Each admission sample collects:

```bash
hostname
date --iso-8601=seconds
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
nvidia-smi pmon -c 1
fuser -v /dev/nvidia* 2>&1
```

Admission requires all three samples to show no compute process or device
owner, GPU and memory utilization at most 5%, memory use at most 512 MiB, and
temperature below 70 C. Any failed or ambiguous query or any user queue entry
is an ABORT. During an admitted test, the mandatory watchdog repeats telemetry
at least every 30 seconds. If unsafe activity appears, it signals and bounds
trainer shutdown and does not restart locally. Local use is limited to one GPU,
one process, at most 10 optimizer steps or 5 minutes (whichever comes first),
`gpu_debug`, and the 35k manifest/index. Admission alone never qualifies:
scientific rendering requires a canonically hashed, admission-bound completion
with normal trainer exit status zero. Never use local V100 for screening,
confirmation, or final evaluation.

## Slurm design and dry-run boundary

No job is submitted in this tranche. At execution time, inspect live state:

```bash
/opt/slurm/bin/squeue -u boyang.yu -o '%.18i %.9P %.24j %.8T %.10M %.6D %R'
/opt/slurm/bin/sinfo -p inter -N -o '%N %T %G %C %m %e'
/opt/slurm/bin/scontrol show node gar-cn-etp01
/opt/slurm/bin/scontrol show node gar-cn-etp02
/opt/slurm/bin/sbatch --help
```

The dedicated `agkuhr` V100 nodes were DRAIN during audit; do not target them
until they are live and authorized. Use partition `inter` with exactly one of
the site-advertised GRES strings. Initial templates are:

```bash
# H200 candidate, verify GRES spelling from live sinfo/site policy
#SBATCH --partition=inter
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue

# V100 alternative; never request generic gpu:1
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
```

The site may run an older Slurm version: verify `--gres`, `--signal`, `--requeue`,
`--test-only`, and output-token syntax against `/opt/slurm/bin/* --help` and site
policy. A generic `--gres=gpu:1` is forbidden because it can land on the wrong
GPU type. Start with 8 CPUs, 64 GiB RAM, and one GPU; use 1 hour for microtest,
8–12 hours for 35k screening, and request 24 hours for 100k only after measured
throughput supports it. Estimates are planning bounds, not measured runtime.

The job must activate a reviewed frozen H200/V100-compatible environment (the
current CPU test environment is not a GPU environment), set deterministic
seeds, stage no mutable source, and write to
`artifacts/runs/<experiment>/<seed>/<slurm_job_id>/`. Capture rendered config,
manifest/index/provenance hashes, `pip freeze` or environment lock, Git
commit/tree/status, `nvidia-smi`, Slurm environment, stdout/stderr, metrics
JSONL, checkpoints, cursor/RNG/scheduler/scaler state, and termination reason.
On USR1, atomically checkpoint then exit with the site-approved requeue code.

Before submission, run `bash -n`, configuration rendering, CPU contract tests,
manifest/index validation, provenance fail-closed validator, and—only if
supported—`sbatch --test-only`. Stop at printed command/dry-run until a later
explicit submission authorization.

## Staged training schedule and promotion gates

All stages use fixed split manifests, train-only normalizers, single GPU, AMP
with explicit FP32 geometry/VIC, gradient clipping, exact checkpoint/resume,
and seeds `20260812`, `20260813`, `20260814`. Event budgets, not vague epochs,
are authoritative; record actual steps, events, wall time, GPU-hours, peak
memory, and discarded/overflow events.

### Pretraining

| Stage | Data/model | Budget | Validation | Seeds | Go gate |
|---|---|---:|---:|---|---|
| CPU smoke | tiny fixture / `tiny_cpu` | 2–10 steps | fixture | 20260812 | Deterministic, finite, exact resume; software only. |
| GPU microtest | 35k / `gpu_debug` | ≤100 steps or 15 min | 256 fixed events | 20260812 | No OOM/NaN, stable throughput, FP32 diagnostics, checkpoint/resume. |
| Screening | 35k / `gpu_debug` | 70k seen events (2 pool passes) | fixed 2k validation | 20260812; paired ablations use same seed | Key validation objectives improve from step-0/frozen baselines; no representation collapse; no unexplained >20:1 objective-gradient dominance; memory headroom ≥20%. |
| Candidate | 100k / smallest convergent preset | 300k seen events (3 passes) | fixed 5k validation | all 3 | All seeds finite; median and each-seed trend improve; topology/parent ranking beats preregistered random/constant baseline; seed spread reported; exact resume passes. |
| Promoted confirmation | 250k / unchanged candidate | 750k seen events (3 passes), with early stop | full 50k validation in final checkpoint sweep | all 3 | Candidate gains persist with paired confidence intervals; no category loses catastrophically; compute scaling justified. |

Use progressive phases: topology/parent/anticollapse, then distance/radius, then
channel contrast, then candidate correctness/hard negatives, with cumulative
objectives unless a preregistered stage disables one. Warmup is 5% of planned
optimizer steps, capped after measured tuning. Early stop after three complete
fixed-validation evaluations without improvement in the preregistered
aggregate, provided each evaluation spans at least 0.25 training-pool passes.
Do not advance to 1M or 10M unless the 250k result is stable, scientifically
useful, and scaling analysis predicts material information gain.

### Reconstruction

Initialize from the selected pretraining checkpoint. The first pass predicts
leaf PID and rebuilds runtime PID/p4/relations; the second pass predicts mother
type, cardinality, daughter pointers, and confidence. Mothers use deterministic
daughter-summed p4. Truth topology remains supervision only and must never
enter inference-compatible inputs.

| Stage | Data/model | Budget | Validation/rollout | Seeds | Go gate |
|---|---|---:|---:|---|---|
| CPU smoke | fixture / `tiny_cpu` | 2–10 steps | all fixture rollout | 20260812 | Inference-boundary tests, finite Hungarian loss, exact resume. |
| GPU microtest | 35k / `gpu_debug` | ≤100 steps or 15 min | 256 / 128 fixed | 20260812 | No OOM/NaN; free rollout completes; checkpoint/resume. |
| Screening | 35k | 105k seen events (3 passes) | fixed 2k / 1k | 20260812 | Rollout composite improves over untrained/frozen and greedy baseline; all-level denominators nonzero; calibration output valid. |
| Candidate | 100k | 300k seen events | fixed 5k / 2k | all 3 | Paired improvement in exact event/tree, level-1 set F1, valid topology, parent/daughter accuracy, and rollout NLL/Brier/ECE without a material p4 or partial-target regression. |
| Confirmation | 250k | 750k seen events, early stop | full 50k validation; 10k rollout during training, full at checkpoint sweep | all 3 | Rollout-selected checkpoint wins across seeds and category/channel slices; bounded beam diagnostic does not reveal a resolver failure that invalidates conclusions. |

Checkpoint selection is lexicographic: reject any checkpoint failing finite,
topology-validity, source-exclusivity, or calibration gates; among survivors,
maximize a preregistered rollout composite of exact reconstruction, level-1 set
F1, valid topology, and calibrated confidence. Teacher-forced loss breaks ties
only. Store every component so the composite cannot hide a regression. Numeric
performance thresholds must be set from the frozen step-0 and first 35k
baseline; no unmeasured absolute performance is claimed here.

## Prioritized ablations without combinatorial explosion

Use paired seeds, identical source manifests, validation UIDs, initialization,
event budget, and evaluation code. Screening is one seed on 35k; only effects
with a clear paired direction and no safety regression advance to 100k with all
three seeds. Confirm at 250k only the final default and at most two unresolved
scientific controls. Never cross-product ablations.

| Priority | Question / arms | Screen | Confirmation rule |
|---:|---|---|---|
| 1 | Representation ladder: flat vs heterogeneous Euclidean vs shared Poincaré | 35k, one paired seed | Advance ladder winner plus scientific baseline if rollout/topology effect is consistent. |
| 2 | Physical relation bias off/on; then optional hyperbolic relation refinement off/on | Sequential, not 2×2 | Test refinement only on physical-bias winner. Default remains physical bias on, refinement off unless confirmed. |
| 3 | Scheduled sampling endpoint 0/10/25/50% | 35k | Confirm best and 0% control at 100k/all seeds. |
| 4 | Leaf PID: soft, temperature, straight-through; hard/canonical pion diagnostic | 35k | Confirm best differentiable mode; hard/pion never promoted solely on teacher-forced loss. |
| 5 | KLM full vs masked, with matched/unmatched KLM cohorts | 35k | Advance only if full KLM helps relevant cohort without broad regression. |
| 6 | Channel memory off/on and pooling (FSP-only vs chosen pooled form) | Two sequential tests | Pooling comparison only if memory is useful; require rare-channel gain and bounded memory. |
| 7 | Level encoding none/Euclidean/bounded tangent and radius target alternatives | Sequential on winner | Confirm only one encoding and one radius target. |
| 8 | Query repulsion off/default/stronger and first-level type-relation bias | 35k diagnostics | Promote only for first-level duplicate/cardinality improvement without calibration loss. |
| 9 | Complete-only vs reconstructable-partial | Final candidate only | Separate denominators and never combine policy metrics. |
| 10 | Greedy resolver vs evaluation-only bounded beam/set packing | No training duplication | Diagnose search error; do not use beam to choose training checkpoints unless preregistered. |

Every ablation reports optimizer events, GPU-hours, peak memory, throughput,
parameters, checkpoint bytes, and evaluation CPU/GPU time, plus exact event,
level-1/all-level set precision/recall/F1, type/cardinality/pointer accuracy,
tree validity, p4 residuals, free-rollout failure modes, PID metrics, NLL/Brier/
ECE, macro category, production-weighted category, rare channel, multiplicity,
depth, KLM matched/unmatched, and complete/partial denominators.

## Final held-out evaluation and reproducibility

The 50k test pool is sealed: do not inspect its model metrics, tune thresholds,
select checkpoints, or choose ablations on it. Unseal once for the frozen final
model/config/evaluator. Evaluate every event and report bootstrap confidence
intervals clustered by source shard, paired against declared baselines. Also
evaluate the separate 35k stress pool by category; stress is diagnostic and is
not merged into the test headline.

The final artifact bundle contains source commit/tree/tag and clean status,
production provenance status, inventory/roles/selection/index hashes, UID gate
report, exact train/validation/test/stress source lists, train-only normalizer,
rendered configs, seeds, environment/container digest, Slurm script and job
metadata, logs, checkpoints and checksums, exact resume state, predictions by
event UID, evaluator version, all metric denominators, bootstrap samples,
category/channel/KLM/partial slices, compute accounting, and a model card that
states the reduced-data skew and unsupported claims.

## Critical path and STOP conditions

Critical path: merge/tags → recover and verify production object → immutable
roles/manifests (done) → full UID/source index → trainer corrections and CPU
tests → guarded local microtest or Slurm dry-run → 35k pretraining → 35k
reconstruction/ablations → 100k three-seed confirmation → optional 250k
confirmation → sealed test → only then consider larger data or DDP.

STOP immediately on any of the following:

- dirty/unarchived experiment source; manifest/index/provenance hash mismatch;
  missing source object/tree proof for scientific submission; duplicate UID;
  source/task leakage; validation/test used for normalization or tuning;
- GPU guard ambiguity or competing workload; wrong/unspecified GRES; drained or
  unauthorized node; environment/GPU incompatibility;
- NaN/Inf, invalid Poincaré points, collapsed variance, exploding gradients,
  unexplained objective dominance, OOM without ≥20% planned headroom, corrupted
  checkpoint, non-exact resume, or changing data/normalizer hash on resume;
- truth topology or truth PID leaking into inference-compatible context;
  recursive-source duplication; daughter-sum p4 contract failure; nonzero
  ingestion anomaly beyond a preregistered tolerance;
- no reproducible learning signal at 35k, inconsistent three-seed effect at
  100k, rollout degradation hidden by teacher-forced loss, severe category/
  rare-channel/KLM regression, or compute scaling unsupported by measured
  throughput and information gain.

When stopped, retain artifacts, label the run failed/diagnostic, identify the
smallest CPU or bounded microtest that discriminates causes, and do not promote
or silently extend the budget.
