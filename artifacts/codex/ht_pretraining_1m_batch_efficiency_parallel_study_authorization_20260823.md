# Phase-3 batch-efficiency parallel-study authorization — 2026-08-23

Status: programming and CPU/static amendment complete. `gpu_calibration_completed=false`, `production_submission_authorized=false`, and `submission_performed=false`. The parent implementation commit/tag and old readiness/selection artifacts are preserved.

The authoritative plan is [ht_pretraining_1m_phase3_parallel_study_v1.json](../../configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json), SHA-256 `b3a8c3936426f3da89623600a3948b3bca05a89c785bd1501990d2ac127cd836`. It binds parent implementation `aa54b4efdf99f7853df480a1dded44214faac770` / `ht-pretraining-1m-phase3-batch-efficiency-implementation-20260823`.

## Authorized calibration matrix

Exactly four distinct, non-production, train-role-only tuples may run concurrently, subject to queue/resources. The coordination registry admits at most four active jobs. Every tuple has an immutable ID, hypothesis, exact GRES, batch size, precision/scaler policy, checkpoint-copy path, disjoint output/attempt roots, metrics path, receipt path, and tuple hash.

| ID | GRES | Batch | Precision/scaler | Hypothesis |
| --- | --- | ---: | --- | --- |
| `ht3-cal-h100nvl-b32-20260823` | `gpu:h100nvl:1` | 32 | BF16 / scaler disabled | H100-NVL batch 32 bounded stability |
| `ht3-cal-h100nvl-b64-20260823` | `gpu:h100nvl:1` | 64 | BF16 / scaler disabled | H100-NVL batch 64 bounded stability |
| `ht3-cal-v100-b32-20260823` | `gpu:v100:1` | 32 | FP16 / scaler enabled | V100 batch 32 bounded stability |
| `ht3-cal-v100-b64-20260823` | `gpu:v100:1` | 64 | FP16 / scaler enabled | V100 batch 64 bounded stability |

Duplicate IDs, tuple keys, tuple hashes, receipt hashes, checkpoint-copy paths, output/attempt roots, or ambiguous owners fail closed. Calibration cannot authorize or run production, and production rendering rechecks the active registry.

Each job performs a fresh exact-GRES one-GPU preflight, synthetic fixture benchmark, and bounded 256-step/900-second synthetic train-role stability pilot. Validation tuning, sealed/stress access, CPU scientific training, source checkpoint mutation, and scheduler submission are forbidden. Receipts must be self-hashed and terminal `healthy`.

## Later Spark commands

Run the following four commands in separate fresh interactive/in-allocation Spark sessions. They are intentionally direct Python invocations; there is no scheduler command here. Set `HT_PHASE3_QUEUE_DELAY_SECONDS` from each fresh allocation record and use a fresh preflight token for each session. The four commands may run concurrently.

```bash
export HT_PHASE3_OWNER=sole-authorized-phase3-follow-up-programming-operator
export HT_PHASE3_QUEUE_DELAY_SECONDS=<fresh-allocation-queue-delay-seconds>
```

Use the corresponding exact command stored in the machine-readable artifact under `later_spark_commands`: `h100nvl_batch32`, `h100nvl_batch64`, `v100_batch32`, and `v100_batch64`. After all four commands exit with terminal healthy receipts, collect the exact configured set:

```bash
uv run python scripts/aggregate_phase3_batch_efficiency_receipts.py --study-plan configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json --output artifacts/codex/calibration/parallel-study-20260823/receipt-aggregation.json
uv run python scripts/select_phase3_batch_efficiency_profile.py --study-plan configs/batch_efficiency/ht_pretraining_1m_phase3_parallel_study_v1.json --receipt-aggregation artifacts/codex/calibration/parallel-study-20260823/receipt-aggregation.json --output artifacts/codex/calibration/parallel-study-20260823/selection.json --authorize-production
```

Aggregation and selection fail closed on missing/nonhealthy receipts, nonfinite values, objective-dominance failure, queue/throughput omissions, checkpoint immutability failure, active calibration, or any set mismatch. Expected completion is `queue_delay_seconds + remaining_presentations / measured_train_throughput`; no validation metric is used.

The current recovery lineage defaults to exactly one explicitly enumerated production continuation. Multiple production variants are not authorized by this amendment; duplicate production contract identities are rejected. The structural provenance gate remains `scientific_slurm_submission_allowed=false`.

## Verification record

The new readiness verifier is [verify_phase3_batch_efficiency_readiness.py](../../scripts/verify_phase3_batch_efficiency_readiness.py), and the schema is [ht_pretraining_1m_phase3_parallel_study_authorization_v1.schema.json](../../schemas/ht_pretraining_1m_phase3_parallel_study_authorization_v1.schema.json). Focused CPU/static tests cover max-four admission, tuple/hash uniqueness, shared-root rejection, duplicate production-contract rejection, production blocking during active calibration, exact receipt aggregation, and exactly-one default production resume.

No Slurm command was run, no GPU allocation was used, no scientific training was run, and no sealed/stress/source payload contents were opened.
