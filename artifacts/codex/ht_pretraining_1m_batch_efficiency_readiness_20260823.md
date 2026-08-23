# Production-1M phase-3 batch-efficiency readiness — 2026-08-23

Status: programming and CPU/static work complete; GPU calibration and production submission remain pending. `submission_performed=false`.

The immutable parent authorization is `de0c54f762aa8ac46484ae28bfb950d035f4e51b` / `ht-pretraining-1m-phase3-execution-authorization-20260823-final`. Its `submission_authorized=true` is the parent operator authorization. This readiness artifact deliberately records the separate gates `gpu_calibration_completed=false` and `production_submission_authorized=false`. The structural provenance exception remains fail-closed: `scientific_slurm_submission_allowed=false`.

## Contract and diagnosis

The source checkpoint is step 54,064 at:

`artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/15933802/checkpoint-step-54064.pt`

with SHA-256 `997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d`. The scientific boundary is exactly 865,024 completed and 865,024 remaining presentations out of 1,730,048. The new progress contract makes one virtual step equal to 16 presentations. Physical optimizer updates remain a distinct counter.

The old trainer rejected a changed batch size as a data-order mismatch and used optimizer steps for phase, validation/checkpoint cadence, and the LR scheduler. The implementation adds:

- explicit presentation progress and exact divisibility checks;
- migration from the legacy step-16 checkpoint while preserving model, optimizer moments, RNG, fixed validation UIDs, and scheduler state;
- virtual-step phase and cosine/warmup positions;
- epoch-tail carry so a larger batch never emits a partial scientific update;
- exact future milestones 67,580, 81,096, 94,612, and 108,128, without repeating the four completed milestones;
- validation of the exact fixed 5,000-event cohort with 79 batches at batch 64 (versus 313 at batch 16);
- device-specific precision: H100 NVL BF16 with GradScaler disabled after BF16 validation, V100 FP16 with GradScaler enabled and BF16 forbidden.

The legacy checkpoint's raw `curriculum_phase_cursor.events_completed=865016` and epoch-relative stream cursor are retained historical telemetry. They reflect the old source-split terminal-tail behavior. The authorized migration does not reinterpret that raw counter as scientific progress: it binds the checkpoint's step 54,064 to `54,064 × 16 = 865,024`, reconstructs the deterministic epoch position from that presentation count, and carries source tails across epoch boundaries. This is why the migration is explicit and fail-closed rather than a silent changed-batch resume.

The audit covered `pretrain_trainer.py`, `learning_rate.py`, `checkpointing.py`, `presentation_progress.py`, `data_module.py`/`streaming.py`, `fixed_validation.py`, the device precision profiles, the new renderer/verifier, the existing Slurm contract verifier, and runtime telemetry in the trainer/monitor/calibration paths.

No LR scaling is asserted. Both candidates preserve `0.0005` at the resume boundary. Any later LR change must be explicit, bounded, and supported by the stability receipt.

## Candidate evidence

H100 NVL batch 16 has measured peak memory 4,459 MiB, peak utilization 54%, and finite throughput around 8 events/s. Batch 32 and 64 are therefore bounded candidates, not measured guarantees. Existing V100 evidence is diagnostic-only (peak 554 MiB and 11% utilization for a reduced diagnostic); it does not establish production batch memory safety. Both profiles consequently use the ladder `[32, 64]`, with 64 as the current candidate and calibration required before selection.

The selection policy is earliest expected completion from measured train-role throughput, tie-broken by lower peak memory and then H100 NVL. H100 and V100 calibration is sequential; exactly one profile can be selected and exactly one scientific production lineage can be rendered/submitted.

## Later Spark commands

Run these in separate fresh interactive/in-allocation sessions, H100 first and V100 second. The harness performs exact-GRES preflight, a synthetic fixture memory/throughput probe, a train-role-only stability pilot from a copy of the checkpoint, finite-loss/raw-gradient/objective-dominance/LR checks, and a self-hashed receipt. It does not call Slurm or submit anything.

```text
HT_PHASE3_CALIBRATION_ACTIVE=1 HT_PHASE3_ALLOCATION_GRES=gpu:h100nvl:1 HT_PHASE3_FRESH_PREFLIGHT_TOKEN=<fresh-token> uv run python scripts/run_phase3_batch_efficiency_calibration.py --profile h100nvl --batch-size 64 --checkpoint-copy artifacts/codex/calibration/h100nvl/checkpoint-step-54064.copy.pt --output artifacts/codex/calibration/h100nvl/receipt.json --pilot-metrics artifacts/codex/calibration/h100nvl/metrics.jsonl --execute-pilot --pilot-command -- <later train-role-only pilot command>

HT_PHASE3_CALIBRATION_ACTIVE=1 HT_PHASE3_ALLOCATION_GRES=gpu:v100:1 HT_PHASE3_FRESH_PREFLIGHT_TOKEN=<fresh-token> uv run python scripts/run_phase3_batch_efficiency_calibration.py --profile v100 --batch-size 64 --checkpoint-copy artifacts/codex/calibration/v100/checkpoint-step-54064.copy.pt --output artifacts/codex/calibration/v100/receipt.json --pilot-metrics artifacts/codex/calibration/v100/metrics.jsonl --execute-pilot --pilot-command -- <later train-role-only pilot command>

uv run python scripts/select_phase3_batch_efficiency_profile.py --h100-receipt artifacts/codex/calibration/h100nvl/receipt.json --v100-receipt artifacts/codex/calibration/v100/receipt.json --output artifacts/codex/calibration/selection-20260823.json --authorize-production

uv run python scripts/render_phase3_batch_efficiency_production_contract.py --selection artifacts/codex/calibration/selection-20260823.json --expected-git-sha <clean-calibration-commit> --expected-git-tag ht-pretraining-1m-phase3-batch-efficiency-implementation-20260823 --output artifacts/slurm/ht-pretrain-1m-phase3-selected-20260823.job-contract.json

uv run python scripts/verify_phase3_batch_efficiency_contract.py artifacts/slurm/ht-pretrain-1m-phase3-selected-20260823.job-contract.json

/opt/slurm/bin/sbatch --account=others --partition=inter --gres=<selected-exact-gres> --cpus-per-task=8 --mem=64G --time=2-00:00:00 --signal=B:USR1@300 --requeue --job-name=ht-pretrain-1m-phase3-selected-20260823 --export=NIL /home/b/Boyang.Yu/HyperTagging_uni/HyperTagging/scripts/slurm/train_one_gpu.sbatch /home/b/Boyang.Yu/HyperTagging_uni/HyperTagging/artifacts/slurm/ht-pretrain-1m-phase3-selected-20260823.job-contract.json
```

The final command is recorded for the later authorized operator only; it was not run in this programming session. The existing historical attempt root must not be reused, and no second production lineage may be submitted.

## CPU/static verification

Focused tests cover virtual presentation arithmetic, divisibility, milestone/phase mapping, scheduler continuity, checkpoint migration, stream carry/resume, precision policy, calibration limits, receipt/contract tamper rejection, queue uniqueness, and no-overwrite behavior. No source batch payload, sealed-test payload, stress payload, Slurm command, or scientific training run was used here.

The machine-readable companion is [ht_pretraining_1m_batch_efficiency_readiness_20260823.json](ht_pretraining_1m_batch_efficiency_readiness_20260823.json). Device configs are [H100 NVL](../../configs/slurm/pretrain_1m_phase3_batch_efficiency_h100nvl_20260823.yaml) and [V100](../../configs/slurm/pretrain_1m_phase3_batch_efficiency_v100_20260823.yaml); the initial selection manifest is [here](ht_pretraining_1m_phase3_batch_efficiency_selection_20260823.json).
