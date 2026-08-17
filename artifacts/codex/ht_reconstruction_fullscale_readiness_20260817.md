# HyperTagging Reconstruction Full-Scale Readiness Report (20260817)

Scope: object-positive-weight 12 / pointer-positive-weight 16 full-scale reconstruction launch readiness for `pretrain-035k` using source checkpoint step `3282`.

## Frozen evidence-backed settings

- Source checkpoint: `3282`, SHA256 `5afdae8ac943163631499674297d6f15986c825f00ccff2d39389f22ff383c79` (from midpoint calibration artifacts).
- Source checkpoint mutation policy: `source_checkpoint_mutation = "forbidden"`.
- Authoritative result commit/tag: `3bd394d`, `pretrain-035k-transfer-positive-weight-calibration-midpoint-v100-results-20260817`.
- Frozen weights: `object_positive_weight = 12`, `pointer_positive_weight = 16`.
- Sealed-test policy: `sealed_test_role_access = forbidden`.
- Roles: `training_role = "train"`, `evaluation_role = "validation"` in transfer probe contracts and preregistration.
- Corrected runtime-only normalization: runtime normalizer must be identity-scaled for zero-training-support slots; this correction is explicitly documented and must remain unchanged for launch.
- Resource policy (frozen): `H100` is preferred; `V100` is allowed only as fallback.
- Checkpoint and gating precedent exists only in transfer-probe contracts at `max_steps = 100` (not full-scale):  
  - `artifacts/studies/contracts/pretrain-035k-transfer-step3282-object12-pointer16-calibration-20260816.json`
  - `artifacts/studies/contracts/pretrain-035k-transfer-step3282-object12-pointer16-calibration-v100-results-20260817.json`
  - `artifacts/codex/ht_positive_weight_midpoint_readiness_20260817.json`

## Required contract gates (scientific/control gates)

- **Validation gate**: 512 fixed validation events, 256 rollout validation events, and 100-step cadence are explicitly documented only for the 100-step frozen-encoder calibration probe.
- **Finite-gradient gate**: full scientific submission path requires finite-gradient checks from execution result/receipt flow (probe run evidence is available for midpoint only).
- **Checkpoint gate**: source checkpoint must be immutable and explicitly bound by step + SHA-256.
- **Resource gate**: accelerator must match allowed policy (`gpu:h100nvl:1` preferred, optional `gpu:v100:1` fallback in probe renderer/verifier).

## Unresolved parameters blocking executable full-scale contract

`NOT_EXECUTABLE_UNTIL_PARAMETERS_RESOLVED`

- No tracked contract template exists in `artifacts/studies/contracts/` for **full-scale reconstruction launch** with a documented, execution-bound schema/validator for object12/pointer16.
- No tracked launch contract maps `object12_pointer16` to full-scale training horizon or validation cadence beyond the probe profile.
- No tracked contract fields define full-scale output/result contract paths for the reconstruction transfer object/pointer run.
- No tracked launch script path currently renders a full-scale reconstruction contract for this arm (existing renderer/verifier only supports transfer-probe contract versions and `max_steps`-100 profiles).

## Evidence paths

- `artifacts/studies/contracts/pretrain-035k-transfer-step3282-object12-pointer16-calibration-20260816.json`
- `artifacts/studies/contracts/pretrain-035k-transfer-step3282-object12-pointer16-calibration-v100-results-20260817.json`
- `artifacts/codex/ht_positive_weight_midpoint_readiness_20260817.json`
- `artifacts/codex/ht_positive_weight_midpoint_audit_20260817.md`
- `artifacts/codex/ht_positive_weight_calibration_preregistration_20260816.json`
- `scripts/slurm/verify_reconstruction_transfer_probe_contract.py`
- `scripts/slurm/render_reconstruction_transfer_probe_job.py`
- `docs/training_execution_plan_20260812.md`
- `docs/training.md`

## Checklist

- Data-role constraints: **preserve** (train/validation separation, no sealed-test role access).
- Source and checkpoint immutability: **preserve**.
- H100-primary, V100-fallback policy: **documented**.
- Runtime-only normalization correction: **preserved**.
- Full-scale execution contract: **missing**.
