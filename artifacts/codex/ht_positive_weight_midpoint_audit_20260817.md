# Positive-weight midpoint calibration audit — 2026-08-17

- Failed bootstrap attempts: 15801263 (FAILED, bootstrap contract hash mismatch, 0 scientific steps), 15801367 (FAILED, bootstrap preregistration hashed-input mismatch, 0 scientific steps).
- Cancelled pending H100: 15801839 (CANCELLED, pending H100 priority; V100 fallback explicitly authorized, 0 scientific steps).
- Successful V100 midpoint result: 15802021 (COMPLETED, 100 steps, elapsed 1,018 s).

## Agreement checks
- Job id across submitted contract, receipt, and result: `15802021`.
- Contract SHA256 across inputs: `ea6721e943f2fdea315ca412597c33b6a234a69a75c797cf81fdc51ad10fe16d`.
- Checkpoint step and hash consistency: 3282 and `5afdae8ac943163631499674297d6f15986c825f00ccff2d39389f22ff383c79`.
- Optimizer steps: 100; test rows: 0; restart count: 0.
- Source checkpoint unchanged: true.

## Metrics and gates (from readiness verdict)
- `finite_metrics_and_losses`: true
- `finite_loadable_checkpoint_family`: true (8/8 loadable, all tensors finite)
- `query_utilization`: 0.36962890625 (>= 0.10)
- `closure_denominator`: 222.0 (>= 32)
- `closure_rate`: 1.0 (== 1.0)
- `tree_validity`: 1.0 (== 1.0)
- `level_1_object_recall`: 0.9921875 (>= 0.50)
- `predicted_edge_f1`: 0.046875 (>= 0.046875)
- `level_1_object_precision`: 0.0995257045774347 (> 0.05965839272548538)
- `duplicate_candidate_rate`: 0.4107818108818002 (< 0.42221732761548914)
- `source_checkpoint_integrity`: true
- `roles_and_restart`: true
- `full_scale_submission_authorized`: false (unchanged)

## Tensor audit and checkpoint family (15802021)
- File: `artifacts/studies/reconstruction-transfer-probe/pretrain-035k-transfer-step3282-object12-pointer16-calibration-20260816-v100-fallback-20260817/15802021/tensor-finiteness-audit.json`
- all_reported_numbers_finite: true
- all checkpoints loadable: true
- all model tensor checks finite: true
- checkpounts: best.pt, best_rollout_edge_f1.pt, best_rollout_tree_validity.pt, best_teacher_forced.pt, checkpoint-step-100.pt, checkpoint.pt, latest.pt, signal-checkpoint.pt

## Timing and resources
- Started/ended: 2026-08-17T11:28:13+00:00 to 2026-08-17T11:45:11+00:00.
- Elapsed seconds (verdict): 1,018.
- Peak GPU utilization: 21 %
- Peak GPU memory: 524 MiB
- Peak node temperature: 39 °C
- Max RSS: 2,614,776 KiB

## Data-access gates
- `sealed_test_role_access` is `forbidden` in submitted contract, runtime result payload, and readiness verdict.
- `source_checkpoint_mutation` is forbidden in the submission contract.
