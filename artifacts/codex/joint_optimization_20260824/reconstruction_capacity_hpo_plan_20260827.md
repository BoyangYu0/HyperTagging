# HyperTagging reconstruction capacity/HPO plan — 2026-08-27

Status: **planning only; all execution, submission, scheduler, throughput, scientific-training and production authorizations are false.**

## Bound evidence

This plan binds the completed aggregate-only q32 terminal report at commit `cd5e9942e05725ccf4e20b5ce6d43e93eeb25db4` (`q32_16036155_terminal.json` SHA256 `0bbb91703a5b79a006f79a105b13281fdc60c392579da317694a31d27b2fdd9f`; Markdown SHA256 `45e2d0c11927d532fac6381c4a466584274a208ffe682da49021b191a5f61a7d`). q32 used one H100-NVL, B64, 512 optimizer steps/32,768 presentations, frozen encoder and transferred leaf head, BF16 with scaler off. It completed finite with zero fallback, peak GPU utilization 39%, peak memory 5,045 MiB, peak temperature 38 C, validation loss 1.8195147991, edge-F1/full exact 0.031, and tree/P4 validity 1.

The q32 evidence remains aggregate-only and non-promotable. No live relation-bias artifact was inspected.

The plan also binds the inert capacity runnable/clearance commits `f8ae47091aba8dbb77e4c12e142f11dc6529f684` and `1171dff3a7bab1a3edc123000b5033e125551021`, plus conditional-HPO verifier hardening commit `4ee5d819e5586dcfdec0848d2e91b917494484cc`. Their bound JSON hashes are recorded in the companion machine-readable artifact.

## Fail-closed prerequisite

Nothing below may execute until controller-approved per-event q32 and relation-bias evidence is complete and matched by data order, seed and validation cohort. Promotion requires edge-F1 delta at least 0.01, 95% paired bootstrap with Holm-adjusted lower bound above zero, all tree/reconstruction/leaf/P4 rates exactly 1, zero fallback, and passing teacher-forced and efficiency guards. Sealed-test and stress roles remain forbidden.

If and only if relation bias passes, fix it on for every later arm; otherwise fix it off. It is not an HPO dimension.

## Ranked recommendation

1. **Frozen encoder, B128:** retain the existing d128/l4 architecture (2,135,433 total parameters; 1,322,658 encoder; 807,486 trainable when encoder and leaf head are frozen). Use 65,536 presentations/512 updates. This corrects obvious H100 underutilization while limiting exposure to about 1.87 passes over train_035k.
2. **Decoder-only capacity:** preserve the exact transferred width-128 encoder and leaf head; add a verified encoder-to-decoder adapter and configurable decoder depth. First target 4–6M total and 2.7–4.7M trainable parameters. Only after that passes, consider 9–12M total.
3. **Separately preregistered staged unfreezing:** use 32,768 frozen presentations followed by 32,768 unfrozen, encoder LR multiplier 0.05, paired against the same-seed fully frozen counterpart.
4. **Width scaling only with a matching checkpoint:** code-only counts are 6,219,801 parameters at d192/l6 and 10,992,297 at d256/l6. The current d128 checkpoint cannot support an honest frozen-transfer comparison at these widths; partial shape-mismatched loading is non-promotable.

Use a throwaway B64/B128/B256 calibration. Default to B128. B256 requires finite/no-OOM evidence, allocated ≤70 GiB, reserved ≤76 GiB, ≥10% unreserved memory, median utilization ≥60%, median throughput ≥1.5× B64 and p10 throughput ≥1.25× B64. There is no automatic fallback.

## HPO and validation

Use ASHA presentation rungs 16,384 → 32,768 → 65,536, with 65,536 as the hard cap. At B128 search LR 0.0007/0.0014, dropout 0.10/0.20, explicit AdamW weight decay 0.001/0.01, and scheduled-sampling maximum 0.10/0.25 in a preregistered 12-arm fractional design. Warm up for 10% of updates, capped at 64, then cosine-decay to a 0.1 LR ratio. The present reconstruction code omits explicit weight decay, so q32 inherited AdamW's 0.01 default; the repair must make this explicit.

Bind one validation manifest and event-UID cohort across arms and seeds. Run teacher-forced validation every 8,192 presentations and rollout at each rung; retain final 2,000 teacher-forced/1,000 rollout denominators. Do not metric-stop before 32,768 presentations. Rank only by validation edge-F1, full exact, tree-edit distance, efficiency, lower parameter count and lower LR. Train loss is diagnostic only.

Every ranked checkpoint must remain finite with tree/reconstruction/leaf/P4 validity exactly 1, zero fallback, and passing teacher/efficiency guards. Final promotion is same-seed paired, requires edge-F1 delta ≥0.01 and a 95% paired-bootstrap Holm-adjusted lower bound >0 across all final candidate comparisons.

The bounded upper allocation is 45 GPU primaries plus 45 after-any controllers, nominally 35–37 GPU-hours with a fail-closed cap of 45 GPU-hours. This is an estimate, not authorization.

## Reuse and parallel work

Extend and repair the v2r1 conditional harness; do not create a duplicate. Add explicit batch/rungs/architecture/optimizer/unfreezing/seed fields, replace the hardcoded step-1352 pointer with an immutable validation-plus-transfer-selected checkpoint binding, add decoder adapter/depth contracts, explicit reconstruction weight decay, exact parameter/transfer/resume/cohort bindings, and negative verifier tests.

Harness/schema tests, decoder-adapter code, parameter-count fixtures, inert configs, cohort binding and throughput-contract review may proceed in parallel with the pretraining ladder as code-only planning. Scheduler polling, submission/cancellation, live relation-bias reads, payload access and scientific training remain forbidden.

This artifact does not authorize any job or mutation of scientific state.
