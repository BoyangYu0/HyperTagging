# Reconstruction post-terminal exact decision plan v1

Status: **planning clearance only; all execution and submission authorization is false**.

This plan governs the decisions after reconstruction primaries `16030125` and `16030126`, their after-any controllers `16030127` and `16030128`, and pretraining job `16028845` have all reached terminal state. It authorizes no polling, submission, payload access, training, or sealed-test use. Its machine-readable companion is the normative source for exact lists and thresholds.

## Decision boundary

The r8 transaction itself is clear and complete: exactly four reconstruction jobs were submitted and jointly released once, with no retry or alternate command. That does not clear any scientific promotion. The next action is evidence collection only after terminal completion.

Proceed only if all five jobs have unique, exact scheduler accounting rows with canonical user `boyang.yu` / UID `12184`, expected names, `COMPLETED`, and `ExitCode=0:0`. The two controllers must bind the exact parent IDs. Preserve raw command argv, stdout, stderr, exit codes, normalized rows, local terminal receipts, after-any receipts, results, telemetry, checkpoints, and source manifests. Every file receives an absolute path, byte length, SHA256, producing job/contract identity, and role in a canonical manifest that is itself hashed and committed. Any query error, nonempty stderr, duplicate, mismatch, missing artifact, nonfinite value, source mutation, or unsupported restart stops the program; no retry is implied.

For each reconstruction arm require the primary receipt, reconciled controller receipt, result, metrics, teacher-forced and rollout event-score artifacts, step-512 checkpoint, telemetry JSONL and recomputed summary, and identical before/after source manifests. For pretraining require the step-1352 checkpoint and its checkpoint, architecture, config, data, normalizer, and state-spec hashes; exact 5,000-event validation receipt; finite audit; terminal/runtime/telemetry receipts; dataset and selection manifests. Train loss is documentary only and may not select anything.

## Paired q32 versus relation bias

The two arms must both complete optimizer step 512 and 32,768 presentations with seed `20260826`, 2,000 teacher-forced validation events, 1,000 rollout events, identical data order, seed manifest, validation UID order, source checkpoint, optimizer schedule, runtime closure, and resource class. The only treatment difference is `type_conditioned_daughter_relation_bias`.

Promotion requires event-level paired inference, not the existing aggregate comparator alone:

1. Use event UID as the pairing unit, 10,000 paired percentile bootstrap samples, seed `20260826`, 95% confidence, Holm alpha 0.05, and the preregistered family size two: rollout predicted-edge F1 and the teacher-forced edge/complete-target promotion endpoint.
2. The primary relbias-minus-q32 rollout edge-F1 point delta must be at least `0.01`, its Holm-adjusted confidence lower bound must be strictly greater than zero, and its Holm-adjusted p-value must be at most 0.05.
3. Both arms must have finite evidence, positive edge denominators, tree validity, reconstruction validity, leaf retention and P4 validity exactly `1.0`, and fallback count zero.
4. `secondary_metrics_nonregressed`, `teacher_forced_constraints_passed`, and `efficiency_floor_passed` must all be true. The teacher-forced direction/margin table must be sealed before result scoring; missing definitions block promotion. Complete-target efficiency may not regress, validation events/second must be at least 90% of q32, and memory/walltime must remain within contract.
5. Sealed-test and stress roles remain forbidden.

If every gate passes, relbias is admitted. Otherwise q32 remains the architecture baseline. A lower training loss, a favorable point estimate without the adjusted interval, or incomplete event scores cannot promote relbias.

## Locator replay and deterministic cohorts

Replay the sealed v1r2 locator binder in metadata-only, fail-closed mode. It must bind opt676 checkpoint `6c3aafc…`, dataset index `eba2a549…`, selection manifest `fe250b70…`, and all four ordered cohorts:

- train order: 32,768 UIDs, fixed-role selector v1, seed `20260826`;
- pretraining validation: 5,000 unique checkpoint-native UIDs, native seed `20260812`, authoritative digest `1e424f72…` under `sha256-u64be-length-prefixed-utf8-v1`;
- teacher-forced validation: 2,000 UIDs, fixed-role selector v1, seed `20260826`;
- rollout validation: 1,000 UIDs, fixed-role selector v1, seed `20260826`.

Reject reordering, duplicates, missing UIDs, wrong roles/counts/seeds/selectors, legacy compact-JSON substitution, symlinks, trusted-root escape, and any artifact/receipt/attestation hash mismatch.

After replay, perform one separately authorized CPU-only locator pilot. In each cohort choose the 64 UIDs with lowest SHA256 of `UTF8("locator-pilot-v1\0" || uid)`, ties by raw UTF-8 bytes. Each must have exactly one correct role-index membership and one stable locator resolution; sealed-test and stress indexes must never be opened. Seal the selected UIDs, per-UID results, source hashes, and zero-miss/zero-duplicate receipt. This pilot is currently unauthorized.

## Transfer ladder: opt676 versus opt1352 versus scratch

The three matched arms use batch 64, 512 optimizer steps, 32,768 presentations, learning rate 0.001, object/pointer weights 12/16, seed `20260826`, frozen encoder, trainable head, and identical train/validation cohorts and normalizer.

Opt676 may use only its already preserved evidence closure, with every currently null architecture, normalizer, state-spec, selection/validation receipt, and audit path filled and hashed. Opt1352 is only a candidate after `16028845` passes the terminal gate and provides the complete immutable binding. Scratch is exactly one deterministic `torch_generator_seeded_normal_v1` state at seed `20260826`, under the bound constructor and architecture state spec, with finite audit and exclusive receipt. Fit one normalizer from the train cohort only and reuse identical bytes across all arms.

For teacher-forced and rollout modes, compute exactly the three signed contrasts `opt1352-opt676`, `opt1352-scratch`, and `opt676-scratch` using 10,000 paired bootstrap samples, seed `20260826`, 95% confidence, and Holm adjustment across the three contrasts separately per mode. A contrast promotes only if delta is at least 0.01, confidence lower bound is above zero, adjusted p is at most 0.05, and it passes in both modes.

Run globally sequentially: one H100 NVL, eight CPUs, 64 GiB, two hours, no requeue/restart per arm; total ceiling six H100-hours.

- If opt1352 beats both opt676 and scratch jointly, issue the formal immutable step-1352 pointer with criterion `validation_plus_transfer`, `train_loss_used=false`, and exact validation/transfer receipt hashes.
- If opt676 beats scratch and opt1352 does not beat opt676, stop before HPO. The current conditional builder hard-codes step 1352, so opt676 requires an independently reviewed generalized-pointer revision.
- If scratch is not beaten or evidence is incomplete/ambiguous, stop. Scratch is a diagnostic baseline and has no automatic HPO branch.

## Bounded conditional HPO

The currently cleared builder can be used only on the opt1352-selected branch, after paired architecture admission, locator replay/pilot, frozen environment validation, a real bundle, exact scheduler/runtime contracts, and separate authorization.

The immutable grid order is:

1. LR 0.0003, scheduled-sampling max 0.1
2. LR 0.0003, scheduled-sampling max 0.3
3. LR 0.0007, scheduled-sampling max 0.1
4. LR 0.0007, scheduled-sampling max 0.3
5. LR 0.0015, scheduled-sampling max 0.1
6. LR 0.0015, scheduled-sampling max 0.3

All arms use query map `[16,8,6,4,3,2]`, global fallback 16, the admitted relation-bias setting, batch 32, 1,094 steps, 35,008 presentations, validation at steps 547 and 1094 on 5,000 events, bf16 without grad scaling, frozen encoder/head policy, and weights 12/16.

Run at most one arm concurrently, each with one H100 NVL, eight CPUs, 64 GiB, 4.5 hours, no requeue/restart; maximum 27 H100-hours. Midpoint pruning is allowed only at step 547 for nonfinite/hard-invariant failure, projected paired-CI upper bound strictly below 0.01, or failed efficiency floor. A pruned/failed arm is nonrankable and consumes its budget. There is no adaptive expansion, replacement arm, or automatic retry.

Select only among complete, finite, hard-invariant-passing arms using: predicted edge F1 descending; full-tree exact match descending; tree-edit distance ascending; validation efficiency descending; learning rate ascending. Training loss is not a selection or pruning variable.

## Exactly one final reconstruction

After HPO, render a fresh immutable fullscale snapshot for exactly one reconstruction using the selected source and HPO configuration, with no post-selection tuning: 1,094 steps, 35,008 presentations, validation at 547 and 1094 on the bound 5,000-event cohort plus the bound 1,000-event rollout cohort. Budget one H100 NVL, eight CPUs, 64 GiB, 4.5 hours, no requeue/restart.

Qualification requires `COMPLETED 0:0`, finite model/optimizer/gradient/loss/forward checks, all structural rates 1.0, zero fallback, complete validation metrics and denominators, verified telemetry and terminal receipts, and an unchanged source manifest. It can update a validation-best pointer only. It is not a sealed-test result and cannot support a final unbiased test-performance claim.

## What is clear and what remains missing

Clear now: exact r8 four-job transaction; immutable q32/relbias contracts; terminal infrastructure; code-only conditional HPO verifier; non-runnable transfer-ladder/scorer; preserved opt676 closure; sealed UID-binder code/provenance; and the main worktree’s immutable validation implementation snapshot.

Missing: all current terminal result manifests; event-level paired bootstrap evidence and sealed guard definitions; real promoted locator/cohorts and pilot; opt1352/scratch/normalizer/runtime bindings; terminal transfer result; real HPO bundle plus resource/scheduler contracts; generalized pointer support for an opt676 branch; and the final immutable production snapshot. None may be inferred from train loss or existing aggregate reports.

## Critical path

Terminal manifests → paired architecture gate → locator replay/pilot → three-arm transfer ladder → opt1352-only current HPO entry (or stop for redesign) → six-arm bounded HPO → exactly one fresh validation-qualified reconstruction. Sealed test remains untouched throughout.

Future resource ceiling: 6 transfer-ladder + 27 HPO + 4.5 final = **37.5 H100-hours**, planning-only and not authorized.
