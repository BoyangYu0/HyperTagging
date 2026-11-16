# Reconstruction study workflow

Use the current controller state and reviewed protocol to choose work. Historical
plans do not authorize new runs. Be practical, not overly defensive: complete
authorized steps, repair concrete failures and reuse valid results instead of
repeating approvals, broad tests or completed training.

## Plan, execute, evaluate, decide

1. Define one scientific question, a paired control, changed factors, fixed data
   roles, finite sample/update counts, metrics, practical margins and a decision rule.
2. Implement the necessary change. Check the affected entry point and meaningful
   failure cases; broaden tests only for unresolved risk or required CI.
3. Run registered operations, retain checkpoints and actual exits, and reconcile
   accepted jobs through completion or failure.
4. Evaluate reconstruction using the [evaluation workflow](full_decay_reconstruction_evaluation.md#evaluation-workflow).
   Compare with the control; distinguish a software repair from a scientific gain.
5. Record the result, publication decision and concrete next experiment or repair.
   A null result selects a useful diagnosis; it does not justify a blind retry.

Use the configured design/review and implementation/execution roles and installed
routing schema, including the actual study ID. A remote check is useful when it
resolves an environment-specific question, not as a compulsory extra stage.

## Use resources in parallel

Keep training and full-reconstruction studies moving together. Evaluate immutable,
completed checkpoints on CPU while GPUs train; prepare data or run independent
registered controls on other available allocations. Parallelize independent work,
not operations that depend on an unfinished result or write the same state.

The existing controller records each operation's owner, source revision, inputs,
output directory, job ID, dependencies and requested resources. Workers write
separate evidence; only the controller updates campaign state. Use immutable
source snapshots when live workers need different code versions.

Size concurrency to allocated GPU memory, CPU threads, RAM and storage bandwidth.
Reuse datasets, normalization and valid checkpoints; measure throughput and memory
before increasing concurrency. Reduce idle time without slowing other users' jobs.
Follow the [shared-GPU rules](../environment/gpu/README.md#shared-gpu-rules).
Project CPU/GPU budgets and elapsed cutoffs remain unset; administrator allocation
limits and supported diagnostic limits still apply. Scientific counts stay finite.

## Track results once

Keep a private study record on the data volume, linked from the campaign ledger:

- Study/operation ID, hypothesis, control, changed factor and decision margins.
- Source/config hashes, data roles and manifest/index hashes, checkpoint lineage,
  evaluator version, inference settings and seeds.
- Jobs, actual exits, sample/update/presentation counts, elapsed time and compute.
- Metrics and denominators, paired differences and uncertainty, coverage,
  failures/unavailable values, artifact locations and interpretation.
- Next action, dependencies/live owners, and publish/defer with its reason.

Retain failed attempts and partial results. Corrected attempts get fresh identities;
never overwrite evidence to make an operation appear successful. Keep reusable
rules in these guides and transient decisions in the study record. See
[storage](repository_storage.md) for file placement and recovery.

## Evaluate and choose the next study

Run native full reconstruction after each reconstruction operation; teacher-forced
loss alone is insufficient. Use continuous source/edge quality to guide progress
while exact recovery is sparse. Exact recursive topology plus PID remains the goal;
zero exact recovery does not erase a supported continuous gain.

Use the registered metrics and fixed cohorts in the
[evaluation contract](full_decay_reconstruction_evaluation.md). Poor TRAIN fit
motivates loss/gradient/matching diagnosis; teacher/native divergence motivates
generated-state training; useful proposals lost during selection motivate a
controlled inference change; structural gain with weak PID motivates PID work.
Use measured trends to justify continuation. TRAIN-only and oracle-only gains
are not native development gains. Repair measurement using existing checkpoints
when possible rather than retraining.

## Keep ownership continuous

Use one controller and operation registry. Return success, null results and
reconciled failures to review with the exact remaining action. Preserve explicit
user pauses. A temporary STOP_AFTER_STAGE ownership marker is reconciled under
the lock by the sole owner after the worker is terminal, then archived before
resuming the saved result. Do not reset consumed identities or duplicate owners.

Continue useful authorized work through routine failures. Stop for an explicit
user request or an evidenced absence of meaningful authorized alternatives; record
what would resume progress. Reserve UNSOLVABLE_TERMINAL for that supported decision.

## When and how to update the dashboard

At every review, record **publish or defer**, why, and what published claim changes.
Publish a supported quality result, accumulated diagnosis that changes the next
experiment, material correction, or meaningful readiness change. Ordinary tests,
queue polls, routine repairs and unchanged failures stay in the private record.

Send the existing publication owner a reviewed handoff with source hashes,
sanitized aggregates, uncertainty, coverage, cost, limitations and next action.
Follow [wiki maintenance](wiki/maintaining.rst): authenticate inputs, update the
allowlisted sources, validate rendering/privacy/links and required exact-commit CI,
then use guarded deployment and verify the live page and downloads.

Expand the latest published study; summarize older studies and retain their
reports/downloads. Label the latest learned evaluation separately from operational
updates. Counts and readiness are not reconstruction efficiency. Publication
repairs should not repeat completed science or block independent study work.

When the local app is unavailable, retain reports in the controller's durable
outbox. On reconnection, deliver one current summary linked to the full archive;
clear only successfully delivered entries. Study work continues independently.
