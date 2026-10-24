# Continuous reconstruction study loop

This is the single reusable study workflow. The latest user instruction and
current reviewed protocol determine the work; dated reports and old handoffs
are evidence only. Obtain the current study and next operation from the sole
controller, not from a hardcoded phase number in documentation.

## Run the cycle

1. **Plan:** choose one improvement question from the previous measured result.
   Define its paired control, changed factor, data roles, finite scientific
   counts, practical margins and next-decision rule before execution.
2. **Implement:** fix the concrete prerequisite or model/algorithm change.
   Exercise the actual affected entry point and relevant failure path, reuse
   unaffected evidence, and return the concrete candidate for review.
3. **Execute:** run the reviewed operations, reuse valid data/checkpoints and
   retain outputs, actual exits and compute. Keep accepted jobs owned through
   terminal reconciliation. Registered parallel work is allowed by the protocol.
4. **Review:** compare reconstruction quality with the control, identify the
   supported bottleneck, and issue one concrete next handoff. Every work stage
   returns to review on success, a scientific null or operational failure.
5. **Continue:** a negative result selects a diagnosis or changed experiment;
   a repairable error selects a root-cause repair. Neither ends the campaign.

Astra medium handles design and independent review; Sol high handles
implementation, execution, monitoring and publication, at the default service
tier. Follow the installed schema and include the actual study_id. Validate the
outgoing route in isolation. Avoid repeated generic checks, blind retries,
unnecessary retraining and unmotivated width/data sweeps. NAF can resolve a
concrete environment-dependent question; it is not a compulsory verification
stage for a locally reproducible defect.

## Evaluate after each reconstruction operation

Use continuous partial quality to select near-term improvements while exact
recovery is sparse. Exact recursive topology plus PID remains the final goal
and a reported endpoint; zero exact recovery does not veto a continuous gain.
The reviewed protocol must bind its continuous metric definitions and observer
version. Shared evaluator definitions and complete-evaluation rules belong in
the [evaluation contract](full_decay_reconstruction_evaluation.md).

The private closeout records:

- Per-generation source F1/IoU, precision/recall and membership-sensitive edge
  quality, separating shallow from deep response and proposal stages from the
  final selected output. Counts alone measure activity, not correctness.
- Selected B quality separately from oracle/pool quality; PID and confidence
  proper scores only when actual probabilities and trained-state provenance
  exist. Teacher-forced fit and native rollout remain distinct.
- Exact, topology-only and inclusive endpoints; each continuum flavour's
  acceptance; nominal, processed, failed, incompatible and unavailable counts.
- Paired control differences, practical margins, whole-collision uncertainty,
  category/level support, update/presentation counts, latency, memory and compute.

Use the registered aggregation consistently for point estimates and bootstrap.
Include unmatched truth and extra wrong proposals appropriately. Keep failed
collisions in nominal accounting and missing truth/scores explicit, with bounds
where defined. Do not fabricate telemetry or add unregistered inference solely
for a report. Reuse existing checkpoints for a reviewed measurement repair.
Small diagnostic panels are useful but do not establish complete evaluation,
independent calibration, confirmation, generalization or physical completeness.

Choose the next intervention from the measured bottleneck: poor TRAIN fit
motivates loss/gradient/matching diagnosis; teacher/native divergence motivates
generated-state training; useful proposals lost at a filter motivate one
controlled inference change; structural gain with weak PID motivates PID work.
Use observed slopes to justify a finite continuation. Do not promote TRAIN-only
or oracle-only gains as native development improvement.

## Keep handoffs continuous

Maintain one controller and operation registry. Each transition must return
both success and reconciled failure to that controller's review path. Preserve
rollback, partial outputs and actual failure; a launch exit or partial readiness
file is not proof of adoption. Corrected attempts use fresh identities.

An automated STOP_AFTER_STAGE used for ownership transfer is temporary. Once
its worker/transition is terminal, the sole owner reconciles state under the
lock, archives the matching marker and resumes the saved result into review.
Preserve explicit user stops/pauses. Never remove a lock, reset a consumed hook,
replay completed work or start a duplicate coordinator. Uncertain ownership
requires continued reconciliation before successor launch.

Project CPU/GPU budgets and elapsed cutoffs are null: queue waiting,
preparation, training, evaluation, owner observation, elapsed watchdog and
reconciliation. Service runtime is unlimited; administrator allocation limits
still apply and must be refreshed at admission. Recoverable query timeouts are
not cumulative deadlines. Scientific sample/update counts remain finite.

Stop only on explicit user request or an evidenced absence of meaningful
authorized alternatives. Record what would resume progress. Routine errors,
zero recovery, missing metrics and completion of a finite study are not terminal
conditions. The controller's UNSOLVABLE_TERMINAL marker is reserved for a
supported terminal decision.

## Decide dashboard publication at every review

Every reconstruction gets a private closeout. Every scientific review records
**publish or defer**, why, and which currently published evidence would change.
Publish a significant reviewed continuous-quality result, a diagnosis or
accumulated negative evidence that changes the recommendation, a substantial
correction, or a material plan/readiness change. Nonzero exact recovery is not
a publication prerequisite; small reused-panel gains remain exploratory.

Do not publish each test, stage, queue poll, ordinary repair or unchanged
failure. Coalesce them until interpretation changes. Show quality, uncertainty,
coverage, cost, limitations and the next experiment using sanitized aggregates.
Expand the latest study admitted for publication; summarize earlier published
studies and retain their downloads. If the latest update concerns metadata or
operations, keep the latest learned evaluation separately labelled. Static
counts and operational readiness are not reconstruction efficiency.
Use the existing publication owner and stable dashboard, with the
[publication procedure](wiki/maintaining.rst) for CI, privacy and live checks.
Publication failures return to repair/review without repeating completed
science or unnecessarily blocking its next authorized step.

A handoff names completed work, preserved failures, metric comparison, exact
remaining operation/dependencies, next action, study_id, publication decision
and any live owner/job. Keep transient study decisions in that record, not in
another Markdown instruction file.
