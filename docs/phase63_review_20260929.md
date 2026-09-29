# Phase63 partial review and masking-only repair

Phase63 is an **incomplete comparison**, not a completed two-arm result.
The corrected-recovery arm (16742232) completed 4,376 optimizer updates and its
seven strict/search/checkpoint evaluation views. The masking-only arm (16742233)
failed before data-module construction or any optimizer update. Both passed GPU
preflight. Historical source, checkpoints and receipts remain unchanged.

The completed run's native receipt, source contracts, checkpoint hashes, cohort
bindings, all seven reports, both retained-tree scopes and every returned beam
candidate were verified. The primary repeated evaluation is exactly identical.
Its 73,159 scalar metric rows are retained outside Git in the data directory
`HyperTagging_artifacts/phase63_review_20260929`; the compact repository evidence
is `artifacts/codex/reconstruction_phase63_partial_review_20260929.json`.

## Completed recovery-arm results

| Endpoint | Count |
| --- | ---: |
| Selection source-set plus mother PID | 292 / 3,718 |
| Configured full roots | 32 / 100 |
| Original full LCAG | 7 / 3,474 |
| Retained full LCAG | 24 / 3,939 |
| Retained half/component LCAG | 23 / 2,168 |
| Exact nontrivial full components | 3 / 155 |
| Exact nontrivial half/components | 2 / 173 |
| Coherent retained forests | 2 / 100 |

All exact nontrivial full components have one mother and retained depth one.
The run fails the preregistered half-source precision and half-perfect-LCAG
gates. Half-source precision is 158/367; original half-perfect LCAG is 2/163.
Configured-root output and mean predicted depth are not evidence of correct
recursive topology. Structural validity and daughter-sum closure are necessary
implementation checks, not reconstruction accuracy.

On the fixed 20-event beam subset, normalized joint ranking gives retained
full LCAG 7/1393, exact nontrivial components 2/34 and coherent forests 1/20;
greedy gives 5/1393, 0/34 and 0/20 respectively. These small diagnostic counts
do not justify selecting a new ranker on the strict cohort or increasing search
budgets before completing the contrast.

Raw Phase62/Phase63 rates are not causal comparisons: the validation cohorts and
training conditions differ. There is no masking-only estimate, paired interval,
or arm winner yet. Do not promote this checkpoint or start a scientific
successor on the basis of the completed arm alone.

## Failure and correction

The preregistered masking-only configuration correctly sets recovery weight to
zero. The trainer's unconditional positive-weight admission check rejected that
inactive coefficient with `ValueError: recovery_objective_weight must be finite
and positive`. Earlier campaign tests checked configuration construction, but
missed entry into the real trainer. No training directory or training result was
created for the failed arm; its frozen-encoder reuse receipt is not a training
result.

The repair accepts finite nonnegative recovery weights when recovery is inactive
and still requires a strictly positive weight for the active recovery objective.
Negative and nonfinite values remain invalid. This changes admission only; it
does not change the objective, optimizer or the completed recovery arm's behavior.

Regression coverage now enters the actual trainer with both preregistered arm
configurations, tests invalid weights, and completes a CPU optimizer step for
masking-only supervision at zero recovery weight. A guarded single-arm
continuation checks the original failed receipt and scheduler state, absence of
training output, identical scientific configuration/cohort/initial checkpoint,
and an exact source diff restricted to the reviewed admission repair. The
completed arm will not be rerun. The repair uses a new immutable source tag and
new scheduler job ID, retaining both source revisions in the study lineage.

## Next decision

Complete the same masking-only arm with the original seed, 70,000 training
events, 4,376-update budget and reserved 1,000/100 selection/strict cohorts.
Reusing this predeclared cohort completes the existing contrast; no replacement
cohort or hyperparameter tuning is authorized by these partial results.

Then compare the two arms with paired event-level uncertainty, nontrivial tree
counts and depth, coherent forests, full/half LCAG and all original gates. If a
recovery benefit is absent, use the simpler masking baseline for the next
controlled experiment. The previously measured radial saturation remains a
priority for a separate reconditioning/parameterization study. Seed replication,
beam-candidate calibration and host/offline search parity remain distinct work.
The sealed test stays closed; there is no automatic successor or promotion.
