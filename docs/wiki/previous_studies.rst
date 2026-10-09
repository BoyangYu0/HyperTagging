Previous studies through Phase83
================================

The accumulated Phase78–83 evidence finds **no demonstrated membership
improvement**. Phase81's explicit pair supervision leaves both arms at zero
exact main memberships and increases background contamination. Phase83 passes
zero of twelve structural improvement gates. These are training-role diagnoses,
not fresh validation, primary evaluation or generalization successes.

This review completes the separately authorized historical publication through
Phase83. It retains the original :doc:`Phase78 report <phase78>` and all earlier
:doc:`studies <studies>` and downloads. Later research is outside this frozen
review. Future incremental research remains private under the user's existing
publication preference.

What the sequence establishes
------------------------------

.. list-table:: Sequential evidence, with unchanged scientific gates
   :header-rows: 1
   :widths: 14 52 34

   * - Study
     - Observed endpoint
     - Interpretation
   * - Phase78: finite linear probe
     - Assessment event-mean pair AUC 0.5333 versus native 0.5452; shuffled null 0.5005. Difference interval [-0.02365, -0.00097].
     - This finite probe exposes no stronger unused pair signal; it does not prove information is absent.
   * - Phase79: optimized linear probe
     - AUC 0.5494 versus native 0.5452; paired 95% interval [-0.00671, 0.01409].
     - Function tolerance converges; strict gradient tolerance 1e-7 fails at 1.88e-7. No demonstrated deployable gain.
   * - Phase80: optimizer step size
     - All twelve native and half-size updates worsen audit risk and B assignment; exact membership remains 0/64.
     - Reducing damage is not improvement and does not justify a learning-rate study.
   * - Phase81: explicit pair supervision
     - Native and pair arms both tiny raw 32/32, accepted 27/32; main raw and accepted 0/1024.
     - Training screen fails; moving foreground out of unassigned increases swaps and background.
   * - Phase82: nonlinear readability
     - Assessment AUC 0.5349 versus native 0.5452 and optimized linear 0.5494; difference from native interval [-0.02587, 0.00426].
     - Readability gate fails. Better fitting risk does not establish assessment information or nonconvex convergence.
   * - Phase83: first-moment history
     - Zero of twelve qualifying updates, against a required nine. All conditions raw and accepted 0/64.
     - All three lower-risk steps add proposal background; no optimizer-history intervention is supported.

Phase80 false background assignments decrease while B assignment worsens;
that tradeoff is not a structural improvement.

Phase78 native replay preserves all 1536 main identities and 24 tiny views.
It excludes and counts 466 main and five tiny shared-source pairs while retaining
all event and native trial denominators. The 128 assessment B collisions are
training-role events, disjoint only from the new probe's fitting events.
Conditional-optimal source assignment 6075/9437 versus a truth-assisted
majority-size comparator 5713/9437 is a diagnostic, not an inference policy.
The original report retains query similarities, margins, pair support,
background scores and all category and overlapping source-size strata.

Phase81: improved foreground counts, worse contamination
--------------------------------------------------------

.. list-table:: Matched native versus pair-supervised final checkpoints
   :header-rows: 1

   * - Endpoint
     - Native
     - Pair supervised
   * - Proposal correct B assignments
     - 3437/9437
     - 4304/9437
   * - Proposal other-B swaps
     - 1828/9437
     - 2411/9437
   * - Proposal background assignments
     - 2981/52590
     - 5362/52590
   * - Refinement correct B assignments
     - 3205/9437
     - 4152/9437
   * - Refinement other-B swaps
     - 1752/9437
     - 2341/9437
   * - Refinement background assignments
     - 2491/52590
     - 5049/52590
   * - Conditional-optimal proposal correctness
     - 6075/9437
     - 6040/9437
   * - Conditional-optimal refinement correctness
     - 6053/9437
     - 6011/9437
   * - Proposal event-mean same-B/cross-B AUC
     - 0.54317
     - 0.54137
   * - Refinement event-mean same-B/cross-B AUC
     - 0.54073
     - 0.53342
   * - Accepted continuum collisions
     - 96/1024
     - 78/1024
   * - Main raw and accepted exact memberships
     - 0/1024
     - 0/1024

Proposal AUC change has paired 95% interval [-0.00496, 0.00142]; refinement
[-0.01101, -0.00348]. Accepted continuum difference is -0.01758 with interval
[-0.03027, -0.00488]. That reduced acceptance does not establish better tagging:
exact recovery remains zero and raw background contamination rises. Source
conflicts remain zero; the five tiny charge-incompatible targets remain rejected.
Native final model and decoder tensors exactly replay the historical control,
with matched initialization and sampling order.

The denominator is 512 B collisions and 1024 correlated B trials, plus 1024
continuum collisions. Each of charged, mixed, ccbar, uubar, ddbar and ssbar has
256 main collisions per arm. The two arms use the same cohort. Zero exact
bootstrap intervals do **not** prove equivalence; the reported collision-success
95% upper bound is 0.00718. Full category, channel, source-size, event and trial
counts, unavailable endpoints and uncertainty remain in the complete downloads.

Phase83: latest completed historical endpoint
---------------------------------------------

The corrected optimizer-history diagnostic exactly replays the native baseline
and update, then compares stored first-moment Adam updates with current-gradient
updates on the same twelve training batches. Second moments, clipping, learning
rates and weights remain matched. No model is selected from these counterfactuals.
Its 96-event training audit contains 16 collisions per category: 32 B collisions,
64 B trials and 64 continuum collisions. The optimizer batches cover 94 distinct
identities. Paired non-minibatch audits retain 94–96 events; the twelve
comparisons share the checkpoint and overlapping events and are not twelve
independent experiments. Baseline support is 585 B assignments and 3242 background
nodes, with continuum acceptance 6/64. The gate requires consistent structural gain with background control,
not merely lower risk. Zero of twelve updates qualifies. All three lower-risk
updates increase proposal background. Raw and accepted exact recovery remains
0/64 in every condition.

Roles, controls and unresolved questions
----------------------------------------

The original authenticated 70000-event corpus stays fixed. Phase81 uses a
1536-event eligible main pool and 24 tiny views, with contextual width 128,
hyperbolic width 32 and head width 256, fixed depth, geometry, PID, normalization,
pretrained initialization and final-checkpoint selection. Each arm receives
1000 tiny and 1500 main updates at batch eight: 8000 tiny and 12000 main
presentations. Actual distinct main identities are 1535/1536; tiny uses all 24.
Shared historical pretraining contributes 8000 presentations per lineage.

The exposure contrast **8000 presentations over 24 tiny identities versus
12000 over 1536 main identities is an unresolved confound**. Tiny memorization
is neither a capacity proof nor proof that the main model is undertrained.
The evidence also does not establish adequate representations, main convergence,
irrecoverable information loss, data scarcity or an optimizer implementation bug.

Probe fitting uses 384 B events and assessment 128 B events, event-disjoint
within the training role. The scientific checkpoint has already seen that role;
repeated exploratory assessment is disclosed. Phase79 and Phase82 fit diagnostic
probes; Phase80 and Phase83 execute counterfactual optimizer steps; Phase81
performs two scientific training fits. Calling the whole sequence zero training
would be incorrect. None uses fresh validation or sealed test data, reserves a
new primary cohort, or establishes generalization. The required complete
post-study evaluation of at least 2000 distinct collisions in each of six
categories per arm has not occurred.

Truth joins follow detached model-generated outputs. Same-B/cross-B labels,
conditional-optimal matching and truth-assisted comparators are supervision or
diagnostics, never deployable search. Exact membership means retained detector
source-set equality. It is not physical reconstruction or FEI efficiency.
Physical trees, physical exact-tree top-1 or beam-pool recovery, legal deep
reachability and p4 closure remain **unavailable** for these flat heads.
Latent source survival and relation counts remain explicitly labelled proxies.

Actual compute and preserved failures
-------------------------------------

.. list-table:: Recorded scheduler resources; all CPU, no requeue
   :header-rows: 1

   * - Work
     - Elapsed seconds
     - Scheduler CPU seconds
     - Allocation
   * - Phase79 optimized probe
     - 43
     - 40.203
     - 2 CPU, 16 GiB, 1 hour
   * - Phase80 step diagnostic
     - 297
     - 291.562
     - 2 CPU, 16 GiB, 1 hour
   * - Phase81 native training
     - 3107
     - 3085.588
     - 2 CPU, 32 GiB, 8 hours
   * - Phase81 pair training
     - 3202
     - 3181.025
     - 2 CPU, 32 GiB, 8 hours
   * - Phase81 review
     - 68
     - 64.464
     - 2 CPU, 16 GiB, 1 hour
   * - Phase82 nonlinear probes
     - 44
     - 40.014
     - 2 CPU, 16 GiB, 1 hour
   * - Phase83 failed replay / corrected diagnostic
     - 40 / 188
     - 36.063 / 181.522
     - 2 CPU, 16 GiB, 1 hour each

All fifteen Phase79–83 jobs, including admission checks, a cancelled own
superseded test and failed replay, total 7596.282 scheduler CPU seconds.
Phase78 separately records 159 successful and 24 failed-preflight scheduler
seconds. Successful Phase78 process CPU is 150.92 seconds, peak RSS 1619072 KiB.
Its two 257-parameter probes perform 1024 updates and 262144 pair presentations.

Phase79 true/null fits use 119/120 function evaluations and approximately
8.120/8.186 million pair presentations including full-risk and Hessian work.
Phase82 fits two 16513-parameter probes for 2048 updates times 256 pairs each,
covering 65436 distinct fitting pairs per head. Neither updates the encoder.
Phase81 has 2708649 optimizer-eligible parameters. Each arm records 36604461
fit node-squared proxy units and 40000 encoder passes; evaluation adds 1560
views and 3120 passes. The pair intervention adds 17879076 supported unordered
pair presentations and 35758152 two-head likelihoods, excluding 5401 shared-source
pairs; unknown support is zero. Native/pair process wall is 3088.37/3183.82 seconds,
CPU 3070.63/3166.15 seconds and peak RSS 1800392/1783724 KiB. Equal update counts
and node-squared proxies do not imply equal FLOPs.

Corrected Phase83 executes 24 optimizer steps, 96 shared gradient presentations,
192 condition exposures and 1344 audit views; encoder passes are 192 gradient
and 2688 evaluation. Fit/audit node-squared proxies are 158429/2273082. Process
wall/CPU is 179.305/177.225 seconds and peak RSS 1492164 KiB. Scheduler and process
measurements have different accounting boundaries and remain separate.

Preserved Phase81 failures concern admission status schema, ordered-list versus
identical UID-set comparison, and a new loss expecting source lists where native
inputs supply boolean source masks. Corrections precede scientific updates and
retain alias, permutation and dtype checks. The cancelled own test is not a pass.
Phase83's original exact replay guard fails at maximum trace difference 9.54e-7
because of historical FP32 initial-plus-delta reconstruction. The correction
applies that reconstruction symmetrically and retains the exact guard. The failed
attempt performed one native step, eight gradient presentations and 192 audit
views before intervention. A mistyped test invocation collected zero tests; the
correct invocation subsequently passed. No failed receipt is relabelled successful.

The recorded broad suite is 2430 plus 41 passed, 34 skipped, at its recorded
source before the wrapper/source-mask corrections. Corrected core/pair checks
have 203 passes and contract checks 10; the final combined diagnostic/pair suite
has 28 passes. These are different scopes and revisions. Publication validation
is recorded separately; CPU fixtures establish implementation behavior, not physics.

Recommendation and complete evidence
------------------------------------

Keep the corpus, widths, geometry and acceptance policy fixed on this evidence.
Do not adopt the tested pair objective or optimizer variants, spend fresh
validation on them, or scale up. No tested mechanism supplies the structural and
background-controlled gain required for another campaign. This is the conclusion
of the frozen previous-studies sequence, not a claim about later research.

The :doc:`dashboard <_generated/status/index>` expands Phase83 only. The
:doc:`download catalogue <_generated/status/downloads>` retains Phase78's original
three event parts and all previous downloads, and adds the complete lossless
Phase79–83 metric records with bounded decoders, source hashes, exact cardinality
and transport integrity. Private identities, paths and operational secrets are
excluded; supported scientific counts and metrics are retained. Full publication
requires exact-commit CPU, all documentation layouts, privacy and download checks,
guarded promotion, and verification of the actual live release.
