Phase76 membership refinement development
==========================================

This separately authorized follow-up to :doc:`Phase75 <phase75>` tests one
refinement-conditioning factor. The original 70000-event corpus stays fixed.
The completed Phase74 factorial did not establish a capacity or pretraining
winner, and Phase75 gradient projection did not improve membership recovery.
Neither result establishes data scarcity or irrecoverably missing encoder
information. No data expansion, primary reservation or wider model is tested.

Source-backed diagnosis
-----------------------

Frozen Phase75 checkpoints were replayed on 1536 training events, 600 already
inspected development events and 24 tiny events per arm. Detached model outputs
were recorded before joining truth. Raw predictions reproduce the immutable
summaries. Tests cover permutation-invariant matching, unassigned sources,
masked unknown supervision and presence versus raw membership. A diagnostic
pairing correction joins the original target identity across proposal/refinement
slot permutations; old traces and initial receipts remain intact.

For the joint control, training targets contain 9437 source assignments across
1024 B trials. The proposal misses 6000, including 4172 assigned to unassigned
and 1828 assigned to the other B. Refinement misses 6232, including 4480 assigned
to unassigned, while reducing extra sources from 3364 to 2961. No exact target
is recovered at either stage. On inspected development, proposal and refinement
miss 2469 and 2501 of 3621 assignments; extra sources fall from 1243 to 1094.
Refinement slightly improves mean best overlap there. Thus refinement has
tradeoffs, rather than uniformly damaging predictions.

Most trials have 5 to 16 sources. Training size-bin denominators are 61, 413,
518 and 32 for sizes 1 to 4, 5 to 8, 9 to 16 and 17 or more; the corresponding
inspected-development denominators are 23, 175, 193 and 9. Complete missing,
extra, unassigned/other-B, category and size counts are downloadable. Tiny raw
membership remains 32 of 32 in both previous arms; five targets violate the
unchanged charge guard, explaining accepted 27 of 32. Main failure is raw
generation, not rejection of an otherwise correct pool.

In the joint control's 512 training B collisions, 3559 of 4096 generated merges
contain unassigned sources, 334 cross B memberships and 319 are clean within-B.
These categories can overlap. A finer rejoin of the same preserved traces
separates 1890 mixed B/unassigned merges from 1669 background-only merges;
background-only merging is not called B contamination. The final partial context
mixes unassigned sources into groups covering 1711 of 9437 B-source nodes. This
additive exploratory analysis does not change the registered contrast or use
fresh development outcomes. Relation supervision labels only identifiable
within-B pairs, while generation ranks all source-disjoint pairs. This mismatch
and contamination motivate a causal refinement test, not a demonstrated benefit.
Using the proposal's B label as a merge filter would retain only 92 of the 319
clean merges, so that unvalidated filter is not substituted for this study.

One controlled contrast
-----------------------

The control retains generated partial-group context; the intervention zeros
only that input block in refinement. This also removes singleton context and
therefore tests the whole block, not only contaminated merges. Both arms retain
differentiable predicted-membership group context, detector features, the same
relation objective and truth-free partition-generation procedure, and joint encoder
adaptation. Realized partitions can differ after learning; only the generation
procedure and bounds are identical. Truth enters supervision or post-generation
analysis only. Allocated parameter counts match, but the disabled block
removes an input pathway: its first-layer input-block weights receive no data gradient.
This is an ablation of the whole pathway, not a claim of identical effective
capacity or measured equal FLOPs.

Both use the same Phase74 128-existing pretraining-final encoder and fresh
identical 256-wide heads. Context width 128, hyperbolic width 32, depth four,
curvature, radius convention, PID decisions/construction and train-fitted
normalization remain fixed. Neither arm uses gradient projection or new
pretraining. Independent tiny branches use 24 events and 1000 updates; main
training uses the same eligible 1536-event pool and 1500 updates, batch eight.
The corpus has 70000 events; this bounded run does not train on all of them.

A fresh development cohort contains 100 events in each of charged, mixed,
ccbar, uubar, ddbar and ssbar. UID/category/source-only deterministic selection
excludes all original training, checkpoint-selection, primary, supplemental
and previous development reservations, including Phase74 and Phase75. No easy
or complete-event filter is applied. Checkpoints are fixed final and presence
threshold is 0.5; held-out data never select thresholds or checkpoints.

Two CPU-only jobs are bounded to two CPUs, 32 GiB and eight hours each, with no
requeue or automatic successor. Full authenticated identity/cache admission and
real-data gradients pass. The largest event has 73 detector sources, no event
is dropped, and membership targets are available throughout the admitted data.
Resource smokes forecast less than eight hours; equal updates are not measured
equal FLOPs. Actual presentations, unique sampled identities, node-pair proxies,
parameters, wall/CPU time and peak memory accompany terminal results.

Endpoints and interpretation
----------------------------

Both trainings completed successfully. The preregistered endpoint
requires tiny raw membership at least 95 percent, count-backed fresh held-out
raw and accepted improvement with paired uncertainty, background control and
zero accepted source conflicts. Sparse or zero successes do not establish
population equivalence. There is one seed and two correlated B trials per
B-bearing collision. No primary scale-up is authorized in this pass regardless
of the outcome.

Fixed final results
~~~~~~~~~~~~~~~~~~~

.. list-table:: Context on versus off, same fresh development cohort
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Context on
     - Context off
   * - Tiny raw membership
     - 32 of 32
     - 32 of 32
   * - Tiny accepted membership
     - 27 of 32
     - 27 of 32
   * - Main-train raw and accepted membership
     - 0 of 1024
     - 0 of 1024
   * - Held-out raw and accepted membership
     - 0 of 400
     - 0 of 400
   * - Main-train continuum accepted events
     - 96 of 1024
     - 117 of 1024
   * - Held-out continuum accepted events
     - 52 of 400
     - 46 of 400
   * - Accepted source conflicts
     - 0
     - 0

All six held-out categories process 100 collisions without execution failures
or unavailable B-membership targets. Charged and mixed each have 200 B trials,
all raw generation failures in both arms. Continuum acceptance is ccbar 26
versus 22, uubar 8 versus 11, ddbar 10 versus 8 and ssbar 8 versus 5, each of
100 collisions. There are no recovered main-train or held-out B source sets to
attribute to channels or sizes; all evaluated denominators remain downloadable.

The five tiny retention failures are charge-incompatible targets. Charge guards
would also reject 134 of 1024 main-training and 54 of 400 held-out targets, but
current failures occur earlier: none of these main/held-out targets is generated
exactly. No trial is dropped to improve a rate. Raw and accepted results remain
separate, and the tiny gate uses raw membership.

The paired continuum difference, off minus on, is minus 0.015, with a
category-stratified collision interval from minus 0.040 to plus 0.0075. Sixteen
collisions are accepted only with context on and ten only with context off;
374 are tied. This does not establish a background improvement. All 200
B-bearing collisions are tied at zero exact recovery. The empirical paired
bootstrap interval of zero to zero is degenerate, not evidence of population
equivalence; each arm's two-sided exact event-level 95 percent interval is zero
to 0.0182753. The two B trials in each collision are correlated.

Tiny raw memorization passes, but the held-out gain and background-control
criteria do not. Removing the whole partial-context block fails the registered
scientific endpoint. This does not prove that repairing contaminated states or
improving proposal formation cannot help. No primary scale-up, model promotion,
new reservation or second successor follows this pass.

Detector within-B relation counts are 8065 versus 8028 correct of 12898.
Generated-state counts are 10111 of 16797 versus 10435 of 17347; supports differ
across learned models. Latent deep source-set proposals are 97 versus 96 of
1486, with 15 versus 18 retained. These are diagnostic source sets, not physical
proposals or verified reachability. Ignored-pair counts are separately measured.

Post-hoc continuum retained-component raw equality is 2 versus 1 of 1184
explicit composite components, and accepted equality is zero in both arms.
The components are ccbar 337, uubar 285, ddbar 292 and ssbar 270. Twenty-four
continuum collisions have no explicit composite root and remain separately
reported. Main-train component counts include two unavailable source mappings
among 3102 components; unavailable is not a measured failure. These accidental
flat-group equalities do not establish physical component reconstruction.

Source-level effects
~~~~~~~~~~~~~~~~~~~~

On fresh held-out B-bearing collisions, proposal source recall falls from
1167 of 3634 with context on to 964 of 3634 with it off: a difference of
minus 0.05586, with paired interval minus 0.07265 to minus 0.03993. Proposal
precision rises from 1167 of 2344 to 964 of 1837, but more target sources are
assigned unassigned. This tradeoff does not support simply deleting the pathway.
After refinement, recall is 1081 versus 1107 of 3634, a difference of 0.00715
with interval minus 0.00554 to plus 0.01952. Refined precision is 1081 of 2102
versus 1107 of 2121; its paired difference also includes zero. These secondary
source-level intervals are descriptive, single-seed and not multiplicity-adjusted.
They do not rescue the failed exact-membership endpoint.

Efficiency and cumulative decision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each arm uses 20000 new fit presentations: 8000 tiny and 12000 main. Actual
main sampling presents 1535 distinct identities from the 1536-event pool; tiny
sampling presents all 24. Both inherit the same 1000 pretraining updates and
8000 pretraining presentations, with no new pretraining. The control exactly
reproduces all earlier training records and final model/decoder states.

There are 1322658 encoder and 1385991 decoder parameters per arm. Recorded
fit/evaluation wall times are 3067.96 and 3004.61 seconds; process CPU times
are 3071.71 and 3009.24 seconds. Process peak RSS is 1826428 and 1835148 KiB;
scheduler-sampled peaks are smaller because sampling does not capture every
instantaneous peak. Each fit processes a sum of 36604461 detector-node-squared
pairs and 40000 encoder passes, plus 3715508 node-squared pairs and 4320 passes
for the 2160 evaluation event views. These are workload proxies, not FLOPs.
Read-only diagnostic replay and trace rejoining are separately accounted for.
The last 100 main-update mean losses are 1.27464 and 1.26743, versus 1.18848
and 1.19677 in the preceding 100. Matched batch composition still produces noisy
losses; convergence is not established and no checkpoint is selected from them.

Keep the corpus and width fixed. Neither the prior capacity/pretraining test,
gradient projection, nor this partial-context ablation establishes an exact
membership gain. Data growth and better pretraining remain unproven remedies.
The supported next direction is proposal-stage B-to-unassigned error correction,
first tested on training-only diagnostic states with background controls and
then, only under a separately admitted study, fresh development assessment.
Do not rerank a nonexistent correct pool or repeat whole-block ablation merely
because a scalar loss or individual relation count changes.

Flat optional-B groups are retained-source proxies. Physical exact trees,
physical beam pools, legal deep-proposal survival and daughter-sum p4 closure
are unavailable. Latent source-set proposal diagnostics are not legal physical
reachability. Post-hoc equality to explicit continuum retained-component sets
is reported separately from fake-B acceptance; it is neither physical component
reconstruction nor parton ancestry. No FEI-equivalent efficiency is claimed.

This 600-event developmental assessment is not a complete post-study primary
evaluation, which requires at least 2000 distinct collisions in each of the six
categories, at least 12000 per arm. See :doc:`evaluation` for common definitions
and :doc:`_generated/status/index` for the latest results and complete downloads.
