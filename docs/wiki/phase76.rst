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
These categories can overlap. Relation supervision labels only identifiable
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
analysis only.

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

Terminal evaluation is pending in this working draft. The preregistered endpoint
requires tiny raw membership at least 95 percent, count-backed fresh held-out
raw and accepted improvement with paired uncertainty, background control and
zero accepted source conflicts. Sparse or zero successes do not establish
population equivalence. There is one seed and two correlated B trials per
B-bearing collision. No primary scale-up is authorized in this pass regardless
of the outcome.

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
