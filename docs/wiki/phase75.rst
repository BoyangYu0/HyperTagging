Phase75 gradient-routing development review
==============================================

The sole bounded successor to :doc:`Phase74 <phase74>` completed on 9 October
2026. Ordinary joint membership/relation gradients were compared with removal of
only the relation-gradient component opposing the membership encoder gradient.
Both arms pass tiny raw memorization but recover no exact main-train or held-out
memberships. The intervention does not establish a downstream efficiency gain.
No further campaign, primary reservation, wider model or model promotion follows.

Fixed final results
-------------------

.. list-table:: Joint versus projection, same fresh development cohort
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Joint
     - Projection
   * - Tiny raw membership
     - 32 of 32
     - 32 of 32
   * - Tiny constraint-accepted membership
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
     - 89 of 1024
   * - Held-out continuum accepted events
     - 47 of 400
     - 40 of 400
   * - Accepted source conflicts
     - 0
     - 0

Every held-out category contributes 100 processed collisions, with no failures,
replacements or unavailable membership trials. Charged and mixed each contribute
200 nominal B trials. Continuum acceptance by category is ccbar 26 versus 23,
uubar 4 versus 3, ddbar 10 versus 8 and ssbar 7 versus 6, each out of 100.
The separate tiny branch contains eight charged, eight mixed and eight ccbar
events; continuum acceptance there is zero of eight in both arms. Five tiny raw
target memberships violate the unchanged charge guard, as in Phase74.

There are no recovered held-out or main-train source sets to attribute to a
channel or size. Complete evaluated channel/size denominators remain in the
download. Physical exact-tree top1, beam-pool recovery, physical deep-proposal
survival and physical p4 closure are unavailable: the model produces flat latent
memberships, not physical mothers or a physical beam. Inclusive top1 refers only
to its predicted partition. Continuum retained-component recovery is unavailable
for this optional-B grouping head. No FEI-equivalent efficiency is claimed.

Paired uncertainty and supported relation diagnostics
------------------------------------------------------

All 200 B-bearing collisions are tied at zero recovery. The empirical paired
bootstrap difference is zero with interval [0, 0], a sparse-data degeneracy,
not proof of zero population efficiency or equivalence. For each arm, zero of
200 collisions with any recovered membership has an exact two-sided 95 percent
binomial interval [0, 0.0182753]. Two B trials within a collision are correlated;
they are not treated as 400 independent events for this bound. One training seed
does not measure seed-to-seed variability.

The continuum difference, projection minus joint, is minus 0.0175 with a
category-stratified collision-paired interval [minus 0.0375, 0.0025]. Thirteen
collisions are accepted only by joint, six only by projection, and 381 have equal
acceptance. This does not establish a background reduction. Equal-category
sampling is not a physical-mixture rate. This fresh cohort differs from Phase74;
47 versus Phase74's 31 for the identical control is not a model change.

Detector-state within-B relation correctness is 7865 versus 7631 of 12694.
Generated-state correctness is 9487 of 16472 versus 9699 of 17211; generated
supports differ across models. Latent deep source-set proposals are 87 versus
92 of 1503, and latent retained source sets are 16 versus 18 of 1503. These are
diagnostic latent merges, not verified legal physical hierarchy proposals.
Per-class and per-category counts are retained. Ignored ambiguous relation-pair
counts were not logged and remain unavailable, rather than zero.

What was controlled and verified
--------------------------------

The training-only Phase74 probe motivated a falsifiable gradient-conflict test,
not a presumption of benefit. Both arms reuse the exact same fixed Phase74
128-existing pretraining checkpoint: 1000 inherited pretraining updates and 8000
presentations, with zero new pretraining. Context width 128, hyperbolic width 32,
depth four, geometry, PID behavior, train-fitted normalization, common 256-wide
predicted-group-conditioned decoder, head initialization, optimizer and schedules
are matched. Encoders adapt jointly; downstream PID weights remain fixed.
Truth enters supervision only, after detector/model-generated states are formed.

Only the encoder relation-gradient component is corrected directly, per event,
before the unchanged global clipping and Adam rules. Decoder loss and gradient
routing rules remain unchanged; subsequent decoder updates may differ as models
diverge and the common clipping scale changes. Ordinary gradient projection does
not guarantee descent after optimizer preconditioning.

Both arms probe 20000 fitting presentations. Opposing gradient dots occur in
4760 joint-arm presentations and 4633 projection-arm presentations. Joint's
summed dot stays minus 1475.1149; projection changes its summed before-correction
dot from minus 590.1822 to 21447.5404 after correction. The per-event orthogonality
guard passes throughout. These sums are local optimization diagnostics, not
measures of physical efficiency. Local correction was implemented successfully
but failed to improve exact membership at the registered endpoints.

The joint control exactly reproduces all 1000 tiny and 1500 main training records
from Phase74, including losses, components and gradient norms. Both its final
encoder/model and decoder states are bitwise equal to the original control.
Terminal summary/checkpoint hashes, steps, optimizer/RNG state, architecture,
source/cache contracts, finite parameters, frozen PID and encoder adaptation
were verified for both arms. Checkpoints and threshold 0.5 were fixed in advance;
no best-checkpoint or threshold search was performed.

Data, compute and admission
---------------------------

The original 70000-event corpus remains fixed. The eligible training subset has
1536 events, 256 in each of the six study categories. With-replacement sampling
presents 1535 distinct events during main training; all 24 tiny events are seen.
Each arm performs 1000 independent tiny and 1500 main updates at batch eight:
20000 new fitting presentations. The inherited pretraining schedule presented
1526 distinct members of the same pool. Pool size, actual distinct events and
presentation counts are separately retained with authenticated order hashes.

A further 600 development identities, 100 per category, were designated before
outcome inspection. Full authentication excludes all training, primary,
supplementary, checkpoint-selection and prior development reservations, including
Phase74. No sealed test or primary reservation is used. Phase74 is now adaptive
exploratory evidence and is not reused as independent confirmation. The original
failed 60-event admission stays invalid and immutable.

The authenticated validation universe has 160000 events, including 20000 taupair
events not selected for this study. The six required categories contain 140000
identities, with 42571 available before the new designation. The immutable data-
design receipt inherited some Phase74 training descriptors; an explicit external
scope clarification preserves that receipt. Those descriptors did not configure
Phase75. Its separately preregistered frozen training contract specifies the two
128-wide arms, common checkpoint and zero new pretraining. No identity, training
setting or threshold was changed by this clarification.

Both jobs completed exit zero under two CPUs, 32 GiB and eight hours per arm,
without GPU or requeue. Scheduler elapsed times were 4738 and 4775 seconds.
Native fit/evaluation wall counters were 4711.79 and 4750.13 seconds; startup
admission/loading is outside those wall counters. Process peak RSS was 1966452
and 1887520 KiB. Each encoder has 1322658 parameters and each decoder 1385991.
Both record 36604461 fitting detector node-pair proxy units, 40000 fitting encoder
passes and 4320 evaluation passes, for 44320 total. The native encoder counter
covers fitting only; the complete download adds explicit evaluation accounting.
These are workload proxies, not measured FLOPs. Both arms run gradient probes;
projection adds correction work. Equal updates are not equal compute.

Recommendation and evidence
---------------------------

The tiny gate passes; the held-out membership-improvement gate fails. End this
single projection contrast without a coefficient grid, wider model, primary
scale-up or automatic successor. No capacity/pretraining winner emerged in
Phase74, and removing this measured gradient conflict did not rescue Phase75.
This does not prove missing encoder information or that every pretraining or
capacity intervention is ineffective. Data growth versus better pretraining
remains unidentified; neither is supported as an uncontrolled next step.

The next analysis should locate proposal/refinement and within-branch membership
errors at fixed corpus and controls, especially the failure to fit the main training
pool. Any use of these now-inspected development events is exploratory.
A new training study would need a new specific hypothesis and admission; none
is launched here. Physical hierarchy training requires the development gates,
and a complete primary evaluation additionally requires at least 2000 distinct
collisions in each of six categories per arm on an authenticated reserved cohort.

The :doc:`dashboard <_generated/status/index>` expands this latest completed study.
The :doc:`download catalogue <_generated/status/downloads>` retains the complete
Phase75 metrics, 5000 update records, sampling/compute counts and hashes in a
lossless envelope with a bounded standard-library decoder and integrity record.
All Phase74 and earlier downloads remain intact. Private identities, full
checkpoint states and scheduler receipts remain in the external overnight audit.
