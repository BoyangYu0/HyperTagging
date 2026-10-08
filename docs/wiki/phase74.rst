Phase74 development factorial review
========================================

Four matched-history development arms completed: contextual widths 128 and 256,
each with existing or assembly-targeted pretraining. Hyperbolic width 32, depth
four, curvature and radius convention remained fixed. This is a 1,536-event
subset of the original 70,000 training corpus, with 600 fresh held-out development
collisions, 100 per category. It is not a complete primary evaluation.

Verified results
----------------

.. list-table:: Fixed-final source-membership results
   :header-rows: 1
   :widths: auto

   * - Arm
     - Tiny raw; accepted
     - Main train raw
     - Held-out raw; accepted
     - Continuum accepted
   * - 128 existing
     - 32; 27 of 32
     - 0 of 1024
     - 0; 0 of 400
     - 31 of 400
   * - 128 assembly
     - 32; 27 of 32
     - 0 of 1024
     - 0; 0 of 400
     - 24 of 400
   * - 256 existing
     - 32; 27 of 32
     - 0 of 1024
     - 0; 0 of 400
     - 52 of 400
   * - 256 assembly
     - 32; 27 of 32
     - 1 of 1024
     - 1; 1 of 400
     - 72 of 400

All arms pass the 95 percent tiny raw gate. Five tiny raw memberships fail the
unchanged charge guard in every arm; their retained target source sets themselves
have unsupported charge. Accepted tiny efficiency is therefore not identical to
memorization. Zero accepted source conflicts were measured. All 600 held-out
collisions processed; no failure, replacement or unavailable membership target.
The four continuum categories each contribute 100 collisions; charged and mixed
each contribute 200 nominal B trials. Category, channel, size and relation counts
are preserved in the complete download.

The single held-out recovery in 256 assembly is a charged-category, three-FSP
retained source set, charge minus one, whose matching truth node has generation
height two. Its retained-channel identifier is 975817919630024005. It is not a
physical B-tree reconstruction. There is no evidence of a capacity/pretraining
winner: one recovered source set accompanies increased continuum acceptance.
Collision-paired factorial effects, interaction, discordant counts and exact
sparse event bounds condition on these four fitted models; one seed does not
measure training-run variability. Correlated B trials are resampled together.
The width effect on accepted membership is 0.00125 with paired interval
[0, 0.00375]; the objective effect has the same estimate and interval. Their
interaction is 0.0025 [0, 0.0075]. These sparse bootstrap intervals do not establish
positive population effects. Continuum acceptance has a width effect of 0.08625
[0.06125, 0.11375], objective effect 0.01625 [-0.0025, 0.035], and interaction
0.0675 [0.0275, 0.1075]. These are differences in fractions on equal-category
samples, not physical-mixture rates or seed-to-seed intervals.

What the models actually learned
--------------------------------

The existing objective retains the native pretraining curriculum. Assembly adds
optional unordered global memberships with an unassigned class and categorical
within-B parent, sibling and other-relative discrimination. Truth is supervision
only. Detector input and model-generated latent partial partitions are separate
from the baseline teacher/corruption views. Predicted soft memberships and pooled
partial groups condition a common 256-wide decoder; 128-wide features are padded
without losing information. Objective arms share initial states within width.
Both widths have fresh, matched finite pretraining histories. The downstream
head resets identically and every encoder adapts, with PID-head weights frozen.

Held-out latent deep source-set proposal counts are 86, 90, 85 and 94 of 1,469;
retention is 4, 10, 10 and 15 respectively. These are diagnostic source-set merges,
not verified legal physical deep proposals. Physical exact-tree top1, beam-pool,
physical deep survival and daughter-sum closure are unavailable because this flat
interface builds neither physical mothers nor competing physical beam hypotheses.
Inclusive top1 denotes membership of the predicted partition only. Continuum
retained-component recovery is unavailable for this optional-B grouping head.
There is no FEI-equivalent efficiency claim.

Actual compute and finite selection
-----------------------------------

Each arm uses 1,000 pretraining updates, an independent 1,000-update tiny branch,
and 1,500 main downstream updates, batch eight: 28,000 fit presentations. Tiny
training cannot alter the main model. Final checkpoints and presence threshold
0.5 were fixed in advance. Original train-fitted normalization is unchanged.
Encoder parameters are 1,322,658 and 5,176,610; the common decoder has 1,385,991.
Wall times in arm order are 5,207.36, 5,700.68, 6,883.21 and 7,601.44 seconds.
All four jobs completed within their two-CPU, 32-GiB, eight-hour bounds. Actual
views, presentations, node-pair proxies, peak memory and every logged loss are
exported. Equal updates are not equal FLOPs. No GPU was used.

Cumulative recommendation and one bounded successor
---------------------------------------------------

Earlier search probes had no correct complete B pool for a reranker to rescue.
Phase72 improved shallow component/background measures without an established
B-efficiency gain. Phase73 memorized tiny memberships but failed held-out recovery
in both frozen and adapted arms. Phase74 controls the previously confounded
capacity/pretraining histories yet still fits very few exact main-train groups.
This does not prove missing encoder information, adequate capacity, data scarcity
or that all targeted pretraining is ineffective. More unique data versus better
pretraining remains unidentified. Keep the 70,000 corpus fixed; neither uncontrolled
training growth nor merely longer unchanged pretraining is justified.

A training-only mechanism probe found negative encoder-gradient dot products
between membership and relation losses in 17 of 32 baseline events; mean cosine
was 0.021. This local association is not causal proof. One preregistered two-arm
Phase75 development study tests ordinary joint gradients against removal of only
the relation-gradient component opposing the membership gradient. Decoder gradients,
scalar losses, initialization, data order and finite schedules remain matched.
Projection acts on ordinary gradients before Adam preconditioning and clipping;
it does not guarantee that every optimizer step lowers membership loss.
Both arms use the fixed Phase74 128-existing pretraining checkpoint and identical
fresh heads, not a downstream model selected for the lone held-out success.

The successor designates a further 600 fresh non-primary development identities,
100 per category, excluding every historical reservation, checkpoint-selection
cohort, original training identity and all Phase74 held-out identities. Phase74
is now adaptive exploratory evidence and is not reused as independent confirmation.
Fixed final checkpoints and threshold 0.5 remain unchanged. Each arm has 1,000 tiny
and 1,500 downstream updates of batch eight, no additional pretraining, at most
two CPUs, 32 GiB and eight hours. There is no automatic requeue or further successor.
Both jobs passed guarded admission and were accepted and started on 9 October
2026. Results are pending in this publication snapshot; no further successor is
authorized. The final overnight report retains scheduler and startup receipts.

Primary scale-up remains closed. Require count-backed held-out membership gain,
background and validity controls as well as tiny raw memorization. A primary study
would additionally need at least 2,000 distinct collisions in each of six categories
per arm on an authenticated reserved cohort. No primary reservation, wider model,
sealed-test access or production hierarchy is part of this pass.

Evidence and downloads
----------------------

The :doc:`dashboard <_generated/status/index>` expands this latest completed study;
:doc:`earlier studies <studies>` and every historical download remain intact.
The :doc:`download catalogue <_generated/status/downloads>` includes the lossless
Phase74 aggregate/history envelope, standard-library decoder and integrity record.
It retains all 14,000 step records and all available role/category/channel/size,
relation, latent-depth, uncertainty and compute results. Private identities,
source positions, checkpoints and scheduler receipts stay in the external audit.
Publication retains fixed file/site/privacy budgets and checks decoded hashes.
