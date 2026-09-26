Phase60: Early pretraining failure before reconstruction
========================================================

Both Phase60 jobs failed before reconstruction. Each completed 546 pretraining
optimizer steps and 17,472 event presentations, then stopped before update 547
because the weighted shared-encoder PID-to-LCA gradient ratio was 28.020851,
above the unchanged limit of 20. The failure reports and recorded scientific
training histories are identical across the two arms, excluding runtime timing,
throughput and memory measurements.

The planned late-PID weights of 0.2 and 0.1 never executed. Both arms were still
in the first curriculum phase with PID weight 1.0. Parent ranking was inactive
in that phase. This is an early objective-stability failure, not evidence that
one late-PID setting is better. Neither arm produced a refined checkpoint,
a reconstruction checkpoint, a validation metric record or a strict result.
Original failed receipts and all native artifacts are preserved.

Phase60 complete available metrics
----------------------------------

The :doc:`dashboard <_generated/status/index>` provides the complete public
metric download: every numeric objective-failure diagnostic, every logged
metric's count, first, last, minimum, maximum and mean, and explicit full/half
availability for every evaluation view. The private complete-metrics archive
also includes all **97,546 recorded scalar values**, raw histories, objective
failure reports, source hashes, scheduler receipts and verification evidence.
Step means describe these logs; they are not global micro efficiencies or
independent statistical replications.

.. list-table::
   :header-rows: 1

   * - Recorded quantity
     - Planned late PID 0.2
     - Planned late PID 0.1
   * - Completed pretraining steps
     - 546 / 2188
     - 546 / 2188
   * - Attempted step rejected before update
     - 547
     - 547
   * - Training event presentations
     - 17472
     - 17472
   * - Weighted shared PID gradient norm
     - 5.702252865
     - 5.702252865
   * - Weighted shared LCA gradient norm
     - 0.203500345
     - 0.203500345
   * - PID / LCA ratio
     - 28.020851046
     - 28.020851046
   * - Guard threshold
     - 20
     - 20
   * - Completed reconstruction steps
     - 0
     - 0
   * - Validation metric records
     - 0
     - 0
   * - Strict events scored
     - 0
     - 0
   * - Beam candidates generated
     - 0
     - 0

The last completed training batch has total loss 1.414527, LCA loss 1.008405,
leaf-PID loss 0.406120 and effective rank 23.781219 in both arms. These are
training diagnostics, not held-out reconstruction or representation quality.
The rejected batch has PID loss 0.522857 and LCA loss 1.018887; its gradient
ratio, rather than its loss ratio, triggers the failure.

Full, half and beam reconstruction availability
-----------------------------------------------

No Phase60 tree exists to score. All requested metrics are explicitly
**UNAVAILABLE_NO_RECONSTRUCTION_CHECKPOINT** with null numerator, denominator
and value. They are not scored failures, zero efficiencies or skipped trees.

.. list-table::
   :header-rows: 1

   * - Evaluation view
     - Full, both arms
     - Half, both arms
   * - Primary direct
     - UNAVAILABLE
     - UNAVAILABLE
   * - Identical strict repeat
     - UNAVAILABLE
     - UNAVAILABLE
   * - Independent complete-target checkpoint
     - UNAVAILABLE
     - UNAVAILABLE
   * - Independent depth checkpoint
     - UNAVAILABLE
     - UNAVAILABLE
   * - Independent tree-validity checkpoint
     - UNAVAILABLE
     - UNAVAILABLE
   * - Contracted diagnostic
     - UNAVAILABLE
     - UNAVAILABLE
   * - Beam direct, every candidate
     - UNAVAILABLE
     - UNAVAILABLE

This covers source precision/recall, LCAG, exact nontrivial components,
mother/root/daughter PID, configured roots, coherent forests, target
representability, depth, validity, source conflicts, micro/macro aggregates,
all deployable beam rankers and post-inference oracle diagnostics. Paired
reconstruction uncertainty is unavailable because no paired predictions exist.
Daughter-sum p4 closure would check implementation consistency; physical
momentum resolution remains unmeasured.

Data integrity and validation reservation
-----------------------------------------

Both jobs passed the immutable contract and GPU preflights, and the actual
validation-selector preflight passed before training: 1,000 ordered selection
events, zero strict overlap and all other 99,000 validation events excluded.
The strict reservation contains 100 events; beam uses its first 20 if reached.

Phase60 had added 50,000 validation events from source files disjoint from the
entire previous inventory, including the initial pretraining pool and sealed
test. Training remains the same 70,000 events. The test payload was not opened.
The full index and train-only normalizer are unchanged for the next study.
Both arms retain the shared level-5 daughter capacity of 14, global limit 16,
and all original acceptance thresholds.

Neither validation metrics nor strict predictions were produced. Phase61
therefore reuses the authenticated reservation and seed without consulting
strict outcomes. All 1,100 reserved events remain reserved, leaving 48,900 of
the added validation events unreserved. This is not a fresh independent seed
replication, and the early-weight choice is adaptive to training diagnostics.

Decision across the accumulated studies
---------------------------------------

* **Training dataset size:** hold 70,000 events. Phase40 changed data size,
  compute and cohort together and supplies no controlled learning curve.
  Phase60 fails before reconstruction; adding training examples does not
  directly address the observed gradient imbalance. No additional validation
  expansion is needed for this pilot after Phase60's 50,000-event addition.
* **Longer pretraining:** defer it. Phases34/42 did not establish a reliable
  benefit from duration alone. Phase60 cannot complete the current budget;
  increasing that budget is not a supported remedy.
* **Pretraining quality and stability:** prioritize a bounded objective-stability
  experiment. Phase41's pointer-weight change, mixed adaptation results in
  Phases43/46/47, corrected PID-update studies49–51 and recovery-dose studies52–54
  did not establish reproducible deep-tree gains. Phase48's intended PID update
  did not execute. Phase55's parent-weight effects were shallow and mixed;
  failed controls in Phases56/58 prevented matched conclusions. Phase57 strict
  contamination prevents an independent quality claim. Phase59 completed both
  arms but only established feasibility on 25 events, with no full roots and
  shallow inconclusive gains. Phase60 adds evidence of seed-dependent early
  instability, not a quality comparison.

Different cohorts, seeds and runtime histories prohibit causal comparisons of
raw rates across phases. Improving pretraining remains a hypothesis; its
advantage over training-set growth has not been measured in a controlled
head-to-head experiment. Stable execution is necessary to test that hypothesis,
but is not itself evidence of better reconstruction. Target-policy
incompatibilities also remain separate from representation quality.

Phase61: Shared early-PID stability pilot
-----------------------------------------

Both arms restart from the same original step-81096 parameters with fresh
train-only normalization, optimizer, schedule, RNG and memory. They use the
same seed 20260927, training payloads and validation reservation as Phase60.
The first two PID weights become **0.5, 0.5** in both arms. The later weights
remain **0.2, 0.2** for control and **0.1, 0.1** for candidate, with parent
weight 2. Budgets remain 2,188 pretraining and 4,376 reconstruction steps.

At fixed gradients, halving PID weight halves the observed PID/LCA ratio.
The new trajectories will differ, so this does not guarantee passing the guard.
The dominance threshold stays 20 with action ``fail``; no failure condition,
capacity check, cohort isolation check or reconstruction gate is relaxed.

This is one bounded adaptive stability pilot, not an independent confirmation
of Phase59 and not a sweep. If training completes, retain all original and
retained full/half metrics, every returned beam candidate, checkpoint tracks,
strict repeats and uncertainty. Require joint topology and coherent-forest
evidence before claiming quality benefit. No promotion, sealed-test access,
automatic follow-on or longer campaign is authorized.

The dashboard records the Phase61 submission snapshot. Submission is separate
from startup and training completion.

Phase61 submission and verification
-----------------------------------

Both Phase61 jobs were submitted from immutable source
``9d235fda39108b739b4c8e55b89dc9a4aff399c7``, tagged
``reconstruction-phase61-early-pid-stability-pilot-20260926``.
Submission receipts, scheduler records and bound inputs were verified.
These are submitted experiments, not completed training results.

The frozen source passed 27 preflight tests. All 193 focused reconstruction/search
tests and both two-step CPU training smokes passed. The broad CPU suite
(excluding the separately checked dashboard tests) passed 1,777 tests with
34 skips. The dashboard run had one stale source-count assertion, which was
corrected and passed a targeted rerun; its other 136 tests passed. Independent verification
matched every exported scalar to its native source, recomputed all history
summaries, and rehashed all 34 selected train/validation shards. All 50 earlier
metric downloads are preserved byte for byte. Publication is subject to the
complete CPU CI suite, strict documentation builds and live download checks.
