Training stages and loss functions
==========================================

Training first learns detector and tree representations, then learns to build
an unordered set of mothers at each generation height. The shared encoder
uses detector-specific adapters and contextual hyperbolic features;
:py:mod:`hypertagging.models.level_autoregressive` predicts leaf PID, rebuilds
PID-dependent quantities, refines context and predicts mother queries.
The :doc:`dataset guide <dataset>` defines inputs, masks and target populations.

Pretraining curriculum
------------------------------

:py:mod:`hypertagging.training.pretraining_curriculum` implements four default
phases, with fractions 0.20, 0.25, 0.30 and 0.25. Scheduling can use optimizer
steps, presentation virtual steps or events; these units are not interchangeable.

#. ``fsp_topology_anticollapse`` uses detector final-state particles, relation
   classification, variance/covariance regularization and leaf PID.
#. ``truth_guided_distance_radius`` introduces retained multilevel composites,
   directed parent ranking, exact tree distance and radius supervision.
#. ``multilevel_channel_memory`` adds cross-event channel representation learning.
#. ``corrupted_composites_hard_negatives`` perturbs composites and adds corruption
   classification, candidate correctness and hard-negative learning. Its channel
   memory enqueue is disabled, so corrupt examples do not populate that memory.

Truth-guided composites are a training view. Strict inference starts from
detector leaves and constructs every composite itself. Enabled phase objectives,
ablations and configured weights determine which computed terms contribute.

Hyperbolic objectives
-----------------------------

:py:mod:`hypertagging.losses.hyperbolic_pretraining` returns a weighted sum
:math:`L = \sum_k w_k L_k` plus component diagnostics. Hyperbolic and
variance/covariance numerical kernels operate in FP32.

.. list-table::
   :header-rows: 1

   * - Component
     - Definition and support
   * - ``lca``
     - Cross-entropy for exact retained-tree relation classes; averages the
       loss of each present class equally to limit pair-class imbalance.
   * - ``parent``
     - Margin ranking makes a child's true parent closer than a topology-safe
       negative. Self, ancestors/descendants, siblings and near-positive
       relations are excluded from negatives. Unsupported children do not
       contribute; eligible-negative coverage is reported.
   * - ``tree_distance``
     - Smooth-L1 regression of hyperbolic distance against a fixed-scale,
       logarithmic exact retained-edge path target. Height differences alone
       cannot specify this distance.
   * - ``depth``
     - Masked radius MSE uses generation-height or exact-root-depth targets.
       Weak/learned mode instead penalizes radii outside the configured range.
       The leaf-outward convention puts leaves farther from the origin.
   * - ``channel``
     - Cross-event branch embedding contrastive loss plus weighted Smooth-L1
       structured-similarity regression, using full-truth/reconstructable
       identities and branch counts, optionally with memory. The compatibility
       channel objective uses similarity MSE within its supplied pool.
   * - ``var``
     - Tangent-space standard-deviation hinge discourages collapsed dimensions;
       the target can scale with embedding dimension. Balanced samples account
       for level, node kind, event and branch.
   * - ``cov``
     - Squared off-diagonal tangent covariance discourages redundant dimensions
       over the same masked balanced sample.
   * - ``same_mother``, ``same_branch``
     - Optional pairwise binary cross-entropy auxiliaries; their default weights
       are zero. They do not replace the principal relation objective.

:py:mod:`hypertagging.training.pretrain_trainer` adds leaf PID cross-entropy
on available truth labels, corruption-type cross-entropy on composite nodes,
candidate-correctness BCE for uncorrupted versus corrupted composites, and a
hard-negative mean hinge ``relu(margin - distance)`` for sampled corrupt pairs.
These auxiliaries are
phase-gated. The trainer records gradients, support counts and collapse
diagnostics so a small loss from an empty mask is distinguishable from learning.

Reconstruction objective and matching
---------------------------------------------

Mothers at a level are an unordered set. In
:py:mod:`hypertagging.losses.set_matching`, Hungarian assignment minimizes type
negative log probability plus soft daughter-set Jaccard cost, with optional
object, cardinality, charge and p4 costs. Production uses SciPy. Assignment
selects query-target pairs; the matching cost is not an extra training loss.
Capacity overflow is explicit rather than silently dropping excess targets.

:py:mod:`hypertagging.losses.level_reconstruction` then computes:

.. list-table::
   :header-rows: 1

   * - Component
     - Definition
   * - ``object``
     - Positive-weighted focal BCE for matched versus unmatched queries.
       Unrepresentable targets can mask uncertain unmatched negatives.
   * - ``type``
     - Mother PID/type cross-entropy on matched queries.
   * - ``pointer``
     - Positive-weighted focal BCE on eligible context daughters for each match.
   * - ``cardinality``
     - Cross-entropy for the number of daughters of each matched mother.
   * - ``confidence``
     - BCE to a detached target: daughter-set Jaccard at pointer threshold 0.5,
       multiplied by correct mother type and within-candidate source-conflict
       validity; unmatched targets are zero. This target precedes inference
       cardinality/top-k, configured pointer threshold, charge and cross-query
       selection. Its calibration is not calibration of completed beam trees.
       Rollout uses learned confidence only when marked trained.
   * - ``physics``
     - Scaled mean squared p4 residual plus charge MSE between soft pointer sums
       and the truth-selected sum of current PID-refined daughters. Stored
       future-mother p4 does not supply this target. There is no free mother p4
       regression head or MC mother momentum target.
   * - ``source_conflict``
     - Differentiable penalty for selecting recursively overlapping sources,
       weighted by query activity and the configured conflict policy.
   * - ``mother_charge``
     - Policy-weighted absolute residual between soft daughter charge and the
       matched mother type's expected charge, enabled for soft charge policies.
   * - ``query_repulsion``
     - Penalizes cosine similarity of active matched query pointer proposals
       whose target daughter sets are disjoint; reduces duplicate proposals.

The daughter-sum terms are implemented in :py:mod:`hypertagging.losses.physics`.
:py:mod:`hypertagging.training.reconstruction_trainer` mixes teacher contexts
and predicted contexts according to its schedule. It adds leaf PID CE once per
event's first selected level, optional auxiliary teacher loss, and a recovery
``softplus(-object_logit)`` loss for missing predicted-context mothers. Recovery
uses the highest object logits up to the missing count. These terms complement
the matched level objective; a teacher-forced score does not measure free rollout.

Validation and checkpoint selection
-------------------------------------------

``scripts/train_hyperbolic_pretrain.py`` and
``scripts/train_level_reconstruction.py`` are indexed in the
:doc:`script catalogue <_generated/catalog/index>`. Configuration precedence is
defaults, YAML, then explicitly supplied CLI flags. Architecture, PID/features,
normalizers, target/constraint policies, sampling and curriculum state are
checkpoint contracts. Encoder transfer measures key/shape coverage.

Pretraining keeps the configured full-objective best checkpoint and separate
optional diagnostic tracks. Reconstruction keeps teacher-forced loss and
rollout edge-F1/validity best tracks independently. Primary rollout selection
requires nonzero rollout/edge/p4 denominators, validity at least 0.999, complete
p4 closure at the configured tolerance and zero source conflicts. Validation
cohorts are fixed; optimizer, scheduler, scaler and RNG states support resume.
Exact mid-epoch replay is supported with zero loader workers.

Compatibility loss families
-----------------------------------

The following implemented objectives belong to historical paths and have their
own scaling; they are not additional terms in the current level objective.

* :py:mod:`hypertagging.losses.embedding_losses`: GraFEI and Toy-MC radius MSE
  against mass/energy-derived targets, Colab radius L1, grouped intra-event
  concentration/variance, inter-pattern or inter-channel separation, and the
  historical VICReg-style similarity/variance/covariance sum. Connection losses
  use weighted BCE on pair predictions or cosine-derived predictions.
  ``CompositeLoss`` combines registered objectives with their supplied weights.
* :py:mod:`hypertagging.losses.link_losses`: masked link cross-entropy and
  argmax accuracy; transfer-link MSE and argmax agreement.
* :py:mod:`hypertagging.losses.reconstruction_losses`: PDG cross-entropy,
  spatial-weighted momentum MSE (weight three in GraFEI, one in the reduced
  variant), plain momentum MSE, embedding MSE and mean cosine distance.
* :py:mod:`hypertagging.losses.gpt_losses`: masked reconstruction distance
  ``10 * MSE + L1`` on nonzero particle targets for the L1 term, and the
  historical autoregressive radius L1 objective.

See :doc:`evaluation` for reductions, accuracies and diagnostics, and the
:doc:`dashboard <_generated/status/index>` for recorded measurements.
