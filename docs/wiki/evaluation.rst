Evaluation losses and metrics
=====================================

Three evaluations answer different questions: held-out objectives test the
training task, free rollout tests reconstruction from detector inputs, and
strict full-decay evaluation measures recovery of specified truth units.
The :doc:`dashboard <_generated/status/index>` keeps recorded exploratory
measurements separate from unavailable or not-run physics evaluations.

Validation losses and representation diagnostics
--------------------------------------------------------

:py:mod:`hypertagging.training.pretrain_trainer` evaluates the same weighted
components described in :doc:`training` on a fixed validation cohort, with
named curriculum views. FSP-only and truth-guided multilevel relation results
have separate denominators. :py:mod:`hypertagging.training.reconstruction_trainer`
separates teacher-forced validation losses from free-rollout performance.
Losses from different weights, masks or target populations are not directly
comparable. An inactive loss can be zero without indicating good predictions.

:py:mod:`hypertagging.losses.hyperbolic_pretraining` and
:py:mod:`hypertagging.evaluation.hierarchical_metrics` implement:

* Tree-relation accuracy: correct argmax relation labels over eligible pairs.
  Parent-ranking accuracy: true parent nearer than the selected safe negative,
  over children with a valid parent and an eligible negative. Negative counts,
  children without negatives and active fractions describe support.
* Tree-distance error and radius diagnostics: held-out geometry loss,
  parent-child radius monotonicity and radius/height correlation. Positive
  relation distance averages classes 1-3; negative distance averages classes
  4 and above. Root-depth and generation-height radius targets differ.
* Collapse diagnostics: mean/minimum tangent dimension standard deviation,
  off-diagonal covariance norm, effective rank and rank within level, node-kind
  and B-side groups. Effective rank is the exponential entropy of normalized
  singular values. Boundary fraction counts embeddings near the Poincare ball
  boundary. Branch angular separation averages angles across different sides.
* Channel retrieval and nearest-neighbor diagnostics: anchor/support counts,
  same-label neighbor fraction, unique-neighbor fraction and mean neighbor
  cosine, excluding self. Exact and structured channel similarities have
  different meanings; support and candidate pools accompany retrieval results.
* Leaf PID accuracy and predictive entropy: correctness on available labels
  and uncertainty of the predicted distribution. Neither includes unavailable
  truth labels as successful predictions.

Representation helpers may return zero for degenerate or empty support. Their
support counts must accompany interpretation; strict decay ratios instead use
``None`` when their denominator is zero.

Teacher-forced query metrics
------------------------------------

:py:mod:`hypertagging.evaluation.hierarchical_metrics` matches queries to targets
before computing mother-type and cardinality accuracy. Pointer precision is
correct selected daughter entries divided by predicted entries; recall divides
by target entries. Object/no-object accuracy thresholds sigmoid scores at 0.5.
Reports also retain true-positive/predicted/truth counts, query utilization,
matched counts and accuracy numerators/denominators. Query utilization counts
active predicted queries relative to query capacity, not reconstructed events.

:py:func:`hypertagging.losses.level_reconstruction.confidence_calibration_metrics`
reports Brier score (mean squared probability error) and expected calibration
error (occupancy-weighted absolute confidence/target gaps in probability bins).
These compare to the detached quality targets defined in :doc:`training`.
:py:mod:`hypertagging.evaluation.query_activation` adds object-logit, activation,
assignment-cost and gradient diagnostics for query collapse; these are
optimization diagnostics rather than reconstruction efficiencies.

Canonical rollout metrics
---------------------------------

The rollout ``micro_complete_target_efficiency`` matches eligible mothers by
recursive detector-source set and mother PID. It does not require matching
internal daughter topology, so its target-level percentage is distinct from
strict LCAG or full-event reconstruction. It is a rollout metric, not
teacher-forced query accuracy. Preserve these historical definitions when
comparing existing checkpoint tracks.

:py:mod:`hypertagging.evaluation.hierarchical_metrics` represents leaves by
source identity and mothers by PID plus recursively sorted daughter signatures.
Generated composite node numbers therefore do not affect canonical matching.

.. list-table::
   :header-rows: 1

   * - Metric family
     - Definition
   * - Edge precision, recall, F1
     - Canonical edge multiset overlap divided by predicted/truth edge counts;
       F1 is their harmonic mean. Mother signatures include PID. Empty-set
       conventions belong to this helper and differ from unavailable ratios.
   * - Full-tree / canonical-subtree exact match
     - Equality of canonical trees / overlap of canonical subtree signatures.
       Subtree overlap includes unchanged input leaves and can be high with
       no reconstructed edges. A valid tree can still have incorrect topology.
   * - Mother-type / leaf-assignment accuracy
     - Agreement of matched mother types / assigned leaf sources under the
       canonical comparison. Source-aligned diagnostics additionally use
       Hungarian source-set Jaccard matching and report its mean overlap.
   * - Recursive-source overlap
     - Mean source-set Jaccard of aligned predicted/truth nodes. Despite its
       name, this is an alignment overlap score, not a source-conflict count.
   * - Tree-edit-like distance
     - Canonical signature mismatch diagnostic; not a general optimal graph
       edit distance.
   * - First divergence / root success
     - First differing generation level; legacy root success means any shared
       root signature, including an isolated input leaf. It is not configured
       Upsilon construction or strict full-root recovery.
   * - Tree validity / p4 closure
     - Structural acyclicity, parent and level checks / mother p4 agreement with
       daughter sums at tolerance. These test construction consistency.
   * - Node counts / maximum level
     - Active predicted and truth multiplicity and generation height; useful
       for recognizing early stopping or excessive composition.

Strict full-decay units and denominators
------------------------------------------------

:py:mod:`hypertagging.evaluation.full_decay_metrics` is the current strict
definition. Full scope supplies one eligible retained root unit per event;
B-half scope supplies two units and separate event-level both-halves results.
Continuum uses explicit retained top-level components. ``checkpoint_direct``
targets use the checkpoint's ontology/policy. Unrepresentable direct targets
remain failed primary trials; contracted topology is a separate diagnostic.

``RatioMetric`` stores numerator, denominator and value. Summaries add counts
before division, including within source-category and target-shape groups.
Failed inference retains eligible truth denominators; unavailable truth has
zero support. Do not average per-event rates into micro efficiencies. For
example, 1/2 and 9/10 combine to 10/12, not the mean of 0.5 and 0.9.

.. list-table::
   :header-rows: 1

   * - Output
     - Meaning
   * - ``source_recall``, ``source_precision``
     - Matched detector source count over truth/predicted source count.
       Missing and extra source keys are recorded separately.
   * - ``structurally_valid``, ``target_representable``
     - Whether the reconstruction is structurally admissible and whether the
       direct target can be expressed by the selected policy. Reasons accompany
       unrepresentable targets.
   * - ``lcag_pair_accuracy``
     - Correct retained lowest-common-ancestor generation relations over truth
       source pairs after source alignment. Missing pairs do not become correct.
   * - ``perfectLCAG``
     - Exact source/topology recovery with valid structure. This topology
       criterion is distinct from the PID-aware canonical rollout tree metric.
   * - ``strict_missing_one_leaf`` / ``missing_one_particle``
     - Exactly one missing truth leaf, no extra sources and correct induced
       topology on the remaining leaves; the latter name is an alias. At least
       four truth leaves are required, leaving at least three for comparison.
   * - ``leave_one_out_lcag``
     - Remove one truth source from both trees and require equal remaining
       source sets and induced topology. Includes exact and strict-missing-one
       cases; eligibility requires at least four truth leaves.
   * - Leaf / mother / root PID accuracy
     - PID correctness after source/topology alignment, on available labels.
       Mother PID coverage measures aligned mothers relative to truth mothers;
       missing alignment is not hidden by reporting conditional accuracy alone.
   * - Confusions and topology counts
     - Leaf/mother truth-predicted PID counts, matched/unmatched mother counts,
       retained depths and leaf multiplicities support stratified analysis.

Kinematics after alignment
----------------------------------

Source/topology matching precedes PID or p4 scoring. Kinematic values never
choose the alignment. ``KinematicErrorMetrics`` reports alignment coverage,
component-mean p4 L1, vector momentum error (p3), relative p3, p3 RMSE,
px/py/pz/energy signed biases, component MAE/RMSE, mass absolute error/bias/RMSE
and absolute momentum-magnitude error. ``energy_error`` aliases energy MAE.
RMSE takes the square root after aggregation of squared errors.

Leaf PID uses all predicted FSPs matched by source, independently of whether
they were linked into the selected reconstructed root.

The default reference is reconstructed-FSP daughter-sum p4. Physical MC
composite momentum error is unavailable when the preprocessed schema does not
retain that reference; the report records the reason. These residuals and
closure must not be presented as measured detector momentum resolution.

Beam and oracle evaluation
----------------------------------

:py:mod:`hypertagging.evaluation.beam_decay_metrics` receives an immutable
model-ranked candidate list after inference. Beam top-1 is deployable ranking
performance; oracle@K asks whether a better truth match occurs in the returned
first K candidates, bounded by the actual candidate count.

Oracle outputs include perfect LCAG, maximum source recall, maximum mother PID
coverage, reciprocal first-exact rank, and mean first-exact rank conditional
on recovery. Unrecovered eligible units contribute zero reciprocal rank.
Unit oracles may recover separate components in different candidates;
``coherent_event_perfect_lcag`` requires all components in one hypothesis.
Both-halves perfect LCAG likewise requires a coherent candidate. First exact
ranks/scores, candidate-count distributions and requested/evaluated K are
retained. All ratios preserve sufficient statistics for micro aggregation.

:py:mod:`hypertagging.evaluation.full_decay_runner` writes event/unit rows,
summaries and beam diagnostics. ``scripts/evaluate_full_decay.py`` and
``scripts/evaluate_hyperbolic_pretrain.py`` appear in the
:doc:`script catalogue <_generated/catalog/index>`. Greedy, beam top-1 and oracle
namespaces are kept separate. Search bounds and target population must accompany
any comparison.

Historical evaluation compatibility
-------------------------------------------

:py:mod:`hypertagging.evaluation.grafei_metrics` retains padded per-level PDG
correctness and summed absolute feature errors, exact LCA equality, and event
columns ``nleaves``, ``depth``, ``pdgAcc``, ``featErr``, ``perfect``, ``failed``
and ``sigProb``. Failed rows use the historical zero values. These padding and
failure conventions differ from strict full-decay unavailable denominators.

The legacy loss helpers in :py:mod:`hypertagging.losses.reconstruction_losses`
report PDG accuracy and momentum MAE alongside their training losses;
:py:mod:`hypertagging.losses.link_losses` reports masked argmax link accuracy
or transfer agreement. Their exact loss variants are listed in :doc:`training`.
Historical ``scripts/evaluate_reconstruction.py`` consumes the GraFEI path and
does not replace the strict evaluator.
