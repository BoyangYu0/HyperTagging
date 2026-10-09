Evaluation losses and metrics
=====================================

Three evaluations answer different questions: held-out objectives test the
training task, free rollout tests reconstruction from detector inputs, and
strict full-decay evaluation measures recovery of specified truth units.
The :doc:`dashboard <_generated/status/index>` keeps recorded exploratory
measurements separate from unavailable or not-run physics evaluations.

Tag efficiency and inclusive efficiency
---------------------------------------

Every new strict evaluation exports ``tag_efficiency``. For N generic B-pair
collisions there are 2N true-B trials. If N1 events recover one distinct true B
and N2 recover both, per-B efficiency is (N1 + 2N2) divided by 2N, and event efficiency
is (N1 + N2) divided by N. Duplicate candidates never increase the success count.

* **Exact tag efficiency:** an accepted B candidate has exactly the retained
  detector sources and the unordered recursive truth topology and PID,
  including the B root and leaves. The other B need not be correct.
* **Inclusive efficiency** (``inclusive_fsp_grouping``): an actual reconstructed
  composite groups all and only one true B's retained FSP sources. PID and
  internal topology are ignored, including the group's PID and wrappers above
  it. Disconnected components cannot be joined after inference to create a
  success, and a group containing both Bs cannot recover either individually.

Report greedy, model-ranked top-1 and retained beam-pool results separately.
Each true B counts at most once at each K; beam width does not enlarge the
2N denominator. Pool recovery is truth-evaluated oracle recall. Recovering two
Bs in incompatible hypotheses does not establish a coherent reconstructed pair.
Truth is never used to generate or rank candidates.

For new studies, candidate-survival diagnostics must distinguish target
eligibility, threshold support, generated proposals, local pruning, forest
survival and final top-1/pool membership. Traces are truth-free; truth joins occur
after search. Surviving clean-root coverage is an optimistic bound, not proof of
legal reachability. The measured structural-study policy is recorded in
``docs/assembly_diagnosis_20261007.md`` and
``configs/reconstruction/next_study_policy.json`` in the repository.

Missing target membership remains unavailable, with its nominal trial retained;
proven success over nominal trials is a lower bound when truth is unavailable.
PID availability is separate: inclusive grouping may be measurable when exact
matching is unavailable. Report counts, denominators, unavailable counts and
covered flags by B channel, with readable retained signatures and stored IDs.

Continuum has no true B pair. Its two nominal attempts per collision measure
fake-B acceptance, separately from retained-component exact/inclusive recovery
and source-type coverage. Original q/qbar efficiencies are unavailable without
parton-to-FSP ancestry. These retained-source proxies are not physical FEI
matching; comparisons require common truth, inputs, selections and working points.
Historical studies are not retroactively assigned the newly added metrics.

The complete metric contract is maintained in
``docs/full_decay_reconstruction_evaluation.md`` and the implementation in
:py:mod:`hypertagging.evaluation.tag_efficiency`.

Complete final evaluation policy
--------------------------------

Every model or arm needs at least **2,000 distinct processed collision events
in each of charged, mixed, ccbar, uubar, ddbar and ssbar**: at least12,000 per
arm, using the same immutable category-stratified cohort across arms and scopes.
These are collision counts, not pooled categories, candidates, B trials or
planned reservations. Fresh deterministic ledger reservations must be disjoint
from training, fitting, checkpoint selection and prior adaptive reservations.
Capacity shortages must be reported; never duplicate events, lower the quota,
condition on easy truth, or open sealed test data. Smaller native, beam, oracle
and auxiliary samples remain visibly diagnostic with their actual category counts.
Nominal unavailable-truth trials and failed-event accounting are retained.

Latest measured efficiencies: Phase70
-------------------------------------

The :doc:`Phase70 decoder evaluation <phase70>` is the current
complete greedy evaluation: 2,000 processed collisions in each of six source
categories, 12,000 per arm and scope. Its 4,000 charged/mixed B-pair collisions
supply 8,000 nominal B trials and 4,000 event trials. Continuum supplies the
remaining 8,000 collisions and has separate fake-B and component metrics.

In both full and half scope, exact retained tagging is **0/8,000** for both
arms. Inclusive grouping is **0/8,000** with bias enabled and **1/8,000 (0.0125%)**
with bias disabled; event-any recovery is respectively **0/4,000** and
**1/4,000 (0.025%)**. Neither arm recovers both Bs in one event. Twenty-four exact
truth trials remain unavailable within the nominal denominators; inclusive
membership is available for all8,000 B trials. Full and half views reuse the same collisions.

The 60-event-per-arm beam is diagnostic: ten collisions per category, with
0/40 exact and inclusive B successes in each arm/scope for top1 and retained
pool. It does not satisfy the complete-evaluation quota. The original
100-event strict and 20-event beam measurements are historical; their smaller
samples must not be pooled with this supplementary cohort. A single inclusive
success does not establish a recursive-quality advantage, and zero observed
successes do not prove zero population efficiency.

See the :doc:`dashboard <_generated/status/index>` for efficiency-first results,
all-study summaries and the complete download catalogue.

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

Flat-membership development availability
----------------------------------------

:doc:`Phase74 <phase74>` evaluates raw and constraint-accepted retained source
memberships on600 development collisions, not policy-sized primary reconstruction.
The flat interface has no physical tree, exact-tree top1, physical beam pool or
p4 closure; those quantities are unavailable. Latent source-set merge survival
is a separate diagnostic. Relation accuracy is conditional on identifiable
within-B pairs; the number of ignored ambiguous pairs was not logged and remains
unavailable. Per-B denominators, collision-any/both counts and continuum acceptance
remain separate. See the full evaluation contract for these definitions.

Phase76 proposal/refinement diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detached model-generated partitions are joined to truth only after generation.
Per-target missing and extra detector sources use a single event-level optimal
B-slot permutation, with unassigned fixed. The best-overlap diagnostic may choose
the better predicted group for each target; it is an optimistic overlap summary,
not exact reconstruction or a globally matched physical hierarchy. Raw exact
source membership and constraint-accepted membership retain separate numerators.
First-error counts distinguish absent raw targets from rejection of raw-exact
targets. Presence, cardinality, charge and source constraints remain unchanged.

Post-hoc continuum membership uses explicit top-level retained composite source
sets, excluding singleton roots, and keeps unavailable mappings separate. It
compares each component against individual predicted flat groups without joining
disconnected groups. This newly measured diagnostic is distinct from fake-B
acceptance and does not supply physical component trees or quark ancestry.

Training-only proposal audit
-----------------------------

:doc:`Phase77 <phase77>` separates foreground omissions from conditional B-slot
confusion using detached logits and supervision joined afterwards. Aggregate-B
probability decisions and truth-assisted oracle corrections are diagnostics,
not deployed thresholds or physical reachability. Keep source, collision and
correlated-B supports separate. Native total-loss and diagnostic hard-error
permutations optimize different criteria. Training-only paired intervals do not
provide fresh confirmation, and zero-success bootstrap bounds are not equality.

Phase78 pair diagnostic scope
------------------------------

Same-B versus cross-B pair discrimination is conditioned on two known B constituents
within one collision and disjoint detector sources. Unknown/shared-source support
is counted separately. Event-mean AUC and pooled-pair AUC have different weights.
Probe fitting and assessment are event-disjoint training-role subsets; both were
seen by the frozen scientific model, so this is not fresh confirmation. The
majority-size comparator and conditional-optimal permutation are truth-assisted
diagnostics, not deployable exact membership. Background pair score frequencies
are not accepted fake-B events. Full per-event metrics and finite fitting curves
remain lossless; physical hierarchy endpoints stay unavailable.
