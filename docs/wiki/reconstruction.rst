Full reconstruction and beam search
===========================================

Reconstruction converts a forest of detector final-state particles into
candidate decay trees, one generation height at a time. The strict entry point
is :py:mod:`hypertagging.reconstruction.hierarchical_inference`; the CLI is
``scripts/evaluate_full_decay.py`` in the
:doc:`script catalogue <_generated/catalog/index>`. Offline inference is CPU-only,
with model evaluation and inference mode enabled.

Detector projection and two-pass PID
--------------------------------------------

The entry point physically projects the event to detector FSPs and removes
stored truth composites, ancestry, geometry, truth PID, completeness and channel
labels. An evaluation-only source-key map stays outside the model. This lets
the same event carry rich training targets without making those targets
available to candidate generation, ranking or stopping.

:py:mod:`hypertagging.models.heterogeneous` adapts detector blocks and builds
shared context. :py:mod:`hypertagging.models.level_autoregressive` first predicts
raw-track PID with charge-compatible choices, reconstructs PID-dependent p4,
then runs refined context and mother/pointer prediction.
:py:mod:`hypertagging.reconstruction.pid_state` owns persistent PID state.
In ``soft_decision_hard_construction`` mode, neural decisions use soft PID
kinematics while persisted daughters use hard PID. Decision and construction
modes are serialized separately.

The reconstruction state
--------------------------------

Each state contains nodes, current parent/daughter links, active roots, p4,
charge, PID, generation heights, recursive detector-source masks and current
composite features. Height is bottom-up: leaves are zero, a new mother is above
its deepest daughter. It is not depth measured from the eventual root.

:py:mod:`hypertagging.reconstruction.constraints` and
:py:mod:`hypertagging.reconstruction.level_rollout` enforce a shared policy:

* A mother can select only pre-existing, eligible, parentless forest roots.
  A query and a daughter are used at most once in a coherent level proposal.
* Mother ontology, cardinality, charge and configured physics checks determine
  admissibility. Finite kinematics and node capacity are required.
* Recursive detector sources must be disjoint. Distinct node IDs can share a
  track, ECL/KLM source or copied origin and therefore conflict.
* Persisted mother p4 and charge are exact daughter sums. Source masks, daughter
  PID histograms and composite features are rebuilt from selected daughters.
  The runtime normalizer is reused after construction.

For example, two otherwise plausible mothers cannot coexist in one hypothesis
if both contain the same ECL source. They may occur in two competing hypotheses.
This distinction is central to event-level beam search.

Greedy decoding
-----------------------

Greedy is the default. At each level it selects valid mother proposals under
the common policy, persists them and continues with the resulting forest.
Object, type, cardinality and pointer outputs determine proposals; confidence
is used only if the checkpoint records that it was trained. Search stops under
the configured root, level, or empty-state conditions. Truth cannot rescue a
rejected proposal or supply a missing parent.

The output records the final forest, construction steps and diagnostics. A
structurally valid forest can still have the wrong daughters, PID or root and
score poorly against truth. Daughter-sum closure is guaranteed construction
consistency, rather than a claim about physical momentum resolution.

Full-depth beam decoding
--------------------------------

The opt-in :py:mod:`hypertagging.reconstruction.beam_search` implementation uses
:py:mod:`hypertagging.reconstruction.bounded_search` for bounded proposals and
is exposed through the same strict inference surface. It retains multiple
coherent states across generation heights:

#. Enumerate bounded mother-type, cardinality and daughter-set alternatives per
   query, backfilling past invalid alternatives within explicit search bounds.
#. Combine compatible proposals into source-exclusive level hypotheses.
#. Persist each hypothesis and score the resulting event state cumulatively.
#. Canonically deduplicate and retain the bounded global beam, continuing each
   surviving state through subsequent levels.

This is a full-depth search: alternatives can survive a locally inferior
decision and later form a better root. Beam width one follows greedy for valid
source-exclusive semantics. Larger widths are approximate; finite proposal,
per-query, level-hypothesis and node limits can discard a useful branch.

Scores combine selected-query log likelihood with omitted-query log no-object
probability, explicit query-count normalization and an optional empty-level
penalty. Cross-query proposal caps use gain over omission, so scores compare
states with different selected query sets. Completed roots precede unfinished
forests. A bounded terminal-root lane retains later root alternatives.
Scores are ranking functions, not calibrated probabilities of full events.

Live-state deduplication includes recursive topology and every future-visible
feature/confidence value: equal topology alone does not imply equal future
predictions. Terminal feature-only duplicates can collapse by physical tree
identity. Canonical ordering makes repeated inference deterministic under the
same model and configuration.

Configuration and reports
---------------------------------

``--beam-search`` enables the evaluator's beam report. The tracked
``configs/reconstruction/full_depth_beam.json`` records a bounded configuration;
the default hypothesis node cap is 256. Width, per-query proposals, global
proposals, level hypotheses, returned candidates and node limits all affect
the explored space. Inspect the CLI catalogue and
:py:mod:`hypertagging.reconstruction.hierarchical_inference` for current argument
names and defaults rather than equating width alone with search effort.

:py:mod:`hypertagging.evaluation.full_decay_runner` serializes search limits,
pruning diagnostics, candidate ranks/scores and evaluated trees. Greedy-only
reports preserve their v3 structure; beam-enabled v4 reports add beam top-1 and
post-inference oracle@K. Existing greedy metric names keep their meaning.
Oracle metrics inspect truth only after all generation/ranking is finished.
See :doc:`evaluation` for unit versus coherent-event oracle denominators.

``scripts/validate_reconstruction_checkpoint_pair.py`` validates encoder and
reconstruction lineage before evaluation of immutable validation/test
selections. :py:mod:`hypertagging.deployment` exports manifest-bound ONNX level
graphs, and :py:mod:`hypertagging.basf2_integration` constructs fresh candidates
inside basf2. See :doc:`basf2` for integration. CPU beam fixtures establish
search behavior; current trained beam physics performance is unavailable in
the dashboard evidence.
