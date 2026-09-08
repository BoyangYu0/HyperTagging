Architecture and scientific contracts
=====================================

The current pipeline uses production ``direct-mdst-tree-v4`` event rows.
Historical GraFEI/Toy-MC/GPT paths remain compatibility implementations; they
must not be substituted for the current strict full-decay evaluator.

.. code-block:: text

   mDST + retained truth for supervision
       | preprocessing/schema_v4.py
       v
   detector inputs | separate targets, masks, provenance
       | streaming.py + dataset_index.py + training/data_module.py
       v
   train-only statistics + immutable source-role selections
       | models/heterogeneous.py
       v
   detector-context encoder -> charge-compatible leaf PID
       | models/level_autoregressive.py + reconstruction/pid_state.py
       v
   PID-dependent p4 rebuild -> refined context -> mother/pointer decoder
       | constraints.py + level_rollout.py
       v
   source-exclusive forest -> daughter-summed persistent mothers -> next height
       | hierarchical_inference.py                 | deployment/export_onnx.py
       v                                           v
   CPU strict rollout -> full-decay metrics       ONNX graphs -> basf2 candidates

Package responsibilities
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Role and boundary
   * - :py:mod:`hypertagging.preprocessing`
     - V4 detector/truth separation, retained trees and channel identities;
       V1-V3 compatibility; V5 experimental nested Arrow storage.
   * - :py:mod:`hypertagging.data`
     - Tensor contracts, lazy event loading, masks, source-safe splits,
       normalization, dataset indexes, and selection manifests.
   * - :py:mod:`hypertagging.models`
     - Detector/composite adapters, shared contextual/Poincare features,
       physical relations, stair causality, PID and mother/pointer prediction.
   * - :py:mod:`hypertagging.losses`
     - Exact retained-tree geometry objectives, FP32 hyperbolic/VIC kernels,
       unordered targets and SciPy Hungarian matching in production.
   * - :py:mod:`hypertagging.training`
     - Real trainers, curriculum, sampling, checkpointing and resume, encoder transfer,
       independent teacher-forced and rollout checkpoint selection.
   * - :py:mod:`hypertagging.reconstruction`
     - PID state, forest constraints, persistent composites, strict detector-only
       inference, greedy decoding and opt-in full-depth beam search.
   * - :py:mod:`hypertagging.evaluation`
     - Checkpoint lineage, source/topology matching, full/B-half denominators,
       top-1 and explicitly separate oracle@K metrics.
   * - :py:mod:`hypertagging.deployment`
     - Trusted checkpoint export and manifest-bound ONNX tensor/feature policy.
   * - :py:mod:`hypertagging.basf2_integration`
     - Reconstructed FSP inputs, ONNX inference and fresh basf2 ParticleLists.
   * - :py:mod:`hypertagging.utils`
     - Seeds, padding, device/resource guards, I/O and trusted checkpoint helpers.

Information boundaries
----------------------

Strict inference physically keeps detector FSPs and scrubs target-only fields.
Truth PID, adjacency, geometry, stored composites, channel/B-side labels and
completeness flags cannot generate, rank, prune or stop reconstruction. The
evaluation-only source-key map stays outside the model. Links created by the
decoder are valid current state.

A new mother uses eligible, existing, parentless forest roots. Each coherent
hypothesis enforces one parent, unique daughter/query use, an allowed mother
ontology, charge/cardinality policy and recursive detector-source disjointness.
Different node IDs can still share a detector source. Competing hypotheses may
reuse that source independently. Persistent mother p4 and charge are exact
recursive daughter sums. P4 closure checks this implementation invariant;
it does not measure physical momentum resolution.

Raw-track PID must respect charge. Checkpoints keep decision and construction
modes distinct: ``soft_decision_hard_construction`` uses soft kinematics for
neural decisions and hard PID to persist daughters. Detector availability masks
distinguish missing measurements from zero placeholders. Static detector blocks
use train-fitted normalization; common/composite blocks stay in physical units
until the model-owned transform, which is reused after each rebuild. Runtime
slots without training observations use identity scaling.

Geometry and search
-------------------

Reconstruction level is generation height, not edge depth. Retained parent,
LCA and path geometry determines supervision; leaves lie farther out in the
radius convention. Greedy is the default. Optional beam search keeps coherent
event states and mother/type/daughter alternatives across all levels, with
explicit proposal/state/node limits, source exclusivity, canonical deduplication,
terminal-root handling and comparable cumulative scores. Truth is inspected for
oracle metrics only after generation and ranking finish.

Source references: ``AGENTS.md``, ``README.md``,
``docs/hyperbolic_level_autoregressive_reconstruction.md``,
``docs/heterogeneous_node_encoding.md``, ``docs/training.md`` and
``docs/full_decay_reconstruction_evaluation.md`` remain in the checkout.
Static signatures and redacted docstrings are in :doc:`_generated/api/index`;
:doc:`_generated/repository/index` summarizes whole-tree coverage with hashes.
