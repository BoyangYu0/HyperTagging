Dataset production and event structure
==============================================

The learning problem starts with reconstructed detector objects and asks which
decay tree could have produced them. Simulation supplies retained ancestry and
particle labels for supervision. The production format is
``direct-mdst-tree-v4``: one event per Parquet row, with a JSON event payload
and schema/feature metadata. This makes variable-size trees streamable without
padding the stored dataset to the largest event. Experimental v5 stores nested
Arrow values; v1-v3 remain compatibility inputs.

From mDST to a retained tree
------------------------------------

The ``scripts/preprocess_mdst.py`` producer (see the
:doc:`script catalogue <_generated/catalog/index>`) runs inside basf2.
:py:mod:`hypertagging.preprocessing.mdst_tree_builder` assembles the event tree;
:py:mod:`hypertagging.preprocessing.pid_filter` defines retained particles and
the reduced PID vocabulary. Primary MC particles and reconstructed-to-MC
relations establish supervision. Retention changes the tree being learned:
its exact edges, roots and paths are the targets, rather than all generator
intermediates.

Tracks use reconstructed fits and detector PID observations. Neutral ECL
clusters use the photon hypothesis, and track-matched clusters are excluded
from that neutral collection. Optional KLM information has its own block and
availability. Release-dependent accessors may leave values unavailable; no
track, calorimeter or KLM measurement is inferred from a zero placeholder.
Fixed-hypothesis candidates and raw tracks have different kinematic contracts.
Raw tracks acquire PID-dependent energy after model PID inference.

:py:mod:`hypertagging.preprocessing.levelize_tree` assigns generation height:
leaves have height zero and a mother is one above its highest daughter.
Copied nodes preserve compatibility views while retaining their original
detector-source identity. A unique node ID therefore does not guarantee a
unique physical source. Reconstructed mother momentum and charge are recursive
daughter sums, including after retention; they are not MC mother values.

Event-row contract
--------------------------

:py:mod:`hypertagging.preprocessing.schema_v4` owns version strings, feature
specifications, event serialization and completion metadata. Important groups
are:

.. list-table::
   :header-rows: 1

   * - Group
     - Meaning and use
   * - Identity and provenance
     - Stable event UID, experiment/run/production, source category and row
       provenance support duplicate detection, selection and reproducibility.
   * - Nodes and links
     - Node kind, node/source/copy identities, parent/daughter links and height
       describe the retained training tree. Inference creates its own links.
   * - Common features
     - Momentum, energy, mass, charge and daughter count share a contract across
       detector and composite nodes. PID, level, active and copied slots are
       categorical/flag inputs, rather than continuous geometry.
   * - Detector blocks
     - Track, ECL and KLM values each carry availability. PID detector names
       cover SVD, CDC, TOP, ARICH, ECL and KLM.
   * - Composite inputs
     - Daughter-sum p4, summed charge, count, pointer confidence mean/minimum
       and copied-daughter fraction can be reconstructed from current state.
   * - Supervision
     - Truth PID and its availability, retained ancestry, exact LCA/path
       geometry, completeness and B/channel targets support losses/evaluation.
       Stored completeness-related composite slots remain target-only.
   * - PID summaries
     - Model-input daughter PID histograms summarize currently constructed
       daughters. Truth-supervision histograms are separate fields. Legacy
       conflated inputs carry an explicit compatibility marker.

Full-truth and reconstructable channel signatures describe different decay
populations. B-side labels support training pools and evaluation, but cannot
tell reconstruction which leaves to combine. Continuum targets use explicit
retained top-level components; they do not invent two B hemispheres.

Loading and tensorization
---------------------------------

:py:mod:`hypertagging.data.streaming` reads event rows lazily.
:py:mod:`hypertagging.data.heterogeneous` adapts schemas and collates variable
node counts. Typical shapes are ``[B, N, F]`` for feature blocks, ``[B, N]`` for
node masks/labels, and ``[B, N, N]`` for adjacency or relation targets. Padding
is masked. Recursive source masks have a separate source axis and identify
overlap even between distinct copied/composite nodes.

:py:mod:`hypertagging.data.dataset_index` records authenticated shard identity
and capacity information. :py:mod:`hypertagging.training.data_module` combines
the index with immutable source-role selections. Splits use stable UIDs and
source grouping; validation cohorts remain fixed. Train, validation and sealed
test roles have distinct purposes. Query count, node count and daughter
cardinality must accommodate the chosen target policy before training.

Static track/ECL normalization uses only available training observations;
the KLM encoder uses its fixed physical scales and availability masks.
Common/composite quantities remain in physical units until the model-owned
runtime transform, which is reapplied after PID and mother construction.
Unobserved runtime slots use identity scaling. Missing values stay masked
after transformation. Statistics and feature/PID contracts travel with the
checkpoint, so evaluation applies the same mapping.

Target populations and quality checks
---------------------------------------------

``complete_only`` selects valid mothers with recursive support in the retained
forest; ``reconstructable_partial`` admits valid retained partial targets.
The producer removes absent truth leaves before computing recursive completeness,
so a mother can be recursively complete and still carry
``partial_missing_daughters=True``. Both policies can select the same native
targets. Full physical-decay completeness, strict-root representability and
diagnostic target populations have separate meanings and denominators.

Use ``scripts/verify_preprocessing.py`` to check tree structure, PID and p4
closure. ``scripts/build_dataset_index.py`` and
``scripts/build_training_selection.py`` prepare indexed role selections; their
options are in the :doc:`catalogue <_generated/catalog/index>`. Production
shards also carry sidecars and completion markers binding schema, counts,
source/config provenance and content hashes. Validation checks unique UIDs,
non-overlapping source ranges and common contracts across shards.

The :doc:`dashboard <_generated/status/index>` records the scientific status.
The current-status record reports no current-revision representative real
pilot and unresolved KLM training scope. Fixture schema checks establish
readability and invariants, not detector completeness or physics efficiency.
