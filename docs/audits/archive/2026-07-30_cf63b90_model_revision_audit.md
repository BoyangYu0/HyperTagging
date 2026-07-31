# Model Revision Audit

Audit date: 2026-07-30  
Audited revision: `cf63b90` (`master`, also `origin/HEAD`)

## Scope and status before this revision

The direct generic-mDST preprocessing path is the verified part of this
repository. It builds a truth-guided retained topology, contracts pruned truth
particles, copies multiply used subtrees, assigns levels, and recursively
recomputes every composite four-vector from retained reconstructed daughters.
This revision treats that behavior as an invariant.

The model path before this revision is CPU-testable scaffolding. It has a flat
node-feature projection, a shared Euclidean/Poincare encoder, preliminary
same-mother/same-branch losses, a parent loss evaluated with Euclidean distance
in the tangent space, and a radius target which increases toward the root. The
level decoder has object, type, pointer, cardinality, and confidence heads with
Hungarian matching. It can predict one level, but it does not implement the
complete re-encode/append/stop rollout cycle. A pairwise relation bias is
computed but discarded before contextualization.

## Exact current parquet contract

The canonical top-level schema is `direct-mdst-tree-v1`. One parquet row holds:

- `schema_version`;
- `events`;
- `legacy_levels`;
- `summary_json`.

Each event contains `event_id`, stable `event_uid`, `experiment`, `run`,
`production`, `root_ids`, level-to-node-ID records, nodes, and `metadata_json`.
Each v1 node contains:

`node_id`, `pdg`, `token`, `charge`, `mass`, `energy`, `px`, `py`, `pz`,
`level`, `parent_id`, `daughter_ids`, `reco_id`, `mc_id`, `copied_from`,
`source_node_id`, `flags`, `prodTime`, `x`, `y`, `z`, `nDaughters`, `mc_px`,
`mc_py`, `mc_pz`, and `mc_energy`.

The compatibility `legacy_levels` rows contain `feature`, `channel`, `evtNum`,
`depth`, `seq_len`, `arrayIndex`, `motherPDG`, `motherIndex`, `E_Rec`, and
`targetNodeIds`. Their ordered feature vector is token, mass, charge, energy,
production time, vertex x/y/z, px/py/pz, and daughter count.

## Reconstructed detector information currently available

The current generic-mDST adapter reads `Tracks` and neutral `ECLClusters`, and
optional explicitly configured uDST `Particle` arrays.

- Tracks currently preserve the fitted momentum, charge sign, a mass-hypothesis
  energy, reconstructed-object ID, optional MC relation, PDG label, and flags.
  No fit-quality, impact-parameter, covariance, PID-likelihood, acceptance, or
  matched-cluster quantities are exported.
- ECL clusters currently preserve the photon-hypothesis four-vector,
  reconstructed-object ID, optional MC relation, photon PDG/zero charge, and
  flags. Track-matched clusters and clusters without the photon hypothesis are
  skipped. No timing, position, shower-shape, crystal-count, matching-distance,
  or cluster-quality quantities are exported.
- Detector-specific zeroes therefore have no documented missing-value meaning.

The v2 reader will only store detector values returned by a supported basf2
accessor and will carry a value-level availability mask. It will not fabricate
missing detector values.

## Reduced PID vocabulary and pruning

`PDG_TOKENS` has 41 entries including token 0 for unknown. The retained
vocabulary is Upsilon(4S), gamma, K_L0, pi0, J/psi, K_S0, charged leptons,
charged kaons/pions/protons, Lambda/Sigma, D/D_s/Lambda_c and their vector
partners, B/B_s, and charge conjugates. `TOKENIZE_DICT` is the deterministic
enumeration in `hypertagging.preprocessing.pid_filter`.

The verified pruning rule keeps primary MC particles whose Belle II name and
PDG are in that legacy vocabulary. Reconstructed-only objects are filtered by
PDG membership. This revision does not change those rules.

## Legacy channel and dictionary-array audit

The checked-in history contains only the unified migration commits. The
`src/hypertagging/legacy` directory has no frozen source. Repository references
to the historical implementations point outside this checkout; the available
contracts preserve only flat `channel`/`channels` and `evtNum`/`evtNums`
identifiers. `MIGRATION_NOTES.md` explicitly calls their complete semantics
ambiguous. No historical dictionary/count-array construction for comparing the
two B decays is recoverable from the repository history or legacy directories.

Consequently v2 will use the explicit fallback specified by the scientific
design: recursively sorted reduced-PID signatures after the existing pruning
and contraction, deterministic hash IDs, an unordered pair for Upsilon(4S),
and depth-aware reduced-PID/multiplicity count arrays. Same-event, B-side,
exact-channel equality, structured similarity, and unordered event-pair
identity remain distinct labels.

## Level and ordering conventions

`assign_levels` defines leaves as level 0 and every mother as
`1 + max(daughter level)`. An adjacent-level sample for target `t+1` uses all
nodes with level at most `t`.

The historical fixed-width GPT-like collator orders fixed-size level blocks and
uses a stair mask. The direct-tree adapter instead sorts nodes by `(level,
node_id)`, exposes only level-0 embeddings as inputs, and uses zero query slots
for higher truth levels. Level-0 queries can attend all level-0 nodes; a
higher-level query can attend strictly lower levels. The preliminary revised
model uses all nodes at levels below the requested target, but does not yet
apply its computed relation bias to contextual attention.

## Notebook and HTCondor conventions

Notebook files are generated deterministically with `nbformat` by scripts under
`scripts/`. Existing notebooks accept an environment-variable parquet path and
use small CPU model samples, but they target schema v1 and real input by
default.

Full preprocessing and training use HTCondor. Renderers create submit and
executable files without submission; submission is explicit. Local CUDA is
blocked except for a tiny, explicitly enabled smoke run after `condor_q` and
`nvidia-smi` checks. This revision retains those conventions and does not add
SLURM.

## Identified limitations addressed by this revision

- no explicit node kind or feature-group/value availability;
- no structured track, ECL-cluster, or composite blocks;
- no structured two-B channel representation;
- flat input projection for all detector and composite types;
- no daughter-pooled composite representation;
- parent ranking does not use Poincare distance;
- radius convention is reversed relative to leaf/root hierarchy;
- no VICReg-style variance/covariance protection or collapse diagnostics;
- redundant same-mother/same-branch objectives are the principal losses;
- relation bias is computed but not applied to attention logits;
- no complete teacher-forced/free-rollout reconstruction cycle;
- no source-aware split/normalization and incomplete evaluation surfaces;
- no fixture-capable dataset, embedding, and rollout inspection notebooks.

## Planned files

The implementation will add versioned schema/channel/data adapters, heterogeneous
encoders, relation-aware attention, revised hyperbolic objectives, rollout and
evaluation helpers, ablation configs, notebook generators/notebooks, and focused
tests. Existing preprocessing builders and v1 exporter behavior will only be
extended through optional fields and an explicit v2 export path.

Expected existing files touched include:

- `src/hypertagging/preprocessing/{mdst_tree_builder,basf2_mdst,export_dataset}.py`;
- `src/hypertagging/data/{level_batch,level_collate,tiny_level_fixtures}.py`;
- `src/hypertagging/models/{hyperbolic,relations,level_autoregressive}.py`;
- `src/hypertagging/losses/hyperbolic_pretraining.py`;
- `src/hypertagging/training/` and `src/hypertagging/reconstruction/`;
- notebook generators, documentation, repository maps, and exports.

## Compatibility strategy

1. `export_trees()` remains a v1 exporter with byte-level field semantics
   unchanged for existing callers.
2. A separate v2 export/migration surface writes `direct-mdst-tree-v2`.
3. The model loader accepts v1 and v2. V1 node kinds are inferred
   conservatively from explicit reconstructed-object ID prefixes and topology;
   ambiguous leaves are `unknown`. All unavailable detector groups and values
   are masked false.
4. Original v1 scalar values, IDs, flags, links, legacy level rows, and
   diagnostic MC fields are preserved by migration.
5. Composite p4 is always recomputed or validated as the deterministic
   daughter sum.

## Tests protecting verified preprocessing semantics

The existing `tests/test_direct_mdst_preprocessing_cpu.py` protects pruning,
truth/reco separation, levelization, shared-daughter copying, generic-mDST
collection, v1 export, and exact mother p4 closure. Existing production tests
protect non-overlapping shard ranges and stable event metadata. Existing direct
GPT tests protect variable-level ordering and masks.

Before changing v1 behavior, this revision adds explicit v1-to-v2 migration
equivalence tests for all original fields, level definitions, links, copy
provenance, and p4. New v2 tests cover node-kind inference, missingness masks,
detector-block separation, channel stability, and round trips.

## Verification boundaries

CPU fixtures can establish software correctness, finite gradients, mask
behavior, deterministic rollout, and notebook execution. They cannot establish
physics performance. No full preprocessing, long training, real-scale GPU run,
or HTCondor submission is part of this revision.
