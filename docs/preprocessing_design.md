# Direct mDST Preprocessing Design

## Legacy Semantics Preserved

- The reduced HyperTagging PID vocabulary is preserved from `HyperTagging/__init__.py`.
- The legacy pruning rule from `HyperTagging/DataProd.py` is centralized in
  `hypertagging.preprocessing.pid_filter`: keep only particles whose Belle II
  name is in the legacy `particle_list` and whose MCParticle is primary.
- Historical tokenization is preserved through `TOKENIZE_DICT`.
- The level-wise output keeps a compatibility view named `legacy_levels` with
  the old fields: `feature`, `channel`, `evtNum`, `depth`, `seq_len`,
  `arrayIndex`, `motherPDG`, `motherIndex`, and `E_Rec`.
- Legacy repeated-leaf semantics are generalized through explicit copied nodes:
  copied nodes carry a unique `node_id`, the same reco-level features, and a
  `copied_from` source.

## Verified v1 behavior

- The new path avoids scanning many prebuilt basf2 `ParticleList`s. The basf2
  steering script reads DataStore arrays from generic mDST input using
  `inputMdst`, then hands compact MC/reco records to pure Python tree logic.
- Tree construction, PID pruning, repeated-node copying, level assignment,
  export, and verification are pure Python modules under
  `hypertagging.preprocessing`.
- The canonical output is an event tree schema (`direct-mdst-tree-v1`) with
  explicit nodes, parent/daughter links, levels, p4, tokens, MC ids, reco ids,
  copy metadata, and diagnostic flags. The legacy level rows are exported next
  to it for existing training code.
- Validation is now first-class: scripts check acyclicity, valid parent/daughter
  ids, level ordering, copy references, PID counters, and four-vector closure.

`export_trees()` remains the `direct-mdst-tree-v1` compatibility exporter. Its
fields and legacy-level semantics are unchanged.

## Versioned heterogeneous v2

`export_trees_v2()` writes `direct-mdst-tree-v2`. It retains every flat v1 node
field and adds:

- explicit `node_kind`;
- shared common features with value-level availability;
- separate track, ECL-cluster, and composite feature blocks and masks;
- daughter PID histograms;
- deterministic two-B signatures, IDs, count arrays, and event-pair identity;
- optional source file/category metadata.

The complete feature specification is stored in `feature_spec_json`. Numeric
zeroes in unavailable positions are tensor-safe storage only; masks are
authoritative.

The current basf2 adapter conditionally preserves track fit p-value, d0, z0,
phi0, omega, and tan-lambda when their accessors exist. ECL records preserve
energy/direction and conditionally timing, E9/E21, crystal count, and minimum
track distance. The photon hypothesis and track-match decision are explicit.
Unsupported values are absent and masked, not inferred.

`load_payload_v2()` and `load_heterogeneous_events()` accept both v1 and v2.
For v1, node kind is inferred only from explicit topology and `reco_id`
prefixes. Detector blocks remain unavailable; composite structure is safely
recovered from existing reco-derived daughter p4 and links.

See `docs/heterogeneous_node_encoding.md` and
`docs/channel_representation.md`.

## Truth Topology vs Reco Kinematics

MC truth is used only for topology and labels:

- MCParticles define possible parent/daughter links for training supervision.
- MC matching/relation ids map reconstructed leaves to MC truth nodes.
- Diagnostic MC four-vectors, if available, are stored only in `mc_px`,
  `mc_py`, `mc_pz`, and `mc_energy`.

Reco kinematics are kept separate:

- Reconstructed leaves get p4 from reconstructed basf2 objects such as
  `Particle`, `Track`, or `ECLCluster` records when available.
- Every retained composite/mother node is recomputed recursively as:

  ```text
  mother.p4 = sum(daughter.p4)
  ```

- MCParticle four-momenta are never used as reconstructed mother p4. The
  verification script fails if a retained mother p4 does not equal the summed
  retained daughter p4 within tolerance.

## CLI Usage

Run inside a basf2 environment:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/generic_mdst.root \
  --output /data/dust/user/boyangyu/hypertagging/processed.parquet \
  --schema-version direct-mdst-tree-v4 \
  --entry-sequence 0:49 \
  --max-events 50 \
  --event-buffer-size 32 \
  --row-group-size 16
```

Omitting `--schema-version` selects the new v4 production default. Explicit
v1/v2/v3 options remain available for compatibility. Existing parquets are
never migrated or overwritten in place.

`--entry-sequence` is repeatable and uses basf2's inclusive entry-range
notation. Supply one sequence per input file. Production events carry
`experiment`, `run`, `production`, and a stable `event_uid` so shards can be
checked for duplicates and split by source without event leakage.

Inspect and validate:

```bash
/data/dust/user/boyangyu/uv_env/bin/python scripts/verify_preprocessing.py \
  --input /data/dust/user/boyangyu/hypertagging/processed.parquet \
  --all-events --check-tree --check-p4 --check-pid
```

The documented real mDST glob is in `file_paths.md`:
`/pnfs/desy.de/belle/local/belle/MC/release-08-03-00/DB00003335/MC16ri_run2/**/*.root`.

## Current Basf2 Adapter Notes

Generic mDST input uses `Tracks` and neutral `ECLClusters` by default. Optional
uDST `Particle` arrays are read only when explicitly passed with
`--particle-array`. Track fits use the Belle II charged-stable hypothesis API;
neutral ECL four-vectors use `ClusterUtils` with the photon hypothesis, and
track-matched ECL clusters are excluded to avoid double counting.

If an input file contains only MCParticles and no usable reconstructed objects,
preprocessing fails loudly.
The debug flag `--allow-mc-leaf-kinematics-for-debug` can synthesize reco leaves
from MC leaves for software tests only; it must not be used for training.

## Visualization Notebook

Generate and execute the real-data four-momentum comparison notebook:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/create_preprocessing_visualization_notebook.py
JUPYTER_CONFIG_DIR=/tmp/hypertagging-jupyter-config \
  /data/dust/user/boyangyu/uv_env/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace \
  notebooks/preprocessing_four_momentum_validation.ipynb
```

The notebook compares computed/reconstructed and MC `E`, `px`, `py`, `pz`, and
invariant mass distributions, plus event-by-event and particle-by-particle
differences. Unmatched reconstructed objects remain in production output but
are excluded from truth residuals.

The executed
`notebooks/inspect_preprocessed_parquet_and_gpt_like.ipynb` complements this
with the parquet schema, representative event/node tables, multiplicity and
depth distributions, tree checks, the direct-tree GPT batch, its attention
mask, and a real-data CPU forward/loss/backward/optimizer smoke test. Regenerate
it with:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/create_parquet_gpt_inspection_notebook.py
JUPYTER_CONFIG_DIR=/tmp/hypertagging-jupyter-config \
  /data/dust/user/boyangyu/uv_env/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace \
  notebooks/inspect_preprocessed_parquet_and_gpt_like.ipynb
```

## GPT-Like Direct-Tree Contract

Real direct-mDST events have variable numbers of nodes and variable tree
depths, so the old fixed-particles-per-level collator is not valid for this
parquet. `hypertagging.data.direct_gpt` instead orders visible leaves first,
then truth-guided higher-level query slots. Leaf queries may attend leaves;
each higher-level query may attend nodes at strictly lower levels. Targets are
node embeddings and link labels map each child position to its parent position.

This adapter makes the current `MultiGPT` implementation executable and
testable on real parquet, but it is teacher-forced integration scaffolding, not
a claim of final training quality. Production training should add feature
normalization, source-aware train/validation/test splits, checkpointing, and
physics performance metrics.

## Ten-Million-Event Production

The planner reads ROOT tree metadata only, interleaves physics categories,
creates exact non-overlapping inclusive entry ranges, and writes an atomic
JSONL manifest. Each array task writes a temporary parquet, validates its
schema, event count, and `event_uid` uniqueness, then publishes atomically.
Completed valid shards are resumable.

The project venv is Python 3.11, while release-08-03-00 embeds Python 3.8.
Compiled packages cannot be shared. A small basf2-compatible dependency target
is therefore required once:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
unset PYTHONPATH
/cvmfs/belle.cern.ch/el9/externals/v02-00-02b/Linux_x86_64/common/bin/python3 \
  -m pip install --target /data/dust/user/boyangyu/basf2_py38 \
  'numpy==1.24.4' 'awkward==2.6.10' 'pyarrow==16.1.0'
```

The submitted worker still activates
`/data/dust/user/boyangyu/uv_env`; it isolates the embedded basf2 subprocess
from that Python-3.11 interpreter and injects the compatible dependency target.
To inspect the existing default 10M manifest and submission command:

```bash
scripts/condor/submit_mdst_production_10m.sh --dry-run
```

Defaults are 10,000,000 input events, 5,000 events per task, and at most 50
concurrent materialized tasks. Override with `TARGET_EVENTS`, `EVENTS_PER_TASK`,
`MAX_CONCURRENT`, and the resource variables documented in
`scripts/condor/README.md`. Run a small pilot before the
full array because the current exporter buffers a shard in memory.

## Corrected v3 leaf contract

V3 never uses related MC PDG to select a TrackFitResult, reconstructed charge,
or an input energy. The fit policy is best reconstructed p-value with a
deterministic pion closest-mass fallback. Raw tracks store reconstructed p3,
charge, e/mu/pi/K/p energy hypotheses, availability, PIDLikelihood logL values
and availability, canonical pion energy, `input_pid_token=0`, and the separate
truth-only target. Fixed basf2 Particle candidates are explicitly tagged
`fixed_hypothesis_candidate`.

The canonical pion value is only the tensor-compatible encoder baseline.
Reconstruction training substitutes the differentiable, charge-compatible
PID-energy mixture before daughter-sum physics losses. Rollout now names the
PID contract explicitly. The compatibility default,
`soft_decision_hard_construction`, computes relation features and pointer/type
logits with soft expected PID kinematics, then uses hard predicted-PID track p4
for exact daughter-sum mother construction. Only
`rollout_pid_kinematics_mode=hard` is a fully hard-conditioned neural rollout.
Temperature-softmax and straight-through-hard modes are explicit alternatives.
Fixed-hypothesis candidates and ECL measurements are not rewritten by this
raw-track path.

Composites store exact recursive daughter sums and recursive reconstructed leaf
sources. Partial targets expose truth/retained/reconstructed daughter counts and
are invalid by default below two daughters. V1/v2 files are adapted without
inventing missing detector values.

Production planning now defaults to 5,000-event bounded shards and writes
schema, PID vocabulary, feature hash, charge-conjugation mode, leaf mode, and
git commit into every manifest row. The worker explicitly passes
`--schema-version direct-mdst-tree-v4`. Run the cross-shard validator with:

```bash
python scripts/mdst_batch_production.py validate --manifest /path/to/manifest.jsonl
```

## Schema-v4 production contract

V4 does not alter v3 semantics in place. It publishes one event per Parquet row
and uses `ParquetEventWriter` with a bounded event buffer, periodic row groups,
an atomic `.partial` file, and a metadata JSON sidecar. A failed writer removes
its unpublished partial file. The sidecar records the schema/PID/feature
contracts, source range, category, preprocessing configuration, event count,
and aggregate capacity, PID, leaf-mode, and completeness statistics.

Composite PID summaries are now unambiguous:
`daughter_input_pid_histogram` contains current data-available PID knowledge;
`daughter_truth_pid_histogram` is target/diagnostic only. Raw tracks are stored
as unknown input PID with explicit `leaf_kinematics_mode_id`; no node mode is
inferred from PID zero. Legacy v1-v3 records are adapted as
`legacy_conflated`, and production training requires explicit diagnostic opt-in.

Recursive completeness is stored independently from local daughter counts.
The default `complete_only` target policy admits only valid, recursively
reconstructable-complete mothers. Capacity scans and training use the same
policy.

## Runtime-scale publication contract

V4 buffers only a configured event window and publishes the parquet and
metadata sidecar followed by an atomic completion marker. Failed basf2
processing aborts and removes unpublished partial files. The sidecar includes
mergeable feature Welford state as well as capacity, PID, completeness, and
actual per-node leaf-mode distributions.

Requested collection mode and actual node provenance are separate. A
fixed-hypothesis request requires explicit Particle arrays and rejects raw
Tracks; validation checks the actual distribution rather than trusting the
manifest label. Mixed raw-track, ECL, and composite events are marked
`mixed_explicit_per_node`.

V4 retains JSON event payloads for compatibility. Experimental
`schema_v5.py` writes native nested Arrow events, and
`scripts/benchmark_parquet_storage.py` measures size, writes, full decode, and
projected reads on a pilot. Production must remain v4 until that benchmark is
reviewed on representative data.

Run progressively larger existing fixture or pilot samples without touching
production parquet files:

```bash
for events in 10 100 1000 10000
do
  python scripts/benchmark_parquet_storage.py \
    --data /data/volume/pilot-or-fixture-manifest.jsonl \
    --max-events "$events" \
    --output-dir "/tmp/hypertagging-storage-${events}"
done
```

Each JSON report contains file sizes, write/full-read/projected-read
throughput, JSON/native decode CPU, process peak RSS (with the Arrow allocation
caveat), and events/s plus nodes/s. This command is diagnostic: schema-v4 stays
the production default, v5 is not promoted, and ten-million-event readiness is
not claimed until representative measurements exist.
