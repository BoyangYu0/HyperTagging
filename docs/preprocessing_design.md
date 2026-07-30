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

## What Changed

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
  --entry-sequence 0:99 \
  --max-events 100
```

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

Defaults are 10,000,000 input events, 25,000 events per task, and at most 50
concurrent materialized tasks. Override with `TARGET_EVENTS`, `EVENTS_PER_TASK`,
`MAX_CONCURRENT`, and the resource variables documented in
`scripts/condor/README.md`. Run a small pilot before the
full array because the current exporter buffers a shard in memory.
