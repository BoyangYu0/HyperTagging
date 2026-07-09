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
basf2 scripts/preprocess_mdst.py -- \
  --input /project/agkuhr/users/boyang/data/MC15/mdst001.root \
  --output /project/agkuhr/users/boyang/data/MC15/HyperTagging_uni/processed.parquet \
  --max-events 100
```

Inspect and validate:

```bash
uv run python scripts/verify_preprocessing.py \
  --input processed.parquet \
  --event 0 \
  --all
```

The documented real mDST glob is in `file_paths.md`:
`/project/agkuhr/users/boyang/data/MC15/mdst*.root`.

## Current Basf2 Adapter Notes

The direct reader tries reconstructed `Particle` arrays first and then mDST
`Tracks`/`ECLClusters` with MC relations when present. If an input file contains
only MCParticles and no usable reconstructed objects, preprocessing fails loudly.
The debug flag `--allow-mc-leaf-kinematics-for-debug` can synthesize reco leaves
from MC leaves for software tests only; it must not be used for training.
