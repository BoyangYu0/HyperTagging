# ONNX full-decay reconstruction in basf2

This integration is additive. It consumes existing reconstructed final-state
`ParticleList` objects and publishes at most one model-selected root to a new
list (by default `Upsilon(4S):HyperTagging`). It does not call MC matching,
load training checkpoints during event processing, alter any input list, or
replace the offline full-decay evaluator.

The event flow is:

```text
reconstructed FSP ParticleLists
  -> truth-free schema-v4 feature blocks
  -> one hash-checked ONNX graph per hierarchy level
  -> constrained mother/daughter proposals
  -> proposal-set beam search across levels
  -> best completed Upsilon(4S) tree in a fresh ParticleList
```

One Python `basf2.Module` owns the level loop because a basf2 path is static
once event processing starts. As in FEI inference, candidates are built
hierarchically and scored before later stages; here, the model's cardinality,
mother-type, pointer, object, leaf-PID, and confidence predictions direct the
tree construction, while the host enforces charge, provenance, and loose
physics constraints. Competing partial trees survive according to a bounded
beam score, so the locally best first composite need not determine the final
tree. Exact within-level proposal-set search is capped at 12 proposals.

The v1 deployment beam differs from the strict offline full-depth beam. It
decodes one mother type, cardinality and daughter set per query, then retains
coherent subsets across levels using the manifest's `sum_confidence` score.
It does not enumerate the offline beam's alternative type/daughter decisions
or its explicit no-object alternative when proposals exist. Its ranking is
therefore a separate search policy, and offline beam gains do not establish
deployment gains. Compare both paths on the same validation events and
checkpoint before making a parity or physics-performance claim.

## Runtime boundary

The runtime loads only `manifest.json`, NumPy, and per-level ONNX graphs. The
manifest fixes tensor names and shapes, feature ordering, PID vocabulary,
normalizers, reconstruction policy, graph SHA-256 hashes, and compatible
basf2 releases. Model and manifest hashes are checked before the first event.
The v1 graphs use fixed `max_nodes` and `max_sources` capacities.

The v1 runtime also requires identical query and daughter-cardinality counts
across all exported levels. Current scientific checkpoints with different
per-level capacities cannot export a full bundle under this contract; they
need a runtime/manifest extension that preserves the trained model's shapes.
Changing or reshaping the checkpoint to fit a uniform graph would change the
model being evaluated.

The tested CVMFS environment is `light-2607-kasei`:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup light-2607-kasei
```

That installed light release supplies ONNX Runtime. No basf2 source checkout
or development-branch pull is needed. Belle II's official setup instructions
are at <https://software.belle2.org/release-09-00-01/sphinx/online_book/basf2/introduction.html>,
and the current software documentation is at <https://software.belle2.org/>.

## Export a trusted checkpoint

Export happens in the training environment, never inside event processing:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_full_decay_onnx.py \
  artifacts/runs/ht-reconstruction-transfer-fullscale-20260824/15979725/training/best_rollout_edge_f1.pt \
  /path/to/new/hypertagging-onnx-bundle \
  --levels 1 2 3 4 5 6 \
  --max-nodes 128 \
  --max-sources 128
```

Use `--dry-run` first to validate the checkpoint and print the resolved
contract without writing a bundle. The destination must be new or empty.
`--force` can replace only an existing bundle whose manifest and graph hashes
still validate; it will not remove an unrelated directory.

The exporter requires a current schema-v4, data-compatible checkpoint with
matching feature/PID contracts and all learned normalizers. It emits one
fixed-shape graph per requested level and validates every graph with ONNX.

## Add it to a basf2 path

For an existing steering file:

```python
from hypertagging.basf2_integration import add_hypertagging_full_decay

add_hypertagging_full_decay(
    path,
    manifest_path="/path/to/hypertagging-onnx-bundle/manifest.json",
    input_particle_lists=("pi+:HyperTaggingFSP", "gamma:HyperTaggingFSP"),
    output_particle_list="Upsilon(4S):HyperTagging",
)
```

Inputs must be reconstructed, daughterless final-state candidates selected in
the same way as the training data. Repeated detector sources are deduplicated;
recursive source masks also prevent an ECL/KLM association from being reused
in two branches. Track PID is predicted by the model with charge-compatible
hard construction. The new list is transient by default.

The standalone runner reads existing mDST/uDST files and writes nothing unless
`--output-udst` is explicitly supplied:

```bash
basf2 scripts/run_basf2_full_decay.py -- \
  --manifest /path/to/hypertagging-onnx-bundle/manifest.json \
  --input /path/to/input.mdst.root \
  --input-list pi+:HyperTaggingFSP \
  --input-list gamma:HyperTaggingFSP \
  --events 1
```

If output is requested, the runner refuses an existing filename. The module
also refuses an output `ParticleList` that already exists, which prevents it
from merging with or overwriting standard/offline reconstruction products.

## Minimal verification

Run the CPU contract/beam tests in the project environment:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_onnx_full_decay_beam_cpu.py
```

Run the synthetic event plus installed-release real-mDST ECL and Track smokes:

```bash
bash tests/run_basf2_onnx_full_decay_smoke.sh
```

The smoke bundle is deliberately deterministic and non-physical: it proves
ONNX execution, multi-level non-greedy beam selection, source exclusivity,
Particle tree materialization, Track PID detector-availability parity, and
non-interference with existing DataStore objects. Before physics use, export
all intended levels/capacities and validate reconstruction efficiency, purity,
calibration, and input-list selection on a representative sample.
