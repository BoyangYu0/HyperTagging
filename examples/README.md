# Examples

These examples are CPU-only smoke runs built from tiny synthetic fixtures and
validated dry-run paths. They do not read full datasets or reproduce physics
performance.

- `toy_mc_minimal/`: validates the Toy-MC contract, computes a tiny embedding
  loss, and prints the legacy Toy-MC preprocessing dry-run command.
- `grafei_minimal/`: validates the GraFEI combined reconstruction contract and
  runs the reconstruction dry-run.
- `gpt_like_minimal/`: validates the GPT-like collate path and runs the combined
  `MultiGPT` dry-run.

Full-data roots documented for later reproduction:

- Toy-MC after BASF2 generation and before preprocessing:
  `/home/boyang/data/MC`
- Original GraFEI before preprocessing:
  `/home/boyang/data/graFEI`

Run all examples individually:

```bash
uv --cache-dir /tmp/uv-cache run python examples/toy_mc_minimal/run_example.py
uv --cache-dir /tmp/uv-cache run python examples/grafei_minimal/run_example.py
uv --cache-dir /tmp/uv-cache run python examples/gpt_like_minimal/run_example.py
```

For current schema-v4 reconstruction, the offline evaluator supports a greedy
and beam comparison in one CPU run; see the
[full-decay evaluation invocation](../docs/full_decay_reconstruction_evaluation.md).
The validated search defaults are in
[`configs/reconstruction/full_depth_beam.json`](../configs/reconstruction/full_depth_beam.json).
In an existing evaluation pipeline with a separately loaded CPU `model` in
evaluation mode, a single-event collated/normalized `truth_batch`, and the
checkpoint-resolved `rollout_config`, run:

```python
import json
from pathlib import Path
from hypertagging.reconstruction import (
    BeamSearchConfig, HierarchicalInferenceConfig, reconstruct_beam_from_fsps,
)
from hypertagging.evaluation import (
    evaluate_full_decay, evaluate_ranked_decay_candidates,
)

search = BeamSearchConfig(**json.loads(
    Path("configs/reconstruction/full_depth_beam.json").read_text()
))
beam = reconstruct_beam_from_fsps(
    model, truth_batch,
    config=HierarchicalInferenceConfig(scope="full", rollout_config=rollout_config),
    beam_config=search,
)
# All truth comparisons happen after the fixed candidate list is returned.
comparison = evaluate_ranked_decay_candidates(
    [evaluate_full_decay(candidate.batch, truth_batch) for candidate in beam.candidates],
    scores=beam.scores,
    oracle_ks=(1, search.beam_width),
)
print(comparison.as_dict())
```

The CLI is preferable for checkpoint-bound reports because it restores lineage,
target policy, confidence policy, event selection, and micro aggregation. API
callers must pass the same checkpoint target policy to `evaluate_full_decay`
when it differs from the default `complete_only`.
