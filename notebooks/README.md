# Notebook evidence suite

[`index.yaml`](index.yaml) is the single machine-readable registry used by the
runner. Fixture notebooks validate software contracts, not physics performance
or throughput. Every generated report records SHA, schema, fixture/real source,
data name/path, checkpoint, seed, and pass/NOT RUN status.

| Group | Responsibility |
|---|---|
| `CORE_CONTRACT` | schema/PID inspection, preprocessing QA and p4 closure, geometry/anti-collapse, checkpoint transfer, level reconstruction |
| `EXTENDED_ENGINEERING` | streaming/index, exact geometry, runtime scaling, capacity, bounded rollout search, manifests |
| `HISTORICAL_COMPATIBILITY` | direct-GPT and legacy-schema compatibility only |
| `DIAGNOSTIC` | first-level ambiguity and objective-gradient conflict |
| `EXTERNAL_SCIENTIFIC` | real mDST pilot and trained held-out physics validation |

Run the default deterministic CPU set with:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-current-head-notebooks
```

`--list` and `--only ID` are derived from the YAML index. The first-level
diagnostic runs with `--diagnostic-first-level-ambiguity`; its type-relation
section exercises the actual model switch and gradients. Objective-gradient
conflict diagnostics are part of the hyperbolic inspection notebook.

The real mDST and trained-physics notebooks are real-only. Missing inputs write
`NOT RUN` and raise a clear guard; they never substitute fixtures.

Overlap is intentional and narrow: dataset inspection owns schema/PID/tree
inspection; preprocessing QA owns the aggregate closure decision; four-vector
validation owns detailed closure visualization; direct-GPT inspection owns
historical compatibility. The index's `scientific_claims_allowed` field is the
authority for claims from each artifact.
