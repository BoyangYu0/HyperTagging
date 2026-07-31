# Post-audit completion report

Date: 2026-07-31

## Repository identity and status

- Actual final `git rev-parse HEAD`:
  `68a7ebe81b14b6c329d78de91be18b5458c49089`.
- Branch: `master`, the current default branch and its inspected HEAD.
- Status: **dirty by design**. The correction is an uncommitted working-tree
  patch on the SHA above; therefore that SHA alone does not contain these
  corrections. `git status --short` lists the modified/new source, tests,
  configs, notebooks, generators, and the three new reports.
- `git diff --check`: passed with no whitespace errors.
- Historical audit snapshots named in the task were not modified.
- No commit, push, PR, or HTCondor submission was made.

The inspected log was:

```text
68a7ebe small fixs
d8be636 corrections
274d2df code audit
d236284 fix data production
9d37fde fix preprocessing
da8f357 update gpt like functions
cf63b90 update data production
7471c97 add new components for GPT training on generic
bbcd120 cleanup and add new data production
82ebd98 init
f7ff44b init
```

## Exact test commands and results

Before editing:

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

Result: `231 passed, 8 skipped, 18 warnings in 351.78s`.

Final complete suite:

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

Result: `241 passed, 8 skipped, 18 warnings in 345.01s (0:05:45)`.
The eight skips are the repository's expected environment-dependent tests.

Additional focused final checks included:

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q \
  tests/test_post_audit_geometry_rollout_cpu.py \
  tests/test_contextual_hyperbolic_encoder_cpu.py \
  tests/test_validation_batch_size_cpu.py \
  tests/test_checkpoint_cpu.py \
  tests/test_checkpoint_architecture_mismatch_cpu.py
```

Result: `16 passed in 23.06s`.

```bash
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q \
  tests/test_revised_notebooks_cpu.py -k real_only
```

Result: `2 passed, 1 deselected in 12.33s`. These tests execute the guard cells
and verify that the trained-physics notebook raises `REAL INPUT REQUIRED` and
the real-pilot notebook forbids fixture substitution.

`python -m compileall -q src scripts tests` and `git diff --check` also passed.

## Exact notebook commands and results

Before editing, the documented runner executed all 12 historical groups:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-post-audit-baseline-notebooks
```

Final run:

```bash
/data/dust/user/boyangyu/uv_env/bin/python \
  scripts/execute_notebook_smoke_tests.py \
  --keep-output /tmp/hypertagging-post-audit-final-notebooks
```

Result: `Executed 15 notebooks on CPU fixtures`. This includes all 12
historical groups plus:

- `inspect_exact_tree_geometry_and_loss_scales.ipynb`;
- `inspect_rollout_search_and_calibration.ipynb`;
- `inspect_runtime_scaling.ipynb`.

Every executed copy is under
`/tmp/hypertagging-post-audit-final-notebooks`; semantic JSON checks passed.
The full pytest run independently executed the same 15-group test again.

Key executed fixture evidence:

- exact direct-daughter/root path: 1 edge; rejected height synonym: 4;
- eligible different-B negatives: positions 3 and 7;
- parent coverage: 8 eligible children, 8 with different-B negatives, 0 with
  no negative, active fraction 1.0, ranking denominator 8;
- preset radius p95: 0.3822 (`tiny_cpu`), 0.3643 (`gpu_debug`), 0.6975
  (`production_baseline`); boundary fraction 0.0 and finite gradients for all;
- all four rollout PID modes were reproducible with p4 closure 1.0 and their
  relation/decision/construction modes were recorded explicitly;
- the search fixture recorded duplicate/overlap/query metrics and a 0.75
  greedy-versus-set-packing and greedy-versus-best-beam typed-set difference;
- runtime artifacts are one-event CPU measurements only and set
  `throughput_claim=false`.

`inspect_trained_physics_validation.ipynb` was not executed with physics inputs
because no explicit real parquet or trained checkpoint was supplied. Its guard
was directly verified; no fixture result is presented as trained physics.

## Real basf2 pilot

Status: **NOT RUN**. `command -v basf2` returned no executable, and no documented
real mDST input was available. The exact operator command embedded in
`inspect_real_mdst_pilot.ipynb` is:

```bash
source /cvmfs/belle.cern.ch/tools/b2setup release-08-03-00
basf2 scripts/preprocess_mdst.py -- \
  --input /path/to/generic_mdst.root \
  --output /data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
  --schema-version direct-mdst-tree-v4 --entry-sequence 0:49 \
  --max-events 50 --event-buffer-size 32 --row-group-size 16
HYPERTAGGING_REAL_PILOT=/data/dust/user/boyangyu/hypertagging/pilot-v4.parquet \
  /data/dust/user/boyangyu/uv_env/bin/python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/inspect_real_mdst_pilot.ipynb
```

No real-pilot, PIDLikelihood, K_L/KLM, fit-choice, or physics result is claimed.

## CI provenance

No CI run corresponds to this uncommitted corrected worktree, so there is no
applicable CI run ID to report. No old audit's CI prose was reused as evidence.

## Acceptance status and unresolved questions

The CPU/software acceptance criteria are directly satisfied: full tests and 15
fixture notebook groups pass; class-4 different-B negatives remain eligible;
exact unbalanced-tree edges pass; graph distance is distinct from generation
level; preset initialization is finite and dimension-aware; PID rollout modes
are explicit; overlapping pair mass/energy is withheld; mother p4 closure and
truth-input exclusion are tested; greedy remains reproducible; and no Condor,
full preprocessing, long training, or fixture-based physics claim occurred.

Still unresolved scientifically:

- real-pilot detector provenance and K_L/KLM coverage;
- trained held-out calibration and physics efficiency/purity;
- optimal geometry/scale/loss weights;
- beam or set-packing promotion versus greedy;
- representative batched-rollout and full-scale runtime/memory;
- real retained-tree contraction, unmatched-object, partial-topology, channel,
  multiplicity, and depth distributions;
- Mbc, DeltaE, missing mass, and rare/unseen-channel performance under explicit
  beam/frame/channel contracts.

These are marked as deferred rather than inferred from fixtures or historical
audit prose.

