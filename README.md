# hypertagging-unified

Planned unified repository for the historical HyperTagging project.

This Phase 1 skeleton intentionally contains no migrated scientific code. Model
definitions, loss functions, preprocessing routines, reconstruction logic, and
evaluation code must only be added in later phases after equivalence tests are
defined.

## Data Roots

- Toy-MC inputs after BASF2 generation and before preprocessing:
  `/home/boyang/data/MC`
- Original GraFEI inputs:
  `/home/boyang/data/graFEI`

Derived/preprocessed folders such as `emb/`, `comb/`, `gpt/`, `ConstEmb/`, and
`RegEmb/` should be produced or configured explicitly in later phases.

## Phase 1 Contents

- Python package skeleton under `src/hypertagging/`.
- Empty configuration folders under `configs/`.
- Test scaffold under `tests/`.
- Placeholder docs for migration notes, repository map, legacy code, notebooks,
  and examples.

## Current Status

Import smoke testing is the only expected test in this phase:

```bash
python -m pytest tests/test_import_cpu.py
```

Full training and scientific reproduction remain out of scope until later
migration phases.
