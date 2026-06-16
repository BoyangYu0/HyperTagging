# Tests

Phase 1 contains only an import smoke test.

Phase 2 adds CPU-only tests for migrated utility functions and checkpoint
loading helpers. These tests intentionally avoid model, loss, preprocessing,
and reconstruction behavior.

Phase 3 adds CPU-only tests for data contracts and tiny synthetic fixtures.
These tests validate keys, ranks, dtype families, and symbolic shape equality;
they do not exercise historical preprocessing.

Phase 4 adds CPU-only tests for preprocessing adapters. These tests validate
dry-run command construction, planned input roots, and environment forwarding;
they do not run BASF2 or historical preprocessing jobs.

Phase 5 adds CPU-only scalar equivalence tests for migrated loss functions on
tiny tensors. Model-dependent historical losses are tested at the tensor-loss
component boundary only.

Phase 6 adds CPU-only model tests for migrated model definitions. The tests
instantiate small models, compare state-dict keys/shapes with legacy classes,
copy legacy state dicts, and compare forward outputs where the historical class
is self-contained.

Phase 7 adds CPU-only dry-run tests for training stages. The tests build the
model, one synthetic batch, stage loss, and optimizer, then run one optional
backward/optimizer step without checkpointing or full training.

Phase 8 adds CPU-only tests for one single-level reconstruction step. The tests
compare recovered PDG, feature masking, energy sorting, embedding terms, and
link terms against historical inline formulas on tiny tensors.

Phase 9 adds CPU-only tests for link prediction modes: from-scratch training on
ground-truth daughters/mothers and fine-tuning with ground-truth daughters plus
reconstructed mothers. It also covers the corrected teacher-logit path and
embedding-link dry-run shape.

Phase 10 adds CPU-only tests for full GraFEI reconstruction evaluation. The
tests cover predicted LCA construction, PDG accuracy, feature error, iterative
stopping, failure rows, and the evaluation dry-run CLI on a tiny event.

Phase 11 adds CPU-only tests for GPT-like/autoregressive components: exact
level-mask construction, flattened collate behavior, GPT reconstructor and
embedding-linker legacy parity, combined `MultiGPT` forward/loss smoke tests,
and the `run_gpt_like.py` dry-run CLI.

Phase 12 adds CPU-only tests that run every minimal example as a subprocess and
verify the JSON summary plus documented local data roots.

Phase 13 documentation cleanup was verified by the full CPU smoke suite. No new
scientific behavior was added.

Future phases must add equivalence tests before migrating scientific behavior.
