# Migration Notes

This file will track behavior-preservation decisions during Goal B.

## Phase 1

- Created package skeleton only.
- No scientific code was migrated.
- No model, loss, preprocessing, reconstruction, evaluation, checkpoint, or data
  behavior has been changed.

## Phase 2

- Migrated only shared utility and checkpoint-loading helpers.
- `pad_to` and `pack_with_evtNum`/`pack_with_evt_num` preserve the historical
  awkward-array padding and event grouping behavior from the source repositories.
- `pad_to(..., pad_length=...)` preserves the GraFEI utility extension.
- `pad_to(..., v_kwargs=...)` preserves the legacy `graFEI_gpt` split behavior,
  including its axis slicing convention.
- checkpoint helpers default to `map_location="cpu"` while preserving the
  historical `model_state_dict` and `epoch` dictionary keys.
- CPU smoke-loaded `/home/boyang/code_backups/HyperTagging/pretrained_AutoInt.pth`
  through the new checkpoint helper: dictionary checkpoint, epoch `99`, 55
  `model_state_dict` entries.
- No models, losses, preprocessing routines, reconstruction routines, notebooks,
  checkpoints, or data files were migrated.

## Phase 3

- Added provisional data contracts and tiny synthetic fixture functions only.
- Contracts document observed batch dictionaries from historical dataset classes
  in `SampleEmbedding.py`, `Link.py`, `Reconstruction.py`, `utils.py`, and
  `produce_train_data_grafei.py`.
- Fixture batches are synthetic NumPy arrays for CPU validation; they are not
  produced by migrated preprocessing and do not encode physics results.
- Ambiguous fields remain provisional, including toy-MC `arrayIndex` and the
  exact required scope of GraFEI `channels`/`evtNums` across all workflows.
- No models, losses, preprocessing routines, reconstruction routines, notebooks,
  checkpoints, or data files were migrated.

## Phase 4

- Added subprocess-based preprocessing adapters only.
- Default planned input roots are `/home/boyang/data/MC` for non-Colab Toy-MC
  inputs after BASF2 generation and before preprocessing, and
  `/home/boyang/data/graFEI` for original GraFEI inputs before preprocessing.
- The adapters build dry-run commands for historical scripts and can pass the
  planned roots through `HYPERTAGGING_INPUT_ROOT` and
  `HYPERTAGGING_OUTPUT_ROOT` to wrapper-aware subprocesses.
- Historical scripts still contain hard-coded legacy paths; no scientific
  preprocessing logic was copied or edited to reinterpret those paths.
- No models, losses, scientific preprocessing algorithms, reconstruction
  routines, notebooks, checkpoints, or data files were migrated.

## Phase 5

- Migrated pure tensor loss functions into grouped modules:
  `embedding_losses.py`, `link_losses.py`, `reconstruction_losses.py`, and
  `gpt_losses.py`.
- Preserved historical variants instead of deduplicating them: GraFEI vs
  Toy-MC radius targets, `graFEI` vs `graFEI_reduced` momentum weighting, Colab
  masked losses, and GPT-like reconstruction/radius losses.
- Model-dependent losses that call pretrained historical models are represented
  only by their tensor-level distance components; model calls remain future
  Phase 6+ work.
- Added CPU-only scalar equivalence tests on tiny tensors.

## Phase 6

- Mechanically migrated self-contained GraFEI/reduced model definitions into
  `models/common.py` and exposed grouped target modules for embedding, link
  prediction, reconstruction, and Toy-MC aliases.
- Added GPT-like `ParticleEmbedder`, `EmbLinker`, and `GPTReconstructor`.
- Did not migrate `MultiGPT`: the historical `graFEI_gpt/models.py` class
  references undefined globals and uninitialized heads in its constructor/body.
- CPU tests compare state-dict keys/shapes and forward outputs against legacy
  classes for representative self-contained models.

## Phase 7

- Added CPU dry-run training loops for embedding, link prediction,
  reconstruction, and GPT-like reconstruction stages.
- Dry runs build a small model, one synthetic batch, stage loss, and optimizer,
  and can run one backward/optimizer step.
- Preserved historical optimizer families at the stage level: `Adam` for
  embedding and `AdamW` for link/reconstruction/GPT-like stages.
- Did not migrate full epoch loops, early stopping, LR scheduler state,
  checkpoint save timing, logging policy, or HPC shell-wrapper behavior.

## Phase 8

- Added a single-level reconstruction-step API preserving the historical order:
  model forward, optional energy sorting, PDG recovery, reconstructed batch
  construction, and PDG/feature/embedding/link loss terms.
- Preserved variant differences for `grafei_reduced`, `grafei`, and `toy_mc`:
  recovered-PDG masking, momentum weighting, embedding distance type, and link
  loss path.
- Added CPU-only equality tests against the historical inline formulas on tiny
  tensors.
- Did not migrate full reconstruction evaluation, recursive/full-event
  reconstruction, or link-prediction workflows beyond the single-step loss hook.

## Phase 9

- Added explicit link-prediction training modes:
  `ground_truth` uses ground-truth daughters and ground-truth mothers;
  `reconstructed_mother` uses ground-truth daughters and reconstructed mothers.
- Added standard/corrected/embedding link surfaces. `CorrectedLinker` is kept as
  an alias of the historical `linearLinker`; corrected behavior is represented
  by the reconstructed-mother input path and optional teacher-logit transfer
  loss.
- Updated the link dry-run CLI to accept `--mode ground_truth` and
  `--mode reconstructed_mother`.
- Added CPU-only tests for logits shape, cross-entropy equality, reconstructed
  mother input construction, corrected teacher path equality, and embedding-link
  dry-run shape.
- Did not migrate full epoch link training, file loaders, early stopping,
  checkpoint save timing, or notebook-only workflows.

## Phase 10

- Migrated the event-level GraFEI full-reconstruction evaluation behavior from
  `graFEI/whole_eva.py`.
- Preserved argmax PDG recovery, link max-value signal scoring, empty-mother
  link remapping, daughter feature aggregation, iterative predicted-LCA
  construction, root stopping on PDG token `13`, and failure-row behavior.
- Added CPU-only tests for predicted LCA, PDG accuracy, feature error, stopping,
  signal probability, and the evaluation dry-run CLI on tiny events.
- Preserved a historical padded-accuracy convention that can produce
  `pdgAcc > 1` on synthetic padded fixtures.
- Did not migrate full-data parquet evaluation jobs, plotting notebooks, or HPC
  batch execution.

## Phase 11

- Added GPT-like/autoregressive data helpers preserving the historical
  `get_level_mask`, shifted-target collate, link-index offsetting, mass-token
  padding, and `lvl_code = exp(-level)` conventions from
  `graFEI_gpt/Reconstruction.py`.
- Added a CPU-capable `MultiGPT` that preserves the verified branches of the
  historical class: autoregressive embedding reconstruction and embedding-link
  prediction.
- The historical `graFEI_gpt/models.py::MultiGPT` class remains ambiguous and
  was not copied literally: it references undefined constructor globals
  (`num_pdg`, `pdg_emb`), undefined PDG/feature heads, and returns an undefined
  `particle_emb`.
- Added `scripts/run_gpt_like.py` for a combined GPT-like CPU dry run.
- Did not migrate full epoch GPT training, file loaders, early stopping,
  checkpoint save timing, or HPC logging/checkpoint behavior.

## Phase 12

- Added runnable CPU-only examples for Toy-MC, GraFEI, and GPT-like workflows
  under `examples/`.
- Examples use tiny synthetic fixtures, migrated contract validators, migrated
  losses or dry-run training paths, and preprocessing dry-run command builders.
- Documented full-data roots as `/home/boyang/data/MC` for Toy-MC after BASF2
  generation and before preprocessing, and `/home/boyang/data/graFEI` for
  original GraFEI before preprocessing.
- Added CPU tests that execute all examples as subprocesses and parse their JSON
  summaries.
- Did not add runnable full-data examples, GPU training examples, performance
  reproduction scripts, or new scientific behavior.

## Phase 13

- Replaced stale Phase 1 placeholder documentation with current migration
  status through Phase 12.
- Synchronized the internal repository map with the migrated package surface
  while leaving the full Phase 0 inventory in `../REPOSITORY_MAP.md`.
- Updated notebook and legacy documentation to clarify that notebooks and frozen
  legacy copies are not migrated implementation sources.
- Documented CPU smoke commands, local data roots, known limitations, and
  equivalence-test status.
- Did not migrate new scientific code.

## Future Notes

For every migrated component, record:

- historical source file;
- target file;
- equivalence test;
- known ambiguities;
- any discrepancy between thesis-level description and repository behavior.
