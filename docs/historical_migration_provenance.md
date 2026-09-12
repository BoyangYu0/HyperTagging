# Historical migration provenance

This consolidated record preserves the migration decisions and historical
source mapping that previously lived in two root-level notes. It preserves
the Phase 1–13 migration history and the later schema-v4/runtime addenda below;
those addenda do not describe the Phase-13 implementation. This is not a
current readiness statement.
Use the [current audit](audits/current_status.md) for mutable status and the
[README](../README.md) for the active repository guide.

## Migration Notes

This file will track behavior-preservation decisions during Goal B.

### Phase 1

- Created package skeleton only.
- No scientific code was migrated.
- No model, loss, preprocessing, reconstruction, evaluation, checkpoint, or data
  behavior has been changed.

### Phase 2

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

### Phase 3

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

### Phase 4

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

### Phase 5

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

### Phase 6

- Mechanically migrated self-contained GraFEI/reduced model definitions into
  `models/common.py` and exposed grouped target modules for embedding, link
  prediction, reconstruction, and Toy-MC aliases.
- Added GPT-like `ParticleEmbedder`, `EmbLinker`, and `GPTReconstructor`.
- Did not migrate `MultiGPT`: the historical `graFEI_gpt/models.py` class
  references undefined globals and uninitialized heads in its constructor/body.
- CPU tests compare state-dict keys/shapes and forward outputs against legacy
  classes for representative self-contained models.

### Phase 7

- Added CPU dry-run training loops for embedding, link prediction,
  reconstruction, and GPT-like reconstruction stages.
- Dry runs build a small model, one synthetic batch, stage loss, and optimizer,
  and can run one backward/optimizer step.
- Preserved historical optimizer families at the stage level: `Adam` for
  embedding and `AdamW` for link/reconstruction/GPT-like stages.
- Did not migrate full epoch loops, early stopping, LR scheduler state,
  checkpoint save timing, logging policy, or HPC shell-wrapper behavior.

### Phase 8

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

### Phase 9

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

### Phase 10

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

### Phase 11

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

### Phase 12

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

### Phase 13

- Replaced stale Phase 1 placeholder documentation with current migration
  status through Phase 12.
- Synchronized the internal repository map with the migrated package surface.
  The surviving historical source map is preserved below.
- Updated notebook and legacy documentation to clarify that notebooks and frozen
  legacy copies are not migrated implementation sources.
- Documented CPU smoke commands, local data roots, known limitations, and
  equivalence-test status.
- Did not migrate new scientific code.

### Future Notes

For every migrated component, record:

- historical source file;
- target file;
- equivalence test;
- known ambiguities;
- any discrepancy between thesis-level description and repository behavior.

## Repository Map

This section records the historical repository mapping. The unified-package
surface and revised-inspection subsections include later v4/runtime annotations.

### Historical Repository Roles

| Repository | Historical role | Dataset/source | Unified status |
|---|---|---|---|
| `HyperTagging` | First Toy-MC HyperTagging studies, including BASF2 generation, preprocessing, embedding, and reconstruction experiments. | Toy-MC inputs after BASF2 generation and before preprocessing under `/home/boyang/data/MC`. | Toy-MC contracts, dry-run preprocessing adapters, selected model aliases, losses, and examples are migrated. Full BASF2/data production is not migrated. |
| `HyperTaggingColab` | Cleaner collaboration-style package for GraFEI embedding utilities, models, losses, trainer patterns, and preprocessing scripts. | GraFEI-derived data, planned original root `/home/boyang/data/graFEI`. | Package shape, losses, training-loop conventions, and reusable utility ideas informed the unified layout. |
| `graFEI` | Early full GraFEI HyperTagging workflow with embedding, link prediction, reconstruction, and full evaluation scripts. | Original GraFEI data under `/home/boyang/data/graFEI`; many historical scripts use hard-coded HPC paths. | Full reconstruction evaluation behavior from `whole_eva.py` is migrated for CPU tiny events. Full-data evaluation remains HPC-only. |
| `graFEI_reduced` | Reduced/final GraFEI workflow and primary source for many self-contained model definitions. | GraFEI original inputs and derived pair/reconstruction data. | Main source for migrated embedding, link, and reconstruction model classes plus single-level reconstruction behavior. |
| `graFEI_gpt` | GPT-like/autoregressive GraFEI branch. | GraFEI original inputs and GPT-like derived flattened data. | GPT-like masks, collate layout, losses, `GPTReconstructor`, `EmbLinker`, and a conservative `MultiGPT` are migrated. Ambiguous historical `MultiGPT` PDG/feature branch is documented but not reimplemented. |

### Unified Package Surface

- `hypertagging.data`: contracts, reduced-token tiny fixtures, schema-v4-first heterogeneous parquet
  loading, source-aware splitting, train-only normalization, historical
  preprocessing command builders, and GPT-like adapters.
- `hypertagging.utils`: padding, checkpoint loading, device, seed, and I/O
  helpers.
- `hypertagging.preprocessing`: verified legacy exports plus corrected
  truth-separated v4, full/reconstructable channel signatures, recursive
  provenance, and v1/v2 compatibility adapters.
- `hypertagging.losses`: historical losses plus balanced LCA relations, true
  Poincare parent ranking, corrected radius depth, and VICReg variance/covariance.
- `hypertagging.models`: historical models plus heterogeneous frontends, one
  shared Poincare encoder, relation-aware set attention, task projections, and
  named ablations.
- `hypertagging.training`: real parquet data module, curriculum pretrainer,
  all-level reconstruction trainer, encoder transfer, atomic checkpoints,
  CPU dry-runs, and JSONL logging.
- `hypertagging.reconstruction`: historical reconstruction plus complete
  teacher-forced/scheduled/free level rollout and overlap resolution.
- `hypertagging.evaluation`: GraFEI metrics plus hierarchical edge/tree,
  closure, parent, pointer, channel, and rare/unseen helpers.
- `scripts/`: CPU dry-run CLIs, preprocessing wrappers, deterministic notebook
  generators/execution, and HTCondor renderers.
- `examples/`: runnable fixture-based CPU examples.
- `configs/ablations/`: flat through full revised experiment controls.

### Revised inspection artifacts

- `docs/audits/current_status.md`: sole current audit; immutable history is in
  `docs/audits/archive/`.
- `docs/channel_representation.md`: exact and structured two-B semantics.
- `docs/heterogeneous_node_encoding.md`: feature blocks and shared encoder.
- `docs/dataset_visualization.md`: fixture and real-data notebook execution.
- `notebooks/inspect_preprocessed_dataset.ipynb`
- `notebooks/inspect_hyperbolic_pretraining.ipynb`
- `notebooks/inspect_level_autoregressive_reconstruction.ipynb`
- `notebooks/preprocessing_qa_report.ipynb`
- `notebooks/inspect_leaf_input_pid_contract.ipynb`
- `notebooks/inspect_query_capacity_and_losses.ipynb`
- `notebooks/inspect_training_pipeline.ipynb`
- `notebooks/inspect_production_manifest.ipynb`

### Checkpoints, Logs, And Data

Historical repositories contain many `.pth` checkpoints and log files. The
unified repository does not copy those artifacts. Checkpoint loading helpers
preserve `model_state_dict` and `epoch` handling and default to CPU loading.

Full-data reproduction requires external data and historical checkpoint
provenance not fully verified from repository contents.

### Obsolete Or Experimental Sources

Historical backup folders, `.ipynb_checkpoints/`, `__pycache__/`, large logs,
and exploratory notebooks remain in their source repositories. They are not
migrated unless a later phase needs them for an equivalence test.

### Components Still Treated As Sensitive

Do not refactor these before stronger equivalence tests exist:

- `baum_utils` LCAG/LCA reconstruction behavior.
- GraFEI level ordering, mass tokenization, and link construction.
- Toy-MC channel/event numbering and preprocessing conventions.
- Loss weights, masking logic, and model constructor defaults.
- Full-reconstruction stopping, signal probability, and link index remapping.
- GPT-like level masks, collate layout, and autoregressive masking.

### Final correctness and scalability revision

- `preprocessing/schema_v4.py`: one-event-per-row schema, bounded atomic
  Parquet writer, metadata sidecar, explicit input/truth daughter PID
  histograms, and legacy adapters.
- `reconstruction/pid_state.py`: authoritative runtime PID probabilities,
  charge-compatible raw-track PID, differentiable p4, and daughter histogram
  rebuilding shared by teacher forcing and rollout.
- `data/streaming.py`: worker-partitioned event iteration, bounded deterministic
  shuffle, and mergeable masked Welford normalization.
- `training/data_module.py`: production-manifest resolution (`output_file`
  included), source-safe splits, legacy safety gate, and lazy batches.
- `training/scheduled_sampling.py`: scheduled context selection and
  recursive-source alignment.
- `models/level_autoregressive.py`: detector-context PID pass followed by
  PID-refined reconstruction context and hyperbolic projection.
- `evaluation/hierarchical_metrics.py`: source/topology alignment independent
  of mother type before type scoring.
- `notebooks/inspect_leaf_pid_and_composite_inputs.ipynb` and
  `notebooks/inspect_streaming_dataset.ipynb`: deterministic fixture/real-data
  inspection for the new contracts.

### Final runtime and scale revision

- `data/streaming.py`: checkpointed runtime normalization, serializable cursor,
  bounded shuffle, and file/row-group worker partitioning.
- `data/dataset_index.py`: versioned mergeable startup index with split,
  Welford, capacity, PID, depth, and completeness statistics.
- `preprocessing/schema_v5.py`: experimental native nested Arrow rows and a
  bounded comparison against JSON-in-Parquet v4.
- `training/reconstruction_trainer.py`: one sampled primary context per
  event/level and multi-event metric aggregation.
- `training/pretraining_curriculum.py`: actual applied corruption labels and
  relation-valid hard negatives.
- `scripts/build_dataset_index.py` and `scripts/benchmark_parquet_storage.py`:
  explicit scale-preparation tools.
- The [archived final runtime and scale audit](audits/archive/2026-07-31_d236284_final_runtime_and_scale_audit.md): baseline, issues, compatibility,
  implementation outcome, and verification boundary.
