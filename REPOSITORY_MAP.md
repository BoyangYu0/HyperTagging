# Repository Map

Compact documentation map for the unified migration repository. The original
Phase 0 inventory remains in `../REPOSITORY_MAP.md`; this file records how that
history maps onto the migrated package surface through Phase 13.

## Historical Repository Roles

| Repository | Historical role | Dataset/source | Unified status |
|---|---|---|---|
| `HyperTagging` | First Toy-MC HyperTagging studies, including BASF2 generation, preprocessing, embedding, and reconstruction experiments. | Toy-MC inputs after BASF2 generation and before preprocessing under `/home/boyang/data/MC`. | Toy-MC contracts, dry-run preprocessing adapters, selected model aliases, losses, and examples are migrated. Full BASF2/data production is not migrated. |
| `HyperTaggingColab` | Cleaner collaboration-style package for GraFEI embedding utilities, models, losses, trainer patterns, and preprocessing scripts. | GraFEI-derived data, planned original root `/home/boyang/data/graFEI`. | Package shape, losses, training-loop conventions, and reusable utility ideas informed the unified layout. |
| `graFEI` | Early full GraFEI HyperTagging workflow with embedding, link prediction, reconstruction, and full evaluation scripts. | Original GraFEI data under `/home/boyang/data/graFEI`; many historical scripts use hard-coded HPC paths. | Full reconstruction evaluation behavior from `whole_eva.py` is migrated for CPU tiny events. Full-data evaluation remains HPC-only. |
| `graFEI_reduced` | Reduced/final GraFEI workflow and primary source for many self-contained model definitions. | GraFEI original inputs and derived pair/reconstruction data. | Main source for migrated embedding, link, and reconstruction model classes plus single-level reconstruction behavior. |
| `graFEI_gpt` | GPT-like/autoregressive GraFEI branch. | GraFEI original inputs and GPT-like derived flattened data. | GPT-like masks, collate layout, losses, `GPTReconstructor`, `EmbLinker`, and a conservative `MultiGPT` are migrated. Ambiguous historical `MultiGPT` PDG/feature branch is documented but not reimplemented. |

## Unified Package Surface

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

## Revised inspection artifacts

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

## Checkpoints, Logs, And Data

Historical repositories contain many `.pth` checkpoints and log files. The
unified repository does not copy those artifacts. Checkpoint loading helpers
preserve `model_state_dict` and `epoch` handling and default to CPU loading.

Full-data reproduction requires external data and historical checkpoint
provenance not fully verified from repository contents.

## Obsolete Or Experimental Sources

Historical backup folders, `.ipynb_checkpoints/`, `__pycache__/`, large logs,
and exploratory notebooks remain in their source repositories. They are not
migrated unless a later phase needs them for an equivalence test.

## Components Still Treated As Sensitive

Do not refactor these before stronger equivalence tests exist:

- `baum_utils` LCAG/LCA reconstruction behavior.
- GraFEI level ordering, mass tokenization, and link construction.
- Toy-MC channel/event numbering and preprocessing conventions.
- Loss weights, masking logic, and model constructor defaults.
- Full-reconstruction stopping, signal probability, and link index remapping.
- GPT-like level masks, collate layout, and autoregressive masking.

## Final correctness and scalability revision

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

## Final runtime and scale revision

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
- `docs/final_runtime_and_scale_audit.md`: baseline, issues, compatibility,
  implementation outcome, and verification boundary.
