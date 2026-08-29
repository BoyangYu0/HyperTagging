# Full-decay evaluator compatibility handoff

This handoff is the training/HPO integration record for the CPU-only offline
full-decay evaluator. It covers schema-v4 preprocessed-mDST input only; it is
not a basf2 or GraFEI reconstruction path. The machine-readable mirror is
[`full_decay_training_hpo_compatibility_handoff.json`](full_decay_training_hpo_compatibility_handoff.json).

## Frozen evidence

- Repository HEAD at audit: `2d5909d5e222874c610e5fd0678f732f9555c59a`;
  every evaluation report hashes raw Git index metadata plus cached diff bytes,
  requires index/worktree equality, and lists untracked names only without
  reading their contents.
- Selection manifest:
  `configs/training_selection/production_1m_20260812/train_035k.json`, contract
  hash `b55cebee16b6dfa55b7fee943e28cca0de325132547b34dc13fffc24c418df72`
  and file SHA-256
  `48108c18e61f5c7494fd11958f9e93911ac21df4e47f8e518025f8a99929a6eb`.
- Promoted `complete_only` index:
  `artifacts/experiment_readiness/production_1m_20260812/train_035k/train_035k.complete_only.index.json`,
  index hash `5fc837315b2e6f5e4783cba2808bfba7672cf4ba12bff1fe5cf050f5d22b6de1`;
  schema `direct-mdst-tree-v4`; 35,000 train and 50,000 validation UIDs;
  no test role.
- Pretraining checkpoint step 54,064, SHA-256
  `997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d`.
- Reconstruction checkpoint step 2,000, SHA-256
  `06e1bf3288648f2e663012f2f58aa9b83451b65f04169d860f2bfc4b455744ee`.
- CPU pair validation: 121/121 transferred encoder tensors have matching keys,
  shapes, dtypes, and bit-exact values; feature, model-feature, and PID contracts
  match. Equal-valued tensors with different dtypes are rejected.
- Historical real acceptance artifact (now cryptographically stale):
  `artifacts/evaluation/full_decay_checkpoint_cohort_smoke_cpu.json`, using the
  first checkpoint rollout UID `1004:0:115966003:52205`. The artifact itself is
  intentionally ignored as generated output; its command and invariants are
  reproducible from the tracked evaluator and this handoff. Its SHA-256 is
  `1ece01688fcd06c9796ff67fadf72d43e81a0398cf57fa19e79890f9142d367b`;
  context loading took 141.23 s and both inference scopes took 6.45 s. That
  artifact predates the fail-closed cohort, manifest-alias, shared-loader, and
  rollout-policy-identity fixes in this audit and must not be reused for
  comparison. A new smoke from a clean commit is required before promotion.

## Compatibility matrix

| ID | Severity / status | Exact evidence and affected files | Tests | Smallest safe upstream action |
|---|---|---|---|---|
| DATA-01 | invariant / verified | `load_trained_evaluation_context` and `build_real_data_module` bind schema, feature/PID contracts, target policy, selection-manifest hash, index hash, source groups, normalizers, and split. Files: `src/hypertagging/evaluation/trained_context.py`, `src/hypertagging/training/data_module.py`, `src/hypertagging/data/dataset_index.py`. | `test_trained_evaluation_restores_normalization_and_heldout_contract`; index/identity suites; real smoke. | Keep the manifest and promoted index immutable and publish both hashes with every HPO result. |
| DATA-02 | P0 fixed in evaluator and shared loader | `load_dataset_index_metadata` authenticates only index JSON object/version/index hash and selection-contract schema/mode/hash fields. It never canonicalizes shard paths, recomputes the selection fingerprint, stats/opens shards, or inspects sidecars. Evaluator and shared loader then authenticate the supplied manifest JSON/hash before full loading. Files: `dataset_index.py`, `trained_context.py`, `data_module.py`. | resolver, `_selection_fingerprint`, publication, selection-publication, and indexed-shard bombs stay at zero for raw-list/wrong-manifest rejection; trained context delegation and invalid metadata-schema tests pass. | Retain this central pre-effect invariant for every caller. |
| DATA-03 | fixed | Split precedence is event override, then source-role override, then stable hash; selected UIDs are checked for uniqueness, train overlap, and requested split. File: `trained_context.py::_evaluation_split_assignment`. | `test_evaluation_split_assignment_honors_source_role_manifest`; `...event_override_first`. | Retain these checks as fail-closed gates. |
| DATA-04 | fixed | Validation `auto` restores the checkpoint's ordered 1,000-UID rollout cohort from its deterministic 2,000-event validation selection, validates selection-manifest identity, applies category filtering without changing cohort rank, and records UID list plus digest. File: `trained_context.py::_resolve_checkpoint_event_selection/_select_evaluation_events`. | `test_checkpoint_rollout_cohort_restores_uid_order_before_category_limit`; real smoke selected the stored first UID. | Use this cohort for comparisons; label `stream` runs diagnostic and never mix them in HPO ranking. |
| DATA-05 | P1 open contract | The promoted index contains train/validation only. CLI `--split test` correctly fails with this index. | Dataset-index validation plus inspected promoted metadata. | Materialize a separate immutable test-role manifest/index after validation policy freeze; never relabel validation. |
| BATCH-01 | invariant / verified | Collation preserves event order and pads to batch maxima. Projection requires exact native shapes: features/availability `[B,N,W]` at widths 12/16/9/9/13, PID histogram `[B,N,41]`, scalar masks/IDs `[B,N]`, p4 `[B,N,4]`, adjacency `[B,N,N]`, and provenance `[B,N,S>0]`. Floats are `float32`, masks/adjacency/provenance `bool`, IDs `int64`. Files: `data/heterogeneous.py`, `reconstruction/hierarchical_inference.py`. | projection/collation suites; `test_projection_rejects_compatibility_adapter_and_non_normalized_batches`; parameterized rank/width mutations. | Keep tensor validation at the evaluator boundary. |
| BATCH-02 | P2 open throughput | Projection and batched rollout support `[B,...]`, but the CLI collates and runs one event at a time. File: `scripts/evaluate_full_decay.py`. | Batch-size-one end-to-end tests; lower-level batched parity tests. | Add a two-event unequal-FSP parity test before exposing a small CPU `--batch-size`. |
| FSP-01 | hard invariant / fixed | Only active level-zero track, ECL, and KLM rows are physically gathered. Stored composites, parents, adjacency, levels, B-side, truth PID targets, truth geometry, and truth ancestry cannot reach the model. Files: `reconstruction/hierarchical_inference.py`, `tests/test_hierarchical_inference_cpu.py`. | projection scrubbing and higher-node perturbation tests; real smoke discarded all 5 active higher nodes and passed only 31 FSPs. | Never weaken this projection; new truth-only fields must be added to its scrub audit. |
| FSP-02 | fixed | Unique FSP evaluation keys/topology membership are rebuilt from adjacency, separately from multi-hot detector-resource provenance used for conflict rejection. Files: `evaluation/full_decay_metrics.py`, `evaluation/full_decay_runner.py`. | detector conflict and topology-key separation tests. | Preserve the two namespaces. |
| FSP-03 | hardening fixed | Stable evaluation keys live in `FSPProjection.evaluation_leaf_source_keys`, are absent from every model-call dictionary, and are attached only after rollout. File: `reconstruction/hierarchical_inference.py`. | scripted model asserts no `evaluation_leaf_source_keys` at every call. | Retain the side-channel boundary. |
| TREE-01 | fixed; policy drift disclosed | Decoding and append accept only unparented forest roots, enforce one parent, and apply detector-source exclusivity even without cardinality decoding. Track-A scheduled training shares this rollout code, so its historical scheduled-training semantics and stored `predicted_edge_f1` are stale after the correction. Files: `reconstruction/level_rollout.py`, `evaluation/full_decay_runner.py`. | parent-reuse, structural-validity, no-cardinality conflict, and reference/batched parity tests. | Re-evaluate every compared checkpoint; historical stored `predicted_edge_f1` is not directly comparable. |
| TREE-02 | fixed; versioned divergence | Offline inference advances through empty eligible levels and reports their count; training's generic rollout default remains stop-on-empty. Policy version is `fsp-forest-root-empty-level-soft-type-pid-parity-v2`. Files: `hierarchical_inference.py`, `level_rollout.py`. | empty-level legacy/continuation/reference parity test. | Adopt only through an explicit new HPO rollout contract; do not silently change training defaults. |
| TREE-03 | P1 fixed | `soft_decision_hard_construction` uses `soft_expectation` for model relations/decisions and hard PID only for persistent construction. Evaluation restores model PID mode from authoritative `feature_contract.pid_reconstruction_mode`, inherits it when config is explicitly null, rejects conflicting non-null config, passes it into construction, and reports the instantiated model's actual mode. Files: `level_rollout.py`, `trained_context.py`. | pre-decoder policy tests; null-inheritance and conflict regressions. | Treat pre-v2 offline reports as superseded and rerun candidates. |
| SEM-01 | invariant / verified | Full scope requires exactly one explicit top-level Upsilon token 1; no largest-root guess. One eligible event is one row; decode failure stays a failure. File: `full_decay_metrics.py::evaluate_full_decay`. | perfect/permuted full tree and no-invented-continuum-root tests. | Keep explicit-root availability separate from model success. |
| SEM-02 | invariant / verified | Signal half scope produces exactly two canonically unordered B-root rows. One-to-one source overlap assignment excludes PID/p4; extra predicted B/component roots are counted. File: `full_decay_metrics.py::evaluate_half_decays`. | unordered halves, ineligible-root multiplicity, conjunction tests. | Optimize only with assignment-parity tests. |
| SEM-03 | invariant / fixed | Continuum half scope is one multiplicity-one row per explicit top-level truth composite, assigned one-to-one by sources. No artificial two hemispheres or B-style both-halves trial; absent roots are unavailable. | continuum multiplicity, no-root, and hallucinated-extra-component tests. | If hemispheres are ever added, version them as a separate truth-only contract. |
| SEM-04 | P0 denominator bias fixed | Primary `checkpoint_direct` keeps a policy-eligible root with an ineligible intermediate or detector conflict as an available failed trial (`target_representable=0`), instead of dropping it. `contracted_diagnostic` is opt-in and non-primary; mother PID/p4 denominators include only policy-eligible mothers. File: `full_decay_metrics.py`. | unary direct-vs-contracted and detector-conflict regressions. | If contracted targets are desired, retrain/version them; never use contraction to improve the current checkpoint's primary denominator. |
| METRIC-01 | verified | Source-keyed LCAG is invariant to tensor/node/query order and PID. `perfectLCAG` also requires exact source set and structural validity. Strict missing-one and induced leave-one-out record denominators and omitted key. File: `full_decay_metrics.py`. | LCAG permutation, one-vs-two missing, DAG/conflict, and trivial-tree tests. | Use numerator/denominator pairs, not event-filtered means. |
| METRIC-02 | verified | Leaf PID is keyed by FSP source; mothers align on exact source clade, induced height, and immediate child clades before PID comparison. Leaf/mother/root accuracy, availability, coverage, and confusion counts are separate. | PID invariance, availability, exact mother-alignment tests. | Keep type tokens out of alignment costs. |
| METRIC-03 | diagnostic only | p4 component bias/MAE/RMSE, p3, relative p3, magnitude, mass, and coverage are implemented after topology alignment. Current retained mother p4 and predicted p4 are both reconstructed FSP daughter sums, so physical MC momentum resolution is unavailable and explicitly labeled false. Closure is eligible only when its denominator is positive; event and aggregate values are `null`, never perfect, at denominator zero. Files: `full_decay_metrics.py`, `full_decay_runner.py`, report/docs. | source-aligned p4 denominator/coverage and zero-denominator event/aggregate closure tests. | Add evaluation-only MC composite p4 in a future schema; scrub it before inference. |
| METRIC-04 | fixed | `rollout_event_valid` now recomputes final acyclic/single-parent/level/source validity instead of meaning merely initial-nonempty; independent runner validity and p4 closure remain reported. File: `level_rollout.py::_batched_event_structural_validity`. | malformed topology/rollout diagnostics and focused rollout suites. | Keep both rollout and independently recomputed runner validity in receipts. |
| CKPT-01 | invariant / verified | The actual pair matches feature/model/PID contracts and 121/121 frozen encoder tensors in key, shape, dtype, and value; dtype-compatible key count is explicit and equal-valued fp32/fp64 tensors are rejected. SHA-256 and steps are listed above. File: `evaluation/checkpoint_pair.py`. | pair-validator tests and real CPU validation command. | Fail closed unless an explicitly disclosed future fine-tuned encoder is intended. |
| CKPT-02 | invariant / verified | Reconstruction restores serialized architecture strictly, four normalizer blocks, constraint policy, PID kinematics, CPU device, and eval mode. Actual architecture has 128 model dims, 32 hyperbolic dims, 32 queries, max cardinality 16, and 41 PID tokens. Decoder shapes are object/confidence `[B,Q]`, type `[B,Q,41]`, pointer `[B,Q,N]`, cardinality `[B,Q,C+1]`. | trained-context and model API suites; real smoke. | Change APIs only with checkpoint-contract migration tests. |
| CKPT-03 | fixed | Pair validation requires nonempty feature/model/PID identifiers before equality; mutually missing fields no longer pass. File: `evaluation/checkpoint_pair.py`. | missing-contract rejection fixture. | Retain nonempty checks in standalone tools. |
| HPO-01 | fixed | Default learned-confidence enablement comes from the checkpoint selection contract (`true` for this checkpoint); CLI overrides are explicitly diagnostic and reported. File: `scripts/evaluate_full_decay.py`. | CLI default/override tests; real smoke reports checkpoint source and `true`. | Use the default for ranking; never combine override results. |
| HPO-02 | fixed receipt; publication action open | Provenance hashes raw `git ls-files -s -z` index metadata plus cached diff bytes, requires index/worktree equality, and lists untracked names without reading arbitrary payload content. Unstaged or untracked state marks provenance incomplete. File: `scripts/evaluate_full_decay.py::_evaluator_code_provenance`. | `restricted.parquet` untracked-name test with `Path.read_bytes` bomb; index/worktree mismatch regression. | Commit/freeze the evaluator before a publication or final HPO table. |
| PERF-01 | invariant / verified | CUDA is hidden before importing torch; checkpoints map to CPU; model must already be CPU/eval; default loader workers are zero and threads are bounded. No training process, checkpoint, optimizer, GPU, or scheduler is mutated. | CPU guard/projection tests; process/GPU inspection; real smoke. | Keep evaluation in a separate CPU process. |
| PERF-02 | P1 open | Context loading re-reads/verifies the index and hashes shards; pair validation and model restoration load the reconstruction checkpoint separately. Final smoke spent 141.23 s loading context versus 6.45 s for two-scope inference. | report phase timers. | Pass one verified index payload and loaded checkpoint payload through the pipeline. |
| PERF-03 | P1 open | `--scope both` performs separate full and half rollouts although most decoding is shared. File: `scripts/evaluate_full_decay.py`. | Current parity is semantic, not compute reuse. | Derive both views from one max-level rollout, then prove parity. |
| PERF-04 | P1 open | Each level appends all `Q=32` slots, including rejected padding; dense width approaches `N_FSP + 8Q`. Recursive closure is recomputed in a `total_count` loop over dense matrices, creating the principal CPU scaling risk. File: `level_rollout.py::batched_level_step`. | Real one-event phase timing; lower-level rollout tests. | Compact accepted slots and update tree closure incrementally before large CPU samples or batching. |
| PERF-05 | fixed | Report includes timers, throughput, policy/configuration, exact UIDs, structural counters, index-bound provenance, and strict finite JSON. Publication uses exclusive `tempfile.mkstemp` creation in `output.parent`, writes/flushes/fsyncs the returned descriptor, atomically replaces, and cleans only its owned temporary. | alias-safe atomic publication and strict-JSON tests. | Optionally add a detached artifact SHA-256 sidecar in publication automation. |
| SAFE-01 | P1 fixed | `--output` is checked before checkpoint/data loading against direct inputs and every manifest shard/sidecar/marker alias. Random exclusively created publication temporaries cannot follow or truncate predictable legacy symlink/hardlink aliases; normal report overwrite remains allowed. | direct/manifest alias tests and preexisting legacy-temp symlink/hardlink sentinel regression. | Reuse both the input preflight and owned-temp helper in future evaluators. |
| BASE-01 | P1 pre-existing, outside evaluator | Broad suite exposes duplicate `empty_channel_memory_expansion_v1` entries because `restore_training_checkpoint` reuses and re-extends the migration list. File: `training/checkpointing.py`; untouched here. | `test_empty_zero_capacity_channel_memory_can_expand_on_explicit_resume` fails. | Use a separate local variable for the channel-memory migration and append it exactly once. |
| BASE-02 | blocked external authorization, untouched | Phase-3 authorization binds runtime-contract SHA `411facd...`, while the current tracked file is `ed1a1d...`; four authorization tests fail before preflight assertions. Slurm files/jobs were not edited, polled, submitted, or cancelled. | four `test_phase3_execution_authorization_cpu.py` failures. | Re-materialize authorization through its owner workflow; do not hand-edit hashes in this task. |
| BASE-03 | environment-only | Full-suite collection lacks optional `nbformat`; notebook-only audit tests cannot collect. | `test_audit_integrity_cpu.py`, `test_revised_notebooks_cpu.py`. | Run notebook tests in the declared notebook-extra environment. |

## Verification summary

- Post-audit synthetic/static boundaries: 33 focused tests and 59 extended
  rollout/checkpoint/report tests passed. No real smoke was rerun.
- Current fail-closed shape/dtype/PID/closure slice: 40 focused tests passed in
  6.62 s and 90 safe-broad synthetic tests passed in 10.08 s using the frozen
  environment with current-source `PYTHONPATH`. No real data or science ran.
- The prior real smoke is cryptographically stale. Promotion requires a new
  smoke generated from a clean commit with the recorded Track A/B rollout
  policy hashes.
- Focused evaluator/model/data/rollout suite: 77 passed after the final policy
  parity and output-safety fixes.
- Final alias/preflight/provenance focused boundary: 59 passed in 5.15 s.
- Final repair SHA-256 bindings: evaluator script `1096c49d...93862`, trained
  context `dbaffb30...1a3f`, shared data module `51d8b60f...db26`, central index
  metadata loader `1a404ebe...0450`, dataset-index selection tests
  `8b1c849b...4adb`, trained-context tests `b28eeda0...2691`, and evaluator CLI
  tests `ebaa6980...f0be`.
- Current explicit-source broad suite, excluding only the documented notebook
  dependency, BASE-01's single failing case, and BASE-02's file: 842 passed,
  8 skipped, and 1 intentionally deselected in 101.50 s. The deselected node is
  `tests/test_channel_cross_event_cpu.py::test_empty_zero_capacity_channel_memory_can_expand_on_explicit_resume`.
- The earlier result of 813 passed, 8 skipped, and 1 deselected is historical
  and superseded. Its preceding unfiltered non-notebook run had 812 passed and
  six failures; it must not be quoted as the current validation boundary.
- `git diff --check`, strict JSON validation, and the final real one-event smoke
  are required immediately before handoff completion.

The separate checkpoint-selection-v4 materialization worktree and all Slurm
job state were intentionally left untouched.
