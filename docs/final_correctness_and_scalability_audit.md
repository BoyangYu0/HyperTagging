# Final Correctness and Scalability Audit

Audit date: 2026-07-30  
Audited branch: `master`  
Audited revision: `9d37fde7df7cd59eda1d3464ad5351f224314d6c`
(`fix preprocessing`)

## Baseline

The worktree was clean before this revision. `origin/HEAD` points to `master`,
and the inspected checkout is the current local default branch.

The literal requested command, `python -m pytest -q`, selects
`/usr/bin/python` 3.9 in this environment and fails before collection because
that interpreter does not have pytest installed. The repository CPU
environment is therefore used for the meaningful baseline:

```text
/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q
```

The baseline completed successfully:

```text
151 passed, 8 skipped, 15 warnings in 375.67s
```

The skips are historical-equivalence cases whose external legacy sources are
not installed. The warnings are the existing odd-attention-head PyTorch
nested-tensor warnings.

## Preserved corrections and scientific invariants

The audit confirms that the current code already enforces these properties,
which this revision must preserve:

1. Generic-mDST raw-track fit selection and reconstructed charge do not use
   related MC PID or charge.
2. Canonical raw-track input energy uses a data-independent pion hypothesis,
   while e/mu/pi/K/p hypotheses and PIDLikelihood availability are stored
   separately.
3. Truth PID/charge and MC four-vectors are target or diagnostic fields only.
4. Model PID/type tensors use the 41-token `PDG_TOKENS` vocabulary, with strict
   range validation.
5. Track, ECL-cluster, and composite inputs use heterogeneous adapters in one
   shared contextual and hyperbolic space.
6. Physical event contextualization precedes the principal hyperbolic
   projection.
7. Hyperbolic parent ranking uses Poincare distance; tree distance,
   leaves-outside radius, tangent variance, and tangent covariance objectives
   are present.
8. Cross-event B-branch supervision, target-level queries, weighted focal
   pointer/object losses, supervised confidence, Hungarian matching, and
   recursive leaf-source masks are present.
9. Reconstructed composite four-momenta are recursive daughter sums; MC mother
   four-momenta are never reconstruction targets.
10. V1, v2, and v3 remain loadable; production configuration and encoder-only
    transfer are already represented in checkpoints/manifests.
11. CPU pilots are local; full preprocessing/training remains HTCondor-only and
    no job is submitted automatically.

## Confirmed remaining issues

### Schema and truth cleanliness

- `export_trees_v3` buffers every event and writes `ak.Array([payload])`, so a
  shard is one top-level parquet row containing the complete event list.
- `_node_record_v3` constructs `daughter_pid_histogram` from daughter
  `pid_target_token`; that target-derived histogram is then consumed by
  `CompositeNodeEncoder`.
- Leaf kinematics modes are strings only and raw/fixed/cluster/composite modes
  are not carried as an explicit tensor.
- V1/v2/v3 legacy-conflated inputs are not rejected by real trainers.

### Reconstruction flow

- The leaf PID head is evaluated after the encoder and pointer decoder.
  Therefore Level-1 relations and pointers still see canonical p4 and stored
  input PID rather than predicted leaf PID.
- The decoder exposes allowed-type and pointer-validity masks, but the
  reconstructor does not provide them.
- Recursive source masks are used for final proposal competition, but not for
  within-proposal constrained decoding or a differentiable overlap penalty.
- Cardinality top-k decoding has no minimum pointer-probability requirement.
- Confidence targets multiply pointer overlap by the model's detached target
  type probability, making the target self-referential.

### Data and training scalability

- Production JSONL `output_file` records are not recognized by
  `resolve_data_paths`.
- Real data loading eagerly materializes all events and all split lists.
- Normalizer fitting concatenates every training node.
- Both trainers materialize `list(data_module.batches(...))`.
- Tiny-pilot split repair is unconditional when the training split is empty.
- Scheduled sampling is validation scaffolding rather than an optimization
  context.
- Resume does not restore AMP scaler or random states and does not validate
  schema/feature/PID/split contracts.
- CLIs parse `--config` but do not apply YAML values.

### Curriculum, targets, and metrics

- Multilevel curriculum batches do not apply a level-causal attention mask.
- Stage-1 radius normalization can depend on only currently visible nodes
  instead of original retained-tree depth.
- Composite corruption changes adjacency/type without rebuilding every derived
  p4, charge, PID-histogram, source-mask, and structural feature.
- Corruption codes and hard-negative pairs are produced but do not materially
  contribute to training objectives.
- Partial/completeness fields are stored in v3 but not propagated through
  `HeterogeneousEvent`, collation, capacity, loss, and evaluation under one
  target policy.
- Canonical metric alignment contains mother type/source identity, making
  post-alignment type and leaf metrics partly tautological.
- Reconstruction validation samples only the first event rather than
  aggregating a configurable validation subset.

### Production and inspection

- New-production defaults still select v3.
- The worker does not apply every scientific option recorded by the manifest.
- Sidecar statistics are not used to avoid a production-scale prescan.
- Notebooks do not yet expose v4 event-row streaming, runtime composite PID
  state, actual scheduled optimization context, or non-tautological alignment.

## Compatibility strategy

1. Keep v1/v2/v3 bytes and loader semantics unchanged.
2. Add `direct-mdst-tree-v4` as a new event-row schema and mark legacy
   conflation explicitly when adapting older data.
3. Preserve the ambiguous v3 histogram only as a named legacy compatibility
   alias. Native v4 uses separate daughter input and truth histograms.
4. Default real training rejects legacy-conflated events; an explicit
   diagnostic-only override is recorded in logs and checkpoints.
5. Centralize runtime PID distributions and daughter histograms so teacher
   forcing, scheduled sampling, and rollout use identical input rules.
6. Preserve all verified preprocessing tests before changing tree or basf2
   code. New truth-invariance and p4-closure tests guard any shared path.
7. Preserve CLI compatibility while moving trainers to restartable iterators,
   online train-only statistics, and explicit pilot split repair.

## Planned implementation stages

1. Add regression tests for composite PID input separation, two-pass leaf PID,
   production manifest resolution, streaming/normalization, scheduled context,
   curriculum consistency, partial policies, decoder validity, resume/config,
   and non-tautological metrics.
2. Add schema-v4, bounded atomic event-row writing, metadata sidecars, legacy
   gate, and streaming event iteration.
3. Add runtime PID state and a two-pass detector/PID then reconstruction
   contextualization path.
4. Propagate partial policies and source/type/charge/cardinality constraints
   into targets, losses, decoding, capacity, and evaluation.
5. Replace eager training flow with streaming batches, online Welford
   normalization, real scheduled-context optimization, and exact resume.
6. Correct curriculum masks/corruptions/objectives and tree alignment.
7. Apply v4 production settings, update deterministic notebooks/CI/docs, and
   run the complete CPU verification surface.

## Acceptance tests and verification boundary

The focused acceptance suite will prove truth-PID invariance of composite
inputs, pointer gradients into leaf PID, streamable event rows, bounded
iteration, online-normalizer equivalence, production-manifest integration,
scheduled-context use, causal curriculum masks, internally consistent
corruptions, target-policy consistency, constrained decoding, confidence-target
independence, exact resume, YAML precedence, and type-independent structural
alignment.

CPU fixtures cannot establish PID calibration, physics efficiency, scientific
improvement, or ten-million-event throughput. No large preprocessing, long
training, unguarded CUDA, or Condor submission is part of this revision.

## Implemented result

The planned stages were completed without changing the verified v1/v2/v3
formats. Native v4 is one event per row, atomically published by a bounded
writer, and carries explicit input/truth PID histograms and leaf-mode IDs. The
model boundary rejects target-derived composite inputs. Runtime soft PID state
is shared by teacher forcing, scheduled contexts, and rollout.

Level-1 reconstruction now uses detector-context PID inference before rebuilding
PID embeddings, track energy, p4, relation features, and reconstruction context.
The pointer gradient therefore reaches the leaf PID head. Streaming iteration,
bounded shuffle, online masked Welford normalization, production `output_file`
manifest loading, exact state restoration, YAML precedence, partial-target
policy, constrained pointers, corrected confidence targets, causal curriculum,
and source/topology-first evaluation are implemented and covered by CPU tests.

## Final verification

The final repository-environment test command completed:

```text
173 passed, 8 skipped, 18 warnings in 343.55s
```

The standalone notebook smoke command executed all 12 deterministic CPU
notebooks and produced the required JSON, CSV, figure, and checkpoint
artifacts under `/tmp/hypertagging-notebook-smoke`.

The two-step real v4 CPU pilots completed:

```text
pretraining: loss 4.628071308135986
reconstruction: loss 2.1927132606506348
encoder transfer: 88 keys loaded, no missing/unexpected/shape-mismatched keys
```

The HTCondor renderer completed in dry-run mode and no job was submitted.
`basf2` is not available in this environment, so no real mDST pilot was run.
The scientific verification boundary remains unchanged: fixtures establish
software correctness, not PID calibration, physics performance, or 10M-event
throughput.
