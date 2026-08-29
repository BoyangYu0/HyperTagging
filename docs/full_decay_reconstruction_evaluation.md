# Offline full-decay reconstruction evaluation

This document defines model evaluation for hierarchical reconstruction from
preprocessed detector final-state particles (FSPs). It is not a basf2 decay
candidate builder and it does not consume the historical GraFEI `pairs`
format. The input is the same `direct-mdst-tree-v4` Parquet data and immutable
selection/index contract used by the current pretraining and reconstruction
jobs.

The central information boundary is:

```text
schema-v4 event
  |-- detector FSP projection: track, ecl_cluster, klm_cluster --> model
  `-- complete retained truth tree and labels -----------------> metrics only
```

No stored composite, MC mother, truth parent, truth level, B-side label, or
truth PID target may enter reconstruction. Higher-level particles are useful
only after inference, when the predicted forest is compared with the retained
truth forest.

## Why the historical full reconstruction is not the evaluator

`src/hypertagging/reconstruction/full_reconstruction.py` preserves the
behavior of the old `graFEI/whole_eva.py` workflow as a compatibility fixture.
For every level it:

1. reads a padded GraFEI `pairs` array;
2. asks a generator for a fixed set of mother PDG classes and mother features;
3. asks a separate linker to assign every daughter to a mother slot;
4. removes empty slots, remaps their link indices, and sums daughter features;
5. follows the slot-index ancestor chains to form an LCA matrix; and
6. repeats until the historical hard-coded root class/token `13`, one object,
   or a step limit is reached.

Its reported PDG accuracy is padded slot accuracy, its feature error depends on
the slot ordering, and `perfect` compares raw LCA arrays in that ordering.
Consequently padded zeros can affect the denominator and two equivalent
unordered trees can compare differently. It also predates current detector
node kinds, recursive reconstructed-source conflicts, leaf-PID inference,
checkpoint feature contracts, and the 41-token reduced-PDG vocabulary.

That implementation and `scripts/evaluate_reconstruction.py` should remain
available for historical regression tests. They must not be used to claim
performance for the current level-autoregressive model.

## Current inference algorithm

The current model factorizes an unordered tree by generation level,

```text
p(tree | FSPs) = product_t p(new mothers at level t + 1 | current forest)
```

and uses one `LevelAutoregressiveReconstructor` with a shared transferred
encoder and learned query slots. Each query predicts object/no-object, mother
type, daughter pointers, cardinality, and confidence. The reconstruction
checkpoint already contains the transferred pretraining encoder; the separate
pretraining checkpoint is used to prove lineage, not as a second sequential
network in the forward pass.

Evaluation proceeds as follows.

1. Load the reconstruction checkpoint, its serialized architecture,
   normalizers, PID vocabulary, constraint policy, and rollout PID mode on
   CPU. Validate the supplied pretraining checkpoint against the transferred
   encoder before processing events.
2. Load a deterministic validation or test split through the schema-v4
   selection manifest and dataset index. Retain an untouched truth batch for
   metrics.
3. Build a physically compact inference batch containing active level-zero
   `track`, `ecl_cluster`, and `klm_cluster` nodes only. Reset generated node
   state, parents, adjacency, and levels. Remove truth composites rather than
   merely padding or masking them. Scrub truth PID targets, B-side labels,
   truth geometry, truth source ancestry, and all other target-only fields.
   Keep only reconstructed provenance needed to detect shared detector
   sources. An evaluation-only map relates compact leaf keys back to the truth
   batch; it is never supplied to the model.
4. Predict leaf PID from detector-compatible features. For raw tracks, update
   energy and p4 according to the checkpoint's recorded rollout PID-kinematics
   mode. The current `soft_decision_hard_construction` contract uses soft
   expectation inside model relations/decisions and the hard PID choice only
   when constructing persistent leaf p4. Fixed-hypothesis candidates and
   cluster measurements keep their detector-defined kinematics.
5. Encode the current forest and decode the next level. A daughter is eligible
   only if it is active, belongs to the current event, existed before this
   decoding step, and has no parent. In other words, only current forest roots
   can become daughters. Apply the checkpoint's cardinality, type, charge,
   physical, confidence, and recursive-source exclusivity constraints. The
   serialized allowed-type mask and empirical soft type bias are injected
   before the mother decoder, as in training, so they also condition expected
   type embeddings and daughter-pointer logits rather than merely filtering a
   post-hoc argmax.
6. Resolve competing accepted queries without sharing a forest root or a
   recursive detector source. Append each accepted mother as a new composite,
   set the selected roots' parent links, and construct

   ```text
   p4(mother) = sum_daughters p4(daughter)
   charge(mother) = sum_daughters charge(daughter).
   ```

   Recursive source masks, current daughter-PID summaries, daughter count, and
   confidence summaries are likewise constructed only from predicted/current
   daughters. No mother p4 is copied from truth or independently regressed.
7. Re-encode the enlarged forest and repeat. In offline evaluation, advance
   the target-level embedding across an empty proposal level: stored schedules
   can contain an empty eligible generation even though a valid direct target
   exists later. Count these empty levels explicitly. Full scope stops on the
   requested root or the configured maximum; half scope scans through the
   maximum so both B candidates remain available. The lower-level rollout API
   retains its historical stop-on-empty default for training callers.

Restricting pointers to parentless forest roots is essential. Merely requiring
a daughter to come from an earlier level would allow a node already consumed
by one mother to be consumed again at a later level, producing an inconsistent
adjacency/parent state and double-counting its detector source.

### Full-decay scope

For a B-pair event with an explicit retained `Upsilon(4S)` root, full scope has
one evaluation unit per event. In the current vocabulary the root is token 1
(PDG 300553). The rollout aims to construct that one declared root component
and may stop when it is accepted. Detector leaves outside the truth root's
retained component can remain as separate forest roots; they are reported as
leftover/contaminating coverage rather than silently attached to make the
forest literally single-rooted.

The primary full-decay denominator is the number of eligible events, not the
number of successfully decoded events. A failed rollout is therefore a failed
unit. Events without an explicit single retained truth root are reported as
`unavailable` with a reason and are not silently assigned a guessed root.

### Half-decay scope

For an eligible B-pair event, half scope has exactly two evaluation units: the
two truth components rooted at `B`/`anti-B`. The model is not given `b_side`,
`b1_root_id`, or `b2_root_id`. It must be allowed to produce both B candidates;
the event must not stop merely because the first B token was accepted.

After inference, predicted B-rooted subtrees are matched as an unordered pair
to the two truth halves using reconstructed leaf-source/topology overlap. PID
and p4 are excluded from this matching cost, so neither quantity can improve
its own evaluation alignment. Missing and extra B candidates remain visible
as unmatched truth and prediction counts.

The output contains two rows per eligible event, each with multiplicity one.
Thus half-level micro summaries have denominator `2 * N_events`. A separate
event-level `both_halves_exact` indicator is true only when both matched halves
have perfect keyed-LCA topology; its denominator is `N_events`. Reporting only
an average event score is insufficient because it can hide one successful and
one failed half.

### Continuum component scope

Native schema-v4 continuum events have `b_side = -1`; they do not define two
hemispheres. Half-scope evaluation therefore uses each explicit top-level
truth composite in the retained continuum forest as one multiplicity-one
`continuum_component` unit. Predicted top-level components are assigned
one-to-one by FSP-source overlap, without PID or p4 in the cost. An event with
three retained top-level continuum particles contributes three rows, not two.

The B-specific `both_halves_exact` statistic has a zero denominator for these
events. The evaluator never manufactures a two-side conjunction from MC
ancestry, the two highest-scoring predictions, or a post-hoc truth partition.
If no explicit truth composite root exists, component scope is explicitly
unavailable. A future versioned truth-only hemisphere label could define an
additional two-side metric, but it would be a different target contract.

### Direct-target representability

The primary `checkpoint_direct` topology is the original retained direct
daughter graph used to form the checkpoint's pointer targets. Policy-ineligible
or unary intermediate composites are not valid emitted mothers. When an
otherwise eligible root depends on such an intermediate, the unit remains in
the primary topology denominator with `target_representable = false` and
`perfectLCAG = false`; removing it as unavailable would bias the result toward
easy target shapes. Recursive detector-source reuse inside a truth unit is
handled the same way.

`contracted_diagnostic` suppresses policy-ineligible intermediates and links
their first retained frontier. It is useful for diagnosing a possible future
training target, but it is not the pointer topology used by the current
checkpoint and must never replace the primary score. PID and p4 denominators
include only policy-eligible truth mothers in either mode.

## Metric contract

Every aggregate is published with its numerator, denominator, value, and
eligibility/failure counts. Decode failures stay in topology and root-success
denominators. Missing truth labels are `unavailable`, not successes and not
decode failures. Results should also be stratified by scope, source category,
target FSP multiplicity, and retained depth.

### Source-keyed perfect LCAG

Tensor positions and generated composite IDs are not stable tree identities.
Each detector leaf is instead keyed by stable reconstructed-source provenance.
For every unordered pair of target leaves, the LCA is represented canonically
by the recursive set of leaf keys below that ancestor (and, where needed, its
generation). This makes the comparison invariant to leaf order, query order,
and generated mother IDs.

`perfectLCAG` is true exactly when:

- the predicted and target leaf-key sets for the evaluation unit are equal;
- every keyed leaf pair has the same canonical LCA entry; and
- the predicted component is structurally valid.

Mother and leaf PID are deliberately excluded. They have separate metrics.
The diagonal carries no topology information and is not counted. For fewer
than two leaves, eligibility and the adopted trivial-tree convention must be
reported explicitly rather than inflating the aggregate.

This replaces the historical comparison of position-indexed LCA arrays.

### Missing-one-particle topology

Two related values should be reported:

- `strict_missing_one`: the prediction has no extra leaves, is missing exactly
  one target leaf, and its keyed LCAG equals the truth LCAG induced on all
  remaining leaves;
- `leave_one_out_lcag`: there exists exactly one target leaf whose removal
  makes the remaining predicted and truth keyed LCAGs exact.

The removed source key should be recorded for diagnostics. A unit is not
eligible for the leave-one-out rate unless at least three leaves remain after
removal; otherwise almost any tiny tree would pass. Publish the mutually
exclusive counts `perfect`, `strict_missing_one`, and `other`, as well as the
leave-one-out numerator and its own eligibility denominator. PID correctness
of the retained leaves is not part of this topology metric.

### PID accuracy

PID is evaluated only after source/topology alignment.

- Leaf PID accuracy compares the hard current PID token inferred from detector
  inputs with the truth PID target for evaluable FSPs. Report correct, total,
  unavailable/unknown, and per-token confusion counts.
- Mother PID accuracy compares the predicted mother token with a truth mother
  aligned by its recursive leaf-source clade/topology. The alignment excludes
  the type token. Report correct over matched mothers, matched-mother coverage
  over truth mothers, and unmatched predicted/truth counts.
- Root PID success is reported separately for full and half roots. It is not a
  substitute for the topology metrics.

This separation prevents a missing mother from disappearing from a conditional
PID accuracy and prevents the expected PID label from selecting its own match.

### Momentum and p4 error

Kinematics are compared only for nodes aligned by source/topology, never by
nearest p4. For matched nodes report at least:

- signed bias, MAE, and RMSE for `px`, `py`, `pz`, and `E`;
- mean/RMSE of `|Delta p|` and a relative value where the reference momentum
  is nonzero;
- invariant-mass bias/MAE, with the numerical mass convention stated; and
- the matched-node numerator, truth-node denominator, and coverage.

Unmatched nodes are represented by coverage/topology failures and are not
assigned an artificial infinite p4 error. Relative quantities exclude a zero
reference norm and publish that reduced denominator.

Also report daughter-sum closure separately:

```text
closure = p4(predicted mother) - sum p4(predicted daughters).
```

Closure tests implementation validity and should be near floating-point zero
by construction. It is not a physics-resolution metric. Schema-v4 retained
mother p4 is also derived from reconstructed daughter inputs, not generator MC
p4. Consequently, for exactly source/topology-aligned mothers, `px`, `py`, and
`pz` agreement is normally tautological daughter-sum consistency; energy and
mass can additionally reflect the inferred leaf-PID mass hypothesis. The
report labels this reference `reconstructed_fsp_daughter_sum` and sets
`physical_momentum_error_available = false`. A physical resolution metric
requires a future evaluation-only MC-composite p4 field that is excluded by
the FSP projection and never passed to the model.

### Validity, coverage, and stopping

At minimum publish:

- acyclic and single-parent fractions;
- forest-root-only pointer and recursive-source-disjoint fractions;
- constraint-valid and exact daughter-sum-closure fractions;
- target leaf-source recall, predicted-source precision, leftover input FSPs,
  and duplicate-source counts;
- matched composite coverage and unmatched composite counts;
- requested-root reconstruction success and predicted root multiplicity;
- rollout stop-reason counts, levels completed, and failed-event count; and
- full-event and both-half conjunctions separately from conditional component
  scores.

A threshold scan may supplement the fixed checkpoint policy, but the primary
result must identify one predeclared policy. Selecting a threshold on the test
sample and quoting the same sample as final performance is not valid.

## CPU-only execution

The scripts hide CUDA before importing PyTorch, load checkpoints with
`map_location="cpu"`, put the model in `eval()` mode, and run under
`torch.inference_mode()`. Keep data-loader workers at zero and bound BLAS/OpenMP
threads so evaluation does not compete with an active training job. If the
current node is busy, run the same read-only command on an available CPU node
with access to the shared filesystem; do not attach to or reuse a training
GPU.

From the repository root, validate the currently selected frozen-transfer pair
first:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ./.venv/bin/python scripts/validate_reconstruction_checkpoint_pair.py \
  --pretraining-checkpoint \
  artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/15933802/checkpoint-step-54064.pt \
  --reconstruction-checkpoint \
  artifacts/runs/ht-reconstruction-transfer-fullscale-20260824/15979725/training/best_rollout_edge_f1.pt
```

The receipt contains checkpoint paths, SHA-256 digests, steps, feature/PID
contract agreement, encoder key coverage, and exact tensor equality. Content
equality is the lineage evidence even if a serialized historical path no
longer resolves. `--allow-finetuned-encoder` is for a future intentionally
fine-tuned run and must be disclosed; it is not appropriate for this frozen
pair.

An offline validation invocation is:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ./.venv/bin/python scripts/evaluate_full_decay.py \
  --pretraining-checkpoint \
  artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/15933802/checkpoint-step-54064.pt \
  --reconstruction-checkpoint \
  artifacts/runs/ht-reconstruction-transfer-fullscale-20260824/15979725/training/best_rollout_edge_f1.pt \
  --data configs/training_selection/production_1m_20260812/train_035k.json \
  --dataset-index \
  artifacts/experiment_readiness/production_1m_20260812/train_035k/train_035k.complete_only.index.json \
  --split validation \
  --scope both \
  --max-events 100 \
  --threads 4 \
  --output artifacts/evaluation/full_decay_validation.json
```

`--data` is the schema-v4 training-selection manifest bound to the promoted
dataset index, not a raw Parquet list, a basf2 mDST file, or a GraFEI `.npy`
file. A source-role index cannot safely be replayed by rehashing raw paths; the
loader rejects that combination. The manifest and index together fix source
roles and deterministic splits. For validation, the default
`--event-selection auto` restores the ordered checkpoint rollout UID cohort
and the checkpoint's learned-confidence policy; `stream` is explicitly
diagnostic. The currently promoted index contains no `test` role, so test
evaluation requires a separately materialized, immutable test-bound index
rather than relabeling validation. Use `test` only after the policy and
thresholds have been fixed on validation. Small smoke
runs should begin with `--max-events 1`. Use repeatable exact filters such as
`--source-category mixed` or `--source-category ccbar` when validating B-pair
and continuum behavior separately; filtering happens after immutable split
assignment and does not change dataset membership.

Before loading any checkpoint or data, the CLI resolves `--output` and rejects
path, symlink, or hardlink aliases of the checkpoints, index, selection
manifest, and every manifest-referenced shard. Existing independent report
files may still be replaced atomically.

## Interpretation limits

- The target is the retained reduced-PDG tree, not every generator particle in
  the original decay record. Primary scoring uses its direct topology;
  contraction is diagnostic only.
- Evaluation is conditional on detector objects exported to schema-v4. A
  physically missing track or cluster cannot be recovered by this tree model.
- `complete_only` selection changes the target population and must be included
  in every result receipt.
- A reconstructed ECL/KLM association can intentionally share a source key;
  the source-conflict policy prevents double use without consulting truth.
- Greedy exclusivity, confidence thresholds, maximum level, and the reduced
  mother ontology bound what can be reconstructed. These settings belong in
  the output metadata.
- Symmetric halves can yield an equivalent unordered alignment. Deterministic
  tie handling should be recorded, but B1/B2 naming itself is not a physics
  error.
- Continuum roots are multiplicity-weighted components. Current schema-v4 data
  cannot support a two-hemisphere or both-continuum-halves claim.
- The evaluator's forest-root and empty-level policy is newer than the current
  reconstruction checkpoint. Historical stored rollout metrics must be
  re-evaluated with the versioned offline policy before comparison.
- Passing CPU unit tests establishes implementation invariants, not scientific
  model quality. Physics claims require a locked held-out dataset, immutable
  checkpoint receipts, and reported statistical uncertainty.
- The output is an evaluation artifact. It does not create basf2 Particles,
  write an mDST, or replace experiment reconstruction software.
