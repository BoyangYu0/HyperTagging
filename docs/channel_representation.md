# Schema-v4 channel representation

The production `direct-mdst-tree-v4` event record keeps channel topology,
reconstructed kinematics, and event membership separate. Channel labels are
truth-guided supervision and diagnostics; they are not model input features at
inference.

## Current v4 contract

A retained node's canonical signature is the recursively sorted pair of its
reduced PID token and daughter signatures after the verified contraction step.
Node IDs, copy IDs, reconstructed object IDs, and four-vectors are excluded.
Canonical compact JSON is assigned a stable 60-bit SHA-256-derived ID; zero
means unavailable. Optional charge-conjugate normalization selects the
lexicographically smaller full recursive signature.

V4 stores separate full-retained and reconstructable-retained signatures and
IDs for both B branches, plus their sorted permutation-invariant Upsilon pair.
B-side ordering is deterministic and has no tag/signal meaning. Upsilon(4S)
discovery requires exactly two direct retained B0/B+ daughters; B_s is excluded
by default and any fallback is flagged.

Structured summaries contain dense reduced-PID counts, depth-by-PID and
relative-depth counts, selected intermediate counts, canonical branch-
multiplicity histograms, node count, and maximum relative depth. The sorted
`branch_multiplicities` list remains serialized for compatibility, but
similarity uses `(multiplicity, count)` records and never traversal or list
position.
`structured_channel_similarity` is a symmetric weighted Jaccard score over
these structures. It is a graded metric-learning target, not a replacement for
exact IDs.

These labels must not be conflated:

- `same_event`: branches originate in one event only;
- B-side membership: a node-to-branch relation;
- exact full/reconstructable equality: canonical signature equality;
- structured similarity: graded count/topology similarity;
- Upsilon pair ID: the unordered pair of B signatures.

The two B mesons in one event are never positive merely because they share the
event. Channel training pools shared contextual node embeddings within each
truth-guided B branch and forms pairs across minibatch branches. The optional
checkpointed ring buffer extends comparisons across minibatches without
copying the full bank at every step.

## Pooling ablations

The baseline is mean pooling over all retained branch nodes. B-root-only,
final-state-only, learned-attention, and level-weighted alternatives are in
`configs/ablations/channel_pool_*.yaml`. Fixtures validate mechanics only;
matched held-out comparison remains external.

## Historical compatibility

No exact historical dictionary-array implementation was recoverable from this
repository, so v4 makes no legacy-equivalence claim. Schema-v2/v3 evolution,
compatibility boundaries, and the earlier fallback design are preserved in
[schema migration history](schema_migration_history.md). Historical audit
claims are indexed under [docs/audits](audits/README.md).
