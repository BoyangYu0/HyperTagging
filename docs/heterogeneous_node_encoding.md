# Schema-v4 heterogeneous node encoding

Production schema `direct-mdst-tree-v4` assigns every node an explicit kind:
`track`, `ecl_cluster`, `composite`, `unknown`, or `other`. Every numeric block
has value-level availability masks; stored zeros are tensor-safe placeholders,
never claims that a detector quantity was measured.

## Current input contract

The common block contains p4, invariant mass, reconstructed charge, current
input PID token, retained-tree level, active/copied flags, daughter count, and
candidate confidence. Track blocks include available helix/fit and
PIDLikelihood-derived quantities. ECL blocks include reconstructed energy,
direction, shower values, and matching state when available. Unsupported
accessors remain masked.

Composite persistent inputs are exact daughter-summed p4 and charge, daughter
count, prediction-confidence summaries, copied-daughter fraction, and the
model-input daughter PID histogram. Truth daughter-PID/channel/MC fields have
separate construction functions and source provenance. They never enter the
encoder, relation features, pointers, type logits, or leaf-PID inference.

## Architecture and composite lifecycle

Common, track, cluster, and composite adapters all return `d_model`. Masked
values are cleaned before projection. The representation combines common and
kind-specific projections, current PID, node kind, level representation, and
availability, then enters a shared relation-aware contextual encoder. Tree,
reconstruction, channel, and hyperbolic projections branch only afterward.

Append-time construction persists physical daughter sums only. The current
baseline is `precontext_daughter_pool`: every subsequent encoder pass
recomputes a permutation-invariant summary from the current adjacency and the
pre-context daughter representations, then builds composite inputs before
Stage-A contextual attention. This avoids circular dependence and target-level
leakage. Truth-guided and predicted states with identical links therefore
follow the same next-pass construction contract. A lower-level contextual
two-pass pool is not silently enabled; it remains a named future ablation.

`learned_euclidean` is the stable level baseline. The optional
`bounded_tangent_level_embedding` is accurately a tangent-space mechanism: it
uses learned directions and ordered bounded radii with leaves outside roots.
It is not described as literal hyperbolic positional encoding.

## Historical compatibility

V1/v2/v3 adapter semantics and schema evolution are preserved in
[schema migration history](schema_migration_history.md). Compatibility paths
remain testable but do not redefine the current v4 contract.
