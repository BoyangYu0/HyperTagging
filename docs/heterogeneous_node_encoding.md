# Heterogeneous node encoding

`direct-mdst-tree-v2` gives every node an explicit kind: `track`,
`ecl_cluster`, `composite`, `unknown`, or `other`. Every stored numeric block
has a parallel value-level availability mask. A stored zero is only a
tensor-safe placeholder and never means that a detector quantity was measured.

## Common block

The shared block has identical semantics for every node:

`px`, `py`, `pz`, `energy`, invariant `mass`, `charge`, `reduced_pid`,
`level`, `active`, `copied`, `n_daughters`, and `candidate_confidence`.

Track-specific fields currently supported when returned by basf2 are fit
p-value, d0, z0, phi0, omega, and tan-lambda. ECL-specific fields include the
cluster energy and p4-derived direction, plus available timing, E9/E21,
crystal count, minimum track distance, photon-hypothesis, and track-match
state. Unsupported accessors stay unavailable.

Composites contain daughter-summed p4/charge, daughter count, pointer
confidence statistics when predicted, copied-daughter fraction, and a
reduced-PID daughter histogram. They never claim track or cluster fields.

## Architecture

`CommonNodeEncoder`, `TrackNodeEncoder`, `ClusterNodeEncoder`, and
`CompositeNodeEncoder` all return `d_model`. Values are cleaned only after
their masks are applied. The shared initial representation sums:

- the common projection;
- the selected type projection;
- reduced-PID, node-kind, and level embeddings;
- an explicit availability projection.

It then applies one shared normalization and MLP. A single shared encoder feeds
tree, reconstruction, and channel projections. The tree projection is mapped
to one Poincare ball; tracks, clusters, and composites do not get unrelated
geometries.

For a composite, masked mean pooling over daughter embeddings is
permutation-invariant. `composite_token_from_daughters` is the common
construction routine used by truth-guided and predicted paths. During every
rollout step, all current nodes are re-encoded, so appended composites receive
the same pooled-daughter treatment as teacher-forced composites.

## V1 adaptation

A v1 leaf is a track or ECL cluster only when its explicit `reco_id` prefix
proves that provenance. Other reconstructed leaves are `other`; absent
provenance is `unknown`. V1 detector blocks are entirely unavailable.
Reco-derived composite structure is recovered from existing daughter links and
stored p4 without inventing detector measurements. All original scalar values,
IDs, links, flags, and diagnostic MC fields remain unchanged.
# Correctness revision

Schema-v3 track blocks add verified PIDLikelihood log-likelihoods and
e/mu/pi/K/p energy hypotheses with per-value availability. These are
data-compatible measurements/derivations; truth PID is a separate target.
Adapters now enter one physical relation-aware contextualizer before task and
hyperbolic projections. Optional hyperbolic relation refinement is downstream,
avoiding circular dependence.
