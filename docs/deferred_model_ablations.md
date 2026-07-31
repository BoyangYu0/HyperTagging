# Deferred model ablations

## Current architecture boundary

The current production model is **not** a mixture-of-experts model. Tracks,
ECL clusters, and composites have heterogeneous input adapters, then share one
contextual encoder, task projections, and hyperbolic latent space. This
correctness patch does not add MoE routing or claim an MoE result.

Level information is also not a literal predefined hyperbolic positional
encoding. The encoder adds a learned Euclidean level embedding before
contextualization; the hyperbolic objective separately supervises radius to be
monotonic with retained-tree level.

The explicit `hyperbolic_level_encoding` ablation has three settings:
`learned_euclidean` (the unchanged default), `hyperbolic_tangent` (learned
hyperbolic level points mapped into the origin tangent space), and `none`.
Promotion requires matched held-out measurements.

## Future optional MoE study

A later held-out HTCondor ablation may retain the shared base encoder and
compare level/topology-specialist reconstruction FFNs or mother-query experts.
A hyperbolic-prototype router or learned router would select/weight experts,
with an explicit load-balancing loss. No expert may be defined by a complete
exclusive decay channel. The present non-MoE network must remain the
matched-split baseline. Router load, expert utilization, next-level loss,
teacher-forced metrics, free rollout, rare channels, and compute/memory cost
must all be reported before any production decision.
