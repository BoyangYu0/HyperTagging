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

## Future optional MoE study

A later held-out HTCondor ablation may retain the shared heterogeneous encoder
and compare level/topology-specialist feed-forward experts. A hyperbolic
prototype router or a learned router would select/weight experts, with an
explicit load-balancing loss. The present non-MoE network must remain the
matched-split baseline. Router load, expert utilization, next-level loss,
teacher-forced metrics, free rollout, rare channels, and compute/memory cost
must all be reported before any production decision.
