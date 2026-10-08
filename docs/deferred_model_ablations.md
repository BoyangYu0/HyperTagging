# Model ablations and deferred designs

## Structural membership development priority

The [2026-10-07 assembly diagnosis](assembly_diagnosis_20261007.md) executes a
frozen-encoder direct-membership feasibility pilot. The experimental
`models/direct_membership.py` head is consumed by
`scripts/probe_direct_membership.py`; it is not a production architecture preset.
It fits a tiny training subset but has no observed development exact-membership
gain. The next bounded development contrast should test encoder adaptation under
global membership supervision against the same frozen-head control. A full
hierarchy, reversible assembly and candidate-specific reranker remain conditional
future designs, not implemented production capabilities or selected winners.

The companion `configs/reconstruction/capacity_pretraining_development_plan.json`
specifies a 128/256 contextual-width by existing/assembly-targeted pretraining
comparison, with hyperbolic width 32 and fixed depth. It is a scientific plan,
not an executable training preset. Global unordered membership, relation-specific
within-B discrimination and detector/generated-state supervision are the target
objectives; existing hierarchy losses already provide some related supervision.
Joint decoder use, comparable pretraining histories, head bottleneck controls and
measured compute are required. Wider hyperbolic embeddings or 512-wide context
follow only reviewed downstream gains. The frozen-head pilot cannot establish
that the current encoder has adequate capacity or has discarded information.

## Current architecture boundary

The production model is not mixture-of-experts. Heterogeneous adapters feed a
shared contextual encoder and shared task geometry. `learned_euclidean` remains
the stable level baseline.

The optional `bounded_tangent_level_embedding` is a learned tangent-space
embedding with bounded, ordered radial prototypes; it is not a literal
hyperbolic positional encoding. The former `hyperbolic_tangent` name was
removed because expmap-at-origin followed by logmap-at-origin was not a
meaningfully distinct geometric mechanism.

## First-level ambiguity ablations

`first_level_type_relation_bias` is wired end to end and disabled by default.
It adds a soft query-to-node compatibility score to pointer logits using the
predicted mother-type distribution, candidate PID distribution, charge, kind,
level, physical relation summaries, and hyperbolic compatibility. Its config,
checkpoint architecture contract, logits, and gradients are tested.

The former `first_level_whole_set_scorer.yaml` and
`first_level_iterative_pointer.yaml` were removed. Their helpers were not
consumed by model logits, loss, or proposal ranking, so runnable configs would
have been decorative. They remain deferred designs:

- a whole-set scorer requires a coherent differentiable training target and a
  specified role in bounded proposal ranking;
- iterative pointer decoding requires a specified differentiable training
  contract and deterministic interaction with exclusivity constraints.

Neither field is accepted silently by the CLI.

## Structural relation-input compatibility ablation

Exact truth parent/ancestor indicators are loss targets by default and do not
enter FSP-only or truth-guided multilevel contextual attention. The explicit
`truth_guided_structural_relation_inputs` pretraining switch restores exact
known-link inputs only as a compatibility ablation. Validation reports that
view separately from FSP-only and target-only multilevel geometry; fixture
results cannot select it for production.

## Future MoE study

A later matched held-out ablation may compare level/topology-specialist FFNs or
query experts with explicit load balancing. No expert may be an exclusive full
decay channel. Utilization, losses, rollout quality, rare channels, and compute
cost must be measured before any production decision.
