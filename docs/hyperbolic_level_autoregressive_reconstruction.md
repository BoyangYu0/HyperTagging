# Hyperbolic-pretrained level-autoregressive set reconstruction

The event is an unordered hierarchy and is factorized as:

```text
p(tree | S0) = product_t p(S_{t+1} | S_{<=t})
```

Level 0 contains reconstructed final-state objects. Levels increase toward the
retained root and obey `L(mother) = 1 + max L(daughter)`.

## Kinematics and topology

MC truth may supervise retained topology, B-side membership, LCA relations,
mother type, and daughter links. It never supplies reconstructed mother
kinematics. At teacher forcing and inference:

```text
p4(mother) = sum p4(selected reconstructed daughters)
```

No head regresses an arbitrary mother p4. Composite p4, charge, daughter
histogram, confidence summaries, and pooled daughter embedding are all created
by the common composite-construction path.

## Shared encoder and task projections

Type-specific track, ECL, and composite adapters feed one `d_model` space and
one shared contextual set transformer. Small tree, reconstruction, and channel
projections specialize the shared representation. One shared tree projection
maps to the Poincare ball.

## Principal hyperbolic objective

The configurable base loss is:

```text
L = lambda_LCA L_LCA
  + lambda_parent L_parent
  + lambda_depth L_depth
  + lambda_channel L_channel
  + lambda_var L_var
  + lambda_cov L_cov
```

The exact LCA relation classes are:

1. same node;
2. same immediate retained mother;
3. same local branch (a retained common ancestor no more than two levels above
   the deeper node);
4. same explicit B branch;
5. different B branches in the same event;
6. unrelated or unavailable truth relation.

`balanced_tree_relation_loss` averages present class losses so unrelated pairs
cannot dominate. Hard parent negatives are the hyperbolically nearest valid
non-parent nodes.

Parent ranking uses the actual Poincare distance:

```text
relu(d_H(child,parent) - d_H(child,negative) + margin)
```

The stable distance implementation uses the mathematically equivalent `asinh`
form, avoiding the infinite derivative of `acosh` at coincident points.

## Radius convention

Leaves are farther from the origin and high-level composites are nearer:

```text
r_target = r_min + (r_max-r_min) * (L_max-L_i) / max(L_max,1)
```

`L_max` is event-specific. Tests explicitly protect the direction.

## Anti-collapse behavior

VICReg-style variance and off-diagonal covariance penalties act on
`logmap0(z)`. Sampling can be deterministically balanced/capped by level and
node kind across the batch. These losses are not pairwise sibling-repulsion
terms.

Logged diagnostics include per-dimension mean/minimum standard deviation,
off-diagonal covariance norm, singular-value effective rank, positive/negative
relation distance, radius-level correlation, angular B-branch separation, and
Poincare-boundary fraction.

## Channel objective

Shared channel projections are pooled over truth-guided B branches. Cosine
similarity is trained toward exact equality and the structured channel
similarity described in `docs/channel_representation.md`. The unordered event
embedding pools the two B embeddings. Full-channel classification is not
applied independently to every final-state token.

## Relation-aware contextual attention

Each contextual layer adds a learned pair bias to QK logits before softmax. Its
inputs include Poincare distance, both radii, level difference, same-level
state, pair mass/energy, summed charge, node-kind compatibility, and copied
source conflicts. Stair-causal and padding masks are applied in the same
softmax. Tests verify that changing only this bias changes outputs and that its
parameters receive finite gradients. `use_relation_bias=False` is the ablation.

## Decoder and full rollout

Learned query slots predict object/no-object, mother token, daughter pointers,
cardinality, and confidence. Hungarian matching makes target-mother ordering
irrelevant.

Each rollout step:

1. encodes all current nodes;
2. contextualizes them with stair-causal relation attention;
3. hard-decodes valid mother proposals;
4. optionally retains competing hypotheses;
5. resolves source-object overlap for an exclusive result;
6. sums selected daughter p4/charge;
7. appends composite tokens with distinct node/reco/source/copy provenance;
8. re-encodes all nodes.

Stopping conditions cover all-no-object, no valid mother, configured root
token, maximum level, repeated state, and invalid/cyclic links. Cycles are
prevented structurally because a new mother may point only to pre-existing
nodes.

Teacher forcing uses truth links but the same reco-derived construction.
Scheduled sampling is seeded and reproducible. Evaluation reports both
teacher-forced next-level metrics and free-rollout edge/tree metrics.

## Verification boundary

Tiny CPU tests verify formulas, masks, matching, p4 closure, finite gradients,
and termination. They do not establish scientific improvement. Real-size
training and evaluation remain HTCondor-only.

## Context-first geometry and curriculum

The corrected order is heterogeneous adapters → Stage-A physical relation
attention → contextual Euclidean nodes → task projections → shared Poincaré
projection. Stage A has no hyperbolic inputs. Optional Stage B then uses
Poincaré distance/radius features and is a separate ablation.

The principal loss is LCA classification, true Poincaré parent ranking, direct
normalized tree-distance regression, leaves-outside radius-depth regression,
cross-event B-branch supervised contrastive learning, and tangent-space VICReg
variance/covariance regularization. One curvature value is propagated through
maps, distance, radius, relation features, losses, and diagnostics.

The curriculum is explicit: FSP-only, truth-guided reconstructed composites,
then corrupted/predicted-like composites (missing/wrong daughters, wrong
types, shared-source conflicts, and physical hard negatives). Stage 1 masks
every truth composite from contextual input. Multilevel attention is
level-causal: level `l` can inspect only levels `<= l`. Radius targets retain
the original full-event depth even when only leaves are visible.

Queries receive a target-level embedding. Pointer scores include the expected
mother-type embedding. Truth query/cardinality overflow raises; sparse
object/pointer losses use configurable focal/positive weighting. Confidence is
trained on matched pointer Jaccard times a hard correct-type and structural
validity indicator; it never depends on the model's own type-confidence
magnitude. It reports Brier/ECE. Exclusive rollout compares recursive
leaf-source sets.

The production reconstructor uses two contextual passes. Pass A uses only
detector-compatible inputs and predicts charge-compatible leaf PID. Its soft
PID distribution rebuilds the PID embedding, track energy, p4, pair relations,
and composite input histograms. Pass B then contextualizes those refined
quantities before the principal hyperbolic projection and pointer decoder.
Thus leaf-PID gradients can originate in Level-1 reconstruction loss.
`canonical_pion_first_level` remains an explicit disabled-by-default ablation.

Corruption training rebuilds all derived p4, charge, source, histogram, level,
and conflict features. Candidate-correctness, corruption-class, and explicit
hard-negative losses consume these outputs; corrupted branches are never
queued as positive channel examples.
