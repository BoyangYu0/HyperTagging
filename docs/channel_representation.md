# Structured two-B channel representation

The `direct-mdst-tree-v2` event record keeps topology labels separate from
reconstructed kinematics and from event membership.

## Canonical signature

For a retained node, the signature is a recursively sorted pair of its reduced
PID token and daughter signatures. It is built after the verified PID
pruning/contraction step. Node IDs, reconstructed four-vectors, reconstructed
object IDs, and copy IDs are absent, so changing the identity of a copied node
does not change the signature.

The implementation serializes the tuple as canonical compact JSON and assigns a
stable 60-bit ID from SHA-256. ID zero means unavailable. Charge-conjugate
normalization is configurable. When enabled, the lexicographically smaller of
the original and fully conjugated recursive signatures is used.

Each event stores:

- `b1_channel_signature`, `b2_channel_signature`;
- `b1_channel_id`, `b2_channel_id`;
- `y4s_channel_signature`, the sorted unordered pair;
- `y4s_channel_id`.

The B-side ordering is deterministic but has no claim of tag/signal semantics.
The Upsilon representation is explicitly permutation-invariant.

## Count arrays and similarity

No historical dictionary-array implementation was recoverable from the
checked-in history; see `docs/model_revision_audit.md`. V2 therefore implements
the explicit fallback design:

- dense reduced-PID counts;
- depth-by-reduced-PID counts;
- relative depth counts;
- selected intermediate-particle counts;
- sorted branch multiplicities;
- node count and maximum relative depth.

The parquet stores dense PID arrays directly and the complete structured
summary as JSON. `structured_channel_similarity` is a symmetric weighted
Jaccard score over PID, depth/PID, and multiplicity components. It supports a
continuous metric-learning target and does not replace exact IDs.

## Labels that must not be conflated

- `same_event` states only that the two branches came from one event.
- B-side membership is a node/branch relation.
- `exact_channel_equal` compares canonical signatures.
- `structured_channel_similarity` compares count/topology summaries.
- `y4s_channel_id` identifies the unordered pair of B signatures.

Two B mesons from the same Upsilon event are not automatically the same
channel. Channel loss pools shared node embeddings within each truth-guided B
branch and compares branch embeddings. Individual final-state tokens are not
required to classify the full B decay.

Exact IDs remain useful for diagnostics, rare-channel stratification, and
seen/unseen evaluation. They are not the only training objective.
