# Batched free-rollout design and evidence boundary

`evaluation_reference_rollout` remains the bounded batch-size-one correctness
oracle. The implemented multi-event path consists of `batched_level_step` and
`batched_rollout_level_transition`; it is not production-ready.

The state contract keeps `active_event_mask`, `stopped_event_mask`,
`levels_completed`, and tensor stop codes per event. At each level, decoded
`[B,Q,N]` daughter masks are intersected with the active-event and padded-node
masks. Accepted mothers occupy padded query slots, which supports a variable
number of accepted mothers without scalar traversal. Exact p4 and charge use
segmented tensor sums; recursive source masks and conflict matrices propagate
from daughter unions; node IDs are derived deterministically from each event's
maximum active ID.

An event with no accepted mother stops independently and cannot append in later
transitions. A future full driver must additionally tensorize constrained hard
decoding, root-token stop masks, repeated-state detection, and optional event
compaction. It must keep weighted set packing and beam search on the bounded
evaluation path. Representative CUDA profiling and a guarded CUDA smoke test
are required before any production-readiness claim.

CPU tests compare a two-event append against `evaluation_reference_rollout`'s
append construction, including exact daughter-summed p4, physical features,
links, and next-pass embeddings. A second test exercises independent event
stopping. These are fixture mechanics, not GPU throughput evidence.
