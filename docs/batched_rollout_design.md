# Batched free-rollout design and evidence boundary

`evaluation_reference_rollout` remains the bounded batch-size-one correctness
oracle. The implemented multi-event path consists of `batched_level_step`,
`batched_decode_level`, and `batched_free_rollout`. It completes multiple
levels for padded events on CPU fixtures, but is not production-ready until a
guarded CUDA smoke and representative profiling exist.

The state contract keeps `active_event_mask`, `stopped_event_mask`,
`levels_completed`, and tensor stop codes per event. At each level, decoded
`[B,Q,N]` daughter masks are intersected with the active-event and padded-node
masks. Accepted mothers occupy padded query slots, which supports a variable
number of accepted mothers without scalar traversal. Exact p4 and charge use
segmented tensor sums; recursive source masks and conflict matrices propagate
from daughter unions; node IDs are derived deterministically from each event's
maximum active ID.

`batched_decode_level` performs cardinality-constrained soft-score decoding,
recursive-source conflict rejection, type/charge/physical compatibility, and
confidence filtering without extracting CUDA tensor scalars. Events stop
independently on no accepted object, configured root completion, or maximum
level. The state remains padded rather than compacted. Weighted set packing and
beam search stay on the bounded evaluation path. Repeated-state detection and
optional compaction remain follow-up engineering work.

CPU tests compare every event and every completed level against independent
batch-size-one reference rollouts, including exact daughter-summed p4,
physical features, deterministic IDs, links, source conflicts, and independent
stopping. These are fixture mechanics, not GPU throughput evidence.
