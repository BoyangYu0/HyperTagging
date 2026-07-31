# Schema migration history

This page preserves compatibility history. Current architecture and channel
documentation describe schema-v4 first.

## V1 to v2

V2 introduced explicit node kinds, common/track/ECL/composite blocks, and
value-level availability masks. A v1 leaf was adapted as a track or ECL cluster
only when its `reco_id` prefix proved that provenance; otherwise it became
`other` or `unknown`. V1 detector blocks stayed unavailable. Existing scalar
values, links, IDs, flags, and diagnostic MC fields were retained without
inventing detector measurements.

The channel fallback used dense PID counts, depth-by-PID counts, relative depth
counts, selected intermediate counts, branch multiplicities, node count, and
maximum relative depth. This was never claimed exactly equivalent to an
unrecoverable historical dictionary-array implementation.

## V2 to v3

V3 added verified PIDLikelihood log-likelihoods, e/mu/pi/K/p energy hypotheses,
and separate full-retained versus reconstructable-retained B signatures. It
also tightened Upsilon discovery and kept target truth separate from detector
inputs. Its shard-wide JSON buffering and publication behavior were not the
production-scale endpoint.

## V3 to v4

V4 made event-row parquet, bounded writing, marker-last publication, sidecars,
dataset indexes, stable global UIDs, and worker partitioning the production
contract. It formalized input/truth daughter histograms, local and recursive
completeness, partial-target policies, source provenance, and streaming
training. Compatibility readers remain available for historical artifacts.

## V5 experimental boundary

Native nested v5 is experimental and disabled by default. Fixture benchmarks
exercise storage mechanics only and cannot justify promotion over v4 or a
ten-million-event readiness claim.
