# Joint terminal-evidence verifier specification v12

Status: **PENDING INDEPENDENT REVIEW; NO ACTION AUTHORIZED**

This is the exact-schema successor to sealed v11 commit `4a32c756cb3c1a9f4de88df8dfbfa5738ab25b26`. It preserves the compiled A01–A05 authority, statistical oracle, state machine, production API and all downstream fail-closed gates. The normative artifact is the sibling JSON; this document is a review index, not a substitute for any exact JSON object.

## Audit-block closure

- ABBA summaries now have exact key sets, scalar types, ranges and recomputation rules. Each block reopens a file-backed normalized-tensor manifest, exact batch plan and before/after state receipt.
- `training_operations` is exact and requires zero backward calls, optimizer/scheduler steps, parameter updates and checkpoint writes, with no optimizer constructed.
- Repeated-arm equality uses an invariant RFC8785-JCS scientific/causal-content digest. Only the explicitly enumerated process/timing/memory/scheduler envelope fields may differ.
- Scheduler evidence includes a third raw command receipt for the exact `${job_id}.batch` query, one LF-terminated five-column `JobIDRaw|ElapsedRaw|MaxRSS|State|ExitCode` row, strict raw-byte/base64/hash checks, `0 < ElapsedRaw <= 14400`, and binary-unit `MaxRSS <= 64 GiB`.
- Every native receipt declares `CommonReceipt.v5` inheritance plus a complete, disjoint extension key/type contract, including nested shapes, literals, ranges, nullable refs, timestamps and exact ref targets.
- Teacher and rollout cohorts are separate (`2000` and `1000`) with the rollout equal to teacher indices `0..999`; no combined count or digest of `3000` is accepted. Individual target/Brier/loss counts may be zero, while all pooled teacher denominators must be recomputed and positive.

## Exact additions

The JSON contains 38 schemas, including `NormalizedTensorManifest.v12`, `BatchPlanReceipt.v12`, and `StateIntegrityReceipt.v12`. T162–T163 are narrowly corrected for `.batch` raw evidence, and T172–T186 cover refs, state, zero-operation counters, invariant digests, summary shapes, scheduler columns/types, separate cohorts and pooled positivity. Every test T001–T186 is required with no skip or xfail.

## Fail-closed boundary

F01–F05, R01–R05 and P01 remain `MISSING_OR_NOT_ACCEPTED`. A structurally valid chain still stops at the first compiled missing downstream gate. No runtime fixture, bundle, environment variable, CLI option, registry, monkeypatch or alternate path can change authority or gate constants.

## Authorization

Authority promotion, implementation, feasibility, recovery, scheduling, submission, payload access, scientific execution and final promotion are all false. No trainer, test, scheduler, payload or science action was run while authoring this specification.

Future implementation receipts must bind the annotated tag `ht-joint-terminal-evidence-verifier-v12-spec-20260826-v1`, its resolved commit, the SHA-256 of the exact tagged JSON bytes and all reviewed authority artifacts. Earlier verifier versions remain lineage only.
