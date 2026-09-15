# Current audit backlog

This unresolved-only view is generated from `issue_ledger.yaml`; do not edit it
by hand. The complete source/test/notebook mapping is in
[`evidence_matrix.md`](evidence_matrix.md).

## Unresolved status counts

| Status | Count |
|---|---:|
| `IMPLEMENTED_NOT_REAL_VERIFIED` | 6 |
| `INTENTIONALLY_DEFERRED_SCIENCE` | 4 |
| `PARTIAL` | 9 |

## Items

### DATA-012: Ten-million-event throughput and memory readiness

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- Next external evidence: representative ten-million-event production run
- Current boundary: Fixture timings are explicitly not throughput evidence.

### MODEL-008: Query duplication and collapse observability

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- Next external evidence: trained held-out duplicate and unused-query rates
- Current boundary: Mechanics and metrics exist; scientific collapse frequency is unmeasured.

### MODEL-019: Soft training PID versus hard rollout PID mismatch invisible

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- Next external evidence: trained held-out comparison of PID construction modes
- Current boundary: Modes and diagnostics work; the best physics mode is unknown.

### TRAIN-012: Channel pooling had only one untested choice

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- Next external evidence: matched held-out channel-pooling ablations
- Current boundary: Modes execute; scientific ranking is deferred.

### TRAIN-016: Scientific performance, calibration, rare-channel quality, and optimal decoding

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- Next external evidence: trained checkpoint, matched held-out real schema-v4 data
- Current boundary: No physics-improvement claim is permitted from fixtures.

### TRAIN-017: KLM and K_L detector completeness

- Status: `PARTIAL`
- Next external evidence: representative study of unmatched KLM clusters and K_L reconstructability across generic categories
- Current boundary: The charged-B pilot collected 92 explicit KLM nodes, but only 16 of 48 K_L-like leaves carried KLM provenance; detector completeness therefore remains partial.

### NB-003: Normal push CI did not run every notebook

- Status: `PARTIAL`
- Next external evidence: remote workflow execution for a committed final SHA
- Current boundary: Local full suite is executable and CI cadence is indexed, but the audited local source has no exact-SHA remote run.

### NB-004: Notebook assertions only proved a figure existed

- Status: `PARTIAL`
- Next external evidence: human visual review for plot legibility
- Current boundary: Core JSON semantics are asserted; visual interpretation remains human review.

### NB-006: Trained rollout notebook lacked divergence, ambiguity, slices, and PID diagnostics

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- Next external evidence: trained checkpoint, held-out real schema-v4 data
- Current boundary: Required diagnostics are implemented but deliberately NOT RUN here.

### FINAL-002: Whole-set compatibility scorer appeared runnable but was disconnected

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- Next external evidence: scientifically coherent proposal-ranking and training design
- Current boundary: Runnable YAML was removed; CLI cannot silently accept the old field.

### FINAL-003: Iterative pointer mask appeared runnable but was disconnected

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- Next external evidence: scientifically coherent differentiable decoding design
- Current boundary: Runnable YAML was removed; bounded iterative decoding remains a design only.

### FINAL-007: Free rollout lacked a batched validation step

- Status: `PARTIAL`
- Next external evidence: guarded CUDA smoke, representative memory and throughput profiling
- Current boundary: batched_free_rollout now completes multiple levels with per-event stopping and matches independent batch-size-one references on deterministic CPU fixtures; GPU readiness remains unverified.

### FINAL-013: GPU throughput and remaining scalar synchronization

- Status: `PARTIAL`
- Next external evidence: guarded CUDA run, representative GPU profiling
- Current boundary: The padded batched free path avoids tensor-scalar extraction during level traversal; the bounded reference rollout and optional diagnostics retain Python work, and no CUDA profile exists.

### DATA-018: Production query and cardinality capacities lacked representative slices

- Status: `PARTIAL`
- Next external evidence: representative multi-category production dataset index
- Current boundary: The index reports per-level quantiles and source/event/neutral/channel-frequency slices with bounded coverage; the 50-event charged-B pilot had zero overflow, but it is not representative of production.

### AUDIT-002: Exact starting-HEAD CI failed because audit ancestry was unavailable in a shallow checkout

- Status: `PARTIAL`
- Next external evidence: successful workflow_dispatch or push run at the audited source or audit commit
- Current boundary: Run 30698801983 passed unit tests and notebook smoke but failed audit integrity; checkout now fetches full history and workflow_dispatch accepts an explicit SHA, but no post-fix remote run exists.

### DATA-020: Full-truth to reconstructable-channel collisions lacked indexed diagnostics

- Status: `PARTIAL`
- Next external evidence: representative multi-category index and signature-level causal attribution of PID reduction, contraction, charge conjugation, and copied-node deduplication
- Current boundary: The index now counts distinct full channels per reconstructable ID and co-occurring mechanisms; it deliberately does not call co-occurrence causal attribution.

### ROLLOUT-007: Batched rollout lacked phase-level profiling instrumentation

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- Next external evidence: guarded representative CUDA memory and throughput profile
- Current boundary: Optional host-phase intervals exist without forcing synchronization; they are not CUDA timing evidence.

### PROD-004: Pilot and canary gates lacked immutable profiles and current real evidence

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- Next external evidence: clean committed current-HEAD 1k-10k multi-category pilot and validated consolidated readiness report
- Current boundary: Pilot and 100k profiles render without submission and 10M fails closed without canary evidence; no current-worktree real pilot was run.

### PROD-005: KLM training scope and production resource decision remain unresolved

- Status: `PARTIAL`
- Next external evidence: representative multi-category KLM/K_L pilot denominators, 100k canary if KLM is included, and real worker/storage/index timings
- Current boundary: The required fields and explicit included/excluded_by_policy/unresolved decision are implemented; current scope is unresolved, resource results are fixture-only, and schema-v4 remains selected.
