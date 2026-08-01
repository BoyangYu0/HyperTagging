# Current audit backlog

This file is generated from `issue_ledger.yaml` by
`scripts/generate_audit_views.py`; do not edit status counts here.

## Ledger status counts

| Status | Count |
|---|---:|
| `FIXED_AND_TESTED` | 64 |
| `IMPLEMENTED_NOT_REAL_VERIFIED` | 4 |
| `INTENTIONALLY_DEFERRED_SCIENCE` | 4 |
| `OBSOLETE_OR_DUPLICATE` | 1 |
| `PARTIAL` | 7 |

## Unresolved or externally bounded items

### DATA-008: Real PIDLikelihood availability and fit selection on basf2 data

- Status: `PARTIAL`
- External evidence: fix real TrackFitResult selection and PIDLikelihood access, rerun bounded pilot
- Notes: The 50-event real pilot ran, but all 392 tracks reported fit choice missing and all five PIDLikelihood features unavailable; truth-derived detector inputs remained zero.

### DATA-012: Ten-million-event throughput and memory readiness

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: representative ten-million-event production run
- Notes: Fixture timings are explicitly not throughput evidence.

### MODEL-008: Query duplication and collapse observability

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: trained held-out duplicate and unused-query rates
- Notes: Mechanics and metrics exist; scientific collapse frequency is unmeasured.

### MODEL-019: Soft training PID versus hard rollout PID mismatch invisible

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: trained held-out comparison of PID construction modes
- Notes: Modes and diagnostics work; the best physics mode is unknown.

### TRAIN-012: Channel pooling had only one untested choice

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: matched held-out channel-pooling ablations
- Notes: Modes execute; scientific ranking is deferred.

### TRAIN-016: Scientific performance, calibration, rare-channel quality, and optimal decoding

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: trained checkpoint, matched held-out real schema-v4 data
- Notes: No physics-improvement claim is permitted from fixtures.

### TRAIN-017: KLM and K_L detector completeness

- Status: `PARTIAL`
- External evidence: real KLMClusters collection and a repeated bounded pilot
- Notes: The real pilot found 52 K_L-like leaves and no KLM provenance fields; current direct collection remains Tracks plus ECLClusters only.

### NB-003: Normal push CI did not run every notebook

- Status: `PARTIAL`
- External evidence: remote workflow execution for a committed final SHA
- Notes: Local full suite is executable; CI cadence is indexed, but the working tree has no run.

### NB-004: Notebook assertions only proved a figure existed

- Status: `PARTIAL`
- External evidence: human visual review for plot legibility
- Notes: Core JSON semantics are asserted; visual interpretation remains human review.

### NB-006: Trained rollout notebook lacked divergence, ambiguity, slices, and PID diagnostics

- Status: `IMPLEMENTED_NOT_REAL_VERIFIED`
- External evidence: trained checkpoint, held-out real schema-v4 data
- Notes: Required diagnostics are implemented but deliberately NOT RUN here.

### FINAL-002: Whole-set compatibility scorer appeared runnable but was disconnected

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: scientifically coherent proposal-ranking and training design
- Notes: Runnable YAML was removed; CLI cannot silently accept the old field.

### FINAL-003: Iterative pointer mask appeared runnable but was disconnected

- Status: `INTENTIONALLY_DEFERRED_SCIENCE`
- External evidence: scientifically coherent differentiable decoding design
- Notes: Runnable YAML was removed; bounded iterative decoding remains a design only.

### FINAL-007: Free rollout lacked a batched validation step

- Status: `PARTIAL`
- External evidence: guarded GPU smoke, full multi-level batched rollout
- Notes: batched_level_step handles one complete masked level and matches the reference fixture.

### FINAL-013: GPU throughput and remaining scalar synchronization

- Status: `PARTIAL`
- External evidence: guarded CUDA run, representative GPU profiling
- Notes: Normal hot paths improved, but reference rollout and diagnostics retain bounded Python work.

### DATA-013: Strict B-root discovery had zero coverage in the bounded real pilot

- Status: `PARTIAL`
- External evidence: diagnose strict-root matching on representative generic mDST and rerun
- Notes: All 50 real events used fallback B-root discovery; zero events satisfied the strict path.
