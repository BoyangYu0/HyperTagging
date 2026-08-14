# Current repository audit status

This is the sole authoritative current audit report. Historical reports under
`archive/` are immutable evidence snapshots and are not current truth.

<!-- GENERATED_STATUS_SUMMARY_START -->
## Generated authoritative summary

- Audited source SHA: `6e46423e5ed4553aae1fccdce33db6e03f9cb4c8`
- Audit metadata HEAD: resolved dynamically with `git rev-parse HEAD`
- Canonical complete CPU pytest result: 463 passed, 8 skipped, 24 warnings
- Human visual review: `NOT_REVIEWED`

| Ledger status | Count |
|---|---:|
| `FIXED_AND_TESTED` | 84 |
| `IMPLEMENTED_NOT_REAL_VERIFIED` | 6 |
| `INTENTIONALLY_DEFERRED_SCIENCE` | 4 |
| `OBSOLETE_OR_DUPLICATE` | 1 |
| `PARTIAL` | 9 |
<!-- GENERATED_STATUS_SUMMARY_END -->
## Recommendation: NO-GO

The production-integrity software is fixed and CPU/fixture tested, but a clean
committed current-HEAD multi-category real pilot has not run. Therefore the
100k HTCondor canary is not yet authorized, and there is explicitly no GO for
10M. The next operator action is to commit the focused worktree, render a new
1k–10k pilot campaign from that clean source, run only its bounded shards,
validate it globally, and review the consolidated report and figures.

## Training-integration runtime evidence — 2026-08-14

Non-scientific Slurm job `15745941` completed in 2:12 on `th-cl-nv01` from
clean source `0c5e054...` with exact `gpu:v100:1` request/allocation, the
frozen CUDA 12.6 environment, the 35k train plus validation-only index, and the
actual `small_candidate`. Four checkpoints prove phase indices 0, 1, 2, and 3.
One two-event validation batch completed across four named views with flushed
JSON progress, elapsed time, event-view count, and throughput. The v2 receipt
internal SHA-256 is `04b8a81c...`; it hashes nine periodic GPU telemetry
samples and exposes peaks of 554 MiB, 11% utilization, and 33 C. Sealed test
remained closed. This closes software/runtime observability evidence only; it
is not convergence or scientific-performance evidence.

The installed uv 0.5.20 now manages both environments. The corrected project
lock SHA-256 is `38e6093...`; it contains all 10 direct runtime dependencies in
editable-root metadata, and consecutive frozen all-extras syncs made no
changes. The static drift checker, uv package check, and SciPy/PyYAML imports
pass. Strict hashed GPU sync remains separate and made no changes. The tracked
activation helper and fresh-shell-verified `htenv`/`htgpu` shortcuts select the
project and frozen GPU environments explicitly.

## Software fixed and tested

- Manifest v2 binds every task to a deterministic campaign-config digest,
  campaign ID/output namespace, exact clean source commit/tree, input size/
  mtime/stable identity, category, half-open source range, event count,
  schema/PID/feature/model-feature contracts, leaf/track/charge policies,
  buffers, campaign stage, KLM scope, and canonical task-record hash.
- Workers recompute the task hash and verify the exact clean checkout and input
  identity before basf2. They never silently use a moving branch tip.
- Parquet metadata, sidecar, completion marker, and worker result carry the
  campaign/source/task provenance. Completion-marker JSON is parsed; its
  schema/count/feature/model/provenance/range fields and mandatory parquet and
  sidecar SHA-256 values are verified.
- Shard state is classified as `COMPLETE_VALID`, `MISSING`,
  `INCOMPLETE_NO_MARKER`, `CORRUPT_HASH`, `METADATA_MISMATCH`,
  `PROVENANCE_MISMATCH`, or `EVENT_COUNT_MISMATCH`. Invalid publications are
  moved into recoverable per-task quarantine before retry; failures write
  structured `.failure.json`. Complete valid shards refuse implicit overwrite.
- The unreachable fixed-hypothesis/raw-track validation is restored and tested.
  Global validation checks unique task IDs/hashes, non-overlapping exact ranges,
  exact planned/produced count, global UIDs, common source/config provenance,
  and all completion markers.
- `status`, `validate`, `list-missing`, and non-submitting `render-resubmit`
  commands exist. Pilot (default 5k) and canary (100k) profiles render without
  submission. A 10M worker refuses unresolved KLM scope or absent representative
  canary readiness-report digest.

All preserved physics/data invariants continue to pass: MC truth supervises
topology only; reconstructed composite p4 is the exact recursive daughter sum;
MC mother p4 does not enter model inputs; schema-v4 compatibility remains; PID
pruning and channel semantics were not migrated.

## Verification

- Complete CPU suite: `334 passed, 8 skipped, 20 warnings in 324.47s`.
- Focused campaign/marker/retry/validator tests passed, including each requested
  interruption window, stale/corrupt marker, cross-task sidecar, valid rerun,
  retry quarantine, and destructive-overwrite refusal.
- Generated source consistency passed for all 18 registered notebooks.
- All 15 default fixture notebooks passed under
  `/tmp/hypertagging-production-integrity-full`.
- The modified dataset, QA, and manifest notebooks also passed as a focused
  three-notebook run. Consolidated production-readiness JSON, Markdown, and
  HTML were generated. Automated fixture PASS remains separate from human
  visual status `NOT_REVIEWED`.
- `compileall`, audit integrity, generated audit view consistency, and
  `git diff --check` are required final checks and are recorded in
  `verification_runs.yaml` after completion.

## Real pilot, KLM, and resource boundary

No real mDST preprocessing ran in this pass. The checkout contains the focused
changes but is dirty, which the new immutable worker gate correctly refuses;
no campaign-bound 1k–10k multi-category paths were supplied. The old 50-event
charged-B run remains historical ancestor evidence only and is not promoted to
current verification.

The updated real-only pilot notebook verifies campaign/source/task provenance,
completion hashes, categories, levels/multiplicities, the full PID vocabulary,
node/availability distributions, trees, B roots, channels/shared nodes, p4
closure, K_L/KLM denominators, query capacity, worker resources, dataset-index
time, and JSON-v4/native-v5 storage measurements. Because it was not run,
`klm_training_scope` remains `unresolved`; detector completeness is not claimed.
If KLM is included, a representative canary is required before 10M. If it is
excluded, the campaign must use `excluded_by_policy`, which disables collection
and persists the exclusion.

There are no current real events/s, peak worker RSS, bytes/event, finalization,
validation, index-build, or row-group findings. Fixture storage benchmarks do
not justify migration, so schema-v4 remains the production format and native-v5
remains optional/experimental.

## Deliberately deferred scientific issues

No HyperTagging model redesign was performed. Trained physics performance,
calibration, rare-channel quality, PID/fit-policy/channel-pooling ablations,
query collapse, GPU throughput, rollout profiling, whole-set scoring, and
iterative pointer decoding remain separate scientific or CUDA/HTCondor work.
No HTCondor job, 100k canary, 10M campaign, long training, or CUDA job was run
or submitted.
