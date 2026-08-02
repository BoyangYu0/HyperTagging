# Production-integrity correction snapshot

Date: 2026-08-02  
Base HEAD: `6e46423e5ed4553aae1fccdce33db6e03f9cb4c8`  
Worktree state: dirty, with the focused corrections listed below  
Scope: direct-mDST campaign planning, shard publication/recovery, global
validation, production-gate notebooks, and audit consolidation

This is an immutable historical snapshot of this correction pass. Current
truth remains `docs/audits/current_status.md`.

## Software fixed and tested

- Manifest v2 binds campaign ID/config digest, exact clean source commit/tree,
  input stat identity, source range, scientific/feature contracts, output
  namespace, buffers, campaign stage, KLM scope, and a canonical task-record
  hash.
- Workers recompute the task hash, verify the exact clean checkout and input
  identity, and never fall back to a moving branch tip.
- Completion markers are parsed as versioned JSON. Required parquet and
  sidecar SHA-256 values plus schema/count/feature/model/campaign/source/task/
  range fields are verified against sidecar and Parquet metadata.
- Incomplete/inconsistent publications are classified and quarantined before
  retry. Complete valid shards short-circuit; destructive overwrite is an
  explicit operator action. Failures publish structured `.failure.json`.
- The unreachable fixed-hypothesis/raw-track check was restored. Raw, fixed,
  fit-policy, campaign, source, task, and range mismatch tests execute.
- Global validation checks exact task/hash/range/count/UID/config/provenance
  consistency and reports category, fit-policy, leaf-mode, KLM, B-root,
  incomplete-branch, node/depth, and bytes/event distributions.
- `status`, `validate`, `list-missing`, and non-submitting
  `render-resubmit` commands are available. Pilot and 100k canary profiles
  render without submission; 10M workers fail closed without a resolved KLM
  policy and representative canary report digest.

## Verification evidence

- Complete CPU pytest: `331 passed, 8 skipped, 20 warnings in 339.38s`.
- Focused production/notebook tests: passed.
- Generated notebook source consistency: 18 notebooks passed.
- Default fixture notebook smoke: 15 notebooks passed under
  `/tmp/hypertagging-production-integrity-full`.
- Consolidated fixture readiness report: JSON, Markdown, and HTML generated;
  automated fixture status is separate from human `NOT_REVIEWED`.

## External and scientific boundary

No current-worktree real mDST pilot was run: the checkout is dirty, so the new
worker contract correctly refuses to call it immutable campaign source, and no
campaign-bound 1k–10k multi-category manifest was supplied. No HTCondor job,
100k canary, 10M campaign, CUDA job, or long training was submitted or run.

`klm_training_scope` remains `unresolved`. The real pilot notebook now reports
the required K_L/KLM denominators and branch-completeness comparison, but no
new real data populate them. Storage/resource findings remain fixture-only;
schema-v4 is retained and schema-v5 is not promoted.

## Recommendation at this snapshot

`NO-GO` for the 100k canary until a clean committed current-HEAD campaign runs
and passes the bounded multi-category pilot, including representative KLM and
resource evidence. This is also explicitly not a GO for 10M.
