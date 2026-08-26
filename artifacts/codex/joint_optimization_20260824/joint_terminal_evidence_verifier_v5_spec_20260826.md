# Joint terminal-evidence verifier v5 — implementation specification and test oracle

Version: `v5`, 2026-08-26

Normative companion: `joint_terminal_evidence_verifier_v5_spec_20260826.json`
Status: specification only; no verifier, scheduler query, payload access, test execution, or scientific execution is authorized.

All authorization flags are false. This document does not make terminal evidence available and does not promote any result.

## Normative authority and fail-closed gaps

The verifier must compile in the full sealed-plan identity:

- commit `b10c65e02e36dcb5878d3aa932adc559d5676a70`;
- JSON SHA-256 `45747d32d5af2682f712c399d23f0a8f04f5f137757b93cc0664d4f5e556dfe2`;
- tag `ht-reconstruction-post-terminal-decision-plan-v1-20260826`.

The reconstruction job IDs, names, primary contract hashes, primary wrapper hashes, controller IDs, parent relationships and controller wrapper hashes are exact constants in the normative JSON. Their authoritative repository paths and hashes are listed separately. Controller wrapper hashes are deliberately not mislabeled as contract hashes: controller producer bindings carry `contract_sha256: null`, the exact wrapper SHA-256, the parent job ID and the exact parent primary contract SHA-256.

The pretraining execution contract, its exact resource contract, the sealed teacher-forced direction/margin table, the secondary-endpoint table and the complete-target-efficiency definition are not authoritative in the current tracked evidence. They are therefore explicit authority gaps `A01` through `A05`, not invented constants. A bundle cannot fill them. Until a later reviewed specification revision replaces each null with an exact committed/tagged artifact SHA-256, verification returns the specified authority error and no architecture or pointer decision exists.

The one authoritative pretraining-validation cohort binding is available from the sealed plan: count `5000`, ordered UID digest `1e424f72a3d3a3efb4c1548579c46bd464794e2594444e36a36e4adbcbc68c8e`, scheme `sha256-u64be-length-prefixed-utf8-v1`.

## Bytes, hashes and evidence policy

There are two intentionally different hashes:

1. Artifact hashes cover exact bytes read once from an absolute, confined, no-follow regular-file descriptor. The verifier compares `fstat` identity and metadata before and after reading and rejects change during verification.
2. Node hashes cover RFC 8785 JCS bytes of the complete node after removing exactly the top-level `digest` key. Every evidence reference remains in the hash. Hydrated artifact contents are forbidden inside nodes.

Producer and consumer use that same node algorithm. No code path may omit `evidence_refs`, and no parsed-object equality may substitute for exact source-manifest byte equality.

Raw scheduler output is strict RFC 4648 base64 in a file-backed receipt. Length and SHA-256 are computed after strict decoding. This preserves arbitrary bytes, including NUL and non-UTF-8 failures, instead of round-tripping through a JSON string. Scheduler parsing begins only after exact argv, zero exit status and zero stderr bytes have been established.

Every JSON object rejects duplicate keys and unknown keys. Decision-bearing decimals are canonical decimal strings and are evaluated with exact decimal arithmetic; JSON binary floats, NaN and infinity are forbidden.

`roles_accessed` is a sorted, duplicate-free array of allowed strings. A scalar string is an error. The sealed-test, stress, restricted-raw and restricted-source roles are forbidden at every artifact and node boundary. Denials are exact false booleans and are never inferred from absence.

## Terminal evidence

The terminal node contains exactly five job entries: q32, relation bias, their two after-any controllers, and pretraining. Each entry has a fixed key set and mandatory file references appropriate to its kind. Nullability is exact; controllers cannot carry science artifacts, and science jobs cannot omit their required evidence.

Scheduler evidence uses fixed binaries and arguments. `sacct` must produce exactly one LF-terminated, headerless row with exactly these 15 fields, in order:

`JobIDRaw,JobName,User,State,ExitCode,DerivedExitCode,Submit,Start,End,Elapsed,Timelimit,NodeList,ReqTRES,AllocTRES,Restarts`.

Identity, `COMPLETED`, both `0:0` exits, zero restarts, ordered timestamps, exact elapsed time, resource TRES and job-specific timelimit are recomputed. `scontrol` independently corroborates identity, canonical `boyang.yu(12184)`, status, resources, restart/requeue state, and exact controller `afterany` parent. Reconstruction primaries are exactly one `h100nvl`, 8 CPUs, 64G and `02:00:00`; controllers are 1 CPU, 1G and `00:10:00`. Pretraining remains blocked on `A05` rather than accepting a guessed resource profile.

Source manifests and artifact manifests use absolute paths and exact file-backed entries. Each entry records exact bytes, producing job, correctly typed producer binding, role and dependency hashes. The dependency graph must be closed and acyclic. Every decision-bearing job reference occurs exactly once in the job artifact manifest. Before and after source-manifest files must have identical raw length and SHA-256, and their listed files must still match.

Typed receipts have exact common and type-specific key sets. Checkpoints bind actual checkpoint bytes, exact step, finite audit, architecture, configuration, data, normalizer, state specification and source audit. Results bind the same checkpoint and exact horizon. Pretraining cannot use a self-referential presentation claim.

Event-score artifacts independently require unique event keys and unique UIDs. The ordered UID digest is recomputed. Q32 and relation-bias rows must match exactly by `(event_key, uid)`, count, order and digest. Structural guards apply to both arms in both teacher-forced and rollout modes.

## Exact family-two decision oracle

The normative JSON fixes a library-independent SHA-256 counter/rejection sampler, seed `20260826`, 10,000 paired resamples, exact-rational add-one directional p-values, fixed endpoint tie order, Holm step-down adjusted p-values, and rank-specific two-sided percentile intervals. Quantile indexing is specified by nearest rank.

Promotion requires the rollout relation-bias-minus-q32 predicted-edge-F1 point delta to be at least `0.01`, its Holm-rank lower interval bound to exceed zero, its adjusted p-value to be at most `0.05`, and the registered teacher family endpoint to pass the same directional interval/p gate. All structural, teacher, secondary, complete-target-efficiency, throughput and exact resource-contract guards must pass.

The exact teacher, secondary and complete-target formulas cannot be reconstructed from prose. The statistical engine is specified, but decision evaluation is deliberately blocked on `A02` through `A04`. A training-loss field is never a substitute.

## Non-bypassable transition model

The only public transition operation is:

`verify_chain(node_refs, trusted_roots) -> VerifiedChain`

It receives an immutable tuple of file references, requires an exact prefix of `terminal → pair → locator → pilot → ladder → pointer → hpo → final`, and reopens and revalidates every node from terminal on every call. `decision()` accepts only the live, private capability returned by that module instance.

There is no public graph constructor, node list, append/add method, hydrated-dictionary input, or caller-deserializable verified state. Every node has an exact schema, key set, false authorization, recomputed digest and predecessor digest. Pairing uses this same path; it cannot be manually inserted.

## Later evidence

Locator evidence contains four file-backed, independently attested ordered cohorts with exact roles, counts, selectors, versions and seeds. The pretraining digest is the plan constant; teacher and rollout digests must match terminal event evidence; all four digests are distinct.

The locator pilot recomputes the exact 64-UID selection per cohort and every resolution row. Claimed counters or a `passed` boolean cannot substitute for receipts.

The transfer ladder contains all six arm/mode cells and six signed contrasts. Every cell is terminal, resource-checked and uses one byte-identical train-only normalizer. Contrasts are recomputed using exact three-member Holm families separately for teacher and rollout. Selection follows the sealed three-way rule and cannot contradict its cells or contrasts.

The selected pointer can progress to the current HPO branch only for opt1352. It must reference the exact terminal pretraining checkpoint bytes and exact ladder selection, optimizer step 1352, authoritative 5,000-UID validation cohort, finite and validation audits, architecture, configuration, data, normalizer, state specification, runtime, telemetry, terminal and source-audit artifacts. Its criterion is `validation_plus_transfer`; train loss remains false.

The HPO terminal node contains the exact six ordered learning-rate/scheduled-sampling configurations, exact 1,094-step horizon, both validation steps, resource ceiling, midpoint policy, global stopping policy, all arm/controller receipts, concurrency proof, retry absence, aggregate H100-hours and independently recomputed final ranking. A plan-only node cannot advance.

The final node is evidence for exactly one completed final reconstruction. It binds the HPO-selected configuration, selected pointer, architecture decision, independently reviewed contract, immutable snapshot, exclusive fresh output root, terminal/runtime/telemetry receipts, step-547 and step-1094 validation, rollout validation, exact cohorts and unchanged source bytes. `validation_qualified` is recomputed; it remains explicitly not sealed-test-qualified.

## Test oracle and acceptance

The normative JSON names `T001` through `T110`, plus parameter expansions. `T001` is a true full end-to-end synthetic path using only the public file-reference API after `A01`–`A05` have been closed in a separately reviewed later specification revision. The production public API has no runtime authority selector and no test profile. The suite separately proves constructor, append, hydrated-dictionary and forged-capability bypasses unavailable. It includes raw binary receipts, fixed scheduler arguments and 15-column parsing, byte-level source drift, provenance closure/cycles, cohort uniqueness and digest recomputation, independent statistical golden vectors, both Holm orderings, missing authority, every guard family, complete ladder selection, actual pointer causality, exact HPO execution evidence and final receipt causality.

No implementation is accepted unless every named case and parameter case passes with no skip or xfail, independent oracle fixtures do not import implementation helpers, static inspection finds no scheduler/submission/payload calls, and a frozen no-sync receipt binds the implementation, tests, this specification and the lockfile.

## Critical implementation checklist

- Close `A01`–`A05` by reviewed, committed and tagged authority artifacts; publish a new spec revision with exact hashes.
- Implement strict duplicate-key JSON, RFC 8785 JCS, exact node evidence policy, canonical decimals and strict base64.
- Implement confined no-follow, single-read artifact verification and full manifest/provenance closure.
- Implement fixed raw scheduler argv and exact semantic parsing for all five jobs.
- Implement exact typed receipt schemas and correct primary/controller/pretraining producer variants.
- Implement UID uniqueness, the authoritative digest algorithm, cohort matching and locator/pilot replay.
- Implement the independent family-two and ladder statistical engines with published golden vectors.
- Expose only immutable `verify_chain`/`decision`; revalidate the full file-backed prefix on every call.
- Require complete terminal ladder, HPO and final receipts; never accept asserted pass/selection/qualification booleans.
- Run all `T001`–`T110` cases frozen/no-sync only after implementation; obtain an independent read-only audit before any use.
