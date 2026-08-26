# Joint terminal-evidence verifier v11 specification

Status: **LUNA EXACT-SCHEMA-CONTRADICTION-RESOLVED SPECIFICATION PENDING INDEPENDENT REVIEW — AUTHORITY COMPILED, DOWNSTREAM ARTIFACTS REQUIRED**

This is a verifier specification and test oracle, not an implementation or an execution authorization. Submission, execution, scheduler access, payload access, scientific execution, feasibility execution, recovery, promotion, and root final go are all false. No tests, scheduler commands, payload reads, or scientific work were performed.

## Normative lineage and Luna v11 amendment

V11 amends blocked verifier v10 commit `b18fe726afa166eccd3e15a16a7b4cfe3281bd3d`, tag `ht-joint-terminal-evidence-verifier-v10-spec-20260826-v1`, JSON SHA-256 `393b0699fc3e1894b49f00cf805d0e28ab4012f4438ab5c992a1dcdc4480f2fb`, and Markdown SHA-256 `8e1226d620a195368dfacfbb840db7d2524bf99318bebb0826edb892a62a4e15`. V5–v10 remain lineage only and are not implementation targets. The scientific authority remains bound to:

- v5 commit `b70b7dc41b9b91ef2c54ab3881698cbbdf469bbc`, tag `ht-joint-terminal-evidence-verifier-v5-spec-20260826-v1`;
- v5 JSON SHA-256 `86f91830f66d93aba8809044b7aea4a599ea67a56f5f044a7237cd3941f0c082`;
- authority discovery commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170`, JSON SHA-256 `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`;
- guard proposal v7 commit `4932c5849ed2d4f450f269ce2954b54c500898ce`, JSON SHA-256 `b76d67b084821b69b13935ef4eec097116f9099a49eaa7c3cfe880910565f419`;
- effective authority transition commit `079f85fb83aeb552acf570294740759318b172c5`, JSON SHA-256 `af8f5c15ee069c55b0dfdf4acb8ea0bad735a9cfb01afe645d27ce2189bcf2c6`;
- clearance verdict `CLEAR_FOR_AUTHORITY_PROMOTION_WITH_FEASIBILITY_ARTIFACTS_REQUIRED`, UTF-8 SHA-256 `664316462cfcc11810d96fbe3dd6194e635c9cd8df3f6da52f904f13d95c6074`.

The relayed Luna v11 contradiction message has UTF-8 SHA-256 `7e16e40d11593e2ddbb3ee5654cb794e23ba0f57dda2739e8546036cd7c5d929`. Read-only tracked-path, history-message, and clean-worktree discovery found no separate tracked Luna report and no untracked partial, so no path or hash is invented. Any later-discovered partial is non-normative until completed, tracked, hashed, and reviewed.

The paired oracle, compiled A01–A05 authority, F01–P01 gate table/order, state machine, canonicalization, production API, and T001–T146 are byte-identical to v10 as parsed JSON values. V11 replaces six schema nodes, adds thirteen typed schemas, and adds T147–T171. The exact allowed pointers are enumerated in JSON.

The corrected preservation sentence is exact: `Within implementation_acceptance, v5 entries 3–5 and 7 remain byte-identical in place; v5 entries 1, 2 and 6 are administratively replaced by the v8 authority-closure, test-range and receipt-binding clauses; v8 entries 8–13 are additive relative to v5.`

## Corrected paired statistical oracle

The legacy per-draw SHA-index algorithm is forbidden. V7 uses two domain-separated `numpy.random.Generator(numpy.random.PCG64(seed_integer))` matrices:

- teacher: domain `ht-guard-v3/teacher/2000/pcg64/20260826`, seed hex `4b004223af6f36bd1cb553952753a421`, shape `10000×2000`, 80,000,000 serialized bytes, SHA-256 `e5a9ba6df91bf1fd58e22c2557e5c8c68a5220783cf5bb66cccf444887b1b3c0`;
- rollout: domain `ht-guard-v3/rollout/1000/pcg64/20260826`, seed hex `af89e3bbe40d7dc29e4b3d6aaf8cc690`, shape `10000×1000`, 40,000,000 serialized bytes, SHA-256 `72e22d5e7c3f13b783d3b8717d878b84a41482620d5031701952903b01ab08a9`.

Seeds are the first 16 SHA-256 bytes of the UTF-8 domain, interpreted unsigned big-endian. Matrices serialize as headerless C-order little-endian `uint32`; non-little-endian hosts byteswap before hashing. The JSON publishes both first-32 golden vectors, exact byte lengths, shapes, and hashes.

For every endpoint, `d=relbias−q32`, bootstrap effect is `d_b`, and centered-null value is `z_b=d_b−d+h`. Both primaries use `h=0` and equality-inclusive upper-tail add-one p `(1+count(z_b>=d))/10001`. Holm step-down applies to the two raw p-values only, with lexical exact-ID tie order. Primary confidence intervals are separately sorted paired percentile effects at fixed zero-based indices 124 and 9874. These are Bonferroni familywise intervals; Holm-rank, unadjusted, interpolated, basic, BCa, and studentized alternatives are forbidden.

Higher-is-better NI uses `h=−margin`, the same equality-inclusive upper tail, p at most 0.05, and strict `d>−margin`. Lower-is-better NI uses `h=+margin`, equality-inclusive lower tail `(1+count(z_b<=d))/10001`, p at most 0.05, and strict `d<+margin`. NI receives no Holm or other multiplicity adjustment. Primary point delta exactly 0.01 and p exactly 0.05 pass; CI lower exactly zero and NI estimate exactly on its margin fail.

Effective bootstrap source SHA-256 is `07e2290870783d0da5af19af4ec8976dd7f9dcb327514ffac14ccd282faa0c1c`; corrected paired-oracle SHA-256 is `89ecb3afc40c84e701e5367a123df93321f430741149091afdf7febf0994f7a3`; their endpoint-bound projection SHA-256 is `7a678b51c05d7c0df0ae6a79a23d661dcf42a247fe9982ac46d1dad0254d3d56`.

## Compiled A01–A05 authority

The v5 gap IDs map to the reviewed evidence as follows:

| v5 gap | Reviewed source | Compiled SHA domain |
|---|---|---|
| A01 pretraining contract | discovery A01 | artifact bytes `b7be3644…`; embedded canonical payload separately `a50408ee…` |
| A02 teacher table | discovery A03 missing boundary, completed by effective v7 | projection `358c5c7c…` |
| A03 secondary table | discovery A04 missing/incompatible boundary, completed by effective v7 | projection `21ad1ea3…` |
| A04 efficiency definition | discovery A05 partial boundary, completed by effective v7 | projection `ef16e0ed…` |
| A05 pretraining resources | discovery A02 | wrapper/resource bytes `df311116…` |

The A01 artifact-file and embedded canonical digests have different domains and are both mandatory. The canonical digest removes only `contract_sha256`, then uses the reviewed sorted-key compact JSON algorithm before UTF-8 SHA-256. The pretraining producer must bind canonical contract `a50408eee13132d7a2338ea1b3f38d9ffa13fecc49d8ade510e504cbf4a891a1` and wrapper `df31111644e38f8abdadf12cd39805fe8aed4e1426437109dafd3edcfc6720f4`.

All five constants are present in this specification, but implementation authority closes only through an independently reviewed eventual v11 transition binding the exact sealed v11 commit, tag, JSON SHA-256, and authority-artifact hashes. Blocked v6–v10 lineage closes no implementation authority. Nulls, mismatches, runtime selection, environment or CLI substitution, fixture overrides, alternate/latest authority, and bundle-supplied constants remain forbidden.

Pretraining resources are exactly partition `inter`, `gpu:h100nvl:1`, one GPU, eight CPUs, 64G memory, six-hour walltime, no requeue, zero restarts, no DDP, and no two-GPU comparison. ReqTRES/AllocTRES are parsed semantically and must contain unique `cpu=8`, `mem=64G`, `gres/gpu=1`, and `gres/gpu:h100nvl=1` components. Scheduler-added billing or node fields are not invented as byte-exact authority. Runtime remains Python 3.11.11, Torch 2.7.1+cu126, CUDA 12.6, uv 0.5.20, exact lock/runtime hashes, and frozen/no-sync invocation.

## Teacher authority

The teacher cohort is exactly 2,000 ordered validation UIDs, 500 immutable batches of four, with zero fallback. The primary teacher endpoint is pooled micro pointer F1 with relbias-minus-q32 delta at least 0.01. Registered one-sided noninferiority guards are pointer precision/recall and object precision/recall at 0.005, mother-type and cardinality accuracy at 0.01, lower-is-better Brier and ten-bin ECE at 0.005, and lower-is-better loss ratio at 0.02.

The normative JSON fixes every formula, denominator, record key, ECE bin key, one-forward-per-batch-level rule, frozen-dtype contribution stage, binary64 cast stage, `math.fsum` ordering, and finite serialization rule. NI uses paired intersection-union tests with no multiplicity adjustment, one-sided p at most 0.05, and the estimate strictly inside its boundary.

## Secondary and structural authority

The rollout cohort is exactly the first 1,000 ordered validation UIDs with singleton materialization and zero fallback. Full-tree exact match, canonical subtree match, edge precision/recall, mother-type accuracy, leaf assignment, source overlap, tree-edit distance, root success, and pooled complete-target efficiency have exact formulas, directions, and margins in JSON. All except complete-target efficiency aggregate as per-UID macro means; complete-target is the sole pooled endpoint and its one NI test is reused by the efficiency gate.

Structural authority requires no predicted self-edge or cycle and strictly descending mother-to-child level; exact four-condition reconstruction validity; truth/predicted active-leaf source unions and intersection; canonical root-Counter intersection; scheduled rollout validity; full truth-source retention; zero recursive conflicts; exact teacher/predicted P4 closure; positive registered denominators; finite values; and no validation fallback.

## Efficiency and replay authority

Replay order is `q32_A, relbias_A, relbias_B, q32_B`, each in a fresh evaluator process and CUDA context. Exact immutable teacher and rollout materialization precedes the first block. Signature warmup includes exact teacher target-level lists and rollout depth × multiplicity bucket (`low<=4`, `medium=5..8`, `high>8`) × mode × up-to-eight-level coverage. Every block measures 500 batch-four teacher batches plus 1,000 singleton rollouts.

Block throughput is `3000 / positive monotonic measured seconds`; each arm is the geometric mean of its two blocks; relbias/q32 must be at least 0.9. Process CUDA allocated/reserved peaks are reset after warmup and measured after synchronized evaluation. Fifteen-second device telemetry is diagnostic only. Recovery requires one `.batch` row, `0<ElapsedRaw<=14400`, and strict binary-unit MaxRSS no more than 64 GiB; original primary elapsed values remain at most 7,200 seconds. The replay allocation is one H100 NVL, eight CPUs, 64G, four hours, no requeue/restarts, at most four GPU-hours.

Repeated arms must have identical statistics, numeric/dtype contract, target-level plan, model, cohort, manifest, batch, and causal hashes. Only timing, memory, PID, block identity/ordinal, and envelope hashes may differ. No training, update, checkpoint write, old aggregate, train loss, inferred timing, sampled telemetry substitution, reduced ABBA, reordered cohort, altered batch size, or second forward is permitted.

## Exact typed event and ABBA receipts

The schema table now contains 35 nodes. `TeacherEventSufficientStatistics.v11` has exactly the authoritative v3 20 per-UID keys: event UID/index, validation role/fallback, target-level count, object and pointer TP/FP/FN, mother-type and cardinality correct/total, Brier sum/count, ten exact ECE-bin sufficient-statistic rows, and loss sum/count. `TeacherStatisticsReceipt.v11` binds exactly 2,000 unique ordered validation UIDs, batch size four, batch boundaries, event manifest, numeric contract, treatment, block, primary/controller jobs, source contract/manifest, checkpoint, evaluator, and digest.

V3 places teacher-P4 counts only on the 1,000 rollout UIDs. V11 therefore does not invent teacher-2000 P4 fields. The teacher receipt must reference the same-arm, same-block `RolloutStatisticsReceipt.v11`, whose exact `RolloutEventSufficientStatistics.v11` records carry `teacher_p4_closed_count`, `teacher_p4_composite_count`, predicted P4 counts, and the full authoritative edge/subtree/mother/leaf/source/tree/root/complete-target/validity/fallback/secondary field set. Every per-UID formula, inequality, pooled-complete-target rule, P4 closure, validity condition, and cohort digest is recomputed.

`EventScoreReceipt.v5` is no longer a loose endpoint map. Its v11 exact schema contains only causal scalar bindings and typed refs to teacher, rollout, checkpoint, controller, contract, source, event manifest, and ABBA evidence. Hydrated event arrays, arbitrary endpoint maps, `p4_valid` shortcuts, and supplied aggregate summaries are forbidden.

`ABBAEvaluationReceipt.v11` contains exactly `q32_A`, `relbias_A`, `relbias_B`, and `q32_B`. Each block binds exact teacher/rollout refs, source jobs/contracts, a fresh PID/starttime and CUDA context, the authoritative monotonic timing fields, process CUDA peak fields, native scheduler resource fields, before/after state refs, and a block digest. Timing, memory, resource ceilings, arm geometric-mean throughput, 0.9 ratio, and repeated-arm causal/statistic equality are recomputed.

## Native causal leaves

Eight exact native receipt schemas cover scheduler, controller, runtime, telemetry, checkpoint, validation, midpoint, and resource evidence. Scheduler evidence reopens exact raw-command receipts and validates fixed argv, strict base64 bytes/length/hash, zero exit, empty stderr, timestamps, the single 15-column `sacct` row, duplicate-free `scontrol`, TRES, dependency, requeue, and restarts. Controller receipts bind the exact parent and `afterany` dependency. Runtime receipts bind executable, frozen argv, lock, source, timestamps, process identity, exit, access domain, and roles. Telemetry remains raw-file-backed and diagnostic-only.

Checkpoint receipts bind exact checkpoint bytes, the name/dtype/shape/contiguous-CPU-byte model-state digest, step, configuration, data, normalizer, finite audit, source audit, and validation cohort. Validation receipts bind checkpoint/runtime/event refs, exact role/UID/event order, metrics, access and fallback. Midpoint receipts recompute the step-547 prune/continue policy. Resource receipts reopen scheduler/contract evidence and recompute TRES, dependencies, intervals, MaxRSS, concurrency, H100-hours, and budgets.

Pair, Ladder, Pointer, HPO, and Final schemas now name the exact native refs. Every leaf is reopened no-follow and checked for bytes, hash, schema, producer, role, job, contract, source, checkpoint, cohort, and digest before summaries, guards, selection, qualification, concurrency, or budgets are recomputed.

## Normative fixtures and tests

Authorities are now non-null, so normative schema/oracle fixtures may contain matching evidence and exercise the complete calculations. A fixture can never supply, select, mutate, or override authority. Production and fixtures use the same parser, schemas, immutable compiled constants, inference code, state machine, and public API.

The production public API contains exactly two callable exports: `verify_chain` and `decision`. It exposes no spec-byte validator, constructor, append/add operation, authority selector, registry, test profile, or runtime-constant accessor. T144 checks all hidden environment, CLI, fixture, registry, monkeypatch, bundle, latest-tag, and test-profile override routes; T145 checks the exact two-callable export surface.

T001–T146 remain byte-identical to v10. T147–T171 are additive and cover positive typed-evidence reopening followed by the immutable F01 block; exact teacher and rollout keys/types/ranges; UID/event/role/fallback/P4 location; recomputation and forged summaries; missing/ref/hash/schema/role/job/contract/source/digest failures; exact four-block ABBA timing/process-memory/resources/repeated identity; raw scheduler bytes and 15 columns; controller dependencies; runtime access; telemetry; checkpoint state; validation; midpoint; resources; and downstream native closure. Every named test and every parameter case T001–T171 must be implemented with no skip or xfail. No test was executed; this revision received schema/equality validation only.

## Normative downstream block and offline spec-build validation

All exact A01–A05 constants are compiled and present. F01–F05, R01–R05, and P01 remain `MISSING_OR_NOT_ACCEPTED`, are immutable at production runtime, and cannot be supplied by fixtures. Therefore T001 must return `E_FEASIBILITY_SHAPE_AUTHORITY` at gate F01, stage `pair`, with `terminal` the last valid stage; `decision` is unreachable. `PASS_FINAL_VALIDATION_QUALIFIED` is forbidden until a future independently reviewed successor compiles accepted exact downstream artifacts. T143 supplies a structurally valid synthetic file-backed chain, validates terminal and the candidate pair schema/digest/linkage through the last pre-gate point, and still returns the same block without any authority override.

Corrupted-spec coverage uses only the separate offline tool `validate_and_compile_spec_bytes(immutable_spec_bytes)`. It accepts one immutable byte string and produces a deterministic build report. It is absent from the production import graph and exports, cannot call `verify_chain` or `decision`, cannot read or mutate loaded runtime constants, and cannot construct a `VerifiedChain`. T113–T115, T118–T120, and T146 mutate in-memory candidate bytes and require offline compile errors. T010, T065, and T068 instead assert that production compiled A01–A05 constants are present and immutable.

## Future implementation seal

The future frozen no-sync implementation receipt must treat the independently reviewed v11 seal as its sole specification target. It must bind the exact commit obtained from annotated tag `ht-joint-terminal-evidence-verifier-v11-spec-20260826-v1^{}`, that fixed tag, the SHA-256 of the exact tagged v11 JSON bytes, implementation and test hashes, and lock/runtime hashes. V5–v10 verifier specs may be retained only as lineage references and may never be implementation targets or alternates.

The same receipt must bind the exact authority discovery commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170` and JSON SHA-256 `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`; effective-v7 proposal commit `4932c5849ed2d4f450f269ce2954b54c500898ce` and JSON SHA-256 `b76d67b084821b69b13935ef4eec097116f9099a49eaa7c3cfe880910565f419`; effective authority-transition commit `079f85fb83aeb552acf570294740759318b172c5` and JSON SHA-256 `af8f5c15ee069c55b0dfdf4acb8ea0bad735a9cfb01afe645d27ce2189bcf2c6`; and v3 authority-base commit `06744aa60e5da9748a55b5ed2385322a046fdfe9` and JSON SHA-256 `2be54d832c357f0b53f092c6878cfd0889a59384db0a67558356fe1a42161489`. Any absent or mismatched binding fails closed.

## Downstream fail-closed boundary

The eleven authority-transition requirements F01–F05, R01–R05, and P01 remain missing or unaccepted. They cover the synthetic shape manifest, feasibility implementation, dominance certificate, feasibility contract/transaction, passing 37,888-forward receipt, accepted terminal checkpoints, causal/materialization/runtime manifests, recovery implementation, recovery transaction, four scientific receipts, independent review, and a later promotion-outcome authorization.

Authority compilation does not authorize any of them. Missing, malformed, stale, mismatched, incomplete, nonexclusive, unreviewed, or failed evidence blocks the exact stage. No scheduler, payload, feasibility, recovery, submission, promotion, or scientific action is authorized.

Disposition: **READY FOR INDEPENDENT SPEC EQUALITY/COMPLETENESS AUDIT**.
