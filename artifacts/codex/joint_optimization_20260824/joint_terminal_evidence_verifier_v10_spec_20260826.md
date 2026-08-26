# Joint terminal-evidence verifier v10 specification

Status: **LUNA NORMATIVE-CONTRADICTION-RESOLVED SPECIFICATION PENDING INDEPENDENT REVIEW — AUTHORITY COMPILED, DOWNSTREAM ARTIFACTS REQUIRED**

This is a verifier specification and test oracle, not an implementation or an execution authorization. Submission, execution, scheduler access, payload access, scientific execution, feasibility execution, recovery, promotion, and root final go are all false. No tests, scheduler commands, payload reads, or scientific work were performed.

## Normative lineage and Luna v10 correction

V10 resolves the normative contradictions in blocked verifier v9 commit `5aa6235c6c04f9d86c33aa6f78c4c7f128a0a22b`, tag `ht-joint-terminal-evidence-verifier-v9-spec-20260826-v1`, JSON SHA-256 `7f0f10b6221df9961b20c25fb49dd6df39a4e598173233f6f1a0c994b35cf692`, and Markdown SHA-256 `9208f0c3da287b56ffead7850d00165d9d0f241cd7f2a2d7b89c861d75070fd5`. V9 is an exact one-block preservation correction of blocked verifier v8 commit `6380cc600df271c7bb698e91f2e139ad6ff988e6`, tag `ht-joint-terminal-evidence-verifier-v8-spec-20260826-v1`, JSON SHA-256 `6925b025bc6ea60622344bca669ec75f6a3ece34405c813bf4c098e4fd1085fd`, and Markdown SHA-256 `e8e7130dca1a27f4006ec3cae3397e7110e4e3aac68a4234e0654e49fc485a36`. V8 is an administrative-only successor to blocked verifier v7 commit `753b56f621807f9ba47a80f04ffbe42b84d6d3ee`, which corrects blocked v6. V6, v7, v8, and v9 are preserved as BLOCKED/superseded lineage only and are not implementation targets. V6 in turn is bound to verifier v5:

- v5 commit `b70b7dc41b9b91ef2c54ab3881698cbbdf469bbc`, tag `ht-joint-terminal-evidence-verifier-v5-spec-20260826-v1`;
- v5 JSON SHA-256 `86f91830f66d93aba8809044b7aea4a599ea67a56f5f044a7237cd3941f0c082`;
- authority discovery commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170`, JSON SHA-256 `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`;
- guard proposal v7 commit `4932c5849ed2d4f450f269ce2954b54c500898ce`, JSON SHA-256 `b76d67b084821b69b13935ef4eec097116f9099a49eaa7c3cfe880910565f419`;
- effective authority transition commit `079f85fb83aeb552acf570294740759318b172c5`, JSON SHA-256 `af8f5c15ee069c55b0dfdf4acb8ea0bad735a9cfb01afe645d27ce2189bcf2c6`;
- clearance verdict `CLEAR_FOR_AUTHORITY_PROMOTION_WITH_FEASIBILITY_ARTIFACTS_REQUIRED`, UTF-8 SHA-256 `664316462cfcc11810d96fbe3dd6194e635c9cd8df3f6da52f904f13d95c6074`.

The independent audit returned `BLOCK` because v6 retained the legacy SHA-index/Holm-rank-CI oracle instead of the effective-v7 PCG64 authority. The relayed review text has UTF-8 SHA-256 `d7ed7f475a71a68dcb27499f4829f43ba4299a4db7ce87d2b525d54b90dff287`. The v8 administrative instruction has UTF-8 SHA-256 `b7a747fb42182c433abcaa6987ca2d43d4b9c350812d2989009da82eab791254`; the v9 instruction has SHA-256 `707ef5e858e14c47552ddb933c6955d83cc643b8d22f55db7beb3d02614a6fe9`; and the relayed Luna v10 stop message has SHA-256 `41194160bfd595de8fcbb2a0ffa12981b1acd980faabf066b4a86ca3b8e97a19`. Read-only tracked-path, history-message, and clean-worktree discovery found no separate tracked Luna stop report and no untracked partial, so no report path or hash is invented. Any later-discovered partial is non-normative until completed, tracked, hashed, and reviewed.

Every v9 schema, scientific oracle, compiled-authority object, state machine, downstream-gate table, and canonicalization value is preserved. Within T001–T142, only T001, T010, T065, T068, T113–T115, and T118–T120 change; T143–T146 are additive. The exact allowed pointers are enumerated in JSON.

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

All five constants are present in this specification, but implementation authority closes only through an independently reviewed eventual v10 transition binding the exact sealed v10 commit, tag, JSON SHA-256, and authority-artifact hashes. Blocked v6/v7/v8/v9 lineage closes no implementation authority. Nulls, mismatches, runtime selection, environment or CLI substitution, fixture overrides, alternate/latest authority, and bundle-supplied constants remain forbidden.

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

## Normative fixtures and tests

Authorities are now non-null, so normative schema/oracle fixtures may contain matching evidence and exercise the complete calculations. A fixture can never supply, select, mutate, or override authority. Production and fixtures use the same parser, schemas, immutable compiled constants, inference code, state machine, and public API.

The production public API contains exactly two callable exports: `verify_chain` and `decision`. It exposes no spec-byte validator, constructor, append/add operation, authority selector, registry, test profile, or runtime-constant accessor. T144 checks all hidden environment, CLI, fixture, registry, monkeypatch, bundle, latest-tag, and test-profile override routes; T145 checks the exact two-callable export surface.

Within T001–T142, the unchanged v9 ranges are exactly T002–T009, T011–T064, T066–T067, T069–T112, T116–T117, and T121–T142. Corrected in-place tests are exactly T001, T010, T065, T068, T113–T115, and T118–T120. T143–T146 are additive. Every named test and every parameter case must be implemented with no skip or xfail. No test was executed; this revision received schema/equality validation only.

## Normative downstream block and offline spec-build validation

All exact A01–A05 constants are compiled and present. F01–F05, R01–R05, and P01 remain `MISSING_OR_NOT_ACCEPTED`, are immutable at production runtime, and cannot be supplied by fixtures. Therefore T001 must return `E_FEASIBILITY_SHAPE_AUTHORITY` at gate F01, stage `pair`, with `terminal` the last valid stage; `decision` is unreachable. `PASS_FINAL_VALIDATION_QUALIFIED` is forbidden until a future independently reviewed successor compiles accepted exact downstream artifacts. T143 supplies a structurally valid synthetic file-backed chain, validates terminal and the candidate pair schema/digest/linkage through the last pre-gate point, and still returns the same block without any authority override.

Corrupted-spec coverage uses only the separate offline tool `validate_and_compile_spec_bytes(immutable_spec_bytes)`. It accepts one immutable byte string and produces a deterministic build report. It is absent from the production import graph and exports, cannot call `verify_chain` or `decision`, cannot read or mutate loaded runtime constants, and cannot construct a `VerifiedChain`. T113–T115, T118–T120, and T146 mutate in-memory candidate bytes and require offline compile errors. T010, T065, and T068 instead assert that production compiled A01–A05 constants are present and immutable.

## Future implementation seal

The future frozen no-sync implementation receipt must treat the independently reviewed v10 seal as its sole specification target. It must bind the exact commit obtained from annotated tag `ht-joint-terminal-evidence-verifier-v10-spec-20260826-v1^{}`, that fixed tag, the SHA-256 of the exact tagged v10 JSON bytes, implementation and test hashes, and lock/runtime hashes. V5, v6, v7, v8, and v9 verifier specs may be retained only as lineage references and may never be implementation targets or alternates.

The same receipt must bind the exact authority discovery commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170` and JSON SHA-256 `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`; effective-v7 proposal commit `4932c5849ed2d4f450f269ce2954b54c500898ce` and JSON SHA-256 `b76d67b084821b69b13935ef4eec097116f9099a49eaa7c3cfe880910565f419`; effective authority-transition commit `079f85fb83aeb552acf570294740759318b172c5` and JSON SHA-256 `af8f5c15ee069c55b0dfdf4acb8ea0bad735a9cfb01afe645d27ce2189bcf2c6`; and v3 authority-base commit `06744aa60e5da9748a55b5ed2385322a046fdfe9` and JSON SHA-256 `2be54d832c357f0b53f092c6878cfd0889a59384db0a67558356fe1a42161489`. Any absent or mismatched binding fails closed.

## Downstream fail-closed boundary

The eleven authority-transition requirements F01–F05, R01–R05, and P01 remain missing or unaccepted. They cover the synthetic shape manifest, feasibility implementation, dominance certificate, feasibility contract/transaction, passing 37,888-forward receipt, accepted terminal checkpoints, causal/materialization/runtime manifests, recovery implementation, recovery transaction, four scientific receipts, independent review, and a later promotion-outcome authorization.

Authority compilation does not authorize any of them. Missing, malformed, stale, mismatched, incomplete, nonexclusive, unreviewed, or failed evidence blocks the exact stage. No scheduler, payload, feasibility, recovery, submission, promotion, or scientific action is authorized.

Disposition: **READY FOR INDEPENDENT SPEC EQUALITY/COMPLETENESS AUDIT**.
