# Joint terminal-evidence verifier v6 specification

Status: **SEALED SPECIFICATION ONLY — AUTHORITY COMPILED, DOWNSTREAM ARTIFACTS REQUIRED**

This is a verifier specification and test oracle, not an implementation or an execution authorization. Submission, execution, scheduler access, payload access, scientific execution, feasibility execution, recovery, promotion, and root final go are all false. No tests, scheduler commands, payload reads, or scientific work were performed.

## Normative lineage

V6 is a strict successor to verifier v5:

- v5 commit `b70b7dc41b9b91ef2c54ab3881698cbbdf469bbc`, tag `ht-joint-terminal-evidence-verifier-v5-spec-20260826-v1`;
- v5 JSON SHA-256 `86f91830f66d93aba8809044b7aea4a599ea67a56f5f044a7237cd3941f0c082`;
- authority discovery commit `1a5c26eab5f7649dac76b95b7c698e1658ce0170`, JSON SHA-256 `81515669819e57b5628985b0422522ed2e1bc584da1af744ce7c0e0dbb2a2ee7`;
- guard proposal v7 commit `4932c5849ed2d4f450f269ce2954b54c500898ce`, JSON SHA-256 `b76d67b084821b69b13935ef4eec097116f9099a49eaa7c3cfe880910565f419`;
- effective authority transition commit `079f85fb83aeb552acf570294740759318b172c5`, JSON SHA-256 `af8f5c15ee069c55b0dfdf4acb8ea0bad735a9cfb01afe645d27ce2189bcf2c6`;
- clearance verdict `CLEAR_FOR_AUTHORITY_PROMOTION_WITH_FEASIBILITY_ARTIFACTS_REQUIRED`, UTF-8 SHA-256 `664316462cfcc11810d96fbe3dd6194e635c9cd8df3f6da52f904f13d95c6074`.

Every v5 value is preserved except the exact authority-integration pointers enumerated in the normative JSON. All v5 schemas and oracles survive; only the two pretraining producer-binding constants and the authority-dependency text change where compilation requires it. T001–T110 remain an exact unchanged prefix. The seven original implementation-acceptance clauses remain an exact unchanged prefix.

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

T001–T110 are preserved exactly. T111–T127 add authority-source equality, digest-domain, semantic-TRES, complete teacher/secondary/efficiency/replay, runtime-override, and eleven-gate fail-closed oracles. No test was executed; this revision received schema-only structural validation.

## Downstream fail-closed boundary

The eleven authority-transition requirements F01–F05, R01–R05, and P01 remain missing or unaccepted. They cover the synthetic shape manifest, feasibility implementation, dominance certificate, feasibility contract/transaction, passing 37,888-forward receipt, accepted terminal checkpoints, causal/materialization/runtime manifests, recovery implementation, recovery transaction, four scientific receipts, independent review, and a later promotion-outcome authorization.

Authority compilation does not authorize any of them. Missing, malformed, stale, mismatched, incomplete, nonexclusive, unreviewed, or failed evidence blocks the exact stage. No scheduler, payload, feasibility, recovery, submission, promotion, or scientific action is authorized.

Disposition: **READY FOR INDEPENDENT SPEC EQUALITY/COMPLETENESS AUDIT**.
