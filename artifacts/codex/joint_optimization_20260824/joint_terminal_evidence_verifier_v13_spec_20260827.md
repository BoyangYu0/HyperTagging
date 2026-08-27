# Joint terminal-evidence verifier specification v13

Status: **PENDING INDEPENDENT REVIEW; NO ACTION AUTHORIZED**

This is the narrowly corrected successor to sealed v12 commit `c47c79f89055896f8196de140f6797ebb6a71509`. It preserves the compiled A01–A05 authority, paired statistical oracle, state machine, production/offline APIs, all downstream fail-closed gates and every v12 value outside the exact allowed-diff manifest. The sibling JSON is normative; this document is only a review index.

## V12 BLOCK closure

- `NormalizedTensorManifest.v12` now compiles exactly 79 raw-UTF-8-sorted tensor keys in three disjoint exhaustive dtype groups. Its 118,500 tensor records are ordered teacher `500 × batch4 × 79` followed by rollout `1000 × singleton × 79`. Every tensor record and every one of 1,500 batch-index entries has an exact RFC8785-JCS UTF-8 projection and SHA-256 algorithm.
- `BatchPlanReceipt.v12` maps each teacher batch4 and rollout singleton one-to-one, in the same fixed order, to the unique reopened normalized-manifest `batch_sha256` over exactly 79 ordered tensor-record hashes. Per-event, per-tensor, positional-fallback and ambiguous mappings reject.
- Teacher and rollout statistics now have explicit block-invariant content projections. Ref-derived hashes come only from reopened `ArtifactRef.v5` objects; rollout content is hashed first; teacher P4 binds that rollout hash; full block-specific receipt digests cannot substitute.
- Block-specific `teacher_statistics_ref` and `rollout_statistics_ref` are explicit permitted envelope exclusions, while their reconstructed invariant projections must remain byte-equal across repeated arms.
- All six before/after state targets map exactly into ABBA scientific content. `runtime_manifest` maps only to the new `runtime_manifest_sha256` field and cannot be replaced by `data_access_sha256`.

## Exact tests and preservation

T172–T173 are narrowly corrected with exact teacher-batch4, rollout-singleton, record/batch JCS and one-to-one BatchPlan mapping cases. T176 is narrowly corrected for projection and runtime-manifest closure. T187–T194 add independent golden serialization, universe/order/dtype/count rejection, exact BatchPlan mapping, positive repeated-arm block-specific refs with equal content, negative projection/exclusion cases and runtime-manifest mapping rejection. Every test T001–T194 and every parameter case is required with no skip or xfail.

The JSON still contains exactly 38 schema nodes. Only `NormalizedTensorManifest.v12`, `BatchPlanReceipt.v12`, `StateIntegrityReceipt.v12`, `ABBAEvaluationReceipt.v11`, T172–T173, T176, T187–T194 and administrative/version binding values may differ from v12. The exact `v13_exact_allowed_v12_changes` pointer list is normative.

## Fail-closed boundary and authorization

F01–F05, R01–R05 and P01 remain `MISSING_OR_NOT_ACCEPTED`. A structurally valid chain still stops at the first compiled missing downstream gate. No runtime fixture, bundle, environment variable, CLI option, registry, monkeypatch or alternate path can change authority or gate constants.

Authority promotion, implementation, feasibility, recovery, scheduling, submission, payload access, scientific execution and final promotion are all false. No implementation, test, scheduler, payload or science action was run while authoring this specification.

Future implementation receipts must bind annotated tag `ht-joint-terminal-evidence-verifier-v13-spec-20260827-v1`, its resolved commit, the SHA-256 of the exact tagged JSON bytes and all reviewed authority artifacts. V5–v12 verifier specifications remain lineage only.
