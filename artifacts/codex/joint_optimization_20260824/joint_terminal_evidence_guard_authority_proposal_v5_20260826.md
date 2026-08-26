# Joint terminal evidence guard authority proposal v5

Date: 2026-08-26

Status: **revised proposal pending fifth independent review; not authority**.

V5 supersedes v4 commit `c9161c5108d9c9dc367f32c5eeb50b65f46ebf65` after a fourth independent `REVISE`. The JSON binds the verbatim review and provenance. All authority, execution, submission, recovery, feasibility, scheduler, payload and promotion flags are false.

## Exact normative overlay

V5 uses v3 JSON SHA `2be54d832c357f0b53f092c6878cfd0889a59384db0a67558356fe1a42161489` as its normative base. Its ordered RFC 6901 replacement map names every replaced or added pointer and the exact inline replacement node. Every v3 field and subfield survives unchanged unless its pointer is listed or descends from a listed replaced pointer. There is no broad prose inheritance and no remove operation.

Explicitly unchanged are v3's causal receipt envelope, source/hash bindings, UID and canonical-byte rules, model-state digest, PCG64/bootstrap/inference protocol, guard margins, timing and process-memory exact keys, hard guards, descriptive-diagnostics exclusion and zero audit counters.

The overlay explicitly retains:

- terminal/controller/result/metrics and data-manifest causal hashes;
- exact 2,000/1,000 UID equality and frozen event/tensor/batch manifests;
- teacher/rollout exact record keys, formulas, ranges and denominators;
- raw p-values, Holm primaries, Bonferroni simultaneous intervals and intersection-union NI;
- unique recovery `.batch` evidence with `0 < ElapsedRaw <= 14,400`;
- strict binary-unit `0 < MaxRSS <= 64 GiB`;
- exact timing/process-CUDA memory receipts;
- 15-second device-wide telemetry as diagnostic only;
- schema-bound per-level diagnostics excluded from authority.

## V5 synthetic feasibility workload

The feasibility order is exactly:

1. `q32_synth_A`
2. `relbias_synth_A`
3. `relbias_synth_B`
4. `q32_synth_B`

Every block is a fresh evaluator subprocess and CUDA context. It instantiates the exact frozen architecture and code with the actual q32 or relbias treatment config, but uses domain-separated deterministic synthetic weights and maximum-shape synthetic control tensors. No terminal checkpoint is opened.

The proposal binds the current config/code hashes, architecture fields, generator domains/seeds, maximum call counts, byte counts, shape-manifest hash slot and workload-digest schema. The conservative per-block ceiling is 37,792 model forwards: 4,000 measured teacher, 32,000 measured rollout, 1,024 maximum teacher-signature warmup and 768 maximum rollout-signature warmup.

The gate runs the exact evaluator orchestration, model-forward and teacher/predicted/bounded/scheduled rollout-control call graph. UIDs, raw or normalized events, labels, targets, checkpoints, endpoints, metrics and outcomes remain prohibited. Outputs are infrastructure timing, memory, operation counts, byte counts and hashes only.

Because label-dependent matching, loss and metric paths cannot run without prohibited data, feasibility authorization additionally requires an independently reviewed operation-and-byte dominance certificate. It must enumerate every excluded frozen operation and prove the synthetic replacement performs at least its maximum operation count and bytes read/written. Missing or failed dominance blocks.

The feasibility gates remain:

- end-to-end time at most 10,800 seconds;
- unique `.batch` MaxRSS at most 48 GiB;
- peak CUDA reserved at most 80% of device total;
- recovery ceiling 14,400 seconds and 64 GiB.

Failure requires a resource amendment and new review; no block, call, byte movement, serialization or fsync work may be truncated.

## Disposition

V5 is pending fifth independent review. The synthetic implementation, reviewed maximum-shape manifest, workload receipt, dominance certificate, accepted terminal checkpoints and scientific recovery implementation/contract do not yet exist. No feasibility or recovery execution is authorized.

Audit counters: zero scheduler calls, payload reads, tests and scientific runs.
