# Repository audit history

[`current_status.md`](current_status.md) is the sole authoritative current audit
report. [`issue_ledger.yaml`](issue_ledger.yaml) is the machine-readable issue
disposition, and [`current_backlog.md`](current_backlog.md) is generated from
that ledger.

Every report under `archive/` is an immutable historical evidence snapshot.
Its numerical results, SHA claims, and worktree descriptions apply only to the
state inspected at the time. Every archive report is historical and is
superseded by the [current status](current_status.md); none is current truth.

## `cf63b90` — 2026-07-30

- Gap analysis: [model revision audit](archive/2026-07-30_cf63b90_model_revision_audit.md)

## `da8f35796ae81835645ac12721a5287829c02c31` — 2026-07-30

- Gap analysis: [correctness revision audit](archive/2026-07-30_da8f357_correctness_revision_audit.md)

## `9d37fde7df7cd59eda1d3464ad5351f224314d6c` — 2026-07-30

- Verification: [correctness and scalability audit](archive/2026-07-30_9d37fde_final_correctness_and_scalability_audit.md)

## `d2362847cf036599d955cd4c70b9d2d9a3d83a08` — 2026-07-31

- Verification: [runtime and scale audit](archive/2026-07-31_d236284_final_runtime_and_scale_audit.md)

## `274d2df8a9df8b25142b68966dfe30c828538b90` — 2026-07-31

- Gap analysis: [baseline-readiness audit](archive/2026-07-31_274d2df_final_baseline_readiness_audit.md)
- Completion: [production-readiness report](archive/2026-07-31_274d2df_final_production_readiness_report.md)

## `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60` — 2026-07-31

- Gap analysis: [historical gap audit](archive/2026-07-31_d8be636_current_head_gap_audit.md)
- Verification: [historical verification report](archive/2026-07-31_d8be636_current_head_verification_report.md)

## `68a7ebe81b14b6c329d78de91be18b5458c49089` — 2026-07-31

- Gap analysis: [geometry and rollout review](archive/2026-07-31_68a7ebe_post_audit_geometry_and_rollout_review.md)
- Verification: [retained-tree definition](archive/2026-07-31_68a7ebe_post_audit_retained_tree_definition.md)
- Completion: [post-audit completion report](archive/2026-07-31_68a7ebe_post_audit_completion_report.md)

## `f064f49985da98b69c99fb02764d854f76c12e3c` — 2026-07-31

- Verification: [historical verification](archive/2026-07-31_f064f49_current_head_verification.md)
- Completion: [historical completion report](archive/2026-07-31_f064f49_current_head_completion_report.md)

## `77707eb181b0ec011663f9fff5f0e0a454dc1758` — 2026-08-01

- Gap analysis: [current-boundary gap audit](archive/2026-08-01_77707eb_current_head_gap_audit.md)

## Current evidence boundary

The focused post-audit pass began from clean `master` commit
`e9b828c45c9c26bf4dcd2b76e96fb788ee6d3f1b`. The independently tested source
boundary is `88270d00fb5c9fc6311daab2f9443832ebe7c3bf`. The commit after that boundary
contains only the exact audit/index paths listed in `issue_ledger.yaml`; the
validator rejects any later source, model, loss, preprocessing, training,
reconstruction, config, runtime, workflow, test, notebook-source, or other
non-allowlisted change. The audit-metadata commit is identified dynamically by
`git rev-parse HEAD`, avoiding a self-referential SHA requirement.

The generated `archive/manifest.yaml` records date, audited SHA, report type,
supersession target, evidence scope, and historical worktree state for every
immutable snapshot.
