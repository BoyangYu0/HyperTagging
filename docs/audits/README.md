# Repository audit history

`current_status.md` is the only authoritative current report. `issue_ledger.yaml`
is the machine-readable disposition of every historical issue. Files under
`archive/` are immutable evidence snapshots: their numerical results, SHA
claims, and worktree descriptions apply only to the state inspected at the
time. They must not be read as current truth.

| Date | Snapshot | Audited SHA | Historical | Superseded by |
|---|---|---|---|---|
| 2026-07-30 | [Model revision audit](archive/2026-07-30_cf63b90_model_revision_audit.md) | `cf63b90` | yes | [current status](current_status.md) |
| 2026-07-30 | [Correctness revision audit](archive/2026-07-30_da8f357_correctness_revision_audit.md) | `da8f35796ae81835645ac12721a5287829c02c31` | yes | [current status](current_status.md) |
| 2026-07-30 | [Correctness and scalability audit](archive/2026-07-30_9d37fde_final_correctness_and_scalability_audit.md) | `9d37fde7df7cd59eda1d3464ad5351f224314d6c` | yes | [current status](current_status.md) |
| 2026-07-31 | [Runtime and scale audit](archive/2026-07-31_d236284_final_runtime_and_scale_audit.md) | `d2362847cf036599d955cd4c70b9d2d9a3d83a08` | yes | [current status](current_status.md) |
| 2026-07-31 | [Baseline-readiness audit](archive/2026-07-31_274d2df_final_baseline_readiness_audit.md) | `274d2df8a9df8b25142b68966dfe30c828538b90` | yes | [current status](current_status.md) |
| 2026-07-31 | [Production-readiness report](archive/2026-07-31_274d2df_final_production_readiness_report.md) | `274d2df8a9df8b25142b68966dfe30c828538b90` | yes | [current status](current_status.md) |
| 2026-07-31 | [Current-head gap audit](archive/2026-07-31_d8be636_current_head_gap_audit.md) | `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60` | yes | [current status](current_status.md) |
| 2026-07-31 | [Current-head verification report](archive/2026-07-31_d8be636_current_head_verification_report.md) | `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60` | yes | [current status](current_status.md) |
| 2026-07-31 | [Geometry and rollout review](archive/2026-07-31_68a7ebe_post_audit_geometry_and_rollout_review.md) | `68a7ebe81b14b6c329d78de91be18b5458c49089` | yes | [current status](current_status.md) |
| 2026-07-31 | [Retained-tree definition](archive/2026-07-31_68a7ebe_post_audit_retained_tree_definition.md) | `68a7ebe81b14b6c329d78de91be18b5458c49089` | yes | [current status](current_status.md) |
| 2026-07-31 | [Post-audit completion report](archive/2026-07-31_68a7ebe_post_audit_completion_report.md) | `68a7ebe81b14b6c329d78de91be18b5458c49089` | yes | [current status](current_status.md) |
| 2026-07-31 | [Current-head verification](archive/2026-07-31_f064f49_current_head_verification.md) | `f064f49985da98b69c99fb02764d854f76c12e3c` | yes | [current status](current_status.md) |
| 2026-07-31 | [Current-head completion report](archive/2026-07-31_f064f49_current_head_completion_report.md) | `f064f49985da98b69c99fb02764d854f76c12e3c` | yes | [current status](current_status.md) |

The current audit began from a clean `master` worktree at committed HEAD
`6f24a7a1729d50a7f98ea62e3c3ffe5e68562eec`. The model, rollout, notebook,
and audit-consolidation corrections described by the previous pass are part of
that commit; they are not an uncommitted patch. Any newer working-tree changes
are identified separately in the authoritative current report.
