# Repository audit history

[`current_status.md`](current_status.md) is the sole authoritative current audit
report. [`issue_ledger.yaml`](issue_ledger.yaml) is the machine-readable authority;
[`current_backlog.md`](current_backlog.md) and
[`evidence_matrix.md`](evidence_matrix.md) are generated views. Structured complete
runs are recorded in [`verification_runs.yaml`](verification_runs.yaml).

Every report under `archive/` is an immutable historical evidence snapshot. Its
numerical results, SHA claims, and worktree descriptions apply only to the state
inspected at the time. Archive inventory and worktree state come only from explicit
[`archive/metadata.yaml`](archive/metadata.yaml), never prose inference.

## Historical snapshots

### `9d37fde7df7cd59eda1d3464ad5351f224314d6c`

- 2026-07-30 · verification · [historical report](archive/2026-07-30_9d37fde_final_correctness_and_scalability_audit.md) · superseded by [current status](current_status.md)

### `cf63b9096a2869ba2dfc8af0f292b4b7387642bc`

- 2026-07-30 · gap · [historical report](archive/2026-07-30_cf63b90_model_revision_audit.md) · superseded by [current status](current_status.md)

### `da8f35796ae81835645ac12721a5287829c02c31`

- 2026-07-30 · gap · [historical report](archive/2026-07-30_da8f357_correctness_revision_audit.md) · superseded by [current status](current_status.md)

### `274d2df8a9df8b25142b68966dfe30c828538b90`

- 2026-07-31 · verification · [historical report](archive/2026-07-31_274d2df_final_baseline_readiness_audit.md) · superseded by [current status](current_status.md)
- 2026-07-31 · completion · [historical report](archive/2026-07-31_274d2df_final_production_readiness_report.md) · superseded by [current status](current_status.md)

### `68a7ebe81b14b6c329d78de91be18b5458c49089`

- 2026-07-31 · completion · [historical report](archive/2026-07-31_68a7ebe_post_audit_completion_report.md) · superseded by [current status](current_status.md)
- 2026-07-31 · gap · [historical report](archive/2026-07-31_68a7ebe_post_audit_geometry_and_rollout_review.md) · superseded by [current status](current_status.md)
- 2026-07-31 · definition · [historical report](archive/2026-07-31_68a7ebe_post_audit_retained_tree_definition.md) · superseded by [current status](current_status.md)

### `d2362847cf036599d955cd4c70b9d2d9a3d83a08`

- 2026-07-31 · verification · [historical report](archive/2026-07-31_d236284_final_runtime_and_scale_audit.md) · superseded by [current status](current_status.md)

### `d8be6363fce7faaf7b86a1e6edd8c1175a8fac60`

- 2026-07-31 · gap · [historical report](archive/2026-07-31_d8be636_current_head_gap_audit.md) · superseded by [current status](current_status.md)
- 2026-07-31 · verification · [historical report](archive/2026-07-31_d8be636_current_head_verification_report.md) · superseded by [current status](current_status.md)

### `f064f49985da98b69c99fb02764d854f76c12e3c`

- 2026-07-31 · completion · [historical report](archive/2026-07-31_f064f49_current_head_completion_report.md) · superseded by [current status](current_status.md)
- 2026-07-31 · verification · [historical report](archive/2026-07-31_f064f49_current_head_verification.md) · superseded by [current status](current_status.md)

### `56e0323a22195457fb69aad35925538219a95c0b`

- 2026-08-01 · verification · [historical report](archive/2026-08-01_56e0323_post_audit_correction_report.md) · superseded by [current status](current_status.md)

### `77707eb181b0ec011663f9fff5f0e0a454dc1758`

- 2026-08-01 · gap · [historical report](archive/2026-08-01_77707eb_current_head_gap_audit.md) · superseded by [current status](current_status.md)

### `6e46423e5ed4553aae1fccdce33db6e03f9cb4c8`

- 2026-08-02 · verification · [historical report](archive/2026-08-02_6e46423_production_integrity_correction.md) · superseded by [current status](current_status.md)

### `ede387e195caabf41b7da0350de15eb4b90b4417`

- 2026-08-02 · verification · [historical report](archive/2026-08-02_ede387e_focused_post_audit_consolidation.md) · superseded by [current status](current_status.md)

## Evidence boundary

The current non-self-referential source boundary and exact post-boundary
allowlist are declared in `issue_ledger.yaml`. The validator requires that the
source commit be an ancestor of HEAD and rejects every later non-audit path.
