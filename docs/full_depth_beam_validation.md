# Full-depth offline beam implementation and verification

Verified on 2026-09-07 against working-tree changes based on
`db7fcbffbbb5e5cc91ac3e165b8b492a336ca0a6` on
`training-integration-20260812`. No commits, pushes, resets, or changes to the
existing untracked basf2/ONNX implementation were made.

## Delivered behavior

- `AGENTS.md` records project architecture, scientific boundaries, training and
  checkpoint contracts, CPU/HPC workflows, and worktree discipline;
  `AGENT.md` is a relative symlink to it.
- `reconstruction/beam_search.py` retains bounded mother-type, cardinality and
  daughter alternatives per query, composes source-exclusive query sets, and
  advances multiple coherent forests through every configured level. Shared
  `level_rollout.py` construction and PID/policy helpers preserve daughter-sum
  kinematics. `bounded_search.py` supplies deterministic pruning without a
  tensor dependency. The former one/two-level rollout delegates to this search.
- `reconstruct_beam_from_fsps` applies the strict native schema-v4 FSP
  projection. The default greedy API is unchanged. Width one uses reference
  greedy decisions, with an additional guard against cross-generation source
  alias reuse; parity is tested for source-exclusive full and half trees.
- Scores accumulate selected-query log likelihood and omitted-query log
  no-object probability, with explicit query-count normalization and an
  optional empty-level penalty. Cross-query caps use gain over the omitted
  score, and a bounded terminal lane preserves later root alternatives.
  Completed roots precede unfinished forests. Live candidate/state identities
  retain recursive topology and all future-visible reconstructed feature state;
  terminal feature-only duplicates collapse by physical tree identity.
- Source-aware daughter enumeration uses the same pointer objective as final
  scoring and backfills past invalid type/cardinality options. Each hypothesis
  has an explicit 256-node default cap in addition to per-query, proposal, and
  global beam bounds. The direct core model-call surface is truth-filtered.
- `evaluate_full_decay.py --beam-search` writes report schema v4 and adds beam
  top-1, ranked candidate metrics/scores, search diagnostics, and
  post-inference oracle@K. Existing
  greedy keys retain their names and meaning; greedy-only output remains v3.
  Micro summaries cover event,
  source-category, and target-shape views. Oracle unit recovery and coherent
  whole-event recovery have separate sufficient statistics.
- `configs/reconstruction/full_depth_beam.json`, the evaluation documentation,
  repository map, README and examples describe the API and configurable bounds.

All package paths above are relative to `src/hypertagging/`.

## Verification environment and results

The host's system Python has no Torch and the historical README environment
path is absent. Tests used the existing frozen interpreter
`/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1/bin/python`
(Torch 2.7.1+cu126), with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, and
`MKL_NUM_THREADS=2`. The frozen environment was not modified. Missing pytest
and notebook dependencies were installed into ignored, task-specific
`artifacts/agent-verification/` directories. Frozen scientific packages took
precedence over those extra packages on `PYTHONPATH`.

Final focused verification after formatting and the independent regression
audits: **234 passed in 12.05 s**. The selected files were:

```text
test_full_depth_beam_cpu.py
test_full_depth_beam_regressions_cpu.py
test_bounded_search_cpu.py
test_hierarchical_inference_cpu.py
test_full_decay_metrics_cpu.py
test_full_decay_cli_cpu.py
test_full_decay_runner_cpu.py
test_recursive_source_exclusivity_cpu.py
test_no_truth_leakage_cpu.py
test_onnx_full_decay_beam_cpu.py
test_basf2_feature_contract_cpu.py
test_post_audit_geometry_rollout_cpu.py
test_rollout_constraint_parity_cpu.py
test_mother_charge_compatibility_cpu.py
test_full_reconstruction_cpu.py
test_validation_micro_aggregation_cpu.py
test_two_pass_leaf_pid_reconstruction_cpu.py
test_current_head_reconstruction_corrections_cpu.py
test_reconstruction_context_collation_cpu.py
```

Run these with the project interpreter and `python -m pytest -q tests/<file>`.
Coverage includes alternative types/daughters/cardinalities; multi-generation
composition; pruning budgets; deterministic topology deduplication; per-query
and recursive source exclusivity; charge/physics/finite-value checks; greedy
compatibility; truth perturbation; a real tiny model's two-pass PID and p4
closure; CLI/config rejection; JSON serialization; unequal-denominator micro
aggregation; and a three-generation event where greedy and beam top-1 fail
but beam rank two has exact topology.

The final dependency-compatible broad `python -m pytest -q` run completed with
**1,277 passed, 8 skipped, 9 failed** in 132.72 s. It preceded the last focused
regression additions; the 234-test run above verifies the final feature code.
All nine failures are a subset of 14 failures reproduced with the original HEAD
source in a separate export under `artifacts/agent-verification/head-baseline`,
without resetting the worktree. The baseline was given read access to existing
ignored checkpoint/index fixtures required by those tests. Five additional
publication tests can fail on the checkout's NFS mount because it rejects
`renameat2(RENAME_NOREPLACE)`; those five passed when the final broad run used a
temporary filesystem that supports the operation.

| Existing failing tests | Final run | Baseline | Reproduced cause |
| --- | ---: | ---: | --- |
| `test_audit_integrity_cpu.py::test_audit_archive_and_issue_ledger_integrity` | 1 | 1 | The historical audit allowlist rejects already committed post-audit paths. |
| `test_channel_cross_event_cpu.py::test_empty_zero_capacity_channel_memory_can_expand_on_explicit_resume` | 1 | 1 | Empty-memory resume duplicates its migration receipt. |
| `test_phase3_execution_authorization_cpu.py`: `test_new_authorization_is_distinct_from_old_false_report_and_self_hashed`, `test_fresh_in_allocation_preflight_is_required_and_bound`, `test_missing_or_mismatched_preflight_fails_closed`, `test_preflight_tampering_fails_even_when_the_contract_is_unchanged` | 4 | 4 | Authorization expects a stale hash for unchanged `environment/gpu/runtime-contract.json`. |
| `test_phase3_parallel_study_cpu.py::test_parallel_matrix_has_max_four_distinct_immutable_slots`, `test_phase3_slurm_calibration_wrapper_cpu.py::test_submit_command_binds_exact_tuple_and_paths` | 2 | 2 | Four existing checkpoint-copy symlinks resolve to one checkpoint, violating the immutable-slot validator. |
| `test_revised_notebooks_cpu.py::test_revised_notebooks_generate_and_execute_on_cpu_fixtures` | 1 | 1 | The existing generated fixture supplies a noncanonical parquet path to the strict indexed-data contract. |
| `test_training_selection_repromotion_cpu.py`: `test_o_excl_namespace_is_read_only_one_shot_and_does_not_move_inputs`, `test_promoted_outputs_pass_public_preflight_without_payload_open` | 0 | 2 | This checkout's NFS filesystem returns `EINVAL` for `renameat2(RENAME_NOREPLACE)`. |
| `test_training_selection_repromotion_publication_cpu.py`: `test_success_writes_accepted_receipt_last_and_keeps_all_authority_inert`, `test_accepted_receipt_failure_strands_package_without_retry`, `test_wrapper_build_opens_only_four_authenticated_metadata_documents` | 0 | 3 | The same unsupported NFS atomic-rename operation. |

These failures were left intact; they are outside the beam feature and include
historical authorization and publication contracts. Initial notebook collection
also required installing missing `nbformat`; collection then succeeded.

Local evidence for the preceding focused/broad runs and the original-HEAD
reproduction is in `artifacts/agent-verification/`: `focused-final.log`,
`full-cpu-suite-with-notebooks.log`, `baseline-audit.log`,
`baseline-phase3-channel-linked-v2.log`, `baseline-phase3-reproduction.json`,
`baseline-repromotion-with-fixture.log`, `baseline-notebooks.log`, and
`current-notebooks-sequential.log`. These generated artifacts are ignored.
`git diff --check` and Ruff undefined/unused-name checks on the new modules
and beam test passed. The final tracked diff and new source/docs were reviewed.

## Limits of the evidence

Beam inference is bounded, approximate, CPU-only through the offline API, and
processes one event per call. Width, proposal limits and score normalization
are analysis choices; no global-optimum or calibrated-event-probability claim
is made. Existing ONNX manifests retain their own policy. No basf2 event
processing, new ONNX export, GPU training, or representative trained physics
performance study was performed. Oracle@K uses truth after inference and is
not deployable top-1 performance.
