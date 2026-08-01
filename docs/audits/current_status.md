# Current repository audit status

This is the sole authoritative current audit report. The focused pass began on
2026-08-01 from clean `master` starting HEAD
`70e99ae489e30ce9c131c6a2228ce3e5d517f584`.

## Non-self-referential evidence boundary

The ledger records audited code SHA
`6f24a7a1729d50a7f98ea62e3c3ffe5e68562eec`, which is an ancestor of the
starting HEAD. The committed `6f24a7a..70e99ae` delta contains audit documents,
the notebook registry, audit/notebook validation tooling, one audit test, and
18 generated notebooks whose normalized content differs only in cell IDs. It
contains no model, loss, preprocessing, training, reconstruction, runtime, or
scientific-config change.

Audit validation no longer requires a tracked report to contain the SHA of the
commit containing that report. Instead it proves ancestry, classifies every
later committed path against an explicit allowlist, and semantically restricts
post-audit notebook changes to cell-ID-only normalization. A later source
change invalidates the boundary. The present uncommitted correction work must
therefore be committed as source first and followed by a separate audit-only
metadata commit before `audited_code_sha` can advance.

GitHub has an exact-starting-SHA `CPU correctness` run, run
[`30692052350`](https://github.com/BoyangYu0/HyperTagging/actions/runs/30692052350),
which completed successfully. It is evidence for committed HEAD `70e99ae` only,
not for this uncommitted worktree.

## Corrections in this focused pass

- Predicted composites now use `runtime_reconstructed` input provenance,
  unavailable truth supervision/targets/counts, and a separate
  `runtime_structurally_valid` flag. Exact daughter-summed p4 is unchanged.
- LCA relation, parent ranking, exact tree distance, radius, channel, variance,
  covariance, leaf PID, corruption class, candidate correctness, and hard
  negative objectives have independent typed weights serialized in checkpoint
  config. Validation reports both `validation_principal_loss` and
  `validation_full_training_objective`; best-checkpoint selection explicitly
  chooses one.
- The daughter-pooling baseline is explicitly `precontext_daughter_pool`.
  Append time persists physical summaries, and the next pass pools pre-context
  daughter representations before Stage-A attention.
- Encoder outputs and the two inspection notebooks separate physical Stage-A
  bias/attention from hyperbolic Stage-B bias/attention. Disabled Stage B
  returns no Stage-B attention matrix.
- Active documentation names the production schema-v4 and
  `physical-relations-overlap-aware-v3` contracts, includes tree-distance in
  the objective formula, and matches implemented daughter pooling.
- Notebook CI selection comes from `notebooks/index.yaml`.
  `--check-generated` generates into temporary storage, normalizes cell IDs,
  and byte-compares normalized sources. Execution outputs include a figure
  index whose visual-review state is explicitly `NOT_REVIEWED`.

## Evidence categories

Independently executable CPU evidence consists of the unit suite, audit
ancestry fixtures, generated-notebook consistency, compile checks, and
deterministic notebook execution. Fixture notebooks demonstrate software
mechanics only. Real-mDST evidence is recorded only if the bounded operator
pilot completes. Trained-checkpoint physics evidence remains absent unless a
matching held-out checkpoint is supplied. The generated
[current backlog](current_backlog.md) is the authoritative concise list of
partial, deferred, open, and externally bounded ledger items.

## Remaining boundaries

Full multi-level batched free rollout is still partial: `batched_level_step`
is a multi-event, one-level reference-equivalent append, while event-specific
multi-level stopping/compaction and guarded CUDA profiling remain future work.
No fixture timing is a throughput claim. Radius target mode, channel pooling,
PID rollout mode, and objective weights remain scientific ablations pending
matched trained held-out evidence. Real basf2 coverage, real detector feature
availability, human visual review, GPU throughput, and ten-million-event scale
remain external until measured.

## Verification result

The repository environment produced the following independently executable
results for this uncommitted worktree:

- `/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q`:
  `278 passed, 8 skipped, 19 warnings in 393.66s (0:06:33)`;
- `scripts/validate_audit_integrity.py`: PASS, 13 immutable archives, 80 ledger
  items, and one current-status document;
- `scripts/execute_notebook_smoke_tests.py --list`: 15 default deterministic
  CPU groups derived from `notebooks/index.yaml`;
- all 15 default fixture notebooks: PASS under
  `/tmp/hypertagging-notebook-audit`, with 29 provenance-bearing JSON reports,
  48 figures, and visual review `NOT_REVIEWED`;
- first-level ambiguity diagnostic: PASS under
  `/tmp/hypertagging-first-level-audit`, with three provenance-bearing JSON
  reports and the actual model ablation changing pointer logits;
- `--check-generated`: PASS for all 18 tracked notebooks;
- `python -m compileall -q src scripts tests` and `git diff --check`: PASS.

The bounded real run processed 50 entries from one generic mDST and published
`/data/dust/user/boyangyu/hypertagging/audit-pilot-v4-20260801.parquet` plus its
metadata and completion marker. The real-only notebook passed and wrote
`/tmp/hypertagging-real-pilot-report-20260801.json`. It found zero truth-derived
detector inputs, zero p4 residual, and zero cycle/link/level failures. It also
found zero strict B-root events (50 fallback), missing recorded fit choice for
all 392 tracks, unavailable PIDLikelihood features for those tracks, and no KLM
provenance for 52 K_L-like leaves. Those shortcomings remain `PARTIAL`.

No trained held-out checkpoint was evaluated, no local CUDA training or
throughput profiling ran, and no HTCondor job was submitted. Archive numerical
results are not copied or rewritten here.
