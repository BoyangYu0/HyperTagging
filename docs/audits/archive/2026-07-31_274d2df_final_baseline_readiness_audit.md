# Final baseline readiness audit

Audit date: 2026-07-31 (Europe/Berlin)

## Starting state

- Audited revision and initial current HEAD: `274d2df8a9df8b25142b68966dfe30c828538b90` (`code audit`).
- Branch: `master`, tracking `origin/master`.
- Initial worktree: clean.
- Documented Python: `/data/dust/user/boyangyu/uv_env/bin/python`.
- Initial unit suite: `183 passed, 8 skipped, 18 warnings in 389.01s` from `/data/dust/user/boyangyu/uv_env/bin/python -m pytest -q`.
- Initial deterministic notebook suite: all 12 notebooks executed successfully on CPU fixtures. Retained executed copies are under `/tmp/hypertagging-final-baseline-notebook-smoke` for this host session.
- GitHub Actions: `.github/workflows/cpu-tests.yml` runs the complete unit suite and a selected deterministic notebook suite on Python 3.11. A current-head push run is available: `CPU correctness` run 30617205526 completed successfully for the exact starting SHA on 2026-07-31.

## Confirmed remaining correctness issues

The baseline is test-green but does not yet meet the requested production-readiness contract:

- The stored v3/v4 composite block contains truth multiplicity/completeness positions and the encoder consumes the complete block.
- Scheduled target alignment drops unrepresentable mothers, which can turn a non-empty truth level into implicit all-no-object supervision.
- Teacher-forcing schedule progress uses `optimizer_step * 100 + target_level`.
- Training, scheduled contexts, validation, and rollout do not share one serialized reconstruction-constraint policy; observed type frequencies are hard masks.
- Scheduled predicted rollouts restart from level zero for every target level, and leaf-PID loss is consequently repeated by level.
- Dataset-index loading trusts stored hashes and has incomplete source/publication/selection/target-policy validation.
- Resume metadata does not yet freeze every data-order-defining setting and the saved epoch semantics are incomplete.
- Validation is event-at-a-time despite `validation_batch_size`, mixes macro and micro aggregation, uses subtree exact match as complete-target efficiency, and emits unbounded channel-pair metric keys.
- Channel tensors do not yet carry all requested depth/multiplicity/intermediate/signature fields through the heterogeneous contract.
- Real training does not uniformly require parquet + sidecar + valid completion marker, and overwrite marker invalidation is not failure-safe.
- Experimental v5 inference needs an explicit feature-spec-derived Arrow schema and expanded benchmark accounting.
- Architecture parameters/presets and encoder transfer coverage are not fully resolved and checkpoint-enforced.
- Pretraining needs a final audit of runtime two-pass PID semantics, corrupted-node structural masks, and invalid/fallback B-root channel masks.

## Planned change surface

The focused implementation is expected to touch the v4 feature specification and adapters; heterogeneous encoders/collation/normalization; scheduled sampling, rollout, reconstruction loss/trainer/validation; dataset index/data module; checkpointing and transfer; pretraining curriculum/loss/trainer; channel preprocessing; v4/v5 publication and benchmark code; resolved configs/CLI; deterministic notebook generators; and corresponding CPU tests and documentation.

## Compatibility strategy

- Preserve the on-disk v1/v2/v3/v4 readers and native schema-v4 truth-clean raw-track contract.
- Keep existing schema-v4 files valid. A versioned model-feature adapter will select/mask runtime composite positions instead of requiring regeneration.
- Keep legacy field names in stored records and target/diagnostic tensors while preventing those tensors from reaching any model representation path.
- Add defaults matching the current single-kernel behavior where safe; correctness-sensitive new defaults are `fallback_teacher`, `complete_only`, strict publication validation for real training, and experimental/default-off v5.
- Store new policies/contracts in checkpoints and reject incompatible exact resume or architecture loads rather than guessing.

## Verification boundary

Verification is restricted to deterministic CPU fixtures, tiny parquet datasets, unit/integration tests, notebook smoke execution, read-only GitHub Actions inspection, and dry-run command construction. No HTCondor submission, long training, ten-million-event production, production-file overwrite, or physics-performance claim is in scope. A real basf2 pilot, if run later by an operator, must contain fewer than 100 events.
