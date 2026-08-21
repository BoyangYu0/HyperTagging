# HyperTagging 1M provenance and historical-failure audit — 2026-08-21

Decision: blocked pending exact external production provenance. The scientific
submission gate remains `submission_authorized=false`; no launch-authorized
contract is rendered from this audit.

## Baseline

The task began from a clean worktree at commit
`39abb57a9a7cde11072496a6e1a831231edbcfc7`, annotated as
`pretrain-035k-reconstruction-object12-pointer16-fullscale-readiness-20260817`.

## Preserved historical failures

Job `15762258` is terminally failed at attempted optimizer step 1155 after
1154 completed steps. Its immutable `nonfinite-gradient-step-1155.json` records
the H100 BF16 run, disabled GradScaler, the truth-guided distance/radius phase,
and nonfinite gradients in 119 parameters. The Slurm stderr records the
fail-closed `clip_grad_norm_(error_if_nonfinite=true)` exception. This is a
finite-gradient safety failure, not evidence for disabling the check.

The corrected path is present in commit `3305f9b`: smooth float32 tangent-norm
bounding (`max_tangent_norm=1.5`) addresses the stiff hyperbolic boundary path,
while the raw-gradient clip and persisted offending-parameter report remain
fail-closed. The full-scale configuration inherits that geometry and the CPU
tests assert the binding.

Job `15763038` is terminally failed during objective pilot preflight. Its
stderr explicitly records repeated zero-positive channel windows with
`channel_memory_size=0`, followed by
`channel:gradient_without_support, channel:loss_without_support`. The channel
objective was therefore not silently accepted as scientifically evaluable.

The repair chain is preserved: `5d22f14` permits only a bounded channel-memory
expansion at the exact channel-phase boundary, `f039869` counts structured
channel loss support in preflight, and `b857909` records the measured late-phase
leaf-PID taper. The new scientific guard rejects non-positive channel memory or
non-failing zero-positive handling before reading data; full scale binds memory
4096 and fail-closed objective actions.

## Exact provenance recovery attempt

The expected source commit is
`f4e54df23b5c60115e475c5d68df4651899d678e`; its expected tree is
`b6e3a4118b960e3a4676a61af9601438d56cef96`.

The configured `origin` was queried with an exact fetch. Git returned:

```
remote error: upload-pack: not our ref f4e54df23b5c60115e475c5d68df4651899d678e
```

Read-only searches of reachable history, reflogs, packed objects, alternates,
bundles, and additional Git repositories under the scoped workspace found no
matching commit or tree. The object cannot be independently verified. The
existing provenance validator itself runs and reports `status=valid`, but its
scientific readiness result is correctly false because this object is absent.

Concrete remediation: the production provenance owner must provide the exact
object, ref, or bundle. Import it only after `git cat-file` proves the exact
commit ID and `git rev-parse <commit>^{tree}` proves the exact tree ID. Do not
replace the object, alter the expected hashes, or bypass the validator.

The machine-readable record is
`artifacts/codex/ht_pretraining_1m_provenance_blocker_20260821.json`.
