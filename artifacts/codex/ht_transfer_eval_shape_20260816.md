# Frozen-encoder transfer diagnosis, 2026-08-16

## Scope and evidence boundaries

- Diagnosed failed downstream Slurm job `15772685`, experiment
  `pretrain-035k-transfer-step3282-collationfix-study-20260816`.
- Treated every prior job artifact and checkpoint as immutable evidence.
- Did not query, cancel, modify, duplicate, or otherwise interact with active
  read-only pretraining validation jobs `15772672` and `15772673`.
- Did not access sealed-test data, submit a Slurm job, run local scientific
  training, or use a local GPU.
- Authorized data inspection was limited to the selected train and validation
  roles in dataset index
  `artifacts/experiment_readiness/production_1m_20260812/train_035k/train_035k.complete_only.index.json`
  (`index_hash=5fc837315b2e6f5e4783cba2808bfba7672cf4ba12bff1fe5cf050f5d22b6de1`).
  The saved split manifest reports `test: 0`, and the scan asserted every
  record's source role was either `train` or `validation`.

## Diagnosis 1: final-validation 40-vs-42 feature mismatch

The traceback in
`artifacts/slurm/pretrain-035k-transfer-step3282-collationfix-study-20260816-15772685.err`
ends in `CompositeNodeEncoder.pid_histogram` with
`mat1 (39x40)` and `mat2 (42x128)`. The saved downstream model confirms
`encoder.composite_encoder.pid_histogram.0.weight` has shape `(128, 42)`,
which is the required 41-token PID histogram plus one availability feature.

The exact cause was shape-based tensor classification in
`reconstruction.level_rollout._select_nodes`. When an event has 41 nodes,
the node feature `daughter_input_pid_histogram` also has shape `[1, 41, 41]`.
The old code mistook it for a pairwise node matrix and selected both axes.
Selecting the 39 level-zero leaves therefore produced `[1, 39, 39]`; adding
histogram availability yielded the observed width 40. Pairwise tensors are
now identified by their authoritative field names, while PID histograms are
selected only on the node axis and retain vocabulary width 41.

## Diagnosis 2: rare finite type-loss spikes

The 100 optimizer-step records have median total loss `3.1301745`; every
non-spike total loss is at most `6.6742191`. Exactly nine records have type-loss
spikes from `126.5259` to `626.7262`, at steps
`20, 46, 58, 65, 72, 73, 80, 90, 93`.

Deterministic replay of the read-only epoch-0 train stream (seed `20260815`,
shuffle buffer 1024, batch size 4) found exactly one complete target excluded
by `reduced-mother-ontology-v1` in each of those same nine batches. Eight are
token 3 (`PDG 130`, `K_L0`); step 73 is token 24 (`PDG 11`, electron). The
decoder filled those target logits with `-1e4`, after which matching and
`cross_entropy` consumed them. The one-for-one step match and approximately
`10000 / denominator` loss scale establish the causal link.

This is an ontology defect, not evidence for discarding targets. A complete
read-only scan of all authorized 35,000 train and 50,000 validation events
found the only static-mask mismatches were genuine complete composite nodes:

| Role | token 3 (`K_L0`) | token 24 (electron) |
| --- | ---: | ---: |
| train | 577 | 300 |
| validation | 683 | 499 |

Many `K_L0` nodes are direct daughters of otherwise-valid complete D/B mothers,
so filtering them would strand valid higher topology and weaken the study.
The dataset index already records tokens 3 and 24 as observed train mother
types. The static ontology is therefore versioned to
`reduced-mother-ontology-v2` and admits precisely those two evidenced tokens.
The level loss also now fails closed before matching if any truth target is
excluded by the effective decoder mask; it does not clip, suppress, or weaken
the loss.

## Changed files

- `src/hypertagging/reconstruction/level_rollout.py`: use named pairwise fields
  during rollout node selection.
- `src/hypertagging/preprocessing/pid_filter.py`: version the corrected static
  mother ontology and admit PDGs 130 and 11.
- `src/hypertagging/losses/level_reconstruction.py`: reject masked truth mother
  targets before matching/cross-entropy.
- `tests/test_transfer_eval_regressions_cpu.py`: exact CPU regressions for the
  41-node/39-leaf histogram collision, corrected target logits/losses, and the
  fail-closed invariant.
- `docs/training.md`: document composite `K_L`/electron target eligibility.
- `artifacts/codex/ht_transfer_eval_shape_20260816.md`: this audit record.

## Verification

Pytest is not installed in the pinned GPU-environment Python, so the repository's
self-running CPU test modules were used. All commands completed successfully
with `PYTHONPATH=src` and
`/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1/bin/python`:

```text
python -m py_compile \
  src/hypertagging/reconstruction/level_rollout.py \
  src/hypertagging/preprocessing/pid_filter.py \
  src/hypertagging/losses/level_reconstruction.py \
  tests/test_transfer_eval_regressions_cpu.py
python tests/test_transfer_eval_regressions_cpu.py
python tests/test_partial_target_policy_cpu.py
python tests/test_reconstruction_context_collation_cpu.py
python tests/test_current_head_reconstruction_corrections_cpu.py
python tests/test_revised_rollout_cpu.py
python tests/test_post_audit_geometry_rollout_cpu.py
git diff --check
```

The intended annotated tag is
`pretrain-035k-transfer-eval-shape-ontology-fix-20260816`.

## Remaining uncertainty and retry recommendation

No full downstream training or final rollout validation was run locally, by
design. The parent orchestrator may retry the same source-bound 100-step frozen
encoder probe from a new output directory under this tagged commit. The retry
should retain the existing dataset index and validation cohort, verify that no
type target reaches cross-entropy outside the effective mask, and require the
complete final teacher/predicted rollout validation and normal scientific
eligibility gates to pass. No Slurm submission was made here.
