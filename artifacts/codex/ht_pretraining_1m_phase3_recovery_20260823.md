# HyperTagging production-1M phase-3 recovery

Status: implementation and contract are complete; no Slurm mutation was performed. The current handoff sets `submission_authorized=false` because no replacement GPU pilot was run in this diagnosis turn. The contract retains the prior explicit operator execution exception for a later, separately verified H100 execution.

## Immutable failure evidence

- Failed Slurm lineage: job `15933802`, `ht-pretrain-production-1m-h100-20260821`, terminal `FAILED`, exit `1:0`, node `usm-cl-nv01`, `gpu:h100nvl:1`, `Restarts=0`, terminal `2026-08-23T11:01:50+02:00`. The attempt receipt reports trainer failure at `2026-08-23T09:01:49+00:00`, no USR1, no termination, and no restart.
- Historical source: commit `93b71c5d7c1bc20181640aafb4e918abb9267362`, tag `ht-pretraining-production-1m-h100-operator-authorized-20260821`.
- Historical contract: `artifacts/slurm/ht-pretrain-production-1m-h100-20260821.operator-authorized.job-contract.json`; canonical SHA256 `2af2b8fc51c7f1bceb26e5013c822967316a0f2b1d09671eb8b10fc0e8fd3406`; file SHA256 `8dfa6b2320c8992e69c68f7d570bcb0e562306b928be57c2ece0c8f8626f5a0d`.
- Historical config SHA256: `dd1947cdb50ffa0c4150ec5ab2a9032f4e6dab3d7f6a6c68bd2f245c2e43b8ad`.
- Selection manifest semantic/file SHA256: `fe250b707242377cbdfea5936a1b7ff160241fb3c223a37b4f3285613062dba2` / `989ac429d7d7cc444d8278b2f3443999999c481c4e202d249e57cdeb1d58a390`.
- Index semantic/file SHA256: `eba2a5498e09d82e1ca1e8d9841ba117423b99b540190088ed4875a8aa296809` / `8b307fe9c7c359b6c2d36908c393aa82fa13afca5ec53144311b5d592eed7be5`.
- Terminal exception: `RuntimeError: objective pilot preflight: weighted_gradient_dominance:leaf_pid/lca=22.894`. The configured ratio remained `20.0`, with `leaf_pid_phase_weights=[1.0,1.0,0.5,0.5]` and `lca_relation_weight=1.0`.
- Step-54064 validation: 5,000 events, 313 batches, full objective `8.664798735239254`, principal loss `8.494534350955448`, relation accuracy `0.8384483626570565`, parent ranking `0.6717389677755368`, leaf PID accuracy `0.8452743453720507`.
- Checkpoint: `artifacts/runs/ht-pretrain-production-1m-h100-20260821/20260812/15933802/checkpoint-step-54064.pt`, 19,371,763 bytes, SHA256 `997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d`.
- `metrics.jsonl` contains 2,176 parseable records and its maximum logged step is 54,275. The terminal gate was on an unlogged continuation batch; this is consistent with the diagnostic cadence and pending channel-support retry behavior, not a restart or overwrite.

The checkpoint loaded read-only on CPU. Its model state has 141 tensors (136 floating), optimizer state 387 floating tensors, and normalizer state 16 floating tensors / 200 elements; all floating and complex tensors in model, optimizer, scheduler, normalizer, training, and streaming state were finite. The cursor is at completed optimizer step 54,064, with epoch 1 / batch index 1 as the next stream position. BF16 and GradScaler were disabled in the historical run. No historical file was rewritten.

## Diagnosis

The checkpoint boundary is semantically correct. The saved curriculum cursor is phase index 1 after 54,064 completed steps. With four equal 27,032-step phases, the next optimizer iteration at cursor step 54,064 resolves to phase index 2, `multilevel_channel_memory`. Phase 3 activates LCA, parent/tree objectives, leaf PID, and channel memory; the channel objective can remain pending, so objective preflight is intentionally retried before the normal logging cadence.

The gate is measuring the intended quantity. Leaf PID uses mean cross-entropy over supported raw-track PID labels. LCA uses its balanced relation-loss reduction. The active denominators are support/finite checks; they do not renormalize the raw objective gradients. The dominance diagnostic uses absolute configured weight multiplied by the shared-encoder gradient norm, with LCA as the reference. Therefore the terminal `22.894` is a weighted, shared-representation dominance event, not a denominator artifact, nonfinite event, or validation-role mix-up.

Prior source-bound evidence in `artifacts/codex/ht_channel_objective_diagnosis_20260815.md` documented an earlier measured `20.1182` event and the already-adopted late-phase taper to `0.5`; it explicitly concluded that the gate must not be raised. The new terminal event shows that `0.5` is insufficient for this later deterministic sample-conditioned batch. The fixed seed, exact checkpoint cursor, zero restart count, and continuous metrics support a deterministic continuation explanation. No sealed-test or stress payload was opened, and no claim is made about the unavailable batch contents beyond the logged exception.

Classification: deterministic sample/gate behavior, not a code defect. The phase transition, denominators, and fail-closed objective gate are behaving as designed. The historical late-phase weight was scientifically too large for the observed shared-encoder gradient imbalance.

## Smallest correction

`configs/slurm/pretrain_1m_phase3_recovery_20260823.yaml` is identical to the historical production config except for `leaf_pid_phase_weights: [1.0, 1.0, 0.4, 0.4]`; its SHA256 is `eced70932466ea07783122f7d2bce7fb344c87f45e72e57622e6363df3a2ad3f`. The dominance threshold remains `20.0`, the violation action remains `fail`, and all finite checks remain active. Because the diagnostic scales the leaf contribution linearly, the measured event maps to `(0.4 / 0.5) * 22.894 = 18.3152`, leaving approximately 1.0926x threshold headroom without disabling the objective or altering LCA/validation weighting.

The replacement contract uses experiment `ht-pretrain-1m-phase3-recovery-20260823`, `initialization_policy=exact_resume_from_checkpoint`, and the exact step-54064 source checkpoint above. Its output templates remain job-ID/attempt scoped and explicitly prohibit silent overwrite; its recovery lineage names historical job 15933802 and states that the replacement attempt root must not be `artifacts/slurm/jobs/15933802/attempt-00`.

Expected continuation: 54,064 completed steps are resumed; the next optimizer step is phase 3 (`multilevel_channel_memory`), followed by phase 4 (`corrupted_composites_hard_negatives`) at step 81,096. The full fixed schedule remains 108,128 optimizer steps with 27,032 steps per phase. All eight validation/checkpoint boundaries are contract-bound, each with both validation and checkpoint expectations: 13,516; 27,032; 40,548; 54,064; 67,580; 81,096; 94,612; and 108,128. The first four are historical evidence; the resumed execution must preserve the remaining four.

## Validation performed

- `uv run pytest -q tests/test_trainer_corrections_cpu.py tests/test_phase3_recovery_cpu.py tests/test_slurm_requeue_and_render_contract_cpu.py tests/test_production_1m_operator_authorization_cpu.py`: 64 passed.
- Focused test reproduces the old 0.5-weight failure and verifies 0.4 passes at 18.3152 with the threshold still 20.0.
- Read-only checkpoint load, tensor finiteness, cursor/phase semantics, historical metrics, receipt restart fields, config-diff, output isolation, sealed-test role, and renderer/verifier bindings passed.
- The renderer performed only read-only Slurm inventory checks and emitted a command; `submission_performed=false`. No training was run on CPU and no Slurm job was submitted, cancelled, requeued, or mutated.

## Immutable replacement identifiers

- Implementation commit: `88b4fcdbd8bec2c1cd772c3e45742aa39ff077b7`.
- Implementation tag: `ht-pretraining-1m-phase3-recovery-implementation-v2-20260823`.
- Final artifact tag to be created on the clean artifact commit: `ht-pretraining-1m-phase3-recovery-20260823-final`.
- Replacement contract: `artifacts/slurm/ht-pretrain-1m-phase3-recovery-20260823.operator-authorized.job-contract.json`.
- Replacement contract canonical SHA256: `20805cd37f914ea9ffb85789a200188bf23b1f6ee23e38067e5512f16393ac94`.
- Replacement contract file SHA256: `2dec2fc5c793230d9decde5f41a6b9e2c83cdc6b6237c1cd5a9145cd1f46857c`.
- Submission authorized: `false` for this handoff. Scientific uncertainty remaining: no new GPU pilot was run, so the later operator must pass fresh in-allocation GPU preflight and all objective/finite/milestone gates. The existing operator provenance exception remains recorded and does not change `scientific_slurm_submission_allowed=false`.
