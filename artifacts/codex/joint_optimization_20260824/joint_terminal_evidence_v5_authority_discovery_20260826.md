# Joint terminal evidence v5 authority discovery

Version: `joint-terminal-evidence-v5-authority-discovery-v1`

Date: 2026-08-26

Scope: tracked config, code, contracts, and reports only. No scheduler calls, payload access, tests, or scientific execution.

## Outcome

| Item | Finding | v6 disposition |
|---|---|---|
| A01 pretraining contract/hash | Authoritative and compatible | Close |
| A02 pretraining resources/TRES/runtime | Authoritative for contracted semantic fields | Close with semantic TRES matching |
| A03 teacher direction/margin table | Missing | New reviewed scientific decision |
| A04 secondary guard table | Missing; nearby calibration preregistration is incompatible | New reviewed scientific decision |
| A05 efficiency definitions/thresholds | Partial: formula, thresholds, and caps exist; measurement semantics do not | Partial close, then new reviewed measurement decision |

The sealed decision plan is commit `b10c65e02e36dcb5878d3aa932adc559d5676a70`, tag `ht-reconstruction-post-terminal-decision-plan-v1-20260826`, path `artifacts/codex/joint_optimization_20260824/reconstruction_post_terminal_exact_decision_plan_v1_20260826.json`, SHA-256 `45747d32d5af2682f712c399d23f0a8f04f5f137757b93cc0664d4f5e556dfe2`.

## A01 — authoritative pretraining contract/hash

The exact job-16028845 execution authority is the transition contract at commit `0c728e7c37ec933f5aad8c724e295e8788baa72a`, tag `ht-pretraining-balanced-resume-attempt9r4r23-transition-20260826-v1`:

- `artifacts/codex/ht_pretraining_balanced_resume_contract_attempt9r4r23_transition_20260826.json`, file SHA-256 `b7be36440996bbb12e842c0cce7bc748ff3779a341873e7604cb2d57c7631e56`.
- `scripts/slurm/run_pretraining_balanced_resume_attempt9r4r23_transition.sbatch`, SHA-256 `df31111644e38f8abdadf12cd39805fe8aed4e1426437109dafd3edcfc6720f4`.
- `configs/slurm/pretrain_capacity_calibration_balanced_resume_attempt9r4r23_20260826.yaml`, SHA-256 `0d5a8556be87ba32e21e6b9ed9f6f0ce9b5ac45a0e32260f6be18afa0eb27db1`.

The executed receipt at commit `98ba2a0a7dfd6bda50da035a32cd2dfb61da9018`, tag `ht-pretraining-balanced-resume-attempt9r4r23-submission-20260826-v1`, path `artifacts/codex/ht_pretraining_balanced_resume_attempt9r4r23_submission_receipt_20260826.json`, SHA-256 `b83ef816ffc6ea245a892fdb7cd86e255af261f8a79fbaef4977230023f2963b`, binds job `16028845`, its exact name, transition commit/tag, contract/config/wrapper hashes, and exact sbatch argv.

The contract's embedded `contract_sha256` is `a50408eee13132d7a2338ea1b3f38d9ffa13fecc49d8ade510e504cbf4a891a1`. It is not the file hash. The tagged verifier defines it as SHA-256 over sorted compact JSON after removing the `contract_sha256` member. A v6 verifier can close A01 only by keeping these two digest domains separately named and validated.

## A02 — exact pretraining resource/TRES/runtime authority

The same tagged transition fixes: partition `inter`; `gpu:h100nvl:1`; one GPU; 8 CPUs per task; `64G`; `06:00:00`; no requeue; zero restarts; no DDP; no two-GPU comparison. Required semantic TRES are `cpu=8`, `mem=64G`, `gres/gpu=1`, and `gres/gpu:h100nvl=1`.

It also fixes Python `3.11.11`, Torch `2.7.1+cu126`, CUDA build `12.6`, `uv 0.5.20`, and `uv run --project <exact root> --frozen --no-sync python`, with these hashes: pyproject `85cf6a6154f4569123a059d22750ee6284cc778d30ae52918c66553a0bf6fe03`; Python pin `49a506dd32096b010d75205acf3430c9ae6c40351888129499e5a5e487126c93`; uv.lock `acdbd38d7c0233cb85cb1db5806bb33df3c99a6228c2aa1d2316a58539111917`; uv executable `ae859d6c2102b58381e9d508fc754ad5483c994aa1060da3fe8e29f4f9c70541`; venv Python `4f88a74c1b135771da9e4d6bce3d47713025c96944b24646d4dd5379152b246c`.

No tracked authority fixes scheduler-added serialized TRES fields such as billing or node count for job 16028845. A v6 verifier may compare the contracted semantic components, but must not invent an exact full TRES string.

## A03 — teacher direction/margin table

Missing. The sealed plan explicitly requires a separately sealed direction/margin table for teacher-forced accuracy, precision, recall, calibration, and loss endpoints, but supplies neither endpoint keys nor directions/margins.

The closest compatible Stage-A comparator is `paired_stage_a_compare.py` at r8 authorization commit `d7068ef529cd3c07764e545f0c83c700d783bdeb`, tag `ht-reconstruction-paired-stage-a-v2r3r2r8-final-authorization-20260826-v1`, path `reconstruction/snapshots/recon_stage_a_paired_q32_h100nvl_20260826_v2r3r2_atomic_restart_evidence_clearance/runtime/scripts/slurm/paired_stage_a_compare.py`, SHA-256 `74cdf5e97a3cdf64c7f1d1fb74e3f87858adbc23ba9646b8a28e673fc7402eae`. It evaluates aggregate rollout edge F1 and structural invariants, not a teacher endpoint table. A03 requires a new reviewed scientific decision.

## A04 — secondary guard table

Missing. The sealed plan requires `secondary_metrics_nonregressed=true` but supplies no endpoint set, direction table, margins, or aggregation rule.

`artifacts/codex/ht_positive_weight_calibration_preregistration_20260816.json`, introduced by commit `ee0648f972f7da469f61d5b4f547d2b4633b010a`, SHA-256 `1f99b05afd193ca7d20361b596ab5ff831da7ae5ca35058913bf07b5e3179801`, is not compatible authority: it covers a different object/pointer positive-weight calibration and ranking criteria, not matched q32-versus-relbias Stage A. A04 requires a new reviewed scientific decision.

## A05 — efficiency definitions and thresholds

Partially authoritative. The sealed plan fixes complete-target efficiency nonregression, validation events/sec at least `0.9` of q32, peak memory within contract, and walltime within contract. The exact complete-target numerator/denominator formula is implemented in `hierarchical_metrics.py` at r8 authorization commit/tag above, path `reconstruction/snapshots/recon_stage_a_paired_q32_h100nvl_20260826_v2r3r2_atomic_restart_evidence_clearance/runtime/src/hypertagging/evaluation/hierarchical_metrics.py`, SHA-256 `6a8c3cc51907b68039816a9988284917b5e10263f7209f3e9693ee5d90080dff`.

The q32 and relbias job contracts at the same commit fix one `gpu:h100nvl:1`, 8 CPUs, `64G`, `02:00:00`, no requeue, and zero restarts. Their paths/hashes are:

- `reconstruction/snapshots/recon_stage_a_paired_q32_h100nvl_20260826_v2r3r2_atomic_restart_evidence_clearance/job_contract.json`, `7298ddde4d7a0458dfda89f11e72e038ef4ed0146929cacfbdfa860ff5a653fe`.
- `reconstruction/snapshots/recon_stage_a_paired_relbias_h100nvl_20260826_v2r3r2_atomic_restart_evidence_clearance/job_contract.json`, `73142dc1dbbc4d96ef9989fb25d3333dd566df18979793161e5f82576cd333b7`.

The remaining semantics are not authoritative: no source selects validation-throughput numerator/timer/window/warmup/aggregation; peak-memory domain among sampled `nvidia-smi memory.used`, CUDA allocated/reserved peaks, and scheduler MaxRSS; or walltime origin/interval among scheduler, wrapper, and scientific-runner elapsed values. Existing training throughput and telemetry code expose possible measurements, but the plan does not choose them. A v6 spec can close the complete-target formula, numeric ratio, and resource caps, while the three operational measurement definitions require a new reviewed decision and must remain fail-closed.

## Authorization

`execution_authorized=false`, `submission_authorized=false`, `scientific_execution_authorized=false`, `scheduler_calls_authorized=false`, and `payload_access_authorized=false`. Audit counters: zero scheduler calls, zero payload reads, zero tests, and zero scientific runs.
