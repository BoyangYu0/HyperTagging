# Reconstruction Stage-A Paired Relation-Bias Terminal Audit

Audit date: 2026-08-27. Scope: relation-bias primary 16036157/controller 16036158; aggregate comparison against q32 primary 16036155/controller 16036156.

## Verdict

COMPLETED_NORMALLY. Recommendation: DO_NOT_PROMOTE. submission_authorized=false.

Both primary allocations exited 0:0 with zero restarts and zero requeue. Relation bias has finite training and aggregate validation, but evidence is exploratory and aggregate-only: no controller-bound per-event paired record, no Holm-adjusted paired confidence interval, and no complete structural/teacher-forcing/efficiency guard set. Edge F1 is 0.039 versus q32 0.031, delta +0.008, below required +0.010. No promotion authorization is granted.

No scientific code was modified. No submission, cancellation, CPU scientific training, or sealed/stress/restricted payload access occurred. Prior qualifiers and artifacts remain preserved.

## Immutable binding and runtime

- Clearance worktree: /home/b/Boyang.Yu/HyperTagging_uni/HyperTagging_reconstruction_hpo_clearance_20260825
- Branch: codex/reconstruction-hpo-clearance-20260825
- Pinned tag: ht-reconstruction-stage-a-paired-rerun-20260827-v2r5r8-pinned-tag-tree-identity-clearance
- Tag commit: 67c5b88d53ef0b9416179bce6833ed7339cf017a
- Snapshot trees: relation-bias 23dcb759eae4358cba7225ad11795bc022ce0091; q32 89d56b16a8e9ae9b7b466e915c7b97ff87a70411
- Relation-bias contract SHA256: d1ae572d7509cf64d535d0bcd2086a4a69234947ba8cc549790a2d0b2895700b
- Shared manifest SHA256: 350e437cc2094e2ca4c5662eab69c52a651a54c778414efb3f23df6a409fc3d3
- Runtime: uv 0.5.20 at /home/b/Boyang.Yu/.local/bin/uv; Python3.11.11; torch2.7.1+cu126; CUDA build12.6; uv.lock SHA256 7a18fbd4feed4371fa8e8a740f87720462d58c3a8e283402870f375ab744ad18.

## Allocation, checks, and artifacts

| Arm | Job | Result | Elapsed | Allocation | Restart/requeue |
|---|---:|---|---|---|---|
| q32 | 16036155 | COMPLETED/0:0 | 01:08:21 | H100-NVL, cpu8, mem64G, 2h | 0/false |
| relation bias | 16036157 | COMPLETED/0:0 | 01:13:02 | H100-NVL, cpu8, mem64G, 2h | 0/false |

Relation-bias source checkpoint is step 54064, SHA256 997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d. It ran 512 optimizer steps/32768 presentations; batch64, LR0.001, BF16, scaler off, frozen encoder and leaf-PID head, object/pointer weights12/16, relation-bias enabled, validation2000/rollout1000. Fresh in-allocation exact gpu:h100nvl:1, H100 NVL, CUDA device0, uv runtime, and finite GPU tensor probe passed. It produced 512 finite rows, zero bad rows, first-20 passed; stdout recorded contract/preflight/trainer/result/telemetry completion; stderr empty.

Hashes: result a236779d4a050a85763773dc790d108d9439157f799804e4c628b9ae65402f20; final checkpoint 4e29a61338425805e3d88096ec4cf6c572a14038bc5131b5cc95da52709b7cc0; best checkpoint 76439a781b143ff010a7d8badc3c214cbcd68123f4570b5613603b3e584a4e79; metrics 5fb42a923679c310f28318462506c4ce1ce25b96445274125d3c550392ae2dce; GPU preflight 9c0571fe1976538da2fbf56f0b9ec195210de928a91566aff4804d6f3e474225; uv/GPU probe fbaa59787ec5650fdf67ad80d62773015d724ad18b98efd0b00bcd4c38ccc51b; telemetry 3306f949b6fe94d5a8cb32b1ba48a141db8ff7fb9956e6221eb56120a88f4110.

Controller 16036158 completed 0:0 in 4 seconds with normal classification. Controller JSON SHA256 b9cc445fcda0d93d03bc1980e341f6c6ea5edc0d8ba271dc7a6a2b85aee0db13; terminal receipt SHA256 e3a81927f81316bdb3b799bf99cb666ba66987f2b17d8716438263b585ef73c0; reconciliation SHA256 802031a1739da0e8c612555d229de21efa7143d37d0fde08a6dbb36c922eafe2. Reconciliation records promotable_terminal_evidence=false.

Telemetry captured 291 samples at 15-second cadence, peak GPU utilization39%, memory5123 MiB, temperature38 C, no error. q32 telemetry was 273 samples, peak utilization39%, memory5045 MiB, temperature38 C.

## Validation-only comparison

| Metric | q32 16036155 | Relation bias 16036157 | Delta |
|---|---:|---:|---:|
| Edge F1 | 0.031 | 0.039 | +0.008 |
| Full-tree exact match | 0.031 | 0.039 | +0.008 |
| Tree validity | 1.000 | 1.000 | 0.000 |
| Canonical subtree | 0.888916 | 0.888916 | 0.000 |
| Pointer precision | 0.184022 | 0.184872 | +0.000850 |
| Pointer recall | 0.432027 | 0.460616 | +0.028590 |
| Validation loss | 1.819515 | 2.461554 | +0.642040 (worse) |
| Query utilization | 0.042750 | 0.041516 | -0.001234 |

Both arms used 2000 validation events and 1000 rollouts. Results are aggregate-only, not controller-bound paired per-event evidence; no Holm-adjusted CI or guard suite is available. Preserve all runs; keep submission_authorized=false; do not promote relation bias, authorize HPO, or authorize a final full run. Any future attempt requires per-event paired evidence, complete guards, and edge-F1 delta at least +0.010 with Holm-adjusted CI lower bound greater than zero.
