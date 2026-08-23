# HyperTagging production-1M phase-3 execution authorization

Status: the corrected recovery lineage is explicitly authorized for one later execution, but no submission was performed and no new GPU pilot is claimed. The prior recovery report remains immutable and retains its deliberate `submission_authorized=false` verdict. This new versioned artifact records the separately authorized state under `authorization_basis=explicit_user_operator_instruction`.

The authorization is limited to the recovery lineage `ht-pretrain-1m-phase3-recovery-20260823`, requires execution by `gpt-5.3-codex-spark medium`, and does not change the structural validator. `structural_scientific_slurm_submission_allowed=false` remains true under the explicit provenance exception. A future allocation must pass a fresh, exact, in-allocation GPU preflight before trainer launch; the preflight is self-hashed and bound to both the authorization artifact and execution contract. A missing, stale, mismatched, or tampered preflight fails closed.

## Authorization state

- `submission_authorized`: `true`
- `authorization_basis`: `explicit_user_operator_instruction`
- `timestamp`: `2026-08-23T21:45:06+02:00`
- `gpu_pilot_completed`: `false`
- `fresh_in_allocation_preflight_required`: `true`
- `submission_performed`: `false`
- `execution_role_required`: `gpt-5.3-codex-spark medium`
- sealed-test/stress payload access: forbidden; no such payload was opened
- accelerator policy: H100 NVL or V100 by evidence-based earliest completion; never CPU

## Immutable lineage and bindings

| Binding | SHA256 / immutable identifier |
|---|---|
| Prior recovery MD | `a9639523e775b59a691abc29b1c0fae99be4f8f452c270f80a9a4decafa3bf7a` |
| Prior recovery JSON | `90e119c1fb121d9cc985d47a6914ac82a67ffc7ee85e6a28a750652d2caf3e75` |
| Failed job | `15933802`; attempt root remains protected |
| Exact resume checkpoint, step 54064 | `997241deb841033598846dea8b3650d31b9511c4241aad44798d83fe0ac5ad7d` |
| Corrected recovery config | `eced70932466ea07783122f7d2bce7fb344c87f45e72e57622e6363df3a2ad3f` |
| Recovery contract canonical/file | `20805cd37f914ea9ffb85789a200188bf23b1f6ee23e38067e5512f16393ac94` / `2dec2fc5c793230d9decde5f41a6b9e2c83cdc6b6237c1cd5a9145cd1f46857c` |
| Historical failed commit | `93b71c5d7c1bc20181640aafb4e918abb9267362` |
| Recovery implementation commit | `88b4fcdbd8bec2c1cd772c3e45742aa39ff077b7` |
| Prior artifact commit | `fcf19adf16b1492fc1c0478fc79fe358e13d1809` |

Preserved tags remain bound to those commits: `ht-pretraining-production-1m-h100-operator-authorized-20260821`, `ht-pretraining-1m-phase3-recovery-implementation-v2-20260823`, and `ht-pretraining-1m-phase3-recovery-20260823-final-v2`.

## Retained gates

- Objective dominance remains fail-closed at `20.0`. The late-phase `leaf_pid` weights remain `[1.0, 1.0, 0.4, 0.4]`; the projected exact ratio is `18.3152` from the observed `22.894` event. No threshold or fail action was weakened.
- Train/validation isolation remains bound to train and validation only, with test count zero. Sealed-test and stress roles remain forbidden.
- Resume is exact from checkpoint step `54064`; output is job-ID/attempt scoped, must not silently overwrite, and must not reuse `artifacts/slurm/jobs/15933802/attempt-00`.
- The curriculum remains four phases of `27032` optimizer steps. The eight fixed validation/checkpoint boundaries remain `13516`, `27032`, `40548`, `54064`, `67580`, `81096`, `94612`, and `108128`.
- Finite-gradient and finite-checkpoint gates remain enabled fail-closed. The preserved checkpoint was read-only loadable and finite.
- The structural provenance exception remains explicit and narrow: missing source commit `f4e54df23b5c60115e475c5d68df4651899d678e`, expected tree `b6e3a4118b960e3a4676a61af9601438d56cef96`, validator unchanged, and structural scientific submission status false.

## Machine-verification path

`schemas/ht_pretraining_1m_phase3_execution_authorization_v1.schema.json` defines the versioned shape. `scripts/slurm/phase3_execution_authorization_v1.py` verifies the artifact, all parent hashes, preserved tags, exact contract, retained gates, and the fresh preflight attestation. `scripts/slurm/verify_phase3_execution_authorization_v1.py` is the CLI verifier. The existing recovery contract verifier invokes the new static authorization check, while `train_one_gpu.sbatch` invokes the fresh in-allocation preflight and the authorization verifier before trainer launch. No Slurm command was submitted, cancelled, requeued, or mutated here.

The new authorization JSON canonical self-hash is `c952524ce32b1c504cc6210cc8bc540bb6180a928c145fe29109b89b3fe3b5e3`; its file SHA256 is `1af20420655a95aa7ce0a3d1ad4a6e357c7fe45510c3f8bafaf80ad3fdbb7991`.

## Validation and audit files

Focused CPU/static command:

`uv run pytest -q tests/test_phase3_execution_authorization_cpu.py tests/test_phase3_recovery_cpu.py tests/test_production_1m_operator_authorization_cpu.py tests/test_slurm_requeue_and_render_contract_cpu.py`

Result: `31 passed`; CPU training run: `false`; Slurm mutation: `false`.

Tracked implementation/support file hashes at this audit:

- `schemas/ht_pretraining_1m_phase3_execution_authorization_v1.schema.json`: `f4f1499309999fea2cb333ed881291023bae644afc35dbd21be24c5f57f8fc8b`
- `scripts/slurm/phase3_execution_authorization_v1.py`: `f1b36167a800d514cf087305e86502a6f590cb2fbb13a1fbdab652471af6f7e7`
- `scripts/slurm/verify_phase3_execution_authorization_v1.py`: `0fb0cb3d9bfa613a4e35e7e014f807f8f13888d7244906a70ba38c1f38ae989a`
- `tests/test_phase3_execution_authorization_cpu.py`: `0ed92db0182bedafad77c3dd8b7ad72c8a2e7c4b1e15921055e30f1518c950a2`
- `scripts/slurm/preflight_gpu_environment.py`: `7f82037a600d91f24f4b6b9e5358e2234bd21599ece06a246f6bb9ccc4439e07`
- `scripts/slurm/train_one_gpu.sbatch`: `82b8e27f4b3b7c43690230108584492fdc8693fb8f0a5c0b40c34c34b3d97099`
- `scripts/slurm/verify_job_contract.py`: `8148dfe3831517d2dff117ee1b7481d43189ff235fa3b74c191baba9437e9ad5`
- `scripts/slurm/render_one_gpu_job.py`: `d48a34b4fc7071c87138c40e49294e944a59009501b0acbdef301a945f6f3d42`
- `scripts/slurm/verify_phase3_recovery_contract.py`: `7f9e98ce982a31cf017c999b27a29670c5228b1366cc5c31432bead2c7ca87bb`

The delivery tag is `ht-pretraining-1m-phase3-execution-authorization-20260823-final`; the final clean commit and annotated-tag object are recorded in the handoff after commit creation. The authorization artifact itself is versioned separately and does not rewrite the prior report.

Canonical SHA256 of this Markdown audit, excluding this sentence: `a9cdae9ed6ae599b0e07f0442f31046aeeb2cfc076bc699d270108b0197fd8ff`.
