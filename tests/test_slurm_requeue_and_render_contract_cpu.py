from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm.verify_job_contract import (  # noqa: E402
    validated_runtime_values,
    verify_contract,
    verify_hashed_inputs,
    verify_rendered_contract_hash,
)
from scripts.slurm import render_one_gpu_job  # noqa: E402
from scripts.slurm.verify_execution_receipt import verify_receipt  # noqa: E402


def test_contract_verifier_bootstrap_does_not_import_torch():
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            (
                "import importlib.util,sys;"
                "spec=importlib.util.spec_from_file_location('verify',"
                "'scripts/slurm/verify_job_contract.py');"
                "module=importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(module);"
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_requeue_wrapper_never_requeues_outside_slurm(tmp_path):
    trainer = tmp_path / "trainer.sh"
    trainer.write_text("#!/usr/bin/env bash\nexit 75\n")
    trainer.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "scripts/slurm/run_with_bounded_requeue.sh",
            "--max-restarts",
            "2",
            "--",
            str(trainer),
        ],
        cwd=ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SLURM_")
        },
        check=False,
    )
    assert result.returncode == 75


def test_requeue_wrapper_is_bounded_before_scontrol(tmp_path):
    trainer = tmp_path / "trainer.sh"
    trainer.write_text("#!/usr/bin/env bash\nexit 75\n")
    trainer.chmod(0o755)
    forbidden = tmp_path / "scontrol"
    forbidden.write_text("#!/usr/bin/env bash\nexit 99\n")
    forbidden.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "scripts/slurm/run_with_bounded_requeue.sh",
            "--max-restarts",
            "2",
            "--scontrol",
            str(forbidden),
            "--",
            str(trainer),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SLURM_JOB_ID": "12",
            "SLURM_RESTART_COUNT": "2",
        },
        check=False,
    )
    assert result.returncode == 76


def test_slurm_templates_forbid_generic_gres_and_submission_side_effects():
    renderer = (ROOT / "scripts/slurm/render_one_gpu_job.py").read_text()
    sbatch = (ROOT / "scripts/slurm/train_one_gpu.sbatch").read_text()
    assert "ALLOWED_SLURM_GRES" in renderer
    assert '"gpu:h200nvl:1"' not in sbatch
    assert "--gres=gpu:1" not in renderer + sbatch
    assert '"submission_performed": False' in renderer
    assert "def submit" not in renderer
    assert '"--export=NIL"' in renderer
    assert '"--export=NONE"' not in renderer
    assert '"--export=ALL"' not in renderer
    assert '"scripts/slurm/train_one_gpu.sbatch",' in renderer
    assert 'SLURM_SUBMIT_DIR' in sbatch
    assert 'dirname "${BASH_SOURCE[0]}"' not in sbatch
    assert 'artifacts/slurm/jobs/${SLURM_JOB_ID}' in sbatch
    assert 'finalize_execution_receipt.py' in sbatch
    for ambient in (
        "HYPERTAGGING_GPU_ENV",
        "HYPERTAGGING_EXPECTED_GRES",
        "HYPERTAGGING_TRAIN_CONFIG",
        "HYPERTAGGING_EXPERIMENT",
        "HYPERTAGGING_SEED",
        "HYPERTAGGING_MAX_RESTARTS",
    ):
        assert ambient not in sbatch
    for token in (
        '"status", "--porcelain"',
        "event_identity_validation",
        "hashed_inputs",
    ):
        assert token in (ROOT / "scripts/slurm/verify_job_contract.py").read_text()


def test_prologue_contract_and_input_hashes_fail_on_mutation(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}\n")
    contract = {
        "contract_version": "hypertagging-slurm-one-gpu-contract-v2",
        "hashed_inputs": [
            {
                "path": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    verify_rendered_contract_hash(contract)
    verify_hashed_inputs(contract["hashed_inputs"], root=tmp_path)
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="hashed job input changed"):
        verify_hashed_inputs(contract["hashed_inputs"], root=tmp_path)
    contract["contract_version"] = "tampered"
    with pytest.raises(RuntimeError, match="contract hash mismatch"):
        verify_rendered_contract_hash(contract)


def test_runtime_values_are_contract_bound_and_shell_constrained():
    contract = {
        "gpu_environment": "/frozen/gpu-env",
        "gres": "gpu:v100:1",
        "train_config": "configs/slurm/pretrain_diagnostic.yaml",
        "experiment": "safe-experiment",
        "seed": 20260812,
        "max_restarts": 2,
    }
    assert validated_runtime_values(contract)["seed"] == "20260812"
    for field, mutation in (
        ("gres", "gpu:1"),
        ("train_config", "../outside.yaml"),
        ("experiment", "bad; touch submitted"),
        ("max_restarts", 11),
        ("seed", -1),
    ):
        with pytest.raises(RuntimeError):
            validated_runtime_values({**contract, field: mutation})


def test_renderer_only_writes_contract_and_prints_exact_sanitized_command(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "job-contract.json"
    monkeypatch.setattr(
        render_one_gpu_job,
        "validate_live_slurm",
        lambda gres: {"exact_gres": gres, "version": "slurm 23.02.8"},
    )
    monkeypatch.setattr(
        render_one_gpu_job,
        "_run",
        lambda command: "41da0a2" if command[:2] == ("git", "rev-parse") else "",
    )
    monkeypatch.setattr(
        render_one_gpu_job.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("renderer attempted a submit side effect"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_one_gpu_job.py",
            "--mode",
            "diagnostic",
            "--gres",
            "gpu:v100:1",
            "--output",
            str(output),
        ],
    )
    assert render_one_gpu_job.main() == 0
    rendered = json.loads(capsys.readouterr().out)
    command = rendered["sbatch_command"]
    assert command.count("--gres=gpu:v100:1") == 1
    assert "--export=NIL" in command
    assert "--export=NONE" not in command
    assert "--export=ALL" not in command
    assert command[-2] == "scripts/slurm/train_one_gpu.sbatch"
    assert command[-1] == str(output.resolve())
    assert Path(command[-1]).is_absolute()
    contract = json.loads(output.read_text())
    assert contract["submission_performed"] is False
    assert contract["export_policy"] == "NIL"
    assert contract["seed"] == 20260812
    assert contract["max_restarts"] == 2
    assert contract["submission_authorized"] is True


def test_live_inventory_selects_h100nvl_before_v100_when_h200_is_absent(
    monkeypatch,
):
    def fake_run(command):
        if command[:2] == ("/opt/slurm/bin/sbatch", "--version"):
            return "slurm 23.02.8\n"
        if command[:2] == ("/opt/slurm/bin/sbatch", "--help"):
            return "--account --partition --gres --signal --requeue --export"
        if command[:2] == ("/opt/slurm/bin/sinfo", "-h"):
            return (
                "usm-cl-nv01|idle|gpu:h100nvl:7(S:0-6)\n"
                "th-cl-nv01|idle|gpu:v100:2(S:0-1)\n"
            )
        if command[:3] == ("/opt/slurm/bin/scontrol", "show", "partition"):
            return "PartitionName=inter State=UP\n"
        if command[:3] == ("/opt/slurm/bin/sacctmgr", "-nP", "show"):
            return "boyang.yu|others|inter|\n"
        raise AssertionError(command)

    monkeypatch.setattr(render_one_gpu_job, "_run", fake_run)
    live = render_one_gpu_job.validate_live_slurm("gpu:h100nvl:1")
    assert live["usable_priority"] == ["gpu:h100nvl:1", "gpu:v100:1"]
    with pytest.raises(RuntimeError, match="violates live accelerator priority"):
        render_one_gpu_job.validate_live_slurm("gpu:v100:1")


@pytest.mark.parametrize(
    "provided_flag", ("--local-admission-receipt", "--local-completion-receipt")
)
def test_scientific_renderer_requires_both_bound_receipts(
    monkeypatch, tmp_path, provided_flag
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_one_gpu_job.py",
            "--mode",
            "scientific",
            "--gres",
            "gpu:h200nvl:1",
            provided_flag,
            str(tmp_path / "receipt.json"),
            "--output",
            str(tmp_path / "contract.json"),
        ],
    )
    with pytest.raises(RuntimeError, match="requires both receipt paths"):
        render_one_gpu_job.main()


def test_scientific_renderer_validates_binds_and_hashes_both_receipts(
    monkeypatch, tmp_path, capsys
):
    admission = tmp_path / "admission.json"
    completion = tmp_path / "completion.json"
    admission.write_text('{"admission":true}\n')
    completion.write_text('{"completion":true}\n')
    gpu_env = tmp_path / "gpu-env"
    (gpu_env / "bin").mkdir(parents=True)
    (gpu_env / "bin" / "python").write_text("")
    validated = {}
    monkeypatch.setattr(
        render_one_gpu_job,
        "load_local_microtest_completion_receipt",
        lambda path, *, admission_path: validated.update(
            completion=path, admission=admission_path
        ),
    )
    monkeypatch.setattr(
        render_one_gpu_job,
        "validate_live_slurm",
        lambda gres: {"exact_gres": gres, "version": "slurm 23.02.8"},
    )
    monkeypatch.setattr(
        render_one_gpu_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "scientific_slurm_submission_allowed": True,
                    "blockers": [],
                }
            ),
        ),
    )
    output = tmp_path / "scientific-contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_one_gpu_job.py",
            "--mode",
            "scientific",
            "--gres",
            "gpu:h200nvl:1",
            "--expected-git-sha",
            "a" * 40,
            "--expected-git-tag",
            "immutable-test-tag",
            "--gpu-env",
            str(gpu_env),
            "--local-admission-receipt",
            str(admission),
            "--local-completion-receipt",
            str(completion),
            "--output",
            str(output),
        ],
    )
    assert render_one_gpu_job.main() == 0
    contract = json.loads(output.read_text())
    assert validated == {
        "admission": admission.resolve(),
        "completion": completion.resolve(),
    }
    assert contract["local_admission_receipt"] == str(admission.resolve())
    assert contract["local_completion_receipt"] == str(completion.resolve())
    receipt_inputs = {
        item["path"]: item["sha256"] for item in contract["hashed_inputs"]
    }
    assert (
        receipt_inputs[str(admission.resolve())]
        == hashlib.sha256(admission.read_bytes()).hexdigest()
    )
    assert (
        receipt_inputs[str(completion.resolve())]
        == hashlib.sha256(completion.read_bytes()).hexdigest()
    )
    command = json.loads(capsys.readouterr().out)["sbatch_command"]
    assert "--gres=gpu:h200nvl:1" in command
    assert "--export=NIL" in command
    assert contract["submission_performed"] is False
    assert contract["export_policy"] == "NIL"


def test_explicit_user_authorization_retains_provenance_blocker_and_h100_label(
    monkeypatch, tmp_path, capsys
):
    admission = tmp_path / "admission.json"
    completion = tmp_path / "completion.json"
    admission.write_text('{"admission":true}\n')
    completion.write_text('{"completion":true}\n')
    gpu_env = tmp_path / "gpu-env"
    (gpu_env / "bin").mkdir(parents=True)
    (gpu_env / "bin" / "python").write_text("")
    monkeypatch.setattr(
        render_one_gpu_job,
        "load_local_microtest_completion_receipt",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        render_one_gpu_job,
        "validate_live_slurm",
        lambda gres: {
            "exact_gres": gres,
            "usable_priority": ["gpu:h100nvl:1", "gpu:v100:1"],
            "selection_reason": "H200 absent; exact H100-NVL is usable",
        },
    )
    monkeypatch.setattr(
        render_one_gpu_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "scientific_slurm_submission_allowed": False,
                    "blockers": ["missing source object/tree"],
                }
            ),
        ),
    )
    output = tmp_path / "authorized-scientific-contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_one_gpu_job.py",
            "--mode",
            "scientific",
            "--gres",
            "gpu:h100nvl:1",
            "--gpu-env",
            str(gpu_env),
            "--expected-git-sha",
            "a" * 40,
            "--local-admission-receipt",
            str(admission),
            "--local-completion-receipt",
            str(completion),
            "--user-authorized-scientific-submit",
            "--output",
            str(output),
        ],
    )
    assert render_one_gpu_job.main() == 0
    contract = json.loads(output.read_text())
    assert contract["gres"] == "gpu:h100nvl:1"
    assert contract["user_submission_authorization"]["authorized"] is True
    assert contract["scientific_submission_blockers"] == [
        "missing source object/tree"
    ]
    assert contract["accelerator_selection"]["selected"] == "gpu:h100nvl:1"
    assert "--gres=gpu:h100nvl:1" in json.loads(capsys.readouterr().out)[
        "sbatch_command"
    ]


def test_scientific_contract_verifier_revalidates_completion_binding(
    monkeypatch, tmp_path
):
    checked = {}
    monkeypatch.setattr(
        "scripts.slurm.verify_job_contract._git",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        "scripts.slurm.verify_job_contract.load_local_microtest_completion_receipt",
        lambda path, *, admission_path: checked.update(
            completion=path, admission=admission_path
        ),
    )
    contract = {
        "contract_version": "hypertagging-slurm-one-gpu-contract-v2",
        "mode": "scientific",
        "export_policy": "NIL",
        "gpu_environment": "/frozen/gpu-env",
        "gres": "gpu:h200nvl:1",
        "train_config": "configs/slurm/pretrain_035k_scientific.yaml",
        "experiment": "safe-experiment",
        "seed": 20260812,
        "max_restarts": 2,
        "expected_git_sha": "a" * 40,
        "expected_git_tag": None,
        "hashed_inputs": [],
        "local_admission_receipt": "/evidence/admission.json",
        "local_completion_receipt": "/evidence/completion.json",
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    verify_contract(path)
    assert checked == {
        "admission": "/evidence/admission.json",
        "completion": "/evidence/completion.json",
    }
    tampered = {**contract, "export_policy": "NONE"}
    tampered.pop("contract_sha256")
    canonical = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    tampered["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="NIL export policy"):
        verify_contract(path)


def test_blocked_scientific_contract_verifies_but_refuses_shell_runtime(
    monkeypatch, tmp_path, capsys
):
    admission = tmp_path / "admission.json"
    completion = tmp_path / "completion.json"
    admission.write_text('{"admission":true}\n')
    completion.write_text('{"completion":true}\n')
    gpu_env = tmp_path / "gpu-env"
    (gpu_env / "bin").mkdir(parents=True)
    (gpu_env / "bin" / "python").write_text("")
    monkeypatch.setattr(
        render_one_gpu_job,
        "load_local_microtest_completion_receipt",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        render_one_gpu_job,
        "validate_live_slurm",
        lambda gres: {"exact_gres": gres, "version": "slurm 23.02.8"},
    )
    monkeypatch.setattr(
        render_one_gpu_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "scientific_slurm_submission_allowed": False,
                    "blockers": ["missing source object/tree"],
                }
            ),
        ),
    )
    output = tmp_path / "blocked-scientific-contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_one_gpu_job.py",
            "--mode",
            "scientific",
            "--gres",
            "gpu:v100:1",
            "--gpu-env",
            str(gpu_env),
            "--expected-git-sha",
            "a" * 40,
            "--local-admission-receipt",
            str(admission),
            "--local-completion-receipt",
            str(completion),
            "--blocked-no-submit",
            "--output",
            str(output),
        ],
    )
    assert render_one_gpu_job.main() == 0
    contract = json.loads(output.read_text())
    assert contract["submission_authorized"] is False
    assert contract["verification_scope"] == "blocked_no_submit"
    assert contract["scientific_submission_blockers"] == [
        "missing source object/tree"
    ]
    assert json.loads(capsys.readouterr().out)["sbatch_command"]

    monkeypatch.setattr(
        "scripts.slurm.verify_job_contract._git",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        "scripts.slurm.verify_job_contract.verify_hashed_inputs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.slurm.verify_job_contract.load_local_microtest_completion_receipt",
        lambda *args, **kwargs: {},
    )
    verified, _, _ = verify_contract(output)
    assert verified["submission_authorized"] is False


def test_wrapper_forwards_usr1_and_requeues_at_most_once(tmp_path):
    child_ready = tmp_path / "ready"
    child_signal = tmp_path / "child-signal"
    requeue_log = tmp_path / "requeues"
    status_file = tmp_path / "wrapper-status.json"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import os, signal, sys, time\n"
        "ready, seen = sys.argv[1:]\n"
        "def handle(*_):\n"
        "    open(seen, 'w').write('USR1')\n"
        "    raise SystemExit(75)\n"
        "signal.signal(signal.SIGUSR1, handle)\n"
        "open(ready, 'w').write(str(os.getpid()))\n"
        "while True: time.sleep(0.05)\n"
    )
    fake_scontrol = tmp_path / "scontrol.sh"
    fake_scontrol.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$REQUEUE_LOG"\n'
    )
    fake_scontrol.chmod(0o755)
    process = subprocess.Popen(
        [
            "bash",
            "scripts/slurm/run_with_bounded_requeue.sh",
            "--max-restarts",
            "2",
            "--scontrol",
            str(fake_scontrol),
            "--status-file",
            str(status_file),
            "--",
            sys.executable,
            str(trainer),
            str(child_ready),
            str(child_signal),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SLURM_JOB_ID": "123",
            "SLURM_RESTART_COUNT": "0",
            "REQUEUE_LOG": str(requeue_log),
        },
    )
    deadline = time.monotonic() + 5
    while not child_ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_ready.exists()
    os.kill(process.pid, signal.SIGUSR1)
    assert process.wait(timeout=5) == 0
    assert child_signal.read_text() == "USR1"
    assert requeue_log.read_text().splitlines() == ["requeue 123"]
    status = json.loads(status_file.read_text())
    assert status["action"] == "requeue_requested"
    assert status["trainer_status"] == 75
    assert status["usr1_received"] == 1


def test_attempt_receipt_hashes_failure_evidence(tmp_path):
    attempt = tmp_path / "attempt-00"
    run = tmp_path / "run"
    attempt.mkdir()
    run.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text('{"contract":true}\n')
    stage_log = attempt / "stages.log"
    stage_log.write_text("stage=trainer_failed\n")
    wrapper_status = attempt / "wrapper-status.json"
    wrapper_status.write_text(
        json.dumps(
            {
                "action": "trainer_exit",
                "trainer_status": 1,
                "wrapper_status": 1,
            }
        )
    )
    metrics = run / "metrics.jsonl"
    metrics.write_text('{"loss":1.0}\n')
    receipt = attempt / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/slurm/finalize_execution_receipt.py",
            "--receipt",
            str(receipt),
            "--contract",
            str(contract),
            "--attempt-root",
            str(attempt),
            "--run-root",
            str(run),
            "--stage-log",
            str(stage_log),
            "--wrapper-status",
            str(wrapper_status),
            "--batch-exit-status",
            "1",
            "--terminal-stage",
            "trainer_failed",
            "--started-at",
            "2026-08-14T10:00:00+00:00",
            "--completed-at",
            "2026-08-14T10:01:00+00:00",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(receipt.read_text())
    stored = payload.pop("receipt_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert stored == hashlib.sha256(canonical.encode()).hexdigest()
    assert payload["status"] == "failed_or_nonterminal"
    assert payload["trainer_status"] == 1
    assert payload["artifacts"]["metrics"]["sha256"] == hashlib.sha256(
        metrics.read_bytes()
    ).hexdigest()
    assert verify_receipt(receipt)["trainer_status"] == 1
    with pytest.raises(RuntimeError, match="normal completion"):
        verify_receipt(receipt, require_completed=True)
    metrics.write_text("changed\n")
    with pytest.raises(RuntimeError, match="artifact (size|hash) changed"):
        verify_receipt(receipt)
