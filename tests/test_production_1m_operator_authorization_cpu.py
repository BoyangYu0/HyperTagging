from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.slurm import render_one_gpu_job  # noqa: E402
from scripts.slurm import verify_job_contract as contract_verifier  # noqa: E402
from scripts.slurm.verify_job_contract import verify_contract  # noqa: E402


PROVENANCE_BLOCKER = (
    "The production source object f4e54df23b5c60115e475c5d68df4651899d678e "
    "remains unavailable locally; independently recover it and verify tree "
    "b6e3a4118b960e3a4676a61af9601438d56cef96."
)


def _render_authorized_contract(monkeypatch, tmp_path, *, capsys):
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
        lambda gres, *, require_highest_priority=True: {
            "exact_gres": gres,
            "usable_priority": ["gpu:h100nvl:1"],
            "selection_reason": "operator-selected exact H100 NVL",
        },
    )
    monkeypatch.setattr(
        render_one_gpu_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "valid",
                    "scientific_slurm_submission_allowed": False,
                    "blockers": [PROVENANCE_BLOCKER],
                }
            ),
        ),
    )
    output = tmp_path / "authorized-production-1m-h100-contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_one_gpu_job.py",
            "--mode",
            "scientific",
            "--gres",
            "gpu:h100nvl:1",
            "--fullscale",
            "--scientific-config",
            "configs/slurm/pretrain_1m_h100_20260821.yaml",
            "--gpu-env",
            str(gpu_env),
            "--expected-git-sha",
            "a" * 40,
            "--expected-git-tag",
            "ht-pretraining-production-1m-h100-operator-authorized-20260821",
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
    rendered = json.loads(output.read_text())
    capsys.readouterr()
    return output, rendered


def _patch_git_and_receipts(monkeypatch):
    monkeypatch.setattr(
        contract_verifier,
        "_git",
        lambda *args: (
            "a" * 40
            if args == ("rev-parse", "HEAD")
            else (
                "a" * 40
                if args
                == (
                    "rev-list",
                    "-n",
                    "1",
                    "ht-pretraining-production-1m-h100-operator-authorized-20260821",
                )
                else ""
            )
        ),
    )
    monkeypatch.setattr(
        contract_verifier,
        "load_local_microtest_completion_receipt",
        lambda *args, **kwargs: {},
    )


def _rehash(contract):
    contract = dict(contract)
    contract.pop("contract_sha256", None)
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return contract


def test_authorized_fullscale_contract_records_exact_exception_and_one_job(
    monkeypatch, tmp_path, capsys
):
    path, contract = _render_authorized_contract(monkeypatch, tmp_path, capsys=capsys)

    authorization = contract["user_submission_authorization"]
    assert contract["submission_authorized"] is True
    assert contract["submission_performed"] is False
    assert contract["execution_authorization"] == {
        "basis": "operator_provenance_exception",
        "execution_authorized": True,
    }
    assert authorization["authorization_date"] == "2026-08-21"
    assert authorization["source"] == "interactive_user_instruction"
    assert authorization["scope"] == (
        "exactly_one_production_1m_pretraining_job_on_gpu:h100nvl:1"
    )
    assert authorization["job_count"] == 1
    assert authorization["gres"] == "gpu:h100nvl:1"

    exception = contract["operator_provenance_exception"]
    assert exception["authorization_date"] == "2026-08-21"
    assert exception["source"] == "interactive_user_instruction"
    assert exception["scope"] == authorization["scope"]
    assert exception["job_count"] == 1
    assert exception["gres"] == "gpu:h100nvl:1"
    assert exception["limitation"] == PROVENANCE_BLOCKER
    assert contract["provenance_status"]["scientific_slurm_submission_allowed"] is False
    assert contract["provenance_validation"]["scientific_slurm_submission_allowed"] is False
    assert contract["provenance_validation"]["expected_missing_source_commit"] == (
        "f4e54df23b5c60115e475c5d68df4651899d678e"
    )
    assert contract["provenance_validation"]["expected_missing_source_tree"] == (
        "b6e3a4118b960e3a4676a61af9601438d56cef96"
    )

    selection = json.loads(
        (ROOT / contract["selection_manifest"]).read_text(encoding="utf-8")
    )
    assert selection["selection_includes_test"] is False
    assert selection["split_counts"] == {
        "test": 0,
        "train": 865000,
        "validation": 50000,
    }
    assert contract["sealed_test_role_access"] == "forbidden"

    _patch_git_and_receipts(monkeypatch)
    verified, _, _ = verify_contract(path)
    assert verified["submission_authorized"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract["hashed_inputs"][0].update({"sha256": "0" * 64}),
            "hashed job input changed",
        ),
        (
            lambda contract: contract["resource_contract"].update({"memory": "128G"}),
            "full-scale resource contract is not exact",
        ),
        (
            lambda contract: contract["operator_provenance_exception"].update(
                {"scope": "two_production_1m_jobs"}
            ),
            "operator provenance exception scope is not exact",
        ),
    ],
)
def test_authorized_contract_rejects_bound_hash_resource_or_scope_mutation(
    monkeypatch, tmp_path, capsys, mutation, message
):
    path, contract = _render_authorized_contract(monkeypatch, tmp_path, capsys=capsys)
    mutation(contract)
    path.write_text(json.dumps(_rehash(contract), indent=2) + "\n")
    _patch_git_and_receipts(monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        verify_contract(path)
