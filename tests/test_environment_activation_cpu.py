from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_activation_helper_must_be_sourced():
    result = subprocess.run(
        ["bash", "scripts/activate_env.sh", "project"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "source this helper" in result.stderr


def test_project_activation_is_sourceable_and_changes_to_repository(tmp_path):
    repository = tmp_path / "repo"
    scripts = repository / "scripts"
    environment = repository / ".venv" / "bin"
    scripts.mkdir(parents=True)
    environment.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/activate_env.sh", scripts / "activate_env.sh")
    (environment / "activate").write_text(
        f'export VIRTUAL_ENV="{repository / ".venv"}"\n'
        f'export PATH="{environment}:$PATH"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                f'cd / && source "{scripts / "activate_env.sh"}" project && '
                "printf '%s|%s|%s' \"$VIRTUAL_ENV\" \"$HYPERTAGGING_ENV_MODE\" \"$PWD\""
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{repository / '.venv'}|project|{repository}"


def test_activation_helper_rejects_unknown_mode_without_changing_caller(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                "before=$PWD; source scripts/activate_env.sh invalid; status=$?; "
                "printf '%s|%s|%s' \"$status\" \"$before\" \"$PWD\""
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == f"64|{ROOT}|{ROOT}"
