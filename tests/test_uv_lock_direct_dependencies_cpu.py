from pathlib import Path

from scripts.check_uv_lock_direct_dependencies import direct_dependency_errors


ROOT = Path(__file__).resolve().parents[1]


def test_uv_lock_root_metadata_matches_all_direct_runtime_dependencies():
    assert direct_dependency_errors() == []


def test_uv_lock_checker_rejects_new_unlocked_direct_dependency(tmp_path):
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    changed = pyproject.replace(
        '  "PyYAML",\n]',
        '  "PyYAML",\n  "example-unlocked-runtime",\n]',
    )
    assert changed != pyproject
    changed_path = tmp_path / "pyproject.toml"
    changed_path.write_text(changed, encoding="utf-8")

    errors = direct_dependency_errors(changed_path, ROOT / "uv.lock")

    assert errors == [
        "root dependencies missing direct runtime dependencies: example-unlocked-runtime",
        "root requires-dist missing direct runtime dependencies: example-unlocked-runtime",
    ]
