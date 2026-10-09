import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "row_delivery", ROOT / "docs/_ext/wiki_historical_rows.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def original():
    return {
        "reconstruction": {
            phase: {
                "metric_rows": json.loads(
                    gzip.decompress(
                        (
                            ROOT / module().SOURCE / f"{phase}-metric-rows.json.gz"
                        ).read_bytes()
                    )
                ),
                "summary_preserved": True,
            }
            for phase in module().PHASES
        }
    }


def test_historical_rows_roundtrip_preserves_every_original_scalar(tmp_path):
    m = module()
    data = original()
    before = copy.deepcopy(data)
    for phase in m.PHASES:
        source = next(
            (ROOT / "artifacts/codex").glob(f"reconstruction_{phase}_closeout_*.json")
        )
        assert (
            data["reconstruction"][phase]["metric_rows"]
            == json.loads(source.read_text())["metric_rows"]
        )
    receipts = m.compact(ROOT, tmp_path, data)
    subprocess.run(
        [sys.executable, str(tmp_path / "phase42-44-row-decoder.txt")],
        cwd=tmp_path,
        check=True,
    )
    assert [r["rows"] for r in receipts] == [16306, 16405, 15434]
    for phase in m.PHASES:
        assert data["reconstruction"][phase]["summary_preserved"]
        assert "metric_rows" not in data["reconstruction"][phase]
        restored = (tmp_path / f"{phase}-metric-rows-decoded.json").read_bytes()
        assert restored == m.canonical(before["reconstruction"][phase]["metric_rows"])


def test_historical_source_change_fails_closed(tmp_path):
    data = original()
    data["reconstruction"]["phase42"]["metric_rows"][0]["value"] += 1
    with pytest.raises(ValueError, match="rows changed"):
        module().compact(ROOT, tmp_path, data)


def test_historical_archive_hash_change_fails_closed(tmp_path):
    m = module()
    source = tmp_path / m.SOURCE
    source.mkdir(parents=True)
    (source / "binding.json").write_bytes(
        (ROOT / m.SOURCE / "binding.json").read_bytes()
    )
    (source / "phase42-metric-rows.json.gz").write_bytes(b"bad archive")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="compressed hash"):
        m.compact(tmp_path, output, original())
